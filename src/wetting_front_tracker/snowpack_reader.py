"""
snowpack_reader.py
==================

xsnow-backed reader for SNOWPACK .pro files.

Reads .pro files via ``xsnow.read()``, then adapts the resulting
5-D dataset to the 2-D ``(timestamp, layer_index)`` structure expected
by the rest of the analysis pipeline.  All public methods and the
``metadata`` dict remain identical to the previous implementation.

Hardware Acceleration:
- GPU Support: Uses CuPy for GPU acceleration of rc_flat if available
- CPU Fallback: Uses NumPy otherwise

Author: Ron Simenhois
"""

import gc
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import xsnow

# Hardware-adaptive array library (used for rc_flat computation)
try:
    import cupy as xp
    _ = xp.arange(1)
    GPU_AVAILABLE = True
    logging.info("GPU detected. Using CuPy for accelerated calculations.")
except (ImportError, RuntimeError):
    import numpy as xp
    GPU_AVAILABLE = False
    logging.info("No GPU or CuPy found. Using NumPy on CPU.")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

logger = logging.getLogger(__name__)

# Physical constants for rc_flat calculation
RHO_ICE = 917.0  # Ice density (kg/m³)
GS_0 = 0.00125   # Reference grain size (m)
G = 9.81         # Gravitational acceleration (m/s²)
A = 4.6e-9       # Material constant
B = -2.0         # Material constant


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _cleanup_gpu_memory() -> None:
    """Explicitly free GPU memory if using CuPy."""
    if GPU_AVAILABLE:
        gc.collect()
        try:
            xp.get_default_memory_pool().free_all_blocks()
            logger.debug("GPU memory cleaned up")
        except AttributeError:
            pass


def _to_cpu_array(data: xr.DataArray) -> xr.DataArray:
    """Convert xarray DataArray from GPU to CPU if needed."""
    if GPU_AVAILABLE and hasattr(data.data, 'get'):
        return xr.DataArray(data.data.get(), dims=data.dims, coords=data.coords)
    return data


def _compute_sqrt_term(term_data: xr.DataArray) -> xr.DataArray:
    """Safely compute square root, replacing non-positive values with NaN."""
    data_copy = term_data.data.copy()
    data_copy[data_copy <= 0] = xp.nan
    sqrt_data = xp.sqrt(data_copy)
    return xr.DataArray(sqrt_data, dims=term_data.dims, coords=term_data.coords)


# ---------------------------------------------------------------------------
# SnowpackProfile Class
# ---------------------------------------------------------------------------

class SnowpackProfile:
    """
    Reads and represents a SNOWPACK .pro file.

    Internally uses ``xsnow.read()`` for I/O, then adapts the dataset to a
    2-D ``(timestamp, layer_index)`` structure.  The public interface is
    identical to the previous implementation.

    Attributes:
        filename: Path to the input .pro file
        metadata: Station parameters extracted from the file
        data: xarray Dataset with dims ``(timestamp, layer_index)``
    """

    def __init__(self, filename: Union[Path, str], _load_data: bool = True):
        self.filename: Path = Path(filename) if isinstance(filename, str) else filename
        self.metadata: Dict[str, Any] = {}
        self.data: Optional[xr.Dataset] = None

        if _load_data:
            self._read_profile()

    def __len__(self) -> int:
        return 0 if self.data is None else len(self.data.timestamp)

    def __repr__(self) -> str:
        device = "GPU" if GPU_AVAILABLE else "CPU"
        return f"<SnowpackProfile(filename='{self.filename}', profiles={len(self)}, device='{device}')>"

    def _read_profile(self) -> None:
        """Load profile data using xsnow and adapt to the (timestamp, layer_index) API."""
        if not self.filename.exists():
            raise FileNotFoundError(f"File not found: {self.filename}")

        try:
            xsnow_ds = xsnow.read(str(self.filename), lazy=False)
        except Exception as e:
            logger.error(f"Error reading {self.filename}: {e}", exc_info=True)
            raise

        if xsnow_ds is None or xsnow_ds.data is None or xsnow_ds.data.time.size == 0:
            logger.warning(f"No profile data parsed from {self.filename}")
            return

        raw = xsnow_ds.data

        # Extract station metadata from xsnow coordinates
        self.metadata = {
            'stationName': str(raw.location.values[0]),
            'latitude':    float(raw.latitude.values[0]),
            'longitude':   float(raw.longitude.values[0]),
            'altitude':    float(raw.altitude.values[0]),
        }

        # Squeeze singleton dims (location, slope, realization) and rename
        # time→timestamp, layer→layer_index to preserve the downstream API.
        data = raw.squeeze(['location', 'slope', 'realization'], drop=True)
        self.data = data.rename({'time': 'timestamp', 'layer': 'layer_index'}).sortby('timestamp')

        self._compute_and_add_depth()
        self._compute_and_add_rc_flat_vectorized()

    def _compute_and_add_depth(self) -> None:
        """Calculate depth of each layer from the snow surface."""
        if self.data is None or 'height' not in self.data.data_vars:
            logger.warning("Cannot calculate depth without 'height' variable")
            return

        height = self.data['height']
        total_height = height.max(dim='layer_index', skipna=True)
        depth = total_height - height

        if GPU_AVAILABLE:
            final_depth_data = xp.where(
                height.notnull().data,
                depth.data,
                xp.nan
            )
            self.data['depth'] = xr.DataArray(
                final_depth_data, dims=depth.dims, coords=depth.coords
            )
        else:
            self.data['depth'] = depth.where(height.notnull())

    def _compute_and_add_rc_flat_vectorized(self) -> None:
        """
        Calculate rc_flat for all profiles using vectorized operations.

        Benefits significantly from GPU acceleration when CuPy is available.
        """
        required_vars = {'density', 'grain_size', 'shear_strength', 'height'}
        if self.data is None or not required_vars.issubset(self.data.data_vars):
            logger.warning("Skipping rc_flat: missing required variables")
            return

        try:
            height = self.data['height']
            density = self.data['density']
            grain_size = self.data['grain_size']
            shear_strength = self.data['shear_strength']

            height_of_bottom = height.shift(layer_index=1, fill_value=0)
            thick = height - height_of_bottom
            layer_load = density * thick * G

            load = (
                layer_load
                .reindex(layer_index=layer_load.layer_index[::-1])
                .cumsum(dim='layer_index')
                .reindex(layer_index=layer_load.layer_index)
            )

            total_thick_above = self.data['height'].max(dim='layer_index', skipna=True) - height
            rho_sl_raw = load / (total_thick_above * G)

            if GPU_AVAILABLE:
                rho_sl_numpy = rho_sl_raw.to_numpy()
                rho_sl_cpu_da = xr.DataArray(
                    rho_sl_numpy, dims=rho_sl_raw.dims, coords=rho_sl_raw.coords
                )
                rho_sl_filled_cpu = rho_sl_cpu_da.bfill(dim='layer_index').ffill(dim='layer_index')
                rho_sl_gpu = xp.asarray(rho_sl_filled_cpu.values)
                rho_sl = xr.DataArray(rho_sl_gpu, dims=rho_sl_raw.dims, coords=rho_sl_raw.coords)
            else:
                rho_sl = rho_sl_raw.bfill(dim='layer_index').ffill(dim='layer_index')

            tau_p = shear_strength * 1000.0
            gs = grain_size * 0.001
            e_prime = 5.07e9 * (rho_sl / RHO_ICE)**5.13 / (1 - 0.2**2)
            dsl_over_sigman = 1.0 / (G * rho_sl)

            term1_under = A * (density / RHO_ICE * gs / GS_0)**B
            term2_under = 2 * tau_p * e_prime * dsl_over_sigman

            term1 = _compute_sqrt_term(term1_under)
            term2 = _compute_sqrt_term(term2_under)

            rc_flat_combined = term1 * term2
            rc_flat_filled_data = xp.nan_to_num(rc_flat_combined.data, nan=9999.0)
            rc_flat_da = xr.DataArray(
                rc_flat_filled_data, dims=rc_flat_combined.dims, coords=rc_flat_combined.coords
            )

            max_heights = height.max(dim='layer_index', skipna=True)
            is_not_surface = height < max_heights

            if GPU_AVAILABLE:
                final_data = xp.where(is_not_surface.data, rc_flat_da.data, 9999.0)
                rc_flat_da = xr.DataArray(final_data, dims=rc_flat_da.dims, coords=rc_flat_da.coords)
            else:
                rc_flat_da = rc_flat_da.where(is_not_surface, 9999.0)

            self.data['rc_flat'] = rc_flat_da.transpose('timestamp', 'layer_index')

            _cleanup_gpu_memory()

        except Exception as e:
            logger.error(f"Failed to calculate rc_flat: {e}", exc_info=True)

    def slice(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> 'SnowpackProfile':
        """Return a new SnowpackProfile containing only the specified date range."""
        if self.data is None or self.data.timestamp.size == 0:
            logger.warning("Cannot slice empty dataset")
            return self

        timestamps = pd.to_datetime(self.data.timestamp.values)

        start_mask = timestamps >= pd.to_datetime(start_date) if start_date else np.ones(len(timestamps), dtype=bool)
        end_mask   = timestamps <= pd.to_datetime(end_date)   if end_date   else np.ones(len(timestamps), dtype=bool)
        combined_mask = start_mask & end_mask

        if not np.any(combined_mask):
            logger.warning(
                f"No data in date range {start_date} to {end_date} for {self.filename}"
            )
            sliced_data = self.data.isel(timestamp=slice(0, 0))
        else:
            sliced_data = self.data.isel(timestamp=np.where(combined_mask)[0])

        new_profile = SnowpackProfile(self.filename, _load_data=False)
        new_profile.data = sliced_data
        new_profile.metadata = self.metadata
        return new_profile

    def save_as_netcdf(self, output_path: Union[str, Path]) -> None:
        """Save the profile dataset to a NetCDF file."""
        if self.data is None or self.data.timestamp.size == 0:
            logger.warning(f"No data to save for {output_path}")
            return

        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            data_to_save = self.data

            if GPU_AVAILABLE:
                cpu_data = xr.Dataset(attrs=self.data.attrs)
                for var_name, data_array in self.data.data_vars.items():
                    cpu_data[var_name] = (data_array.dims, data_array.data.get())
                coords_dict = {
                    coord_name: (
                        coord_val.dims,
                        coord_val.data.get() if hasattr(coord_val.data, 'get') else coord_val
                    )
                    for coord_name, coord_val in self.data.coords.items()
                }
                cpu_data = cpu_data.assign_coords(coords_dict)
                data_to_save = cpu_data

            data_to_save.to_netcdf(output_path)
            logger.debug(f"Saved profile to NetCDF: {output_path}")

        except Exception as e:
            logger.error(f"Failed to save NetCDF to {output_path}: {e}", exc_info=True)

    def _process_summary_row(
        self,
        profile_layers: pd.DataFrame,
        parameters_to_calculate: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate summary statistics for a single profile timestamp."""
        summary_row = {}

        for name, calc in parameters_to_calculate.items():
            if callable(calc):
                try:
                    result = calc(profile_layers)
                    summary_row.update(self._unpack_result(name, result))
                except Exception as e:
                    logger.warning(f"Custom function for '{name}' failed: {e}")
                    summary_row[name] = np.nan
                continue

            if isinstance(calc, str):
                param, calc_type = name.split('-')[0], calc
            elif isinstance(calc, tuple):
                param, calc_type = calc
            else:
                continue

            if param not in profile_layers.columns:
                continue

            series = profile_layers[param].dropna()
            if series.empty:
                continue

            if param == 'hand_hardness':
                series = series.abs()

            if calc_type == 'min':
                summary_row[name] = series.min()
                if 'height' in profile_layers.columns:
                    summary_row[f"{name}-height"] = profile_layers.loc[series.idxmin()]['height']
            elif calc_type == 'max':
                summary_row[name] = series.max()
                if 'height' in profile_layers.columns:
                    summary_row[f"{name}-height"] = profile_layers.loc[series.idxmax()]['height']
            elif calc_type == 'mean':
                summary_row[name] = series.mean()
            elif calc_type == 'median':
                summary_row[name] = series.median()
            elif calc_type in ['weighted_mean', 'weighted_sum']:
                weights = profile_layers.loc[series.index]['thickness']
                weighted_sum_val = (series * weights).sum()
                if calc_type == 'weighted_sum':
                    summary_row[name] = weighted_sum_val
                elif weights.sum() > 0:
                    summary_row[name] = weighted_sum_val / weights.sum()

        return summary_row

    def _unpack_result(self, name: str, result: Any) -> Dict[str, Any]:
        """Unpack tuple results into named dictionary entries."""
        if result is None:
            return {name: np.nan}
        if isinstance(result, tuple):
            unpacked = {f"{name}_value": result[0] if result[0] is not None else np.nan}
            if len(result) > 1:
                unpacked[f"{name}_height"] = result[1] if result[1] is not None else np.nan
            return unpacked
        return {name: result}

    def get_profile_summary(
        self,
        parameters_to_calculate: Dict[str, Any],
        from_height: Optional[float] = None,
        above_or_below: str = 'above',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Extract daily summary statistics (uses noon profile for each day)."""
        sliced_profile = self.slice(start_date, end_date)
        if sliced_profile.data is None or sliced_profile.data.timestamp.size == 0:
            return pd.DataFrame()

        data_in_range = sliced_profile.data
        if GPU_AVAILABLE:
            data_in_range = data_in_range.as_numpy()

        full_df = data_in_range.to_dataframe().reset_index()

        noon_time = full_df['timestamp'].dt.normalize() + pd.Timedelta(hours=12)
        full_df['time_from_noon'] = (full_df['timestamp'] - noon_time).abs()
        closest_indices = full_df.loc[
            full_df.groupby(full_df['timestamp'].dt.date)['time_from_noon'].idxmin()
        ]
        noon_df = closest_indices.copy()

        if noon_df.empty:
            return pd.DataFrame()

        noon_df['date'] = noon_df['timestamp'].dt.normalize()

        summary_list = []
        for ts in noon_df['timestamp']:
            summary_row = {'date': ts.normalize()}
            profile_layers = full_df[full_df['timestamp'] == ts].copy()

            if from_height is not None:
                if above_or_below == 'above':
                    profile_layers = profile_layers[profile_layers['height'] > from_height]
                else:
                    profile_layers = profile_layers[profile_layers['height'] <= from_height]

            if not profile_layers.empty:
                profile_layers = profile_layers.sort_values('height').copy()
                profile_layers['thickness'] = profile_layers['height'].diff()
                base_h = from_height if (from_height is not None and above_or_below == 'above') else 0
                first_row_idx = profile_layers.index[0]
                profile_layers.loc[first_row_idx, 'thickness'] = float(
                    profile_layers['height'].iloc[0] - base_h
                )

            summary_row.update(self._process_summary_row(profile_layers, parameters_to_calculate))
            summary_list.append(summary_row)

        if not summary_list:
            return pd.DataFrame()

        return pd.DataFrame(summary_list).set_index('date')

    def get_full_timeseries_summary(
        self,
        parameters_to_calculate: Dict[str, Callable],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Calculate summary statistics for every timestamp (high resolution)."""
        sliced_profile = self.slice(start_date, end_date)
        if sliced_profile.data is None or sliced_profile.data.timestamp.size == 0:
            return pd.DataFrame()

        data_in_range = sliced_profile.data
        summary_list = []

        for ts_val in data_in_range.timestamp.values:
            summary_row = {'timestamp': ts_val}
            single_profile_ds = data_in_range.sel(timestamp=ts_val)

            if GPU_AVAILABLE:
                profile_layers = single_profile_ds.as_numpy().to_dataframe().reset_index()
            else:
                profile_layers = single_profile_ds.to_dataframe().reset_index()

            if profile_layers.empty:
                continue

            for name, calc in parameters_to_calculate.items():
                if not callable(calc):
                    logger.warning(f"Calculation for '{name}' is not callable. Skipping.")
                    continue
                try:
                    result = calc(profile_layers)
                    summary_row.update(self._unpack_result(name, result))
                except Exception as e:
                    logger.warning(f"Function '{name}' failed at {ts_val}: {e}")
                    summary_row[name] = np.nan

            summary_list.append(summary_row)

        if not summary_list:
            return pd.DataFrame()

        return pd.DataFrame(summary_list).set_index('timestamp')

    def find_layer_by_criteria(
        self,
        criteria: Dict[str, str],
        search_from: str = 'top',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Find the layer best matching a set of prioritised criteria (daily resolution)."""
        import re
        sliced_profile = self.slice(start_date, end_date)
        if sliced_profile.data is None or sliced_profile.data.timestamp.size == 0:
            return pd.DataFrame()

        data_in_range = sliced_profile.data
        if GPU_AVAILABLE:
            data_in_range = data_in_range.as_numpy()

        full_df = data_in_range.to_dataframe().reset_index()

        noon_time = full_df['timestamp'].dt.normalize() + pd.Timedelta(hours=12)
        full_df['time_from_noon'] = (full_df['timestamp'] - noon_time).abs()
        closest_indices = full_df.loc[
            full_df.groupby(full_df['timestamp'].dt.date)['time_from_noon'].idxmin()
        ]
        noon_df = closest_indices.copy()

        if noon_df.empty:
            return pd.DataFrame()

        results_list = []
        op_pattern = re.compile(r'([<>=!]+)\s*(\S+)')
        ordered_criteria = list(criteria.items())
        num_criteria = len(ordered_criteria)

        for ts in tqdm(noon_df['timestamp'], desc="Finding Layers by Criteria"):
            daily_result = {'date': ts.normalize()}
            df = full_df[full_df['timestamp'] == ts].copy()

            if df.empty:
                continue

            if 'depth' in criteria and 'height' in df.columns:
                df['depth'] = df['height'].max() - df['height']

            if 'hand_hardness' in criteria and 'hand_hardness' in df.columns:
                df['hand_hardness'] = df['hand_hardness'].abs()

            score = pd.Series(0, index=df.index, dtype=int)
            criteria_masks = {}

            for i, (param, condition) in enumerate(ordered_criteria):
                weight = 2**(num_criteria - 1 - i)

                if param not in df.columns:
                    logger.warning(f"Parameter '{param}' not found. Skipping.")
                    continue

                condition_mask = pd.Series(False, index=df.index)

                if ' to ' in condition:
                    try:
                        low_str, high_str = condition.split(' to ', 1)
                        low, high = float(low_str), float(high_str)
                        condition_mask = (df[param] >= low) & (df[param] <= high)
                    except (ValueError, IndexError):
                        logger.warning(f"Invalid range format: '{condition}'")
                        continue
                else:
                    match = op_pattern.match(condition)
                    if not match:
                        logger.warning(f"Invalid condition format: '{condition}'")
                        continue
                    op, value_str = match.groups()
                    try:
                        value = float(value_str)
                    except ValueError:
                        logger.warning(f"Could not convert '{value_str}' to float")
                        continue

                    ops = {'<': df[param].__lt__, '>': df[param].__gt__,
                           '<=': df[param].__le__, '>=': df[param].__ge__,
                           '==': df[param].__eq__, '!=': df[param].__ne__}
                    if op not in ops:
                        raise ValueError(f"Unsupported operator '{op}'")
                    condition_mask = ops[op](value)

                criteria_masks[param] = condition_mask.fillna(False)
                score += criteria_masks[param].astype(int) * weight

            max_score = score.max()

            if max_score == 0:
                daily_result['height'] = None
                daily_result['matching_criteria_count'] = 0
                daily_result['matching_parameters'] = ''
                for param, _ in ordered_criteria:
                    daily_result[param] = None
            else:
                best_matching_layers = df[score == max_score]

                if search_from == 'top':
                    target_layer_index = best_matching_layers.index[-1]
                elif search_from == 'bottom':
                    target_layer_index = best_matching_layers.index[0]
                else:
                    raise ValueError("'search_from' must be 'top' or 'bottom'")

                target_layer = df.loc[target_layer_index]
                daily_result['height'] = target_layer['height']

                matched_params = [
                    param for param, mask in criteria_masks.items()
                    if mask.loc[target_layer_index]
                ]
                daily_result['matching_criteria_count'] = len(matched_params)
                daily_result['matching_parameters'] = ', '.join(matched_params)

                for param, _ in ordered_criteria:
                    daily_result[param] = target_layer[param] if param in target_layer else None

            results_list.append(daily_result)

        if not results_list:
            return pd.DataFrame()

        return pd.DataFrame(results_list).set_index('date')


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def read_snowpack(pro_file_path: Union[str, Path]) -> Optional[SnowpackProfile]:
    """
    Read a SNOWPACK profile, using a cached NetCDF when available.

    On the first call for a given .pro file the profile is parsed and the
    result is written to a .nc sidecar for fast re-loading on subsequent runs.
    """
    pro_path = Path(pro_file_path)
    nc_path = pro_path.with_suffix('.nc')

    if nc_path.exists():
        try:
            data = xr.open_dataset(nc_path)
            profile = SnowpackProfile(str(pro_path), _load_data=False)
            profile.data = data
            logger.debug(f"Loaded from cached NetCDF: {nc_path}")
            return profile
        except Exception as e:
            logger.warning(f"Could not read cached NetCDF {nc_path}, falling back to .pro: {e}")

    try:
        profile = SnowpackProfile(str(pro_path))
        if profile.data is not None and profile.data.timestamp.size > 0:
            profile.save_as_netcdf(str(nc_path))
        return profile
    except Exception as e:
        logger.error(f"Failed to read .pro file {pro_path}: {e}")
        return None


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print(f"GPU Available: {GPU_AVAILABLE}")
    print(f"Array Backend: {'CuPy' if GPU_AVAILABLE else 'NumPy'}")
