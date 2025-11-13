"""
param_config.py
===============

Comprehensive configuration for SNOWPACK layer parameters.
Maps parameter codes to names, units, and extraction functions.

This module defines all parameters available in SNOWPACK .pro files
and provides utilities for extracting them from profile data.

Author: Ron Simenhois
Created: November 2025
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# SNOWPACK Parameter Definitions
# ---------------------------------------------------------------------------

@dataclass
class ParameterDefinition:
    """
    Definition of a SNOWPACK parameter.
    
    Attributes:
        code: SNOWPACK parameter code (e.g., '0502')
        name: Human-readable name
        units: Physical units
        description: Brief description
        column_name: Name in xarray Dataset/DataFrame
        is_layer_param: True if parameter exists for each layer
        compute_diff: Whether to compute interface differences
        compute_ratio: Whether to compute interface ratios
    """
    code: str
    name: str
    units: str
    description: str
    column_name: str
    is_layer_param: bool = True
    compute_diff: bool = True
    compute_ratio: bool = False


# All SNOWPACK parameters we want to extract
# Based on the comprehensive list provided by user
SNOWPACK_PARAMETERS = {
    # Core structural parameters
    '0501': ParameterDefinition(
        code='0501',
        name='height',
        units='cm',
        description='Element height (top of layer)',
        column_name='height',
        is_layer_param=True,
        compute_diff=True,
        compute_ratio=False
    ),
    '0502': ParameterDefinition(
        code='0502',
        name='density',
        units='kg/m³',
        description='Element density',
        column_name='density',
        is_layer_param=True,
        compute_diff=True,
        compute_ratio=True
    ),
    '0503': ParameterDefinition(
        code='0503',
        name='temperature',
        units='°C',
        description='Element temperature',
        column_name='temperature',
        is_layer_param=True,
        compute_diff=True,
        compute_ratio=False
    ),
    '0504': ParameterDefinition(
        code='0504',
        name='element_ID',
        units='',
        description='Unique element identifier',
        column_name='element_ID',
        is_layer_param=True,
        compute_diff=False,
        compute_ratio=False
    ),
    '0505': ParameterDefinition(
        code='0505',
        name='age',
        units='days',
        description='Element age since deposition',
        column_name='age',
        is_layer_param=True,
        compute_diff=True,
        compute_ratio=True
    ),
    '0506': ParameterDefinition(
        code='0506',
        name='lwc',
        units='% vol',
        description='Liquid water content by volume',
        column_name='lwc',
        is_layer_param=True,
        compute_diff=True,
        compute_ratio=True
    ),
    
    # Microstructure parameters
    '0508': ParameterDefinition(
        code='0508',
        name='dendricity',
        units='',
        description='Degree of dendritic structure (0-1)',
        column_name='dendricity',
        is_layer_param=True,
        compute_diff=True,
        compute_ratio=False
    ),
    '0509': ParameterDefinition(
        code='0509',
        name='sphericity',
        units='',
        description='Degree of rounded grains (0-1)',
        column_name='sphericity',
        is_layer_param=True,
        compute_diff=True,
        compute_ratio=False
    ),
    '0510': ParameterDefinition(
        code='0510',
        name='coordination_number',
        units='',
        description='Number of grain-to-grain bonds',
        column_name='coord_number',
        is_layer_param=True,
        compute_diff=True,
        compute_ratio=True
    ),
    '0511': ParameterDefinition(
        code='0511',
        name='bond_size',
        units='mm',
        description='Size of intergranular bonds',
        column_name='bond_size',
        is_layer_param=True,
        compute_diff=True,
        compute_ratio=True
    ),
    '0512': ParameterDefinition(
        code='0512',
        name='grain_size',
        units='mm',
        description='Average grain size',
        column_name='grain_size',
        is_layer_param=True,
        compute_diff=True,
        compute_ratio=True
    ),
    '0513': ParameterDefinition(
        code='0513',
        name='grain_type',
        units='',
        description='Grain type (Swiss Code F1F2F3)',
        column_name='grain_type',
        is_layer_param=True,
        compute_diff=False,  # Categorical
        compute_ratio=False
    ),
    '0535': ParameterDefinition(
        code='0535',
        name='optical_grain_size',
        units='mm',
        description='Optical equivalent grain size',
        column_name='opt_equ_grain_size',
        is_layer_param=True,
        compute_diff=True,
        compute_ratio=True
    ),
    
    # Volumetric fractions
    '0515': ParameterDefinition(
        code='0515',
        name='ice_volume_fraction',
        units='%',
        description='Ice volume fraction',
        column_name='ice_content',
        is_layer_param=True,
        compute_diff=True,
        compute_ratio=True
    ),
    '0516': ParameterDefinition(
        code='0516',
        name='air_volume_fraction',
        units='%',
        description='Air volume fraction (porosity)',
        column_name='air_content',
        is_layer_param=True,
        compute_diff=True,
        compute_ratio=True
    ),
    '0519': ParameterDefinition(
        code='0519',
        name='soil_volume_fraction',
        units='%',
        description='Soil volume fraction',
        column_name='soil_content',
        is_layer_param=True,
        compute_diff=True,
        compute_ratio=True
    ),
    
    # Mechanical properties
    '0517': ParameterDefinition(
        code='0517',
        name='stress',
        units='kPa',
        description='Stress in element',
        column_name='stress',
        is_layer_param=True,
        compute_diff=True,
        compute_ratio=True
    ),
    '0518': ParameterDefinition(
        code='0518',
        name='viscosity',
        units='GPa·s',
        description='Snow viscosity',
        column_name='viscosity',
        is_layer_param=True,
        compute_diff=True,
        compute_ratio=True
    ),
    '0523': ParameterDefinition(
        code='0523',
        name='viscous_deformation_rate',
        units='1e-6/s',
        description='Viscous deformation rate',
        column_name='viscous_deformation_rate',
        is_layer_param=True,
        compute_diff=True,
        compute_ratio=True
    ),
    '0534': ParameterDefinition(
        code='0534',
        name='hand_hardness',
        units='',
        description='Hand hardness index (1-6)',
        column_name='hand_hardness',
        is_layer_param=True,
        compute_diff=True,
        compute_ratio=False
    ),
    '0601': ParameterDefinition(
        code='0601',
        name='shear_strength',
        units='kPa',
        description='Snow shear strength',
        column_name='shear_strength',
        is_layer_param=True,
        compute_diff=True,
        compute_ratio=True
    ),
    
    # Thermal properties
    '0520': ParameterDefinition(
        code='0520',
        name='temperature_gradient',
        units='K/m',
        description='Temperature gradient in layer',
        column_name='temperature_gradient',
        is_layer_param=True,
        compute_diff=True,
        compute_ratio=False
    ),
    '0521': ParameterDefinition(
        code='0521',
        name='thermal_conductivity',
        units='W/(K·m)',
        description='Thermal conductivity',
        column_name='thermal_conductivity',
        is_layer_param=True,
        compute_diff=True,
        compute_ratio=True
    ),
    '0522': ParameterDefinition(
        code='0522',
        name='absorbed_shortwave',
        units='W/m²',
        description='Absorbed shortwave radiation',
        column_name='absorbed_shortwave',
        is_layer_param=True,
        compute_diff=True,
        compute_ratio=False
    ),
    
    # Stability indices
    '0531': ParameterDefinition(
        code='0531',
        name='stability_sdef',
        units='',
        description='Deformation rate stability index',
        column_name='stab_deformation_rate',
        is_layer_param=True,
        compute_diff=True,
        compute_ratio=True
    ),
    '0532': ParameterDefinition(
        code='0532',
        name='stability_sn38',
        units='',
        description='Natural stability index (Sn38)',
        column_name='sn38',
        is_layer_param=True,
        compute_diff=True,
        compute_ratio=True
    ),
    '0533': ParameterDefinition(
        code='0533',
        name='stability_sk38',
        units='',
        description='Stability index Sk38',
        column_name='sk38',
        is_layer_param=True,
        compute_diff=True,
        compute_ratio=True
    ),
    '0604': ParameterDefinition(
        code='0604',
        name='ssi',
        units='',
        description='Structural stability index',
        column_name='ssi',
        is_layer_param=True,
        compute_diff=True,
        compute_ratio=True
    ),
    
    # Interface-specific parameters
    '0602': ParameterDefinition(
        code='0602',
        name='grain_size_difference',
        units='mm',
        description='Grain size difference at interface',
        column_name='gs_difference',
        is_layer_param=True,
        compute_diff=False,  # Already a difference
        compute_ratio=False
    ),
    '0603': ParameterDefinition(
        code='0603',
        name='hardness_difference',
        units='',
        description='Hardness difference at interface',
        column_name='hardness_difference',
        is_layer_param=True,
        compute_diff=False,  # Already a difference
        compute_ratio=False
    ),
    '0605': ParameterDefinition(
        code='0605',
        name='inverse_texture_index',
        units='Mg/m⁴',
        description='Inverse texture index (ITI)',
        column_name='inverse_texture_index',
        is_layer_param=True,
        compute_diff=True,
        compute_ratio=True
    ),
    '0606': ParameterDefinition(
        code='0606',
        name='critical_cut_length',
        units='m',
        description='Critical cut length for crack propagation',
        column_name='critical_cut_length',
        is_layer_param=True,
        compute_diff=True,
        compute_ratio=True
    ),
}


# ---------------------------------------------------------------------------
# Parameter Groups
# ---------------------------------------------------------------------------

def get_parameter_groups() -> Dict[str, List[str]]:
    """
    Organize parameters into functional groups.
    
    Returns:
        Dictionary mapping group name to list of parameter codes
    """
    return {
        'core': ['0501', '0502', '0503', '0504', '0505', '0506'],
        'microstructure': ['0508', '0509', '0510', '0511', '0512', '0513', '0535'],
        'volumetric': ['0515', '0516', '0519'],
        'mechanical': ['0517', '0518', '0523', '0534', '0601'],
        'thermal': ['0520', '0521', '0522'],
        'stability': ['0531', '0532', '0533', '0604'],
        'interface': ['0602', '0603', '0605', '0606'],
    }


def get_essential_parameters() -> List[str]:
    """
    Get list of essential parameters that should always be extracted.
    
    Returns:
        List of parameter codes
    """
    return [
        '0501',  # height
        '0502',  # density
        '0503',  # temperature
        '0504',  # element_ID
        '0506',  # lwc
        '0512',  # grain_size
        '0513',  # grain_type
    ]


def get_parameters_for_differences() -> List[str]:
    """
    Get parameters that should have interface differences computed.
    
    Returns:
        List of parameter codes
    """
    return [
        code for code, param in SNOWPACK_PARAMETERS.items()
        if param.compute_diff
    ]


def get_parameters_for_ratios() -> List[str]:
    """
    Get parameters that should have interface ratios computed.
    
    Returns:
        List of parameter codes
    """
    return [
        code for code, param in SNOWPACK_PARAMETERS.items()
        if param.compute_ratio
    ]


# ---------------------------------------------------------------------------
# Parameter Access Utilities
# ---------------------------------------------------------------------------

def get_parameter_name(code: str) -> str:
    """Get human-readable name for parameter code."""
    return SNOWPACK_PARAMETERS.get(code, ParameterDefinition('', 'unknown', '', '', '')).name


def get_parameter_units(code: str) -> str:
    """Get units for parameter code."""
    return SNOWPACK_PARAMETERS.get(code, ParameterDefinition('', '', 'unknown', '', '')).units


def get_column_name(code: str) -> str:
    """Get DataFrame column name for parameter code."""
    return SNOWPACK_PARAMETERS.get(code, ParameterDefinition('', '', '', '', 'unknown')).column_name


def get_all_column_names() -> List[str]:
    """Get all DataFrame column names."""
    return [param.column_name for param in SNOWPACK_PARAMETERS.values()]


def get_available_parameters(df_columns: Set[str]) -> Dict[str, ParameterDefinition]:
    """
    Filter parameter definitions to only those available in a DataFrame.
    
    Args:
        df_columns: Set of column names in DataFrame
        
    Returns:
        Dictionary of available parameters
    """
    return {
        code: param for code, param in SNOWPACK_PARAMETERS.items()
        if param.column_name in df_columns
    }


# ---------------------------------------------------------------------------
# Feature Name Generators
# ---------------------------------------------------------------------------

def generate_layer_feature_names(prefix: str = 'above') -> List[str]:
    """
    Generate feature names for a layer (above or below).
    
    Args:
        prefix: Prefix for feature names ('above' or 'below')
        
    Returns:
        List of feature names
    """
    return [
        f'{prefix}_{param.name}'
        for param in SNOWPACK_PARAMETERS.values()
        if param.is_layer_param
    ]


def generate_interface_feature_names() -> List[str]:
    """
    Generate feature names for interface properties.
    
    Returns:
        List of interface feature names
    """
    features = []
    
    # Differences
    for code, param in SNOWPACK_PARAMETERS.items():
        if param.compute_diff:
            features.append(f'interface_{param.name}_diff')
    
    # Ratios
    for code, param in SNOWPACK_PARAMETERS.items():
        if param.compute_ratio:
            features.append(f'interface_{param.name}_ratio')
    
    # Gradients (for applicable parameters)
    gradient_params = ['0502', '0503', '0506', '0512']  # density, temp, lwc, grain_size
    for code in gradient_params:
        param = SNOWPACK_PARAMETERS[code]
        features.append(f'interface_{param.name}_gradient')
    
    return features


def get_all_feature_names() -> List[str]:
    """
    Get complete list of all feature names.
    
    Returns:
        List of all feature names
    """
    features = []
    
    # Event metadata
    features.extend([
        'event_id',
        'station_name',
        'pro_file',
        'start_time',
        'stall_height',
        'stall_layer_id',
        'layer_above_id',
        'layer_below_id',
        'feature_extraction_time',
        'lookback_hours',
    ])
    
    # Layer features
    features.extend(generate_layer_feature_names('above'))
    features.extend(generate_layer_feature_names('below'))
    
    # Interface features
    features.extend(generate_interface_feature_names())
    
    # Target variable
    features.append('stalled')
    
    return features


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

def print_parameter_summary():
    """Print summary of all available parameters."""
    print("=" * 80)
    print("SNOWPACK Parameter Configuration")
    print("=" * 80)
    print(f"\nTotal parameters: {len(SNOWPACK_PARAMETERS)}")
    
    groups = get_parameter_groups()
    print(f"\nParameter groups: {len(groups)}")
    for group_name, codes in groups.items():
        print(f"  {group_name:15s}: {len(codes):2d} parameters")
    
    print(f"\nEssential parameters: {len(get_essential_parameters())}")
    print(f"Parameters with differences: {len(get_parameters_for_differences())}")
    print(f"Parameters with ratios: {len(get_parameters_for_ratios())}")
    
    print(f"\nTotal features per event: {len(get_all_feature_names())}")
    print("=" * 80)


if __name__ == '__main__':
    # Print summary when run directly
    print_parameter_summary()
    
    # Example usage
    print("\nExample parameter details:")
    print("-" * 80)
    for code in ['0502', '0506', '0512']:
        param = SNOWPACK_PARAMETERS[code]
        print(f"\nCode {param.code}: {param.name}")
        print(f"  Units: {param.units}")
        print(f"  Column: {param.column_name}")
        print(f"  Compute diff: {param.compute_diff}")
        print(f"  Compute ratio: {param.compute_ratio}")