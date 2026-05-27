"""
model_trainer.py
==================

ML model training, tuning, and comparison for wetting front stall prediction.
"""

import logging
import json
import joblib
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    cross_val_score, 
    TimeSeriesSplit,
    RandomizedSearchCV,
    GridSearchCV,
    train_test_split
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available - install with: pip install shap")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    use_time_series_cv: bool = True
    remove_low_variance: bool = True
    variance_threshold: float = 0.01
    remove_correlated: bool = True
    correlation_threshold: float = 0.95
    scale_features: bool = True
    tune_hyperparameters: bool = True
    tuning_method: str = 'random'
    n_iter_random: int = 50
    tuning_cv_folds: int = 3
    models_to_train: List[str] = field(default_factory=lambda: [
        'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm'
    ])
    compute_permutation_importance: bool = True
    compute_shap_values: bool = True

def get_model_configs() -> Dict[str, Dict[str, Any]]:
    configs = {
        'random_forest': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'param_grid': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.3, 0.5],
                'class_weight': ['balanced', None]
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 0.9, 1.0]
            }
        },
        'xgboost': {
            'model': xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss'),
            'param_grid': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2],
                'scale_pos_weight': [1, 2, 3]
            }
        },
        'lightgbm': {
            'model': lgb.LGBMClassifier(random_state=42, n_jobs=1, verbose=-1),
            'param_grid': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10, -1],
                'num_leaves': [31, 50, 70, 100],
                'min_child_samples': [20, 30, 50],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'class_weight': ['balanced', None]
            }
        },
        'logistic_regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
            'param_grid': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'class_weight': ['balanced', None]
            }
        }
    }
    return configs

# ---------------------------------------------------------------------------
# Feature Selection
# ---------------------------------------------------------------------------

class FeatureSelector:
    def __init__(self, variance_threshold: float = 0.01, correlation_threshold: float = 0.95):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
    
    def remove_low_variance(self, X: pd.DataFrame) -> pd.DataFrame:
        if X.isnull().any().any():
            variances = X.fillna(X.median()).var()
        else:
            variances = X.var()
        low_var = variances[variances < self.variance_threshold].index.tolist()
        if low_var:
            logger.info(f"Removed {len(low_var)} low-variance features")
        return X.drop(columns=low_var)
    
    def remove_correlated(self, X: pd.DataFrame) -> pd.DataFrame:
        X_calc = X.fillna(X.median()) if X.isnull().any().any() else X
        corr_matrix = X_calc.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.correlation_threshold)]
        if to_drop:
            logger.info(f"Removed {len(to_drop)} highly correlated features")
        return X.drop(columns=to_drop)

# ---------------------------------------------------------------------------
# Model Trainer
# ---------------------------------------------------------------------------

class ModelTrainer:
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.models_ = {}
        self.results_ = {}
        self.scaler_ = None
        self.imputer_ = None
        self.feature_selector_ = None
        self.feature_names_ = None
        self.best_model_name_ = None
        self.best_model_ = None
        
        self.X_train = pd.DataFrame()
        self.y_train = pd.Series(dtype=float)
        self.X_val = pd.DataFrame()
        self.y_val = pd.Series(dtype=float)
        self.X_test = pd.DataFrame()
        self.y_test = pd.Series(dtype=float)

    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info(f"Initial features: {X.shape[1]}")
        
        # 1. Feature Selection
        self.feature_selector_ = FeatureSelector(
            variance_threshold=self.config.variance_threshold,
            correlation_threshold=self.config.correlation_threshold
        )
        if self.config.remove_low_variance:
            X = self.feature_selector_.remove_low_variance(X)
        if self.config.remove_correlated:
            X = self.feature_selector_.remove_correlated(X)
            
        # 2. Remove 100% NaN Columns
        all_nan_cols = X.columns[X.isna().all()].tolist()
        if all_nan_cols:
            logger.warning(f"Dropping {len(all_nan_cols)} columns that are 100% NaN")
            X = X.drop(columns=all_nan_cols)

        self.feature_names_ = X.columns.tolist()
        logger.info(f"Selected {len(self.feature_names_)} features")

        # 3. Impute
        self.imputer_ = SimpleImputer(strategy='median')
        X_imputed = self.imputer_.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=self.feature_names_, index=X.index)
        
        # 4. Scale
        if self.config.scale_features:
            self.scaler_ = StandardScaler()
            constant_cols = X.columns[X.var() == 0].tolist()
            if constant_cols:
                X = X.drop(columns=constant_cols)
                self.feature_names_ = X.columns.tolist()
            
            if X.shape[1] == 0:
                raise ValueError("No features left after preprocessing.")
                
            X_scaled = self.scaler_.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=self.feature_names_, index=X.index)
            logger.info("Features scaled")
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series):
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.config.test_size,
            random_state=self.config.random_state, stratify=y
        )
        val_size_adj = self.config.validation_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adj,
            random_state=self.config.random_state, stratify=y_temp
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def tune_hyperparameters(self, model, param_grid, X_train, y_train):
        cv = TimeSeriesSplit(n_splits=self.config.tuning_cv_folds) \
             if self.config.use_time_series_cv else self.config.tuning_cv_folds
        
        if self.config.tuning_method == 'random':
            search = RandomizedSearchCV(
                model, param_grid, n_iter=self.config.n_iter_random,
                cv=cv, scoring='roc_auc', n_jobs=-1,
                random_state=self.config.random_state, verbose=1
            )
        else:
            search = GridSearchCV(
                model, param_grid, cv=cv, scoring='roc_auc',
                n_jobs=-1, verbose=1
            )
        search.fit(X_train, y_train)
        return search.best_estimator_
    
    def train_model(self, model_name, X_train, y_train, X_val, y_val):
        logger.info(f"Training {model_name}...")
        configs = get_model_configs()
        model = configs[model_name]['model']
        
        if self.config.tune_hyperparameters:
            model = self.tune_hyperparameters(
                model, configs[model_name]['param_grid'], X_train, y_train
            )
        else:
            model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
        
        res = {
            'model': model,
            'accuracy': accuracy_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_val, y_proba) if y_proba is not None else 0
        }
        logger.info(f"{model_name} Val AUC: {res['roc_auc']:.4f}")
        return res
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ModelTrainer":
        X, y = self.prepare_data(X, y)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        
        for name in self.config.models_to_train:
            try:
                self.results_[name] = self.train_model(name, X_train, y_train, X_val, y_val)
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
        
        best_score = -1
        for name, res in self.results_.items():
            if res['roc_auc'] > best_score:
                best_score = res['roc_auc']
                self.best_model_name_ = name
                self.best_model_ = res['model']
        
        logger.info(f"Best model: {self.best_model_name_}")
        test_pred = self.best_model_.predict(X_test)
        test_proba = self.best_model_.predict_proba(X_test)[:, 1]
        
        self.results_[f'{self.best_model_name_}_test'] = {
            'accuracy': accuracy_score(y_test, test_pred),
            'roc_auc': roc_auc_score(y_test, test_proba),
            'report': classification_report(y_test, test_pred)
        }
        return self
    
    def _preprocess_for_prediction(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names_ is None:
            raise ValueError("Model not trained.")
        missing = set(self.feature_names_) - set(X.columns)
        if missing:
            for c in missing: X[c] = np.nan
        X = X[self.feature_names_]
        if self.imputer_:
            X = pd.DataFrame(self.imputer_.transform(X), columns=self.feature_names_, index=X.index)
        if self.scaler_:
            X = pd.DataFrame(self.scaler_.transform(X), columns=self.feature_names_, index=X.index)
        return X

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.best_model_ is None: raise ValueError("No model loaded.")
        X = self._preprocess_for_prediction(X)
        return self.best_model_.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.best_model_ is None: raise ValueError("No model loaded.")
        X = self._preprocess_for_prediction(X)
        return self.best_model_.predict_proba(X)
    
    def save_model(self, save_dir: Union[str, Path]) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_model_, save_dir / 'model.joblib')
        if self.scaler_: joblib.dump(self.scaler_, save_dir / 'scaler.joblib')
        if self.imputer_: joblib.dump(self.imputer_, save_dir / 'imputer.joblib')
        with open(save_dir / 'feature_names.json', 'w') as f:
            json.dump(self.feature_names_, f, indent=2)
        with open(save_dir / 'metadata.json', 'w') as f:
            json.dump({'best_model': self.best_model_name_}, f, indent=2)
            
    @classmethod
    def load_model(cls, load_dir: Union[str, Path]) -> 'ModelTrainer':
        load_dir = Path(load_dir)
        trainer = cls()
        trainer.best_model_ = joblib.load(load_dir / 'model.joblib')
        if (load_dir / 'scaler.joblib').exists(): trainer.scaler_ = joblib.load(load_dir / 'scaler.joblib')
        if (load_dir / 'imputer.joblib').exists(): trainer.imputer_ = joblib.load(load_dir / 'imputer.joblib')
        with open(load_dir / 'feature_names.json', 'r') as f: trainer.feature_names_ = json.load(f)
        return trainer

# ---------------------------------------------------------------------------
# Feature Importance Analysis
# ---------------------------------------------------------------------------

class FeatureImportanceAnalyzer:
    def __init__(self, model, X, feature_names):
        self.model = model
        self.X = X
        self.feature_names = feature_names
        self.importance_scores_ = {}

    def analyze_all(self, X_val, y_val, compute_shap=True, compute_permutation=True, shap_sample_size=100):
        res = {}
        if hasattr(self.model, 'feature_importances_'):
            res['builtin'] = pd.Series(self.model.feature_importances_, index=self.feature_names).sort_values(ascending=False)
        
        if compute_permutation:
            logger.info("Computing permutation importance...")
            try:
                perm = permutation_importance(self.model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)
                res['permutation'] = pd.Series(perm.importances_mean, index=self.feature_names).sort_values(ascending=False)
            except Exception as e:
                logger.warning(f"Permutation importance failed: {e}")

        if compute_shap and SHAP_AVAILABLE:
            try:
                # Save the specific sample used so we can plot it later!
                X_samp = X_val.sample(min(len(X_val), shap_sample_size), random_state=42)
                
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*LightGBM binary classifier.*")
                    if hasattr(self.model, 'estimators_') or 'XGB' in str(type(self.model)) or 'LGBM' in str(type(self.model)):
                        explainer = shap.TreeExplainer(self.model)
                    else:
                        explainer = shap.KernelExplainer(self.model.predict_proba, X_samp)
                    
                    shap_values = explainer.shap_values(X_samp)
                
                if isinstance(shap_values, list): shap_values = shap_values[1]
                elif len(np.shape(shap_values)) == 3: shap_values = shap_values[:, :, 1]
                
                res['shap'] = pd.Series(np.abs(shap_values).mean(axis=0), index=self.feature_names).sort_values(ascending=False)
                res['shap_values'] = shap_values
                res['shap_data'] = X_samp  # <--- CRITICAL: Return the data matching the values
                res['shap_explainer'] = explainer
            except Exception as e:
                logger.warning(f"SHAP failed: {e}")
        
        self.importance_scores_ = res
        return res

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_model_comparison(results, save_path=None):
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    data = []
    for model_name, res in results.items():
        if 'test' in model_name: continue
        row = {'Model': model_name}
        for m in metrics:
            if m in res: row[m] = res[m]
        data.append(row)
    
    df = pd.DataFrame(data)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, metric in enumerate(metrics):
        if metric in df.columns:
            ax = axes[idx]
            df.sort_values(metric, ascending=False).plot(kind='barh', x='Model', y=metric, ax=ax, legend=False)
            ax.set_title(metric.upper())
    fig.delaxes(axes[-1])
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_feature_importance(importance_dict, top_n=20, save_path=None):
    methods = [k for k in importance_dict.keys() if k not in ['shap_values', 'shap_explainer', 'shap_data']]
    if not methods: return
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 8))
    if len(methods) == 1: axes = [axes]
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        series = importance_dict[method].head(top_n).iloc[::-1]
        ax.barh(range(len(series)), series.values)
        ax.set_yticks(range(len(series)))
        ax.set_yticklabels(series.index)
        ax.set_title(f'{method.title()} Importance')
    
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_shap_summary(shap_values, X, save_path=None):
    """Create SHAP summary plot (Beeswarm)."""
    if not SHAP_AVAILABLE: return None
    fig = plt.figure(figsize=(10, 8))
    # Use the exact X data that generated the shap_values
    shap.summary_plot(shap_values, X, show=False)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved SHAP summary plot to {save_path}")
    return fig

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Model Trainer Module")