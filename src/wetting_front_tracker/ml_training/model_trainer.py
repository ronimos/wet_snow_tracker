"""
model_trainer.py
==================

ML model training, tuning, and comparison for wetting front stall prediction.

This module provides:
- Multiple model training and comparison
- Hyperparameter tuning
- Feature selection (statistical, permutation, SHAP)
- Cross-validation with proper time-series handling
- Model evaluation and visualization

Author: Ron Simenhois
Created: November 2025
"""

import logging
from dataclasses import dataclass, field
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
    GridSearchCV
)
from sklearn.preprocessing import StandardScaler
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
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb

# Optional imports for advanced features
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
    """Configuration for model training."""
    
    # Data splitting
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    
    # Cross-validation
    cv_folds: int = 5
    use_time_series_cv: bool = True  # Important for temporal data
    
    # Feature selection
    remove_low_variance: bool = True
    variance_threshold: float = 0.01
    remove_correlated: bool = True
    correlation_threshold: float = 0.95
    
    # Scaling
    scale_features: bool = True
    
    # Hyperparameter tuning
    tune_hyperparameters: bool = True
    tuning_method: str = 'random'  # 'random' or 'grid'
    n_iter_random: int = 50
    tuning_cv_folds: int = 3
    
    # Model selection
    models_to_train: List[str] = field(default_factory=lambda: [
        'random_forest',
        'gradient_boosting', 
        'xgboost',
        'lightgbm',
        'logistic_regression'
    ])
    
    # Feature importance
    compute_permutation_importance: bool = True
    compute_shap_values: bool = True
    n_permutations: int = 10


# ---------------------------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------------------------

def get_model_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get default model configurations and hyperparameter search spaces.
    
    Returns:
        Dictionary mapping model names to their configs
    """
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
            'model': xgb.XGBClassifier(
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            ),
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
            'model': lgb.LGBMClassifier(
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
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
            'model': LogisticRegression(
                random_state=42,
                max_iter=1000,
                n_jobs=-1
            ),
            'param_grid': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet', None],
                'solver': ['liblinear', 'saga'],
                'class_weight': ['balanced', None]
            }
        },
        'extra_trees': {
            'model': ExtraTreesClassifier(random_state=42, n_jobs=-1),
            'param_grid': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.3],
                'class_weight': ['balanced', None]
            }
        }
    }
    
    return configs


# ---------------------------------------------------------------------------
# Feature Selection
# ---------------------------------------------------------------------------

class FeatureSelector:
    """
    Comprehensive feature selection using multiple strategies.
    """
    
    def __init__(
        self,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95
    ):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.removed_features_ = {}
        self.feature_stats_ = {}
    
    def remove_low_variance(
        self,
        X: pd.DataFrame,
        y: pd.Series = None
    ) -> pd.DataFrame:
        """Remove features with low variance."""
        variances = X.var()
        low_var = variances[variances < self.variance_threshold].index.tolist()
        
        self.removed_features_['low_variance'] = low_var
        self.feature_stats_['variances'] = variances
        
        logger.info(f"Removed {len(low_var)} low-variance features")
        return X.drop(columns=low_var)
    
    def remove_correlated(
        self,
        X: pd.DataFrame,
        y: pd.Series = None
    ) -> pd.DataFrame:
        """Remove highly correlated features."""
        corr_matrix = X.corr().abs()
        
        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation > threshold
        to_drop = [
            column for column in upper.columns
            if any(upper[column] > self.correlation_threshold)
        ]
        
        self.removed_features_['high_correlation'] = to_drop
        self.feature_stats_['correlation_matrix'] = corr_matrix
        
        logger.info(f"Removed {len(to_drop)} highly correlated features")
        return X.drop(columns=to_drop)
    
    def get_feature_importance_scores(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'mutual_info'
    ) -> pd.Series:
        """
        Get feature importance scores using various methods.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: 'mutual_info', 'chi2', or 'f_classif'
        """
        from sklearn.feature_selection import (
            mutual_info_classif,
            chi2,
            f_classif
        )
        
        if method == 'mutual_info':
            scores = mutual_info_classif(X, y, random_state=42)
        elif method == 'chi2':
            # Ensure non-negative features for chi2
            X_nonneg = X - X.min() + 1e-5
            scores, _ = chi2(X_nonneg, y)
        elif method == 'f_classif':
            scores, _ = f_classif(X, y)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return pd.Series(scores, index=X.columns).sort_values(ascending=False)
    
    def select_k_best(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: int = 50,
        method: str = 'mutual_info'
    ) -> pd.DataFrame:
        """Select top k features by importance score."""
        scores = self.get_feature_importance_scores(X, y, method)
        top_features = scores.head(k).index.tolist()
        
        removed = set(X.columns) - set(top_features)
        self.removed_features_[f'select_{k}_best'] = list(removed)
        self.feature_stats_[f'{method}_scores'] = scores
        
        logger.info(f"Selected top {k} features by {method}")
        return X[top_features]


# ---------------------------------------------------------------------------
# Model Trainer
# ---------------------------------------------------------------------------

class ModelTrainer:
    """
    Train, tune, and compare multiple ML models.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config or ModelConfig()
        self.models_ = {}
        self.results_ = {}
        self.scaler_ = None
        self.feature_selector_ = None
        self.feature_names_ = None
        self.best_model_name_ = None
        self.best_model_ = None
    
    def prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_selection: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_selection: Whether to perform feature selection
            
        Returns:
            Processed X and y
        """
        logger.info(f"Initial features: {X.shape[1]}")
        
        # Handle missing values
        if X.isnull().any().any():
            logger.warning("Missing values detected - imputing with median")
            X = X.fillna(X.median())
        
        # Feature selection
        if feature_selection:
            self.feature_selector_ = FeatureSelector(
                variance_threshold=self.config.variance_threshold,
                correlation_threshold=self.config.correlation_threshold
            )
            
            if self.config.remove_low_variance:
                X = self.feature_selector_.remove_low_variance(X, y)
            
            if self.config.remove_correlated:
                X = self.feature_selector_.remove_correlated(X, y)
            
            logger.info(f"After feature selection: {X.shape[1]} features")
        
        # Scale features
        if self.config.scale_features:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            logger.info("Features scaled")
        
        self.feature_names_ = X.columns.tolist()
        return X, y
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
               pd.Series, pd.Series, pd.Series]:
        """
        Split data into train/val/test sets.
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = self.config.validation_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.config.random_state,
            stratify=y_temp
        )
        
        logger.info(f"Data split: train={len(X_train)}, "
                   f"val={len(X_val)}, test={len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def tune_hyperparameters(
        self,
        model,
        param_grid: Dict,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ):
        """
        Tune model hyperparameters.
        
        Returns:
            Best estimator
        """
        cv = TimeSeriesSplit(n_splits=self.config.tuning_cv_folds) \
             if self.config.use_time_series_cv \
             else self.config.tuning_cv_folds
        
        if self.config.tuning_method == 'random':
            search = RandomizedSearchCV(
                model,
                param_grid,
                n_iter=self.config.n_iter_random,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=self.config.random_state,
                verbose=1
            )
        else:  # grid
            search = GridSearchCV(
                model,
                param_grid,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
        
        search.fit(X_train, y_train)
        logger.info(f"Best params: {search.best_params_}")
        logger.info(f"Best CV score: {search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def train_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """Train a single model and evaluate it."""
        logger.info(f"Training {model_name}...")
        
        model_configs = get_model_configs()
        if model_name not in model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = model_configs[model_name]
        model = config['model']
        
        # Tune hyperparameters if requested
        if self.config.tune_hyperparameters:
            model = self.tune_hyperparameters(
                model,
                config['param_grid'],
                X_train,
                y_train
            )
        else:
            model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
        
        results = {
            'model': model,
            'model_name': model_name,
            'params': model.get_params(),
            'accuracy': accuracy_score(y_val, y_val_pred),
            'precision': precision_score(y_val, y_val_pred, zero_division=0),
            'recall': recall_score(y_val, y_val_pred, zero_division=0),
            'f1': f1_score(y_val, y_val_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_val, y_val_pred)
        }
        
        if y_val_proba is not None:
            results['roc_auc'] = roc_auc_score(y_val, y_val_proba)
            results['y_val_proba'] = y_val_proba
        
        logger.info(f"{model_name} - Val ROC-AUC: {results.get('roc_auc', 'N/A'):.4f}")
        
        return results
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Dict[str, Any]]:
        """Train all configured models."""
        results = {}
        
        for model_name in self.config.models_to_train:
            try:
                results[model_name] = self.train_model(
                    model_name,
                    X_train,
                    y_train,
                    X_val,
                    y_val
                )
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        return results
    
    def select_best_model(
        self,
        results: Dict[str, Dict[str, Any]],
        metric: str = 'roc_auc'
    ) -> Tuple[str, Any]:
        """Select the best model based on a metric."""
        best_score = -np.inf
        best_name = None
        
        for name, res in results.items():
            if metric in res and res[metric] > best_score:
                best_score = res[metric]
                best_name = name
        
        logger.info(f"Best model: {best_name} ({metric}={best_score:.4f})")
        return best_name, results[best_name]['model']
    
    def evaluate_test_set(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """Final evaluation on test set."""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        if y_proba is not None:
            results['roc_auc'] = roc_auc_score(y_test, y_proba)
        
        return results
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> "ModelTrainer":
        """
        Complete training pipeline.
        
        Args:
            X: Feature matrix
            y: Target variable (binary: 0=no stall, 1=stall)
            
        Returns:
            Self for chaining
        """
        logger.info("Starting model training pipeline")
        
        # Prepare data
        X, y = self.prepare_data(X, y)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Train all models
        self.results_ = self.train_all_models(X_train, y_train, X_val, y_val)
        
        # Select best model
        self.best_model_name_, self.best_model_ = self.select_best_model(
            self.results_
        )
        
        # Final evaluation on test set
        test_results = self.evaluate_test_set(self.best_model_, X_test, y_test)
        self.results_[f'{self.best_model_name_}_test'] = test_results
        
        logger.info(f"Test set results:\n{test_results['classification_report']}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the best model."""
        if self.best_model_ is None:
            raise ValueError("No model trained. Call fit() first.")
        
        # Apply same preprocessing
        if self.scaler_ is not None:
            X = pd.DataFrame(
                self.scaler_.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        return self.best_model_.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using the best model."""
        if self.best_model_ is None:
            raise ValueError("No model trained. Call fit() first.")
        
        # Apply same preprocessing
        if self.scaler_ is not None:
            X = pd.DataFrame(
                self.scaler_.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        return self.best_model_.predict_proba(X)


# ---------------------------------------------------------------------------
# Feature Importance Analysis
# ---------------------------------------------------------------------------

class FeatureImportanceAnalyzer:
    """
    Analyze feature importance using multiple methods including SHAP.
    """
    
    def __init__(self, model, X: pd.DataFrame, feature_names: List[str]):
        """
        Initialize analyzer.
        
        Args:
            model: Trained model
            X: Feature matrix used for training
            feature_names: Names of features
        """
        self.model = model
        self.X = X
        self.feature_names = feature_names
        self.importance_scores_ = {}
    
    def get_builtin_importance(self) -> Optional[pd.Series]:
        """Get built-in feature importance if available."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return pd.Series(importances, index=self.feature_names).sort_values(ascending=False)
        return None
    
    def get_permutation_importance(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_repeats: int = 10
    ) -> pd.DataFrame:
        """
        Calculate permutation importance.
        
        Returns:
            DataFrame with importance scores and std
        """
        from sklearn.inspection import permutation_importance
        
        logger.info("Computing permutation importance...")
        result = permutation_importance(
            self.model,
            X_val,
            y_val,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        return importance_df
    
    def get_shap_values(
        self,
        X_sample: Optional[pd.DataFrame] = None,
        sample_size: int = 100
    ) -> Tuple[Any, np.ndarray]:
        """
        Calculate SHAP values.
        
        Args:
            X_sample: Sample data for SHAP (use subset for speed)
            sample_size: Number of samples to use
            
        Returns:
            Tuple of (explainer, shap_values)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Install with: pip install shap")
        
        # Sample data for computational efficiency
        if X_sample is None:
            if len(self.X) > sample_size:
                X_sample = self.X.sample(n=sample_size, random_state=42)
            else:
                X_sample = self.X
        
        logger.info(f"Computing SHAP values for {len(X_sample)} samples...")
        
        # Choose appropriate explainer based on model type
        if isinstance(self.model, (RandomForestClassifier, ExtraTreesClassifier, 
                                  GradientBoostingClassifier)):
            explainer = shap.TreeExplainer(self.model)
        elif isinstance(self.model, (xgb.XGBClassifier, lgb.LGBMClassifier)):
            explainer = shap.TreeExplainer(self.model)
        else:
            # Use KernelExplainer for other models (slower)
            explainer = shap.KernelExplainer(
                self.model.predict_proba,
                shap.sample(self.X, 100)
            )
        
        shap_values = explainer.shap_values(X_sample)
        
        # For binary classification, some models return list of arrays
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        return explainer, shap_values
    
    def get_shap_feature_importance(
        self,
        shap_values: np.ndarray,
        X_sample: pd.DataFrame
    ) -> pd.Series:
        """
        Get feature importance from SHAP values.
        
        Args:
            shap_values: SHAP values array
            X_sample: Sample used for SHAP
            
        Returns:
            Series of mean absolute SHAP values per feature
        """
        # Mean absolute SHAP value for each feature
        importance = np.abs(shap_values).mean(axis=0)
        return pd.Series(importance, index=X_sample.columns).sort_values(ascending=False)
    
    def analyze_all(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        compute_shap: bool = True,
        shap_sample_size: int = 100
    ) -> Dict[str, Any]:
        """
        Run all feature importance analyses.
        
        Returns:
            Dictionary with all importance scores
        """
        results = {}
        
        # Built-in importance
        builtin = self.get_builtin_importance()
        if builtin is not None:
            results['builtin'] = builtin
            logger.info(f"Top 5 features (built-in): {builtin.head().to_dict()}")
        
        # Permutation importance
        perm_importance = self.get_permutation_importance(X_val, y_val)
        results['permutation'] = perm_importance
        logger.info(f"Top 5 features (permutation): "
                   f"{perm_importance.head()['feature'].tolist()}")
        
        # SHAP values
        if compute_shap and SHAP_AVAILABLE:
            try:
                explainer, shap_values = self.get_shap_values(
                    X_sample=X_val.head(shap_sample_size)
                )
                shap_importance = self.get_shap_feature_importance(
                    shap_values,
                    X_val.head(shap_sample_size)
                )
                results['shap'] = shap_importance
                results['shap_values'] = shap_values
                results['shap_explainer'] = explainer
                logger.info(f"Top 5 features (SHAP): {shap_importance.head().to_dict()}")
            except Exception as e:
                logger.error(f"Error computing SHAP values: {e}")
        
        self.importance_scores_ = results
        return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_model_comparison(
    results: Dict[str, Dict[str, Any]],
    save_path: Optional[Path] = None
):
    """Plot comparison of model performances."""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # Prepare data
    data = []
    for model_name, res in results.items():
        if 'test' in model_name:
            continue
        row = {'Model': model_name}
        for metric in metrics:
            if metric in res:
                row[metric] = res[metric]
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        if metric in df.columns:
            ax = axes[idx]
            df_sorted = df.sort_values(metric, ascending=False)
            ax.barh(df_sorted['Model'], df_sorted[metric])
            ax.set_xlabel(metric.upper())
            ax.set_title(f'Model Comparison - {metric.upper()}')
            ax.grid(axis='x', alpha=0.3)
    
    # Remove empty subplot
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved model comparison plot to {save_path}")
    
    return fig


def plot_feature_importance(
    importance_dict: Dict[str, Any],
    top_n: int = 20,
    save_path: Optional[Path] = None
):
    """Plot feature importance from multiple methods."""
    n_methods = len([k for k in importance_dict.keys() 
                     if k not in ['shap_values', 'shap_explainer']])
    
    fig, axes = plt.subplots(1, n_methods, figsize=(7 * n_methods, 8))
    if n_methods == 1:
        axes = [axes]
    
    idx = 0
    
    # Built-in importance
    if 'builtin' in importance_dict:
        ax = axes[idx]
        top_features = importance_dict['builtin'].head(top_n)
        ax.barh(range(len(top_features)), top_features.values)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features.index)
        ax.set_xlabel('Importance')
        ax.set_title('Built-in Feature Importance')
        ax.invert_yaxis()
        idx += 1
    
    # Permutation importance
    if 'permutation' in importance_dict:
        ax = axes[idx]
        perm_df = importance_dict['permutation'].head(top_n)
        ax.barh(range(len(perm_df)), perm_df['importance_mean'].values)
        ax.set_yticks(range(len(perm_df)))
        ax.set_yticklabels(perm_df['feature'].values)
        ax.set_xlabel('Importance')
        ax.set_title('Permutation Feature Importance')
        ax.invert_yaxis()
        idx += 1
    
    # SHAP importance
    if 'shap' in importance_dict:
        ax = axes[idx]
        top_features = importance_dict['shap'].head(top_n)
        ax.barh(range(len(top_features)), top_features.values)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features.index)
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title('SHAP Feature Importance')
        ax.invert_yaxis()
        idx += 1
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {save_path}")
    
    return fig


def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    save_path: Optional[Path] = None
):
    """Create SHAP summary plot."""
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available")
        return None
    
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved SHAP summary plot to {save_path}")
    
    return fig


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example workflow
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("ML Model Training and Comparison Module")
    print("=" * 80)
    print("\nThis module provides:")
    print("  • Multiple model training and tuning")
    print("  • Feature selection (statistical + SHAP)")
    print("  • Cross-validation")
    print("  • Comprehensive evaluation")
    print("  • Feature importance analysis")
    print("=" * 80)
