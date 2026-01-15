"""
Model training and optimization pipeline for Carthage Insurance predictive model.
Implements Random Forest and Gradient Boosting with SMOTETomek resampling
and decision threshold optimization.
"""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.combine import SMOTETomek


def train_baseline_rf(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> RandomForestClassifier:
    """Train baseline uncalibrated Random Forest model with default tree settings."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_optimized_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune_hyperparameters: bool = False,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train hyperparameter-optimized Random Forest model.
    Uses pre-validated optimal grid params (max_depth=5, n_estimators=200, min_samples_leaf=2)
    or executes 5-fold CV grid search if requested.
    """
    if tune_hyperparameters:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
        }
        grid = GridSearchCV(
            estimator=RandomForestClassifier(random_state=random_state),
            param_grid=param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        return grid.best_estimator_

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting_smotetomek(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Tuple[GradientBoostingClassifier, SMOTETomek, pd.DataFrame, pd.Series]:
    """
    Resample unbalanced training set with SMOTETomek (oversampling + cleaning Tomek links)
    and train regularized Gradient Boosting Classifier.
    """
    smote_tomek = SMOTETomek(random_state=random_state)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

    gbc = GradientBoostingClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.9,
        max_features='sqrt',
        random_state=random_state,
    )
    gbc.fit(X_resampled, y_resampled)
    return gbc, smote_tomek, X_resampled, y_resampled


def find_optimal_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    start: float = 0.25,
    stop: float = 0.51,
    step: float = 0.02,
) -> Tuple[float, float]:
    """Search for decision threshold that maximizes minority class (claim=1) F1-score."""
    from sklearn.metrics import precision_recall_fscore_support

    best_thr = 0.50
    best_f1 = -1.0

    for thr in np.arange(start, stop, step):
        y_pred = (y_proba >= thr).astype(int)
        _, _, f1s, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1], zero_division=0
        )
        if f1s[1] > best_f1:
            best_f1 = float(f1s[1])
            best_thr = float(thr)

    return round(best_thr, 2), best_f1
