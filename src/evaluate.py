"""
Evaluation metrics, feature importance extraction, and benchmark reporting
for Carthage Insurance risk predictive modeling.
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
) -> Dict[str, Any]:
    """Compute comprehensive classification performance metrics including confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics: Dict[str, Any] = {
        'Modèle': model_name,
        'AUC': roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision (classe 1)': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'Recall (classe 1)': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'F1-score (classe 1)': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        'Spécificité': specificity,
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp),
    }
    return metrics


def extract_feature_importances(
    model: Any, feature_names: List[str]
) -> pd.DataFrame:
    """Extract and sort feature importances from a tree-based estimator."""
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not expose feature_importances_ attribute.")

    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    df_imp['Percentage'] = (df_imp['Importance'] * 100).round(2)
    return df_imp


def generate_benchmark_table(metrics_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate aligned benchmark comparison DataFrame across all evaluated models."""
    df = pd.DataFrame(metrics_list)
    column_order = [
        'Modèle',
        'AUC',
        'Accuracy',
        'Precision (classe 1)',
        'Recall (classe 1)',
        'F1-score (classe 1)',
        'Spécificité',
        'TN',
        'FP',
        'FN',
        'TP',
    ]
    return df[column_order]
