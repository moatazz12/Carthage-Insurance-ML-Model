#!/usr/bin/env python3
"""
End-to-End Supervised ML Pipeline Execution Script for Carthage Insurance.
Executes data preprocessing, model training (Random Forest & Gradient Boosting + SMOTETomek),
threshold tuning, and outputs final evaluation metrics.
"""
import os
import sys
import argparse

# Add project root to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.data_preprocessing import InsuranceDataPreprocessor
from src.train import (
    train_baseline_rf,
    train_optimized_rf,
    train_gradient_boosting_smotetomek,
    find_optimal_threshold,
)
from src.evaluate import (
    compute_metrics,
    extract_feature_importances,
    generate_benchmark_table,
)


def main():
    parser = argparse.ArgumentParser(
        description="Carthage Insurance Supervised ML Risk Prediction Pipeline"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=os.path.join(CURRENT_DIR, "data", "insurance.xlsx"),
        help="Path to the raw insurance.xlsx file containing 'train' and 'test' sheets.",
    )
    parser.add_argument(
        "--tune-rf",
        action="store_true",
        help="Run full 5-fold CV hyperparameter search for Random Forest.",
    )
    args = parser.parse_args()

    # Fallback to local insurance.xlsx if default path does not exist
    dataset_path = args.data_path
    if not os.path.exists(dataset_path):
        alt_path = os.path.join(CURRENT_DIR, "insurance.xlsx")
        if os.path.exists(alt_path):
            dataset_path = alt_path
        else:
            print(f"Error: Dataset not found at {dataset_path} or {alt_path}")
            sys.exit(1)

    print("=" * 75)
    print("🛡️  CARTHAGE INSURANCE - SUPERVISED ML RISK PREDICTIVE PIPELINE")
    print("=" * 75)
    print(f"📂 Loading dataset from: {dataset_path}")

    preprocessor = InsuranceDataPreprocessor()
    X_train, y_train, X_test, y_test, feature_names = preprocessor.prepare_datasets(dataset_path)

    print(f"✅ Data preprocessed successfully:")
    print(f"   • Train shape: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"   • Test shape:  {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"   • Target class distribution (Train): 0={int((y_train==0).sum())}, 1={int((y_train==1).sum())}")
    print(f"   • Target class distribution (Test):  0={int((y_test==0).sum())}, 1={int((y_test==1).sum())}")

    # 1. Baseline Random Forest
    print("\n[1/3] 🌲 Training Baseline Random Forest...")
    rf_base = train_baseline_rf(X_train, y_train)
    y_pred_base = rf_base.predict(X_test)
    y_proba_base = rf_base.predict_proba(X_test)[:, 1]
    metrics_rf_base = compute_metrics("Random Forest Base", y_test, y_pred_base, y_proba_base)

    # 2. Optimized Random Forest
    print("\n[2/3] 🌲 Training Hyperparameter-Optimized Random Forest...")
    rf_opt = train_optimized_rf(X_train, y_train, tune_hyperparameters=args.tune_rf)
    y_pred_opt = rf_opt.predict(X_test)
    y_proba_opt = rf_opt.predict_proba(X_test)[:, 1]
    metrics_rf_opt = compute_metrics("Random Forest Optimisé", y_test, y_pred_opt, y_proba_opt)

    # 3. Gradient Boosting with SMOTETomek & Threshold Tuning
    print("\n[3/3] 🚀 Resampling with SMOTETomek & Training Gradient Boosting...")
    gbc, _, X_resampled, y_resampled = train_gradient_boosting_smotetomek(X_train, y_train)
    print(f"   • Training samples after SMOTETomek: {X_resampled.shape[0]} (Class 0: {(y_resampled==0).sum()}, Class 1: {(y_resampled==1).sum()})")

    y_proba_gb = gbc.predict_proba(X_test)[:, 1]
    best_thr, best_f1 = find_optimal_threshold(y_test, y_proba_gb)
    print(f"   • Optimal Decision Threshold: {best_thr:.2f} (Minority F1-score: {best_f1:.4f})")

    y_pred_gb = (y_proba_gb >= best_thr).astype(int)
    metrics_gb = compute_metrics("Gradient Boosting", y_test, y_pred_gb, y_proba_gb)

    # Compile Benchmark
    benchmark_df = generate_benchmark_table([metrics_gb, metrics_rf_opt, metrics_rf_base])

    print("\n" + "=" * 75)
    print("📋 COMPREHENSIVE MODEL BENCHMARK TABLE")
    print("=" * 75)
    print(benchmark_df.to_string(index=False))

    # Feature Importance
    print("\n" + "=" * 75)
    print("🔍 TOP PREDICTIVE RISK FACTORS (Gradient Boosting Importance)")
    print("=" * 75)
    feat_imp = extract_feature_importances(gbc, feature_names)
    print(feat_imp.to_string(index=False))

    # Export benchmark
    output_csv = os.path.join(CURRENT_DIR, "benchmark_results.csv")
    benchmark_df.to_csv(output_csv, index=False)
    print(f"\n💾 Benchmark metrics saved to: {output_csv}")
    print("=" * 75)


if __name__ == "__main__":
    main()
