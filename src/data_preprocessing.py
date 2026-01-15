"""
Data preprocessing, cleaning, imputation, and feature engineering pipeline
for Carthage Insurance risk predictive modeling.
"""
from typing import Tuple, List, Optional
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, KBinsDiscretizer, OrdinalEncoder
from sklearn.feature_selection import VarianceThreshold


class InsuranceDataPreprocessor:
    """
    Handles end-to-end data preparation including:
    - Loading train and test sets from Excel
    - Dropping empty columns and unnecessary metadata
    - Conditional median imputation for numerical surface areas
    - Intelligent domain-specific imputation for categorical features
    - Outlier-robust scaling using RobustScaler
    - Discretization of temporal features
    - Categorical ordinal encoding
    - Near-zero variance filtering
    """

    def __init__(self, variance_threshold: float = 0.001, random_state: int = 42):
        self.variance_threshold = variance_threshold
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.discretizer = KBinsDiscretizer(
            n_bins=5,
            encode='ordinal',
            strategy='quantile',
            random_state=self.random_state
        )
        self.encoder = OrdinalEncoder()
        self.thresholder = VarianceThreshold(threshold=self.variance_threshold)
        self.median_area_non_residential: Optional[float] = None
        self.median_area_residential: Optional[float] = None
        self.feature_names: List[str] = []

    def load_data(self, filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test sheets from the raw Excel workbook."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found at: {filepath}")

        df_train = pd.read_excel(filepath, sheet_name=0)
        df_test = pd.read_excel(filepath, sheet_name=1)
        return df_train, df_test

    def clean_and_impute(
        self, df_train_raw: pd.DataFrame, df_test_raw: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Perform data cleaning and conditional imputation."""
        df_train = df_train_raw.copy()
        df_test = df_test_raw.copy()

        # 1. Drop columns that are completely empty
        df_train = df_train.dropna(axis=1, how='all')
        df_test = df_test.dropna(axis=1, how='all')

        # 2. Drop risk_score in test if present (not present in train target schema)
        if 'risk_score' in df_test.columns:
            df_test.drop(columns=['risk_score'], inplace=True)

        # 3. Drop rows that are completely empty
        df_train = df_train.dropna(how='all')
        df_test = df_test.dropna(how='all')

        # 4. Conditional median imputation for area_m2 based on residential status
        self.median_area_non_residential = float(
            df_train[df_train['is_residential'] == 0]['area_m2'].median()
        )
        self.median_area_residential = float(
            df_train[df_train['is_residential'] == 1]['area_m2'].median()
        )

        df_train['area_m2'] = df_train.apply(
            lambda r: self.median_area_non_residential
            if pd.isna(r['area_m2']) and r['is_residential'] == 0
            else (
                self.median_area_residential
                if pd.isna(r['area_m2']) and r['is_residential'] == 1
                else r['area_m2']
            ),
            axis=1,
        )

        df_test['area_m2'] = df_test.apply(
            lambda r: self.median_area_non_residential
            if pd.isna(r['area_m2']) and r['is_residential'] == 0
            else (
                self.median_area_residential
                if pd.isna(r['area_m2']) and r['is_residential'] == 1
                else r['area_m2']
            ),
            axis=1,
        )

        # 5. Intelligent domain imputation for has_garden based on locality mapping
        # Locality 'U' (Urban) -> 'V' (has garden / vegetated), 'R' (Rural) -> '0'
        df_train['has_garden'] = df_train['has_garden'].fillna(
            df_train['locality'].map({'U': 'V', 'R': '0'})
        )
        df_test['has_garden'] = df_test['has_garden'].fillna(
            df_test['locality'].map({'U': 'V', 'R': '0'})
        )

        # 6. Drop metadata/identifiers
        cols_to_drop = ['Geo_Code', 'policy_id', 'risk_score']
        for col in cols_to_drop:
            if col in df_train.columns:
                df_train.drop(columns=[col], inplace=True)
            if col in df_test.columns:
                df_test.drop(columns=[col], inplace=True)

        return df_train, df_test

    def transform_features(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply feature scaling, binning, and categorical encoding."""
        train_df = df_train.copy()
        test_df = df_test.copy()

        # 1. RobustScaler for area_m2 (resilient to severe outliers)
        train_df['area_m2'] = self.scaler.fit_transform(train_df[['area_m2']])
        test_df['area_m2'] = self.scaler.transform(test_df[['area_m2']])

        # 2. Window count discrete mapping
        window_mapping = {'without': 0, '>=10': 10}
        train_df['window_count'] = train_df['window_count'].replace(window_mapping).astype(int)
        test_df['window_count'] = test_df['window_count'].replace(window_mapping).astype(int)

        # 3. Discretization of the year feature into 5 quantile-based bins
        train_df['year'] = self.discretizer.fit_transform(train_df[['year']])
        test_df['year'] = self.discretizer.transform(test_df[['year']])

        # 4. Ordinal encoding for categorical columns & claim target
        cat_cols = ['is_finished_and_fenced', 'has_garden', 'locality', 'structure_type']
        if 'claim' in train_df.columns:
            cat_cols.append('claim')

        train_df[cat_cols] = self.encoder.fit_transform(train_df[cat_cols])
        test_df[cat_cols] = self.encoder.transform(test_df[cat_cols])

        return train_df, test_df

    def prepare_datasets(
        self, filepath: str
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
        """Run full preprocessing pipeline returning (X_train, y_train, X_test, y_test, features)."""
        raw_train, raw_test = self.load_data(filepath)
        cleaned_train, cleaned_test = self.clean_and_impute(raw_train, raw_test)
        transformed_train, transformed_test = self.transform_features(cleaned_train, cleaned_test)

        X_train = transformed_train.drop(columns=['claim'])
        y_train = transformed_train['claim'].astype(int)

        X_test = transformed_test.drop(columns=['claim'])
        y_test = transformed_test['claim'].astype(int)

        # Feature selection with VarianceThreshold
        self.thresholder.fit(X_train)
        selected_cols = self.thresholder.get_feature_names_out(X_train.columns).tolist()
        self.feature_names = selected_cols

        X_train_filtered = pd.DataFrame(
            self.thresholder.transform(X_train),
            columns=selected_cols,
            index=X_train.index
        )
        X_test_filtered = pd.DataFrame(
            self.thresholder.transform(X_test),
            columns=selected_cols,
            index=X_test.index
        )

        return X_train_filtered, y_train, X_test_filtered, y_test, selected_cols
