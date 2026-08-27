"""
house_pipeline_utils.py
Production utilities and custom transformers for the House Price Prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class RawHouseFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom Feature Engineering Transformer for raw residential property data.
    
    Transforms raw house attributes into high-signal valuation features:
    1. Total usable square footage (TotalSF)
    2. Combined total bathrooms (TotalBath)
    3. Property age and remodel age at time of sale (HouseAge, RemodAge)
    4. Remodel status indicator (IsRemodeled)
    5. Total outdoor porch/deck area (TotalPorchSF)
    6. Quality-size interaction index (Qual_x_SF)
    """
    def __init__(self):
        self.drop_cols = ["Id", "Alley", "PoolQC", "Fence", "MiscFeature", "FireplaceQu", "MasVnrType"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df_out = X.copy()
        
        # Drop non-predictive or high-missingness columns if present
        cols_to_drop = [c for c in self.drop_cols if c in df_out.columns]
        if cols_to_drop:
            df_out = df_out.drop(columns=cols_to_drop)
            
        # 1. Total Usable Square Footage (Basement + 1st Floor + 2nd Floor)
        df_out["TotalSF"] = (
            df_out["TotalBsmtSF"].fillna(0) +
            df_out["1stFlrSF"].fillna(0) +
            df_out["2ndFlrSF"].fillna(0)
        )
        
        # 2. Combined Total Bathrooms (Full + 0.5 * Half across above-grade and basement)
        df_out["TotalBath"] = (
            df_out["FullBath"].fillna(0) +
            0.5 * df_out["HalfBath"].fillna(0) +
            df_out["BsmtFullBath"].fillna(0) +
            0.5 * df_out["BsmtHalfBath"].fillna(0)
        )
        
        # 3. Property Age and Remodeling Age at Time of Sale
        df_out["HouseAge"] = (df_out["YrSold"] - df_out["YearBuilt"]).clip(lower=0)
        df_out["RemodAge"] = (df_out["YrSold"] - df_out["YearRemodAdd"]).clip(lower=0)
        df_out["IsRemodeled"] = (df_out["YearRemodAdd"] != df_out["YearBuilt"]).astype(int)
        
        # 4. Total Outdoor Porch / Deck Area
        df_out["TotalPorchSF"] = (
            df_out["WoodDeckSF"].fillna(0) +
            df_out["OpenPorchSF"].fillna(0) +
            df_out["EnclosedPorch"].fillna(0) +
            df_out["3SsnPorch"].fillna(0) +
            df_out["ScreenPorch"].fillna(0)
        )
        
        # 5. Quality-Size Interaction Feature
        df_out["Qual_x_SF"] = df_out["OverallQual"] * df_out["TotalSF"]
        
        return df_out
