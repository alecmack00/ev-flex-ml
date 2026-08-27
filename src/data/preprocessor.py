"""
Preprocessing, session feature engineering, cyclical temporal encoding, and tensor scaling.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.logger import setup_logger

logger = setup_logger("preprocessor")


class SessionPreprocessor:
    """Extracts machine learning features from raw EV charging session logs and scales input tensors."""

    def __init__(self) -> None:
        """Initializes scaler objects for input features and targets."""
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.is_fitted = False

        self.feature_cols = [
            "arrival_hour_sin",
            "arrival_hour_cos",
            "day_sin",
            "day_cos",
            "month_sin",
            "month_cos",
            "battery_capacity_kwh",
            "initial_soc",
            "target_soc",
            "required_energy_kwh",
            "max_charger_power_kw",
            "is_weekend",
        ]
        self.target_cols = ["duration_hours", "required_energy_kwh"]

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes engineered temporal cyclical encodings and session metrics.

        Args:
            df: Input DataFrame containing session records.

        Returns:
            pd.DataFrame: Augmented DataFrame with feature columns.
        """
        data = df.copy()

        if not pd.api.types.is_datetime64_any_dtype(data["arrival_time"]):
            data["arrival_time"] = pd.to_datetime(data["arrival_time"])

        arrival_hours = data["arrival_time"].dt.hour + data["arrival_time"].dt.minute / 60.0
        days_of_week = data["arrival_time"].dt.dayofweek
        months = data["arrival_time"].dt.month

        # Cyclical temporal features
        data["arrival_hour_sin"] = np.sin(2.0 * np.pi * arrival_hours / 24.0)
        data["arrival_hour_cos"] = np.cos(2.0 * np.pi * arrival_hours / 24.0)

        data["day_sin"] = np.sin(2.0 * np.pi * days_of_week / 7.0)
        data["day_cos"] = np.cos(2.0 * np.pi * days_of_week / 7.0)

        data["month_sin"] = np.sin(2.0 * np.pi * months / 12.0)
        data["month_cos"] = np.cos(2.0 * np.pi * months / 12.0)

        data["is_weekend"] = (days_of_week >= 5).astype(float)

        if "duration_hours" not in data.columns and "departure_time" in data.columns:
            if not pd.api.types.is_datetime64_any_dtype(data["departure_time"]):
                data["departure_time"] = pd.to_datetime(data["departure_time"])
            data["duration_hours"] = (
                data["departure_time"] - data["arrival_time"]
            ).dt.total_seconds() / 3600.0

        return data

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Engineers features and fits standard scalers on training data.

        Args:
            df: Raw training DataFrame.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (Scaled X feature array, Scaled y target array).
        """
        data = self.extract_features(df)
        X = data[self.feature_cols].values.astype(np.float32)
        y = data[self.target_cols].values.astype(np.float32)

        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y)

        self.is_fitted = True
        logger.info(f"Fitted SessionPreprocessor on {len(data)} samples across {len(self.feature_cols)} features.")
        return X_scaled, y_scaled

    def transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transforms validation or test DataFrame using fitted scalers.

        Args:
            df: Input validation/inference DataFrame.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: (Scaled X feature array, Scaled y array or None).
        """
        if not self.is_fitted:
            raise RuntimeError("SessionPreprocessor must be fitted before calling transform().")

        data = self.extract_features(df)
        X = data[self.feature_cols].values.astype(np.float32)
        X_scaled = self.feature_scaler.transform(X)

        y_scaled = None
        if all(col in data.columns for col in self.target_cols):
            y = data[self.target_cols].values.astype(np.float32)
            y_scaled = self.target_scaler.transform(y)

        return X_scaled, y_scaled

    def inverse_transform_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        """Reverts scaled target predictions back to physical units (hours, kWh).

        Args:
            y_scaled: Scaled predictions.

        Returns:
            np.ndarray: Unscaled targets in original units.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before inverse transforming targets.")
        return self.target_scaler.inverse_transform(y_scaled)
