"""
Unit tests for data loader, ingestion, synthetic generation, and preprocessor.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.data_loader import ACNDataLoader, ENTSOEPriceFetcher, SyntheticDataGenerator
from src.data.preprocessor import SessionPreprocessor


def test_synthetic_data_generator_sessions():
    gen = SyntheticDataGenerator(seed=42)
    df = gen.generate_sessions(num_sessions=50)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50
    assert "arrival_time" in df.columns
    assert "required_energy_kwh" in df.columns
    assert (df["duration_hours"] > 0).all()
    assert (df["required_energy_kwh"] > 0).all()


def test_synthetic_data_generator_prices():
    gen = SyntheticDataGenerator(seed=42)
    df = gen.generate_price_signal(num_days=2, step_minutes=15)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2 * 24 * 4  # 192 quarter-hourly steps
    assert "price_eur_kwh" in df.columns
    assert (df["price_eur_kwh"] > 0).all()


def test_session_preprocessor_fit_transform():
    gen = SyntheticDataGenerator(seed=42)
    df = gen.generate_sessions(num_sessions=30)

    preprocessor = SessionPreprocessor()
    assert len(preprocessor.feature_cols) == 12
    assert "month_sin" in preprocessor.feature_cols
    assert "month_cos" in preprocessor.feature_cols

    X_scaled, y_scaled = preprocessor.fit_transform(df)

    assert X_scaled.shape[0] == 30
    assert X_scaled.shape[1] == 12
    assert y_scaled.shape[0] == 30
    assert y_scaled.shape[1] == 2

    # Test inverse transform
    y_orig = preprocessor.inverse_transform_targets(y_scaled)
    assert y_orig.shape == (30, 2)
    assert np.allclose(y_orig[:, 1], df["required_energy_kwh"].values, atol=1e-3)
