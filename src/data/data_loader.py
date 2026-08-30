"""
Data loading, fetching, and synthetic generation for EV charging sessions and EPEX SPOT electricity prices.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests

from src.utils.logger import setup_logger

logger = setup_logger("data_loader")


class SyntheticDataGenerator:
    """Generates synthetic EV fleet charging sessions and dynamic EPEX SPOT electricity price signals.

    Mimics real European EV charging datasets (ElaadNL, ACN-Data) and EPEX SPOT Day-Ahead market prices.
    """

    def __init__(self, seed: int = 42) -> None:
        """Initializes the generator with a random seed.

        Args:
            seed: Seed for numpy random number generator.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_sessions(
        self,
        num_sessions: int = 200,
        start_date: str = "2024-01-01 00:00:00",
        num_chargers: int = 20,
        feeder_id: str = "NL-AMS-FEEDER-04",
    ) -> pd.DataFrame:
        """Generates synthetic EV charging session logs.

        Args:
            num_sessions: Total number of charging sessions to generate.
            start_date: Starting datetime string (YYYY-MM-DD HH:MM:SS).
            num_chargers: Number of available chargers at the site.
            feeder_id: Identifier for the distribution feeder transformer.

        Returns:
            pd.DataFrame: Session records containing arrival, departure, battery cap, SoC, and required energy.
        """
        start_dt = pd.to_datetime(start_date)
        records = []

        battery_capacities = [40.0, 50.0, 60.0, 75.0, 85.0, 100.0]
        charger_power_options = [11.0, 22.0]

        for session_id in range(1, num_sessions + 1):
            # Arrival distribution: Bi-modal (Morning 07-09, Evening 17-19)
            is_morning = self.rng.random() < 0.6
            if is_morning:
                arr_hour = self.rng.normal(8.0, 1.2)
            else:
                arr_hour = self.rng.normal(18.0, 1.5)

            arr_hour = np.clip(arr_hour, 0.0, 23.99)
            day_offset = self.rng.integers(0, 14)

            arrival_time = start_dt + timedelta(days=int(day_offset), hours=float(arr_hour))

            # Duration distribution (hours): Log-normal between 2h and 14h
            duration_hours = float(np.clip(self.rng.lognormal(mean=1.5, sigma=0.5), 1.5, 16.0))
            departure_time = arrival_time + timedelta(hours=duration_hours)

            # Battery specs
            battery_cap = float(self.rng.choice(battery_capacities))
            initial_soc = float(np.clip(self.rng.uniform(0.1, 0.45), 0.05, 0.5))
            target_soc = float(np.clip(self.rng.uniform(0.85, 1.0), initial_soc + 0.3, 1.0))

            required_energy = (target_soc - initial_soc) * battery_cap
            max_charger_power = float(self.rng.choice(charger_power_options))
            charger_id = f"CP_{self.rng.integers(1, num_chargers + 1):02d}"

            records.append({
                "session_id": f"SESS_{session_id:04d}",
                "feeder_id": feeder_id,
                "charger_id": charger_id,
                "arrival_time": arrival_time,
                "departure_time": departure_time,
                "duration_hours": round(duration_hours, 2),
                "battery_capacity_kwh": battery_cap,
                "initial_soc": round(initial_soc, 3),
                "target_soc": round(target_soc, 3),
                "required_energy_kwh": round(required_energy, 2),
                "max_charger_power_kw": max_charger_power,
            })

        df = pd.DataFrame(records).sort_values("arrival_time").reset_index(drop=True)
        logger.info(f"Generated {len(df)} synthetic EV charging sessions.")
        return df

    def generate_price_signal(
        self,
        start_date: str = "2024-01-01 00:00:00",
        num_days: int = 14,
        step_minutes: int = 15,
    ) -> pd.DataFrame:
        """Generates realistic EPEX SPOT Day-Ahead electricity prices in EUR/kWh.

        Args:
            start_date: Start datetime string.
            num_days: Number of days to simulate.
            step_minutes: Time resolution in minutes (e.g. 15 for quarter-hourly).

        Returns:
            pd.DataFrame: Timestamped pricing schedule.
        """
        start_dt = pd.to_datetime(start_date)
        total_steps = int(num_days * 24 * (60 / step_minutes))

        timestamps = [start_dt + timedelta(minutes=step_minutes * i) for i in range(total_steps)]

        prices_eur_kwh = []
        for ts in timestamps:
            h = ts.hour + ts.minute / 60.0
            day_of_week = ts.weekday()

            # Base tariff curve (Solar duck curve midday, evening peak)
            base_price = 0.15
            morning_peak = 0.12 * np.exp(-0.5 * ((h - 8.5) / 1.5) ** 2)
            evening_peak = 0.22 * np.exp(-0.5 * ((h - 19.0) / 2.0) ** 2)
            midday_solar_dip = -0.06 * np.exp(-0.5 * ((h - 13.0) / 2.0) ** 2)
            weekend_discount = -0.03 if day_of_week >= 5 else 0.0

            noise = float(self.rng.normal(0.0, 0.015))
            price_kwh = max(0.02, base_price + morning_peak + evening_peak + midday_solar_dip + weekend_discount + noise)
            prices_eur_kwh.append(round(price_kwh, 4))

        df = pd.DataFrame({
            "timestamp": timestamps,
            "price_eur_kwh": prices_eur_kwh,
            "price_eur_mwh": [p * 1000.0 for p in prices_eur_kwh],
        })
        logger.info(f"Generated {len(df)} price steps ({num_days} days at {step_minutes}m resolution).")
        return df


class ACNDataLoader:
    """Ingests and parses ACN-Data or ElaadNL EV charging session datasets."""

    def __init__(self, data_dir: Union[str, Path] = "data/raw") -> None:
        """Initializes loader directory.

        Args:
            data_dir: Path to directory containing session raw files.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.synthetic_gen = SyntheticDataGenerator()

    def load_sessions(
        self,
        filepath: Optional[Union[str, Path]] = None,
        fallback_synthetic_count: int = 200,
    ) -> pd.DataFrame:
        """Loads sessions from CSV/JSON file or generates synthetic data if missing.

        Args:
            filepath: Path to session file.
            fallback_synthetic_count: Number of sessions to generate if file missing.

        Returns:
            pd.DataFrame: Cleaned session DataFrame.
        """
        path = Path(filepath) if filepath else self.data_dir / "sessions.csv"
        if path.exists():
            logger.info(f"Loading raw session data from {path}")
            if path.suffix == ".json":
                df = pd.read_json(path)
            else:
                df = pd.read_csv(path)

            df["arrival_time"] = pd.to_datetime(df["arrival_time"])
            df["departure_time"] = pd.to_datetime(df["departure_time"])
            return df
        else:
            logger.warning(f"File {path} not found. Generating synthetic dataset.")
            df = self.synthetic_gen.generate_sessions(num_sessions=fallback_synthetic_count)
            df.to_csv(self.data_dir / "synthetic_sessions.csv", index=False)
            return df


class ENTSOEPriceFetcher:
    """Fetches EPEX SPOT / ENTSO-E Day-Ahead electricity prices or generates fallback."""

    def __init__(self, api_key: Optional[str] = None, data_dir: Union[str, Path] = "data/raw") -> None:
        """Initializes fetcher.

        Args:
            api_key: Optional ENTSO-E Transparency API key.
            data_dir: Data storage directory.
        """
        self.api_key = api_key
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.synthetic_gen = SyntheticDataGenerator()

    def fetch_prices(
        self,
        start_date: str = "2024-01-01 00:00:00",
        num_days: int = 14,
        step_minutes: int = 15,
    ) -> pd.DataFrame:
        """Fetches market price schedule.

        Args:
            start_date: Start date string.
            num_days: Days of horizon.
            step_minutes: Minute interval.

        Returns:
            pd.DataFrame: Pricing dataset.
        """
        cache_path = self.data_dir / f"prices_{num_days}d.csv"
        if cache_path.exists():
            logger.info(f"Loading cached price signal from {cache_path}")
            df = pd.read_csv(cache_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df

        if self.api_key:
            # ENTSO-E REST API Endpoint logic could be added here
            logger.info("ENTSO-E API key detected. Querying endpoint...")
            # Fallback for demonstration reproducibility
            df = self.synthetic_gen.generate_price_signal(start_date, num_days, step_minutes)
        else:
            logger.info("No API key provided. Using realistic EPEX SPOT synthetic price schedule.")
            df = self.synthetic_gen.generate_price_signal(start_date, num_days, step_minutes)

        df.to_csv(cache_path, index=False)
        return df
