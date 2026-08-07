from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseSim2RealLoader(ABC):
    """Abstract Base Class for all trajectory data loaders in SimGap."""

    @abstractmethod
    def load_sim_data(self, path: str, episode_idx: int = 0) -> pd.DataFrame:
        """Loads simulation telemetry into a normalized DataFrame."""
        pass

    @abstractmethod
    def load_real_data(self, path: str, episode_idx: int = 0) -> pd.DataFrame:
        """Loads real robot telemetry into a normalized DataFrame."""
        pass

    def align_trajectories(
        self, sim_df: pd.DataFrame, real_df: pd.DataFrame, target_fps: float = 50.0
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aligns simulation and real trajectories onto a uniform time grid using linear interpolation.
        """
        # Zero-offset timestamps
        sim_time = sim_df["timestamp"].values - sim_df["timestamp"].iloc[0]
        real_time = real_df["timestamp"].values - real_df["timestamp"].iloc[0]

        # Define uniform time grid based on target FPS
        max_duration = min(sim_time[-1], real_time[-1])
        num_samples = int(max_duration * target_fps)
        common_time = np.linspace(0.0, max_duration, num_samples)

        aligned_sim = {"timestamp": common_time}
        aligned_real = {"timestamp": common_time}

        # Interpolate numeric columns common to both DataFrames
        common_cols = [
            col
            for col in sim_df.columns
            if col != "timestamp" and col in real_df.columns
        ]

        for col in common_cols:
            aligned_sim[col] = np.interp(common_time, sim_time, sim_df[col].values)
            aligned_real[col] = np.interp(common_time, real_time, real_df[col].values)

        return pd.DataFrame(aligned_sim), pd.DataFrame(aligned_real)