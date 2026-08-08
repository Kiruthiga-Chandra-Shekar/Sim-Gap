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

    def align_trajectories(self, sim_df: pd.DataFrame, real_df: pd.DataFrame, fps: float = 50.0) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aligns simulation and real trajectories onto a uniform time grid.
        """
        # Fallback timestamp creation if missing
        if "timestamp" not in sim_df.columns:
            sim_df["timestamp"] = np.arange(len(sim_df)) / fps
        if "timestamp" not in real_df.columns:
            real_df["timestamp"] = np.arange(len(real_df)) / fps

        sim_time = sim_df["timestamp"].values - sim_df["timestamp"].iloc[0]
        real_time = real_df["timestamp"].values - real_df["timestamp"].iloc[0]

        max_duration = min(sim_time[-1], real_time[-1])
        if max_duration <= 0:
            return sim_df, real_df

        uniform_grid = np.arange(0, max_duration, 1.0 / fps)

        sim_aligned = pd.DataFrame({"timestamp": uniform_grid})
        real_aligned = pd.DataFrame({"timestamp": uniform_grid})

        # Interpolate numeric columns onto uniform grid
        for col in sim_df.select_dtypes(include=[np.number]).columns:
            if col != "timestamp":
                sim_aligned[col] = np.interp(uniform_grid, sim_time, sim_df[col].values)

        for col in real_df.select_dtypes(include=[np.number]).columns:
            if col != "timestamp":
                real_aligned[col] = np.interp(uniform_grid, real_time, real_df[col].values)

        return sim_aligned, real_aligned