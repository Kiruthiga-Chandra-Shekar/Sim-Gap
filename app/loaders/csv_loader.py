import os
import pandas as pd
import numpy as np
from app.loaders.base_loader import BaseSim2RealLoader


class CSVLoader(BaseSim2RealLoader):
    """
    Loader for flat CSV telemetry logs. Standardizes time columns
    and maps joint/sensor features into unified DataFrame schemas.
    """

    def __init__(self, default_fps: float = 50.0):
        self.default_fps = default_fps

    def load_sim_data(self, path: str, episode_idx: int = 0) -> pd.DataFrame:
        """Loads simulation CSV telemetry into a normalized DataFrame."""
        return self._parse_csv(path)

    def load_real_data(self, path: str, episode_idx: int = 0) -> pd.DataFrame:
        """Loads real robot CSV telemetry into a normalized DataFrame."""
        return self._parse_csv(path)

    def _parse_csv(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")

        df = pd.read_csv(path)

        # Normalize timestamp column name
        time_cols = [c for c in df.columns if c.lower() in ["time", "timestamp", "t", "step"]]
        if time_cols:
            df.rename(columns={time_cols[0]: "timestamp"}, inplace=True)
        else:
            # Fallback: create uniform time grid using default FPS
            dt = 1.0 / self.default_fps
            df["timestamp"] = np.arange(len(df)) * dt

        return df


if __name__ == "__main__":
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
        tmp.write("time,joint_1_pos,joint_2_pos\n0.0,0.1,0.2\n0.02,0.15,0.25\n")
        tmp_name = tmp.name

    loader = CSVLoader()
    df = loader.load_sim_data(tmp_name)
    print("CSVLoader Test Output:")
    print(df)
    os.remove(tmp_name)