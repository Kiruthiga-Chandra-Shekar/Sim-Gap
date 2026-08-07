import os
import h5py
import numpy as np
import pandas as pd
from app.loaders.base_loader import BaseSim2RealLoader


class HDF5Loader(BaseSim2RealLoader):
    """
    Parser for HDF5 trajectory files supporting both:
    1. Robomimic schema: data/demo_<idx>/[obs, actions, states, ...]
    2. LeRobot schema: observation.state, action, timestamp, etc.
    """

    def __init__(self, default_fps: float = 50.0):
        self.default_fps = default_fps

    def load_sim_data(self, path: str, episode_idx: int = 0) -> pd.DataFrame:
        """Loads simulation HDF5 telemetry into a normalized DataFrame."""
        return self._parse_hdf5(path, episode_idx=episode_idx)

    def load_real_data(self, path: str, episode_idx: int = 0) -> pd.DataFrame:
        """Loads real robot HDF5 telemetry into a normalized DataFrame."""
        return self._parse_hdf5(path, episode_idx=episode_idx)

    def _parse_hdf5(self, path: str, episode_idx: int = 0) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"HDF5 file not found: {path}")

        with h5py.File(path, "r") as f:
            # Auto-detect schema format
            if "data" in f and isinstance(f["data"], h5py.Group):
                return self._parse_robomimic(f, episode_idx)
            else:
                return self._parse_lerobot(f, episode_idx)

    def _parse_robomimic(self, f: h5py.File, episode_idx: int) -> pd.DataFrame:
        """
        Parses Robomimic structured HDF5 files.
        Structure: data/demo_<idx>/obs, actions, rewards, etc.
        """
        demos = list(f["data"].keys())
        if not demos:
            raise ValueError("No demos found under 'data/' group in Robomimic HDF5 file.")

        # Resolve demo key
        demo_key = f"demo_{episode_idx}" if f"demo_{episode_idx}" in demos else demos[min(episode_idx, len(demos)-1)]
        demo_group = f["data"][demo_key]

        records = {}
        num_samples = None

        # Extract Robomimic actions
        if "actions" in demo_group:
            actions = np.array(demo_group["actions"])
            num_samples = len(actions)
            for i in range(actions.shape[1]):
                records[f"action_{i}"] = actions[:, i]

        # Extract observations (joint positions, velocities, eef pose)
        if "obs" in demo_group:
            obs_group = demo_group["obs"]
            for key in obs_group.keys():
                data = np.array(obs_group[key])
                if num_samples is None:
                    num_samples = len(data)

                # Flatten 2D matrices (e.g. joint states [N, 6]) into individual columns
                if data.ndim == 1:
                    records[key] = data
                elif data.ndim == 2:
                    for dim in range(data.shape[1]):
                        col_name = self._map_joint_column_name(key, dim)
                        records[col_name] = data[:, dim]

        # Handle Timestamps
        if "timestamps" in demo_group:
            records["timestamp"] = np.array(demo_group["timestamps"])
        elif num_samples is not None:
            dt = 1.0 / self.default_fps
            records["timestamp"] = np.arange(num_samples) * dt
        else:
            raise ValueError(f"Could not determine sequence length for {demo_key}")

        return pd.DataFrame(records)

    def _parse_lerobot(self, f: h5py.File, episode_idx: int) -> pd.DataFrame:
        """
        Parses LeRobot format HDF5 files.
        Structure: Root level datasets (observation.state, action, timestamp, episode_index).
        """
        records = {}
        num_samples = None

        # Check if episode indexing dataset exists
        if "episode_index" in f:
            ep_indices = np.array(f["episode_index"]).flatten()
            mask = ep_indices == episode_idx
            if not np.any(mask):
                mask = ep_indices == ep_indices[0]  # Fallback to first available episode
        else:
            mask = None

        for key in f.keys():
            item = f[key]
            if isinstance(item, h5py.Dataset):
                data = np.array(item)

                # Filter by episode mask if present
                if mask is not None and len(data) == len(mask):
                    data = data[mask]

                # Skip non-time-series scalar metadata
                if data.ndim == 0 or (data.ndim == 1 and len(data) == 1 and key not in ["timestamp", "time"]):
                    continue

                if num_samples is None and data.ndim > 0:
                    num_samples = len(data)

                clean_key = key.replace("/", ".").strip(".")

                if data.ndim == 1:
                    records[clean_key] = data
                elif data.ndim == 2:
                    for dim in range(data.shape[1]):
                        col_name = self._map_joint_column_name(clean_key, dim)
                        records[col_name] = data[:, dim]

        # Normalize timestamp field
        if "timestamp" in records:
            records["timestamp"] = records.pop("timestamp")
        elif "time" in records:
            records["timestamp"] = records.pop("time")
        elif num_samples is not None:
            dt = 1.0 / self.default_fps
            records["timestamp"] = np.arange(num_samples) * dt

        return pd.DataFrame(records)

    @staticmethod
    def _map_joint_column_name(base_key: str, index: int) -> str:
        """Standardizes joint and state vector keys across different schemas."""
        key_lower = base_key.lower()
        if "joint" in key_lower and "pos" in key_lower:
            return f"joint_{index+1}_pos"
        elif "joint" in key_lower and "vel" in key_lower:
            return f"joint_{index+1}_vel"
        elif "state" in key_lower:
            return f"joint_{index+1}_pos"
        elif "action" in key_lower:
            return f"action_{index+1}"
        else:
            return f"{base_key}_{index}"


if __name__ == "__main__":
    import tempfile
    import os
    import numpy as np
    import h5py

    # 1. Create a named temporary file
    tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    tmp_path = tmp.name
    tmp.close()  # Release the Windows file handle immediately

    try:
        # 2. Populate test HDF5 file structure
        with h5py.File(tmp_path, "w") as f:
            demo = f.create_group("data/demo_0")
            demo.create_dataset("actions", data=np.random.randn(100, 6))
            obs = demo.create_group("obs")
            obs.create_dataset("joint_states", data=np.random.randn(100, 6))

        # 3. Test Loader
        loader = HDF5Loader()
        df = loader.load_real_data(tmp_path, episode_idx=0)

        print("Successfully parsed test HDF5 file!")
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")

    finally:
        # 4. Clean up temporary file after handle is safely closed
        if os.path.exists(tmp_path):
            os.remove(tmp_path)