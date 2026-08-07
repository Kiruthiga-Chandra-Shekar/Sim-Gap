import os
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
from app.loaders.base_loader import BaseSim2RealLoader

class RLDSFanucLoader(BaseSim2RealLoader):
    """
    Loader for Berkeley FANUC Manipulation TFRecord datasets 
    formatted as TensorFlow Datasets (TFDS) RLDS structures.
    """
    def __init__(self, dataset_dir: str, default_fps: float = 50.0):
        self.dataset_dir = dataset_dir
        self.default_fps = default_fps

    def load_real_data(self, split_name: str = "pick_and_place", episode_idx: int = 0) -> pd.DataFrame:
        """
        Loads an episode from the specified split (e.g., 'pick_and_place', 'open_drawer')
        and maps all tensors to flat column names.
        """
        # Load dataset builder pointing to the local directory containing dataset_info.json
        builder = tfds.builder_from_directory(self.dataset_dir)
        
        if split_name not in builder.info.splits:
            available_splits = list(builder.info.splits.keys())
            raise ValueError(f"Split '{split_name}' not found. Available splits: {available_splits}")

        # Load split as a streamable tf.data.Dataset
        ds = builder.as_dataset(split=split_name)
        
        # Extract the target episode
        for curr_idx, episode in enumerate(ds):
            if curr_idx == episode_idx:
                return self._parse_episode_to_dataframe(episode)
                
        raise IndexError(f"Episode index {episode_idx} out of bounds for split '{split_name}'.")

    def load_sim_data(self, path: str, episode_idx: int = 0) -> pd.DataFrame:
        """Not used directly by RLDS loader; simulation logs use MuJoCo/Genesis runners."""
        raise NotImplementedError("Use MuJoCo/Genesis runners or HDF5Loader for simulation traces.")

    def _parse_episode_to_dataframe(self, episode: dict) -> pd.DataFrame:
        records = []
        dt = 1.0 / self.default_fps
        
        # Iterate over RLDS step dictionary
        for step_idx, step in enumerate(episode['steps']):
            obs = step['observation']
            action = step['action'].numpy()
            
            # Extract Tensors from numpy arrays
            joint_state = obs['state'].numpy()                  # 13D
            ee_state = obs['end_effector_state'].numpy()        # 7D
            
            row = {
                'timestamp': round(step_idx * dt, 4),
                
                # --- Joint Positions (1-6) & Status ---
                'joint_1_pos': joint_state[0],
                'joint_2_pos': joint_state[1],
                'joint_3_pos': joint_state[2],
                'joint_4_pos': joint_state[3],
                'joint_5_pos': joint_state[4],
                'joint_6_pos': joint_state[5],
                'gripper_status': joint_state[6],
                
                # --- Joint Velocities (1-6) ---
                'joint_1_vel': joint_state[7],
                'joint_2_vel': joint_state[8],
                'joint_3_vel': joint_state[9],
                'joint_4_vel': joint_state[10],
                'joint_5_vel': joint_state[11],
                'joint_6_vel': joint_state[12],
                
                # --- End Effector State [x, y, z, qx, qy, qz, qw] ---
                'ee_x': ee_state[0],
                'ee_y': ee_state[1],
                'ee_z': ee_state[2],
                'ee_qx': ee_state[3],
                'ee_qy': ee_state[4],
                'ee_qz': ee_state[5],
                'ee_qw': ee_state[6],
                
                # --- Command Actions [dx, dy, dz, droll, dpitch, dyaw] ---
                'action_dx': action[0],
                'action_dy': action[1],
                'action_dz': action[2],
                'action_droll': action[3],
                'action_dpitch': action[4],
                'action_dyaw': action[5],
            }
            
            records.append(row)
            
        return pd.DataFrame(records)