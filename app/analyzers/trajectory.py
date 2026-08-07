import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


class TrajectoryAnalyzer:
    """
    Kinematic Sim2Real Gap Analyzer.
    Computes trajectory alignment error, DTW distance, and joint-level drift
    between simulation and real robot telemetry.
    """

    def __init__(self):
        pass

    def compute_gap_metrics(
        self, sim_df: pd.DataFrame, real_df: pd.DataFrame
    ) -> dict:
        """
        Calculates full suite of Sim2Real gap metrics across joint space 
        and end-effector space.
        
        Assumes DataFrames are already time-aligned (e.g. via loader.align_trajectories).
        """
        metrics = {}

        # 1. Joint Position RMSE and Max Error
        joint_cols = [c for c in sim_df.columns if "joint_" in c and "_pos" in c]
        if joint_cols:
            joint_rmse_list = []
            for col in joint_cols:
                if col in real_df.columns:
                    err = sim_df[col].values - real_df[col].values
                    rmse = np.sqrt(np.mean(err**2))
                    max_err = np.max(np.abs(err))
                    metrics[f"rmse/{col}"] = float(rmse)
                    metrics[f"max_err/{col}"] = float(max_err)
                    joint_rmse_list.append(rmse)

            if joint_rmse_list:
                metrics["summary/mean_joint_pos_rmse"] = float(np.mean(joint_rmse_list))

        # 2. Joint Velocity RMSE
        vel_cols = [c for c in sim_df.columns if "joint_" in c and "_vel" in c]
        if vel_cols:
            vel_rmse_list = []
            for col in vel_cols:
                if col in real_df.columns:
                    err = sim_df[col].values - real_df[col].values
                    rmse = np.sqrt(np.mean(err**2))
                    metrics[f"rmse/{col}"] = float(rmse)
                    vel_rmse_list.append(rmse)

            if vel_rmse_list:
                metrics["summary/mean_joint_vel_rmse"] = float(np.mean(vel_rmse_list))

        # 3. End-Effector Cartesian Trajectory Gap (x, y, z)
        ee_cols = ["ee_x", "ee_y", "ee_z"]
        if all(col in sim_df.columns and col in real_df.columns for col in ee_cols):
            sim_pos = sim_df[ee_cols].values
            real_pos = real_df[ee_cols].values

            # Pointwise Euclidean Distance at each timestamp
            pointwise_dist = np.linalg.norm(sim_pos - real_pos, axis=1)

            metrics["ee_cartesian/pos_rmse"] = float(
                np.sqrt(np.mean(pointwise_dist**2))
            )
            metrics["ee_cartesian/max_divergence"] = float(np.max(pointwise_dist))
            metrics["ee_cartesian/mean_divergence"] = float(np.mean(pointwise_dist))

            # Dynamic Time Warping (DTW) on 3D Cartesian Path
            dtw_distance, _ = fastdtw(sim_pos, real_pos, dist=euclidean)
            metrics["ee_cartesian/dtw_distance"] = float(dtw_distance)

        # 4. Joint Space Dynamic Time Warping (DTW)
        if joint_cols:
            sim_joints = sim_df[joint_cols].values
            real_joints = real_df[joint_cols].values
            dtw_joint_dist, _ = fastdtw(sim_joints, real_joints, dist=euclidean)
            metrics["joint_space/dtw_distance"] = float(dtw_joint_dist)

        return metrics


# Standalone Verification Execution
if __name__ == "__main__":
    # Generate dummy test trajectories
    t = np.linspace(0, 2.0, 100)
    
    sim_data = pd.DataFrame({
        "timestamp": t,
        "joint_1_pos": np.sin(t),
        "joint_2_pos": np.cos(t),
        "ee_x": 0.5 * t,
        "ee_y": 0.2 * np.sin(t),
        "ee_z": 0.1 * np.cos(t)
    })

    # Real data with slight noise + 20ms delay shift
    real_data = pd.DataFrame({
        "timestamp": t,
        "joint_1_pos": np.sin(t - 0.02) + np.random.normal(0, 0.01, 100),
        "joint_2_pos": np.cos(t - 0.02) + np.random.normal(0, 0.01, 100),
        "ee_x": 0.5 * t + 0.005,
        "ee_y": 0.2 * np.sin(t - 0.02),
        "ee_z": 0.1 * np.cos(t - 0.02)
    })

    analyzer = TrajectoryAnalyzer()
    results = analyzer.compute_gap_metrics(sim_data, real_data)

    print("--- Trajectory Analyzer Test Output ---")
    for key, val in results.items():
        print(f"{key:35s}: {val:.5f}")