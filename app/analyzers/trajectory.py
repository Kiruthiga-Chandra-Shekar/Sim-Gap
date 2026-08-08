import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


class TrajectoryAnalyzer:
    """
    Analyzes physical trajectory divergence between real robot telemetry 
    and simulated environment outputs using Dynamic Time Warping (DTW).
    """

    def compute_dtw_alignment(self, sim_data: np.ndarray, real_data: np.ndarray):
        """
        Computes dynamic time warping path to align sim and real time-series 
        independently of constant or variable execution latency.
        """
        N, M = len(sim_data), len(real_data)
        cost_matrix = cdist(sim_data, real_data, metric="euclidean")

        # Cumulative cost matrix initialization
        D = np.full((N + 1, M + 1), np.inf)
        D[0, 0] = 0.0

        for i in range(1, N + 1):
            for j in range(1, M + 1):
                D[i, j] = cost_matrix[i - 1, j - 1] + min(
                    D[i - 1, j],    # Insertion
                    D[i, j - 1],    # Deletion
                    D[i - 1, j - 1] # Match
                )

        # Backtrack optimal warping path
        i, j = N, M
        path_sim, path_real = [], []
        while i > 0 and j > 0:
            path_sim.append(i - 1)
            path_real.append(j - 1)
            idx = np.argmin([D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]])
            if idx == 0:
                i -= 1
            elif idx == 1:
                j -= 1
            else:
                i -= 1
                j -= 1

        return path_sim[::-1], path_real[::-1]

    def compute_gap_metrics(self, sim_df: pd.DataFrame, real_df: pd.DataFrame) -> dict:
        # Extract joint position columns
        pos_cols = [c for c in real_df.columns if "joint_" in c and "_pos" in c]
        if not pos_cols:
            pos_cols = [c for c in real_df.columns if "action" in c]

        # Filter available columns common to both DataFrames
        pos_cols = [c for c in pos_cols if c in sim_df.columns and c in real_df.columns]

        sim_pos = sim_df[pos_cols].values
        real_pos = real_df[pos_cols].values

        # -------------------------------------------------------------
        # 1. Compute Raw (Time-Shifted) Position RMSE
        # -------------------------------------------------------------
        min_len = min(len(sim_pos), len(real_pos))
        raw_pos_rmse = float(np.sqrt(np.mean((sim_pos[:min_len] - real_pos[:min_len]) ** 2)))

        # -------------------------------------------------------------
        # 2. Compute Phase-Aligned (DTW) Spatial Position RMSE
        # -------------------------------------------------------------
        # Subsample traces if length > 400 frames to prevent DTW latency spikes
        stride = max(1, len(sim_pos) // 400)
        sim_sub = sim_pos[::stride]
        real_sub = real_pos[::stride]

        sub_sim_idx, sub_real_idx = self.compute_dtw_alignment(sim_sub, real_sub)

        # Map subsampled DTW path back to full index array
        sim_idx = np.array(sub_sim_idx) * stride
        real_idx = np.array(sub_real_idx) * stride

        aligned_diffs = sim_pos[sim_idx] - real_pos[real_idx]
        dtw_aligned_pos_rmse = float(np.sqrt(np.mean(aligned_diffs ** 2)))

        # -------------------------------------------------------------
        # 3. Compute Velocity RMSE (Required for Kd Damping Diagnosis)
        # -------------------------------------------------------------
        vel_cols = [c for c in real_df.columns if "joint_" in c and "_vel" in c]
        vel_cols = [c for c in vel_cols if c in sim_df.columns and c in real_df.columns]

        if vel_cols:
            sim_vel = sim_df[vel_cols].values
            real_vel = real_df[vel_cols].values
            aligned_vel_diffs = sim_vel[sim_idx] - real_vel[real_idx]
            dtw_aligned_vel_rmse = float(np.sqrt(np.mean(aligned_vel_diffs ** 2)))
        else:
            # Fallback derivative estimate if velocity columns are absent
            dt = 0.02
            sim_vel_est = np.gradient(sim_pos, axis=0) / dt
            real_vel_est = np.gradient(real_pos, axis=0) / dt
            aligned_vel_diffs = sim_vel_est[sim_idx] - real_vel_est[real_idx]
            dtw_aligned_vel_rmse = float(np.sqrt(np.mean(aligned_vel_diffs ** 2)))

        # -------------------------------------------------------------
        # 4. Return Unified Metric Dictionary
        # -------------------------------------------------------------
        return {
            "summary/raw_joint_pos_rmse": raw_pos_rmse,
            "summary/mean_joint_pos_rmse": dtw_aligned_pos_rmse,  # DTW spatial error
            "summary/mean_joint_vel_rmse": dtw_aligned_vel_rmse,  # Velocity error for Kd
            "summary/latency_phase_penalty": raw_pos_rmse - dtw_aligned_pos_rmse,
            "joint_space/dtw_distance": float(np.sum(np.abs(aligned_diffs))),
        }