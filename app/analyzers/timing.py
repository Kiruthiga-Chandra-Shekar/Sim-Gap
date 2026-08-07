import numpy as np
import pandas as pd
from scipy.signal import correlate


class TimingAnalyzer:
    """
    Analyzes temporal discrepancies between simulation and real robot logs,
    including command-to-execution latency, phase lag, and sample jitter.
    """

    def compute_timing_metrics(
        self, sim_df: pd.DataFrame, real_df: pd.DataFrame, target_fps: float = 50.0
    ) -> dict:
        metrics = {}
        dt = 1.0 / target_fps

        # 1. Estimate Control Latency via Cross-Correlation (Joint 1 / Primary Actuator)
        joint_col = "joint_1_pos"
        if joint_col in sim_df.columns and joint_col in real_df.columns:
            sim_sig = sim_df[joint_col].values - np.mean(sim_df[joint_col].values)
            real_sig = real_df[joint_col].values - np.mean(real_df[joint_col].values)

            # Compute cross-correlation
            corr = correlate(real_sig, sim_sig, mode="full")
            lags = np.arange(-len(sim_sig) + 1, len(sim_sig))
            best_lag_idx = np.argmax(corr)
            lag_steps = lags[best_lag_idx]

            latency_ms = float(lag_steps * dt * 1000.0)
            metrics["timing/latency_ms"] = latency_ms
            metrics["timing/lag_steps"] = int(lag_steps)

        # 2. Sample Timestamp Jitter (Real Hardware Loop Stability)
        if "timestamp" in real_df.columns:
            timestamps = real_df["timestamp"].values
            dt_series = np.diff(timestamps)
            jitter_ms = float(np.std(dt_series) * 1000.0)
            metrics["timing/real_loop_jitter_ms"] = jitter_ms
            metrics["timing/max_loop_dt_ms"] = float(np.max(dt_series) * 1000.0)

        return metrics


if __name__ == "__main__":
    t = np.linspace(0, 2.0, 100)
    sim = pd.DataFrame({"timestamp": t, "joint_1_pos": np.sin(2 * np.pi * t)})
    # Shift real trajectory by 2 steps (40 ms at 50 Hz)
    real = pd.DataFrame({"timestamp": t, "joint_1_pos": np.sin(2 * np.pi * (t - 0.04))})

    analyzer = TimingAnalyzer()
    print("TimingAnalyzer Test Output:")
    print(analyzer.compute_timing_metrics(sim, real))