import numpy as np
import pandas as pd


class SensorAnalyzer:
    """
    Evaluates sensor quality, noise characteristics, and high-frequency drift
    between simulation and physical sensors.
    """

    def compute_sensor_metrics(
        self, sim_df: pd.DataFrame, real_df: pd.DataFrame
    ) -> dict:
        metrics = {}

        # Evaluate noise on joint velocities / torques
        target_cols = [c for c in real_df.columns if "_vel" in c or "force" in c or "torque" in c]

        if not target_cols:
            # Fallback to joint positions if velocities aren't available
            target_cols = [c for c in real_df.columns if "joint_" in c and "_pos" in c]

        for col in target_cols:
            if col in sim_df.columns:
                real_sig = real_df[col].values
                sim_sig = sim_df[col].values

                # Signal Noise Floor via High-Pass Difference
                real_noise = np.diff(real_sig)
                sim_noise = np.diff(sim_sig)

                real_std = float(np.std(real_noise))
                sim_std = float(np.std(sim_noise))

                metrics[f"sensor_noise_std/real/{col}"] = real_std
                metrics[f"sensor_noise_std/sim/{col}"] = sim_std
                metrics[f"sensor_noise_ratio/{col}"] = (
                    real_std / (sim_std + 1e-8)
                )

        return metrics


if __name__ == "__main__":
    t = np.linspace(0, 1.0, 50)
    sim = pd.DataFrame({"joint_1_vel": np.zeros(50)})
    real = pd.DataFrame({"joint_1_vel": np.random.normal(0, 0.05, 50)})

    analyzer = SensorAnalyzer()
    print("SensorAnalyzer Test Output:")
    print(analyzer.compute_sensor_metrics(sim, real))