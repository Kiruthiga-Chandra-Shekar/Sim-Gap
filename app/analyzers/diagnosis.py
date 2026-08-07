import pandas as pd


class Sim2RealDiagnosisEngine:
    """
    Aggregates metrics from trajectory, timing, and sensor analyzers
    to output actionable engineering diagnoses for domain randomization
    and physics calibration.
    """

    def __init__(
        self,
        latency_threshold_ms: float = 30.0,
        rmse_threshold_rad: float = 0.05,
        noise_ratio_threshold: float = 3.0,
    ):
        self.latency_threshold_ms = latency_threshold_ms
        self.rmse_threshold_rad = rmse_threshold_rad
        self.noise_ratio_threshold = noise_ratio_threshold

    def diagnose(self, combined_metrics: dict) -> dict:
        diagnoses = []
        recommendations = []

        # 1. Latency & Delay Issues
        latency = combined_metrics.get("timing/latency_ms", 0.0)
        if abs(latency) > self.latency_threshold_ms:
            diagnoses.append(
                f"HIGH CONTROL LATENCY: Real robot lags simulation by {latency:.1f} ms."
            )
            recommendations.append(
                f"Add action buffer delay of {int(abs(latency))} ms in simulation policy environment."
            )

        # 2. Tracking Error / Gain Discrepancy
        mean_rmse = combined_metrics.get("summary/mean_joint_pos_rmse", 0.0)
        if mean_rmse > self.rmse_threshold_rad:
            diagnoses.append(
                f"KINEMATIC DIVERGENCE: Joint position tracking RMSE ({mean_rmse:.4f} rad) exceeds limit."
            )
            recommendations.append(
                "Tune simulator PD gains (kp/kd) or model joint stiffness/backlash."
            )

        # 3. Sensor / Actuator Noise Mis-modeling
        noise_ratios = [
            v for k, v in combined_metrics.items() if "sensor_noise_ratio" in k
        ]
        if noise_ratios and max(noise_ratios) > self.noise_ratio_threshold:
            diagnoses.append(
                f"UNDER-MODELED SENSOR NOISE: Real hardware noise is {max(noise_ratios):.1f}x higher than simulation."
            )
            recommendations.append(
                "Increase Gaussian observation noise during RL training domain randomization."
            )

        if not diagnoses:
            diagnoses.append("EXCELLENT ALIGNMENT: Sim2Real gap is within calibrated limits.")
            recommendations.append("Policy is ready for zero-shot real-world deployment.")

        return {
            "diagnoses": diagnoses,
            "actionable_recommendations": recommendations,
            "status": "PASS" if len(diagnoses) == 1 and "EXCELLENT" in diagnoses[0] else "FAIL",
        }


if __name__ == "__main__":
    sample_metrics = {
        "timing/latency_ms": 45.0,
        "summary/mean_joint_pos_rmse": 0.08,
        "sensor_noise_ratio/joint_1_vel": 4.2,
    }

    engine = Sim2RealDiagnosisEngine()
    report = engine.diagnose(sample_metrics)
    print("DiagnosisEngine Test Output:")
    print("Status:", report["status"])
    print("Diagnoses:", report["diagnoses"])
    print("Recommendations:", report["actionable_recommendations"])