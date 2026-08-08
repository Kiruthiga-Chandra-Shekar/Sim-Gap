import pandas as pd


class Sim2RealDiagnosisEngine:
    """
    Aggregates metrics from trajectory, timing, and sensor analyzers
    to output actionable engineering diagnoses for domain randomization
    and physics calibration.
    """
    
    def diagnose(
        self, 
        metrics: dict, 
        current_kp: float = 50.0, 
        current_kd: float = 5.0,
        max_kp: float = 250.0,
        max_kd: float = 20.0
    ) -> dict:
        recommendations = []
        status = "PASS"

        # 1. Latency & Delay Tuning
        latency_ms = metrics.get("timing/latency_ms", 0.0)
        if latency_ms > 20.0:
            status = "FAIL"
            recommendations.append(
                f"[LATENCY GAP] Real robot lags simulation by {latency_ms:.1f} ms. "
                f"Actionable Fix: Insert an action buffer delay of {int(round(latency_ms))} ms into simulation environment."
            )

        # 2. Quantitative Proportional Gain (Kp) Tuning with Physical Cap Patch
        mean_pos_rmse = metrics.get("summary/mean_joint_pos_rmse", 0.0)
        if mean_pos_rmse > 0.05:
            status = "FAIL"
            if current_kp >= max_kp:
                recommendations.append(
                    f"[SYSTEMIC/DRIVE GAP] Joint position RMSE is high ({mean_pos_rmse:.4f} rad), "
                    f"but Kp is already at physical ceiling ({current_kp:.1f} N·m/rad). "
                    f"Actionable Fix: Do not increase Kp further. Add joint 'armature' (rotor inertia) "
                    f"or 'frictionloss' to default joints in XML model to match real drive resistance."
                )
            else:
                kp_adjustment_percent = min(max((mean_pos_rmse / 0.05) * 15.0, 10.0), 100.0)
                target_kp = min(current_kp * (1.0 + kp_adjustment_percent / 100.0), max_kp)
                delta_kp = target_kp - current_kp
                recommendations.append(
                    f"[STIFFNESS GAP] Joint position tracking RMSE is high ({mean_pos_rmse:.4f} rad). "
                    f"Actionable Fix: Increase simulator joint actuator Kp by +{delta_kp:.1f} N·m/rad "
                    f"(from {current_kp:.1f} to {target_kp:.1f} N·m/rad) to reduce compliance."
                )

        # 3. Quantitative Damping Gain (Kd) / Oscillation Tuning with Guard
        mean_vel_rmse = metrics.get("summary/mean_joint_vel_rmse", 0.0)
        if mean_vel_rmse > 0.5:
            status = "FAIL"
            if current_kd >= max_kd:
                recommendations.append(
                    f"[SYSTEMIC DAMPING] High velocity variance ({mean_vel_rmse:.4f} rad/s), "
                    f"but Kd is at ceiling ({current_kd:.2f} N·m·s/rad). "
                    f"Actionable Fix: Check trajectory synchronization or motor current ripple."
                )
            else:
                target_kd = min(current_kd * 1.25, max_kd)
                delta_kd = target_kd - current_kd
                recommendations.append(
                    f"[DAMPING GAP] High velocity variance/oscillation observed ({mean_vel_rmse:.4f} rad/s). "
                    f"Actionable Fix: Increase joint damping Kd by +{delta_kd:.2f} N·m·s/rad "
                    f"(from {current_kd:.2f} to {target_kd:.2f} N·m·s/rad) to stabilize transient response."
                )

        # 4. Success State Output
        if status == "PASS":
            recommendations.append(
                f"[SYSTEM CALIBRATED] Simulation metrics are within target tolerances! "
                f"(Position RMSE: {mean_pos_rmse:.4f} rad <= 0.05, Latency: {latency_ms:.1f} ms <= 20.0 ms)."
            )

        return {
            "status": status,
            "actionable_recommendations": recommendations,
        }


if __name__ == "__main__":
    sample_metrics = {
        "timing/latency_ms": 12.0,
        "summary/mean_joint_pos_rmse": 0.03,
        "summary/mean_joint_vel_rmse": 0.2,
    }

    engine = Sim2RealDiagnosisEngine()
    report = engine.diagnose(sample_metrics, current_kp=150.0, current_kd=8.0)
    print("DiagnosisEngine Test Output:")
    print("Status:", report["status"])
    print("Recommendations:", report["actionable_recommendations"])