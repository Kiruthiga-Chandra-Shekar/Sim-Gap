import mujoco
import numpy as np
import pandas as pd


class MuJoCoSimRunner:
    def __init__(self, xml_path: str = "models/fanuc_mate200id.xml"):
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

    def run_replay(
        self, 
        real_actions: np.ndarray, 
        kp: float = None, 
        kd: float = None, 
        delay_ms: float = 0.0,
        render_height: int = 480,
        render_width: int = 640
    ):
        """
        Replays real action trajectories inside MuJoCo physics simulation.
        Dynamically overrides Kp, Kd, and delay buffer at runtime without modifying XML files.
        """
        # 1. Reset simulation state
        mujoco.mj_resetData(self.model, self.data)

        # 2. Dynamic Kp Override (Position Actuator Gains)
        if kp is not None:
            self.model.actuator_gainprm[:, 0] = kp
            self.model.actuator_biasprm[:, 1] = -kp  # Balance bias vector for position control

        # 3. Dynamic Kd Override (Joint Damping)
        if kd is not None:
            self.model.dof_damping[:self.model.nv] = kd

        # 4. Dynamic Action Delay Buffer Injection
        dt = self.model.opt.timestep if self.model.opt.timestep > 0 else 0.002
        delay_steps = int((delay_ms / 1000.0) / dt) if dt > 0 else 0

        if delay_steps > 0:
            initial_action = real_actions[0:1]
            delay_buffer = np.repeat(initial_action, delay_steps, axis=0)
            buffered_actions = np.vstack([delay_buffer, real_actions])
        else:
            buffered_actions = real_actions

        # 5. Initialize Offscreen Renderer
        renderer = None
        try:
            renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)
        except Exception as e:
            print(f"[WARNING] Offscreen renderer fallback: {e}")

        sim_records = []
        rendered_frames = []

        # 6. Step Physics Through Buffered Trajectory
        for t, action in enumerate(buffered_actions):
            n_act = min(len(action), self.model.nu)
            self.data.ctrl[:n_act] = action[:n_act]
            
            # Step forward physics engine
            mujoco.mj_step(self.model, self.data)

            # Record telemetry state with explicit timestamp
            joint_positions = self.data.qpos[:n_act].copy()
            joint_velocities = self.data.qvel[:n_act].copy()

            step_dict = {
                "timestep": t,
                "timestamp": t * dt,
            }
            for i in range(n_act):
                step_dict[f"joint_{i+1}_pos"] = joint_positions[i]
                step_dict[f"joint_{i+1}_vel"] = joint_velocities[i]
            
            sim_records.append(step_dict)

            # Render camera frame
            if renderer is not None:
                try:
                    renderer.update_scene(self.data)
                    rgb_frame = renderer.render()
                    rendered_frames.append(rgb_frame)
                except Exception:
                    pass

        sim_df = pd.DataFrame(sim_records)
        return sim_df, rendered_frames