import mujoco
import numpy as np
import pandas as pd


class MuJoCoSimRunner:
    def __init__(self, xml_path: str = "models/fanuc_mate200id.xml"):
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

    def run_replay(self, real_actions: np.ndarray, render_width: int = 320, render_height: int = 240):
        """
        Executes forward simulation replay and renders offscreen RGB frames.
        """
        mujoco.mj_resetData(self.model, self.data)
        
        # Initialize renderer for visual camera output
        renderer = None
        try:
            renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)
        except Exception as e:
            print(f"[WARNING] Offscreen renderer initialization fallback: {e}")

        sim_records = []
        rendered_frames = []

        for t, action in enumerate(real_actions):
            # Apply control actions
            n_act = min(len(action), self.model.nu)
            self.data.ctrl[:n_act] = action[:n_act]
            
            # Step physics engine
            mujoco.mj_step(self.model, self.data)

            # Record telemetry state
            joint_positions = self.data.qpos[:n_act].copy()
            joint_velocities = self.data.qvel[:n_act].copy()

            step_dict = {"timestep": t}
            for i in range(n_act):
                step_dict[f"joint_{i+1}_pos"] = joint_positions[i]
                step_dict[f"joint_{i+1}_vel"] = joint_velocities[i]
            
            sim_records.append(step_dict)

            # Render offscreen camera frame
            if renderer is not None:
                try:
                    renderer.update_scene(self.data)
                    rgb_frame = renderer.render()
                    rendered_frames.append(rgb_frame)
                except Exception:
                    pass

        sim_df = pd.DataFrame(sim_records)
        return sim_df, rendered_frames