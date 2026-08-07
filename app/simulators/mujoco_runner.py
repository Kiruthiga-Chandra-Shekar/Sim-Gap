import mujoco
import numpy as np
import pandas as pd
import h5py

class MuJoCoSimRunner:
    def __init__(self, xml_path: str):
        # Load FANUC Mate 200iD or generic arm MJCF model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

    def run_replay(self, real_actions: np.ndarray, dt: float = 0.02) -> pd.DataFrame:
        """
        Executes the real dataset actions inside MuJoCo physics and records state.
        """
        self.model.opt.timestep = dt
        sim_records = []

        for t_idx, action in enumerate(real_actions):
            # Apply control actions to actuators (e.g. joint position targets or torques)
            # Assuming action corresponds to 6 joint targets
            if len(action) >= self.model.nu:
                self.data.ctrl[:self.model.nu] = action[:self.model.nu]

            # Step the MuJoCo physics engine
            mujoco.mj_step(self.model, self.data)

            # Record simulated joint positions and velocities
            qpos = self.data.qpos.copy()
            qvel = self.data.qvel.copy()
            actuator_force = self.data.actuator_force.copy()

            row = {'timestamp': t_idx * dt}
            for j in range(min(6, len(qpos))):
                row[f'joint_{j+1}_pos'] = qpos[j]
                row[f'joint_{j+1}_vel'] = qvel[j]
                row[f'joint_{j+1}_torque'] = actuator_force[j]

            sim_records.append(row)

        return pd.DataFrame(sim_records)

# Example Usage:
# runner = MuJoCoSimRunner("models/fanuc_mate200id.xml")
# sim_df = runner.run_replay(real_action_array)
# sim_df.to_csv("sample_data/sim_mujoco_trace.csv", index=False)