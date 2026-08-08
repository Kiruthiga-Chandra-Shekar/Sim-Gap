# SimGap - A Sim2Real Diagnostic and Calibration Framework for Robotic Manipulation.

SimGap is an open-source robotics infrastructure tool for analyzing and diagnosing the gap between real robot behavior and simulated robot behavior.

Sim2Real remains one of the major challenges in robot learning and physical AI. A manipulation policy can perform well in simulation while behaving differently on a physical robot because of differences in:
actuator dynamics, friction, damping, control gains, latency, sensor noise, timing, model simplifications, contact dynamics, robot hardware.

SimGap is designed as a diagnostic layer between simulation and real-world robot data. Instead of treating Sim2Real as a single performance number, SimGap breaks the discrepancy down into measurable components such as:

- Joint-space trajectory error
- Velocity tracking error
- Temporal/command latency
- Dynamic Time Warping (DTW) alignment
- Simulation parameter sensitivity
- Actuation and controller mismatch
- Sensor/noise discrepancies
- Simulation stability

The framework replays real robot trajectories through a parameterized MuJoCo FANUC Mate 200iD model, compares the simulated response against the original robot trajectory, and produces engineering recommendations for reducing the observed gap.

### Quickstart

``` bash
# Clone repository
git clone https://github.com/Kiruthiga-Chandra-Shekar/Sim-Gap.git
cd Sim-Gap

#Create Virtual Environment
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows
# .venv\Scripts\activate

#Install requirements
pip install -r requirements.txt

Download the FANUC dataset from:
https://sites.google.com/berkeley.edu/fanuc-manipulation

Place it at this location in the Sim-Gap directory:
sample_data/fanuc_manipulation/1.0.0/

Then launch:
streamlit run app/dashboard.py

Open: http://localhost:8501 in your browser

Select a task → select an episode → configure simulation parameters → Execute Pipeline Benchmark.

```

#### Note

The dataset is intentionally not included in this GitHub repository because the dataset files are large and are not suitable for storing directly in the repository.

The Berkeley FANUC dataset is distributed using the TensorFlow/RLDS ecosystem. TFRecord is a binary data format commonly used for storing large machine-learning datasets.

Instead of storing each observation as a separate CSV row, TFRecord packages structured examples containing information such as:

Observation
- RGB images
- Joint positions
- Joint velocities
- Gripper state
- Language instruction
- Other metadata

Action
- Cartesian-space action

SimGap includes an RLDS-based loader that converts the dataset into a normalized internal trajectory representation before running the analysis. This keeps the rest of the framework independent of the original dataset format.

### Current Scope

This project currently focuses on trajectory replay and simulator calibration rather than learned-policy transfer. The real robot's observed trajectory is replayed through the MuJoCo FANUC model. 

### Trajectory Analysis

SimGap compares simulated and real trajectories in joint space.The trajectory analyzer calculates metrics including:

#### Raw Joint RMSE

Measures the direct difference between simulated and real joint trajectories (RMSE_raw).

#### Dynamic Time Warping

Robot executions may not progress through the same trajectory at exactly the same rate. SimGap therefore uses Dynamic Time Warping (DTW) to identify a better temporal correspondence between trajectories.

### Timing Analysis

SimGap estimates temporal mismatch between simulation and real execution. The timing analyzer uses signal correlation to estimate the relative delay between trajectories. The framework can report:estimated command latency, real-loop jitter, maximum timestep, temporal alignment characteristics

For example:

Estimated latency: 42 ms

A significant latency discrepancy can indicate that the simulator should incorporate an equivalent control/action delay.

### Understanding the Results

The current version of SimGap provides a first-pass Sim2Real diagnostic based on predefined thresholds for trajectory error, velocity error, and latency. A **FAIL** status means that one or more metrics are still outside the current acceptance range; it does not necessarily mean that the recommended parameter change was unsuccessful. The diagnosis may continue recommending higher **Kp or Kd** when position or velocity errors remain above their thresholds, even if previous adjustments have already improved the results. Since Sim2Real discrepancies can also come from friction, damping, inertia, actuator dynamics, model fidelity, and other factors, these recommendations should be treated as **candidate calibration actions rather than definitive causes**. Iterative baseline comparison, parameter sensitivity analysis, and residual-gap diagnosis are planned for the next phase.

### Limitations

The current implementation has several important limitations.

#### 1. Simplified robot model

The included FANUC MuJoCo model is a simplified representation and should not be treated as a manufacturer-accurate digital twin.

Differences may remain in:
link geometry,
mass,
inertia,
actuator dynamics,
friction,
gearbox behavior,
controller implementation,
hardware limits.

#### 2. Trajectory replay

The current benchmark evaluates trajectory replay rather than learned-policy transfer.

#### 3. Dataset dependency

The FANUC dataset must be downloaded separately because of its size.

#### 4. Sensor analysis

The sensor analysis module is currently an extensible foundation and does not yet provide a complete camera/depth/tactile Sim2Real benchmark.

#### 5. Simulation stability

Some combinations of aggressive controller parameters may result in unstable MuJoCo dynamics. Such failures are useful signals for future simulator-health diagnostics.

## Citation

If you use the FANUC Manipulation Dataset in your research or project, please cite the original paper:

```bibtex
@misc{zhu2023fanuc,
  title={Fanuc Manipulation: A Dataset for Learning-based Manipulation with FANUC Mate 200iD Robot},
  author={Zhu, Xinghao and Tian, Ran and Xu, Chenfeng and Huo, Mingxiao and Zhan, Wei and Tomizuka, Masayoshi and Ding, Mingyu},
  howpublished={\url{[https://sites.google.com/berkeley.edu/fanuc-manipulation](https://sites.google.com/berkeley.edu/fanuc-manipulation)}},
  year={2023}
}
```

The dataset is released under the license terms outlined on the official Berkeley MSC Lab Dataset Page. Please review the licensing terms prior to redistribution or commercial use.
