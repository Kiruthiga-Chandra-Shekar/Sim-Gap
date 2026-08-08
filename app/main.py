import os
import sys
import glob
from pathlib import Path
import pandas as pd

# Suppress TensorFlow C++ & oneDNN informational logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.append(str(Path(__file__).parent.parent))

from app.loaders.rlds_loader import RLDSFanucLoader
from app.simulators.mujoco_runner import MuJoCoSimRunner
from app.analyzers.trajectory import TrajectoryAnalyzer
from app.analyzers.timing import TimingAnalyzer
from app.analyzers.diagnosis import Sim2RealDiagnosisEngine


def get_available_tasks(dataset_path: str) -> list[str]:
    """Scans dataset directory and extracts available RLDS task split names."""
    tfrecord_files = glob.glob(os.path.join(dataset_path, "*.tfrecord*"))
    tasks = set()
    for filepath in tfrecord_files:
        filename = os.path.basename(filepath)
        # File pattern: fanuc_manipulation-<task_name>.tfrecord-00000-of-0000X
        if "fanuc_manipulation-" in filename:
            task_part = filename.split("fanuc_manipulation-")[1]
            task_name = task_part.split(".tfrecord")[0]
            # Sanitize any accidental whitespace
            task_name = task_name.replace(" ", "").strip()
            tasks.add(task_name)
    return sorted(list(tasks))


def run_pipeline(target_task: str = "close_drawer", episode_idx: int = 0):
    dataset_path = "sample_data/fanuc_manipulation/1.0.0"
    model_path = "models/fanuc_mate200id.xml"

    available_tasks = get_available_tasks(dataset_path)

    print("================================================================================")
    print("                        SIM2REAL GAP DIAGNOSTIC REPORT                          ")
    print("================================================================================")
    print(f"Available Tasks Found ({len(available_tasks)}) : {available_tasks}")
    print("--------------------------------------------------------------------------------")

    if not available_tasks:
        print(f"[ERROR] No tfrecord files found in {dataset_path}.")
        return

    if target_task not in available_tasks:
        print(f"[WARNING] Target task '{target_task}' not found in directory.")
        print(f"Defaulting to first available task: '{available_tasks[0]}'\n")
        target_task = available_tasks[0]

    print(f"Target Task Split    : {target_task} (Episode #{episode_idx})")
    print(f"Target Real Dataset  : {dataset_path}")
    print(f"Simulation Physics   : MuJoCo ({model_path})")
    print("--------------------------------------------------------------------------------")

    # Load RLDS Data
    loader = RLDSFanucLoader(dataset_dir=dataset_path)
    real_data_output = loader.load_real_data(split_name=target_task, episode_idx=episode_idx)
    real_df = real_data_output[0] if isinstance(real_data_output, tuple) else real_data_output
    print(f"Real trajectory loaded: {len(real_df)} timesteps.")

    # Select joint positions or action signals
    joint_cols = [c for c in real_df.columns if "joint_" in c and "_pos" in c]
    if not joint_cols:
        joint_cols = [c for c in real_df.columns if "action" in c]

    # Target 3 main arm joints for the physical simulation model
    real_actions = real_df[joint_cols].iloc[:, :3].values

    # Forward Replay in MuJoCo
    sim_runner = MuJoCoSimRunner(xml_path=model_path)
    sim_output = sim_runner.run_replay(real_actions=real_actions)
    
    # Safely unpack sim_df if run_replay returns a tuple (e.g., (df, metrics))
    sim_df = sim_output[0] if isinstance(sim_output, tuple) else sim_output

    # Time Alignment
    sim_aligned, real_aligned = loader.align_trajectories(sim_df, real_df)

    # Analyzers
    traj_analyzer = TrajectoryAnalyzer()
    timing_analyzer = TimingAnalyzer()
    diag_engine = Sim2RealDiagnosisEngine()

    traj_metrics = traj_analyzer.compute_gap_metrics(sim_aligned, real_aligned)
    timing_metrics = timing_analyzer.compute_timing_metrics(sim_aligned, real_aligned)

    combined_metrics = {**traj_metrics, **timing_metrics}

    print("--------------------------------------------------------------------------------")
    print("KINEMATIC & TEMPORAL METRICS:")
    for metric, value in combined_metrics.items():
        if isinstance(value, float):
            print(f"  {metric:35s}: {value:.5f}")
        else:
            print(f"  {metric:35s}: {value}")

    # Generate Diagnosis
    report = diag_engine.diagnose(combined_metrics)
    print("--------------------------------------------------------------------------------")
    print(f"DIAGNOSIS STATUS: [{report['status']}]")
    print("Automated Recommendations:")
    for rec in report["actionable_recommendations"]:
        print(f"  -> {rec}")
    print("================================================================================")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "close_drawer"
    run_pipeline(target_task=target, episode_idx=0)