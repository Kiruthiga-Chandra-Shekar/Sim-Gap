# app/main.py
import sys
from pathlib import Path

# Add project root to path so 'app' imports work smoothly
sys.path.append(str(Path(__file__).parent.parent))

from app.loaders.rlds_loader import RLDSFanucLoader
from app.simulators.mujoco_runner import MuJoCoSimRunner
from app.analyzers.trajectory import TrajectoryAnalyzer

def run_pipeline():
    # Point directly to your dataset directory
    dataset_path = "sample_data/fanuc_manipulation/1.0.0"
    
    print(f"Loading real dataset from: {dataset_path}")
    loader = RLDSFanucLoader(dataset_dir=dataset_path)
    
    # Extract episode 0 from split 'pick_and_place'
    real_df = loader.load_real_data(split_name="pick_and_place", episode_idx=0)
    print(f"Real trajectory loaded: {len(real_df)} timesteps.")

    # Extract action array
    action_cols = ['action_dx', 'action_dy', 'action_dz', 'action_droll', 'action_dpitch', 'action_dyaw']
    real_actions = real_df[action_cols].values

    # Run MuJoCo forward replay (make sure your model path exists or use a generic robot model)
    sim_runner = MuJoCoSimRunner(xml_path="models/fanuc_mate200id.xml")
    sim_df = sim_runner.run_replay(real_actions=real_actions)

    # Time align
    sim_aligned, real_aligned = loader.align_trajectories(sim_df, real_df)

    # Analyze gap
    analyzer = TrajectoryAnalyzer()
    metrics = analyzer.compute_gap_metrics(sim_aligned, real_aligned)

    print("\n================ Sim2Real Metrics ================")
    for k, v in metrics.items():
        print(f"{k:35s}: {v:.5f}")

if __name__ == "__main__":
    run_pipeline()