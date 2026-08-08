import os
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.append(str(Path(__file__).parent.parent))

from app.loaders.rlds_loader import RLDSFanucLoader
from app.simulators.mujoco_runner import MuJoCoSimRunner
from app.analyzers.trajectory import TrajectoryAnalyzer
from app.analyzers.timing import TimingAnalyzer
from app.analyzers.diagnosis import Sim2RealDiagnosisEngine


st.set_page_config(
    page_title="SimGap - Visual Sim2Real Benchmark",
    page_icon="🤖",
    layout="wide",
)


@st.cache_data
def get_available_tasks(dataset_path: str) -> list[str]:
    tfrecord_files = glob.glob(os.path.join(dataset_path, "*.tfrecord*"))
    tasks = set()
    for filepath in tfrecord_files:
        filename = os.path.basename(filepath)
        if "fanuc_manipulation-" in filename:
            task_part = filename.split("fanuc_manipulation-")[1]
            task_name = task_part.split(".tfrecord")[0]
            tasks.add(task_name)
    return sorted(list(tasks)) if tasks else ["close_drawer"]


def run_benchmark_pipeline(task_name: str, 
    episode_idx: int, 
    kp: float, 
    kd: float, 
    delay_ms: float):
    dataset_path = "sample_data/fanuc_manipulation/1.0.0"
    model_path = "models/fanuc_mate200id.xml"

    loader = RLDSFanucLoader(dataset_dir=dataset_path)
    real_df = loader.load_real_data(split_name=task_name, episode_idx=episode_idx)

    joint_cols = [c for c in real_df.columns if "joint_" in c and "_pos" in c]
    if not joint_cols:
        joint_cols = [c for c in real_df.columns if "action" in c]

    # ✅ Pass ALL 6 joint actions
    real_actions = real_df[joint_cols].values

    sim_runner = MuJoCoSimRunner(xml_path=model_path)
    sim_df, rendered_frames = sim_runner.run_replay(real_actions=real_actions, kp=kp, kd=kd, delay_ms=delay_ms)

    sim_aligned, real_aligned = loader.align_trajectories(sim_df, real_df)

    traj_analyzer = TrajectoryAnalyzer()
    timing_analyzer = TimingAnalyzer()
    diag_engine = Sim2RealDiagnosisEngine()

    traj_metrics = traj_analyzer.compute_gap_metrics(sim_aligned, real_aligned)
    timing_metrics = timing_analyzer.compute_timing_metrics(sim_aligned, real_aligned)
    metrics = {**traj_metrics, **timing_metrics}
    diagnosis = diag_engine.diagnose(metrics, current_kp=kp, current_kd=kd)

    return sim_aligned, real_aligned, metrics, diagnosis, rendered_frames


# --- HEADER ---
st.title("🤖 SimGap: Visual Sim2Real Diagnostics & Benchmarking Suite")
st.markdown(
    "Quantify and visualize real hardware execution versus MuJoCo forward simulation. "
    "Get exact numerical parameter recommendations to bridge physical domain gaps."
)

# --- SIDEBAR ---
st.sidebar.header("Benchmark Controls")
dataset_path = "sample_data/fanuc_manipulation/1.0.0"
available_tasks = get_available_tasks(dataset_path)

selected_task = st.sidebar.selectbox("Select Target Task Split", options=available_tasks, index=0)
selected_episode = st.sidebar.number_input("Episode Index", min_value=0, max_value=10, value=0, step=1)

st.sidebar.divider()
st.sidebar.subheader("🎛️ Dynamic Sim2Real Tuning Controls")
st.sidebar.caption("Adjust parameters to test recommended fixes in real-time without modifying source files.")

# Sliders initialized with default XML values (Kp=50.0, Kd=5.0, Latency=0ms)
selected_kp = st.sidebar.slider("Actuator Stiffness (Kp)", min_value=10.0, max_value=150.0, value=50.0, step=0.5)
selected_kd = st.sidebar.slider("Joint Damping (Kd)", min_value=0.1, max_value=20.0, value=5.0, step=0.25)
selected_delay = st.sidebar.slider("Action Delay Buffer (ms)", min_value=0, max_value=2000, value=0, step=20)

if st.sidebar.button("Execute Pipeline Benchmark", type="primary"):
    with st.spinner(f"Simulating '{selected_task}' with active dynamic overrides..."):
        sim_df, real_df, metrics, diagnosis, sim_frames = run_benchmark_pipeline(
            selected_task, 
            selected_episode,
            kp=selected_kp,
            kd=selected_kd,
            delay_ms=selected_delay
        )
        st.session_state["sim_df"] = sim_df
        st.session_state["real_df"] = real_df
        st.session_state["metrics"] = metrics
        st.session_state["diagnosis"] = diagnosis
        st.session_state["sim_frames"] = sim_frames

if "sim_df" in st.session_state:
    sim_df = st.session_state["sim_df"]
    real_df = st.session_state["real_df"]
    metrics = st.session_state["metrics"]
    diagnosis = st.session_state["diagnosis"]
    sim_frames = st.session_state["sim_frames"]

    # --- METRICS ROW ---
    col1, col2, col3, col4 = st.columns(4)
    status = diagnosis.get("status", "PASS")
    col1.metric("Status", status, delta="Match" if status == "PASS" else "Gap Detected", delta_color="inverse")
    col2.metric("Joint Pos RMSE", f"{metrics.get('summary/mean_joint_pos_rmse', 0.0):.4f} rad")
    col3.metric("Command Latency", f"{metrics.get('timing/latency_ms', 0.0):.1f} ms")
    col4.metric("DTW Distance", f"{metrics.get('joint_space/dtw_distance', 0.0):.2f}")

    st.divider()

    # --- TRAJECTORY ALIGNMENT VISUALIZATION ---
    st.subheader("📈 Trajectory Comparison: Raw vs. DTW Phase-Aligned")

    # Extract Joint Names
    joint_cols = [c for c in real_df.columns if "joint_" in c and "_pos" in c]
    if not joint_cols:
        joint_cols = [c for c in real_df.columns if "action" in c]

    selected_joint = st.selectbox(
        "Select Joint to Inspect", 
        options=joint_cols, 
        index=0
    )

    # Extract Trajectory Arrays
    sim_pos_all = sim_df[joint_cols].values
    real_pos_all = real_df[joint_cols].values
    joint_idx = joint_cols.index(selected_joint)

    # Compute DTW Alignment Path for Visualization
    traj_analyzer = TrajectoryAnalyzer()
    stride = max(1, len(sim_pos_all) // 400)
    sim_sub = sim_pos_all[::stride]
    real_sub = real_pos_all[::stride]

    sub_sim_idx, sub_real_idx = traj_analyzer.compute_dtw_alignment(sim_sub, real_sub)
    sim_dtw_idx = np.array(sub_sim_idx) * stride
    real_dtw_idx = np.array(sub_real_idx) * stride

    # Construct Plotly Figure
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=(
            f"Raw Unaligned Stream ({selected_joint})", 
            f"DTW Phase-Aligned Match ({selected_joint})"
        )
    )

    # Panel 1: Raw Index-to-Index
    fig.add_trace(
        go.Scatter(y=sim_pos_all[:, joint_idx], mode="lines", name="Simulated", line=dict(color="#1f77b4")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=real_pos_all[:, joint_idx], mode="lines", name="Real Hardware", line=dict(color="#ff7f0e", dash="dash")),
        row=1, col=1
    )

    # Panel 2: DTW Phase-Aligned Mapping
    fig.add_trace(
        go.Scatter(y=sim_pos_all[sim_dtw_idx, joint_idx], mode="lines", name="Sim (DTW)", line=dict(color="#1f77b4")),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(y=real_pos_all[real_dtw_idx, joint_idx], mode="lines", name="Real (DTW)", line=dict(color="#2ca02c", dash="solid")),
        row=1, col=2
    )

    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Frame / Step", row=1, col=1)
    fig.update_xaxes(title_text="Aligned Step Index", row=1, col=2)
    fig.update_yaxes(title_text="Position (rad)", row=1, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # --- ACTIONABLE RECOMMENDATIONS ---
    st.subheader("🎯 Quantitative Tuning & Recommendations")
    for rec in diagnosis.get("actionable_recommendations", []):
        st.warning(rec)

    st.divider()

    # --- VISUAL RENDERING SECTION ---
    st.subheader("🎥 Visual Execution & Frame Step View")
    
    max_aligned_len = min(len(sim_df), len(real_df), len(sim_frames))

    if max_aligned_len > 0:
        frame_idx = st.slider(
            "Step Through Execution Timeline (Frame / Timestep)",
            min_value=0,
            max_value=max_aligned_len - 1,
            value=0,
            step=1
        )

        joint_cols = [c for c in real_df.columns if "joint_" in c and "_pos" in c]
        if not joint_cols:
            joint_cols = [c for c in real_df.columns if "pos" in c or "joint" in c]

        vcol1, vcol2 = st.columns(2)
        
        with vcol1:
            st.markdown("#### Simulated Physics Execution (MuJoCo Render)")
            st.image(sim_frames[frame_idx], caption=f"MuJoCo Step {frame_idx}")
            
        with vcol2:
            st.markdown("#### Target Real Hardware State")
            st.info(f"Target Task: {selected_task} | Timestep: {frame_idx} / {max_aligned_len - 1}")
            
            # ✅ Extracts all 6 joint positions safely without [:3] restriction
            if joint_cols:
                current_real_pos = real_df[joint_cols].iloc[frame_idx].to_dict()
                st.write("**Real Joint Positions (rad):**", current_real_pos)

    st.divider()

    # --- TELEMETRY CHARTS ---
    st.subheader("📈 Synchronized Telemetry Plots")
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True, 
        subplot_titles=("Joint Angle Trajectories (rad)", "Absolute Error Delta (rad)")
    )

    # ✅ Collect all 6 joints for plotting
    joint_cols = [c for c in real_df.columns if "joint_" in c and "_pos" in c]
    if not joint_cols:
        joint_cols = [c for c in real_df.columns if "pos" in c or "joint" in c]

    time_axis = np.arange(len(real_df))

    for idx, col in enumerate(joint_cols):
        if col in sim_df.columns and col in real_df.columns:
            fig.add_trace(go.Scatter(x=time_axis, y=real_df[col], name=f"Real {col}"), row=1, col=1)
            fig.add_trace(go.Scatter(x=time_axis, y=sim_df[col], name=f"Sim {col}", line=dict(dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(x=time_axis, y=np.abs(real_df[col] - sim_df[col]), name=f"Error {col}"), row=2, col=1)

    fig.update_layout(height=500, template="plotly_white")
    st.plotly_chart(fig)