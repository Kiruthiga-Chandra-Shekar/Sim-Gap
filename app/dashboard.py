import os
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Suppress verbose TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from app.loaders.rlds_loader import RLDSFanucLoader
from app.simulators.mujoco_runner import MuJoCoSimRunner
from app.analyzers.trajectory import TrajectoryAnalyzer
from app.analyzers.timing import TimingAnalyzer
from app.analyzers.diagnosis import Sim2RealDiagnosisEngine


st.set_page_config(
    page_title="SimGap - Sim2Real Telemetry Dashboard",
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


@st.cache_data
def run_benchmark_pipeline(task_name: str, episode_idx: int):
    dataset_path = "sample_data/fanuc_manipulation/1.0.0"
    model_path = "models/fanuc_mate200id.xml"

    loader = RLDSFanucLoader(dataset_dir=dataset_path)
    real_df = loader.load_real_data(split_name=task_name, episode_idx=episode_idx)

    joint_cols = [c for c in real_df.columns if "joint_" in c and "_pos" in c]
    if not joint_cols:
        joint_cols = [c for c in real_df.columns if "action" in c]

    real_actions = real_df[joint_cols].iloc[:, :3].values

    sim_runner = MuJoCoSimRunner(xml_path=model_path)
    sim_df = sim_runner.run_replay(real_actions=real_actions)

    sim_aligned, real_aligned = loader.align_trajectories(sim_df, real_df)

    traj_analyzer = TrajectoryAnalyzer()
    timing_analyzer = TimingAnalyzer()
    diag_engine = Sim2RealDiagnosisEngine()

    traj_metrics = traj_analyzer.compute_gap_metrics(sim_aligned, real_aligned)
    timing_metrics = timing_analyzer.compute_timing_metrics(sim_aligned, real_aligned)
    metrics = {**traj_metrics, **timing_metrics}
    diagnosis = diag_engine.diagnose(metrics)

    return sim_aligned, real_aligned, metrics, diagnosis


# --- UI HEADER ---
st.title("🤖 SimGap: Interactive Sim2Real Telemetry Dashboard")
st.markdown(
    "Quantify kinematic divergence, control loop latency, and physical discrepancies "
    "between **MuJoCo Forward Physics** and **Real Hardware Logs**."
)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Pipeline Configuration")
dataset_path = "sample_data/fanuc_manipulation/1.0.0"
available_tasks = get_available_tasks(dataset_path)

selected_task = st.sidebar.selectbox("Select Task Split", options=available_tasks, index=0)
selected_episode = st.sidebar.number_input("Episode Index", min_value=0, max_value=10, value=0, step=1)

run_button = st.sidebar.button("Run Diagnostic Benchmark", type="primary")

# --- MAIN DASHBOARD BODY ---
if run_button or "benchmark_run" not in st.session_state:
    st.session_state["benchmark_run"] = True
    with st.spinner(f"Running physics replay and analysis for '{selected_task}'..."):
        sim_df, real_df, metrics, diagnosis = run_benchmark_pipeline(selected_task, selected_episode)
        st.session_state["sim_df"] = sim_df
        st.session_state["real_df"] = real_df
        st.session_state["metrics"] = metrics
        st.session_state["diagnosis"] = diagnosis

sim_df = st.session_state["sim_df"]
real_df = st.session_state["real_df"]
metrics = st.session_state["metrics"]
diagnosis = st.session_state["diagnosis"]

# --- METRIC SUMMARY CARDS ---
col1, col2, col3, col4 = st.columns(4)

status = diagnosis.get("status", "PASS")
col1.metric("Diagnosis Status", status, delta="Optimal" if status == "PASS" else "-Critical Gap", delta_color="inverse")

mean_rmse = metrics.get("summary/mean_joint_pos_rmse", 0.0)
col2.metric("Mean Joint Position RMSE", f"{mean_rmse:.4f} rad")

latency = metrics.get("timing/latency_ms", 0.0)
col3.metric("Command Latency", f"{latency:.1f} ms")

dtw_dist = metrics.get("joint_space/dtw_distance", 0.0)
col4.metric("DTW Distance", f"{dtw_dist:.2f}")

st.divider()

# --- DIAGNOSTIC RECOMMENDATIONS ---
st.subheader("💡 Automated Root-Cause Diagnosis & Tuning Recommendations")
if diagnosis.get("actionable_recommendations"):
    for rec in diagnosis["actionable_recommendations"]:
        st.info(f"👉 {rec}")
else:
    st.success("Simulation parameters match real hardware within tolerance bounds.")

st.divider()

# --- INTERACTIVE TELEMETRY PLOTS ---
st.subheader("📈 Trajectory Comparison: Simulation vs. Real Hardware")

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=("Joint Positions over Time (rad)", "Joint Error / Divergence over Time"),
)

joint_cols = [c for c in real_df.columns if "joint_" in c and "_pos" in c][:3]
if not joint_cols:
    joint_cols = [c for c in real_df.columns if "action" in c][:3]

time_axis = real_df.index.values if "timestamp" not in real_df.columns else real_df["timestamp"].values

colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for idx, col in enumerate(joint_cols):
    if col in sim_df.columns and col in real_df.columns:
        fig.add_trace(
            go.Scatter(x=time_axis, y=real_df[col], mode="lines", name=f"Real {col}", line=dict(color=colors[idx % 3])),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=time_axis, y=sim_df[col], mode="lines", name=f"Sim {col}", line=dict(color=colors[idx % 3], dash="dash")),
            row=1, col=1,
        )
        # Error delta
        error = np.abs(real_df[col] - sim_df[col])
        fig.add_trace(
            go.Scatter(x=time_axis, y=error, mode="lines", name=f"Error {col}", line=dict(width=1)),
            row=2, col=1,
        )

fig.update_layout(height=600, template="plotly_white", hovermode="x unified")
fig.update_xaxes(title_text="Timestep / Time (s)", row=2, col=1)
fig.update_yaxes(title_text="Position (rad)", row=1, col=1)
fig.update_yaxes(title_text="Absolute Error (rad)", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# --- RAW DATA TABLE ---
with st.expander("🔍 View Raw Telemetry Data Table"):
    st.dataframe(real_df.head(50))