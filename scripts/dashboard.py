import streamlit as st
import pandas as pd
import time
import os
import glob
import subprocess

st.set_page_config(
    page_title="Tricked AI Telemetry",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for SOTA look
st.markdown(
    """
<style>
    .reportview-container {
        background: #0e1117;
    }
    .metric-container {
        background-color: #1e212b;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🚀 Tricked AI Telemetry Dashboard")
st.markdown("Real-time, zero-overhead metrics visualization.")


def get_available_runs():
    runs_dir = "runs"
    if not os.path.exists(runs_dir):
        return []

    files = glob.glob(f"{runs_dir}/**/*metrics.csv", recursive=True)
    files.sort(key=os.path.getmtime, reverse=True)
    return files


# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    available_runs = get_available_runs()
    env_path = os.environ.get("CSV_PATH", "")

    if available_runs:
        default_index = 0
        if env_path and env_path in available_runs:
            default_index = available_runs.index(env_path)

        selected_run = st.selectbox(
            "Select Run (Auto-Detected)", available_runs, index=default_index
        )
        csv_path = st.text_input("Manual Override Path:", value=selected_run)
    else:
        st.warning("No runs found in `runs/`. Ensure the engine is saving metrics.")
        csv_path = st.text_input(
            "Metrics CSV Path:", value=env_path if env_path else "metrics.csv"
        )

    refresh_rate = st.slider(
        "Refresh Rate (seconds)", min_value=1, max_value=60, value=2
    )
    auto_refresh = st.checkbox("Auto Refresh", value=True)


@st.cache_data(ttl=refresh_rate)
def load_data(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return pd.DataFrame()


df = load_data(csv_path)

if df.empty:
    st.warning(f"No data found at `{csv_path}`. Waiting for engine to start logging...")
else:
    # ---------------------------------------------------------
    # 1. Top Level Metrics
    # ---------------------------------------------------------
    latest = df.iloc[-1]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Current Step", f"{latest.get('step', 0):,}")
    col2.metric("Total Loss", f"{latest.get('total_loss', 0):.4f}")
    col3.metric("Max Game Score", f"{latest.get('game_score_max', 0):,.0f}")
    col4.metric("Avg MCTS Depth", f"{latest.get('mcts_depth_mean', 0):.1f}")
    col5.metric("GPU Utilization", f"{latest.get('gpu_usage_pct', 0):.1f}%")

    st.divider()

    # ---------------------------------------------------------
    # 2. Charts Section
    # ---------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🚀 Training", "🎮 Gameplay", "🧠 MCTS", "💻 Hardware"]
    )

    with tab1:
        st.subheader("Loss Metrics")
        if "total_loss" in df.columns:
            loss_cols = [
                c
                for c in ["total_loss", "policy_loss", "value_loss", "reward_loss"]
                if c in df.columns
            ]
            st.line_chart(df.set_index("step")[loss_cols])

        st.subheader("Learning Rate")
        if "lr" in df.columns:
            st.line_chart(df.set_index("step")["lr"], color="#f5a623")

    with tab2:
        st.subheader("Scores (Min, Med, Mean, Max)")
        score_cols = [
            c
            for c in [
                "game_score_min",
                "game_score_med",
                "game_score_mean",
                "game_score_max",
            ]
            if c in df.columns
        ]
        if score_cols:
            st.line_chart(df.set_index("step")[score_cols])

        st.subheader("Lines Cleared")
        if "game_lines_cleared" in df.columns:
            st.line_chart(df.set_index("step")["game_lines_cleared"], color="#00ffaa")

    with tab3:
        colA, colB = st.columns(2)
        with colA:
            st.subheader("Search Depth")
            if "mcts_depth_mean" in df.columns:
                st.line_chart(df.set_index("step")["mcts_depth_mean"], color="#aa00ff")
        with colB:
            st.subheader("Search Time (ms)")
            if "mcts_search_time_mean" in df.columns:
                st.line_chart(
                    df.set_index("step")["mcts_search_time_mean"], color="#ff00aa"
                )

    with tab4:
        st.subheader("Resource Usage")
        colC, colD = st.columns(2)
        with colC:
            st.markdown("**CPU, GPU, and Disk Usage (%)**")
            hw_pct = [
                c
                for c in ["cpu_usage_pct", "gpu_usage_pct", "disk_usage_pct"]
                if c in df.columns
            ]
            if hw_pct:
                st.line_chart(df.set_index("step")[hw_pct])
        with colD:
            st.markdown("**Memory Usage (MB)**")
            hw_mem = [c for c in ["ram_usage_mb", "vram_usage_mb"] if c in df.columns]
            if hw_mem:
                st.line_chart(df.set_index("step")[hw_mem])

if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
