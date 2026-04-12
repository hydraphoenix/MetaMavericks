import streamlit as st
import numpy as np
import os
import sys
import time
import imageio
from PIL import Image
import io

# Ensure components are discoverable
sys.path.append(os.path.join(os.path.dirname(__file__), 'metamavericks_env'))
from metamavericks_env.gym_wrapper import MetamavericksGymWrapper
from stable_baselines3 import PPO

st.set_page_config(
    page_title="AquaSAR-Env | MetaMavericks",
    page_icon="🌊",
    layout="wide"
)

# Custom CSS for a polished look
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        background-color: white;
        border: 1px solid #ddd;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title("🌊 AquaSAR-Env: Autonomous Maritime Search & Rescue")
    st.markdown("### 🚀 Engineered by Team MetaMavericks")

    # Initialize environment in session state
    if 'env' not in st.session_state:
        st.session_state.env = MetamavericksGymWrapper()
        st.session_state.obs, _ = st.session_state.env.reset()
        st.session_state.total_reward = 0.0
        st.session_state.steps = 0
        st.session_state.done = False
        st.session_state.history = []

    # Sidebar for Controls and Info
    with st.sidebar:
        st.header("🎮 Control Center")
        mode = st.radio("Execution Mode", ["Manual", "RL Agent (PPO)", "LLM Commander"])
        
        st.divider()
        
        if st.button("🔄 Reset Environment"):
            st.session_state.obs, _ = st.session_state.env.reset()
            st.session_state.total_reward = 0.0
            st.session_state.steps = 0
            st.session_state.done = False
            st.session_state.history = []
            st.rerun()

        st.divider()
        st.info("**AquaSAR-Env** coordinates a UAV and two USVs in a 500m grid to rescue a survivor drifting in stochastic ocean currents.")

    # Main Dashboard Layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📡 Live Environment Feed")
        
        # Action Logic
        if not st.session_state.done:
            if mode == "Manual":
                st.write("Enter velocities [-15 to 15]:")
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    uav_x = st.number_input("UAV X", -15.0, 15.0, 0.0)
                    uav_y = st.number_input("UAV Y", -15.0, 15.0, 0.0)
                with m_col2:
                    usv1_x = st.number_input("USV1 X", -8.0, 8.0, 0.0)
                    usv1_y = st.number_input("USV1 Y", -8.0, 8.0, 0.0)
                with m_col3:
                    usv2_x = st.number_input("USV2 X", -8.0, 8.0, 0.0)
                    usv2_y = st.number_input("USV2 Y", -8.0, 8.0, 0.0)
                
                if st.button("Step ➡️"):
                    action = np.array([uav_x, uav_y, usv1_x, usv1_y, usv2_x, usv2_y])
                    st.session_state.obs, reward, term, trunc, info = st.session_state.env.step(action)
                    st.session_state.total_reward += reward
                    st.session_state.steps += 1
                    st.session_state.done = term or trunc
                    st.rerun()

            elif mode == "RL Agent (PPO)":
                if st.button("▶️ Run RL Step"):
                    model_path = "ppo_aquasar_model"
                    if os.path.exists(f"{model_path}.zip"):
                        model = PPO.load(model_path)
                        action, _ = model.predict(st.session_state.obs, deterministic=True)
                        st.session_state.obs, reward, term, trunc, info = st.session_state.env.step(action)
                        st.session_state.total_reward += reward
                        st.session_state.steps += 1
                        st.session_state.done = term or trunc
                        st.rerun()
                    else:
                        st.error("Model not found. Please train the agent first.")

        # Render Output
        frame = st.session_state.env.render()
        if frame is not None:
            st.image(frame, use_container_width=True)
        
        if st.session_state.done:
            st.success(f"Episode Finished! Final Reward: {st.session_state.total_reward:.2f}")

    with col2:
        st.subheader("📊 Telemetry")
        st.metric("Total Reward", f"{st.session_state.total_reward:.2f}")
        st.metric("Step Count", st.session_state.steps)
        
        st.divider()
        st.subheader("🕵️ Agent Status")
        # Extract status from metadata if available
        # features idx 15, 31, 47 are status codes (0.0 active, 1.0 penalty)
        status_drone = "🔴 PENALTY" if st.session_state.obs[15] > 0.5 else "🟢 ACTIVE"
        status_usv1 = "🔴 PENALTY" if st.session_state.obs[31] > 0.5 else "🟢 ACTIVE"
        status_usv2 = "🔴 PENALTY" if st.session_state.obs[47] > 0.5 else "🟢 ACTIVE"
        
        st.write(f"**UAV (Drone):** {status_drone}")
        st.write(f"**USV1 (Boat):** {status_usv1}")
        st.write(f"**USV2 (Boat):** {status_usv2}")

    st.divider()
    st.markdown("---")
    st.markdown("#### 📖 Project Documentation")
    with st.expander("Show Executive Summary"):
        st.write("""
        AquaSAR-Env is a high-impact, state-of-the-art multi-agent reinforcement learning environment. 
        It simulates a critical maritime search and rescue (SAR) operation coordinating one aerial drone (UAV) 
        and two surface boats (USVs) to locate and rescue a drifting survivor.
        """)

if __name__ == "__main__":
    main()
