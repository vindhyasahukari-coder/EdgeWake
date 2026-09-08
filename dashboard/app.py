import streamlit as st
import pandas as pd
import sqlite3
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from metrics.db import get_connection

st.set_page_config(page_title="EdgeWake Portal", layout="wide", initial_sidebar_state="collapsed")

# --- Session State ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

# --- DB Helpers ---
def clear_logs():
    try:
        conn = get_connection()
        conn.execute("DELETE FROM events")
        conn.commit()
        conn.close()
    except Exception as e:
        pass

def load_data():
    try:
        conn = get_connection()
        sys_df = pd.read_sql_query("SELECT * FROM sys_metrics ORDER BY id DESC LIMIT 50", conn)
        inf_df = pd.read_sql_query("SELECT * FROM inference_metrics ORDER BY id DESC LIMIT 100", conn)
        evt_df = pd.read_sql_query("SELECT * FROM events ORDER BY id DESC LIMIT 10", conn)
        conn.close()
        return sys_df, inf_df, evt_df
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# --- Dynamic CSS Theme ---
if st.session_state.theme == "Dark":
    bg_color = "#0E1117"
    card_bg = "#1A1C23"
    text_color = "#FFFFFF"
    sub_text = "#A0AEC0"
    border = "#2D3748"
else:
    bg_color = "#F7FAFC"
    card_bg = "#FFFFFF"
    text_color = "#1A202C"
    sub_text = "#718096"
    border = "#E2E8F0"

st.markdown(f"""
<style>
    .stApp {{
        background-color: {bg_color};
    }}
    
    .stMarkdown, .stMarkdown p {{
        color: {text_color} !important;
    }}
    
    h1, h2, h3 {{
        color: {text_color} !important;
        font-family: 'Inter', sans-serif;
    }}
    
    div[data-testid="metric-container"] {{
        background-color: {card_bg} !important;
        border: 1px solid {border} !important;
        padding: 1.5rem !important;
        border-radius: 0.75rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }}
    div[data-testid="metric-container"] label {{
        color: {sub_text} !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    div[data-testid="metric-container"] div {{
        color: {text_color} !important;
    }}

    .login-container {{
        background-color: {card_bg};
        border: 1px solid {border};
        padding: 3rem;
        border-radius: 12px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-top: 5vh;
        margin-bottom: 2rem;
    }}
</style>
""", unsafe_allow_html=True)


# --- Login Screen ---
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown(f"""
            <div class="login-container">
                <h1 style="font-size: 2.5rem; margin-bottom: 0;">🌐 EdgeWake Portal</h1>
                <p style="color: {sub_text} !important; font-size: 1.1rem; margin-top: 5px;">Enterprise Hardware Telemetry</p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            st.markdown(f"<p style='color: {text_color} !important; font-weight: 600;'>Authentication Required</p>", unsafe_allow_html=True)
            user = st.text_input("Username", placeholder="admin")
            pwd = st.text_input("Password", type="password", placeholder="admin")
            submitted = st.form_submit_button("Secure Login", use_container_width=True)
            
            if submitted:
                if user == "admin" and pwd == "admin":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid credentials. Use admin / admin")
    st.stop()


# --- Main Dashboard ---
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")
theme_toggle = st.sidebar.radio("UI Theme", ["Dark", "Light"], index=0 if st.session_state.theme == "Dark" else 1)
if theme_toggle != st.session_state.theme:
    st.session_state.theme = theme_toggle
    st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Transcript Logs", use_container_width=True):
    clear_logs()
    st.sidebar.success("Logs Cleared!")

st.title("Live Hardware Telemetry")
st.markdown(f"<p style='color: {sub_text} !important; font-size: 1.1rem; margin-bottom: 2rem;'>Monitoring EdgeWake ESP32-S3 IoT nodes in real-time.</p>", unsafe_allow_html=True)

placeholder = st.empty()

while True:
    sys_df, inf_df, evt_df = load_data()
    
    with placeholder.container():
        # KPI Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            val = f"{sys_df['cpu_percent'].iloc[0]:.1f}%" if not sys_df.empty else "--%"
            st.metric(label="CPU Usage", value=val)
                
        with col2:
            val = f"{sys_df['ram_percent'].iloc[0]:.1f}%" if not sys_df.empty else "--%"
            st.metric(label="RAM Usage", value=val)
                
        with col3:
            val = f"{sys_df['model_size_kb'].iloc[0]:.1f} KB" if not sys_df.empty else "-- KB"
            st.metric(label="Model Size (INT8)", value=val)
                
        with col4:
            val = f"{inf_df['inference_time_ms'].mean():.1f} ms" if not inf_df.empty else "-- ms"
            st.metric(label="Avg Inference Time", value=val)
                
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts Row
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown(f"<h3 style='color: {text_color} !important;'>Live Confidence Score</h3>", unsafe_allow_html=True)
            if not inf_df.empty:
                chart_data = inf_df.sort_values(by="id").copy()
                st.line_chart(chart_data.set_index("timestamp")["confidence"])
            else:
                st.write("Awaiting telemetry...")
                
        with col_chart2:
            st.markdown(f"<h3 style='color: {text_color} !important;'>Latency Breakdown</h3>", unsafe_allow_html=True)
            if not evt_df.empty:
                lat_data = evt_df.sort_values(by="id")
                st.bar_chart(lat_data.set_index("id")[["network_latency_ms", "asr_latency_ms"]])
            else:
                st.write("Awaiting detection events...")
                
        st.markdown("<hr style='opacity: 0.2;'>", unsafe_allow_html=True)
        
        # Data Log Row
        st.markdown(f"<h3 style='color: {text_color} !important;'>Event and Transcript Log</h3>", unsafe_allow_html=True)
        if not evt_df.empty:
            display_df = evt_df[["timestamp", "transcript", "network_latency_ms", "total_latency_ms", "audio_sent_bytes"]].copy()
            display_df['timestamp'] = pd.to_datetime(display_df['timestamp'], unit='s').dt.strftime('%H:%M:%S.%f').str[:-3]
            display_df.rename(columns={
                'timestamp': 'Timestamp',
                'transcript': 'Transcript',
                'network_latency_ms': 'Network Latency [ms]',
                'total_latency_ms': 'Total Latency [ms]',
                'audio_sent_bytes': 'Payload Size [Bytes]'
            }, inplace=True)
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("System is listening. Waiting for wake word detection.")
            
    time.sleep(1.0)
