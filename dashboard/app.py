import streamlit as st
import pandas as pd
import sqlite3
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from metrics.db import get_connection

st.set_page_config(page_title="Voice Edge Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Inject Custom CSS for Premium Look
st.markdown("""
<style>
    /* Global Font and Colors */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    /* Hide Streamlit Default Elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Clean Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #1E1E1E;
        border: 1px solid #333333;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    div[data-testid="metric-container"] > label {
        font-weight: 500;
        color: #A0AEC0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    div[data-testid="metric-container"] > div {
        color: #FFFFFF;
        font-weight: 700;
        font-size: 1.8rem;
    }

    /* Headers */
    h1, h2, h3 {
        font-weight: 600;
        color: #F7FAFC;
    }
    
    h1 {
        margin-bottom: 0.5rem;
        font-size: 2.2rem;
    }
    
    .subtitle {
        color: #A0AEC0;
        font-size: 1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* DataFrames */
    .stDataFrame {
        border-radius: 0.5rem;
        overflow: hidden;
        border: 1px solid #333333;
    }
    
    /* Divider */
    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border-color: #333333;
    }
</style>
""", unsafe_allow_html=True)

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

st.title("Edge-to-Cloud Voice Dashboard")
st.markdown("<div class='subtitle'>Real-time system telemetrics for the Ultra-Lightweight Custom Keyword Spotting architecture.</div>", unsafe_allow_html=True)

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
                
        st.markdown("---")
        
        # Charts Row
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("Live Confidence Score")
            if not inf_df.empty:
                chart_data = inf_df.sort_values(by="id").copy()
                st.line_chart(chart_data.set_index("timestamp")["confidence"])
            else:
                st.write("Awaiting inference telemetry...")
                
        with col_chart2:
            st.subheader("Latency Breakdown")
            if not evt_df.empty:
                lat_data = evt_df.sort_values(by="id")
                st.bar_chart(lat_data.set_index("id")[["network_latency_ms", "asr_latency_ms"]])
            else:
                st.write("Awaiting detection events...")
                
        st.markdown("---")
        
        # Data Log Row
        st.subheader("Event and Transcript Log")
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
