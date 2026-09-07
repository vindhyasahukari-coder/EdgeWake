import sqlite3
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

DB_PATH = os.path.join(config.DATA_DIR, "metrics.db")

def init_db():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    # Enable WAL mode for better concurrency
    conn.execute('PRAGMA journal_mode=WAL;')
    
    with conn:
        # Table for system metrics (CPU, RAM)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS sys_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                cpu_percent REAL,
                ram_percent REAL,
                model_size_kb REAL
            )
        ''')
        
        # Table for inference stats (confidence, time)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS inference_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                confidence REAL,
                inference_time_ms REAL,
                is_wake_word BOOLEAN
            )
        ''')
        
        # Table for end-to-end events (transcripts, latency, bandwidth)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                transcript TEXT,
                network_latency_ms REAL,
                asr_latency_ms REAL,
                total_latency_ms REAL,
                audio_captured_bytes INTEGER,
                audio_sent_bytes INTEGER,
                bandwidth_saved_pct REAL
            )
        ''')
    conn.close()

def get_connection():
    # Return a connection that waits if database is locked
    return sqlite3.connect(DB_PATH, timeout=10.0)

if __name__ == "__main__":
    init_db()
    print(f"Initialized database at {DB_PATH}")
