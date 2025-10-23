# gateway.py
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading, time, json, sqlite3, yaml, requests, statistics
from collections import defaultdict, deque
from network_sim import simulate_send
from tensorflow.keras.models import load_model
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# -----------------------
# Load ML model
# -----------------------
MODEL = load_model("ecg_classifier_model.h5")
CLASS_MAP = {0: "normal", 1: "tachycardia", 2: "arrhythmia", 3: "bradycardia"}

# -----------------------
# Load configuration
# -----------------------
with open("config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

GATEWAY_PORT = cfg["gateway"].get("http_port", 8000)
CLOUD_URL = "http://localhost:9000/cloud_ingest"

# -----------------------
# Database setup
# -----------------------
DB_FILE = "gateway.db"
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device TEXT,
    seq INTEGER,
    hr REAL,
    avg_hr REAL,
    temp REAL,
    ts REAL,
    priority TEXT,
    num_peaks INTEGER,
    num_samples INTEGER
)
""")
conn.commit()

# Add 'event' column if missing
c.execute("PRAGMA table_info(readings)")
columns = [col[1] for col in c.fetchall()]
if "event" not in columns:
    c.execute("ALTER TABLE readings ADD COLUMN event TEXT")
    conn.commit()

# -----------------------
# Deduplication & buffers
# -----------------------
seen = set()
recent_hr = defaultdict(lambda: deque(maxlen=20))
last_alert_time = defaultdict(float)

# -----------------------
# Fuzzy logic setup
# -----------------------
confidence = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'confidence')
arrhythmia_prob = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'arrhythmia_prob')
alert = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'alert')

confidence['low'] = fuzz.trimf(confidence.universe, [0, 0, 0.5])
confidence['medium'] = fuzz.trimf(confidence.universe, [0.3, 0.5, 0.7])
confidence['high'] = fuzz.trimf(confidence.universe, [0.6, 1, 1])

arrhythmia_prob['low'] = fuzz.trimf(arrhythmia_prob.universe, [0, 0, 0.4])
arrhythmia_prob['medium'] = fuzz.trimf(arrhythmia_prob.universe, [0.3, 0.5, 0.7])
arrhythmia_prob['high'] = fuzz.trimf(arrhythmia_prob.universe, [0.6, 1, 1])

alert['normal'] = fuzz.trimf(alert.universe, [0, 0, 0.4])
alert['moderate'] = fuzz.trimf(alert.universe, [0.3, 0.6, 0.8])
alert['critical'] = fuzz.trimf(alert.universe, [0.7, 1, 1])

rule1 = ctrl.Rule(confidence['high'] & arrhythmia_prob['low'], alert['normal'])
rule2 = ctrl.Rule(confidence['medium'] & arrhythmia_prob['medium'], alert['moderate'])
rule3 = ctrl.Rule(confidence['low'] | arrhythmia_prob['high'], alert['critical'])

alert_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
alert_sim = ctrl.ControlSystemSimulation(alert_ctrl)

# -----------------------
# Helper functions
# -----------------------
def compute_statistics(device):
    hr_values = list(recent_hr[device])
    if len(hr_values) < 3:
        return None
    avg_hr = statistics.mean(hr_values)
    std_hr = statistics.pstdev(hr_values) if len(hr_values) > 1 else 0
    trend = hr_values[-1] - avg_hr
    return dict(avg_hr=avg_hr, std_hr=std_hr, trend=trend)

def detect_cardiac_event(device, hr, avg_hr, std_hr, signal=None):
    """
    Returns:
        event_type: 'normal', 'tachycardia', 'bradycardia', 'arrhythmia'
        priority: 'normal', 'medium', 'high'
    """
    event_type = "normal"
    priority = "normal"

    # --- ML prediction if enough signal ---
    if signal is not None and len(signal) >= 250:
        x = np.array(signal[-250:]).reshape(1, 250, 1)
        pred = MODEL.predict(x, verbose=0)[0]
        cls_idx = np.argmax(pred)
        ml_event = CLASS_MAP[cls_idx]
        confidence_score = float(np.max(pred))
        arrhythmia_score = float(pred[2])
    else:
        ml_event = "invalid"
        confidence_score = 0
        arrhythmia_score = 0

    # --- Fuzzy alert ---
    alert_sim.input['confidence'] = confidence_score
    alert_sim.input['arrhythmia_prob'] = arrhythmia_score
    alert_sim.compute()
    fuzzy_result = alert_sim.output['alert']

    if fuzzy_result > 0.7:
        priority = "high"
    elif fuzzy_result > 0.4:
        priority = "medium"

    # --- Heuristic rules for arrhythmia detection ---
    # sudden HR spikes/drops or high variability
    if hr is not None and std_hr is not None:
        if hr < 50:
            event_type = "bradycardia"
        elif hr > 120:
            event_type = "tachycardia"
        elif std_hr > 15:  # high variability triggers arrhythmia
            event_type = "arrhythmia"
        else:
            event_type = ml_event  # fallback to ML prediction

    return event_type, priority


def forward_to_cloud(payload):
    delay_ms, success = simulate_send(distance_km=5, payload_bytes=len(json.dumps(payload).encode()), sf=9)
    if not success:
        print("[GATEWAY] ⚠ Cloud link unstable")
        return
    try:
        r = requests.post(CLOUD_URL, json=payload, timeout=10)
        print(f"[GATEWAY → CLOUD] 🌐 Sent seq={payload.get('seq')} status={r.status_code}")
    except Exception as e:
        print("[GATEWAY] ❌ Cloud send error:", e)

# -----------------------
# Data ingestion
# -----------------------
def receive_from_router(payload_bytes):
    try:
        data = json.loads(payload_bytes.decode())
    except Exception as e:
        print("[GATEWAY] ❌ Failed to parse JSON:", e)
        return

    key = (data["device"], data.get("seq"))
    if key in seen:
        return
    seen.add(key)

    device = data["device"]
    hr = data.get("hr")
    ts = data.get("ts")
    preprocessed_ecg = data.get("preprocessed_ecg")
    if hr:
        recent_hr[device].append(hr)

    stats = compute_statistics(device)
    if stats:
        avg_hr = stats["avg_hr"]
        std_hr = stats["std_hr"]
        trend = stats["trend"]
    else:
        avg_hr = hr
        std_hr = 0
        trend = 0

    event_type, fuzzy_priority = detect_cardiac_event(device, hr, avg_hr, std_hr, signal=preprocessed_ecg)

    now = time.time()
    priority = "normal"
    alert_flag = event_type in ["arrhythmia", "bradycardia", "tachycardia"]
    if alert_flag or fuzzy_priority == "high":
        if now - last_alert_time[device] > 10:
            print(f"[GATEWAY ALERT] ⚠ {device.upper()} → {event_type.upper()} | HR={hr:.1f}, AVG={avg_hr:.1f}")
            last_alert_time[device] = now
            priority = "high"
        elif fuzzy_priority == "medium":
            priority = "medium"

    # Save to DB
    conn_local = sqlite3.connect(DB_FILE, check_same_thread=False)
    c_local = conn_local.cursor()
    c_local.execute("""
        INSERT INTO readings (device, seq, hr, avg_hr, temp, ts, priority, num_peaks, num_samples, event)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        device, data.get("seq"), hr, avg_hr, data.get("temp"),
        ts, priority, data.get("num_peaks"), data.get("num_samples"), event_type
    ))
    conn_local.commit()
    conn_local.close()

    hr_str = f"{hr:.1f}" if hr else "0.0"
    avg_hr_str = f"{avg_hr:.1f}" if avg_hr else "0.0"
    print(f"[GATEWAY] ✅ Stored {device} seq={data.get('seq')} HR={hr_str} AVG={avg_hr_str} [{priority}]")

    # Forward to cloud
    cloud_packet = {
        "device": device,
        "seq": data.get("seq"),
        "hr": hr,
        "avg_hr": avg_hr,
        "std_hr": std_hr,
        "trend": trend,
        "event": event_type,
        "timestamp": ts,
        "priority": priority
    }
    forward_to_cloud(cloud_packet)

# -----------------------
# REST API
# -----------------------
class SimpleHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/ingest":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            receive_from_router(body)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path.startswith("/last"):
            c.execute("""
                SELECT device, seq, hr, avg_hr, temp, ts, priority, num_peaks, num_samples, event
                FROM readings ORDER BY id DESC LIMIT 50
            """)
            rows = c.fetchall()
            out = [
                dict(
                    device=r[0],
                    seq=r[1],
                    hr=r[2],
                    avg_hr=r[3],
                    temp=r[4],
                    ts=r[5],
                    priority=r[6],
                    num_peaks=r[7],
                    num_samples=r[8],
                    event=r[9]
                )
                for r in rows
            ]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(out, indent=2).encode())
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Gateway running")

# -----------------------
# HTTP Server
# -----------------------
def run_server():
    server = HTTPServer(("0.0.0.0", GATEWAY_PORT), SimpleHandler)
    print(f"[GATEWAY] HTTP API running on :{GATEWAY_PORT}")
    server.serve_forever()

# -----------------------
# Main loop
# -----------------------
if __name__ == "__main__":
    threading.Thread(target=run_server, daemon=True).start()
    print("[GATEWAY] Listening for router data...")
    while True:
        time.sleep(5)
