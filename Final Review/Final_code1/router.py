import time, json, argparse, yaml, socket, threading
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, iirnotch
from collections import defaultdict
from utils_crypto import aes_decrypt
from network_sim import simulate_send
from gateway import receive_from_router
import pywt

# -----------------------
# CLI args
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--id", default="router-01")
parser.add_argument("--key", default="00112233445566778899aabbccddeeff")
parser.add_argument("--port", type=int, default=5005)
args = parser.parse_args()
KEY = bytes.fromhex(args.key)

# -----------------------
# Load config
# -----------------------
with open("config.yaml", encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

router_cfg = next(r for r in cfg["routers"] if r["id"] == args.id)
devices_for_router = router_cfg.get("devices", [])
gateway_loc = cfg["gateway"]["location_km"]
distance_km = abs(router_cfg["location_km"] - gateway_loc)
sf = cfg["loRa_defaults"]["sf"]

# -----------------------
# Globals
# -----------------------
seen_seq = defaultdict(set)
last_received = defaultdict(lambda: time.time())
sampling_rate = 5  # base interval (sec)

# 🔹 adaptive state tracker per device
adaptive_sampling = defaultdict(lambda: sampling_rate)
adaptive_timer = defaultdict(lambda: 0)

# -----------------------
# ECG Processing
# -----------------------
def preprocess_ecg(ecg_signal, fs=250):
    ecg = np.array(ecg_signal)
    if len(ecg) == 0:
        return [], 0.0, 0

    b, a = butter(2, [0.5/(fs/2), 40/(fs/2)], btype='band')
    filtered = filtfilt(b, a, ecg)

    b_n, a_n = iirnotch(50/(fs/2), 30)
    filtered = filtfilt(b_n, a_n, filtered)

    coeffs = pywt.wavedec(filtered, 'db4', level=3)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2*np.log(len(filtered)))
    coeffs = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]
    denoised = pywt.waverec(coeffs, 'db4')
    if len(denoised) > len(ecg):
        denoised = denoised[:len(ecg)]

    signal_max = np.max(np.abs(denoised))
    if signal_max < 0.1:
        return denoised.tolist(), 0.0, 0

    thresh = 0.3 * signal_max
    min_distance = int(fs * 0.3)
    peaks, _ = find_peaks(denoised, distance=min_distance, height=thresh)

    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs
        mean_rr = np.mean(rr_intervals)
        hr = 60 / mean_rr if mean_rr > 0 else 0.0
    elif len(peaks) == 1:
        hr = 60 * len(peaks) / (len(denoised) / fs)
    else:
        hr = 0.0

    return denoised.tolist(), hr, len(peaks)


# -----------------------
# Classification
# -----------------------
def classify_status(hr, peaks_count, num_samples, fs=250):
    if peaks_count == 0 or hr == 0:
        return "alert:flatline"
    duration_s = num_samples / fs
    expected_min_peaks = duration_s * (30 / 60)
    if peaks_count < expected_min_peaks:
        return "alert:flatline"
    if hr < 40:
        return "alert:severe_bradycardia"
    elif hr < 60:
        return "alert:bradycardia"
    elif hr > 140:
        return "alert:severe_tachycardia"
    elif hr > 100:
        return "alert:tachycardia"
    else:
        return "normal"


# -----------------------
# Missing Data Alert
# -----------------------
def handle_missing_data(device):
    now = time.time()
    if now - last_received[device] > 2:
        print(f"[{args.id}] ⚠️ Missing data for {device}, possible heart attack")
        summary = {
            "from_router": args.id,
            "device": device,
            "seq": None,
            "hr": 0.0,
            "status": "alert:missing_data",
            "ts": now
        }
        out_bytes = json.dumps(summary).encode()
        delay_ms, success = simulate_send(distance_km, len(out_bytes), sf=sf)
        if success:
            receive_from_router(out_bytes)


# -----------------------
# ECG Packet Handler
# -----------------------
# -----------------------
# ECG Packet Handler
# -----------------------
def handle_ecg_packet(data):
    device = data.get("device_id")
    if device not in devices_for_router:
        print(f"[{args.id}] Ignored {device} — not assigned to this router")
        return

    seq = data.get("seq")
    fs = data.get("fs", 250)
    ecg = data.get("ecg", [])

    if seq in seen_seq[device]:
        print(f"[{args.id}] ⚠️ Duplicate seq={seq} from {device} dropped")
        return
    seen_seq[device].add(seq)

    # Preprocessing
    filtered, hr, peaks_count = preprocess_ecg(ecg, fs)

    # Missing-data alert
    now = time.time()
    if device not in last_received or now - last_received[device] > 5:
        handle_missing_data(device)
    last_received[device] = now

    # ----------------------
    # Adaptive sampling logic
    # ----------------------
    global sampling_rate
    if hr > 140 or hr < 40:
        sampling_rate = 1   # faster sampling during abnormal HR
    else:
        sampling_rate = 5   # normal condition

    # 🩺 print the current sampling time
    print(f"[{args.id}] ⏱️ Current sampling interval = {sampling_rate} sec (based on HR={hr:.1f})")

    # Status classification
    status = classify_status(hr, peaks_count, len(ecg), fs)

    # Log with more details
    print(f"[{args.id}] Device={device} Seq={seq} HR={hr:.1f} Peaks={peaks_count} Samples={len(ecg)} → {status}")

    summary = {
        "from_router": args.id,
        "device": device,
        "seq": seq,
        "hr": round(hr, 2),
        "avg_hr": round(hr, 2),
        "num_peaks": peaks_count,
        "num_samples": len(ecg),
        "fs": fs,
        "ts": data.get("ts", int(now * 1000)),
        "status": status,
        "preprocessed_ecg": filtered[:100]
    }

    out_bytes = json.dumps(summary).encode()
    delay_ms, success = simulate_send(distance_km, len(out_bytes), sf=sf)
    if success:
        receive_from_router(out_bytes)
        print(f"[{args.id}] ✅ Forwarded {status} seq={seq} delay={delay_ms:.1f}ms")
    else:
        print(f"[{args.id}] ⚠️ Packet loss simulated for seq={seq}")


# -----------------------
# ALERT Packet Handler
# -----------------------
def handle_alert_packet(device, alert_type, ts):
    if device not in devices_for_router:
        print(f"[{args.id}] Ignored ALERT from {device} — not assigned to this router")
        return

    print(f"[{args.id}] ⚡ ALERT received from {device}: {alert_type} @ {ts}")
    summary = {
        "from_router": args.id,
        "device": device,
        "seq": None,
        "hr": 0.0,
        "status": f"alert:{alert_type.lower()}",
        "ts": int(ts)
    }
    out_bytes = json.dumps(summary).encode()
    delay_ms, success = simulate_send(distance_km, len(out_bytes), sf=sf)
    if success:
        receive_from_router(out_bytes)
        delay_str = f"{delay_ms:.1f}ms" if isinstance(delay_ms, (int, float)) else "N/A"
        print(f"[{args.id}] ✅ Forwarded {alert_type} to gateway (delay={delay_str})")
    else:
        print(f"[{args.id}] ⚠️ Packet loss simulated for ALERT from {device}")


# -----------------------
# UDP Listener
# -----------------------
def on_receive(payload: bytes, addr=None):
    if not payload:
        return
    pkt_type = payload[0:1]
    data = payload[1:]

    if pkt_type == b"E":
        try:
            raw = aes_decrypt(KEY, data)
            data_json = json.loads(raw.decode())
            handle_ecg_packet(data_json)
        except Exception as e:
            print(f"[{args.id}] ❌ AES/JSON parse failed: {e}")

    elif pkt_type == b"A":
        try:
            msg = data.decode()
            parts = msg.split(",")
            if len(parts) >= 4:
                device, _, alert_type, ts = parts
                handle_alert_packet(device, alert_type, ts)
        except Exception as e:
            print(f"[{args.id}] ⚠️ ALERT parse failed: {e}")
    else:
        print(f"[{args.id}] ⚠️ Unknown packet type received")


def listen_udp():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", args.port))
    print(f"[{args.id}] Listening on UDP port {args.port}...")
    while True:
        data, addr = sock.recvfrom(65535)
        threading.Thread(target=on_receive, args=(data, addr), daemon=True).start()


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    print(f"[{args.id}] started — distance to gateway: {distance_km} km (sf={sf})")
    threading.Thread(target=listen_udp, daemon=True).start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"[{args.id}] shutting down.")
