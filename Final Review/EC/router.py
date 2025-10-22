# router.py
import time, json, argparse, yaml, socket, threading
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from collections import defaultdict
from utils_crypto import aes_decrypt
from network_sim import simulate_send
from gateway import receive_from_router
import pywt
from scipy.signal import iirnotch

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
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

router_cfg = next(r for r in cfg["routers"] if r["id"] == args.id)
gateway_loc = cfg["gateway"]["location_km"]
distance_km = abs(router_cfg["location_km"] - gateway_loc)
sf = cfg["loRa_defaults"]["sf"]

# -----------------------
# Global dedup tracker
# -----------------------
seen_seq = defaultdict(set)

# -----------------------
# ECG Processing Function - DO NOT CHANGE
# -----------------------
def preprocess_ecg(ecg_signal, fs=250):
    ecg = np.array(ecg_signal)
    if len(ecg) == 0:
        return [], 0.0, 0

    # Bandpass
    b,a = butter(2,[0.5/(fs/2),40/(fs/2)],btype='band')
    filtered = filtfilt(b,a,ecg)

    # Notch 50Hz
    b_n,a_n = iirnotch(50/(fs/2),30)
    filtered = filtfilt(b_n,a_n,filtered)

    # Wavelet denoising
    coeffs = pywt.wavedec(filtered,'db4',level=3)
    sigma = np.median(np.abs(coeffs[-1]))/0.6745
    uthresh = sigma*np.sqrt(2*np.log(len(filtered)))
    coeffs = [pywt.threshold(c,value=uthresh,mode='soft') for c in coeffs]
    denoised = pywt.waverec(coeffs,'db4')

    # HR and peaks
    thresh = np.mean(denoised)+0.3*np.std(denoised)
    peaks,_ = find_peaks(denoised,distance=fs*0.28,height=thresh)
    hr = len(peaks)*(60/len(denoised)*fs) if len(peaks)>0 else 0.0
    return denoised.tolist(), hr, len(peaks)

# -----------------------
# Missing Data Alert - DO NOT CHANGE
# -----------------------
last_received = defaultdict(lambda: time.time())
sampling_rate = 5  # normal

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
        delay_ms, success = simulate_send(distance_km,len(out_bytes),sf=sf)
        if success: receive_from_router(out_bytes)

# -----------------------
# ECG Packet Handler - DO NOT CHANGE
# -----------------------
def handle_ecg_packet(data):
    device = data.get("device_id")
    seq = data.get("seq")
    fs = data.get("fs", 250)
    ecg = data.get("ecg", [])

    if seq in seen_seq[device]:
        print(f"[{args.id}] ⚠️ Duplicate seq={seq} from {device} dropped")
        return
    seen_seq[device].add(seq)

    # ----------------------
    # Preprocessing
    # ----------------------
    filtered, hr, peaks_count = preprocess_ecg(ecg, fs)

    # ----------------------
    # Missing-data alert (2 seconds)
    # ----------------------
    now = time.time()
    if device not in last_received or now - last_received[device] > 2:
        print(f"[{args.id}] ⚠️ Missing data for {device}, possible heart attack")
        summary_alert = {
            "from_router": args.id,
            "device": device,
            "seq": None,
            "hr": 0.0,
            "status": "alert:missing_data",
            "ts": now
        }
        out_bytes = json.dumps(summary_alert).encode()
        delay_ms, success = simulate_send(distance_km, len(out_bytes), sf=sf)
        if success:
            receive_from_router(out_bytes)
    last_received[device] = now

    # ----------------------
    # Adaptive sampling
    # ----------------------
    if hr > 140 or hr < 40:
        current_sampling = 1  # faster sampling for 2 minutes
    else:
        current_sampling = sampling_rate  # normal

    # ----------------------
    # Status classification (simple thresholds)
    # ----------------------
    if peaks_count == 0 or hr == 0:
        status = "alert:flatline"
    elif hr > 150:
        status = "alert:tachycardia"
    elif hr < 40:
        status = "alert:bradycardia"
    else:
        status = "normal"

    print(f"[{args.id}] Device={device} Seq={seq} HR={hr:.1f} Peaks={peaks_count} → {status}")

    summary = {
        "from_router": args.id,
        "device": device,
        "seq": seq,
        "hr": round(hr, 2),
        "avg_hr": round(hr, 2),
        "num_peaks": peaks_count,
        "num_samples": len(ecg),
        "fs": fs,
        "ts": data.get("ts", now),
        "status": status,
        "preprocessed_ecg": filtered  # send filtered signal to gateway for ML
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
        # Ensure delay_ms is a number before formatting
        if isinstance(delay_ms, (int, float)):
            delay_str = f"{delay_ms:.1f}ms"
        else:
            delay_str = "N/A"

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

    if pkt_type == b"E":  # Encrypted ECG
        try:
            raw = aes_decrypt(KEY, data)
            data_json = json.loads(raw.decode())
            handle_ecg_packet(data_json)
        except Exception as e:
            print(f"[{args.id}] ❌ AES/JSON parse failed: {e}")

    elif pkt_type == b"A":  # Plaintext ALERT
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
