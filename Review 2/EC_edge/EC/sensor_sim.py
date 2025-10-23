import time, json, socket
import numpy as np
from utils_crypto import aes_encrypt

# -----------------------
# Config: Two routers, 5 devices allocated efficiently
# -----------------------
DEVICE_ROUTER_MAP = {
    # Router 1: 2 devices
    "device_01": ("127.0.0.1", 5005, "normal"),
    "device_02": ("127.0.0.1", 5005, "bradycardia"),
    
    # Router 2: 3 devices
    "device_03": ("127.0.0.1", 5006, "tachycardia"),
    "device_04": ("127.0.0.1", 5006, "arrhythmia"),
    "device_05": ("127.0.0.1", 5006, "arrest")
}

KEY_HEX = "00112233445566778899aabbccddeeff"
KEY = bytes.fromhex(KEY_HEX)

seq_num = {device: 0 for device in DEVICE_ROUTER_MAP}

# -----------------------
# Improved ECG waveform generator with realistic QRS complexes
# -----------------------
def generate_qrs_complex(duration=0.1, fs=250):
    """Generate a realistic QRS complex waveform"""
    t = np.linspace(0, duration, int(fs * duration))
    # Simplified QRS: Q dip, R spike, S dip
    qrs = np.zeros(len(t))
    center = len(t) // 2
    
    # Q wave (small negative)
    q_start = max(0, center - 15)
    q_end = center - 5
    qrs[q_start:q_end] = -0.2
    
    # R wave (large positive spike)
    r_width = 10
    r_indices = np.arange(max(0, center - r_width//2), min(len(t), center + r_width//2))
    qrs[r_indices] = 1.5 * np.exp(-((r_indices - center)**2) / (r_width/3)**2)
    
    # S wave (negative dip)
    s_start = center + 5
    s_end = min(len(t), center + 15)
    qrs[s_start:s_end] = -0.3
    
    return qrs

def generate_ecg_wave(scenario="normal", duration_s=2, fs=250):
    """Generate realistic ECG waveform based on scenario"""
    t = np.linspace(0, duration_s, int(fs * duration_s), endpoint=False)
    ecg = np.zeros(len(t))
    
    qrs = generate_qrs_complex(duration=0.1, fs=fs)
    qrs_samples = len(qrs)
    
    if scenario == "normal":
        # 60-100 bpm, use 75 bpm (0.8s intervals)
        beat_interval = 0.8
        num_beats = int(duration_s / beat_interval)
        
        for i in range(num_beats):
            beat_idx = int(i * beat_interval * fs)
            if beat_idx + qrs_samples <= len(ecg):
                ecg[beat_idx:beat_idx + qrs_samples] = qrs
    
    elif scenario == "bradycardia":
        # <60 bpm, use 45 bpm (1.33s intervals)
        beat_interval = 1.33
        num_beats = int(duration_s / beat_interval) + 1
        
        for i in range(num_beats):
            beat_idx = int(i * beat_interval * fs)
            if beat_idx + qrs_samples <= len(ecg):
                ecg[beat_idx:beat_idx + qrs_samples] = qrs * 0.8  # Lower amplitude
    
    elif scenario == "tachycardia":
        # >100 bpm, use 150 bpm (0.4s intervals)
        beat_interval = 0.4
        num_beats = int(duration_s / beat_interval) + 1
        
        for i in range(num_beats):
            beat_idx = int(i * beat_interval * fs)
            if beat_idx + qrs_samples <= len(ecg):
                ecg[beat_idx:beat_idx + qrs_samples] = qrs * 1.2  # Higher amplitude
    
    elif scenario == "arrhythmia":
        # Irregular intervals: alternating between 0.4s and 1.0s
        beat_times = []
        current_time = 0
        irregular_pattern = [0.4, 1.0, 0.5, 0.9, 0.45, 1.1]  # Irregular pattern
        pattern_idx = 0
        
        while current_time < duration_s:
            beat_times.append(current_time)
            current_time += irregular_pattern[pattern_idx % len(irregular_pattern)]
            pattern_idx += 1
        
        for beat_time in beat_times:
            beat_idx = int(beat_time * fs)
            if beat_idx + qrs_samples <= len(ecg):
                # Variable amplitude for arrhythmia
                amplitude = np.random.uniform(0.7, 1.3)
                ecg[beat_idx:beat_idx + qrs_samples] = qrs * amplitude
    
    elif scenario == "arrest":
        # Flatline - no beats
        ecg = np.zeros(len(t))
        # Add very small noise to simulate monitoring
        ecg += 0.01 * np.random.randn(len(t))
        return ecg.tolist()
    
    # Add baseline wander and noise
    baseline = 0.05 * np.sin(2 * np.pi * 0.5 * t)  # Slow baseline drift
    noise = 0.05 * np.random.randn(len(t))
    ecg = ecg + baseline + noise
    
    return ecg.tolist()


# -----------------------
# Socket setup
# -----------------------
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# -----------------------
# Main simulation loop
# -----------------------
try:
    print("[SENSOR] Starting ECG simulation...")
    print("[SENSOR] Scenarios: normal(75bpm), bradycardia(45bpm), tachycardia(150bpm), arrhythmia(irregular), arrest(flatline)")
    
    while True:
        for device, (ip, port, scenario) in DEVICE_ROUTER_MAP.items():
            seq_num[device] += 1
            ecg_segment = generate_ecg_wave(scenario)

            data = {
                "device_id": device,
                "seq": seq_num[device],
                "fs": 250,
                "ecg": ecg_segment,
                "ts": int(time.time() * 1000)  # milliseconds
            }

            encrypted_payload = aes_encrypt(KEY, json.dumps(data).encode())
            sock.sendto(b"E" + encrypted_payload, (ip, port))
            print(f"[SENSOR] {device} ({scenario:12s}) seq={seq_num[device]:3d} -> {ip}:{port}")

        time.sleep(2)

except KeyboardInterrupt:
    print("\n[SENSOR] Simulation stopped.")