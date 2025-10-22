# sensor_sim.py
import time, json, socket
import numpy as np
from utils_crypto import aes_encrypt

# -----------------------
# Config
# -----------------------
DEVICE_ROUTER_MAP = {
    "device_01": ("127.0.0.1", 5005, "normal"),
    "device_02": ("127.0.0.1", 5006, "arrest")
}
KEY_HEX = "00112233445566778899aabbccddeeff"
KEY = bytes.fromhex(KEY_HEX)

seq_num = {device: 0 for device in DEVICE_ROUTER_MAP}

# -----------------------
# ECG waveform generator
# -----------------------
def generate_ecg_wave_scenario(scenario="normal", duration_s=2, fs=250):
    t = np.linspace(0, duration_s, int(fs*duration_s), endpoint=False)
    if scenario == "normal":
        ecg = 0.1 * np.sin(2*np.pi*1.3*t)
        ecg += 0.1
        ecg += np.exp(-((t % 1)-0.5)**2 / 0.002) * 1.5
        ecg += 0.05*np.random.randn(len(t))
    elif scenario == "arrest":
        ecg = np.zeros(len(t))  # flatline
    else:
        ecg = np.zeros(len(t))
    return ecg.tolist()

# -----------------------
# Socket setup
# -----------------------
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# -----------------------
# Main loop
# -----------------------
try:
    while True:
        for device, (ip, port, scenario) in DEVICE_ROUTER_MAP.items():
            seq_num[device] += 1
            ecg_segment = generate_ecg_wave_scenario(scenario)
            data = {
                "device_id": device,
                "seq": seq_num[device],
                "fs": 250,
                "ecg": ecg_segment,
                "ts": int(time.time())
            }
            # Encrypted ECG packet with prefix 'E'
            encrypted_payload = aes_encrypt(KEY, json.dumps(data).encode())
            sock.sendto(b"E" + encrypted_payload, (ip, port))
            print(f"[SENSOR] {device} sent {scenario} ECG seq={seq_num[device]} -> {ip}:{port}")

            # If scenario is arrest, also send plaintext ALERT with prefix 'A'
            if scenario == "arrest":
                alert_pkt = f"{device},ALERT,FLATLINE,{int(time.time()*1000)}"
                sock.sendto(b"A" + alert_pkt.encode(), (ip, port))
                print(f"[SENSOR] {device} sent ALERT FLATLINE -> {ip}:{port}")

        time.sleep(2)

except KeyboardInterrupt:
    print("[SENSOR] stopping simulation")
