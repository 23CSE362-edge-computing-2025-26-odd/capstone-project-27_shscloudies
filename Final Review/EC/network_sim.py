# network_sim.py
import time, random, math

def distance_to_delay_ms(distance_km, propagation_speed_m_per_s=300_000_000.0):
    # propagation delay = distance (m) / speed (m/s), convert to ms
    dist_m = distance_km * 1000.0
    prop_s = dist_m / propagation_speed_m_per_s
    return prop_s * 1000.0

def lora_toa_ms(payload_bytes, sf=9, bw=125000, cr=4/7):
    base_ms_per_byte = 0.5  # base cost
    sf_factor = 2 ** (sf - 7)  # SF7 -> 1, SF9 -> 4, SF12 -> 32
    return payload_bytes * base_ms_per_byte * sf_factor

def simulate_send(distance_km, payload_bytes, sf=9, bw=125000, base_loss=0.01):
    prop_ms = distance_to_delay_ms(distance_km)
    toa_ms = lora_toa_ms(payload_bytes, sf=sf, bw=bw)
    processing_ms = random.uniform(1.0, 5.0)
    jitter_ms = random.uniform(0, 20.0)
    delay_ms = prop_ms + toa_ms + processing_ms + jitter_ms

    # packet loss increases with distance and SF (longer ToA increases collision risk)
    loss_prob = base_loss + (distance_km / 10.0) + ( (sf - 7) * 0.01 )
    loss_prob = min(loss_prob, 0.5)
    success = random.random() > loss_prob

    # simulate the delay
    time.sleep(delay_ms / 1000.0)
    return delay_ms, success
