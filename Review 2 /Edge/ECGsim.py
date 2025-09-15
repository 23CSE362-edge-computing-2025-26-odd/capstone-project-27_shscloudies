import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import time
import sys

fs = 250
dt = 1.0 / fs
mean_hr = 75
hrv_std = 0.02
display_seconds = 10
normal_duration = 10
flatline_duration = 60

def ecg_template(t):
    return (
        0.1 * np.exp(-((t-0.2)/0.05)**2) +
        -0.15 * np.exp(-((t-0.3)/0.015)**2) +
        1.2 * np.exp(-((t-0.32)/0.01)**2) +
        -0.25 * np.exp(-((t-0.34)/0.015)**2) +
        0.3 * np.exp(-((t-0.55)/0.1)**2)
    )

plt.ion()
fig, ax = plt.subplots(figsize=(12,4))
line, = ax.plot([], [], lw=1.5, color="green")
ax.set_ylim(-1.5, 2)
ax.set_xlim(0, display_seconds)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude (mV)")
ax.set_title("ECG")
ax.grid(which='both', color='lightgray', linestyle='-', linewidth=0.5)
ax.grid(which='major', color='gray', linestyle='-', linewidth=1)

buffer = []
time_buffer = []
t_total = 0.0
start_time = time.time()
update_interval = 0.04
last_update = 0.0
last_written_sec = -1

with open("ecg_amplitudes.txt", "w") as f:   # overwrite each run
    # Normal ECG phase
    while t_total < normal_duration:
        rr_interval = 60.0 / mean_hr * (1 + np.random.normal(0, hrv_std))
        t_beat = np.linspace(0, rr_interval, int(rr_interval*fs))
        beat = ecg_template(t_beat)
        beat += np.random.normal(0, 0.02, len(beat))
        beat += 0.05 * np.sin(2 * np.pi * 0.5 * t_beat)

        for sample in beat:
            buffer.append(sample)
            time_buffer.append(t_total)
            t_total += dt

            # write once per second
            current_sec = int(t_total)
            if current_sec != last_written_sec:
                f.write(f"{t_total:.3f}, {sample:.3f}\n")
                f.flush()
                last_written_sec = current_sec

            if len(buffer) > fs*display_seconds:
                buffer = buffer[-fs*display_seconds:]
                time_buffer = time_buffer[-fs*display_seconds:]

            if t_total - last_update >= update_interval:
                line.set_xdata(time_buffer)
                line.set_ydata(buffer)
                ax.set_xlim(time_buffer[0], time_buffer[0] + display_seconds)
                start_tick = int(np.floor(time_buffer[0]))
                ax.set_xticks(np.arange(start_tick, start_tick + display_seconds + 1, 1))
                ax.set_xticks(np.arange(time_buffer[0], time_buffer[0] + display_seconds + 0.2, 0.2), minor=True)
                fig.canvas.draw()
                fig.canvas.flush_events()
                last_update = t_total

            elapsed = time.time() - start_time
            sleep_time = t_total - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # Flatline phase
    flatline_samples = int(flatline_duration * fs)
    for _ in range(flatline_samples):
        sample = 0.0
        buffer.append(sample)
        time_buffer.append(t_total)
        t_total += dt

        current_sec = int(t_total)
        if current_sec != last_written_sec:
            f.write(f"{t_total:.3f}, {sample:.3f}\n")
            f.flush()
            last_written_sec = current_sec

        if len(buffer) > fs*display_seconds:
            buffer = buffer[-fs*display_seconds:]
            time_buffer = time_buffer[-fs*display_seconds:]

        if t_total - last_update >= update_interval:
            line.set_xdata(time_buffer)
            line.set_ydata(buffer)
            ax.set_xlim(time_buffer[0], time_buffer[0] + display_seconds)
            start_tick = int(np.floor(time_buffer[0]))
            ax.set_xticks(np.arange(start_tick, start_tick + display_seconds + 1, 1))
            ax.set_xticks(np.arange(time_buffer[0], time_buffer[0] + display_seconds + 0.2, 0.2), minor=True)
            fig.canvas.draw()
            fig.canvas.flush_events()
            last_update = t_total

        elapsed = time.time() - start_time
        sleep_time = t_total - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

plt.close(fig)
sys.exit()
