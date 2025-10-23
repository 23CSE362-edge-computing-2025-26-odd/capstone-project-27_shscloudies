import streamlit as st
import requests
import pandas as pd
import time

# -----------------------
# Page setup
# -----------------------
st.set_page_config(page_title="Wearable Gateway Dashboard", layout="wide")
st.title("🏥 Real-Time Wearable Health Monitor")

placeholder = st.empty()

# -----------------------
# Fetch data
# -----------------------
def get_data():
    try:
        r = requests.get("http://localhost:8000/last", timeout=2)
        if r.status_code == 200:
            df = pd.DataFrame(r.json())
            if not df.empty:
                df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            return df
    except Exception as e:
        st.warning(f"Failed to fetch data: {e}")
    return pd.DataFrame()

# -----------------------
# Status Badge
# -----------------------
def status_badge(hr, event):
    if event == "arrhythmia":
        return "⚠ Arrhythmia"
    elif hr == 0:
        return "🚨 CARDIAC ARREST"
    elif hr < 50:
        return "⚠ Bradycardia"
    elif hr > 120:
        return "⚠ Tachycardia"
    else:
        return "✅ Normal"

# -----------------------
# Main Dashboard
# -----------------------
refresh_interval = 5  # seconds
data_placeholder = st.empty()

try:
    while True:
        start_time = time.time()  # record start time for sampling
        df = get_data()
        with data_placeholder.container():
            if df.empty:
                st.info("No data available yet.")
            else:
                df = df.sort_values("ts", ascending=True).reset_index(drop=True)

                # ======================
                # ALERTS SECTION
                # ======================
                cardiac_devices = df[df["hr"] == 0]["device"].unique().tolist()
                warning_df = df[
                    (df["hr"] < 40) | (df["hr"] > 140) | (df["event"] == "arrhythmia")
                ]
                warning_df = warning_df[~warning_df["device"].isin(cardiac_devices)]

                # 💥 LIGHT RED CARDIAC ALERT
                if cardiac_devices:
                    devices_str = ", ".join(cardiac_devices)
                    st.markdown(
                        f"""
                        <div style="background-color:#dc143c; padding:40px; border-radius:15px; 
                                    color:white; text-align:center; font-size:28px; font-weight:bold;
                                    box-shadow:0 0 15px rgba(255,0,0,0.5);">
                            🚨 CARDIAC ARREST DETECTED 🚨<br>
                            <span style="font-size:22px;">Device(s): <b>{devices_str}</b></span><br>
                            <span style="font-size:18px;">⚠ Immediate Medical Attention Required ⚠</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        """
                        <div style="background-color:#22c55e; padding:20px; border-radius:10px; 
                                    color:white; text-align:center; font-size:20px;">
                            ✅ No cardiac arrest detected.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # 💛 Three-line Yellow Warnings for Other Issues
                arrhythmia_devices = df[df["event"] == "arrhythmia"]["device"].unique().tolist()
                brady_devices = df[df["hr"] < 50]["device"].unique().tolist()
                tachy_devices = df[df["hr"] > 120]["device"].unique().tolist()

                yellow_alerts = [
                    ("⚠ Arrhythmia", arrhythmia_devices),
                    ("⚠ Bradycardia (HR < 50 bpm)", brady_devices),
                    ("⚠ Tachycardia (HR > 120 bpm)", tachy_devices)
                ]

                st.markdown("<br>", unsafe_allow_html=True)

                # ------------- ALL ALERTS IN ONE ROW -------------
                warning_cols = st.columns(3)
                for i, (alert, devices) in enumerate(yellow_alerts):
                    if devices:
                        devices_str = ", ".join(devices)
                        with warning_cols[i]:
                            st.markdown(
                                f"""
                                <div style="background-color:#fff3b0; padding:15px; border-radius:10px; 
                                            color:#111; font-size:18px; text-align:center; font-weight:600; 
                                            margin-bottom:10px; box-shadow:0 0 6px rgba(0,0,0,0.1);">
                                    {alert} → Device(s): <b>{devices_str}</b>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                # ======================
                # Device Data Section
                # ======================
                st.subheader("🩺 Patient Devices")

                devices_in_df = sorted(df["device"].unique())
                cols = st.columns(min(len(devices_in_df), 5))  # keep 5 in one row

                for i, device in enumerate(devices_in_df):
                    device_df = df[df["device"] == device].sort_values("ts")
                    last_hr = round(device_df["hr"].iloc[-1], 1)
                    last_event = device_df["event"].iloc[-1]
                    badge = status_badge(last_hr, last_event)

                    with cols[i % 5]:
                        st.markdown(f"### {device}")
                        st.metric("Current HR", f"{last_hr} bpm")

                        if badge.startswith("🚨"):
                            st.error(badge)
                        elif badge.startswith("⚠"):
                            st.warning(badge)
                        else:
                            st.success(badge)

                        st.line_chart(device_df.set_index("ts")[["hr"]], height=180)

                # ======================
                # Footer
                # ======================
                st.markdown(
                    f"""
                    <div style="text-align:center; margin-top:25px; color:gray;">
                        ⏱ Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} 
                        | Refresh interval: {refresh_interval} seconds
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Print sampling time in terminal
        print(f"Sampling at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | Devices: {len(df)}")

        time.sleep(refresh_interval)

except KeyboardInterrupt:
    print("\nDashboard stopped by user.")
