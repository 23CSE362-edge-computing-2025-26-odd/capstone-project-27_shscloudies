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
def status_badge(hr):
    if hr == 0:
        return "🚨 CARDIAC ARREST"
    elif hr < 50:
        return "⚠️ Low HR"
    elif hr > 120:
        return "⚠️ High HR"
    else:
        return "✅ Normal"


# -----------------------
# Main Dashboard (Live Update)
# -----------------------
# This keeps the layout same, but values refresh every 5 seconds smoothly
refresh_interval = 5  # seconds

data_placeholder = st.empty()

# Live update loop
while True:
    df = get_data()
    with data_placeholder.container():
        if df.empty:
            st.info("No data available yet.")
        else:
            df = df.sort_values("ts", ascending=True).reset_index(drop=True)

            # ======================
            # Alerts Section
            # ======================
            critical_devices = df[df["hr"] == 0]["device"].unique().tolist()
            if critical_devices:
                devices_str = ", ".join(critical_devices)
                st.markdown(
                    f"""
                    <div style="background-color:#ff4d4d; padding:20px; border-radius:10px; color:white; font-size:18px;">
                        🚨 <b>CRITICAL ALERT:</b> Cardiac Arrest detected on <b>{devices_str}</b>! Immediate Attention Required!
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div style="background-color:#0fa84f; padding:15px; border-radius:10px; color:white; font-size:16px;">
                        ✅ All patients stable. No critical alerts.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ======================
            # Device Cards
            # ======================
            st.subheader("🩺 Patient Devices")
            devices = df["device"].unique()
            cols = st.columns(len(devices))

            for i, device in enumerate(devices):
                device_df = df[df["device"] == device].sort_values("ts")
                last_hr = round(device_df["hr"].iloc[-1], 1)
                avg_hr = round(device_df["hr"].mean(), 1)
                min_hr = int(device_df["hr"].min())
                max_hr = int(device_df["hr"].max())
                badge = status_badge(last_hr)

                with cols[i]:
                    st.markdown(f"### {device}")
                    st.metric("Current HR", f"{last_hr} bpm")

                    if badge.startswith("🚨"):
                        st.error(badge)
                    elif badge.startswith("⚠️"):
                        st.warning(badge)
                    else:
                        st.success(badge)

                    # Trend chart
                    st.line_chart(
                        device_df.set_index("ts")[["hr"]],
                        height=180,
                        use_container_width=True
                    )

                    # HR Summary box
                    st.markdown(
                        f"""
                        <div style="
                            background-color:#f0f2f6;
                            padding:10px;
                            border-radius:8px;
                            text-align:center;
                            font-size:16px;
                            color:#111;">
                            <b>HR Summary</b><br>
                            Avg: <b>{avg_hr} bpm</b> &nbsp; | &nbsp;
                            Min: <b>{min_hr} bpm</b> &nbsp; | &nbsp;
                            Max: <b>{max_hr} bpm</b>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # ======================
            # Footer Info
            # ======================
            st.markdown(
                f"""
                <div style="text-align:center; margin-top:25px; color:gray;">
                    ⏱ Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | Refresh interval: {refresh_interval} seconds
                </div>
                """,
                unsafe_allow_html=True
            )

    # Wait before updating — no reloads
    time.sleep(refresh_interval)
