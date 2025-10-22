import streamlit as st
import requests
import pandas as pd
import time

st.set_page_config(page_title="Wearable Gateway Dashboard", layout="wide")
st.title("Simulated Wearable Dashboard (Gateway)")

# Placeholder container for dynamic updates
placeholder = st.empty()

# Fetch data from gateway
def get_data():
    try:
        r = requests.get("http://localhost:8000/last", timeout=2)
        if r.status_code == 200:
            df = pd.DataFrame(r.json())
            if not df.empty:
                # Convert milliseconds timestamp to datetime
                df["ts"] = pd.to_datetime(df["ts"], unit='ms')
            return df
    except Exception as e:
        st.warning(f"Failed to fetch data: {e}")
    return pd.DataFrame()

# -----------------------
# Main live update loop
# -----------------------
while True:
    df = get_data()
    with placeholder.container():
        if df.empty:
            st.info("No data available yet.")
        else:
            df = df.sort_values("ts", ascending=False).reset_index(drop=True)

            # Latest readings table
            st.subheader("Latest Readings")
            st.dataframe(df)

            st.subheader("Heart Rate Trends by Device")
            for device in df["device"].unique():
                device_df = df[df["device"] == device].sort_values("ts")

                # Show alert if flatline detected
                if (device_df["hr"] == 0).any():
                    st.warning(f"🚨 {device}: CARDIAC ARREST DETECTED (Flatline)")

                st.line_chart(
                    device_df.set_index("ts")[["hr"]],
                    height=250,
                    use_container_width=True
                )
                st.markdown(f"**{device}** — showing {len(device_df)} samples")

            # Summary stats per device
            # Summary stats per device
            st.subheader("Device HR Summary")

# Ensure 'status' column exists
            if "status" not in df.columns:
                df["status"] = ""

            summary = df.groupby("device").agg(
                last_hr=("hr", "last"),
                avg_hr=("hr", "mean"),
                max_hr=("hr", "max"),
                min_hr=("hr", "min"),
                alerts=("status", lambda x: x.str.contains("alert", na=False).sum())
                ).reset_index()

            st.dataframe(summary)


    time.sleep(5)  # refresh every 5 seconds
