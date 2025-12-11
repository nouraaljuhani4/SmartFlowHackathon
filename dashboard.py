import streamlit as st
import json

st.title("SmartFlow Dashboard")

with open("../detection/status.json", "r") as f:
    status = json.load(f)

for cam_id, info in status.items():
    st.subheader(info["name"])
    st.metric("Vehicles Detected", info["vehicles"])
    st.metric("Congestion Level", info["level"])
