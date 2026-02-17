from __future__ import annotations
import json
import os
import glob
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Hybrid FL Simulation Dashboard", layout="wide")

def find_latest_run(outputs_dir="outputs"):
    runs = sorted(glob.glob(os.path.join(outputs_dir, "*")), key=os.path.getmtime)
    return runs[-1] if runs else None

def load_events(path: str, max_lines: int = 2000):
    if not os.path.exists(path):
        return []
    # 读最后若干行，够显示 current status + log
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()[-max_lines:]
    events = []
    for ln in lines:
        try:
            events.append(json.loads(ln))
        except Exception:
            pass
    return events

def latest_round_from_metrics(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    return int(df["round"].max())

def extract_round_status(events, r: int):
    # 找到该 round 最近的 SELECT/RECV/VERDICT/AGG
    sel = recv = keep = drop = []
    agg = None
    log_lines = []
    for e in events:
        if e.get("round") != r:
            continue
        t = e.get("type")
        p = e.get("payload", {})
        if t == "SELECT":
            sel = p.get("selected", sel)
        elif t == "RECV":
            recv = p.get("received", recv)
        elif t == "VERDICT":
            keep = p.get("keep", keep)
            drop = p.get("drop", drop)
        elif t == "AGG":
            agg = p.get("agg_scalar", agg)
        # 事件日志行（简化显示）
        ts = e.get("ts", "")
        log_lines.append(f"[{ts}] {t}: {p}")
    # 生成类似“Current Round Status”表
    rows = []
    # 这里只能展示 selection/verdict；后续你接 SUMO/OMNeT 再补 location/delay
    all_c = sorted(set(sel) | set(recv) | set(keep) | set(drop))
    for cid in all_c:
        status = "UNKNOWN"
        if cid in keep:
            status = "SUCCESS"
        elif cid in drop:
            status = "FAILED/DROPPED"
        elif cid in sel:
            status = "SELECTED"
        rows.append({"Client ID": cid, "Status": status})
    return sel, recv, keep, drop, agg, rows, log_lines[-50:]  # 只显示最后 50 行

st.title("Hybrid FL Simulation Dashboard")

run_dir = st.sidebar.text_input("Run directory", value=find_latest_run() or "")
auto = st.sidebar.checkbox("Auto refresh (2s)", value=True)
# if auto:
#     st.experimental_set_query_params(_="refresh")
#     st.autorefresh(interval=2000, key="refresh")

# # Auto refresh (works on most recent Streamlit)
# if auto:
#     st.autorefresh(interval=2000, key="refresh")

if auto:
    try:
        st.autorefresh(interval=2000, key="refresh")
    except Exception:
        st.info("Auto refresh not supported in this Streamlit version. Press R to refresh the browser.")


if not run_dir or not os.path.isdir(run_dir):
    st.warning("No valid run directory found. Run the server first to generate outputs/<run_id>/")
    st.stop()

metrics_path = os.path.join(run_dir, "round_metrics.csv")
events_path = os.path.join(run_dir, "events.jsonl")

col1, col2 = st.columns([2, 2])

df = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else pd.DataFrame()
events = load_events(events_path)

r_latest = latest_round_from_metrics(df)

with col1:
    st.subheader("Global Metric vs. Round")
    if not df.empty and "agg_scalar" in df.columns:
        # agg_scalar 可能是空字符串，转数值
        df_plot = df.copy()
        df_plot["agg_scalar"] = pd.to_numeric(df_plot["agg_scalar"], errors="coerce")
        st.line_chart(df_plot.set_index("round")[["agg_scalar"]])
    else:
        st.info("No metrics yet.")

with col2:
    st.subheader("Cumulative (Placeholder)")
    # 你后面把 carbon/cumulative_carbon 写进 round_metrics.csv，这里直接画即可
    if not df.empty and "carbon_g" in df.columns:
        df_plot = df.copy()
        df_plot["carbon_g"] = pd.to_numeric(df_plot["carbon_g"], errors="coerce")
        df_plot["cum_carbon_g"] = df_plot["carbon_g"].fillna(0).cumsum()
        st.line_chart(df_plot.set_index("round")[["cum_carbon_g"]])
    else:
        st.info("Carbon not logged in T0. (Will appear once you add accounting.)")

st.subheader(f"Current Round Status (Round {r_latest})")
sel, recv, keep, drop, agg, rows, log_lines = extract_round_status(events, r_latest)

c1, c2, c3, c4 = st.columns(4)
c1.metric("SELECT", len(sel))
c2.metric("RECV", len(recv))
c3.metric("KEEP", len(keep))
c4.metric("DROP", len(drop))
st.write(f"AGG_SCALAR: {agg}")

st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.subheader("Event Log (tail)")
st.code("\n".join(log_lines) if log_lines else "(empty)")
