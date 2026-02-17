from __future__ import annotations
import json
import os
import glob
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Hybrid FL Simulation Dashboard", layout="wide")


def find_latest_run(outputs_dir="outputs"):
    # prefer outputs/runs/* (aligned orchestrator+server)
    cand = []
    runs_dir = os.path.join(outputs_dir, "runs")
    if os.path.isdir(runs_dir):
        cand = sorted(glob.glob(os.path.join(runs_dir, "*")), key=os.path.getmtime)
    if not cand:
        cand = sorted(glob.glob(os.path.join(outputs_dir, "*")), key=os.path.getmtime)
    return cand[-1] if cand else None


def load_events(path: str, max_lines: int = 2000):
    if not os.path.exists(path):
        return []
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
    if df is None or df.empty or "round" not in df.columns:
        return 0
    return int(df["round"].max())


def extract_round_status(events, r: int):
    sel = recv = keep = drop = []
    agg = None
    train_loss = None
    train_acc = None
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
            train_loss = p.get("train_loss", train_loss)
            train_acc = p.get("train_acc", train_acc)

        ts = e.get("ts", "")
        log_lines.append(f"[{ts}] {t}: {p}")

    rows = []
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

    return sel, recv, keep, drop, agg, train_loss, train_acc, rows, log_lines[-50:]


st.title("Hybrid FL Simulation Dashboard")

run_dir = st.sidebar.text_input("Run directory", value=find_latest_run() or "")
auto = st.sidebar.checkbox("Auto refresh (2s)", value=True)

if auto:
    try:
        st.autorefresh(interval=2000, key="refresh")
    except Exception:
        st.info("Auto refresh not supported in this Streamlit version. Press R to refresh the browser.")

if not run_dir or not os.path.isdir(run_dir):
    st.warning("No valid run directory found. Run the server first to generate outputs/runs/<run_id>/")
    st.stop()

metrics_path = os.path.join(run_dir, "round_metrics.csv")
events_path = os.path.join(run_dir, "events.jsonl")
orch_round_path = os.path.join(run_dir, "server_round.csv")  # from orchestrator

df = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else pd.DataFrame()
events = load_events(events_path)
r_latest = latest_round_from_metrics(df)

col1, col2 = st.columns([2, 2])

with col1:
    st.subheader("Global Metric vs. Round")

    if df is None or df.empty:
        st.info("No metrics yet.")
    else:
        df_plot = df.copy()

        # coerce numeric columns (may be empty strings)
        for c in ["agg_scalar", "train_loss", "train_acc"]:
            if c in df_plot.columns:
                df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce")

        # metric chooser (default: train_acc if exists)
        candidates = []
        if "train_acc" in df_plot.columns:
            candidates.append("train_acc")
        if "train_loss" in df_plot.columns:
            candidates.append("train_loss")
        if "agg_scalar" in df_plot.columns:
            candidates.append("agg_scalar")

        if not candidates:
            st.info("No known metric columns found.")
        else:
            default_idx = 0
            metric = st.selectbox("Metric", candidates, index=default_idx)
            st.line_chart(df_plot.set_index("round")[[metric]])

with col2:
    st.subheader("Cumulative Carbon")

    if os.path.exists(orch_round_path):
        srv = pd.read_csv(orch_round_path)
        if "round" in srv.columns:
            srv = srv.sort_values("round")
        # most important column in your orchestrator schema is co2_total_g
        if "co2_total_g" in srv.columns:
            srv["co2_total_g"] = pd.to_numeric(srv["co2_total_g"], errors="coerce").fillna(0.0)
            srv["cum_co2_g"] = srv["co2_total_g"].cumsum()
            st.metric("Cumulative CO2 (g)", f"{srv['cum_co2_g'].iloc[-1]:.2f}" if len(srv) else "0.00")
            st.line_chart(srv.set_index("round")[["cum_co2_g"]] if "round" in srv.columns else srv[["cum_co2_g"]])
        else:
            st.info("server_round.csv found but 'co2_total_g' column not present.")
    else:
        st.info("server_round.csv not found in this run dir. "
                "Tip: ensure server logs to outputs/runs/<run_id> (aligned with orchestrator).")

st.subheader(f"Current Round Status (Round {r_latest})")
sel, recv, keep, drop, agg, t_loss, t_acc, rows, log_lines = extract_round_status(events, r_latest)

c1, c2, c3, c4 = st.columns(4)
c1.metric("SELECT", len(sel))
c2.metric("RECV", len(recv))
c3.metric("KEEP", len(keep))
c4.metric("DROP", len(drop))

m1, m2, m3 = st.columns(3)
m1.metric("AGG_SCALAR", f"{agg:.6f}" if isinstance(agg, (int, float)) else "n/a")
m2.metric("TRAIN_LOSS (kept-wavg)", f"{t_loss:.6f}" if isinstance(t_loss, (int, float)) else "n/a")
m3.metric("TRAIN_ACC (kept-wavg)", f"{t_acc:.4f}" if isinstance(t_acc, (int, float)) else "n/a")

st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.subheader("Event Log (tail)")
st.code("\n".join(log_lines) if log_lines else "(empty)")
