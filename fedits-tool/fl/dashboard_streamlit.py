#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh  # 添加这一行

try:
    import plotly.express as px
except Exception:
    px = None


def reason_group(raw_reason: str) -> str:
    r = (raw_reason or "").strip().lower()

    if r in {"veh_missing", "mobility_missing", "veh_not_found", "left_map", "out_of_map", "veh_gone", "no_host"}:
        return "out_of_map"
    if "missing" in r or "left_map" in r or "out_of_map" in r:
        return "out_of_map"

    if r in {"left_coverage", "out_of_range", "dl_left_range", "ul_left_range", "ul_out_of_range", "dl_out_of_range"}:
        return "out_of_range"
    if "coverage" in r or "out_of_range" in r or "left_range" in r:
        return "out_of_range"

    if r in {"deadline", "dl_deadline_miss", "ul_deadline_miss", "ul_start_after_deadline", "link_down", "ul_link_down", "bad_signal"}:
        return "bad_signal_or_deadline"
    if "deadline" in r or "signal" in r or "link" in r or "per" in r or "pdr" in r:
        return "bad_signal_or_deadline"

    return "bad_signal_or_deadline"


def discover_runs(outputs_dir: str = "outputs") -> list[str]:
    base = Path(outputs_dir)
    if not base.exists():
        return []
    runs = [p for p in base.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in runs]


def safe_read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def safe_read_events(path: str, max_lines: int = 5000) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    rows = []
    try:
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# def plot_line(df: pd.DataFrame, x: str, y: str, title: str):
#     if df.empty or x not in df.columns or y not in df.columns:
#         st.info(f"Missing data for: {title}")
#         return
#     if px is None:
#         st.line_chart(df.set_index(x)[y], height=260)
#     else:
#         st.plotly_chart(px.line(df, x=x, y=y, markers=True, title=title), use_container_width=True)

def plot_line(df: pd.DataFrame, x: str, y: str, title: str):
    # 1. 检查数据
    if df.empty or x not in df.columns or y not in df.columns:
        st.info(f"Missing data for: {title}")
        return

    # 2. 【关键修改】在画图前，强制显示标题
    # 使用 st.subheader 或者 st.markdown("**" + title + "**") 都可以
    st.subheader(title)

    # 3. 开始画图
    if px is None:
        # st.line_chart 没有 title 参数，全靠上面的 st.subheader
        st.line_chart(df.set_index(x)[y], height=260)
    else:
        # 如果是 Plotly，因为我们在上面已经打印了标题，这里就不要再传 title 参数了
        # 否则会出现“头顶一个大标题，图里又一个小标题”的重复情况
        st.plotly_chart(px.line(df, x=x, y=y, markers=True), use_container_width=True)

def plot_stacked_bar(df: pd.DataFrame, x: str, y: str, color: str, title: str):
    if df.empty or any(c not in df.columns for c in [x, y, color]):
        st.info(f"Missing data for: {title}")
        return
    if px is None:
        st.dataframe(df)
    else:
        st.plotly_chart(px.bar(df, x=x, y=y, color=color, barmode="stack", title=title), use_container_width=True)


st.set_page_config(page_title="FL Orchestrator Dashboard", layout="wide")
st.title("FL Orchestrator Dashboard")

# runs = discover_runs("outputs")
runs = discover_runs("outputs/runs")
if not runs:
    st.warning("No runs found under ./outputs")
    st.stop()

with st.sidebar:
    st.header("Run selection")
    run_dir = st.selectbox("Run directory", runs, index=0)
    auto_refresh = st.checkbox("Auto refresh", value=False)
    refresh_s = st.slider("Refresh period (s)", 1, 10, 2) if auto_refresh else 0

if auto_refresh:
    # st.autorefresh(interval=refresh_s * 1000, key="autorefresh")
    # 注意这里改成了 st_autorefresh
    st_autorefresh(interval=refresh_s * 1000, key="autorefresh")

clients_csv = os.path.join(run_dir, "clients_round.csv")
server_csv = os.path.join(run_dir, "server_round.csv")
round_csv = os.path.join(run_dir, "round_metrics.csv")
events_jsonl = os.path.join(run_dir, "events.jsonl")

df_clients = safe_read_csv(clients_csv)
df_server = safe_read_csv(server_csv)
df_round = safe_read_csv(round_csv)
df_events = safe_read_events(events_jsonl)

if not df_clients.empty:
    if "drop_group" not in df_clients.columns:
        df_clients["drop_group"] = df_clients.get("drop_reason", "").astype(str).apply(reason_group)
    df_clients["drop_reason"] = df_clients.get("drop_reason", "").fillna("").astype(str)

tab_overview, tab_drop, tab_events = st.tabs(["Overview", "Drop analysis", "Events"])

with tab_overview:
    c1, c2 = st.columns(2)

    with c1:
        # st.subheader("Training / aggregation metrics")
        # if not df_round.empty and {"round", "eval_accuracy"}.issubset(df_round.columns):
        #     plot_line(df_round, "round", "eval_accuracy", "Eval accuracy")
        # elif not df_round.empty and {"round", "agg_scalar"}.issubset(df_round.columns):
        #     plot_line(df_round, "round", "agg_scalar", "Agg scalar (proxy)")

        # if not df_round.empty and {"round", "eval_loss"}.issubset(df_round.columns):
        #     plot_line(df_round, "round", "eval_loss", "Eval loss")
        
        if not df_round.empty and {"round", "train_acc"}.issubset(df_round.columns):
            plot_line(df_round, "round", "train_acc", "Train accuracy")
        elif not df_round.empty and {"round", "agg_scalar"}.issubset(df_round.columns):
            plot_line(df_round, "round", "agg_scalar", "Agg scalar (proxy)")

        if not df_round.empty and {"round", "train_loss"}.issubset(df_round.columns):
            plot_line(df_round, "round", "train_loss", "Train loss")

        if not df_server.empty and {"round", "dropout_rate"}.issubset(df_server.columns):
            plot_line(df_server, "round", "dropout_rate", "Dropout rate")

    with c2:
        st.subheader("Carbon per round (gCO2e)")
        if not df_server.empty and {"round", "co2_committed_g"}.issubset(df_server.columns):
            melt_cols = [c for c in ["co2_committed_g", "co2_dropped_g", "co2_total_g"] if c in df_server.columns]
            df_m = df_server[["round"] + melt_cols].melt(id_vars="round", var_name="metric", value_name="gco2e")
            if px is None:
                st.line_chart(df_server.set_index("round")[melt_cols], height=300)
            else:
                st.plotly_chart(px.line(df_m, x="round", y="gco2e", color="metric", markers=True,
                                        title="Committed vs Dropped vs Total"), use_container_width=True)
        else:
            st.info("server_round.csv missing carbon columns.")

    st.subheader("Latest rows")
    l1, l2 = st.columns(2)
    with l1:
        st.caption("clients_round.csv (tail)")
        st.dataframe(df_clients.tail(20) if not df_clients.empty else pd.DataFrame())
    with l2:
        st.caption("server_round.csv (tail)")
        st.dataframe(df_server.tail(20) if not df_server.empty else pd.DataFrame())

with tab_drop:
    st.subheader("Drop counts by reason group (3 buckets)")

    if df_clients.empty:
        st.info("clients_round.csv not found or empty.")
    else:
        df_d = df_clients[df_clients.get("committed", 1) == 0].copy()
        if df_d.empty:
            st.success("No drops in this run.")
        else:
            df_cnt = (df_d.groupby(["round", "drop_group"]).size()
                      .reset_index(name="count").sort_values(["round", "drop_group"]))
            plot_stacked_bar(df_cnt, "round", "count", "drop_group", "Drops per round (count)")

            if "co2_total_g" in df_d.columns:
                df_co2 = (df_d.groupby(["round", "drop_group"])["co2_total_g"].sum()
                          .reset_index().rename(columns={"co2_total_g": "gco2e"}))
                plot_stacked_bar(df_co2, "round", "gco2e", "drop_group", "Dropped CO2e per round (g) by group")

            st.subheader("Raw reason mapping (debug)")
            df_map = (df_d.groupby(["drop_reason", "drop_group"]).size()
                      .reset_index(name="count").sort_values("count", ascending=False))
            st.dataframe(df_map, use_container_width=True)

            st.subheader("Evidence plots (optional)")
            evidence_cols = [c for c in [
                "dl_dist_end_m", "ul_dist_end_m",
                "dl_sinr_db", "ul_sinr_db",
                "dl_rx_power_dbm", "ul_rx_power_dbm",
                "dl_per", "ul_per"
            ] if c in df_d.columns]

            if not evidence_cols:
                st.info("No evidence columns found yet. (Add dist/sinr/rx_power/per from Veins to enable.)")
            else:
                col = st.selectbox("Pick evidence column", evidence_cols, index=0)
                df_show = df_d[["round", "drop_group", "drop_reason", col]].copy()
                df_show[col] = pd.to_numeric(df_show[col], errors="coerce")
                df_show = df_show.dropna(subset=[col])
                if df_show.empty:
                    st.info(f"{col} present but all NaN.")
                elif px is None:
                    st.dataframe(df_show.head(300))
                else:
                    st.plotly_chart(px.box(df_show, x="drop_group", y=col, points="all",
                                           title=f"{col} distribution by drop_group"),
                                    use_container_width=True)

with tab_events:
    st.subheader("events.jsonl")
    if df_events.empty:
        st.info("No events.jsonl found (or empty).")
    else:
        st.dataframe(df_events.tail(400), use_container_width=True)
