#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
home_rto_h3.py
------------------------------------------

Daily Home Return-To-Origin (RTO) diagnostic for H3 PEP mobility.

This script measures how far, on average, travelers move
*before returning to their own origin cell*.

For each origin o, HomeRTO(o,t) is defined as the median distance
of self-loop trips:

    HomeRTO(o,t) = D(o,o,t)

It then aggregates this into a **city-wide, flow-weighted index**
and tracks its evolution across days.

----------------------------------------------------------------------
INPUTS
----------------------------------------------------------------------

1) PEP OD data

    {RUN_DIR}/csvs/pep_od.csv

    Required columns:
        start_h3, end_h3, date, time,
        trip_count, mean_length_m

----------------------------------------------------------------------
COMPUTATION
----------------------------------------------------------------------

For each day in [SWEEP_START_DATE, SWEEP_END_DATE):

1) Select trips in the daily window:
       WINDOW_START → WINDOW_END

2) Keep only self-loop trips:
       start_h3 == end_h3

3) For each origin and time-bin:
       HomeRTO(o,t) = median(mean_length_m)

4) Compute start-flow weights:
       π(o,t) ∝ total trips starting at o at time t

5) Compute per-bin city mean:
       HomeRTO_city(t) = Σ_o π(o,t) · HomeRTO(o,t)

6) Compute daily average:
       HomeRTO_city(day) = mean_t HomeRTO_city(t)

----------------------------------------------------------------------
OUTPUTS
----------------------------------------------------------------------

All outputs are written to:

    {RUN_DIR}/home_rto/

1) CSV time series

    home_rto_h3_{SWEEP_START_DATE}_{SWEEP_END_DATE}_\
        {WINDOW_START}_{WINDOW_END}.csv

    Columns:
        date, HomeRTO

2) Interactive HTML plot

    home_rto_h3_{SWEEP_START_DATE}_{SWEEP_END_DATE}_\
        {WINDOW_START}_{WINDOW_END}.html

    • x-axis: date
    • y-axis: city-average HomeRTO (meters)

----------------------------------------------------------------------
INTERPRETATION
----------------------------------------------------------------------

High HomeRTO implies:
    • longer local tours before returning home
    • weaker neighborhood containment
    • more spatial mixing

Low HomeRTO implies:
    • compact local mobility
    • strong neighborhood loops
    • short-range daily travel

This metric complements:
    • effective distance (system-level)
    • path decomposition (route-level)
    • persistence sweeps (temporal stability)

----------------------------------------------------------------------
PIPELINE CONTEXT
----------------------------------------------------------------------

1) graph_builder_h3.py
2) pep_generator_h3.py
3) effective_distance_pep_h3.py
4) effective_path_decomposition_h3.py
5) time_sweep_effective_distance_h3.py
6) home_rto_h3.py   ← (this)

----------------------------------------------------------------------
AUTHOR
----------------------------------------------------------------------

Asif Shakeel  
ashakeel@ucsd.edu
"""


import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import write_html

import effective_distance_pep_h3 as h3ed   # only for RUN_DIR + column names


# ============================================================
# USER CONFIG
# ============================================================

PEP_SPAN_START = "2025-06-01 00:00:00"
PEP_SPAN_END   = "2025-06-30 00:00:00"

# ensure run directory matches PEP span
h3ed.PEP_SPAN_START = PEP_SPAN_START
h3ed.PEP_SPAN_END   = PEP_SPAN_END

h3ed.H3_RESOLUTION       = 6
h3ed.NEIGHBOR_TOPOLOGY   = "h3"
h3ed.GRAPH_MODE =  "corridors"   # "corridors" | "grid"
h3ed.FEEDER_MODE         = "single"
h3ed.EDGE_DIRECTION_MODE = "potential"
h3ed.INCLUDE_BACKGROUND_EDGES = False
h3ed.H3_OVERLAY_FLAG     = 0
h3ed.INIT_X_MODE         = "periodic_fixed_point"
h3ed.TIME_RES_MIN        = 30


SWEEP_START_DATE = "2025-06-01"
SWEEP_END_DATE   = "2025-06-30"

WINDOW_START = "06:00:00"
WINDOW_END   = "21:00:00"

OUTPUT_SUBDIR = "home_rto"

TITLE_SIZE      = 24
AXIS_TITLE_SIZE = 22
TICK_SIZE       = 20

from datetime import datetime, timedelta

display_end_date = (
    datetime.strptime(SWEEP_END_DATE, "%Y-%m-%d") - timedelta(days=1)
).strftime("%Y-%m-%d")
# ============================================================
# LOCAL PATH HELPERS (to avoid ED imports)
# ============================================================

def h3_to_latlon_local(h):
    """
    Local replacement for h3ed.h3_to_latlon or build_node_meta.
    Works if H3 library is installed; otherwise returns (0,0).
    """
    try:
        import h3
        lat, lon = h3.h3_to_geo(h)
        return float(lat), float(lon)
    except Exception:
        return 0.0, 0.0


def build_node_meta_local(nodes):
    """
    Local metadata builder.

    Returns:
        {h3_id: {"lat": ..., "lon": ...}}
    """
    meta = {}
    for h in nodes:
        lat, lon = h3_to_latlon_local(h)
        meta[h] = {"lat": lat, "lon": lon}
    return meta


def load_manifest_local(path):
    """
    Placeholder manifest loader for compatibility.
    Returns empty manifest (home RTO does not use it).
    """
    return {}


# ============================================================
# LOAD PEP FOR H3 (self-contained)
# ============================================================

def load_pep_h3(path):
    df = pd.read_csv(path)

    # DATE_COL is YYYYMMDD int
    date = df[h3ed.DATE_COL].astype(str)
    time = df[h3ed.TIME_COL].astype(str)

    dt = pd.to_datetime(date + " " + time, format="%Y%m%d %H:%M:%S")
    df["_t0"] = dt
    df["_date"] = dt.dt.date

    # Normalize H3 hex IDs
    df[h3ed.START_COL] = df[h3ed.START_COL].astype(str).str.lower()
    df[h3ed.END_COL]   = df[h3ed.END_COL].astype(str).str.lower()

    return df


# ============================================================
# UTIL
# ============================================================

def banner(x): print(f"\n=== {x} ===", flush=True)
def info(x):   print(f"[INFO] {x}", flush=True)


# ============================================================
# MAIN
# ============================================================

def main():

    banner("HOME RTO — H3 (LOCAL-PATH VERSION)")

    # Compute RUN_DIR (consistent with ED engine)
    h3ed.SPAN_START_TS = pd.to_datetime(PEP_SPAN_START)
    h3ed.SPAN_END_TS   = pd.to_datetime(PEP_SPAN_END)
    h3ed.RUN_DIR = h3ed.make_h3_run_directory()

    RUN_DIR = h3ed.RUN_DIR
    od_path = os.path.join(RUN_DIR, "csvs", "pep_od.csv")

    info(f"RUN_DIR: {RUN_DIR}")
    info(f"OD_PATH: {od_path}")

    df_all = load_pep_h3(od_path)

    days = pd.date_range(SWEEP_START_DATE, SWEEP_END_DATE, freq="D").date
    rows = []   # daily window-averaged city home RTO

    for di, day in enumerate(days, 1):
        print(f"\r[DAY {di}/{len(days)}] {day}", end='', flush=True)

        t0 = pd.to_datetime(f"{day} {WINDOW_START}")
        t1 = pd.to_datetime(f"{day} {WINDOW_END}")

        df_win = df_all[(df_all["_t0"] >= t0) & (df_all["_t0"] < t1)]
        if df_win.empty:
            continue

        # Self-loop rows (o→o)
        self_df = df_win[df_win[h3ed.START_COL] == df_win[h3ed.END_COL]]
        if self_df.empty:
            continue

        # per-bin per-origin home distance
        g = self_df.groupby([h3ed.START_COL, "_t0"], as_index=False)[h3ed.DIST_COL].median()
        g.rename(columns={h3ed.DIST_COL: "home_rto"}, inplace=True)

        # bin start-probabilities
        outflow = df_win.groupby([h3ed.START_COL, "_t0"])[h3ed.COUNT_COL].sum().reset_index()

        merged = g.merge(outflow, on=[h3ed.START_COL, "_t0"], how="left")

        # compute per-bin city average
        def compute_city_avg(df_bin):
            total = df_bin[h3ed.COUNT_COL].sum()
            if total <= 0:
                return np.nan
            pi = df_bin[h3ed.COUNT_COL] / total
            return float((pi * df_bin["home_rto"]).sum())

        city_bin_vals = merged.groupby("_t0").apply(compute_city_avg).dropna()
        if city_bin_vals.empty:
            continue

        rows.append({
            "date": pd.to_datetime(day),
            "HomeRTO": float(city_bin_vals.mean())
        })

    print()

    if not rows:
        banner("NO DATA — EXITING")
        return

    df_out = pd.DataFrame(rows)

    # Output directory
    out_dir = os.path.join(RUN_DIR, OUTPUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(
        out_dir,
        f"home_rto_h3_{SWEEP_START_DATE.replace('-','')}_"
        f"{SWEEP_END_DATE.replace('-','')}_"
        f"{WINDOW_START.replace(':','')}_"
        f"{WINDOW_END.replace(':','')}.csv"
    )
    df_out.to_csv(csv_path, index=False)
    info(f"Saved CSV → {csv_path}")

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=df_out["date"],
        y=df_out["HomeRTO"],
        mode="lines+markers",
        line=dict(width=2, color="black"),
        marker=dict(size=7, color="black"),
        name="Home RTO"
    ))

    fig.update_layout(
        title=dict(
            text=(
                f"Home RTO — City Average<br>"
                f"{SWEEP_START_DATE} → {display_end_date}, "
                f"{WINDOW_START}–{WINDOW_END}"
            ),
            font=dict(size=TITLE_SIZE)
        ),
        xaxis=dict(
            title=dict(text="Date", font=dict(size=AXIS_TITLE_SIZE)),
            tickfont=dict(size=TICK_SIZE)
        ),
        yaxis=dict(
            title=dict(text="RTO distance (m)", font=dict(size=AXIS_TITLE_SIZE)),
            tickfont=dict(size=TICK_SIZE),
            rangemode="tozero"
        ),
        template="plotly_white"
    )



    html_path = os.path.join(
        out_dir,
        f"home_rto_h3_{SWEEP_START_DATE.replace('-','')}_"
        f"{SWEEP_END_DATE.replace('-','')}_"
        f"{WINDOW_START.replace(':','')}_"
        f"{WINDOW_END.replace(':','')}.html"
    )
    write_html(fig, html_path, include_plotlyjs="cdn")
    info(f"Saved HTML → {html_path}")

    banner("DONE")


if __name__ == "__main__":
    main()
