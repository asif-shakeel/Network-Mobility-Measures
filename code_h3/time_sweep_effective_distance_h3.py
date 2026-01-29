#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
time_sweep_effective_distance_h3.py
------------------------------------------

Time-sweep persistence analysis of extreme effective distances.

This script tracks *which origin–destination pairs repeatedly exhibit
extreme effective distance* across many days, within a fixed daily
time window.

For each day in a date range, it:

    • computes D_eff(o,d) for all OD pairs
    • filters by P_hit ≥ MIN_P_HIT
    • optionally restricts to GUP pairs
    • selects the top percentile band of D_eff
    • records which OD pairs appear

It then aggregates across days and produces a single interactive
scatter plot showing:

    • when each extreme OD appears
    • how large its D_eff is
    • how persistent it is (number of days in the band)

Legend entries are ordered by descending persistence.

----------------------------------------------------------------------
INPUTS
----------------------------------------------------------------------

1) PEP OD data (from pep_generator_h3.py)

    {RUN_DIR}/csvs/pep_od.csv

    Required columns:
        start_h3, end_h3, date, time, trip_count, mean_length_m

2) Effective distance engine
    effective_distance_pep_h3.py (imported as h3ed)

    Provides:
        • pack builder
        • LSCC extraction
        • batched elapsed distances
        • P_hit
        • GUP detection
        • regression for D_direct

----------------------------------------------------------------------
PARAMETERS
----------------------------------------------------------------------

Date sweep:
    SWEEP_START_DATE, SWEEP_END_DATE

Daily time window:
    WINDOW_START, WINDOW_END

Filters:
    MIN_P_HIT
    PERCENTILE_Q
    MODE = "gup" | "non-gup" | "all"

Graph + model configuration (overrides h3ed):
    REGION_NAME
    H3_RESOLUTION
    GRAPH_MODE
    FEEDER_MODE
    EDGE_DIRECTION_MODE
    TIME_RES_MIN
    INIT_X_MODE
    etc.

----------------------------------------------------------------------
COMPUTATION
----------------------------------------------------------------------

For each day in the sweep:

1) Subset OD to [day + WINDOW_START, day + WINDOW_END)

2) Compute LSCC of the OD graph

3) Build Markov packs M(t)

4) Compute:
       D_elapsed(i,j)
       P_hit(i,j)

5) Fit regional regression once:
       D_direct = α + β · geo_dist

6) For all OD pairs:
       D_eff = D_elapsed / D_direct

7) Filter:
       • P_hit ≥ MIN_P_HIT
       • MODE restriction (GUP or non-GUP)

8) Select top-q percentile of D_eff

9) Record (origin, dest, D_eff, date)

----------------------------------------------------------------------
OUTPUTS
----------------------------------------------------------------------

All outputs are written to:

    {RUN_DIR}/time_sweep_effective_distance/

1) Interactive HTML plot

    time_sweep_h3_{MODE}_q{PERCENTILE_Q}_\
        {SWEEP_START_DATE}_{SWEEP_END_DATE}_\
        {WINDOW_START}_{WINDOW_END}.html

    Each point represents one OD in the extreme band on one day.

    Axes:
        x = date
        y = D_eff

    Color:
        unique per OD pair

    Legend:
        OD → OD (n = number of days in band),
        ordered by descending n

    Hover tooltip:
        OD pair, date, D_eff, persistence

----------------------------------------------------------------------
INTERPRETATION
----------------------------------------------------------------------

This plot reveals:

    • structurally persistent bottlenecks
    • temporally stable anomalous corridors
    • OD pairs that are *systemically inefficient*,
      not just outliers on a single day.

It complements:

    • effective_distance_pep_h3.py (system-level)
    • effective_path_decomposition_h3.py (path-level)

by adding **temporal persistence**.

----------------------------------------------------------------------
PIPELINE CONTEXT
----------------------------------------------------------------------

1) graph_builder_h3.py
2) pep_generator_h3.py
3) effective_distance_pep_h3.py
4) effective_path_decomposition_h3.py
5) time_sweep_effective_distance_h3.py  ← (this)

----------------------------------------------------------------------
AUTHOR
----------------------------------------------------------------------

Asif Shakeel  
ashakeel@ucsd.edu
"""


import os
import colorsys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import write_html

import effective_distance_pep_h3 as h3ed
from datetime import datetime, timedelta


# ============================================================
# CONFIGURATION (safe overrides into h3ed)
# ============================================================

PEP_SPAN_START = "2025-06-01 00:00:00"
PEP_SPAN_END   = "2025-06-30 00:00:00"

h3ed.PEP_SPAN_START = PEP_SPAN_START
h3ed.PEP_SPAN_END   = PEP_SPAN_END

h3ed.COUNTRY     = "USA"
h3ed.REGION_NAME = "Atlanta, Georgia"
h3ed.H3_RESOLUTION       = 6
h3ed.NEIGHBOR_TOPOLOGY   = "h3"
h3ed.GRAPH_MODE = "corridors"
h3ed.FEEDER_MODE         = "single"
h3ed.EDGE_DIRECTION_MODE = "potential"
h3ed.INCLUDE_BACKGROUND_EDGES = False
h3ed.H3_OVERLAY_FLAG     = 0
h3ed.INIT_X_MODE         = "periodic_fixed_point"
h3ed.TIME_RES_MIN        = 30

SWEEP_START_DATE = "2025-06-01"
SWEEP_END_DATE   = "2025-06-30"

WINDOW_START = "06:00:00"
WINDOW_END   = "08:00:00"

MIN_P_HIT     = 1e-6
PERCENTILE_Q  = 99.0
MODE          = "gup" # "all" , "gup"

SHOW_PROGRESS = True

OUTPUT_SUBDIR = "time_sweep_effective_distance"

TITLE_SIZE      = 24
AXIS_TITLE_SIZE = 22
TICK_SIZE       = 20
LEGEND_TITLE_SIZE = 20
LEGEND_TEXT_SIZE  = 18

DISPLAY_END_DATE = (
    datetime.strptime(SWEEP_END_DATE, "%Y-%m-%d") - timedelta(days=1)
).strftime("%Y-%m-%d")

# ============================================================
# FORCE OVERRIDE OF PEP SPAN + REBUILD RUN_DIR
# ============================================================

h3ed.SPAN_START_TS = pd.to_datetime(h3ed.PEP_SPAN_START)
h3ed.SPAN_END_TS   = pd.to_datetime(h3ed.PEP_SPAN_END)
h3ed.RUN_DIR = h3ed.make_h3_run_directory()

print("[DEBUG] New RUN_DIR:", h3ed.RUN_DIR)


# ============================================================
# Utilities
# ============================================================

def banner(msg): print(f"\n=== {msg} ===", flush=True)
def info(msg):   print(f"[INFO] {msg}", flush=True)
def warn(msg):   print(f"[WARN] {msg}", flush=True)


def build_color_map(od_pairs: List[Tuple[str,str]]) -> Dict[Tuple[str,str], str]:
    cmap = {}
    for k, od in enumerate(od_pairs):
        hue = (k * 137.508) % 360
        r, g, b = colorsys.hsv_to_rgb(hue / 360., 0.78, 0.92)
        cmap[od] = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
    return cmap


# ============================================================
# MAIN
# ============================================================

def main():

    banner("TIME-SWEEP EFFECTIVE DISTANCE — H3")

    RUN_DIR = h3ed.RUN_DIR
    od_path = os.path.join(RUN_DIR, "csvs", "pep_od.csv")

    info(f"RUN_DIR: {RUN_DIR}")
    info(f"Loading OD: {od_path}")

    # ------------------------------------------------------------
    # Load OD
    # ------------------------------------------------------------
    df_all = h3ed.load_pep(od_path)

    df_all[h3ed.START_COL] = df_all[h3ed.START_COL].astype(str).str.lower()
    df_all[h3ed.END_COL]   = df_all[h3ed.END_COL].astype(str).str.lower()

    alpha, beta, rmse = h3ed.fit_region_regression(df_all)

    # ------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------
    days = pd.date_range(SWEEP_START_DATE, SWEEP_END_DATE, freq="D").date

    rows = []

    banner("DAILY COMPUTATION")

    for di, day in enumerate(days, 1):

        if SHOW_PROGRESS:
            print(f"\r[DAY {di}/{len(days)}] {day}", end='', flush=True)

        t0 = pd.to_datetime(f"{day} {WINDOW_START}")
        t1 = pd.to_datetime(f"{day} {WINDOW_END}")

        df_win0 = df_all[(df_all["_t0"] >= t0) & (df_all["_t0"] < t1)].copy()
        if df_win0.empty:
            continue

        lscc_nodes = [h.lower() for h in h3ed.compute_lscc_nodes(df_win0)]
        if not lscc_nodes:
            continue

        df_win = df_win0[
            df_win0[h3ed.START_COL].isin(lscc_nodes) &
            df_win0[h3ed.END_COL].isin(lscc_nodes)
        ].copy()
        if df_win.empty:
            continue

        nodes = sorted(lscc_nodes)
        slices, node_index, packs = h3ed.build_packs(df_win, nodes)

        pos_union = df_win.groupby(h3ed.START_COL)[h3ed.COUNT_COL].sum()
        reachable = [node_index[o] for o in pos_union.index if o in node_index]
        if not reachable:
            continue

        gup_pairs = h3ed.detect_gup_pairs(df_all, nodes)

        X_elapsed, P_hit = h3ed.batched_elapsed_all_origins(packs, len(nodes), reachable)

        all_od = [(o, d) for o in nodes for d in nodes if o != d]
        od_df = pd.DataFrame(all_od, columns=[h3ed.START_COL, h3ed.END_COL])

        i_idx = od_df[h3ed.END_COL].map(node_index).to_numpy(int)
        j_idx = od_df[h3ed.START_COL].map(node_index).to_numpy(int)

        od_df["D_elapsed"] = X_elapsed[i_idx, j_idx]
        od_df["P_hit"] = P_hit[i_idx, j_idx]

        od_df["D_direct"] = [
            h3ed.predict_direct_distance(o, d, alpha, beta)
            for (o, d) in all_od
        ]

        od_df["D_eff"] = od_df["D_elapsed"] / od_df["D_direct"]
        od_df = od_df.replace([np.inf, -np.inf], np.nan)
        od_df = od_df.dropna(subset=["D_eff"])
        od_df = od_df[od_df["P_hit"] >= MIN_P_HIT]

        if MODE == "gup":
            od_df = od_df[
                od_df.apply(lambda r: (r[h3ed.START_COL], r[h3ed.END_COL]) in gup_pairs, axis=1)
            ]
        elif MODE == "non-gup":
            od_df = od_df[
                od_df.apply(lambda r: (r[h3ed.START_COL], r[h3ed.END_COL]) not in gup_pairs, axis=1)
            ]

        if od_df.empty:
            continue

        thr = float(np.nanpercentile(od_df["D_eff"], PERCENTILE_Q))
        band = od_df[od_df["D_eff"] >= thr].copy()
        if band.empty:
            continue

        band["_date"] = pd.to_datetime(day)
        rows.append(band[[h3ed.START_COL, h3ed.END_COL, "D_eff", "_date"]])

    print()

    if not rows:
        warn("No sweep data — exiting.")
        return

    # ------------------------------------------------------------
    # Aggregate & persistence counts
    # ------------------------------------------------------------
    df_points = pd.concat(rows, ignore_index=True)
    df_points.rename(columns={
        h3ed.START_COL: "origin",
        h3ed.END_COL:   "dest",
        "_date":        "date"
    }, inplace=True)

    pair_counts = (
        df_points
        .groupby(["origin", "dest"])["date"]
        .nunique()
        .reset_index(name="n_days")
        .sort_values("n_days", ascending=False)
    )

    ordered_pairs = list(zip(pair_counts.origin, pair_counts.dest))
    count_lookup = {
        (r.origin, r.dest): int(r.n_days)
        for r in pair_counts.itertuples()
    }

    cmap = build_color_map(ordered_pairs)

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig = go.Figure()

    for (o, d) in ordered_pairs:

        sub = df_points[(df_points.origin == o) & (df_points.dest == d)]
        if sub.empty:
            continue

        n_days = count_lookup[(o, d)]

        fig.add_trace(go.Scattergl(
            x=sub["date"],
            y=sub["D_eff"],
            mode="markers",
            marker=dict(size=6, color=cmap[(o, d)]),
            name=f"{o}→{d} (n={n_days})",
            text=[f"{o}→{d}"] * len(sub),
            hovertemplate=(
                "OD: %{text}<br>"
                "Date: %{x|%Y-%m-%d}<br>"
                "D_eff: %{y:.2f}<br>"
                f"Occurrences: {n_days}"
                "<extra></extra>"
            )
        ))

    fig.update_layout(
        title=dict(
            text=(
                f"Effective Distance — Time Sweep ({MODE.upper()})<br>"
                f"{SWEEP_START_DATE} → {DISPLAY_END_DATE}, "
                f"Window {WINDOW_START}–{WINDOW_END}, "
                f"P_hit ≥ {MIN_P_HIT}, q={PERCENTILE_Q}%"
            ),
            font=dict(size=TITLE_SIZE)
        ),
        xaxis=dict(
            title=dict(text="Date", font=dict(size=AXIS_TITLE_SIZE)),
            tickfont=dict(size=TICK_SIZE)
        ),
        yaxis=dict(
            title=dict(
    text="D<sup>eff</sup><sub>ij</sub>",
    font=dict(size=AXIS_TITLE_SIZE)
),
            tickfont=dict(size=TICK_SIZE),
            rangemode="tozero"
        ),
        template="plotly_white",
        hovermode="closest",
        legend=dict(
            title=dict(
                text="OD pairs (ranked by persistence)",
                font=dict(size=LEGEND_TITLE_SIZE)
            ),
            font=dict(size=LEGEND_TEXT_SIZE),
            itemsizing="constant"
        ),
        margin=dict(l=70, r=300, t=100, b=70),
    )




    out_dir = os.path.join(RUN_DIR, OUTPUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    fname = (
        f"time_sweep_h3_{MODE}_q{str(PERCENTILE_Q).replace('.','_')}_"
        f"{SWEEP_START_DATE.replace('-','')}_"
        f"{SWEEP_END_DATE.replace('-','')}_"
        f"{WINDOW_START.replace(':','')}_"
        f"{WINDOW_END.replace(':','')}.html"
    )

    out_html = os.path.join(out_dir, fname)
    write_html(
        fig,
        out_html,
        include_plotlyjs="cdn",
        config={"mathjax": "cdn"}
    )
    banner(f"PLOT SAVED → {out_html}")
    banner("DONE")


if __name__ == "__main__":
    main()
