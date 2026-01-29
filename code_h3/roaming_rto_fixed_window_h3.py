#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
roaming_rto_fixed_window_h3.py
------------------------------------------

Roaming Return-To-Origin (RTO) diagnostic for H3 PEP mobility.

This script measures how far, on average, agents travel
*before returning to their own origin cell*, using the
**effective-distance first-hitting process**.

For each origin o on each day:

    RTO(o) = D_elapsed(o,o) / P_hit(o,o)

where
    D_elapsed(o,o) = expected distance accumulated before first return
    P_hit(o,o)     = probability of eventually returning to o
                    within the window.

It then aggregates this into a **flow-weighted city-level index**
and tracks both origin-wise and city-average RTO across time.

This corresponds to Section 5 and Fig. 9(a,b) of the paper.

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

2) Restrict to the LSCC of the OD graph.

3) Build time-slice Markov kernels and distance operators.

4) Compute first-hitting effective distances:
       X_elapsed(i,j),  P_hit(i,j)

5) Origin-wise roaming RTO:
       RTO(o) = X_elapsed(o,o) / P_hit(o,o)

   (only if P_hit(o,o) ≥ MIN_P_HIT)

6) Compute start-flow weights:
       π(o) ∝ total trips starting at o

7) City-wide average:
       RTO_city = Σ_o π(o) · RTO(o)

----------------------------------------------------------------------
OUTPUTS
----------------------------------------------------------------------

All outputs are written to:

    {RUN_DIR}/roaming_rto_fixed_window/

1) Origin-wise CSV

    rto_originwise_{SWEEP_START_DATE}_{SWEEP_END_DATE}_\
        {WINDOW_START}_{WINDOW_END}.csv

    Columns:
        origin, date, RTO

2) City-average CSV

    rto_cityavg_{SWEEP_START_DATE}_{SWEEP_END_DATE}_\
        {WINDOW_START}_{WINDOW_END}.csv

    Columns:
        origin="CITY_AVG", date, RTO

3) City-average HTML plot
    • single black curve
    • x-axis: date
    • y-axis: RTO (meters)

4) Origin-wise HTML plot
    • one colored trajectory per origin
    • same axes as above

----------------------------------------------------------------------
INTERPRETATION
----------------------------------------------------------------------

High RTO implies:
    • long roaming tours before returning home
    • weak spatial containment
    • strong long-range circulation

Low RTO implies:
    • fast local returns
    • compact roaming behavior
    • strong neighborhood loops

This metric is the *self-return* analogue of effective distance
and complements:

    • HomeRTO (empirical self-loops)
    • Effective distance (OD-level)
    • Path decomposition (route-level)
    • Persistence sweeps (temporal structure)

----------------------------------------------------------------------
PIPELINE CONTEXT
----------------------------------------------------------------------

1) graph_builder_h3.py
2) pep_generator_h3.py
3) effective_distance_pep_h3.py
4) effective_path_decomposition_h3.py
5) time_sweep_effective_distance_h3.py
6) home_rto_h3.py
7) roaming_rto_fixed_window_h3.py   ← (this)

----------------------------------------------------------------------
AUTHOR
----------------------------------------------------------------------

Asif Shakeel  
ashakeel@ucsd.edu
"""


import os
import colorsys
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import write_html

import effective_distance_pep_h3 as h3ed


# ============================================================
# USER CONFIG
# ============================================================

PEP_SPAN_START = "2025-06-01 00:00:00"
PEP_SPAN_END   = "2025-06-30 00:00:00"

h3ed.PEP_SPAN_START = PEP_SPAN_START
h3ed.PEP_SPAN_END   = PEP_SPAN_END

h3ed.COUNTRY     = "USA" # "Mexico"
h3ed.REGION_NAME = "Atlanta, Georgia" # "Mexico City, Mexico"
h3ed.H3_RESOLUTION         = 6
h3ed.GRAPH_MODE =   "corridors"   # "corridors" | "grid"
h3ed.NEIGHBOR_TOPOLOGY     = "h3"
h3ed.FEEDER_MODE           = "single"
h3ed.EDGE_DIRECTION_MODE   = "potential"
h3ed.INCLUDE_BACKGROUND_EDGES = False
h3ed.H3_OVERLAY_FLAG       = 0
h3ed.INIT_X_MODE           = "periodic_fixed_point"
h3ed.TIME_RES_MIN          = 30

SWEEP_START_DATE = "2025-06-01"
SWEEP_END_DATE   = "2025-06-30"

WINDOW_START = "06:00:00"
WINDOW_END   = "21:00:00"

MIN_P_HIT = 1e-6

OUTPUT_SUBDIR = "roaming_rto_fixed_window"

TITLE_SIZE      = 24
AXIS_TITLE_SIZE = 22
TICK_SIZE       = 20

from datetime import datetime, timedelta

display_end_date = (
    datetime.strptime(SWEEP_END_DATE, "%Y-%m-%d") - timedelta(days=1)
).strftime("%Y-%m-%d")
# ============================================================
# UTILITIES
# ============================================================

def banner(x): print(f"\n=== {x} ===", flush=True)
def info(x):   print(f"[INFO] {x}", flush=True)

def build_color_map(keys: List[str]) -> Dict[str,str]:
    """Assign stable colors."""
    cmap = {}
    for k, key in enumerate(keys):
        hue = (k * 127.713) % 360
        r,g,b = colorsys.hsv_to_rgb(hue/360., 0.75, 0.92)
        cmap[key] = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
    return cmap


# ============================================================
# MAIN
# ============================================================

def main():

    banner("RTO — H3")

    # Force RUN_DIR to match PEP_SPAN
    h3ed.SPAN_START_TS = pd.to_datetime(h3ed.PEP_SPAN_START)
    h3ed.SPAN_END_TS   = pd.to_datetime(h3ed.PEP_SPAN_END)
    h3ed.RUN_DIR = h3ed.make_h3_run_directory()

    RUN_DIR = h3ed.RUN_DIR
    od_path = os.path.join(RUN_DIR, "csvs", "pep_od.csv")

    info(f"RUN_DIR: {RUN_DIR}")
    info(f"OD_PATH: {od_path}")

    # Load OD
    df_all = h3ed.load_pep(od_path)
    df_all[h3ed.START_COL] = df_all[h3ed.START_COL].astype(str).str.lower()
    df_all[h3ed.END_COL]   = df_all[h3ed.END_COL].astype(str).str.lower()

    days = pd.date_range(SWEEP_START_DATE, SWEEP_END_DATE, freq="D").date
    rows = []   # both origin-wise and city-average stored here

    banner("DAILY RTO COMPUTATION")

    for di, day in enumerate(days, 1):
        print(f"\r[DAY {di}/{len(days)}] {day}", end='', flush=True)

        t0 = pd.to_datetime(f"{day} {WINDOW_START}")
        t1 = pd.to_datetime(f"{day} {WINDOW_END}")

        df_win0 = df_all[(df_all["_t0"] >= t0) & (df_all["_t0"] < t1)]
        if df_win0.empty:
            continue

        lscc = h3ed.compute_lscc_nodes(df_win0)
        lscc = [x.lower() for x in lscc]
        if not lscc:
            continue

        df_win = df_win0[
            df_win0[h3ed.START_COL].isin(lscc) &
            df_win0[h3ed.END_COL].isin(lscc)
        ].copy()

        nodes = sorted(lscc)
        if not nodes:
            continue

        # Packs for ED
        slices, node_index, packs = h3ed.build_packs(df_win, nodes)
        N = len(nodes)

        # Reachable origins
        pos_union = df_win.groupby(h3ed.START_COL)[h3ed.COUNT_COL].sum()
        reachable = [node_index[o] for o in pos_union.index if o in node_index]
        if not reachable:
            continue

        # ED
        X_elapsed, P_hit = h3ed.batched_elapsed_all_origins(packs, N, reachable)

        # ------------------------------------------------------------
        # (1) Origin-level RTO(o,t)
        # ------------------------------------------------------------
        rto_dict = {}
        for o in nodes:
            i = node_index[o]
            x = X_elapsed[i, i]
            p = P_hit[i, i]
            if p >= MIN_P_HIT:
                rto_dict[o] = x 

        if not rto_dict:
            continue

        # ------------------------------------------------------------
        # (2) Starting probability π(o,t)
        # ------------------------------------------------------------
        outflow = df_win.groupby(h3ed.START_COL)[h3ed.COUNT_COL].sum()
        outflow = outflow.reindex(nodes).fillna(0)

        total_out = outflow.sum()
        pi = outflow / total_out if total_out > 0 else outflow * 0

        # ------------------------------------------------------------
        # (3) City average
        # ------------------------------------------------------------
        city_rto = float(sum(pi[o] * rto_dict[o] for o in rto_dict))

        rows.append({
            "origin": "CITY_AVG",
            "date": pd.to_datetime(day),
            "RTO": city_rto
        })

        # ------------------------------------------------------------
        # (4) Origin-wise rows
        # ------------------------------------------------------------
        for o, v in rto_dict.items():
            rows.append({
                "origin": o,
                "date": pd.to_datetime(day),
                "RTO": v
            })

    print()

    if not rows:
        banner("NO RTO DATA — EXITING")
        return

    df_rto = pd.DataFrame(rows)

    # ============================================================
    # SAVE BOTH CSVs
    # ============================================================

    out_dir = os.path.join(RUN_DIR, OUTPUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    # City average only
    df_city = df_rto[df_rto["origin"] == "CITY_AVG"].copy()
    csv_city = os.path.join(
        out_dir,
        f"rto_cityavg_{SWEEP_START_DATE.replace('-','')}_"
        f"{SWEEP_END_DATE.replace('-','')}_{WINDOW_START.replace(':','')}_"
        f"{WINDOW_END.replace(':','')}.csv"
    )
    df_city.to_csv(csv_city, index=False)
    info(f"Saved CITY-AVG CSV → {csv_city}")

    # Origin-wise only
    df_orig = df_rto[df_rto["origin"] != "CITY_AVG"].copy()
    csv_orig = os.path.join(
        out_dir,
        f"rto_originwise_{SWEEP_START_DATE.replace('-','')}_"
        f"{SWEEP_END_DATE.replace('-','')}_{WINDOW_START.replace(':','')}_"
        f"{WINDOW_END.replace(':','')}.csv"
    )
    df_orig.to_csv(csv_orig, index=False)
    info(f"Saved ORIGIN-WISE CSV → {csv_orig}")

    # ============================================================
    # PLOTS
    # ============================================================

    # ---- City-average plot (single line) ----
    fig_city = go.Figure()
    fig_city.add_trace(go.Scattergl(
        x=df_city["date"],
        y=df_city["RTO"],
        mode="lines+markers",
        marker=dict(size=7, color="black"),
        line=dict(width=2, color="black"),
        name="City Average"
    ))

    fig_city.update_layout(
        title=dict(
            text=(
                f"Roaming RTO — City Average<br>"
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

    html_city = os.path.join(
        out_dir,
        f"rto_cityavg_{SWEEP_START_DATE.replace('-','')}_"
        f"{SWEEP_END_DATE.replace('-','')}_"
        f"{WINDOW_START.replace(':','')}_"
        f"{WINDOW_END.replace(':','')}.html"
    )
    write_html(fig_city, html_city, include_plotlyjs="cdn")
    info(f"Saved CITY-AVG plot → {html_city}")


    # ---- Origin-wise plot ----
    cmap = build_color_map(sorted(df_orig["origin"].unique()))

    fig_orig = go.Figure()
    for o, sub in df_orig.groupby("origin", sort=False):
        fig_orig.add_trace(go.Scattergl(
            x=sub["date"],
            y=sub["RTO"],
            mode="lines+markers",
            marker=dict(size=5, color=cmap[o]),
            line=dict(width=1.3, color=cmap[o]),
            name=o
        ))

    fig_orig.update_layout(
        title=dict(
            text=f"Roaming RTO — Origin-wise<br>{SWEEP_START_DATE} to {SWEEP_END_DATE}",
            font=dict(size=TITLE_SIZE)
        ),
        xaxis=dict(
            title=dict(text="Date", font=dict(size=AXIS_TITLE_SIZE)),
            tickfont=dict(size=TICK_SIZE)
        ),
        yaxis=dict(
            title=dict(text="RTO (m)", font=dict(size=AXIS_TITLE_SIZE)),
            tickfont=dict(size=TICK_SIZE),
            rangemode="tozero"
        ),

        template="plotly_white",
        hovermode="closest"
    )



    html_orig = os.path.join(
        out_dir,
        f"rto_originwise_{SWEEP_START_DATE.replace('-','')}_"
        f"{SWEEP_END_DATE.replace('-','')}_"
        f"{WINDOW_START.replace(':','')}_"
        f"{WINDOW_END.replace(':','')}.html"
    )
    write_html(fig_orig, html_orig, include_plotlyjs="cdn")
    info(f"Saved ORIGIN-WISE plot → {html_orig}")


    banner("DONE")


if __name__ == "__main__":
    main()
