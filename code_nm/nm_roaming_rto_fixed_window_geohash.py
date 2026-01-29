#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nm_roaming_rto_fixed_window_geohash.py
================================

Roaming RTO from Aggregated Geohash OD Data
------------------------------------------

This script computes **Roaming RTO** for a single city / region using
time-resolved **aggregated origin–destination (OD) flows** on a
**geohash tiling** (NetMob 2024 format).

Roaming RTO measures the characteristic spatial distance of
**return-to-origin mobility** in the *network sense*, i.e. the
time-elapsed effective distance of returning to the same tile
after arbitrarily many intermediate moves.

The computation uses the same **LSCC + time-recursive first-hitting
distance pipeline** as the effective-distance engine described in:

    Network-Level Measures of Mobility from Aggregated Origin-Destination Data
    https://arxiv.org/abs/2502.04162


======================================================================
DEFINITION
======================================================================

Let M(t) be the column-stochastic transition matrix derived from
aggregated OD flows in time bin t.

Let X_elapsed(i,j) and P_hit(i,j) be the time-elapsed distance and
total first-hit probability obtained from the recursion in Sec. 4.1
of the paper.

For each origin tile o and time window t:

    RTO(o,t) = X_elapsed(o,o)        (with P_hit(o,o) > 0)

Let π(o,t) be the starting probability:

    π(o,t) = outflow(o,t) / Σ_o outflow(o,t)

Then the **city-average roaming RTO** is:

    CITY_AVG(t) = Σ_o π(o,t) · RTO(o,t)

Daily values are computed over a fixed within-day window and
aggregated across the sweep.


======================================================================
INPUT
======================================================================

Aggregated OD CSV (NetMob format):

    • start_geohash
    • end_geohash
    • date        (YYYYMMDD)
    • time        (HH:MM:SS)
    • trip_count
    • median_length_m

Example:
    od_mx_agg5_3h_MexicoCity_20190101T000000_20191231T000000.csv

Geohash resolution:
    GEOHASH_PRECISION = 5

Time resolution:
    TIME_RES_MIN = 30 minutes

Daily sweep:
    SWEEP_START_DATE → SWEEP_END_DATE

Within-day window:
    WINDOW_START → WINDOW_END


======================================================================
OUTPUT ARTIFACTS
======================================================================

All outputs are written to:

    <BASE_DIR>/roaming_rto_fixed_window/

Let <stem> be the input filename without extension.

----------------------------------------------------------------------
1) City-average roaming RTO (window-averaged per day)
----------------------------------------------------------------------

    <stem>_rto_cityavg_<dates>_<window>.csv

Columns:
    date, RTO

    <stem>_rto_cityavg_<dates>_<window>.html
        • interactive time series plot


----------------------------------------------------------------------
2) Origin-wise roaming RTO (window-averaged per day)
----------------------------------------------------------------------

    <stem>_rto_originwise_<dates>_<window>.csv

Columns:
    origin, date, RTO

    <stem>_rto_originwise_<dates>_<window>.html
        • interactive multi-line time series plot


======================================================================
INTERPRETATION
======================================================================

Roaming RTO captures the *network-scale radius of return*:
how far, on average, a random walker must travel before first
returning to its origin tile, accounting for indirect paths,
temporal ordering, and flow asymmetries.

High roaming RTO indicates long-range circulation before return;
low roaming RTO indicates tight local mobility loops.

For formal definitions and theoretical interpretation, see:

    Network-Level Measures of Mobility from Aggregated Origin-Destination Data
    https://arxiv.org/abs/2502.04162


======================================================================
AUTHOR
======================================================================

Asif Shakeel  
ashakeel@ucsd.edu
"""


import os
import colorsys
from typing import Dict, List

# NOTE(dead-code): numpy imported but unused.
import pandas as pd
import plotly.graph_objects as go
from plotly.io import write_html

# Import your geohash ED module
import nm_effective_distance_pep_geohash as ed

BASE_DIR = "/Users/asif/Documents/nm24/outputs/nm_outputs"
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_OD_FILE = "od_mx_agg5_3h_MexicoCity_20190101T000000_20191231T000000.csv"



# ============================================================
# USER CONFIG
# ============================================================

# PEP generation span used for this run tag (just like effective_distance_pep.py)
PEP_SPAN_START = "2019-06-01 00:00:00"
PEP_SPAN_END   = "2019-06-15 00:00:00"

COUNTRY = "Mexico"
REGION_NAME = "Mexico City, Mexico"

ed.COUNTRY = COUNTRY     
ed.REGION_NAME = REGION_NAME

# Override the module so its tag helpers see the same span
ed.PEP_SPAN_START = PEP_SPAN_START
ed.PEP_SPAN_END   = PEP_SPAN_END

# Geohash / region config (must match those used when generating pep_od.csv)
ed.GEOHASH_PRECISION       = 5
ed.NEIGHBOR_TOPOLOGY       = "4"
ed.FEEDER_MODE             = "single"
ed.EDGE_DIRECTION_MODE     = "potential"
ed.INCLUDE_BACKGROUND_EDGES = False
ed.TIME_RES_MIN            = 30

START_COL = ed.START_COL      # "start_geohash"
END_COL   = ed.END_COL        # "end_geohash"
COUNT_COL = ed.COUNT_COL      # "trip_count"



TIME_RES_MIN = ed.TIME_RES_MIN
GEOHASH_PRECISION = ed.GEOHASH_PRECISION
NEIGHBOR_TOPOLOGY = ed.NEIGHBOR_TOPOLOGY
FEEDER_MODE       = ed.FEEDER_MODE
EDGE_DIRECTION_MODE = ed.EDGE_DIRECTION_MODE
INCLUDE_BACKGROUND_EDGES = ed.INCLUDE_BACKGROUND_EDGES

# Sweep dates
SWEEP_START_DATE = "2019-01-01"
SWEEP_END_DATE   = "2019-12-31"

# Daily fixed window [WINDOW_START, WINDOW_END)
WINDOW_START = "06:00:00"
WINDOW_END   = "21:00:00"

MIN_P_HIT = 1e-6

RTO_SUBDIR = "roaming_rto_fixed_window"

TITLE_SIZE      = 24
AXIS_TITLE_SIZE = 22
TICK_SIZE       = 20

from datetime import datetime, timedelta

display_end_date = (
    datetime.strptime(SWEEP_END_DATE, "%Y-%m-%d") - timedelta(days=1)
).strftime("%Y-%m-%d")
# ============================================================
# PATH HELPERS (MATCHING effective_distance_pep.py)
# ============================================================


OUTPUT_DIR = os.path.join(BASE_DIR, "effective_path_decomposition")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OD_INPUT_PATH = os.path.join(DATA_DIR, INPUT_OD_FILE)

# RTO-specific output directory
RTO_OUTPUT_DIR = os.path.join(BASE_DIR, "roaming_rto_fixed_window")
os.makedirs(RTO_OUTPUT_DIR, exist_ok=True)


# ============================================================
# UTILITIES
# ============================================================

def banner(x): print(f"\n=== {x} ===", flush=True)
def info(x):   print(f"[INFO] {x}", flush=True)

def build_color_map(keys: List[str]) -> Dict[str,str]:
    """Assign stable colors to origins."""
    cmap = {}
    for k, key in enumerate(keys):
        hue = (k * 137.713) % 360
        r,g,b = colorsys.hsv_to_rgb(hue/360., 0.72, 0.92)
        cmap[key] = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
    return cmap


# ============================================================
# MAIN
# ============================================================

def main():

    banner("RTO — GEOHASH")

    info(f"BASE_DIR: {BASE_DIR}")
    info(f"DATA_DIR: {DATA_DIR}")
    info(f"OD_INPUT_PATH: {OD_INPUT_PATH}")
    info(f"OUTPUT_DIR: {OUTPUT_DIR}")

    input_stem = os.path.splitext(os.path.basename(OD_INPUT_PATH))[0]


    if not os.path.exists(OD_INPUT_PATH):
        raise FileNotFoundError(OD_INPUT_PATH)

    # ----------------------------------------
    # Load full PEP OD (geohash)
    # ----------------------------------------
    df_all = ed.load_pep(OD_INPUT_PATH)
    df_all[START_COL] = df_all[START_COL].astype(str).str.lower()
    df_all[END_COL]   = df_all[END_COL].astype(str).str.lower()

    days = pd.date_range(SWEEP_START_DATE, SWEEP_END_DATE, freq="D").date
    rows = []   # all (origin, date, RTO) including CITY_AVG

    banner("DAILY RTO COMPUTATION")

    for di, day in enumerate(days, 1):
        print(f"\r[DAY {di}/{len(days)}] {day}", end="", flush=True)

        t0 = pd.to_datetime(f"{day} {WINDOW_START}")
        t1 = pd.to_datetime(f"{day} {WINDOW_END}")

        # Restrict to this day's window
        df_win0 = df_all[(df_all["_t0"] >= t0) & (df_all["_t0"] < t1)]
        if df_win0.empty:
            continue

        # LSCC on union graph
        lscc = ed.compute_lscc_nodes(df_win0)
        lscc = [x.lower() for x in lscc]
        if not lscc:
            continue

        df_win = df_win0[
            df_win0[START_COL].isin(lscc) &
            df_win0[END_COL].isin(lscc)
        ].copy()

        nodes = sorted(lscc)
        if not nodes:
            continue

        # Packs for ED recursion
        slices, node_index, packs = ed.precompute_slices(df_win, nodes)
        N = len(nodes)

        # Reachable origins (any outflow in window)
        pos_union = df_win.groupby(START_COL)[COUNT_COL].sum()
        reachable = [node_index[o] for o in pos_union.index if o in node_index]
        if not reachable:
            continue

        # ED recursion: X_elapsed(i,j), P_hit(i,j)
        X_elapsed, P_hit = ed.batched_elapsed_all_origins(packs, N, reachable)

        # ------------------------------------------------
        # (1) RTO(o,t) = X_elapsed(o,o)/P_hit(o,o)
        # ------------------------------------------------
        rto_dict = {}
        for o in nodes:
            i = node_index[o]
            x = X_elapsed[i, i]
            p = P_hit[i, i]
            if p >= MIN_P_HIT:
                rto_dict[o] = x 

        if not rto_dict:
            continue

        # ------------------------------------------------
        # (2) Starting probability π(o,t)
        # ------------------------------------------------
        outflow = df_win.groupby(START_COL)[COUNT_COL].sum()
        outflow = outflow.reindex(nodes).fillna(0.0)

        total_out = float(outflow.sum())
        pi = outflow / total_out if total_out > 0.0 else outflow * 0.0

        # ------------------------------------------------
        # (3) City-average RTO(t)
        # ------------------------------------------------
        city_rto = float(sum(pi[o] * rto_dict[o] for o in rto_dict))

        rows.append({
            "origin": "CITY_AVG",
            "date": pd.to_datetime(day),
            "RTO": city_rto
        })

        # ------------------------------------------------
        # (4) Origin-wise rows
        # ------------------------------------------------
        for o, val in rto_dict.items():
            rows.append({
                "origin": o,
                "date": pd.to_datetime(day),
                "RTO": val
            })

    print()

    if not rows:
        banner("NO RTO DATA — EXITING")
        return

    df_rto = pd.DataFrame(rows)

    # ============================================================
    # SPLIT + SAVE CSVs
    # ============================================================
    df_city = df_rto[df_rto["origin"] == "CITY_AVG"].copy()
    df_orig = df_rto[df_rto["origin"] != "CITY_AVG"].copy()

    csv_city = os.path.join(
        RTO_OUTPUT_DIR,
        f"{input_stem}_rto_cityavg_{SWEEP_START_DATE.replace('-','')}_"
        f"{SWEEP_END_DATE.replace('-','')}_"
        f"{WINDOW_START.replace(':','')}_"
        f"{WINDOW_END.replace(':','')}.csv"
    )
    csv_orig = os.path.join(
        RTO_OUTPUT_DIR,
        f"{input_stem}_rto_originwise_{SWEEP_START_DATE.replace('-','')}_"
        f"{SWEEP_END_DATE.replace('-','')}_"
        f"{WINDOW_START.replace(':','')}_"
        f"{WINDOW_END.replace(':','')}.csv"
    )

    df_city.to_csv(csv_city, index=False)
    df_orig.to_csv(csv_orig, index=False)

    info(f"Saved CITY-AVG CSV → {csv_city}")
    info(f"Saved ORIGIN-WISE CSV → {csv_orig}")

    # ============================================================
    # PLOTS
    # ============================================================

    # ---- City-average plot ----
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
                f"{REGION_NAME} – Roaming RTO — City Average<br>"
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
        RTO_OUTPUT_DIR,
        f"{input_stem}_rto_cityavg_{SWEEP_START_DATE.replace('-','')}_"
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
        title=(
            f"{REGION_NAME} Roaming RTO — Origin-wise<br>"
            f"{SWEEP_START_DATE} → {SWEEP_END_DATE}, "
            f"{WINDOW_START}–{WINDOW_END}"
        ),
        xaxis_title="Date",
        yaxis_title="RTO ()",
        template="plotly_white",
        hovermode="closest"
    )

    html_orig = os.path.join(
        RTO_OUTPUT_DIR,
        f"{input_stem}_rto_originwise_{SWEEP_START_DATE.replace('-','')}_"
        f"{SWEEP_END_DATE.replace('-','')}_"
        f"{WINDOW_START.replace(':','')}_"
        f"{WINDOW_END.replace(':','')}.html"
    )
    write_html(fig_orig, html_orig, include_plotlyjs="cdn")
    info(f"Saved ORIGIN-WISE plot → {html_orig}")

    banner("DONE")


if __name__ == "__main__":
    main()
