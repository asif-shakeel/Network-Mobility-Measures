"""
time_sweep_effective_distance_geohash.py
=======================================

Time-Sweep Effective Distance from Aggregated NetMob OD Data
------------------------------------------------------------

This script performs a **daily time sweep of network-level effective
distances** computed from *aggregated origin–destination (OD) flows*
from the NetMob 2024 Data Challenge.

It applies the time-dependent Markov-chain framework introduced in:

    Network-Level Measures of Mobility from Aggregated Origin-Destination Data
    https://arxiv.org/abs/2502.04162

The data are spatially discretized on a **geohash tiling** and treated
as a time-ordered sequence of column-stochastic transition operators.

For each day in the sweep, the script:
    1) extracts a fixed intraday time window,
    2) builds a time-dependent mobility operator,
    3) computes time-elapsed distances D_elapsed(i,j),
    4) normalizes by the staged baseline D_direct(i,j),
    5) selects OD pairs in the top D_eff percentile band,
    6) tracks their persistence across days.

The result is a **temporal fingerprint of structural mobility barriers**
that repeatedly emerge in the extreme tail of the effective-distance
distribution.


----------------------------------------------------------------------
INPUT
----------------------------------------------------------------------

Aggregated OD CSV (NetMob format):

    • start_geohash   : origin cell
    • end_geohash     : destination cell
    • date            : YYYYMMDD
    • time            : HH:MM:SS (bin start)
    • trip_count      : number of trips
    • median_length_m : median path length (meters)

Example:
    od_mx_agg5_3h_MexicoCity_20190101T000000_20191231T000000.csv


----------------------------------------------------------------------
OUTPUT
----------------------------------------------------------------------

1) Interactive HTML plot:

    time_sweep_geohash_<MODE>_q<PERCENTILE>_<DATE_RANGE>_<WINDOW>.html

Each point represents an OD pair (i,j) whose D_eff(i,j) lies in the
selected percentile band on that day.

Legend entries are ranked by **persistence** (number of days appearing).

2) (Implicit data product)

    The union of all percentile-band OD points across the sweep:
        (origin, destination, date, D_eff)


----------------------------------------------------------------------
DEFINITIONS
----------------------------------------------------------------------

Let P(b) be the column-stochastic mobility operator for time bin b:

    P[d, o] = Pr( X_{t+1} = d | X_t = o )

Let D(b) be the corresponding distance-weighted operator.

From the sequence {P(b), D(b)} over the window, the script computes:

    • D_elapsed(i,j) : expected path length to first hit i from j
    • D_direct(i,j) : staged empirical baseline (A–E in the paper)
    • D_eff(i,j)    : D_elapsed / D_direct

Only OD pairs with non-negligible hit probability are retained:

    P_hit(i,j) ≥ MIN_P_HIT


----------------------------------------------------------------------
SWEEP LOGIC
----------------------------------------------------------------------

For each day d in [SWEEP_START_DATE, SWEEP_END_DATE):

    • restrict OD data to WINDOW_START–WINDOW_END
    • compute D_eff(i,j) over that window
    • select OD pairs with:
            D_eff ≥ q-th percentile
            P_hit ≥ MIN_P_HIT
    • store (i,j,d,D_eff)

All stored points are then plotted as a time series, with colors
assigned by OD pair and legend entries ranked by frequency.


----------------------------------------------------------------------
AUTHOR
----------------------------------------------------------------------

Asif Shakeel  
email: ashakeel@ucsd.edu
"""

import os
import colorsys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import write_html

import nm_effective_distance_pep_geohash as gh
from datetime import datetime, timedelta


# ============================================================
# USER CONFIG
# ============================================================

PEP_SPAN_START = "2019-06-01 00:00:00"
PEP_SPAN_END   = "2019-06-15 00:00:00"

SWEEP_START_DATE = "2019-01-01"
SWEEP_END_DATE   = "2019-12-31"

WINDOW_START = "06:00:00"
WINDOW_END   = "12:00:00"

MIN_P_HIT    = 1e-6
PERCENTILE_Q = 99.0
MODE         = 'gup'   # {'gup','non-gup','all'}

SHOW_PROGRESS = True


BASE_DIR = "/Users/asif/Documents/nm24/outputs/nm_outputs"
DATA_DIR = os.path.join(BASE_DIR, "data")

INPUT_OD_FILE = "od_mx_agg5_3h_MexicoCity_20190101T000000_20191231T000000.csv"
OD_INPUT_PATH = os.path.join(DATA_DIR, INPUT_OD_FILE)

OUTPUT_DIR = os.path.join(BASE_DIR, "effective_distance_time_sweep")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REGION_NAME ="Mexico City, Mexico"
NEIGHBOR_TOPOLOGY = "4"
GEOHASH_PRECISION = 5
FEEDER_MODE = "single"
EDGE_DIRECTION_MODE = "potential"
INCLUDE_BACKGROUND_EDGES = False
TIME_RES_MIN = 30

TITLE_SIZE      = 24
AXIS_TITLE_SIZE = 22
TICK_SIZE       = 20
LEGEND_TITLE_SIZE = 20
LEGEND_TEXT_SIZE  = 18

DISPLAY_END_DATE = (
    datetime.strptime(SWEEP_END_DATE, "%Y-%m-%d") - timedelta(days=1)
).strftime("%Y-%m-%d")

# ============================================================
# UTIL
# ============================================================

def build_color_map(od_pairs: List[Tuple[str,str]]) -> Dict[Tuple[str,str], str]:
    cmap = {}
    for k, od in enumerate(od_pairs):
        hue = (k * 137.508) % 360
        r, g, b = colorsys.hsv_to_rgb(hue / 360., 0.78, 0.92)
        cmap[od] = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
    return cmap


def banner(x): print(f"\n=== {x} ===", flush=True)
def info(x):   print(f"[INFO] {x}", flush=True)
def warn(x):   print(f"[WARN] {x}", flush=True)


# ============================================================
# MAIN
# ============================================================

def main():

    banner("TIME-SWEEP EFFECTIVE DISTANCE — GEOHASH")

    info(f"OD_INPUT_PATH: {OD_INPUT_PATH}")

    # ------------------------------------------------------------
    # Load OD
    # ------------------------------------------------------------
    df_all = gh.load_pep(OD_INPUT_PATH)

    df_all[gh.START_COL] = df_all[gh.START_COL].astype(str).str.lower()
    df_all[gh.END_COL]   = df_all[gh.END_COL].astype(str).str.lower()

    alpha, beta, rmse = gh.fit_region_regression(df_all)

    # ------------------------------------------------------------
    # Day sweep
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

        lscc = gh.compute_lscc_nodes(df_win0)
        lscc = [x.lower() for x in lscc]
        if not lscc:
            continue

        df_win = df_win0[
            df_win0[gh.START_COL].isin(lscc) &
            df_win0[gh.END_COL].isin(lscc)
        ].copy()
        if df_win.empty:
            continue

        nodes = sorted(lscc)

        slices, node_index, packs = gh.precompute_slices(df_win, nodes)
        N = len(nodes)
        if N == 0:
            continue

        pos_union = df_win.groupby(gh.START_COL)[gh.COUNT_COL].sum()
        reachable = [node_index[o] for o in pos_union.index if o in node_index]
        if not reachable:
            continue

        gup_pairs = gh.detect_gup_pairs_not_in_D(df_all, nodes)

        X_elapsed, P_hit = gh.batched_elapsed_all_origins(
            packs, N, reachable
        )

        all_od = [(o, d) for o in nodes for d in nodes if o != d]
        od_df = pd.DataFrame(all_od, columns=[gh.START_COL, gh.END_COL])

        i_idx = od_df[gh.END_COL].map(node_index).to_numpy(int)
        j_idx = od_df[gh.START_COL].map(node_index).to_numpy(int)

        od_df["D_elapsed"] = X_elapsed[i_idx, j_idx]
        od_df["P_hit"]     = P_hit[i_idx, j_idx]

        cand = od_df[[gh.START_COL, gh.END_COL, "D_elapsed", "P_hit"]].copy()

        eff_df = gh.staged_baseline_vectorized(
            df_all=df_all,
            cand_df=cand,
            slices=slices,
            reg_params=(alpha, beta, rmse),
            add_rmse_sigma=gh.ADD_RMSE_SIGMA,
            date0=slices[0],
        )

        od_df = eff_df.replace([np.inf, -np.inf], np.nan)
        od_df = od_df.dropna(subset=["D_eff"])
        od_df = od_df[od_df["P_hit"] >= MIN_P_HIT]

        if od_df.empty:
            continue

        if MODE == "gup":
            od_df = od_df[
                od_df.apply(lambda r: (r[gh.START_COL], r[gh.END_COL]) in gup_pairs, axis=1)
            ]
        elif MODE == "non-gup":
            od_df = od_df[
                od_df.apply(lambda r: (r[gh.START_COL], r[gh.END_COL]) not in gup_pairs, axis=1)
            ]

        if od_df.empty:
            continue

        thr = float(np.nanpercentile(od_df["D_eff"], PERCENTILE_Q))
        band = od_df[od_df["D_eff"] >= thr].copy()
        if band.empty:
            continue

        band["_date"] = pd.to_datetime(day)

        rows.append(
            band[[gh.START_COL, gh.END_COL, "D_eff", "_date"]]
        )

    print()

    if not rows:
        warn("No sweep data — exiting.")
        return

    # ------------------------------------------------------------
    # Assemble sweep results
    # ------------------------------------------------------------
    df_points = pd.concat(rows, ignore_index=True)
    df_points.rename(columns={
        gh.START_COL: "origin",
        gh.END_COL:   "dest",
        "_date":      "date"
    }, inplace=True)

    # ------------------------------------------------------------
    # Frequency of occurrence per OD pair
    # ------------------------------------------------------------
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
                f"{REGION_NAME} – Effective Distance — Time Sweep ({MODE.upper()})<br>"
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


    fname = (
        f"{os.path.splitext(os.path.basename(OD_INPUT_PATH))[0]}_"
        f"time_sweep_geohash_{MODE}_q{PERCENTILE_Q}_"
        f"{SWEEP_START_DATE.replace('-','')}_"
        f"{SWEEP_END_DATE.replace('-','')}_"
        f"{WINDOW_START.replace(':','')}_"
        f"{WINDOW_END.replace(':','')}.html"
    )

    out_html = os.path.join(OUTPUT_DIR, fname)
    # write_html(fig, out_html, include_plotlyjs="cdn")
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
