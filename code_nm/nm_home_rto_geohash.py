#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
home_rto_geohash.py
==================

Home-RTO from Aggregated Geohash OD Data
---------------------------------------

This script computes **Home-RTO** for a single city / region using
time-resolved **aggregated origin–destination (OD) flows** on a
**geohash tiling** (NetMob 2024 format).

Home-RTO measures the *effective spatial radius of staying local*:
it is the characteristic distance traveled by trips that begin and
end in the **same spatial tile** (self-loops).


======================================================================
DEFINITION
======================================================================

For each origin geohash o and time bin t, define

    HomeRTO(o,t) = D_self(o,t)

where:

    D_self(o,t) = median distance of OD entries
                  with start_geohash = end_geohash = o
                  in bin t.

Let the starting probability be

    π(o,t) = outflow(o,t) / Σ_o outflow(o,t)

where outflow(o,t) is the total number of trips starting in o at time t.

Then the **city-average Home-RTO** at time t is

    CITY_AVG(t) = Σ_o π(o,t) · HomeRTO(o,t)

and the **window-average** city Home-RTO is

    ⟨CITY_AVG⟩ = mean_t CITY_AVG(t).


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

    <BASE_DIR>/home_rto/

Let <stem> be the input filename without extension.

----------------------------------------------------------------------
1) Origin-wise Home-RTO (window-averaged per day)
----------------------------------------------------------------------

    <stem>_home_rto_originwise_<dates>_<window>.csv

Columns:
    origin, date, HomeRTO


----------------------------------------------------------------------
2) City-average Home-RTO (window-averaged per day)
----------------------------------------------------------------------

    <stem>_home_rto_cityavg_<dates>_<window>.csv

Columns:
    date, HomeRTO


----------------------------------------------------------------------
3) Interactive HTML plots
----------------------------------------------------------------------

    <stem>_home_rto_cityavg_*.html
        • CITY_AVG(t) time series

    <stem>_home_rto_originwise_*.html
        • HomeRTO(o,t) for all origins


======================================================================
INTERPRETATION
======================================================================

Home-RTO quantifies how “local” mobility is at the tile level.
Large Home-RTO indicates that even trips classified as “staying local”
in OD space still involve long physical distances, reflecting
spatial spread or coarse tiling effects.

For interpretation in the network context, see:

    Network-Level Measures of Mobility from Aggregated Origin-Destination Data
    https://arxiv.org/abs/2502.04162


======================================================================
AUTHOR
======================================================================

Asif Shakeel  
ashakeel@ucsd.edu
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import write_html


BASE_DIR = "/Users/asif/Documents/nm24/outputs/nm_outputs"
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_OD_FILE = "od_mx_agg5_3h_MexicoCity_20190101T000000_20191231T000000.csv"

# ============================================================
# USER CONFIG
# ============================================================



COUNTRY = "Mexico"
REGION_NAME = "Mexico City, Mexico"
NEIGHBOR_TOPOLOGY   = "4"
FEEDER_MODE         = "single"
EDGE_DIRECTION_MODE = "potential"
GEOHASH_PRECISION = 5
INCLUDE_BACKGROUND_EDGES = False

# Geohash parameters
START_COL = "start_geohash"
END_COL   = "end_geohash"
DATE_COL  = "date"
TIME_COL  = "time"
COUNT_COL = "trip_count"
DIST_COL  = "median_length_m"

TIME_RES_MIN = 30

# Time sweep
WINDOW_START = "06:00:00"
WINDOW_END   = "21:00:00"

PEP_SPAN_START = "2019-06-01 00:00:00"
PEP_SPAN_END   = "2019-06-30 00:00:00"

SWEEP_START_DATE = "2019-01-01"
SWEEP_END_DATE   = "2019-12-31"

OUTPUT_SUBDIR = "home_rto"

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


def load_pep(path):
    """Load PEP OD and create _t0 timestamps for binning."""
    banner("LOAD PEP OD CSV")
    df = pd.read_csv(path)

    # Convert YYYYMMDD int column to string
    date_str = df[DATE_COL].astype(str)

    # TIME_COL is already HH:MM:SS
    time_str = df[TIME_COL].astype(str)

    # Build datetime
    dt = pd.to_datetime(date_str + " " + time_str, format="%Y%m%d %H:%M:%S")

    df["_t0"] = dt
    df["_t1"] = dt + pd.to_timedelta(TIME_RES_MIN, unit="min")
    df["_date"] = dt.dt.date
    df["_dow"] = dt.dt.dayofweek

    # Normalize geohashes
    df[START_COL] = df[START_COL].astype(str).str.lower()
    df[END_COL]   = df[END_COL].astype(str).str.lower()

    info(f"rows = {len(df):,}")
    info(f"time span = {df['_t0'].min()} → {df['_t0'].max()}")
    return df



# ============================================================
# MAIN
# ============================================================

def main():

    banner("HOME RTO — GEOHASH")

    # Reconstruct RUN_NAME path same way geohash ED scripts do:



    DATA_DIR   = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(BASE_DIR, OUTPUT_SUBDIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    OD_PATH = os.path.join(DATA_DIR, INPUT_OD_FILE)  # or od_mx_agg*.csv

    info(f"OD_PATH = {OD_PATH}")

    df_all = load_pep(OD_PATH)

    # Days
    days = pd.date_range(SWEEP_START_DATE, SWEEP_END_DATE, freq="D").date

    rows_originwise = []
    rows_cityavg = []

    banner("COMPUTING HOME RTO (BIN-BY-BIN)")

    input_stem = os.path.splitext(os.path.basename(OD_PATH))[0]
    
    for di, day in enumerate(days, 1):
        print(f"\r[DAY {di}/{len(days)}] {day}", end='', flush=True)

        # Window selection
        t0 = pd.to_datetime(f"{day} {WINDOW_START}")
        t1 = pd.to_datetime(f"{day} {WINDOW_END}")

        df_win = df_all[(df_all["_t0"] >= t0) & (df_all["_t0"] < t1)].copy()
        if df_win.empty:
            continue

        # Group by bin time + origin tile
        gb = df_win.groupby(["_t0", START_COL])

        # Compute:
        #   self-distance D_self(o,t)
        #   starting probability π(o,t) = outflow(o,t) / total_outflow(t)
        # ------------------------------------------------------------
        # total outflow per bin across ALL origins
        bin_outflow = (
            df_win
            .groupby(["_t0", START_COL])[COUNT_COL]
            .sum()
            .rename("total_outflow")
            .reset_index()
        )


        bin_records = []  # for city average

        for (bin_t, o), sub in gb:

            # Outflow count for π(o,t)
            outflow = bin_outflow.loc[
                (bin_outflow["_t0"] == bin_t) &
                (bin_outflow[START_COL] == o),
                "total_outflow"
            ].sum()


            # Self-loop subset
            self_sub = sub[sub[END_COL] == o]
            if self_sub.empty:
                continue

            # Median D[o,o] for this bin
            d_self = self_sub[DIST_COL].median()

            if not np.isfinite(d_self):
                continue

            bin_records.append({
                "origin": o,
                "bin": bin_t,
                "D_self": float(d_self),
                "outflow": float(outflow)
            })

        if not bin_records:
            continue

        df_bins = pd.DataFrame(bin_records)

        # Compute bin-level starting probs π(o,t)
        # ---------------------------------------
        for bin_t, sub in df_bins.groupby("bin"):

            total_out = sub["outflow"].sum()
            if total_out <= 0:
                continue

            sub = sub.copy()
            sub["pi"] = sub["outflow"] / total_out

            # City-average Home RTO for this bin
            city_val = float((sub["pi"] * sub["D_self"]).sum())

            rows_cityavg.append({
                "date": pd.to_datetime(day),
                "bin": bin_t,
                "HomeRTO": city_val
            })

            # Also store origin-wise bin values
            for _, r in sub.iterrows():
                rows_originwise.append({
                    "date": pd.to_datetime(day),
                    "bin": bin_t,
                    "origin": r["origin"],
                    "HomeRTO": r["D_self"]
                })

    print()

    if not rows_originwise:
        banner("NO HOME-RTO DATA — EXITING")
        return

    # Aggregate to window-level averages
    df_orig = pd.DataFrame(rows_originwise)
    df_city = pd.DataFrame(rows_cityavg)

    # Window-averaged HomeRTO per origin
    df_orig_daily = (
        df_orig.groupby(["origin", "date"], as_index=False)["HomeRTO"].mean()
    )

    # Window-averaged city average
    df_city_daily = (
        df_city.groupby(["date"], as_index=False)["HomeRTO"].mean()
    )

    # ============================================================
    # SAVE CSVs
    # ============================================================

    csv_orig = os.path.join(
        OUTPUT_DIR,
        f"{input_stem}_home_rto_originwise_{SWEEP_START_DATE.replace('-','')}_"
        f"{SWEEP_END_DATE.replace('-','')}_"
        f"{WINDOW_START.replace(':','')}_"
        f"{WINDOW_END.replace(':','')}.csv"
    )
    csv_city = os.path.join(
        OUTPUT_DIR,
        f"{input_stem}_home_rto_cityavg_{SWEEP_START_DATE.replace('-','')}_"
        f"{SWEEP_END_DATE.replace('-','')}_"
        f"{WINDOW_START.replace(':','')}_"
        f"{WINDOW_END.replace(':','')}.csv"
    )

    df_orig_daily.to_csv(csv_orig, index=False)
    df_city_daily.to_csv(csv_city, index=False)

    info(f"Saved ORIGIN-WISE CSV → {csv_orig}")
    info(f"Saved CITY-AVG CSV → {csv_city}")

    # ============================================================
    # PLOTS
    # ============================================================

    # City-average plot
    fig_city = go.Figure()
    fig_city.add_trace(go.Scattergl(
        x=df_city_daily["date"],
        y=df_city_daily["HomeRTO"],
        mode="lines+markers",
        marker=dict(size=7, color="black"),
        line=dict(width=2, color="black"),
        name="City Average"
    ))
    fig_city.update_layout(
        title=dict(
            text=(
                f"{REGION_NAME} – Home RTO — City Average<br>"
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


    html_city = csv_city.replace(".csv", ".html")
    write_html(fig_city, html_city, include_plotlyjs="cdn")
    info(f"Saved CITY-AVG plot → {html_city}")

    # Origin-wise plot
    fig_orig = go.Figure()
    for o, sub in df_orig_daily.groupby("origin"):
        fig_orig.add_trace(go.Scattergl(
            x=sub["date"],
            y=sub["HomeRTO"],
            mode="lines+markers",
            name=o,
            line=dict(width=1),
            marker=dict(size=4)
        ))
    fig_orig.update_layout(
        title="Home RTO — Origin-wise",
        xaxis_title="Date",
        yaxis_title="Home RTO (m)",
        template="plotly_white"
    )

    html_orig = csv_orig.replace(".csv", ".html")
    write_html(fig_orig, html_orig, include_plotlyjs="cdn")
    info(f"Saved ORIGIN-WISE plot → {html_orig}")

    banner("DONE")


if __name__ == "__main__":
    main()
