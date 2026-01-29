#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================

BASE_DIR = Path("/Users/asif/Documents/nm24")
DATA_DIR = BASE_DIR / "outputs" / "nm_outputs" / "data"
OUT_DIR  = BASE_DIR / "outputs" / "nm_outputs" / "effective_distance_regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = DATA_DIR / "od_mx_agg5_3h_MexicoCity_20190101T000000_20191231T000000.csv"

OUTPUT_PDF = OUT_DIR / "mexico_city_distance_regression_override.pdf"
OUTPUT_PNG = OUT_DIR / "mexico_city_distance_regression_override.png"

START_COL = "start_geohash"
END_COL   = "end_geohash"
COUNT_COL = "trip_count"
DIST_COL  = "median_length_m"

# Fixed regression coefficients (override)
ALPHA = 441.834
BETA  = 0.935048
SIGMA = 617.829   # shown only for reference

# Axis trimming (visualization only)
X_MAX_QUANTILE = 99.5
Y_MAX_QUANTILE = 99.5

# =========================
# GEO HELPERS
# =========================

_base32 = "0123456789bcdefghjkmnpqrstuvwxyz"

def geohash_to_latlon(gh):
    even = True
    lat = [-90.0, 90.0]
    lon = [-180.0, 180.0]
    for c in gh.lower():
        cd = _base32.find(c)
        if cd == -1:
            break
        for mask in [16, 8, 4, 2, 1]:
            if even:
                mid = (lon[0] + lon[1]) / 2.0
                lon[0] = mid if (cd & mask) else lon[0]
                lon[1] = lon[1] if (cd & mask) else mid
            else:
                mid = (lat[0] + lat[1]) / 2.0
                lat[0] = mid if (cd & mask) else lat[0]
                lat[1] = lat[1] if (cd & mask) else mid
            even = not even
    return ((lat[0] + lat[1]) / 2.0, (lon[0] + lon[1]) / 2.0)

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*(math.sin(dlmb/2)**2)
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

# =========================
# MAIN
# =========================

def main():
    print("Loading OD data…")
    df = pd.read_csv(INPUT_CSV)

    print("Aggregating OD pairs…")
    g = (
        df.groupby([START_COL, END_COL])
          .agg(
              y=(DIST_COL, "median"),
              w=(COUNT_COL, "sum")
          )
          .reset_index()
    )

    print("Computing geodesic distances…")
    d_gc = []
    for o, d in zip(g[START_COL], g[END_COL]):
        lat_o, lon_o = geohash_to_latlon(o)
        lat_d, lon_d = geohash_to_latlon(d)
        d_gc.append(haversine_m(lat_o, lon_o, lat_d, lon_d))
    g["d_gc"] = np.array(d_gc)

    g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["d_gc", "y", "w"])
    print(f"Plotting {len(g):,} OD pairs")

    # -------------------------
    # Axis trimming (visual only)
    # -------------------------
    x = g["d_gc"].to_numpy(float)
    y = g["y"].to_numpy(float)
    w = g["w"].to_numpy(float)

    x_max = np.percentile(x, X_MAX_QUANTILE)
    y_max = np.percentile(y, Y_MAX_QUANTILE)

    mask = (x <= x_max) & (y <= y_max)
    x, y, w = x[mask], y[mask], w[mask]

    # =========================
    # PLOT
    # =========================

    plt.figure(figsize=(7, 6))

    sizes = 10 + 40 * (w / w.max())
    plt.scatter(
        x,
        y,
        s=sizes,
        alpha=0.45,
        edgecolors="none"
    )

    xx = np.linspace(0, x_max, 400)
    yy = ALPHA + BETA * xx

    plt.plot(
        xx,
        yy,
        color="black",
        lw=2,
        label=(
            r"$y = \alpha + \beta d_{\mathrm{gc}}$" "\n"
            rf"$\alpha={ALPHA:.1f},\;\beta={BETA:.3f}$\n"
            f"$\\sigma={SIGMA:.1f}$"
        )
    )

    plt.xlim(0, x_max)
    plt.ylim(0, y_max)

    plt.xlabel("Geographical distance $d_{gc}$ (m)")
    plt.ylabel("Observed median trip distance (m)")
    plt.title("Mexico City — OD distance baseline (override)")
    plt.legend(frameon=True)
    plt.grid(alpha=0.3)

    # plt.text(
    #     0.02, 0.95,
    #     f"Axes trimmed at {X_MAX_QUANTILE:.1f}th percentile",
    #     transform=plt.gca().transAxes,
    #     fontsize=9,
    #     va="top"
    # )

    plt.tight_layout()
    plt.savefig(OUTPUT_PDF)
    plt.savefig(OUTPUT_PNG, dpi=150)
    plt.close()

    print("Saved:")
    print(f"  {OUTPUT_PDF}")
    print(f"  {OUTPUT_PNG}")

if __name__ == "__main__":
    main()
