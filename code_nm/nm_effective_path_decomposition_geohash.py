#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nm_effective_path_decomposition_geohash.py
==========================================

Effective Path Decomposition from Aggregated Geohash OD Data
------------------------------------------------------------

This script performs **effective path decomposition** for a selected
origin–destination (OD) pair using time-resolved **aggregated OD flows**
(NetMob 2024 format) on a **geohash tiling**.

It reconstructs the *first-hitting path ensemble* that gives rise to the
time-elapsed effective distance between two tiles, using the same
time-recursive Markov framework as:

    Network-Level Measures of Mobility from Aggregated Origin-Destination Data
    https://arxiv.org/abs/2502.04162


======================================================================
CONCEPT
======================================================================

Given a sequence of column-stochastic mobility operators:

    x_{t+1} = M(t) · x_t

and distance operators D(t), this script computes:

    • X_elapsed(i,j): time-elapsed distance of first hitting j from i
    • P_hit(i,j): total probability of first hitting j from i

For a chosen OD pair (o,d), it:

    1) computes D_elapsed(o,d) = X_elapsed(d,o)
    2) estimates a direct baseline distance:
           D_direct(o,d) = α + β · d_gc(o,d)
    3) defines effective distance:
           D_eff(o,d) = D_elapsed / D_direct
    4) enumerates all first-hitting paths contributing to D_elapsed(o,d)
    5) visualizes the OD edge and all contributing paths on an interactive map


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
    TIME_RES_MIN = 180 minutes

Analysis window:
    SPAN_START → SPAN_END

OD pairs to decompose:
    OD_PAIRS = [(origin, destination), ...]


======================================================================
PIPELINE
======================================================================

1) Load aggregated OD flows
2) Restrict to [SPAN_START, SPAN_END)
3) Compute LSCC on union graph
4) Build per-bin transition + distance packs
5) Compute X_elapsed and P_hit via time recursion
6) Fit regression α + β·d_gc for baseline
7) Enumerate all first-hitting paths for each OD
8) Render an interactive Folium map


======================================================================
OUTPUT
======================================================================

One interactive HTML map per run:

    <BASE_DIR>/effective_path_decomposition/
        <input_stem>_path_decomposition_geohash__
        <SPAN_START>_to_<SPAN_END>.html

The map contains:

    • tile centroids
    • OD straight-line edge
    • all first-hitting paths (toggleable layers)
    • arrowed direction fields
    • per-path probability, distance, and step count
    • optional metro / hub overlays (if manifest enabled)


======================================================================
INTERPRETATION
======================================================================

Each drawn path represents a **first-hitting trajectory** contributing
to the effective distance between two tiles. The union of these paths
forms a probabilistic “effective corridor” whose weighted mean length
is D_elapsed(o,d).

For formal definitions and theory, see:

    Network-Level Measures of Mobility from Aggregated Origin-Destination Data
    https://arxiv.org/abs/2502.04162


======================================================================
AUTHOR
======================================================================

Asif Shakeel  
ashakeel@ucsd.edu
"""

import os
import math
from typing import List, Tuple, Dict, Any, Set
# NOTE(dead-code): Optional import is unused.

import numpy as np
import pandas as pd
import folium

# === manifest + metadata imports from ED engine ===
# from effective_distance_pep_geohash import (
#     load_corridor_manifest,
#     build_node_meta_from_manifest,
#     build_metro_pairs_from_manifest,
# )
from nm_effective_distance_pep_geohash import (
    build_node_meta_from_manifest,
    build_metro_pairs_from_manifest,
    fit_region_regression,
)

# ============================================================
# USER CONFIG
# ============================================================

COUNTRY      = "Mexico"
REGION_NAME = "Mexico City, Mexico"
NEIGHBOR_TOPOLOGY   = "4"
FEEDER_MODE         = "single"
EDGE_DIRECTION_MODE = "potential"
GEOHASH_PRECISION   = 5
INCLUDE_BACKGROUND_EDGES = False

BASE_DIR = "/Users/asif/Documents/nm24/outputs/nm_outputs"
DATA_DIR = os.path.join(BASE_DIR, "data")

# Where the path-decomposition outputs will go
OUTPUT_DIR = os.path.join(BASE_DIR, "effective_path_decomposition")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# OD aggregate file (same as cloned_effective_distance_pep_geohash)
INPUT_OD_FILE = "od_mx_agg5_3h_MexicoCity_20190101T000000_20191231T000000.csv"
OD_INPUT_PATH = os.path.join(DATA_DIR, INPUT_OD_FILE)

OUTPUTS_DIR = BASE_DIR            # /Users/asif/Documents/nm24/outputs
# RUNS_DIR is no longer needed for this script

START_COL = "start_geohash"
END_COL   = "end_geohash"
DATE_COL  = "date"
TIME_COL  = "time"
COUNT_COL = "trip_count"
DIST_COL  = "median_length_m"

TIME_RES_MIN = 180

SPAN_START = "2019-08-22 06:00:00"
SPAN_END   = "2019-08-22 12:00:00"


REGRESSION_ON_LSCC = True

PEP_SPAN_START = "2019-06-01 00:00:00"
PEP_SPAN_END   = "2019-06-15 00:00:00"

START_TS = pd.to_datetime(PEP_SPAN_START)
END_TS   = pd.to_datetime(PEP_SPAN_END)

OD_PAIRS = [
    ("9g3w6", "9g3w0"),   # example pair
]

FOLIUM_TILES = "cartodbpositron"

PATH_EDGE_WEIGHT = 2
OD_EDGE_WEIGHT   = 4

PATH_COLORS = [
    "#2FBF71", "#E4572E", "#4D8BFF", "#F1C40F", "#9B59B6", "#17A589",
    "#E67E22", "#2E86C1", "#C0392B", "#27AE60", "#7F8C8D", "#D35400",
]

ARROW_FRAC       = 0.10
ARROW_MIN_LEN_M  = 30
ARROW_MAX_LEN_M  = 5000
ARROW_TIP_ANGLE  = 22.0

SELF_LOOP_RADIUS_M = 120   # in meters


# ============================================================
# HELPERS
# ============================================================

def info(x: str) -> None:
    print(f"[INFO] {x}", flush=True)

def warn(x: str) -> None:
    print(f"[WARN] {x}", flush=True)


def _city_tag() -> str:
    return (REGION_NAME.split(',')[0]).replace(' ', '') or "Region"

def _top_tag() -> str:
    return f"gh-{NEIGHBOR_TOPOLOGY}-res-{GEOHASH_PRECISION}"

def _fdr_tag() -> str:
    return f"fdr-{str(FEEDER_MODE).lower()}"

def _edge_tag() -> str:
    if EDGE_DIRECTION_MODE == "potential":
        return "edge-pot"
    if EDGE_DIRECTION_MODE == "geometric":
        return "edge-geom"
    return f"edge-{EDGE_DIRECTION_MODE}"

def _overlay_tag() -> str:
    return "ovr-1" if INCLUDE_BACKGROUND_EDGES else "ovr-0"


city_tag    = _city_tag()
top_tag     = _top_tag()
fdr_tag     = _fdr_tag()
edge_tag    = _edge_tag()
overlay_tag = _overlay_tag()

# --------- IMPORTANT: tags must match ED engine run layout ----------
# Use T and NO space, NO seconds: 20190601T0000_20190615T0000
PEP_SPAN_START_TS = pd.to_datetime(PEP_SPAN_START)
PEP_SPAN_END_TS   = pd.to_datetime(PEP_SPAN_END)
start_tag = PEP_SPAN_START_TS.strftime("%Y%m%dT%H%M")
end_tag   = PEP_SPAN_END_TS.strftime("%Y%m%dT%H%M")

# RUN_NAME = (
#     f"{city_tag}/"
#     f"{top_tag}/"
#     f"{fdr_tag}/"
#     f"{edge_tag}/"
#     f"{overlay_tag}/"
#     f"{start_tag}_{end_tag}/"
#     f"m{TIME_RES_MIN}/x-periodic_fixed_point"
# )

# # ---------------------------
# # Manifest path (match ED engine naming)
# # ---------------------------
# CORRIDOR_MANIFEST_FILENAME = (
#     f"corridors_manifest_{city_tag}_mode-geohash_"
#     f"{top_tag}_{fdr_tag}_{edge_tag}_{overlay_tag}.json"
# )

# # manifest lives under OUTPUTS_DIR/corridors_outputs_geohash/...
# CORRIDOR_MANIFEST_PATH = os.path.join(
#     OUTPUTS_DIR,
#     "corridors_outputs_geohash",
#     CORRIDOR_MANIFEST_FILENAME
# )


# ----------------------------
# Geohash decode
# ----------------------------
_base32 = "0123456789bcdefghjkmnpqrstuvwxyz"

def geohash_to_latlon(gh: str) -> Tuple[float, float]:
    even = True
    lat = [-90.0, 90.0]
    lon = [-180.0, 180.0]
    for c in gh:
        cd = _base32.find(c)
        if cd == -1:
            break
        for mask in [16, 8, 4, 2, 1]:
            if even:
                mid = (lon[0] + lon[1]) / 2.0
                if cd & mask:
                    lon[0] = mid
                else:
                    lon[1] = mid
            else:
                mid = (lat[0] + lat[1]) / 2.0
                if cd & mask:
                    lat[0] = mid
                else:
                    lat[1] = mid
            even = not even
    return ((lat[0] + lat[1]) / 2.0, (lon[0] + lon[1]) / 2.0)


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = φ2 - φ1
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*(math.sin(dλ/2)**2)
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))


# ============================================================
# LOAD / WINDOW
# ============================================================

def load_pep(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[START_COL] = df[START_COL].astype(str).str.lower()
    df[END_COL]   = df[END_COL].astype(str).str.lower()

    dt = pd.to_datetime(df[DATE_COL].astype(str) + " " + df[TIME_COL].astype(str))
    df["_t0"]   = dt
    df["_t1"]   = dt + pd.to_timedelta(TIME_RES_MIN, "min")
    df["_dow"]  = dt.dt.dayofweek
    df["_date"] = dt.dt.date
    return df


def subset_window(df: pd.DataFrame) -> pd.DataFrame:
    s = pd.to_datetime(SPAN_START)
    e = pd.to_datetime(SPAN_END)
    w = df[(df["_t0"] >= s) & (df["_t0"] < e)].copy()
    info(f"Window rows = {len(w):,}")
    return w


# ============================================================
# LSCC
# ============================================================

import networkx as nx

def compute_lscc(df: pd.DataFrame) -> List[str]:
    G = nx.DiGraph()
    nodes = sorted(set(df[START_COL]).union(set(df[END_COL])))
    G.add_nodes_from(nodes)

    pos = df[df[COUNT_COL] > 0]
    grouped = pos.groupby([START_COL, END_COL])[COUNT_COL].sum().reset_index()

    for _, r in grouped.iterrows():
        if r[START_COL] != r[END_COL]:
            G.add_edge(r[START_COL], r[END_COL])

    if G.number_of_edges() == 0:
        return []

    sccs = list(nx.strongly_connected_components(G))
    sccs.sort(key=len, reverse=True)
    return sorted(sccs[0])


# ============================================================
# PACK BUILDER
# ============================================================

def build_packs(df_win: pd.DataFrame, nodes: List[str]):
    idx = {g: k for k, g in enumerate(nodes)}
    slices = sorted(df_win["_t0"].unique().tolist())

    packs = []
    for s in slices:
        g = df_win[df_win["_t0"] == s]
        agg = g.groupby([START_COL, END_COL]).agg(
            flow=(COUNT_COL, "sum"),
            dist=(DIST_COL, "median"),
        ).reset_index()

        i_list: List[int] = []
        j_list: List[int] = []
        f_list: List[float] = []
        d_list: List[float] = []

        for _, r in agg.iterrows():
            o = r[START_COL]
            d = r[END_COL]
            if o not in idx or d not in idx:
                continue
            j = idx[o]
            i = idx[d]
            if r["flow"] <= 0:
                continue

            i_list.append(i)
            j_list.append(j)
            f_list.append(r["flow"])
            d_list.append(r["dist"])

        packs.append({
            "i": np.asarray(i_list, int),
            "j": np.asarray(j_list, int),
            "c": np.asarray(f_list, float),
            "d": np.asarray(d_list, float),
        })

    return slices, idx, packs


# ============================================================
# ELAPSED DISTANCE (scalar)
# ============================================================

from scipy import sparse

def build_Mt_Dt(pack: Dict[str, Any], N: int):
    i = pack["i"]
    j = pack["j"]
    c = pack["c"]
    d = pack["d"]

    M = sparse.coo_matrix((c, (i, j)), shape=(N, N)).tocsr()
    colsum = np.array(M.sum(axis=0)).ravel()
    colsum[colsum == 0] = 1
    M = M @ sparse.diags(1 / colsum)

    D = sparse.coo_matrix((d, (i, j)), shape=(N, N)).tocsr()
    return M, D


def batched_elapsed_all_origins(
    packs: List[Dict[str, Any]], N: int, origin_indices: List[int]
):
    K = len(packs)

    Ms = []
    Ds = []
    for p in packs:
        M, D = build_Mt_Dt(p, N)
        Ms.append(M)
        Ds.append(D)

    Xbar = np.full((N, N), np.nan)
    P_hit = np.zeros((N, N))

    for j in origin_indices:

        for i in range(N):

            M1 = Ms[0]
            D1 = Ds[0]

            p = M1[:, j].toarray().ravel()
            p[j] = 0
            y = D1[:, j].toarray().ravel()

            num = 0.0
            den = 0.0

            pi1 = p[i]
            if pi1 > 0:
                num += y[i] * pi1
                den += pi1

            for t in range(1, K):
                Mt = Ms[t]
                Dt = Ds[t]

                p_tilde = p.copy()
                p_tilde[i] = 0
                p_tilde[j] = 0

                p_next = Mt @ p_tilde

                MD = (Mt.multiply(Dt)) @ p_tilde
                MY = Mt @ (p_tilde * y)

                y_next = np.zeros(N)
                mask = p_next > 0
                y_next[mask] = (MD[mask] + MY[mask]) / p_next[mask]

                pi_t = p_next[i]
                if pi_t > 0:
                    num += y_next[i] * pi_t
                    den += pi_t

                p = p_next
                y = y_next

            if den > 0:
                Xbar[i, j] = num / den
                P_hit[i, j] = den

    return Xbar, P_hit


# ============================================================
# REGRESSION α + β·d_gc
# ============================================================

def fit_regression(df_all: pd.DataFrame, nodes: List[str]) -> Tuple[float, float]:
    df = df_all[
        df_all[START_COL].isin(nodes)
        & df_all[END_COL].isin(nodes)
        & (df_all[START_COL] != df_all[END_COL])
    ]

    if df.empty:
        warn("No regression data.")
        return (1.0, 1.0)

    rows = []
    for (o, d), grp in df.groupby([START_COL, END_COL]):
        lat_o, lon_o = geohash_to_latlon(o)
        lat_d, lon_d = geohash_to_latlon(d)
        dgc = haversine_m(lat_o, lon_o, lat_d, lon_d)
        dist = grp[DIST_COL].median()
        rows.append((dgc, dist))

    arr = np.array(rows)
    X = arr[:, 0]
    y = arr[:, 1]

    A = np.vstack([np.ones_like(X), X]).T
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

    alpha, beta = float(coef[0]), float(coef[1])
    info(f"Regression fit: α={alpha:.3f}, β={beta:.6f}")
    return alpha, beta


def predict_direct(o: str, d: str, alpha: float, beta: float) -> float:
    lat_o, lon_o = geohash_to_latlon(o)
    lat_d, lon_d = geohash_to_latlon(d)
    dgc = haversine_m(lat_o, lon_o, lat_d, lon_d)
    return alpha + beta * dgc


# ============================================================
# FIRST-HIT PATH ENUMERATION
# ============================================================

def enumerate_first_hitting_paths(
    od: Tuple[str, str],
    packs: List[Dict[str, Any]],
    nodes: List[str],
    node_index: Dict[str, int],
):
    o, d = od
    if o not in node_index or d not in node_index:
        warn(f"OD {o}->{d} not in LSCC.")
        return {"paths": [], "paths_geo": [], "P_sum": 0.0}

    j = node_index[o]
    i = node_index[d]

    Ms = []
    Ds = []
    N = len(nodes)
    for p in packs:
        M, D = build_Mt_Dt(p, N)
        Ms.append(M)
        Ds.append(D)

    frontier = []
    final_paths = []
    P_sum = 0.0
    K = len(Ms)

    M1 = Ms[0]
    D1 = Ds[0]
    col = M1[:, j].toarray().ravel()

    for u in range(N):
        if col[u] > 0 and u != j:
            prob = float(col[u])
            dist = float(D1[u, j])
            if u == i:
                final_paths.append({
                    "path": [j, i],
                    "prob": prob,
                    "dist": dist,
                    "t": 1,
                })
                P_sum += prob
            else:
                frontier.append({
                    "node": u,
                    "path": [j, u],
                    "prob": prob,
                    "dist": dist,
                })

    for t in range(2, K + 1):
        Mt = Ms[t - 1]
        Dt = Ds[t - 1]
        new_frontier = []

        for entry in frontier:
            u = entry["node"]
            col_u = Mt[:, u].toarray().ravel()

            for v in range(N):
                if col_u[v] <= 0:
                    continue
                if v == j and i != j:
                    continue

                prob = entry["prob"] * float(col_u[v])
                dist = entry["dist"] + float(Dt[v, u])

                if v == i:
                    final_paths.append({
                        "path": entry["path"] + [i],
                        "prob": prob,
                        "dist": dist,
                        "t": t,
                    })
                    P_sum += prob
                else:
                    new_frontier.append({
                        "node": v,
                        "path": entry["path"] + [v],
                        "prob": prob,
                        "dist": dist,
                    })

        frontier = new_frontier
        if not frontier:
            break

    paths_geo = []
    for p in final_paths:
        nodes_geo = [nodes[k] for k in p["path"]]
        q = p.copy()
        q["path_geo"] = nodes_geo
        paths_geo.append(q)

    return {"paths": final_paths, "paths_geo": paths_geo, "P_sum": P_sum}


# ============================================================
# MAP HELPERS
# ============================================================

def _merc_xy(lat, lon):
    R = 6378137.0
    x = R * math.radians(lon)
    lat_c = max(min(lat, 85.05112878), -85.05112878)
    y = R * math.log(math.tan(math.pi/4 + math.radians(lat_c)/2))
    return x, y

def _merc_inv(x, y):
    R = 6378137.0
    lat = math.degrees(math.atan(math.sinh(y / R)))
    lon = math.degrees(x / R)
    return lat, lon

def arrow_triangle(lat1, lon1, lat2, lon2, scale=1.0):
    x1, y1 = _merc_xy(lat1, lon1)
    x2, y2 = _merc_xy(lat2, lon2)
    vx, vy = x2 - x1, y2 - y1
    seg = math.hypot(vx, vy)
    if seg < 1e-9:
        return None

    L = seg * ARROW_FRAC
    L = max(ARROW_MIN_LEN_M, min(ARROW_MAX_LEN_M, L))
    L *= scale

    ux, uy = vx / seg, vy / seg

    bx, by = x2 - L * ux, y2 - L * uy
    px, py = -uy, ux
    half_base = L * math.tan(math.radians(ARROW_TIP_ANGLE) / 2)

    leftx, lefty   = bx + half_base * px, by + half_base * py
    rightx, righty = bx - half_base * px, by - half_base * py

    return [
        _merc_inv(x2, y2),
        _merc_inv(leftx, lefty),
        _merc_inv(rightx, righty),
    ]


def draw_self_loop(lat, lon, radius_m, color, fg):
    folium.Circle(
        location=(lat, lon),
        radius=radius_m,
        color=color,
        fill=False,
        weight=2,
        opacity=0.9,
    ).add_to(fg)


# ============================================================
# MAP BUILDER
# ============================================================

def build_map(ctx: Dict[str, Any], od_pairs: List[Tuple[str, str]], out_html: str):

    nodes      = ctx["nodes"]
    node_index = ctx["node_index"]
    node_meta  = ctx["node_meta"]
    packs      = ctx["packs"]
    X_elapsed  = ctx["X_elapsed"]
    alpha      = ctx["alpha"]
    beta       = ctx["beta"]
    metro_pairs = ctx["metro_pairs"]

    lats = [node_meta[n]["lat"] for n in nodes]
    lons = [node_meta[n]["lon"] for n in nodes]
    mlat, mlon = float(np.mean(lats)), float(np.mean(lons))

    m = folium.Map(location=(mlat, mlon), zoom_start=11, tiles=FOLIUM_TILES)


    # ------------------------------------------------------------
    # Small unobtrusive map title (same style as H3 version)
    # ------------------------------------------------------------
    title = (
        f"Effective Path Decomposition — {REGION_NAME}<br>"
        f"OD: {od_pairs[0][0]} → {od_pairs[0][1]} · "
        f"{SPAN_START} → {SPAN_END}<br>"
        f"Geohash res={GEOHASH_PRECISION} · m{TIME_RES_MIN}"
    )

    m.get_root().html.add_child(
        folium.Element(
            f"""
            <div style="
                position: fixed;
                top: 8px;
                left: 50%;
                transform: translateX(-50%);
                font-size: 13px;
                color: #333;
                background-color: rgba(255,255,255,0.85);
                padding: 4px 8px;
                border-radius: 4px;
                z-index: 9999;
                box-shadow: 0 0 3px rgba(0,0,0,0.15);
            ">
            {title}
            </div>
            """
        )
    )



    # ================================
    # DRAW TILES, HUBS, AND CENTERS
    # ================================
    g_tiles   = folium.FeatureGroup("Tiles", show=True)
    g_hubs    = folium.FeatureGroup("Hubs", show=False)
    g_centers = folium.FeatureGroup("Centers", show=False)

    for h in nodes:
        meta = node_meta[h]
        lat, lon = meta["lat"], meta["lon"]

        if meta.get("is_hub", False):
            folium.CircleMarker(
                location=(lat, lon),
                radius=6,
                color="#cc0000",
                fill=True,
                fill_color="#ff4444",
                tooltip=f"Hub {h}",
            ).add_to(g_hubs)

        elif meta.get("is_center", False):
            folium.CircleMarker(
                location=(lat, lon),
                radius=4,
                color="#0066cc",
                fill=True,
                fill_color="#66aaff",
                tooltip=f"Center {h}",
            ).add_to(g_centers)

        else:
            folium.CircleMarker(
                location=(lat, lon),
                radius=2,
                color="#555",
                fill=True,
                fill_color="#aaa",
                tooltip=h,
            ).add_to(g_tiles)

    g_tiles.add_to(m)
    # g_hubs.add_to(m)
    # g_centers.add_to(m)

    # ================================
    # METRO EDGES
    # ================================
    g_metro = folium.FeatureGroup("Metro Edges", show=False)

    for (o, d) in metro_pairs:
        if o in node_meta and d in node_meta:
            lo = node_meta[o]
            ld = node_meta[d]
            folium.PolyLine(
                [(lo["lat"], lo["lon"]), (ld["lat"], ld["lon"])],
                color="#444",
                weight=3,
                opacity=0.9,
            ).add_to(g_metro)

    # g_metro.add_to(m)

    # ================================
    # OD + PATHS
    # ================================
    for idx, (o, d) in enumerate(od_pairs):

        if o not in node_index or d not in node_index:
            warn(f"OD {o}->{d} absent from LSCC.")
            continue

        io = node_index[o]
        idd = node_index[d]

        D_el  = float(X_elapsed[idd, io])
        D_dir = predict_direct(o, d, alpha, beta)
        D_eff = D_el / D_dir if D_dir > 0 else float("nan")

        info(f"[OD {o}->{d}] D_elapsed={D_el:.1f}  D_direct={D_dir:.1f}  D_eff={D_eff:.3f}")

        # coordinates for OD endpoints
        lat_o = node_meta[o]["lat"]
        lon_o = node_meta[o]["lon"]
        lat_d = node_meta[d]["lat"]
        lon_d = node_meta[d]["lon"]

        # OD markers
        g_od_markers = folium.FeatureGroup(f"OD Markers {o}->{d}", show=True)

        folium.CircleMarker(
            location=(lat_o, lon_o),
            radius=7,
            color="#0D47A1",
            fill=True,
            fill_color="#0D47A1",
            fill_opacity=0.9,
            tooltip=f"Origin {o}",
        ).add_to(g_od_markers)

        folium.CircleMarker(
            location=(lat_d, lon_d),
            radius=7,
            color="#C62828",
            fill=True,
            fill_color="#C62828",
            fill_opacity=0.9,
            tooltip=f"Destination {d}",
        ).add_to(g_od_markers)

        g_od_markers.add_to(m)

        # OD straight line
        fg_od = folium.FeatureGroup(f"OD {o}->{d}", show=True)

        folium.PolyLine(
            [(lat_o, lon_o), (lat_d, lon_d)],
            color="#002060",
            weight=OD_EDGE_WEIGHT,
            opacity=0.95,
            tooltip=f"OD {o}->{d}\nD_eff={D_eff:.3f}",
        ).add_to(fg_od)

        tri = arrow_triangle(lat_o, lon_o, lat_d, lon_d, scale=2.0)
        if tri:
            folium.Polygon(
                tri,
                color="#002060",
                fill=True,
                fill_color="#002060",
                fill_opacity=0.95,
            ).add_to(fg_od)

        fg_od.add_to(m)

        # Enumerate paths
        res = enumerate_first_hitting_paths((o, d), packs, nodes, node_index)
        paths_geo = res["paths_geo"]
        info(f"  #paths = {len(paths_geo)}   P_sum={res['P_sum']:.5f}")

        for k, p in enumerate(paths_geo):
            color = PATH_COLORS[k % len(PATH_COLORS)]
            gh_path = p["path_geo"]
            prob    = p["prob"]
            dist_m  = p["dist"]
            steps   = p["t"]

            fg = folium.FeatureGroup(f"Path {k+1}: {o}->{d}", show=True)

            tooltip = (
                f"Path {k+1}<br>"
                f"P={prob:.3e}<br>"
                f"dist={dist_m:.1f} m<br>"
                f"steps={steps}"
            )

            # step through edges
            for u, v in zip(gh_path[:-1], gh_path[1:]):
                lat_u, lon_u = node_meta[u]["lat"], node_meta[u]["lon"]
                lat_v, lon_v = node_meta[v]["lat"], node_meta[v]["lon"]

                if u == v:   # self-loop
                    draw_self_loop(lat_u, lon_u, SELF_LOOP_RADIUS_M, color, fg)
                    continue

                folium.PolyLine(
                    [(lat_u, lon_u), (lat_v, lon_v)],
                    color=color,
                    weight=PATH_EDGE_WEIGHT,
                    opacity=0.95,
                    dash_array="6,6",
                    tooltip=tooltip,
                ).add_to(fg)

                tri2 = arrow_triangle(lat_u, lon_u, lat_v, lon_v, scale=1.4)
                if tri2:
                    folium.Polygon(
                        tri2,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.95,
                    ).add_to(fg)

            fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_html)
    info(f"[MAP SAVED] {out_html}")


# ============================================================
# CONTEXT BUILDER
# ============================================================

def build_context(pep_path: str) -> Dict[str, Any]:

    info("Loading PEP...")
    df_all = load_pep(pep_path)

    info("Windowing...")
    df_win0 = subset_window(df_all)

    info("LSCC...")
    lscc_nodes = compute_lscc(df_win0)
    info(f"LSCC size = {len(lscc_nodes)}")

    df_win = df_win0[
        df_win0[START_COL].isin(lscc_nodes)
        & df_win0[END_COL].isin(lscc_nodes)
    ].copy()

    nodes = sorted(lscc_nodes)
    slices, node_index, packs = build_packs(df_win, nodes)
    N = len(nodes)

    union = df_win.groupby(START_COL)[COUNT_COL].sum()
    origins = [node_index[o] for o in union.index if o in node_index]

    info("Computing elapsed distances...")
    Xbar, P_hit = batched_elapsed_all_origins(packs, N, origins)

    info("Fitting regression...")
    info("Using regression from cloned effective-distance engine...")
    reg = fit_region_regression(df_all)


    if reg is None:
        warn("Regression unavailable; aborting path decomposition.")
        return {
            "nodes": [],
            "node_index": {},
            "packs": [],
            "X_elapsed": np.empty((0, 0)),
            "P_hit": np.empty((0, 0)),
            "alpha": float("nan"),
            "beta": float("nan"),
            "node_meta": {},
            "metro_pairs": set(),
        }

    alpha, beta, rmse = reg


    # --- Manifest + metadata + metro edges ---
    manifest = None
    info("Manifest disabled for path decomposition (external OD mode).")


    if manifest is None:
        warn("No manifest found or failed to load; falling back to pure geohash lat/lon, no hubs, no metro edges.")
        node_meta: Dict[str, Dict[str, Any]] = {}
        for g in nodes:
            lat, lon = geohash_to_latlon(g)
            node_meta[g] = {
                "lat": lat,
                "lon": lon,
                "is_hub": False,
                "is_center": False,
            }
        metro_pairs: Set[Tuple[str, str]] = set()
    else:
        node_meta = build_node_meta_from_manifest(manifest, nodes)
        metro_pairs = build_metro_pairs_from_manifest(manifest)

    return dict(
        df_all=df_all,
        df_win=df_win,
        nodes=nodes,
        node_index=node_index,
        slices=slices,
        packs=packs,
        X_elapsed=Xbar,
        P_hit=P_hit,
        alpha=alpha,
        beta=beta,
        node_meta=node_meta,
        metro_pairs=metro_pairs,
        manifest=manifest,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    # run_dir = os.path.join(RUNS_DIR, RUN_NAME)
    pep_path = OD_INPUT_PATH
    input_stem = os.path.splitext(os.path.basename(OD_INPUT_PATH))[0]
    # info(f"RUN_NAME = {RUN_NAME}")
    # info(f"PEP path = {pep_path}")

    if not os.path.exists(pep_path):
        warn("PEP OD CSV not found at expected path.")
        raise FileNotFoundError(pep_path)


    out_dir = OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    out_html = os.path.join(
        out_dir,
        f"{input_stem }_path_decomposition_geohash__{SPAN_START.replace(':','-')}_to_{SPAN_END.replace(':','-')}.html"
    )

    ctx = build_context(pep_path)
    build_map(ctx, OD_PAIRS, out_html)


if __name__ == "__main__":
    main()
