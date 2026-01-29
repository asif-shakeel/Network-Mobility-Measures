#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nm_effective_distance_geohash.py
================================

Network-Level Effective Distance from Aggregated OD Data (Geohash)
------------------------------------------------------------------

This module computes **time-elapsed and effective distances** on a
mobility network constructed from *aggregated origin–destination (OD)
flows* (NetMob 2024 format) for a **single city / region**, spatially
discretized using **geohash cells**.

It implements the full time-dependent Markov-chain framework introduced in:

    Network-Level Measures of Mobility from Aggregated Origin-Destination Data
    https://arxiv.org/abs/2502.04162

Unlike trajectory-based approaches, all quantities here are derived
*directly from aggregated OD flows* without individual paths.


======================================================================
INPUT
======================================================================

Aggregated OD CSV (NetMob format):

    • start_geohash   : origin geohash
    • end_geohash     : destination geohash
    • date            : YYYYMMDD
    • time            : HH:MM:SS (bin start)
    • trip_count      : number of trips in bin
    • median_length_m : median path length (meters)

Example:
    od_mx_agg5_3h_MexicoCity_20190101T000000_20191231T000000.csv

Spatial resolution:
    GEOHASH_PRECISION = 5

Temporal resolution:
    TIME_RES_MIN = 180   # bin width (minutes)

Window analyzed:
    SPAN_START → SPAN_END

All computations are restricted to the **largest strongly connected
component (LSCC)** of the union graph within the selected window.


======================================================================
MODEL
======================================================================

Let F(t) be the OD flow matrix at time bin t:

    F(t)[d, o] = number of trips from origin o to destination d.

Define the column-stochastic transition operator:

    M(t) = F(t) · diag(1 / Σ_d F(t)[d,o]).

Let D(t) be the matrix of observed median path lengths at time t.

Over a sequence of K time bins, the **time-elapsed distance**
from origin j to destination i is defined by the first-hitting recursion
(Section 4.1 of the paper):

    D_elapsed(i,j) = Σ_t y_t^{ij}(i) · π_t^{ij} / Σ_t π_t^{ij},

where:

    π_t^{ij} = probability of first hitting i from j at time t,
    y_t^{ij}(i) = expected path length conditional on that first hit.

The total hitting probability is:

    P_hit(i,j) = Σ_t π_t^{ij}.


======================================================================
BASELINE AND EFFECTIVE DISTANCE
======================================================================

A direct-distance baseline D_direct(i,j) is computed using the staged
A–E hierarchy (Section 4.2):

    A: same date + same time-of-day
    B: same day-of-week + same time-of-day
    C: any date + same time-of-day
    D: global OD median
    E: regression on geodesic distance

The **effective distance** is:

    D_eff(i,j) = D_elapsed(i,j) / D_direct(i,j).


======================================================================
GUP PROXY (STRUCTURAL GAPS)
======================================================================

A structural “gap” proxy is defined as:

    GUP = all (o,d) in the LSCC that never appear with positive flow
          anywhere in the full dataset (“not-in-D” proxy).

This is a static approximation of the GUP notion used in the paper.


======================================================================
OUTPUT ARTIFACTS
======================================================================

Let <stem> be the input filename without extension and
<span_tag> = <SPAN_START>__<SPAN_END>.

All files are written to:

    OUTPUT_DIR = <BASE_DIR>/effective_distance/

----------------------------------------------------------------------
1) Full effective-distance table
----------------------------------------------------------------------

    <stem>_eff_all_<span_tag>.csv

Columns:
    start_geohash, end_geohash,
    D_elapsed, D_direct, D_eff, P_hit, baseline_src, is_gup


----------------------------------------------------------------------
2) Subsets (all rows, before probability gating)
----------------------------------------------------------------------

    <stem>_eff_gup_all_<span_tag>.csv
    <stem>_eff_non_gup_all_<span_tag>.csv


----------------------------------------------------------------------
3) Probability-gated percentile bands (for visualization)
----------------------------------------------------------------------

    <stem>_eff_gup_q<L>_to_q<U>_<span_tag>.csv
    <stem>_eff_non_gup_q<L>_to_q<U>_<span_tag>.csv

where:
    [L,U] = percentile band in D_eff
    and P_hit ≥ MIN_P_HIT_{GUP|NON_GUP}


----------------------------------------------------------------------
4) Interactive maps (Folium)
----------------------------------------------------------------------

    <stem>_map_gup_q<L>_to_q<U>_<span_tag>.html
    <stem>_map_non_gup_q<L>_to_q<U>_<span_tag>.html

Each map shows directed OD edges colored by D_eff, with arrowheads,
tooltips, and popups displaying:

    D_elapsed, D_direct, D_eff, P_hit, GUP flag

The maximum-D_eff edge is optionally highlighted.


----------------------------------------------------------------------
5) Regression diagnostic
----------------------------------------------------------------------

    <stem>_regression_fit.pdf

Observed median OD distance vs geodesic distance
with fitted baseline (E-stage).


----------------------------------------------------------------------
6) First-hitting path enumeration (optional debug)
----------------------------------------------------------------------

    <stem>_paths_<origin>_<dest>.csv

All first-hitting paths contributing to D_elapsed for a selected OD pair,
including path sequence, probability, distance, and hitting time.


======================================================================
INTERPRETATION
======================================================================

Effective distance identifies OD pairs whose mobility is *structurally
suppressed* relative to their geometric baseline.

High D_eff indicates network-level barriers, bottlenecks, or latent
segmentation in urban mobility, even when aggregated flows are used.

For full theory, proofs, and interpretation:

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
import time
from typing import Optional, Tuple, List, Dict, Union, Set

import numpy as np
import pandas as pd
from scipy import sparse
import networkx as nx
import folium
from branca.colormap import LinearColormap

# ========================
# CONFIG / PATHS
# ========================

BASE_DIR = "/Users/asif/Documents/nm24/outputs/nm_outputs"
DATA_DIR = os.path.join(BASE_DIR, "data")

INPUT_OD_FILE = "od_mx_agg5_3h_MexicoCity_20190101T000000_20191231T000000.csv"   # in BASE_DIR
OD_INPUT_PATH = os.path.join(DATA_DIR, INPUT_OD_FILE)

OUTPUT_DIR =os.path.join(BASE_DIR, "effective_distance") 
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Geohash / OD columns
GEOHASH_PRECISION = 5

# Time resolution and window
TIME_RES_MIN = 180
SPAN_START = "2019-09-23 06:00:00"
SPAN_END   = "2019-09-23 12:00:00"

# PEP Generation time span (for RUN_NAME tags)
# PEP_SPAN_START   = "2019-06-01 00:00:00"
# PEP_SPAN_END     = "2019-07-01 00:00:00"

COUNTRY = "Mexico"
REGION_NAME = "Mexico City, Mexico"

# neighbor / feeder / directions (for tags + manifest)
NEIGHBOR_TOPOLOGY   = "4"
FEEDER_MODE         = "single" # "hubs" # 
EDGE_DIRECTION_MODE = "potential" # "geometric" #
INCLUDE_BACKGROUND_EDGES = False

# ============================================================
# MAP HIGHLIGHT: MAX EFFECTIVE DISTANCE EDGE
# ============================================================

SHOW_MAX_ED_EDGE = True          # master switch
MAX_ED_EDGE_COLOR = "#ff0000"    # bright red
MAX_ED_EDGE_WEIGHT = 3           # thicker than normal
MAX_ED_EDGE_OPACITY = 1.0
MAX_ED_EDGE_LABEL = "MAX D_eff"



# How many OD pairs and PEP samples to inspect
TOP_N_OD = 10            # number of largest D_eff OD pairs to analyze
SAMPLE_PEPS_PER_OD = 3   # how many example PEP_IDs to store per OD
# How many OD pairs and PEP samples to inspect (separately by GUP / non-GUP)
TOP_N_OD_GUP      = 10   # top-N for is_gup == True
TOP_N_OD_NON_GUP  = 10     # top-N for is_gup == False
SAMPLE_PEPS_PER_OD = 3   # how many example PEP_IDs to store per OD

# ---- Probability thresholds for using an OD pair ----
# P_hit(O,D) = total probability of ever hitting D from O in the window (Eq. 4 denom)
# Only ODs with P_hit >= these thresholds will be considered for:
#   - maps (after this, D_eff percentiles are for visualization only)
#   - top-N PEP path statistics (no further D_eff percentile for top-N)
MIN_P_HIT_GUP     = 1.0e-6   # tweak as you like
MIN_P_HIT_NON_GUP = 1.0e-6

START_COL = "start_geohash"
END_COL   = "end_geohash"
DATE_COL  = "date"
TIME_COL  = "time"
COUNT_COL = "trip_count"
DIST_COL  = "median_length_m"

# Regression fallback (E)
ADD_RMSE_SIGMA = 1.0
REGRESSION_ON_LSCC = True
REGRESSION_OVERRIDE: Optional[Tuple[float, float, float]] = None  # (alpha, beta, rmse) #(-2870,1.47, 1330) #
# Batched recursion
BATCH_SIZE_MODE: Union[str, int] = "all_reachable"  # "auto", "all_reachable", or int
MEMORY_CAP_MB = 2048
MAX_ORIGINS_FOR_TEST: Optional[int] = None

# GUP switches (currently "not-in-D" proxy)
USE_GUP_ONLY_EDGES   = False   # compute D_eff only for these edges
SHOW_ONLY_GUP_ON_MAP = True    # map & percentile only on GUP subset

# Map/render
# Map/render
# Map/render: percentile bands for D_eff, per subset
# Keep only edges whose D_eff lies between these percentiles.
LOWER_PERCENTILE_Q_GUP     = 98.5
UPPER_PERCENTILE_Q_GUP     = 100

LOWER_PERCENTILE_Q_NON_GUP = 98.5
UPPER_PERCENTILE_Q_NON_GUP = 100

EDGE_WEIGHT_PX = 2

EDGE_WEIGHT_PX   = 2      # thinner lines than before
EDGE_CMAP_COLORS = ['magenta', 'navy']
FOLIUM_TILES     = "cartodbpositron"
SHOW_MAP_TITLE   = True

# Map UI
LEGEND_POSITION = 'bottomright'   # {'bottomright','bottomleft','topright','topleft'}
TITLE_TOP_PX    = 20             # vertical offset for the title banner

# Arrowheads – tuned to be smaller / closer to pmcm_map
ARROW_FRAC_OF_SEG = 0.1   # was 0.12
ARROW_MIN_LEN_M   = 30     # was 30
ARROW_MAX_LEN_M   = 5000   # was 5000
ARROW_TIP_ANGLE   = 22.0

DO_PATH_ENUM= True   # or just remove this guard
od_pair=("9g3qy", "9g3qv") # ("djgyc","djgyd") #("9g3s0", "9g3s7")


# Logging cadence
LOG_EVERY_ORIGIN_FRACTION = 10

def banner(msg: str): print(f"\n=== {msg} ===", flush=True)
def info(msg: str):   print(f"[INFO] {msg}", flush=True)
def warn(msg: str):   print(f"[WARN] {msg}", flush=True)

# ========================
# PATH HELPERS
# ========================

def _city_tag() -> str:
    return (REGION_NAME.split(',')[0]).replace(' ', '') or "Region"

def _top_tag() -> str:
    try:
        return f"gh-{NEIGHBOR_TOPOLOGY}-res-{GEOHASH_PRECISION}"
    except Exception:
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

# PEP_SPAN_START_TS = pd.to_datetime(PEP_SPAN_START)
# PEP_SPAN_END_TS   = pd.to_datetime(PEP_SPAN_END)
# start_tag = PEP_SPAN_START_TS.strftime("%Y%m%dT%H%M")
# end_tag   = PEP_SPAN_END_TS.strftime("%Y%m%dT%H%M")

# RUN_NAME = (
#     f"{city_tag}/"
#     f"{top_tag}/"
#     f"{fdr_tag}/"
#     f"{edge_tag}/"
#     f"{overlay_tag}/"
#     f"{start_tag}_{end_tag}/"
#     f"m{TIME_RES_MIN}/x-periodic_fixed_point"
# )

# BASE_DIR   = os.path.join(RUN_DIR, RUN_NAME, "csvs")
# OUTPUT_DIR = os.path.join(RUN_DIR, RUN_NAME, "effective_distance")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# OD_INPUT_PATH = os.path.join(BASE_DIR, "pep_od.csv")

# CORRIDOR_MANIFEST_FILENAME = f"corridors_manifest_{city_tag}_mode-geohash_{top_tag}_{fdr_tag}_{edge_tag}_{overlay_tag}.json"
# CORRIDOR_MANIFEST_PATH     = os.path.join(BASE_DIR, "corridors_outputs_geohash", CORRIDOR_MANIFEST_FILENAME)

# ========================
# GEO / TIME HELPERS
# ========================

_base32 = "0123456789bcdefghjkmnpqrstuvwxyz"

def geohash_to_latlon(gh: str) -> Tuple[float, float]:
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

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*(math.sin(dlmb/2)**2)
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

def vectorized_load_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse TIME_COL into _t0, _t1. For PEP OD, local_time is a single timestamp
    like '2025-06-06 00:00:00'. We still support 'start - end' ranges if present.
    """
    s = df[TIME_COL].astype(str).str.strip()
    has_range = s.str.contains(r"\s-\s", regex=True)

    if not has_range.any():
        # Simple timestamp only
        t0 = pd.to_datetime(s, errors="coerce")
        if t0.isna().any():
            warn("Some local_time values could not be parsed; they will be NaT.")
        t1 = t0 + pd.to_timedelta(TIME_RES_MIN, unit="min")
        df["_t0"]   = t0
        df["_t1"]   = t1
        df["_date"] = t0.dt.date
        df["_dow"]  = t0.dt.dayofweek
        return df

    # Range case: 'START - END'
    parts = s.str.split(r"\s-\s", n=1, expand=True)
    left  = parts[0].str.strip()
    right = parts[1].str.strip()

    t0 = pd.to_datetime(left, errors="coerce")
    is_time_only = right.str.match(r"^\d{2}:\d{2}:\d{2}$", na=False)

    t1_full = pd.to_datetime(right.where(~is_time_only), errors="coerce")
    t1_time = pd.to_datetime(right.where(is_time_only), format="%H:%M:%S", errors="coerce")

    t1 = t1_full.copy()
    if is_time_only.any():
        combo = t0.dt.strftime("%Y-%m-%d") + " " + t1_time.dt.strftime("%H:%M:%S")
        combo = combo.where(is_time_only)
        t1 = pd.to_datetime(combo, errors="coerce")

    t1 = t1.where(t1.notna(), t0 + pd.to_timedelta(TIME_RES_MIN, unit="min"))

    df["_t0"]   = t0
    df["_t1"]   = t1
    df["_date"] = t0.dt.date
    df["_dow"]  = t0.dt.dayofweek
    return df

def load_data(path: str) -> pd.DataFrame:
    banner("LOAD PEP OD CSV")
    t0 = time.time()
    try:
        df = pd.read_csv(path, engine="pyarrow")
    except Exception:
        df = pd.read_csv(path)
    df = vectorized_load_time(df)
    info(f"rows={len(df):,}  took {time.time()-t0:.2f}s")
    if "_t0" in df.columns:
        info(f"_t0 range: {df['_t0'].min()} → {df['_t0'].max()}")
    return df

# ========================
# MANIFEST HELPERS
# ========================

def load_corridor_manifest(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        warn(f"corridor manifest not found: {path}")
        return None
    import json
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        warn(f"failed to load manifest: {e}")
        return None

def build_node_meta_from_manifest(manifest: Optional[dict],
                                  nodes: List[str]) -> Dict[str, Dict[str, Union[float,bool]]]:
    """
    Return meta for each node in 'nodes':
      {
        gh: {
          'lat': float,
          'lon': float,
          'is_hub': bool,
          'is_center': bool
        }
      }
    Uses manifest if available; falls back to geohash decode.
    """
    meta: Dict[str, Dict[str, Union[float,bool]]] = {}
    hubs_set: Set[str] = set()
    centers_set: Set[str] = set()
    node_info: Dict[str, dict] = {}

    if manifest is not None:
        hubs = manifest.get("hubs", {})
        hubs_set = set(hubs.keys())
        potentials = manifest.get("potentials", {})
        centers = potentials.get("centers", []) or []
        centers_set = set(centers)
        node_info = manifest.get("nodes", {}) or {}

    for gh in nodes:
        gh_str = str(gh)
        if gh_str in node_info:
            rec = node_info[gh_str]
            lat = float(rec.get("lat", geohash_to_latlon(gh_str)[0]))
            lon = float(rec.get("lon", geohash_to_latlon(gh_str)[1]))
        else:
            lat, lon = geohash_to_latlon(gh_str)

        # Hubs and centers come ONLY from manifest lists — not from rec fields
        is_hub    = (gh_str in hubs_set)
        is_center = (gh_str in centers_set)

        meta[gh_str] = {
            "lat": lat,
            "lon": lon,
            "is_hub": is_hub,
            "is_center": is_center,
        }

        meta[gh_str] = {
            "lat": lat,
            "lon": lon,
            "is_hub": is_hub,
            "is_center": is_center,
        }
    return meta

def build_metro_pairs_from_manifest(manifest: Optional[dict]) -> Set[Tuple[str,str]]:
    """
    Extract metro edges (origin,dest) from manifest edges["AM"][...]["is_metro"] == 1.
    """
    metro_pairs: Set[Tuple[str,str]] = set()
    if manifest is None:
        return metro_pairs
    edges = manifest.get("edges", {})
    am = edges.get("AM", [])
    for e in am:
        try:
            if int(e.get("is_metro", 0)) == 1:
                o = str(e.get("origin"))
                d = str(e.get("dest"))
                if o and d and (o != d):
                    metro_pairs.add((o, d))
        except Exception:
            continue
    return metro_pairs

# ========================
# LSCC + SLICES
# ========================

def compute_lscc_nodes(df_win: pd.DataFrame) -> List[str]:
    # banner("LSCC (window union graph)")
    G = nx.DiGraph()
    nodes = sorted(set(df_win[START_COL].astype(str)).union(set(df_win[END_COL].astype(str))))
    G.add_nodes_from(nodes)
    pos = df_win[df_win[COUNT_COL] > 0]
    edges = pos.groupby([START_COL, END_COL], as_index=False)[COUNT_COL].sum()
    for _, r in edges.iterrows():
        o = str(r[START_COL])
        d = str(r[END_COL])
        if o != d:
            G.add_edge(o, d)
    if G.number_of_edges() == 0:
        warn("no edges in window; LSCC empty")
        return []
    sccs = list(nx.strongly_connected_components(G))
    sccs.sort(key=len, reverse=True)
    lscc = sorted(sccs[0]) if sccs else []
    # info(f"LSCC size={len(lscc)}  (union nodes={len(G.nodes)}, edges={len(G.edges)})")
    return lscc

def precompute_slices(df_win: pd.DataFrame,
                      nodes: List[str]) -> Tuple[List[pd.Timestamp], Dict[str,int], List[Dict[str,np.ndarray]]]:
    # banner("SLICE PRECOMPUTE")
    idx = {g: k for k, g in enumerate(nodes)}
    slices = sorted(df_win["_t0"].unique().tolist())
    packs: List[Dict[str,np.ndarray]] = []

    for s in slices:
        g = df_win[df_win["_t0"] == s]
        agg = g.groupby([START_COL, END_COL], as_index=False).agg(
            **{COUNT_COL: (COUNT_COL, "sum"), DIST_COL: (DIST_COL, "median")}
        )

        i_idx: List[int] = []
        j_idx: List[int] = []
        flow: List[float] = []
        dist: List[float] = []
        outflow: Dict[int,float] = {}

        for _, r in agg.iterrows():
            o = str(r[START_COL])
            d = str(r[END_COL])
            j = idx.get(o)
            i = idx.get(d)
            if j is None or i is None:
                continue
            c_val = float(r[COUNT_COL])
            d_val = float(r[DIST_COL]) if pd.notna(r[DIST_COL]) else 0.0
            if c_val <= 0:
                continue
            i_idx.append(i)
            j_idx.append(j)
            flow.append(c_val)
            dist.append(d_val)
            outflow[j] = outflow.get(j, 0.0) + c_val

        # ensure each origin has at least a self-loop
        # for j in range(len(nodes)):
            # if outflow.get(j, 0.0) <= 0.0:
            #     i_idx.append(j)
            #     j_idx.append(j)
            #     flow.append(5.0)
            #     dist.append(0.0)

        packs.append({
            "i": np.asarray(i_idx, dtype=np.int32),
            "j": np.asarray(j_idx, dtype=np.int32),
            "c": np.asarray(flow, dtype=np.float64),
            "d": np.asarray(dist, dtype=np.float64),
        })

    # info(f"precompute done (slices={len(slices)})")
    return slices, idx, packs

def build_Mt_Dt_from_pack(pack: Dict[str,np.ndarray], N: int) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    i = pack["i"]
    j = pack["j"]
    c = pack["c"]
    d = pack["d"]
    M = sparse.coo_matrix((c, (i, j)), shape=(N, N)).tocsr()
    col_sums = np.array(M.sum(axis=0)).ravel()
    col_sums[col_sums == 0.0] = 1.0
    M = M @ sparse.diags(1.0 / col_sums)
    D = sparse.coo_matrix((d, (i, j)), shape=(N, N)).tocsr()
    return M, D

# ========================
# BATCHED ELAPSED-DISTANCE
# ========================

def auto_batch_size(N: int, cap_mb: int = MEMORY_CAP_MB) -> int:
    bytes_cap = cap_mb * 1_000_000
    safety   = 1.5
    B = int(bytes_cap / (safety * 4 * 8 * max(1, N)))
    return max(1, min(N, B))

def resolve_batch_size(N: int, origin_indices: np.ndarray) -> int:
    if isinstance(BATCH_SIZE_MODE, int):
        return max(1, min(int(BATCH_SIZE_MODE), len(origin_indices)))
    mode = str(BATCH_SIZE_MODE).lower()
    if mode == "all_reachable":
        return max(1, len(origin_indices))
    return auto_batch_size(N)

from typing import Tuple


def batched_elapsed_all_origins(packs, N, origin_indices, batch_size=128):
    """
    Compute time-elapsed distance D_elapsed(i,j) and hit probability P_hit(i,j)
    for all origins j in origin_indices and all destinations i in {0,...,N-1}.

    This implements the recursion of Section 4.1 in the paper:
        p^{ij,t}    = probability vector at time t
        y^{ij,t}    = accumulated-distance conditional expectation
        π_t^{ij}    = first-hit probability at time t
        D_elapsed   = sum_t ( y^{ij,t}_i * π_t^{ij} ) / sum_t π_t^{ij}

    Parameters
    ----------
    packs : list of dicts
        Each pack represents a time slice s and contains:
            'i' : array of destination indices
            'j' : array of origin indices
            'c' : array of flow counts
            'd' : array of distances
    N : int
        Number of nodes (LSCC size)
    origin_indices : list or array
        Origins for which to compute all D_elapsed(:, origin)
    batch_size : int
        Not used here; included for interface compatibility

    Returns
    -------
    Xbar  : (N,N) array
        Time-elapsed distances: Xbar[i,j] = D_elapsed(i,j)
    P_hit : (N,N) array
        Total hit probability: P_hit[i,j] = sum_t π_t^{ij}
    """

    # ---------------------------------------------------------------
    # (1) Build all per-slice transition matrices M^t and distance D^t
    # ---------------------------------------------------------------
    Ms = []
    Ds = []

    for pack in packs:
        i_idx = pack["i"]
        j_idx = pack["j"]
        c = pack["c"].astype(float)
        d = pack["d"].astype(float)

        # Sparse transition matrix (column stochastic)
        M = sparse.coo_matrix((c, (i_idx, j_idx)), shape=(N, N)).tocsr()
        colsum = np.array(M.sum(axis=0)).ravel()
        colsum[colsum == 0] = 1.0
        M = M @ sparse.diags(1.0 / colsum)

        # Sparse distance matrix
        D = sparse.coo_matrix((d, (i_idx, j_idx)), shape=(N, N)).tocsr()

        Ms.append(M)
        Ds.append(D)

    K = len(Ms)


    # ---------------------------------------------------------------
    # Prepare outputs
    # ---------------------------------------------------------------
    Xbar = np.full((N, N), np.nan, float)
    P_hit = np.zeros((N, N), float)

    # ---------------------------------------------------------------
    # (2) Loop over origins j
    # ---------------------------------------------------------------
    for j in origin_indices:

        # Probability at t=0 is concentrated at origin
        # p^{ij,1} will come from M^1[:, j]
        M1 = Ms[0]
        D1 = Ds[0]

        for i in range(N):
            # if i == j:
            #     continue

            # t = 1 probability
            p = M1[:, j].toarray().ravel()
            p[j] = 0.0

            # accumulated distance after the *first step*: y^1_k = D1[k, j]
            # (distance from origin j to current state k after one step)
            y = D1[:, j].toarray().ravel()

            num = 0.0
            den = 0.0

            # FIRST HIT at t = 1  (now use y[i] instead of re-reading D1)
            pi_1 = p[i]
            if pi_1 > 0.0:
                num += y[i] * pi_1
                den += pi_1

            # -------------------------------------------------------
            # RECURSION for t = 2,...,K
            # -------------------------------------------------------
            for t in range(1, K):
                Mt = Ms[t]
                Dt = Ds[t]

                # p̃^{ij,t}: enforce no hitting i or returning to j *before* transition
                p_tilde = p.copy()
                p_tilde[i] = 0.0
                p_tilde[j] = 0.0

                # p^{ij,t+1} = M^{t+1} p̃
                p_next = Mt @ p_tilde

                # skip nodes with zero probability
                mask = p_next > 0

                # Compute:
                # MD = (M ∘ D) p̃     ← expected new-step distance
                # MY = M (p̃ ⊙ y)     ← expected old-distance continuation
                MD = (Mt.multiply(Dt)) @ p_tilde
                MY = Mt @ (p_tilde * y)

                y_next = np.zeros(N)
                y_next[mask] = (MD[mask] + MY[mask]) / p_next[mask]

                # First-hit probability at t+1
                pi_t = p_next[i]

                # Contribution from y^{ij,t}_i * π_t^{ij}
                if pi_t > 0.0:
                    num += y_next[i] * pi_t
                    den += pi_t

                # advance
                p = p_next
                y = y_next

            # -------------------------------------------------------
            # Final time-elapsed distance
            # -------------------------------------------------------
            if den > 0.0:
                Xbar[i, j] = num / den
                P_hit[i, j] = den

    return Xbar, P_hit

def elapsed_all_origins_full3d(packs, N):
    """
    Fully 3D NumPy version of the Sec. 4.1 recursion.

    Indices:
      - k: current state (0..N-1)
      - j: origin index (0..N-1)
      - i: destination index (0..N-1)

    We compute, for ALL (i,j) pairs simultaneously:
      D_elapsed[i,j] and P_hit[i,j] = sum_t π_t^{ij}.

    This is the "no batching" case: we treat all N origins
    and all N destinations in one shot, with a state tensor:

      P[k,j,i] = p^{ij}_t(k)   (prob. at time t, at state k)
      Y[k,j,i] = y^{ij}_t(k)   (expected accumulated distance at state k)
    """

    import numpy as np

    # ------------------------------------------------------------
    # Build dense Mt (transition) and Dt (distance) matrices
    # ------------------------------------------------------------
    Mt_list = []
    Dt_list = []
    for pack in packs:
        M, D = build_Mt_Dt_from_pack(pack, N)  # same helper as scalar version
        Mt_list.append(M.toarray())            # (N, N)
        Dt_list.append(D.toarray())            # (N, N)

    Mt_list = np.stack(Mt_list, axis=0)        # (K, N, N)
    Dt_list = np.stack(Dt_list, axis=0)        # (K, N, N)
    K = Mt_list.shape[0]


    # ------------------------------------------------------------
    # Outputs: D_elapsed[i,j] and P_hit[i,j]
    # ------------------------------------------------------------
    D_elapsed = np.full((N, N), np.nan, dtype=float)
    P_hit     = np.zeros((N, N), dtype=float)

    # We'll fill these at the end for all i,j
    # (they are just num/den across time).
    # Internally we'll call them num, den:
    num = np.zeros((N, N), dtype=float)   # numerator over t
    den = np.zeros((N, N), dtype=float)   # denominator over t = total hit prob

    # ------------------------------------------------------------
    # t = 1  (one-step transition from each origin j)
    # ------------------------------------------------------------
    M1 = Mt_list[0]        # (N, N)
    D1 = Dt_list[0]        # (N, N)

    # P1(k, j) = M1[k, j] for all origins j
    # With shape (N, N):
    P1 = M1.copy()

    # Remove self-loop at origin: P1[j, j] = 0
    idx = np.arange(N)
    P1[idx, idx] = 0.0

    # For t = 1, the distance if you hit i from j is exactly D1[i, j].
    Y1 = D1.copy()   # Y1(k, j) = distance of the first step j→k

    # First-hit at t=1:
    #   π_1^{ij} = P1[i, j]
    #   y_1^{ij} = D1[i, j]
    pi1 = P1.copy()   # (N, N) interpreting row=i, col=j
    y1  = Y1.copy()   # same shape

    num += y1 * pi1
    den += pi1

    # ------------------------------------------------------------
    # Lift to 3D: P(k, j, i) and Y(k, j, i)
    # ------------------------------------------------------------
    # At t=1, the underlying p^j_t(k) and y^j_t(k) don't depend on i yet:
    # same distribution for *every* destination i. So we replicate along i.
    #
    # P has axes: (k, j, i)
    P = P1[:, :, None] * np.ones((1, 1, N), dtype=float)   # (N, N, N)
    Y = Y1[:, :, None] * np.ones((1, 1, N), dtype=float)   # (N, N, N)

    # ------------------------------------------------------------
    # Build time-invariant mask(k, j, i)
    # ------------------------------------------------------------
    # mask[k, j, i] = True means "allowed to stay at k before transition"
    # We want to enforce:
    #   - no return to origin j: k != j
    #   - no early hit of destination i: k != i
    # mask = np.ones((N, N, N), dtype=bool)

    # # No return to origin:
    # #   For each origin j, state k=j is forbidden for ALL i.
    # #   That's mask[j, j, :] = False for each j.
    # mask[idx, idx, :] = False

    # # No early hit of destination:
    # #   For each destination i, state k=i is forbidden for ALL j.
    # #   That's mask[i, :, i] = False for each i.
    # mask[idx, :, idx] = False


    # # axes: P[k, j, i]
    # mask = np.ones((N, N, N), dtype=bool)
    # k_idx, j_idx, i_idx = np.indices((N, N, N))

    # # forbid k == i (no early hit of destination)
    # # forbid k == j (no return to origin)
    # mask[(k_idx == i_idx) | (k_idx == j_idx)] = False
    mask = np.ones((N, N, N), dtype=bool)
    idx = np.arange(N)
    mask[idx, idx, :] = False
    mask[idx, :, idx] = False



    # ------------------------------------------------------------
    # Time recursion: t = 2 ... K
    # ------------------------------------------------------------
    for t in range(1, K):
        Mt = Mt_list[t]    # (N, N)
        Dt = Dt_list[t]    # (N, N)

        # Apply mask: set forbidden states to 0 BEFORE transition
        P_tilde = np.where(mask, P, 0.0)   # (N, N, N)

        # ----- Probability update --------------------------------
        # P_next[k, j, i] = sum_l Mt[k, l] * P_tilde[l, j, i]
        # -> tensordot over l:
        #   l is axis 1 in Mt, axis 0 in P_tilde
        P_next = np.tensordot(Mt, P_tilde, axes=(1, 0))   # (N, N, N)

        # ----- New-step distance term MD -------------------------
        # MD[k, j, i] = sum_l Mt[k, l] * Dt[k, l] * P_tilde[l, j, i]
        MtDt = Mt * Dt   # elementwise, shape (N, N)
        MD   = np.tensordot(MtDt, P_tilde, axes=(1, 0))   # (N, N, N)

        # ----- Old-distance continuation MY ----------------------
        # MY[k, j, i] = sum_l Mt[k, l] * (P_tilde[l, j, i] * Y[l, j, i])
        Z  = P_tilde * Y                                   # (N, N, N)
        MY = np.tensordot(Mt, Z, axes=(1, 0))             # (N, N, N)

        # ----- Conditional expectation update -------------------
        with np.errstate(divide="ignore", invalid="ignore"):
            Y_next = np.where(P_next > 0, (MD + MY) / P_next, 0.0)

        # ----- First-hit probabilities at this step -------------
        # By construction, P_next is the distribution at time t+1.
        # For (i,j), the first-hit probability at t+1 is:
        #   π_{t+1}^{ij} = P_next[i, j, i]
        # and the corresponding conditional distance:
        #   y_{t+1}^{ij} = Y_next[i, j, i]
        idx = np.arange(N)
        pi_t = P_next[idx, :, idx]   # (N, N)   (row=i, col=j)
        y_t  = Y_next[idx, :, idx]   # (N, N)

        # Ignore self-pairs i==j, to match scalar behavior
        pi_t[idx, idx] = 0.0
        y_t[idx, idx]  = 0.0

        num += y_t * pi_t
        den += pi_t

        P = P_next
        Y = Y_next

    # ------------------------------------------------------------
    # Final D_elapsed[i,j] and P_hit[i,j]
    # ------------------------------------------------------------
    # If we never hit i from j, den[i,j] = 0 and D_elapsed[i,j] stays NaN.
    hit_mask = den > 0.0
    D_elapsed[hit_mask] = num[hit_mask] / den[hit_mask]
    P_hit[hit_mask]     = den[hit_mask]

    # idx = np.arange(N)
    # D_elapsed[idx, idx] = np.nan
    # P_hit[idx, idx]     = 0.0

    return D_elapsed, P_hit





def enumerate_first_hitting_paths(
    od_pair: Tuple[str, str],
    packs: List[Dict[str, np.ndarray]],
    nodes: List[str],
    node_index: Dict[str, int],
    write_csv: bool = True,
    csv_path = None
):
    import csv

    origin_geo, dest_geo = od_pair
    if origin_geo not in node_index or dest_geo not in node_index:
        warn(f"OD pair {origin_geo}->{dest_geo} not in LSCC; skipping path enumeration.")
        return {
            "origin": origin_geo,
            "destination": dest_geo,
            "P_sum": 0.0,
            "paths": [],
            "paths_geo": [],
            "K": 0
        }

    j = node_index[origin_geo]
    i = node_index[dest_geo]
    N = len(nodes)

    Ms, Ds = zip(*(build_Mt_Dt_from_pack(p, N) for p in packs))
    K = len(Ms)

    final_paths = []
    P_sum = 0.0

    # ------------------------------------------------
    # t = 1
    # ------------------------------------------------
    M1, D1 = Ms[0], Ds[0]
    col_j = M1[:, j].toarray().ravel()
    frontier = []

    for u in range(N):
        p_ju = col_j[u]
        if p_ju <= 0.0 or u == j:
            continue

        dist_ju = float(D1[u, j])

        if u == i:
            final_paths.append({
                "path": [j, i],
                "prob": p_ju,
                "dist": dist_ju,
                "t": 1,
            })
            P_sum += p_ju
        else:
            frontier.append({
                "node": u,
                "path": [j, u],
                "prob": p_ju,
                "dist": dist_ju,
            })

    # ------------------------------------------------
    # t = 2..K
    # ------------------------------------------------
    for t in range(2, K + 1):
        Mt, Dt = Ms[t - 1], Ds[t - 1]
        new_frontier = []

        for entry in frontier:
            u = entry["node"]
            col_u = Mt[:, u].toarray().ravel()

            for v in range(N):
                p_uv = col_u[v]
                if p_uv <= 0.0 or (v==j and  i!=j):
                    continue

                new_prob = entry["prob"] * p_uv
                new_dist = entry["dist"] + float(Dt[v, u])

                if v == i:
                    final_paths.append({
                        "path": entry["path"] + [i],
                        "prob": new_prob,
                        "dist": new_dist,
                        "t": t,
                    })
                    P_sum += new_prob
                else:
                    new_frontier.append({
                        "node": v,
                        "path": entry["path"] + [v],
                        "prob": new_prob,
                        "dist": new_dist,
                    })

        frontier = new_frontier
        if not frontier:
            break

    # ------------------------------------------------
    # Produce geo-path versions
    # ------------------------------------------------
    paths_geo = []
    for p in final_paths:
        geo = [nodes[idx] for idx in p["path"]]
        out = p.copy()
        out["path_geo"] = geo
        paths_geo.append(out)

    # ------------------------------------------------
    # Summary
    # ------------------------------------------------
    print(f"\n[first-hit paths] {origin_geo} → {dest_geo}: {len(final_paths)} paths")
    print(f"[first-hit paths] total probability mass P_sum = {P_sum:.8f}")

    if write_csv and final_paths:
        if not csv_path:
            csv_path = os.path.join(OUTPUT_DIR, f"paths_{origin_geo}_{dest_geo}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["origin", "destination", "t", "path", "probability", "distance_m"])
            for p in final_paths:
                path_geo = [nodes[idx] for idx in p["path"]]
                writer.writerow([
                    origin_geo,
                    dest_geo,
                    p["t"],
                    " -> ".join(path_geo),
                    p["prob"],
                    p["dist"],
                ])

        info(f"[first-hit paths] wrote CSV: {csv_path}")
    else:
        if not write_csv:
            info("[first-hit paths] write_csv=False → skipping CSV output")

    # ------------------------------------------------
    # Return a rich structured object with everything needed
    # ------------------------------------------------
    return {
        "origin": origin_geo,
        "destination": dest_geo,
        "P_sum": P_sum,
        "paths": final_paths,
        "paths_geo": paths_geo,
        "K": K
    }



import pandas as pd

def write_paths_csv(records, out_csv):
    import pandas as pd
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"[INFO] wrote {out_csv}    ({len(df)} paths)")



# ========================
# BASELINE (STAGED A–E)
# ========================

def staged_baseline_vectorized(df_all: pd.DataFrame,
                               cand_df: pd.DataFrame,
                               slices: List[pd.Timestamp],
                               reg_params: Optional[Tuple[float,float,float]],
                               add_rmse_sigma: float,
                               date0: pd.Timestamp) -> pd.DataFrame:
    """
    For each candidate OD, compute D_direct using staged A–E baseline:
      A: specific date0 & times-of-day
      B: same day-of-week as date0 & times-of-day
      C: any date but times-of-day in window
      D: any date/time (global)
      E: regression fallback on great-circle distance
    """
    key = [START_COL, END_COL]
    base_cols = key + ["D_elapsed"]
    if "P_hit" in cand_df.columns:
        base_cols.append("P_hit")
    outs = cand_df[base_cols].drop_duplicates().reset_index(drop=True)




    def gmed(df_sel):
        if df_sel.empty:
            return pd.DataFrame(columns=key + ["val"])
        g = (df_sel.groupby(key, as_index=False)[DIST_COL]
             .median()
             .rename(columns={DIST_COL: "val"}))
        return g

    slices_local = set(pd.Timestamp(s).time() for s in slices)
    mask_win = df_all["_t0"].dt.time.isin(slices_local)

    # A: same date + time-of-day
    A = df_all[(df_all["_date"] == date0.date()) & mask_win]
    gA = gmed(A); gA["src"] = "A"

    # B: same dow + time-of-day
    dow0 = int(pd.Timestamp(date0).dayofweek)
    B = df_all[(df_all["_dow"] == dow0) & mask_win]
    gB = gmed(B); gB["src"] = "B"

    # C: any date but matching times-of-day
    C = df_all[mask_win]
    gC = gmed(C); gC["src"] = "C"

    # D: global median
    D = gmed(df_all); D["src"] = "D"

    base = (outs
            .merge(gA, on=key, how="left")
            .merge(gB, on=key, how="left", suffixes=(None, "_B"))
            .merge(gC, on=key, how="left", suffixes=(None, "_C"))
            .merge(D,  on=key, how="left", suffixes=(None, "_D")))

    vals = base[["val", "val_B", "val_C", "val_D"]].to_numpy(float)
    srcs = base[["src", "src_B", "src_C", "src_D"]].to_numpy(object)
    idx_first = np.argmax(~np.isnan(vals), axis=1)
    row = np.arange(len(base))
    denom = vals[row, idx_first]
    src   = srcs[row, idx_first]

    need_E = np.isnan(denom)
    if reg_params is not None and need_E.any():
        alpha, beta, rmse = reg_params
        sub = base.loc[need_E, key].copy()
        lat_o, lon_o = zip(*[geohash_to_latlon(g) for g in sub[START_COL].astype(str)])
        lat_d, lon_d = zip(*[geohash_to_latlon(g) for g in sub[END_COL].astype(str)])
        dgc = np.array(
            [haversine_m(a, b, c, d) for a, b, c, d in zip(lat_o, lon_o, lat_d, lon_d)],
            dtype=float
        )
        pred = alpha + beta * dgc
        if add_rmse_sigma > 0 and np.isfinite(rmse):
            pred = pred + add_rmse_sigma * rmse
        denom[need_E] = pred
        src[need_E]   = "E"

    outs["D_direct"]     = denom
    outs["baseline_src"] = src
    outs["D_eff"]        = outs["D_elapsed"] / outs["D_direct"]
    return outs

# ========================
# REGRESSION (E)
# ========================
def plot_region_regression(od: pd.DataFrame,
                           alpha: float,
                           beta: float,
                           rmse: float,
                           title: str = "Region regression",
                           show: bool = False,
                           save_path: str = None):
    """
    Plot observed OD distances vs geodesic distance with fitted regression line.

    od must contain:
        - 'd_gc' : geodesic distance
        - 'y'    : observed median distance
        - 'w'    : weights (trip counts)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    x = od["d_gc"].to_numpy(float)
    y = od["y"].to_numpy(float)
    w = od["w"].to_numpy(float)

    # Scatter (size ∝ weight)
    plt.figure(figsize=(6, 5))
    plt.scatter(
        x, y,
        s=10 + 40 * (w / w.max()),
        alpha=0.5,
        edgecolor="none"
    )

    # Regression line
    xx = np.linspace(x.min(), x.max(), 300)
    yy = alpha + beta * xx
    plt.plot(xx, yy, color="black", lw=2,
             label=f"y = {alpha:.1f} + {beta:.3g}·d\nRMSE = {rmse:.1f}")

    plt.xlabel("Geodesic distance d_gc (m)")
    plt.ylabel("Observed median distance (m)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

def fit_region_regression(df_reg: pd.DataFrame, return_od=False) -> Optional[Tuple[float,float,float]]:
    banner("REGRESSION (fallback E)")
    def od_geo(group):
        o = group.name[0]
        d = group.name[1]
        lat_o, lon_o = geohash_to_latlon(o)
        lat_d, lon_d = geohash_to_latlon(d)
        return pd.Series({
            "d_gc": haversine_m(lat_o, lon_o, lat_d, lon_d),
            "y": group[DIST_COL].median(),
            "w": group[COUNT_COL].sum(),
        })

    try:
        od = (df_reg.groupby([START_COL, END_COL])
              .apply(od_geo, include_groups=False)
              .reset_index())
        od = od.replace([np.inf, -np.inf], np.nan).dropna(subset=["d_gc", "y", "w"])
        if od.empty:
            warn("insufficient data for regression")
            return None
        X = od["d_gc"].to_numpy(float)
        y = od["y"].to_numpy(float)
        w = od["w"].to_numpy(float)

        Xmat = np.vstack([np.ones_like(X), X]).T
        try:
            W = np.diag(w / (w.max() if w.max() > 0 else 1.0))
            beta_hat = np.linalg.pinv(Xmat.T @ W @ Xmat) @ (Xmat.T @ W @ y)
        except Exception:
            beta_hat = np.linalg.lstsq(Xmat, y, rcond=None)[0]

        alpha, beta = float(beta_hat[0]), float(beta_hat[1])
        pred = alpha + beta * X
        rmse = float(np.sqrt(np.average((y - pred)**2, weights=w)))

        info(f"fit: α={alpha:.3f}, β={beta:.6f}, σ={rmse:.3f}, n={len(od):,}")
        if return_od:
            return alpha, beta, rmse, od
        else:
            return alpha, beta, rmse
    except Exception as e:
        warn(f"regression failed: {e}")
        return None

# ========================
# GUP PROXY (NOT-IN-D)
# ========================

def detect_gup_pairs_not_in_D(df_region: pd.DataFrame, lscc_nodes: List[str]) -> Set[Tuple[str,str]]:
    """
    GUP proxy: all (o,d) with o!=d in LSCC that never appear with positive
    COUNT_COL in df_region across the span. This is the same 'not-in-D' proxy
    used in the template script. (The full paper's GUP is stricter/time-respecting.)
    """
    pool = df_region.copy()
    pool = pool[pool[START_COL].astype(str).isin(lscc_nodes) &
                pool[END_COL].astype(str).isin(lscc_nodes)]
    present = pool.groupby([START_COL, END_COL], as_index=False)[COUNT_COL].sum()
    present = present[present[COUNT_COL] > 0]
    present[START_COL] = present[START_COL].astype(str)
    present[END_COL]   = present[END_COL].astype(str)
    present_pairs = set(map(tuple, present[[START_COL, END_COL]].to_numpy()))
    all_pairs = set((o, d) for o in lscc_nodes for d in lscc_nodes if o != d)
    return all_pairs - present_pairs

# ========================
# PEP RAW: EMPIRICAL O→D PATHS FROM TRAJECTORIES
# ========================


def load_pep(path: str) -> pd.DataFrame:
    banner("LOAD PEP OD CSV")
    try:
        df = pd.read_csv(path, engine="pyarrow")
    except Exception:
        df = pd.read_csv(path)

    dt = pd.to_datetime(df[DATE_COL].astype(str) + " " + df[TIME_COL].astype(str),
                        errors="coerce")

    df["_t0"] = dt
    df["_t1"] = dt + pd.to_timedelta(TIME_RES_MIN, unit="min")
    df["_dow"] = df["_t0"].dt.dayofweek
    df["_date"] = df["_t0"].dt.date

    info(f"rows = {len(df):,}")
    info(f"time span = {df['_t0'].min()} → {df['_t0'].max()}")
    return df

# ============================================================
# WINDOW
# ============================================================

def subset_window(df: pd.DataFrame,
                  start: str,
                  end: str) -> pd.DataFrame:
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    w = df[(df["_t0"] >= s) & (df["_t0"] < e)].copy()
    info(f"window rows = {len(w):,}  [{start} → {end})")
    return w


# ========================
# MAP HELPERS (pmcm_map-style)
# ========================

def _mercator_xy(lat, lon, R=6378137.0):
    x = R * math.radians(lon)
    latc = max(min(lat, 85.05112878), -85.05112878)
    y = R * math.log(math.tan(math.pi/4.0 + math.radians(latc)/2.0))
    return x, y

def _mercator_inv(x, y, R=6378137.0):
    lat = math.degrees(math.atan(math.sinh(y / R)))
    lon = math.degrees(x / R)
    return lat, lon

def arrow_triangle_points(lat1, lon1, lat2, lon2,
                          min_len_m=ARROW_MIN_LEN_M,
                          max_len_m=ARROW_MAX_LEN_M,
                          frac_of_seg=ARROW_FRAC_OF_SEG,
                          tip_angle_deg=ARROW_TIP_ANGLE):
    """
    Build a small arrowhead triangle at the destination end of the segment,
    using Web Mercator geometry, with tuned scale.
    """
    x1, y1 = _mercator_xy(lat1, lon1)
    x2, y2 = _mercator_xy(lat2, lon2)
    vx, vy = (x2 - x1), (y2 - y1)
    seg = (vx*vx + vy*vy) ** 0.5
    if seg < 1e-6:
        return None
    L = max(min_len_m, min(max_len_m, seg * frac_of_seg))
    ux, uy = vx/seg, vy/seg
    bx, by = x2 - L*ux, y2 - L*uy
    px, py = -uy, ux
    half_base = L * math.tan(math.radians(tip_angle_deg) / 2.0)
    leftx,  lefty  = bx + half_base*px, by + half_base*py
    rightx, righty = bx - half_base*px, by - half_base*py
    tip_lat,   tip_lon   = _mercator_inv(x2,  y2)
    left_lat,  left_lon  = _mercator_inv(leftx,  lefty)
    right_lat, right_lon = _mercator_inv(rightx, righty)
    return [(tip_lat, tip_lon), (left_lat, left_lon), (right_lat, right_lon)]

def add_title_banner(m: folium.Map, text_html: str):
    """
    pmcm_map-style title banner, fixed at top center.
    """
    title_div = folium.Element(f"""
    <div style="
        position: fixed; top: {TITLE_TOP_PX}px; left: 50%; transform: translateX(-50%);
        z-index: 9999; background: rgba(255,255,255,.95);
        padding: 8px 12px; border-radius: 8px; box-shadow: 0 1px 6px rgba(0,0,0,.15);
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, 'Helvetica Neue', Arial, sans-serif;
        font-size: 14px; font-weight: 600; color: #222;">
      {text_html}
    </div>
    """)
    m.get_root().html.add_child(title_div)

def _legend_css(position: str, title_top_px: int = 10) -> str:
    """
    Place legend container depending on corner preference.
    """
    if position == "bottomleft":
        return ".legend{left:10px;right:auto;bottom:28px;top:auto;}"
    if position == "topleft":
        return f".legend{{left:10px;right:auto;top:{max(title_top_px+40,60)}px;bottom:auto;}}"
    if position == "topright":
        return f".legend{{right:10px;left:auto;top:{max(title_top_px+40,60)}px;bottom:auto;}}"
    return ".legend{right:10px;left:auto;bottom:28px;top:auto;}"

def make_effective_distance_map(eff_df: pd.DataFrame,
                                node_meta: Dict[str, Dict[str,Union[float,bool]]],
                                metro_pairs: Set[Tuple[str,str]],
                                html_path: str,
                                percentile_q: float,
                                show_only_gup: bool = False,  # kept for API compat; not used
                                map_title: Optional[str] = None):
    """
    Build pmcm_map-style Folium map:

    - eff_df is assumed to be ALREADY FILTERED (e.g. GUP-only + percentile,
      or non-GUP-only + percentile).
    - We do NOT do any additional filtering here; we just draw what we are given.
    - All LSCC nodes from node_meta are drawn as tiles/hubs.
    - Map bounds are fit to the nodes that actually appear on effective-distance
      edges in eff_df; if none, fall back to all LSCC nodes.
    """
    df = eff_df.copy()
    if df.empty or not np.isfinite(df["D_eff"]).any():
        warn("no D_eff to draw; skipping map")
        return

    # ---------- Identify max-D_eff edge ----------
    max_row = None
    if SHOW_MAX_ED_EDGE:
        df_valid = df[np.isfinite(df["D_eff"])]
        if not df_valid.empty:
            max_row = df_valid.loc[df_valid["D_eff"].idxmax()]

    # ---------- Map center from ALL LSCC nodes ----------
    all_nodes = sorted(node_meta.keys())
    if not all_nodes:
        warn("node_meta empty; cannot build map")
        return

    all_lats = [node_meta[g]["lat"] for g in all_nodes]
    all_lons = [node_meta[g]["lon"] for g in all_nodes]

    center_lat = float(np.mean(all_lats))
    center_lon = float(np.mean(all_lons))
    m = folium.Map(location=(center_lat, center_lon),
                   zoom_start=11,
                   tiles=FOLIUM_TILES)

    # Feature groups
    g_tiles = folium.FeatureGroup(name="Tiles", show=True)
    g_hubs  = folium.FeatureGroup(name="Hubs", show=True)
    g_metro = folium.FeatureGroup(name="Metro edges", show=True)
    g_eff   = folium.FeatureGroup(name="Effective distance edges", show=True)
    g_max = folium.FeatureGroup(name="Max effective distance", show=True)

    # ---------- Nodes: tiles + hubs (ALL LSCC NODES) ----------
    for gh in all_nodes:
        meta = node_meta[gh]
        lat = meta["lat"]
        lon = meta["lon"]
        is_hub = bool(meta.get("is_hub", False))
        if is_hub:
            folium.CircleMarker(
                location=(lat, lon),
                radius=6,
                color="#cc0000",
                weight=2,
                fill=True,
                fill_color="#ff4444",
                fill_opacity=0.9,
                tooltip=f"Hub {gh}",
            ).add_to(g_hubs)
        else:
            folium.CircleMarker(
                location=(lat, lon),
                radius=2,
                color="#666666",
                weight=1,
                fill=True,
                fill_color="#999999",
                fill_opacity=0.7,
                tooltip=f"Tile {gh}",
            ).add_to(g_tiles)

    # ---------- Metro edges (optional layer) ----------
    metro_drawn = 0
    for (o, d) in metro_pairs:
        if (o not in node_meta) or (d not in node_meta):
            continue
        lat_o, lon_o = node_meta[o]["lat"], node_meta[o]["lon"]
        lat_d, lon_d = node_meta[d]["lat"], node_meta[d]["lon"]
        folium.PolyLine(
            [(lat_o, lon_o), (lat_d, lon_d)],
            color="#ff8800",
            weight=4,
            opacity=0.9,
            tooltip=f"Metro {o}→{d}",
        ).add_to(g_metro)
        metro_drawn += 1
    info(f"[MAP] metro edges drawn: {metro_drawn}")

    # ---------- Effective distance edges (the subset eff_df) ----------
    vmin = float(df["D_eff"].min())
    vmax = float(df["D_eff"].max())

    # Clean up vmin/vmax & ticks for legend
    if vmin == vmax:
        vmin -= 0.1
        vmax += 0.1

    # Round ends to keep legend readable
    vmin = float(f"{vmin:.2f}")
    vmax = float(f"{vmax:.2f}")

    n_steps = 6
    ticks = np.linspace(vmin, vmax, n_steps)
    ticks = np.round(ticks, 2)

    cmap = LinearColormap(
        colors=EDGE_CMAP_COLORS,
        vmin=vmin,
        vmax=vmax
    ).to_step(n_steps)

    # numeric tick positions (branca needs floats)
    cmap.index = list(ticks)

    # Track nodes actually used by effective-distance edges (for zoom)
    used_nodes: Set[str] = set()

    # Draw edges as lines + arrowheads
    # Draw edges as lines + arrowheads
    # Ensures high-value edges are drawn last (on top)
    df = df.sort_values("D_eff", ascending=True)   # or sort by "D_eff", depending on layer

    for row in df.itertuples(index=False):

        o = str(getattr(row, START_COL))
        d = str(getattr(row, END_COL))
        if o == d or (o not in node_meta) or (d not in node_meta):
            continue

        val       = float(row.D_eff)
        d_elapsed = float(row.D_elapsed)
        d_direct  = float(row.D_direct)
        is_gup    = bool(getattr(row, "is_gup", False))
        p_hit     = float(getattr(row, "P_hit", np.nan))

        color = cmap(val)
        lat_o, lon_o = node_meta[o]["lat"], node_meta[o]["lon"]
        lat_d, lon_d = node_meta[d]["lat"], node_meta[d]["lon"]

        used_nodes.add(o)
        used_nodes.add(d)

        label_html = (
            f"<b>Origin:</b> {o}<br>"
            f"<b>Destination:</b> {d}<br>"
            f"<b>D_elapsed:</b> {d_elapsed:.1f} m<br>"
            f"<b>D_direct:</b> {d_direct:.1f} m<br>"
            f"<b>D_eff:</b> {val:.3f}<br>"
            f"<b>P_hit:</b> {p_hit:.3e}<br>"
            f"<b>GUP:</b> {is_gup}"
        )

        tooltip_text = (
            f"{o} → {d} | D_eff={val:.3f} | P_hit={p_hit:.1e}"
        )

        # Main line
        line = folium.PolyLine(
            [(lat_o, lon_o), (lat_d, lon_d)],
            color=color,
            weight=EDGE_WEIGHT_PX,
            opacity=0.9,
        )
        line.add_child(folium.Popup(label_html, max_width=260))
        line.add_child(folium.Tooltip(tooltip_text))
        line.add_to(g_eff)

        # Arrowhead polygon
        tri = arrow_triangle_points(
            lat_o, lon_o, lat_d, lon_d,
            min_len_m=ARROW_MIN_LEN_M,
            max_len_m=ARROW_MAX_LEN_M,
            frac_of_seg=ARROW_FRAC_OF_SEG,
            tip_angle_deg=ARROW_TIP_ANGLE,
        )
        if tri is not None:
            poly = folium.Polygon(
                tri,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
            )
            poly.add_child(folium.Popup(label_html, max_width=260))
            poly.add_child(folium.Tooltip(tooltip_text))
            poly.add_to(g_eff)

    # ---------- Highlight MAX D_eff edge ----------
    if SHOW_MAX_ED_EDGE and max_row is not None:

        o = str(getattr(max_row, START_COL))
        d = str(getattr(max_row, END_COL))

        if o in node_meta and d in node_meta:

            lat_o, lon_o = node_meta[o]["lat"], node_meta[o]["lon"]
            lat_d, lon_d = node_meta[d]["lat"], node_meta[d]["lon"]

            val       = float(max_row.D_eff)
            d_elapsed = float(max_row.D_elapsed)
            d_direct  = float(max_row.D_direct)
            p_hit     = float(getattr(max_row, "P_hit", np.nan))
            is_gup    = bool(getattr(max_row, "is_gup", False))

            popup_html = (
                f"<b>{MAX_ED_EDGE_LABEL}</b><br>"
                f"<b>Origin:</b> {o}<br>"
                f"<b>Destination:</b> {d}<br>"
                f"<b>D_elapsed:</b> {d_elapsed:.1f} m<br>"
                f"<b>D_direct:</b> {d_direct:.1f} m<br>"
                f"<b>D_eff:</b> {val:.3f}<br>"
                f"<b>P_hit:</b> {p_hit:.3e}<br>"
                f"<b>GUP:</b> {is_gup}"
            )

            tooltip_text = (
                f"{MAX_ED_EDGE_LABEL}: {o} → {d} | D_eff={val:.3f}"
            )

            # Thick highlighted line
            line = folium.PolyLine(
                [(lat_o, lon_o), (lat_d, lon_d)],
                color=MAX_ED_EDGE_COLOR,
                weight=MAX_ED_EDGE_WEIGHT,
                opacity=MAX_ED_EDGE_OPACITY,
            )
            line.add_child(folium.Popup(popup_html, max_width=280))
            line.add_child(folium.Tooltip(tooltip_text))
            line.add_to(g_max)

            # Arrowhead
            tri = arrow_triangle_points(
                lat_o, lon_o, lat_d, lon_d,
                min_len_m=ARROW_MIN_LEN_M,
                max_len_m=ARROW_MAX_LEN_M,
                frac_of_seg=ARROW_FRAC_OF_SEG,
                tip_angle_deg=ARROW_TIP_ANGLE,
            )
            if tri is not None:
                folium.Polygon(
                    tri,
                    color=MAX_ED_EDGE_COLOR,
                    fill=True,
                    fill_color=MAX_ED_EDGE_COLOR,
                    fill_opacity=MAX_ED_EDGE_OPACITY,
                ).add_to(g_max)

    # ---------- Add layers ----------
    g_tiles.add_to(m)
    # g_hubs.add_to(m)
    # g_metro.add_to(m)
    g_eff.add_to(m)
    g_max.add_to(m)

    # ---------- Legend CSS + colorbar ----------
    m.get_root().html.add_child(
        folium.Element(f"<style>{_legend_css(LEGEND_POSITION, TITLE_TOP_PX)}</style>")
    )

    # Build legend HTML and normalize number formatting inside it
    legend_inner = cmap._repr_html_()

    # Force all decimal numbers in the legend HTML to 2 decimals,
    # so you don't see things like 99.7299999999.
    try:
        import re as _re
        def _fmt_match(mobj):
            s = mobj.group(0)
            try:
                return f"{float(s):.2f}"
            except Exception:
                return s
        legend_inner = _re.sub(r"\d+\.\d+", _fmt_match, legend_inner)
    except Exception:
        # If anything goes wrong, just fall back to the raw legend
        pass

    legend_html = f"""
    <div class="legend" style="position: fixed; z-index: 9998;
        background: rgba(255,255,255,.95); padding: 8px 8px;
        border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.2);
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, 'Helvetica Neue', Arial, sans-serif;
        font-size: 11px;">
      <div style="margin-bottom:4px;"><b>D_eff (effective distance)</b></div>
      {legend_inner}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ---------- D_eff summary box (min/median/max over THIS subset) ----------
    try:
        med = float(df["D_eff"].median())
        extra_legend = folium.Element(f"""
        <div style="
            position: fixed;
            {'left:10px;right:auto;' if LEGEND_POSITION in ('bottomleft','topleft') else 'right:10px;left:auto;'}
            bottom: 5px;
            z-index: 9997;
            background: rgba(255,255,255,.9);
            padding: 4px 8px;
            border-radius: 6px;
            box-shadow: 0 1px 4px rgba(0,0,0,.2);
            font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, 'Helvetica Neue', Arial, sans-serif;
            font-size: 11px;">
          D_eff summary: min={vmin:.2f}, median={med:.2f}, max={vmax:.2f}
        </div>
        """)
        m.get_root().html.add_child(extra_legend)
    except Exception:
        pass

    # ---------- Fit bounds: based on actually drawn edges, fallback to all LSCC ----------
    try:
        if used_nodes:
            lat_list = [node_meta[g]["lat"] for g in used_nodes]
            lon_list = [node_meta[g]["lon"] for g in used_nodes]
        else:
            # No effective-distance edges drawn? use all nodes
            lat_list = all_lats
            lon_list = all_lons

        min_lat, max_lat = min(lat_list), max(lat_list)
        min_lon, max_lon = min(lon_list), max(lon_list)

        lat_span = max_lat - min_lat
        lon_span = max_lon - min_lon
        pad_lat = (lat_span * 0.05) if lat_span > 0 else 0.001
        pad_lon = (lon_span * 0.05) if lon_span > 0 else 0.001

        m.fit_bounds([
            (min_lat - pad_lat, min_lon - pad_lon),
            (max_lat + pad_lat, max_lon + pad_lon),
        ])
    except Exception as e:
        warn(f"fit_bounds failed; using default center. err={e}")

    # ---------- Title + layer control ----------
    if SHOW_MAP_TITLE and map_title:
        add_title_banner(m, map_title)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(html_path)
    info(f"map: {html_path}")

 

# ========================
# MAIN
# ========================

def gup_mode_tag() -> str:
    if USE_GUP_ONLY_EDGES and SHOW_ONLY_GUP_ON_MAP:
        return "gup_compute+map"
    if USE_GUP_ONLY_EDGES:
        return "gup_compute"
    if SHOW_ONLY_GUP_ON_MAP:
        return "gup_map"
    return "all"

def top_percent_from_q(q: float) -> float:
    return max(0.0, 100.0 - float(q))

def main():
    banner("BOOT")
    info(f"cwd={os.getcwd()}")
    info(f"OD input={OD_INPUT_PATH}")
    info("manifest=None (external OD mode)")

    input_stem = os.path.splitext(os.path.basename(OD_INPUT_PATH))[0]
    # Load PEP OD
    df_all = load_pep(OD_INPUT_PATH)
    print(df_all.head())
    # Window subset
    banner("WINDOW SUBSET")
    # ============================================================
    # NORMALIZE *ALL* H3 IDs TO LOWERCASE (CRITICAL FIX)
    # ============================================================
    df_all[START_COL] = df_all[START_COL].astype(str).str.lower()
    df_all[END_COL]   = df_all[END_COL].astype(str).str.lower()

    # ============================================================
    # DEBUG: RAW UNIQUE OD PAIRS IN df_all
    # ============================================================
    # print("\n=== DEBUG RAW df_all OD PAIRS ===")

    # # Groupby to find unique OD with positive flow
    # raw_pos = df_all[df_all[COUNT_COL] > 0].groupby(
    #     [START_COL, END_COL],
    #     as_index=False
    # )[COUNT_COL].sum()

    # raw_pos[START_COL] = raw_pos[START_COL].astype(str)
    # raw_pos[END_COL]   = raw_pos[END_COL].astype(str)

    # raw_pairs = set(map(tuple, raw_pos[[START_COL, END_COL]].to_numpy()))

    # print(f"Raw positive-flow OD pairs   = {len(raw_pairs):,}")

    # # Unique start/dest
    # raw_starts = {o for o, _ in raw_pairs}
    # raw_ends   = {d for _, d in raw_pairs}

    # print(f"Unique start nodes           = {len(raw_starts):,}")
    # print(f"Unique destination nodes     = {len(raw_ends):,}")

    # # Number of all possible OD from these nodes
    # print(f"Possible OD universe         = {len(raw_starts) * (len(raw_starts) - 1):,}")

    # # Sample some pairs
    # print("Sample raw pairs:", list(raw_pairs)[:20])


    # window
    df_win0 = subset_window(df_all, SPAN_START, SPAN_END)
    df_win0[START_COL] = df_win0[START_COL].astype(str)
    df_win0[END_COL]   = df_win0[END_COL].astype(str)

    info(f"window rows={len(df_win0):,}  [{SPAN_START} → {SPAN_END})")
    if df_win0.empty:
        warn("empty window; abort.")
        return

    # LSCC on union graph
    lscc_nodes = compute_lscc_nodes(df_win0)
    if not lscc_nodes:
        warn("LSCC empty; abort.")
        return

    df_win = df_win0[
        df_win0[START_COL].astype(str).isin(lscc_nodes) &
        df_win0[END_COL].astype(str).isin(lscc_nodes)
    ].copy()
    info(f"LSCC-filtered window rows={len(df_win):,}")

    # Regression pool (PEP-based)
    reg_df = (df_all[
        df_all[START_COL].astype(str).isin(lscc_nodes) &
        df_all[END_COL].astype(str).isin(lscc_nodes)
    ].copy() if REGRESSION_ON_LSCC else df_all)

    # Regression parameters
    if REGRESSION_OVERRIDE is not None:
        alpha, beta, rmse = REGRESSION_OVERRIDE
        info(f"using regression override: α={alpha:.3f}, β={beta:.6f}, σ={rmse:.3f}")
        reg_params = REGRESSION_OVERRIDE
    else:
        reg = fit_region_regression(df_all, return_od=True)


    alpha, beta, rmse, od = reg
    reg_params=reg[:3]

    plot_region_regression(
        od,
        alpha,
        beta,
        rmse,
        title=f"Regression — {REGION_NAME}",
        save_path=os.path.join(OUTPUT_DIR, f"{input_stem}_regression_fit.pdf")
    )

    # Slices / packs
    slices, node_index, packs = precompute_slices(df_win, sorted(lscc_nodes))
    nodes = sorted(lscc_nodes)
    N = len(nodes)
    info(f"N={N}, slices={len(slices)}")

    # Origins with any outgoing flow
    pos_union = df_win.groupby([START_COL, END_COL], as_index=False)[COUNT_COL].sum()
    union_out = set(pos_union[START_COL].astype(str))
    reachable_origins = np.array([node_index[o] for o in nodes if o in union_out], dtype=int)
    if reachable_origins.size == 0:
        warn("no reachable origins in union graph; abort.")
        return

    # GUP proxy = not-in-D
    gup_pairs = detect_gup_pairs_not_in_D(reg_df, nodes)
    info(f"[GUP] pairs in LSCC (not-in-D proxy): {len(gup_pairs):,}")

    # Numerators via batched recursion
    banner("COMPUTE (batched elapsed distance, Sec. 4.1)")
    batch_size = resolve_batch_size(N, reachable_origins)
    info(f"batch_size mode={BATCH_SIZE_MODE} → using {batch_size}")
    Xbar, P_hit = batched_elapsed_all_origins(
        packs, N, reachable_origins, batch_size=N
    )


    # Xbar_2d, P_hit_2d = batched_elapsed_all_origins(packs, N, origin_indices, batch_size=some_value)

    # Use a small batch_size here, e.g. 4 or 8
    #X_numpy, P_numpy = batched_elapsed_all_origins_vectorized_numpy(packs, N, reachable_origins, batch_size=N)
    X_numpy, P_numpy = elapsed_all_origins_full3d(packs, N)

    # Simple sanity check
    diff_X = np.nan_to_num(Xbar - X_numpy, nan=0.0)
    diff_P = np.nan_to_num(P_hit - P_numpy, nan=0.0)

    print("max |ΔXbar| =", np.max(np.abs(diff_X)))
    print("max |ΔP_hit| =", np.max(np.abs(diff_P)))

    # Candidate set: all OD in LSCC (exclude self)
    rows = [(o, d) for o in nodes for d in nodes if o != d]
    cand = pd.DataFrame(rows, columns=[START_COL, END_COL]).drop_duplicates()
    i_idx = cand[END_COL].map(node_index)
    j_idx = cand[START_COL].map(node_index)

    i_idx_np = i_idx.to_numpy(int)    # dest indices
    j_idx_np = j_idx.to_numpy(int)    # origin indices

    cand["D_elapsed"] = Xbar[i_idx_np, j_idx_np]
    cand["P_hit"]     = P_hit[i_idx_np, j_idx_np]   # NEW: true Σ_t π_{ij}^t

    info(f"cand rows={len(cand):,}  compute took 0.0s")


    # Optional GUP-only compute
    if USE_GUP_ONLY_EDGES:
        mask_gup = cand.apply(lambda r: (r[START_COL], r[END_COL]) in gup_pairs, axis=1)
        cand_use = cand[mask_gup].copy()
        info(f"USE_GUP_ONLY_EDGES=True → restricting compute to {len(cand_use):,} rows")
    else:
        cand_use = cand

    # Baseline + D_eff
    eff_df = staged_baseline_vectorized(
        df_all=reg_df,
        cand_df=cand_use,
        slices=slices,
        reg_params=reg_params,
        add_rmse_sigma=ADD_RMSE_SIGMA,
        date0=slices[0],
    )

    # Attach is_gup flag
    eff_df["is_gup"] = eff_df.apply(
        lambda r: (str(r[START_COL]), str(r[END_COL])) in gup_pairs, axis=1
    )

    # Node meta + metro pairs from manifest
    node_meta = build_node_meta_from_manifest(None, nodes)
    metro_pairs = set()


    # # Output naming
    # gup_tag = gup_mode_tag()
    # span_tag = f"{SPAN_START.replace(':','').replace(' ','_')}__{SPAN_END.replace(':','').replace(' ','_')}"
    # # q_tag    = f"q{str(PERCENTILE_Q_GUP).replace('.','_')}"
    # # top_pct  = top_percent_from_q(PERCENTILE_Q_GUP)

    # csv_path = os.path.join(
    #     OUTPUT_DIR,
    #     f"eff_{gup_tag}_unfiltered_{span_tag}.csv"
    # )
    # eff_df.to_csv(csv_path, index=False)
    # info(f"wrote: {csv_path}")



    # ------------------------
    # OUTPUT & SUBSETS
    # ------------------------
    span_start_ts = pd.to_datetime(SPAN_START)
    span_end_ts   = pd.to_datetime(SPAN_END)


    # 1) Save full eff_df (all ODs)
    span_tag = f"{span_start_ts.strftime('%Y%m%dT%H%M')}__{span_end_ts.strftime('%Y%m%dT%H%M')}"
    all_csv_path = os.path.join(OUTPUT_DIR, f"{input_stem}_eff_all_{span_tag}.csv")
    eff_df.to_csv(all_csv_path, index=False)
    info(f"wrote full eff_df: {all_csv_path}")

    # Split into GUP / non-GUP subsets
    eff_gup     = eff_df[eff_df["is_gup"] == True].copy()
    eff_non_gup = eff_df[eff_df["is_gup"] == False].copy()
    info(f"GUP subset rows={len(eff_gup):,}, non-GUP subset rows={len(eff_non_gup):,}")

    # Probability gating: only keep ODs with non-negligible hit probability
    eff_gup_prob = eff_gup[eff_gup["P_hit"] >= MIN_P_HIT_GUP].copy()
    eff_non_gup_prob = eff_non_gup[eff_non_gup["P_hit"] >= MIN_P_HIT_NON_GUP].copy()
    info(f"GUP after P_hit≥{MIN_P_HIT_GUP}: {len(eff_gup_prob):,} rows")
    info(f"non-GUP after P_hit≥{MIN_P_HIT_NON_GUP}: {len(eff_non_gup_prob):,} rows")

    # info(f"GUP subset rows={len(eff_gup):,}, non-GUP subset rows={len(eff_non_gup):,}")


    # Save raw subsets
    gup_all_csv = os.path.join(OUTPUT_DIR, f"{input_stem}_eff_gup_all_{span_tag}.csv")
    eff_gup.to_csv(gup_all_csv, index=False)
    info(f"wrote GUP-only eff_df: {gup_all_csv}")

    non_gup_all_csv = os.path.join(OUTPUT_DIR, f"{input_stem}_eff_non_gup_all_{span_tag}.csv")
    eff_non_gup.to_csv(non_gup_all_csv, index=False)
    info(f"wrote non-GUP eff_df: {non_gup_all_csv}")

    # Percentile thresholds within each subset
    # def _top_by_percentile(sub_df: pd.DataFrame,
    #                     label: str,
    #                     q: float) -> pd.DataFrame:
    #     if sub_df.empty or not np.isfinite(sub_df["D_eff"]).any():
    #         warn(f"[{label}] empty or no finite D_eff; returning empty subset.")
    #         return sub_df.iloc[0:0].copy()
    #     thr = float(np.nanpercentile(sub_df["D_eff"].to_numpy(float), q))
    #     info(f"[{label}] q={q}% → thr={thr:.3f}")
    #     return sub_df[sub_df["D_eff"] >= thr].copy()



    # eff_gup_top     = _top_by_percentile(eff_gup,     label="GUP",     q=PERCENTILE_Q_GUP)
    # eff_non_gup_top = _top_by_percentile(eff_non_gup, label="non-GUP", q=PERCENTILE_Q_NON_GUP)

    # Percentile band selection within each subset
    def _band_by_percentile(sub_df: pd.DataFrame,
                            label: str,
                            lower_q: float,
                            upper_q: float) -> pd.DataFrame:
        """
        Return rows of sub_df with D_eff between the lower_q and upper_q
        percentiles (inclusive).
        """
        if sub_df.empty or not np.isfinite(sub_df["D_eff"]).any():
            warn(f"[{label}] empty or no finite D_eff; returning empty subset.")
            return sub_df.iloc[0:0].copy()

        vals = sub_df["D_eff"].to_numpy(float)
        lo_thr = float(np.nanpercentile(vals, lower_q))
        hi_thr = float(np.nanpercentile(vals, upper_q))
        info(f"[{label}] band q=[{lower_q}%, {upper_q}%] → D_eff∈[{lo_thr:.3f}, {hi_thr:.3f}]")

        mask = (sub_df["D_eff"] >= lo_thr) & (sub_df["D_eff"] <= hi_thr)
        return sub_df[mask].copy()



    eff_gup_band = _band_by_percentile(
        eff_gup_prob, label="GUP",
        lower_q=LOWER_PERCENTILE_Q_GUP,
        upper_q=UPPER_PERCENTILE_Q_GUP,
    )
    eff_non_gup_band = _band_by_percentile(
        eff_non_gup_prob, label="non-GUP",
        lower_q=LOWER_PERCENTILE_Q_NON_GUP,
        upper_q=UPPER_PERCENTILE_Q_NON_GUP,
    )

   


    # Save percentile-banded subsets (what maps will actually draw)
    q_tag_gup = f"q{str(LOWER_PERCENTILE_Q_GUP).replace('.','_')}" \
                f"_to_q{str(UPPER_PERCENTILE_Q_GUP).replace('.','_')}"
    q_tag_non_gup = f"q{str(LOWER_PERCENTILE_Q_NON_GUP).replace('.','_')}" \
                    f"_to_q{str(UPPER_PERCENTILE_Q_NON_GUP).replace('.','_')}"

    gup_band_csv = os.path.join(
        OUTPUT_DIR, f"{input_stem}_eff_gup_{q_tag_gup}_{span_tag}.csv"
    )
    eff_gup_band.to_csv(gup_band_csv, index=False)
    info(f"wrote GUP D_eff band subset: {gup_band_csv}")

    non_gup_band_csv = os.path.join(
        OUTPUT_DIR, f"{input_stem}_eff_non_gup_{q_tag_non_gup}_{span_tag}.csv"
    )
    eff_non_gup_band.to_csv(non_gup_band_csv, index=False)
    info(f"wrote non-GUP D_eff band subset: {non_gup_band_csv}")

    # # Save percentile-filtered subsets (what maps will actually draw)
    # q_tag_gup     = f"q{str(PERCENTILE_Q_GUP).replace('.','_')}"
    # q_tag_non_gup = f"q{str(PERCENTILE_Q_NON_GUP).replace('.','_')}"

    # gup_q_csv = os.path.join(OUTPUT_DIR, f"eff_gup_{q_tag_gup}_{span_tag}.csv")
    # eff_gup_top.to_csv(gup_q_csv, index=False)
    # info(f"wrote GUP q≥{PERCENTILE_Q_GUP}% subset: {gup_q_csv}")

    # non_gup_q_csv = os.path.join(OUTPUT_DIR, f"eff_non_gup_{q_tag_non_gup}_{span_tag}.csv")
    # eff_non_gup_top.to_csv(non_gup_q_csv, index=False)
    # info(f"wrote non-GUP q≥{PERCENTILE_Q_NON_GUP}% subset: {non_gup_q_csv}")


    # ------------------------
    # MAPS
    # ------------------------

    # GUP map
    banner("PERCENTILE & MAPS")

    # GUP map
    if not eff_gup_band.empty:
        gup_title_bits = [
            "Effective distance — GUP only",
            REGION_NAME,
            f"{SPAN_START} → {SPAN_END}",
            f"D_eff in [{LOWER_PERCENTILE_Q_GUP:.1f}%, {UPPER_PERCENTILE_Q_GUP:.1f}%] band",
        ]
        gup_title_html = " — ".join(gup_title_bits)

        gup_map_path = os.path.join(
            OUTPUT_DIR,
            f"{input_stem}_map_gup_{q_tag_gup}_{span_tag}.html",
        )
        make_effective_distance_map(
            eff_df=eff_gup_band,
            node_meta=node_meta,
            metro_pairs=metro_pairs,
            html_path=gup_map_path,
            percentile_q=UPPER_PERCENTILE_Q_GUP,  # just for legend text if needed
            show_only_gup=False,
            map_title=gup_title_html if SHOW_MAP_TITLE else None,
        )
    else:
        warn("No GUP edges in percentile band; skipping GUP map.")

    # non-GUP map
    if not eff_non_gup_band.empty:
        non_gup_title_bits = [
            "Effective distance — non-GUP",
            REGION_NAME,
            f"{SPAN_START} → {SPAN_END}",
            f"D_eff in [{LOWER_PERCENTILE_Q_NON_GUP:.1f}%, {UPPER_PERCENTILE_Q_NON_GUP:.1f}%] band",
        ]
        non_gup_title_html = " — ".join(non_gup_title_bits)

        non_gup_map_path = os.path.join(
            OUTPUT_DIR,
            f"{input_stem}_map_non_gup_{q_tag_non_gup}_{span_tag}.html",
        )
        make_effective_distance_map(
            eff_df=eff_non_gup_band,
            node_meta=node_meta,
            metro_pairs=metro_pairs,
            html_path=non_gup_map_path,
            percentile_q=UPPER_PERCENTILE_Q_NON_GUP,
            show_only_gup=False,
            map_title=non_gup_title_html if SHOW_MAP_TITLE else None,
        )
    else:
        warn("No non-GUP edges in percentile band; skipping non-GUP map.")

    if DO_PATH_ENUM:   # or just remove this guard
        csv_path = os.path.join(OUTPUT_DIR, f"{input_stem}_paths_{od_pair[0]}_{od_pair[1]}.csv")
        result = enumerate_first_hitting_paths(
            od_pair=od_pair,
            packs=packs,
            nodes=nodes,
            node_index=node_index,
            write_csv=True,
            csv_path=csv_path
        )

        P_sum     = result["P_sum"]


        print(f"Probability sum = {P_sum:.8f}")

    banner("DONE")


if __name__ == "__main__":
    main()
