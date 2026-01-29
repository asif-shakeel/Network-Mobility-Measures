#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
effective_distance_pep_h3.py
------------------------------------------

H3-native effective-distance pipeline using PEP OD flows.

For all origin–destination pairs in the LSCC of the observed H3 OD graph,
this script computes:

    D_elapsed(o,d)  — expected path length until first hit
    D_direct(o,d)   — geometric regression baseline
    D_eff(o,d)      = D_elapsed / D_direct
    P_hit(o,d)      — total first-hit probability mass

All calculations are fully Markovian and H3-native:
    • no geohash
    • no path enumeration
    • no tensor powers
    • stable batched recursion

----------------------------------------------------------------------
INPUTS
----------------------------------------------------------------------

1) PEP OD (from pep_generator_h3.py)

    {RUN_DIR}/csvs/pep_od.csv

    Required columns:
        start_h3, end_h3, date, time, trip_count, mean_length_m

2) Corridor / grid manifest (from graph_builder_h3.py)

    outputs/corridors_outputs_h3/

        corridors_manifest_{city}_mode-h3_fdr-{fdr}_edge-{edge}_ovr-{ovr}.json
        OR
        grid_manifest_{city}_mode-h3.json

----------------------------------------------------------------------
COMPUTATION
----------------------------------------------------------------------

1) Subset OD to time window [SPAN_START, SPAN_END)
2) Compute LSCC on directed H3 OD graph
3) Build time-sliced Markov packs M(t), D(t)
4) Compute batched first-hit recursion:

        D_elapsed(i,j)
        P_hit(i,j)

5) Fit geometric baseline:

        D_direct = α + β · great_circle_distance

6) Compute:

        D_eff = D_elapsed / D_direct

7) Detect GUP pairs:
        GUP = LSCC × LSCC − raw positive-flow OD

8) Extract extreme percentile bands (default: 99–100%)

9) (Optional) Enumerate exact first-hit paths for a chosen OD

----------------------------------------------------------------------
OUTPUTS
----------------------------------------------------------------------

All outputs are written under:

    {RUN_DIR}/effective_distance/

where RUN_DIR is:

    outputs/runs/{city}/h3res-{res}/graph-{mode}/fdr-{fdr}/edge-{edge}/ovr-{ovr}/{time_window}/m{Δt}/x-{init}

The script produces:

1) Full effective-distance table
    eff_h3_all_{span}.csv

    Columns:
        start_h3, end_h3,
        D_elapsed, D_direct, D_eff, P_hit, is_gup

2) GUP / non-GUP splits
    eff_h3_gup_all_{span}.csv
    eff_h3_non_gup_all_{span}.csv

3) Extreme percentile bands (default 99–100%)
    eff_h3_band_q{LOW_Q}_to_q{HIGH_Q}_{span}.csv
    eff_h3_gup_band_q{LOW_Q}_to_q{HIGH_Q}_{span}.csv
    eff_h3_non_gup_band_q{LOW_Q}_to_q{HIGH_Q}_{span}.csv

4) Interactive maps (Folium HTML)

    map_h3_ed_q{LOW_Q}_to_{HIGH_Q}_{span}.html
    map_h3_gup_q{LOW_Q}_to_{HIGH_Q}_{span}.html
    map_h3_non_gup_q{LOW_Q}_to_{HIGH_Q}_{span}.html

    Each map shows:
        • ED-colored OD edges
        • hubs
        • metro links
        • corridor overlay edges
        • arrowheads for direction
        • highlighted max D_eff edge

5) Overlay endpoint diagnostic map
    map_h3_overlay_max_edge_{span}.html

    Shows only the graph structure with the
    endpoints of the maximum-D_eff OD pair highlighted.

6) Optional first-hit path enumeration (if DO_PATH_ENUM=True)

    paths_{origin_h3}_{dest_h3}.csv

    Columns:
        origin_h3, dest_h3, t,
        path_h3 (H3 sequence),
        probability, distance_m

----------------------------------------------------------------------
KEY FEATURES
----------------------------------------------------------------------

• H3-only (no geohash, no projection artifacts)
• Exact first-hit recursion (no truncation)
• Periodic time-slice model
• Regression-based geometric baseline
• GUP (structurally unreachable) detection
• Extreme-value OD diagnostics
• Full visual diagnostics

----------------------------------------------------------------------
AUTHOR
----------------------------------------------------------------------

Asif Shakeel  
ashakeel@ucsd.edu
"""


import os
import json
import math
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
from scipy import sparse
import networkx as nx
import folium
from branca.colormap import LinearColormap

# ============================================================
# LOGGING
# ============================================================

def banner(msg: str):
    print(f"\n=== {msg} ===", flush=True)

def info(msg: str):
    print(f"[INFO] {msg}", flush=True)

def warn(msg: str):
    print(f"[WARN] {msg}", flush=True)


# ============================================================
# CONFIG — EDIT AS NEEDED
# ============================================================

COUNTRY     = "USA" # "Mexico"
REGION_NAME = "Atlanta, Georgia" # "Mexico City, Mexico"

H3_RESOLUTION = 6
GRAPH_MODE =  "corridors"   # "corridors" | "grid"
NEIGHBOR_TOPOLOGY = "h3"
FEEDER_MODE = "single"
EDGE_DIRECTION_MODE = "potential"
INCLUDE_BACKGROUND_EDGES = False
H3_OVERLAY_FLAG = 0
INIT_X_MODE = "periodic_fixed_point"    #   "periodic_fixed_point"   # or "flat"

BASE_DIR   = "/Users/asif/Documents/nm24"
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
RUNS_DIR   = os.path.join(OUTPUTS_DIR, "runs")

# ============================================================
# MAP HIGHLIGHT OPTIONS
# ============================================================

SHOW_MAX_ED_EDGE = True          # master switch
MAX_ED_EDGE_COLOR = "#ff0000"    # outline color
MAX_ED_EDGE_WEIGHT = 3           # thicker than normal
MAX_ED_EDGE_OPACITY = 1.0
MAX_ED_EDGE_LABEL = "MAX D_eff"


START_COL = "start_h3"
END_COL   = "end_h3"
DATE_COL  = "date"
TIME_COL  = "time"
COUNT_COL = "trip_count"
DIST_COL  = "mean_length_m"

TIME_RES_MIN = 30
SPAN_START   = "2025-06-01 06:00:00"
SPAN_END     = "2025-06-01 08:00:00"

PEP_SPAN_START   = "2025-06-01 00:00:00"
PEP_SPAN_END     = "2025-06-30 00:00:00"


SPAN_START_TS = pd.to_datetime(PEP_SPAN_START)
SPAN_END_TS   = pd.to_datetime(PEP_SPAN_END)



OUTPUT_SUBDIR = "effective_distance"

# Maps: percentile band to show
LOW_Q = 99
HIGH_Q = 100

# Map style
FOLIUM_TILES = "cartodbpositron"
EDGE_WEIGHT_PX = 2
EDGE_CMAP_COLORS = ["magenta", "navy"]

# Arrow geometry
ARROW_MIN_LEN_M = 30
ARROW_MAX_LEN_M = 4000
ARROW_FRAC_OF_SEG = 0.10
ARROW_TIP_ANGLE = 22.0

od_pair= ('8644c1057ffffff',	'8644c10efffffff') #("86499596fffffff", "86499594fffffff")
DO_PATH_ENUM = True

def sanitize_region(r: str) -> str:
    return "".join(ch for ch in r if ch.isalnum())


def edge_tag() -> str:
    return "geom" if EDGE_DIRECTION_MODE=="geometric" else "pot"


def make_h3_run_directory() -> str:
    region_core = REGION_NAME.split(",")[0]
    tag = sanitize_region(region_core)

    span_tag = (
        SPAN_START_TS.strftime("%Y%m%dT%H%M") + "_" +
        SPAN_END_TS.strftime("%Y%m%dT%H%M")
    )

    if GRAPH_MODE == "corridors":
        run_dir = os.path.join(
            OUTPUTS_DIR, "runs", tag,
            f"h3res-{H3_RESOLUTION}",
            f"graph-{GRAPH_MODE}",  
            f"fdr-{FEEDER_MODE}",
            f"edge-{edge_tag()}",
            f"ovr-{H3_OVERLAY_FLAG}",
            span_tag,
            f"m{TIME_RES_MIN}",
            f"x-{INIT_X_MODE}"
        )
    else:
        run_dir = os.path.join(
            OUTPUTS_DIR, "runs", tag,
            f"h3res-{H3_RESOLUTION}",
             f"graph-{GRAPH_MODE}",           
            span_tag,
            f"m{TIME_RES_MIN}",
            f"x-{INIT_X_MODE}"
        )        

    for sub in ["npz","csvs","maps","logs"]:
        os.makedirs(os.path.join(run_dir,sub), exist_ok=True)

    print(f"[INFO] Run directory: {run_dir}")
    return run_dir

RUN_DIR = make_h3_run_directory()

# print(f"[INFO] Loading manifest: {MANIFEST_PATH}")
if GRAPH_MODE == "corridors":
    CORRIDOR_MANIFEST_FILENAME = f"corridors_manifest_{sanitize_region(REGION_NAME.split(',')[0])}_mode-h3_fdr-{FEEDER_MODE}_edge-{edge_tag()}_ovr-{H3_OVERLAY_FLAG}.json"
else:
    CORRIDOR_MANIFEST_FILENAME = f"grid_manifest_{sanitize_region(REGION_NAME.split(',')[0])}_mode-h3.json"


MANIFEST_PATH = os.path.join(OUTPUTS_DIR, "corridors_outputs_h3", CORRIDOR_MANIFEST_FILENAME)

OUTPUT_DIR=os.path.join(RUN_DIR,"effective_distance")

# ============================================================
# LOAD PEP OD
# ============================================================

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


# ============================================================
# LSCC on H3 graph
# ============================================================

def compute_lscc_nodes(df_win: pd.DataFrame) -> List[str]:
    # banner("LSCC COMPUTATION (H3)")

    g = nx.DiGraph()
    nodes = sorted(set(df_win[START_COL]).union(df_win[END_COL]))
    g.add_nodes_from(nodes)

    pos = df_win[df_win[COUNT_COL] > 0]
    edges = pos.groupby([START_COL, END_COL], as_index=False)[COUNT_COL].sum()

    for _, r in edges.iterrows():
        o, d = r[START_COL], r[END_COL]
        if o != d:
            g.add_edge(o, d)

    if g.number_of_edges() == 0:
        warn("LSCC empty")
        return []

    sccs = list(nx.strongly_connected_components(g))
    sccs.sort(key=len, reverse=True)
    lscc = sorted(sccs[0])

    # info(f"LSCC size = {len(lscc)}")
    return lscc


# ============================================================
# BUILD PACKS
# ============================================================

def build_packs(df_win: pd.DataFrame,
                nodes: List[str]):
    # banner("BUILD PACKS (H3)")

    idx = {h: i for i, h in enumerate(nodes)}
    slices = sorted(df_win["_t0"].unique().tolist())
    packs = []

    for s in slices:
        g = df_win[df_win["_t0"] == s]

        agg = g.groupby([START_COL, END_COL], as_index=False).agg(
            trip_sum=(COUNT_COL, "sum"),
            dist_med=(DIST_COL, "median")
        )

        i_idx = []
        j_idx = []
        flows = []
        dvals = []
        outflow = {}

        for _, r in agg.iterrows():
            o, d = r[START_COL], r[END_COL]
            if o not in idx or d not in idx:
                continue
            j = idx[o]
            i = idx[d]
            # if i == j:
            #     continue

            c = float(r["trip_sum"])
            if c <= 0:
                continue

            dist = float(r["dist_med"]) if not pd.isna(r["dist_med"]) else 0.0

            i_idx.append(i)
            j_idx.append(j)
            flows.append(c)
            dvals.append(dist)
            outflow[j] = outflow.get(j, 0.0) + c

        # ensure self-loop if needed
        N = len(nodes)
        for j in range(N):
            if outflow.get(j, 0.0) <= 0.0:
                i_idx.append(j)
                j_idx.append(j)
                flows.append(1.0)
                dvals.append(0.0)

        packs.append({
            "i": np.array(i_idx, dtype=np.int32),
            "j": np.array(j_idx, dtype=np.int32),
            "c": np.array(flows, dtype=np.float64),
            "d": np.array(dvals, dtype=np.float64),
        })

    # info(f"slices = {len(slices)}")
    return slices, idx, packs


# ============================================================
# Mt and Dt
# ============================================================

def build_Mt_Dt_from_pack(p: Dict[str, np.ndarray],
                          N: int):
    i = p["i"]
    j = p["j"]
    c = p["c"]
    d = p["d"]

    M = sparse.coo_matrix((c, (i, j)), shape=(N, N)).tocsr()
    colsum = np.array(M.sum(axis=0)).ravel()
    colsum[colsum == 0] = 1
    M = M @ sparse.diags(1.0 / colsum)

    D = sparse.coo_matrix((d, (i, j)), shape=(N, N)).tocsr()
    return M, D


# ============================================================
# Batched elapsed-distance (stable)
# ============================================================

def batched_elapsed_all_origins(packs, N, origin_indices, batch_size=128):
    """
    Exact H3 version of the original geohash recursion.
    100% faithful: no shortcuts, no approximations, no changes.

    Computes D_elapsed(i,j) and P_hit(i,j) for all i and all origins j.

    Parameters
    ----------
    packs : list of dicts
        Each pack contains:
            'i' : array of destination *node indices*
            'j' : array of origin *node indices*
            'c' : array of flow counts
            'd' : array of distances (meters)
        NOTE: This is identical to geohash version; only the node IDs differ
              upstream. Recursion operates purely on indices.
    N : int
        LSCC size
    origin_indices : iterable of int
        Origins whose columns we compute
    batch_size : unused
        Included for interface compatibility.

    Returns
    -------
    Xbar  : (N,N) array of floats
        Time-elapsed distances
    P_hit : (N,N) array of floats
        Total first-hit probabilities
    """

    # ---------------------------------------------------------------
    # (1) Build per-slice transition matrices M^t and distance D^t
    # ---------------------------------------------------------------
    from scipy import sparse
    import numpy as np

    Ms = []
    Ds = []

    for pack in packs:
        i_idx = pack["i"]        # destination indices
        j_idx = pack["j"]        # origin indices
        c = pack["c"].astype(float)
        d = pack["d"].astype(float)

        # Sparse transition (column-stochastic)
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
    # Prepare output
    # ---------------------------------------------------------------
    Xbar = np.full((N, N), np.nan, float)
    P_hit = np.zeros((N, N), float)

    # ---------------------------------------------------------------
    # (2) Loop over origins j   <— EXACTLY as original
    # ---------------------------------------------------------------
    for j in origin_indices:

        M1 = Ms[0]
        D1 = Ds[0]

        for i in range(N):
            # if i == j:
            #     continue

            # --- t = 1 ---
            p = M1[:, j].toarray().ravel()
            p[j] = 0.0

            y = D1[:, j].toarray().ravel()

            num = 0.0
            den = 0.0

            # First hit at t=1
            pi_1 = p[i]
            if pi_1 > 0.0:
                num += y[i] * pi_1
                den += pi_1

            # --- t = 2 … K ---
            for t in range(1, K):
                Mt = Ms[t]
                Dt = Ds[t]

                # enforce absorbing boundary:
                #   cannot hit i early
                #   cannot return to j
                p_tilde = p.copy()
                p_tilde[i] = 0.0
                p_tilde[j] = 0.0

                # probability update
                p_next = Mt @ p_tilde
                mask = p_next > 0

                # MD = new-step distance
                # MY = carry-forward of old distance
                MD = (Mt.multiply(Dt)) @ p_tilde
                MY = Mt @ (p_tilde * y)

                y_next = np.zeros(N)
                y_next[mask] = (MD[mask] + MY[mask]) / p_next[mask]

                # first-hit probability at t+1
                pi_t = p_next[i]
                if pi_t > 0.0:
                    num += y_next[i] * pi_t
                    den += pi_t

                # step
                p = p_next
                y = y_next

            # --- final D_elapsed(i,j) ---
            if den > 0.0:
                Xbar[i, j] = num / den
                P_hit[i, j] = den

    return Xbar, P_hit



# ============================================================
# BASELINE (staged A–D)
# ============================================================
def staged_baseline_linear(df_all,
                           cand_df: pd.DataFrame,
                           slices: List[pd.Timestamp],
                           reg_params: Tuple[float, float]):
    """
    Hybrid baseline: A/B/C/D then fallback to linear regression:
        D_direct = alpha + beta * direct_dist
    """
    banner("BASELINE (A–E + regression)")

    alpha, beta = reg_params
    key = [START_COL, END_COL]
    outs = cand_df.copy()

    def gmed(df):
        if df.empty:
            return pd.DataFrame(columns=key + ["val"])
        g = df.groupby(key, as_index=False)[DIST_COL].median()
        g = g.rename(columns={DIST_COL: "val"})
        return g

    slice_times = set(pd.Timestamp(s).time() for s in slices)
    mask_slice = df_all["_t0"].dt.time.isin(slice_times)

    date0 = pd.Timestamp(slices[0]).date()
    dow0  = pd.Timestamp(slices[0]).dayofweek

    # A: same date
    A = df_all[(df_all["_date"] == date0) & mask_slice]
    gA = gmed(A); gA["src"] = "A"

    # B: same weekday
    B = df_all[(df_all["_dow"] == dow0) & mask_slice]
    gB = gmed(B); gB["src"] = "B"

    # C: same slices (all dates)
    C = df_all[mask_slice]
    gC = gmed(C); gC["src"] = "C"

    # D: global median
    gD = gmed(df_all); gD["src"] = "D"

    base = outs.merge(gA, on=key, how="left") \
               .merge(gB, on=key, how="left", suffixes=(None, "_B")) \
               .merge(gC, on=key, how="left", suffixes=(None, "_C")) \
               .merge(gD, on=key, how="left", suffixes=(None, "_D"))

    vals = base[["val", "val_B", "val_C", "val_D"]].to_numpy(float)
    srcs = base[["src", "src_B", "src_C", "src_D"]].to_numpy(object)

    idx = np.argmax(~np.isnan(vals), axis=1)
    row = np.arange(len(base))

    denom = vals[row, idx]
    src   = srcs[row, idx]

    # Replace NaN baseline with regression (E)
    needE = np.isnan(denom)

    if np.any(needE):
        # We need direct_dist = DIST_COL baseline candidate
        # (average of all slices in cand_df)
        # But easiest: use raw df_all to get median distances
        direct_merge = df_all.groupby(key, as_index=False)[DIST_COL].median()
        direct_merge = direct_merge.rename(columns={DIST_COL: "direct_med"})
        base2 = outs.merge(direct_merge, on=key, how="left")

        direct_vals = base2["direct_med"].to_numpy(float)

        denom[needE] = alpha + beta * direct_vals[needE]
        src[needE] = "E"

    outs["D_direct"] = denom
    outs["baseline_src"] = src
    outs["D_eff"] = outs["D_elapsed"] / outs["D_direct"]

    return outs

def staged_baseline(df_all,
                    cand_df: pd.DataFrame,
                    slices: List[pd.Timestamp]):
    banner("BASELINE (A–D)")

    key = [START_COL, END_COL]
    outs = cand_df.copy()

    def gmed(df):
        if df.empty:
            return pd.DataFrame(columns=key + ["val"])
        g = df.groupby(key, as_index=False)[DIST_COL].median()
        g = g.rename(columns={DIST_COL: "val"})
        return g

    slice_times = set(pd.Timestamp(s).time() for s in slices)
    mask_slice = df_all["_t0"].dt.time.isin(slice_times)

    date0 = pd.Timestamp(slices[0]).date()

    # A: same date
    A = df_all[(df_all["_date"] == date0) & mask_slice]
    gA = gmed(A); gA["src"] = "A"

    # B: same weekday
    dow0 = pd.Timestamp(slices[0]).dayofweek
    B = df_all[(df_all["_dow"] == dow0) & mask_slice]
    gB = gmed(B); gB["src"] = "B"

    # C: same slices
    C = df_all[mask_slice]
    gC = gmed(C); gC["src"] = "C"

    # D: global
    gD = gmed(df_all); gD["src"] = "D"

    base = outs.merge(gA, on=key, how="left") \
               .merge(gB, on=key, how="left", suffixes=(None, "_B")) \
               .merge(gC, on=key, how="left", suffixes=(None, "_C")) \
               .merge(gD, on=key, how="left", suffixes=(None, "_D"))

    vals = base[["val", "val_B", "val_C", "val_D"]].to_numpy(float)
    srcs = base[["src", "src_B", "src_C", "src_D"]].to_numpy(object)

    idx = np.argmax(~np.isnan(vals), axis=1)
    row = np.arange(len(base))

    denom = vals[row, idx]
    src = srcs[row, idx]

    # fallback (all NaN)
    needE = np.isnan(denom)
    denom[needE] = 1000.0
    src[needE] = "E"

    outs["D_direct"] = denom
    outs["baseline_src"] = src
    outs["D_eff"] = outs["D_elapsed"] / outs["D_direct"]

    return outs


# ============================================================
# Manifest → nodes + hubs + metro
# ============================================================

def load_manifest(path: str):
    if not os.path.exists(path):
        warn(f"manifest not found: {path}")
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        warn(f"cannot read manifest: {e}")
        return None


# ============================================================
# Manifest → nodes + hubs + metro  (FIXED guaranteed-working)
# ============================================================

import h3
import h3.api.basic_str as h3s

def build_node_meta(manifest: dict, nodes: List[str]):
    """
    Build lat/lon + hub flags for each H3 node.
    Always compute coordinates from H3 API (manifest coords not needed).
    """
    hubs = set()
    if manifest:
        hubs = set(manifest.get("hubs", {}).get("h3", {}).keys())


    node_meta = {}
    for h in nodes:
        lat, lon = h3s.cell_to_latlng(h)
        node_meta[h] = {
            "lat":  lat,
            "lon":  lon,
            "is_hub": h in hubs
        }
    return node_meta


def build_metro_pairs(manifest: dict):
    out = set()
    if not manifest:
        return out

    edges = manifest.get("edges", {})
    am = edges.get("AM", [])

    for rec in am:
        try:
            if int(rec.get("is_metro", 0)) == 1:
                o = str(rec.get("start_h3")).lower()
                d = str(rec.get("end_h3")).lower()
                out.add((o, d))
        except Exception:
            continue

    return out

def build_regular_edge_pairs(manifest: dict,
                             include_background: bool = False):
    """
    Extract NON-metro edges from manifest.

    overlay_flag semantics (fixed):
      overlay_flag == 1 → corridor / feeder / overlay edge
      overlay_flag == 0 → background adjacency edge

    Returns:
        overlay_edges, background_edges
        (each is a set of (o,d) directed pairs)
    """
    overlay_edges = set()
    background_edges = set()

    if not manifest:
        return overlay_edges, background_edges

    edges = manifest.get("edges", {})
    am = edges.get("AM", [])

    for rec in am:
        try:
            o = str(rec.get("start_h3")).lower()
            d = str(rec.get("end_h3")).lower()

            if int(rec.get("is_metro", 0)) == 1:
                continue   # skip metro, already handled

            if int(rec.get("overlay_flag", 0)) == 1:
                overlay_edges.add((o, d))
            else:
                background_edges.add((o, d))

        except Exception:
            continue

    if include_background:
        return overlay_edges | background_edges, set()

    return overlay_edges, background_edges

def haversine_m(lat1, lon1, lat2, lon2, R=6371000.0):
    """
    Great-circle distance in meters between (lat1,lon1) and (lat2,lon2).
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def geo_dist_m_from_h3(h1: str, h2: str) -> float:
    """
    Great-circle distance (meters) between H3 cell centroids,
    using your haversine_m().
    """
    lat1, lon1 = h3s.cell_to_latlng(h1)
    lat2, lon2 = h3s.cell_to_latlng(h2)
    return haversine_m(lat1, lon1, lat2, lon2)


# ============================================================
# Arrowheads
# ============================================================

def _mercator_xy(lat, lon, R=6378137.0):
    x = R * math.radians(lon)
    lat_c = max(min(lat, 85.05112878), -85.05112878)
    y = R * math.log(math.tan(math.pi/4 + math.radians(lat_c)/2))
    return x, y

def _mercator_inv(x, y, R=6378137.0):
    lat = math.degrees(math.atan(math.sinh(y / R)))
    lon = math.degrees(x / R)
    return lat, lon

def arrowhead(lat1, lon1, lat2, lon2):
    x1, y1 = _mercator_xy(lat1, lon1)
    x2, y2 = _mercator_xy(lat2, lon2)

    vx = x2 - x1
    vy = y2 - y1
    seg = math.hypot(vx, vy)
    if seg < 1e-6:
        return None

    L = max(ARROW_MIN_LEN_M, min(ARROW_MAX_LEN_M, seg * ARROW_FRAC_OF_SEG))

    ux = vx / seg
    uy = vy / seg

    bx = x2 - L * ux
    by = y2 - L * uy

    px = -uy
    py = ux
    half_base = L * math.tan(math.radians(ARROW_TIP_ANGLE) / 2)

    leftx = bx + half_base * px
    lefty = by + half_base * py
    rightx = bx - half_base * px
    righty = by - half_base * py

    return [
        _mercator_inv(x2, y2),
        _mercator_inv(leftx, lefty),
        _mercator_inv(rightx, righty)
    ]


# ============================================================
# FOLIUM MAP (effective distance edges)
# ============================================================

def make_ed_map(
    eff: pd.DataFrame,
    node_meta: Dict[str, dict],
    metro_pairs,
    overlay_edges,
    background_edges,
    html_path,
    title
):
    # ============================
    # Safety: empty → skip
    # ============================
    if eff.empty:
        warn("no ED edges → empty map")
        return

    # ============================
    # Identify max D_eff edge
    # ============================
    max_row = None
    if SHOW_MAX_ED_EDGE:
        eff_valid = eff[np.isfinite(eff["D_eff"])]
        if not eff_valid.empty:
            max_row = eff_valid.loc[eff_valid["D_eff"].idxmax()]

    max_endpoints = set()
    if max_row is not None:
        max_endpoints = {
            str(getattr(max_row, START_COL)),
            str(getattr(max_row, END_COL)),
        }

    # ============================
    # Compute map center
    # ============================
    lats = [meta["lat"] for meta in node_meta.values()]
    lons = [meta["lon"] for meta in node_meta.values()]
    mlat, mlon = float(np.mean(lats)), float(np.mean(lons))

    m = folium.Map(location=(mlat, mlon), zoom_start=11, tiles=FOLIUM_TILES)

    # ============================
    # Layers
    # ============================
    g_tiles = folium.FeatureGroup("Tiles", show=True)
    g_hubs  = folium.FeatureGroup("Hubs", show=True)
    g_metro = folium.FeatureGroup("Metro", show=True)
    g_eff   = folium.FeatureGroup("Effective distance edges", show=True)
    g_overlay = folium.FeatureGroup("Graph: overlay edges", show=False)
    g_bg      = folium.FeatureGroup("Graph: background edges", show=False)
    g_max = folium.FeatureGroup("Max effective distance", show=True)



    # ============================
    # Tiles + hubs
    # ============================
    for h, meta in node_meta.items():
        lat, lon = meta["lat"], meta["lon"]

        if h in max_endpoints:
            # endpoints of max D_eff edge (distinct from hubs)
            folium.CircleMarker(
                location=(lat, lon),
                radius=7,
                color="#00897b",          # teal
                fill=True,
                fill_color="#00897b",
                fill_opacity=0.95,
                tooltip=f"Max D_eff endpoint: {h}",
            ).add_to(g_tiles)

        elif meta.get("is_hub", False):
            folium.CircleMarker(
                location=(lat, lon),
                radius=6,
                color="#d81b60",          # magenta
                fill=True,
                fill_color="#d81b60",
                fill_opacity=0.9,
                tooltip=f"Hub {h}",
            ).add_to(g_hubs)

        else:
            folium.CircleMarker(
                location=(lat, lon),
                radius=2,
                color="#555",
                fill=True,
                fill_color="#777",
                fill_opacity=0.7,
                tooltip=h,
            ).add_to(g_tiles)

    # ============================
    # Metro edges
    # ============================
    for o, d in metro_pairs:
        if o in node_meta and d in node_meta:
            lo = node_meta[o]
            ld = node_meta[d]
            folium.PolyLine(
                [(lo["lat"], lo["lon"]), (ld["lat"], ld["lon"])],
                color="#ffa700",
                weight=4,
                opacity=0.9
            ).add_to(g_metro)

    # ============================
    # Regular graph edges
    # ============================

    for (o, d) in overlay_edges:
        if o in node_meta and d in node_meta:
            lo, ld = node_meta[o], node_meta[d]
            folium.PolyLine(
                [(lo["lat"], lo["lon"]), (ld["lat"], ld["lon"])],
                color="#bbbbbb",
                weight=1.5,
                opacity=0.6
            ).add_to(g_overlay)

    for (o, d) in background_edges:
        if o in node_meta and d in node_meta:
            lo, ld = node_meta[o], node_meta[d]
            folium.PolyLine(
                [(lo["lat"], lo["lon"]), (ld["lat"], ld["lon"])],
                color="#dddddd",
                weight=1,
                opacity=0.3,
                dash_array="4,4"
            ).add_to(g_bg)

    # ============================
    # Color scale (fix: ensure valid vmin/vmax)
    # ============================
    vmin = float(eff["D_eff"].min())
    vmax = float(eff["D_eff"].max())

    # Round for clean colorbar
    vmin = round(vmin, 2)
    vmax = round(vmax, 2)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = 0.0
        vmax = 1.0


    cmap = LinearColormap(
        colors=EDGE_CMAP_COLORS,
        vmin=vmin,
        vmax=vmax
    )

    # ============================
    # Effective distance edges
    # ============================
    for r in eff.itertuples(index=False):
        o = getattr(r, START_COL)
        d = getattr(r, END_COL)
        if o not in node_meta or d not in node_meta:
            continue

        lo = node_meta[o]
        ld = node_meta[d]
        color = cmap(float(r.D_eff))

        tooltip = (
        f"{o} → {d} | "
        f"D_eff={float(r.D_eff):.3f}, "
        f"P_hit={float(r.P_hit):.3e}"
        )

        popup = (
        f"<b>{o} → {d}</b><br>"
        f"D_elapsed = {float(r.D_elapsed):.1f} m<br>"
        f"D_direct  = {float(r.D_direct):.1f} m<br>"
        f"D_eff     = {float(r.D_eff):.3f}<br>"
        f"P_hit     = {float(r.P_hit):.3e}<br>"
        )


        base = folium.PolyLine(
            [(lo["lat"], lo["lon"]), (ld["lat"], ld["lon"])],
            color=color,
            weight=EDGE_WEIGHT_PX,
            opacity=0.9
        )
        base.add_child(folium.Tooltip(tooltip))
        base.add_child(folium.Popup(popup))
        base.add_to(g_eff)

        # Arrowhead
        tri = arrowhead(lo["lat"], lo["lon"], ld["lat"], ld["lon"])
        if tri:
            poly = folium.Polygon(
                tri,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9
            )
            poly.add_child(folium.Tooltip(tooltip))
            poly.add_child(folium.Popup(popup))
            poly.add_to(g_eff)

    # ============================
    # Highlight max D_eff edge
    # ============================
    if SHOW_MAX_ED_EDGE and max_row is not None:

        o = getattr(max_row, START_COL)
        d = getattr(max_row, END_COL)

        if o in node_meta and d in node_meta:
            lo = node_meta[o]
            ld = node_meta[d]

            tooltip = (
                f"{MAX_ED_EDGE_LABEL}<br>"
                f"{o} → {d}<br>"
                f"D_eff = {float(max_row.D_eff):.3f}"
            )

            popup = (
                f"<b>{MAX_ED_EDGE_LABEL}</b><br>"
                f"{o} → {d}<br>"
                f"D_elapsed = {float(max_row.D_elapsed):.1f} m<br>"
                f"D_direct  = {float(max_row.D_direct):.1f} m<br>"
                f"D_eff     = {float(max_row.D_eff):.3f}<br>"
                f"P_hit     = {float(max_row.P_hit):.3e}"
            )

            # thick highlighted line
            line = folium.PolyLine(
                [(lo["lat"], lo["lon"]), (ld["lat"], ld["lon"])],
                color=MAX_ED_EDGE_COLOR,
                weight=MAX_ED_EDGE_WEIGHT,
                opacity=MAX_ED_EDGE_OPACITY
            )
            line.add_child(folium.Tooltip(tooltip))
            line.add_child(folium.Popup(popup))
            line.add_to(g_max)

            # arrowhead
            tri = arrowhead(lo["lat"], lo["lon"], ld["lat"], ld["lon"])
            if tri:
                folium.Polygon(
                    tri,
                    color=MAX_ED_EDGE_COLOR,
                    fill=True,
                    fill_color=MAX_ED_EDGE_COLOR,
                    fill_opacity=MAX_ED_EDGE_OPACITY
                ).add_to(g_max)

    # Add layers to map
    g_tiles.add_to(m)
    g_hubs.add_to(m)
    g_metro.add_to(m)
    g_eff.add_to(m)
    g_overlay.add_to(m)
    g_bg.add_to(m)
    g_max.add_to(m)

    # ============================
    # Legend
    # ============================
    legend = cmap._repr_html_()
    box = f"""
    <div style="
        position: fixed;
        bottom: 20px; right: 20px;
        z-index: 9999;
        background: rgba(255,255,255,.9);
        padding: 8px 10px;
        border-radius: 6px;
        box-shadow: 0 1px 4px rgba(0,0,0,.3);
    ">
        <b>Effective Distance</b><br>
        {legend}
    </div>
    """
    m.get_root().html.add_child(folium.Element(box))

    # ============================
    # Title
    # ============================
    tbox = f"""
    <div style="
        position: fixed;
        top: 10px; left: 50%;
        transform: translateX(-50%);
        background: rgba(255,255,255,.9);
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        z-index: 9999;
    ">
        {title}
    </div>
    """
    m.get_root().html.add_child(folium.Element(tbox))

    folium.LayerControl(collapsed=False).add_to(m)

    # Save
    m.save(html_path)
    info(f"map saved: {html_path}")


def make_overlay_endpoint_map(
    node_meta: Dict[str, dict],
    overlay_edges,
    background_edges,
    metro_pairs,
    max_edge: Optional[Tuple[str, str]],
    html_path: str,
    title: str,
):
    """
    Overlay-only map highlighting the endpoints of the max D_eff edge.
    No effective-distance coloring.
    """

    # ----------------------------
    # Map center
    # ----------------------------
    lats = [m["lat"] for m in node_meta.values()]
    lons = [m["lon"] for m in node_meta.values()]
    mlat, mlon = float(np.mean(lats)), float(np.mean(lons))

    m = folium.Map(location=(mlat, mlon), zoom_start=11, tiles=FOLIUM_TILES)

    g_nodes = folium.FeatureGroup("Nodes", show=True)
    g_overlay = folium.FeatureGroup("Overlay edges", show=True)
    g_bg = folium.FeatureGroup("Background edges", show=False)
    g_metro = folium.FeatureGroup("Metro", show=True)

    # ----------------------------
    # Endpoint set
    # ----------------------------
    endpoints = set(max_edge) if max_edge else set()

    # ----------------------------
    # Nodes
    # ----------------------------
    for h, meta in node_meta.items():
        lat, lon = meta["lat"], meta["lon"]

        if h in endpoints:
            # highlight endpoints (distinct from hubs)
            folium.CircleMarker(
                location=(lat, lon),
                radius=8,
                color="#00897b",          # teal
                fill=True,
                fill_color="#00897b",
                fill_opacity=0.95,
                tooltip=f"Max D_eff endpoint: {h}",
            ).add_to(g_nodes)
        elif meta.get("is_hub", False):
            folium.CircleMarker(
                location=(lat, lon),
                radius=6,
                color="#d81b60",
                fill=True,
                fill_color="#d81b60",
                fill_opacity=0.9,
                tooltip=f"Hub {h}",
            ).add_to(g_nodes)

        else:
            folium.CircleMarker(
                location=(lat, lon),
                radius=2,
                color="#666",
                fill=True,
                fill_color="#888",
                fill_opacity=0.6,
                tooltip=h,
            ).add_to(g_nodes)

    # ----------------------------
    # Overlay edges
    # ----------------------------
    for o, d in overlay_edges:
        if o in node_meta and d in node_meta:
            lo, ld = node_meta[o], node_meta[d]
            folium.PolyLine(
                [(lo["lat"], lo["lon"]), (ld["lat"], ld["lon"])],
                color="#888",
                weight=2,
                opacity=0.7,
            ).add_to(g_overlay)

    # ----------------------------
    # Background edges
    # ----------------------------
    for o, d in background_edges:
        if o in node_meta and d in node_meta:
            lo, ld = node_meta[o], node_meta[d]
            folium.PolyLine(
                [(lo["lat"], lo["lon"]), (ld["lat"], ld["lon"])],
                color="#cccccc",
                weight=1,
                opacity=0.3,
                dash_array="4,4",
            ).add_to(g_bg)

    # ----------------------------
    # Metro
    # ----------------------------
    for o, d in metro_pairs:
        if o in node_meta and d in node_meta:
            lo, ld = node_meta[o], node_meta[d]
            folium.PolyLine(
                [(lo["lat"], lo["lon"]), (ld["lat"], ld["lon"])],
                color="#ffa700",
                weight=4,
                opacity=0.9,
            ).add_to(g_metro)

    # ----------------------------
    # Title
    # ----------------------------
    tbox = f"""
    <div style="
        position: fixed;
        top: 10px; left: 50%;
        transform: translateX(-50%);
        background: rgba(255,255,255,.9);
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        z-index: 9999;
    ">
        {title}
    </div>
    """
    m.get_root().html.add_child(folium.Element(tbox))

    g_nodes.add_to(m)
    g_overlay.add_to(m)
    g_bg.add_to(m)
    g_metro.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(html_path)

    info(f"overlay endpoint map saved: {html_path}")


def enumerate_first_hitting_paths_h3(
    od_pair: Tuple[str, str],
    packs: List[Dict[str, np.ndarray]],
    nodes: List[str],
    node_index: Dict[str, int],
    write_csv: bool = True,
    prob_epsilon: float = 0.0,     # prune truly-zero transitions (recommended)
    frontier_max: int = 2_000_000, # safety cap (never triggers in real data)
):
    """
    Correct, faithful H3 version of enumerate_first_hitting_paths,
    with:
      • ZERO changes to enumeration logic
      • progress printing
      • pruning of only zero-probability transitions
      • diagonals allowed (critical for RTO)
    """

    import csv

    origin_h3, dest_h3 = od_pair

    if origin_h3 not in node_index or dest_h3 not in node_index:
        warn(f"[H3 enum] OD {origin_h3}->{dest_h3} not in LSCC; skipping.")
        return {
            "origin": origin_h3,
            "destination": dest_h3,
            "P_sum": 0.0,
            "paths": [],
            "paths_geo": [],
            "K": 0,
        }

    j = node_index[origin_h3]
    i = node_index[dest_h3]
    N = len(nodes)

    # Build Mt, Dt
    Ms, Ds = zip(*(build_Mt_Dt_from_pack(p, N) for p in packs))
    K = len(Ms)

    print(f"[ENUM] {origin_h3}→{dest_h3}")
    print(f"[ENUM] slices={K}")
    print("[ENUM] starting enumeration...")

    final_paths = []
    P_sum = 0.0

    # ------------------------------------------------
    # t = 1
    # ------------------------------------------------
    M1, D1 = Ms[0], Ds[0]
    col_j = M1[:, j].toarray().ravel()

    frontier = []

    hits_t = 0
    for u in range(N):
        p_ju = col_j[u]
        if p_ju <= prob_epsilon or u == j:      # prune zero-prob transitions only
            continue

        dist_ju = float(D1[u, j])

        if u == i:
            # first-hit path of length 1
            final_paths.append({
                "path": [j, i],
                "prob": p_ju,
                "dist": dist_ju,
                "t": 1,
            })
            P_sum += p_ju
            hits_t += 1
        else:
            frontier.append({
                "node": u,
                "path": [j, u],
                "prob": p_ju,
                "dist": dist_ju,
            })

    print(f"[ENUM t=1] frontier={len(frontier)}, hits={hits_t}, P_sum={P_sum:.6e}")

    # ------------------------------------------------
    # t = 2..K
    # ------------------------------------------------
    for t in range(2, K + 1):
        Mt, Dt = Ms[t - 1], Ds[t - 1]
        new_frontier = []
        hits_t = 0

        for entry in frontier:
            u = entry["node"]

            if entry["prob"] <= prob_epsilon:  # skip dead branches
                continue

            col_u = Mt[:, u].toarray().ravel()

            for v in range(N):
                p_uv = col_u[v]
                if p_uv <= prob_epsilon or  (v==j and  i!=j):  # prune exactly zero transitions
                    continue

                new_prob = entry["prob"] * p_uv
                new_dist = entry["dist"] + float(Dt[v, u])

                if v == i:
                    # We have a first-hit at time t
                    final_paths.append({
                        "path": entry["path"] + [i],
                        "prob": new_prob,
                        "dist": new_dist,
                        "t": t,
                    })
                    P_sum += new_prob
                    hits_t += 1
                else:
                    new_frontier.append({
                        "node": v,
                        "path": entry["path"] + [v],
                        "prob": new_prob,
                        "dist": new_dist,
                    })

        frontier = new_frontier

        print(f"[ENUM t={t}] frontier={len(frontier)}, hits={hits_t}, P_sum={P_sum:.6e}")

        # Safety valve (will basically never be hit)
        if len(frontier) > frontier_max:
            warn(f"[ENUM] frontier exceeded limit ({frontier_max}); stopping early.")
            break

        if not frontier:
            break

    # ------------------------------------------------
    # Convert raw index paths → H3 strings
    # ------------------------------------------------
    paths_geo = []
    for p in final_paths:
        g = [nodes[idx] for idx in p["path"]]
        q = dict(p)
        q["path_geo"] = g
        paths_geo.append(q)

    print(f"\n[ENUM RESULT] Paths={len(paths_geo)}, P_sum={P_sum:.6f}")

    # ------------------------------------------------
    # Optional CSV output
    # ------------------------------------------------
    if write_csv and len(paths_geo) > 0:
        out_csv = os.path.join(
            OUTPUT_DIR,
            f"paths_{origin_h3}_{dest_h3}.csv"
        )
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "origin_h3", "dest_h3", "t",
                "path_h3", "probability", "distance_m"
            ])
            for p in paths_geo:
                w.writerow([
                    origin_h3,
                    dest_h3,
                    p["t"],
                    " -> ".join(p["path_geo"]),
                    p["prob"],
                    p["dist"],
                ])
        info(f"[ENUM wrote CSV] {out_csv}")

    return {
        "origin": origin_h3,
        "destination": dest_h3,
        "P_sum": P_sum,
        "paths": final_paths,
        "paths_geo": paths_geo,
        "K": K,
    }
# ========================
# GUP PROXY (NOT-IN-D) — H3
# ========================
# ========================
# TRUE GUP DETECTOR (H3)
# ========================
def detect_gup_pairs(df_raw: pd.DataFrame,
                     lscc_nodes: List[str]) -> Set[Tuple[str,str]]:

    # Raw single-hop edges
    raw_pos = df_raw[df_raw[COUNT_COL] > 0]
    raw_pos = raw_pos[
        raw_pos[START_COL].isin(lscc_nodes) &
        raw_pos[END_COL].isin(lscc_nodes)
    ]
    raw_pairs = set(
        (str(o), str(d))
        for o, d in raw_pos[[START_COL, END_COL]].to_numpy()
        if o != d
    )

    # Full LSCC OD universe
    all_pairs = set(
        (o, d)
        for o in lscc_nodes
        for d in lscc_nodes
        if o != d
    )

    # GUP = universe − raw
    return all_pairs - raw_pairs

def fit_region_regression(df: pd.DataFrame):
    """
    Fit:  mean_length_m ≈ α + β * great_circle_distance_m
    using haversine distance between H3 centroids.
    """
    rows = []
    for r in df.itertuples(index=False):
        o = getattr(r, START_COL)
        d = getattr(r, END_COL)
        m = float(getattr(r, DIST_COL))  # mean_length_m

        if m <= 0 or o == d:
            continue

        geo_m = geo_dist_m_from_h3(o, d)
        if not np.isfinite(geo_m) or geo_m <= 0:
            continue

        rows.append((geo_m, m))

    if not rows:
        raise RuntimeError("Regression pool empty.")

    X = np.array([r[0] for r in rows], dtype=float)
    Y = np.array([r[1] for r in rows], dtype=float)

    A = np.vstack([np.ones_like(X), X]).T
    theta, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    alpha, beta = theta

    pred = alpha + beta * X
    rmse = np.sqrt(np.mean((Y - pred)**2))

    info(f"[REG] alpha={alpha:.3f}, beta={beta:.6f}, rmse={rmse:.2f}")
    return alpha, beta, rmse


def predict_direct_distance(o: str, d: str, alpha: float, beta: float) -> float:
    """
    Predict D_direct (meters) = alpha + beta * great_circle_distance(o,d)
    """
    geo_m = geo_dist_m_from_h3(o, d)
    return alpha + beta * geo_m

# ============================================================
# MAIN
# ============================================================

def main():
    banner("H3 EFFECTIVE DISTANCE PIPELINE")

    # paths
    od_path = os.path.join(RUN_DIR,
                           "csvs/pep_od.csv")

    output_dir = os.path.join(RUN_DIR,
                              OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)

    info(f"OD input = {od_path}")
    info(f"manifest = {MANIFEST_PATH}")
    info(f"output_dir = {output_dir}")

    # load OD
    df_all = load_pep(od_path)

    # ============================================================
    # NORMALIZE *ALL* H3 IDs TO LOWERCASE (CRITICAL FIX)
    # ============================================================
    df_all[START_COL] = df_all[START_COL].astype(str).str.lower()
    df_all[END_COL]   = df_all[END_COL].astype(str).str.lower()



    # window
    df_win0 = subset_window(df_all, SPAN_START, SPAN_END)
    df_win0[START_COL] = df_win0[START_COL].astype(str)
    df_win0[END_COL]   = df_win0[END_COL].astype(str)

    if df_win0.empty:
        warn("empty window — abort")
        return

    # LSCC
    lscc_nodes = compute_lscc_nodes(df_win0)
    lscc_nodes = [h.lower() for h in lscc_nodes]

    if not lscc_nodes:
        warn("LSCC empty — abort")
        return

    df_win = df_win0[df_win0[START_COL].isin(lscc_nodes) &
                     df_win0[END_COL].isin(lscc_nodes)].copy()

    df_win[START_COL] = df_win[START_COL].astype(str)
    df_win[END_COL]   = df_win[END_COL].astype(str)

    nodes = sorted([h.lower() for h in lscc_nodes])
    node_index = {h: i for i, h in enumerate(nodes)}
    # print(node_index)
    info(f"LSCC-filtered rows = {len(df_win):,}")

    # packs
    slices, node_index, packs = build_packs(df_win, nodes)
    N = len(nodes)
    info(f"N={N}, slices={len(slices)}")

    # reachable origins
    union = df_win.groupby([START_COL], as_index=False)[COUNT_COL].sum()
    reachable_origins = np.array(
        [node_index[o] for o in union[START_COL].astype(str) if o in node_index],
        dtype=int
    )
    if reachable_origins.size == 0:
        warn("no origins have outflow")
        return

    # elapsed
    X_elapsed, P_hit = batched_elapsed_all_origins(packs, N, reachable_origins)

    # FULL LSCC OD UNIVERSE (necessary for GUP detection)
    rows = [(o, d) for o in nodes for d in nodes if o != d]
    cand = pd.DataFrame(rows, columns=[START_COL, END_COL])

    cand[START_COL] = cand[START_COL].astype(str).str.lower()
    cand[END_COL]   = cand[END_COL].astype(str).str.lower()

    i_idx = cand[END_COL].map(node_index).to_numpy(int)
    j_idx = cand[START_COL].map(node_index).to_numpy(int)

    cand["D_elapsed"] = X_elapsed[i_idx, j_idx]
    cand["P_hit"]     = P_hit[i_idx, j_idx]




    # Regression
    alpha, beta, rmse = fit_region_regression(df_all)

    rows = []
    for o in nodes:
        for d in nodes:
            if o == d:
                continue
            i_idx = node_index[d]
            j_idx = node_index[o]

            d_elapsed = float(X_elapsed[i_idx, j_idx])
            p_hit = float(P_hit[i_idx, j_idx])
            if not np.isfinite(d_elapsed):
                continue

            d_direct = predict_direct_distance(o, d, alpha, beta)
            d_eff = d_elapsed / d_direct if d_direct > 0 else np.nan

            rows.append({
                START_COL: o,
                END_COL: d,
                "D_elapsed": d_elapsed,
                "D_direct": d_direct,
                "D_eff": d_eff,
                "P_hit": p_hit
            })

    eff_df = pd.DataFrame(rows)

    # Debug print ONLY for the OD pairs requested
    print("\n=== DEBUG D_eff for requested OD pairs ===")
    for (o, d) in [od_pair]:
        row = eff_df[
            (eff_df[START_COL] == o) &
            (eff_df[END_COL] == d)
        ]
        if row.empty:
            print(f"[DEBUG] {o} → {d} : NOT FOUND in eff_df")
            continue

        r = row.iloc[0]
        print(
            f"[DEBUG] {o} → {d} :  "
            f"D_elapsed={r['D_elapsed']:.3f} m,  "
            f"D_direct={r['D_direct']:.3f} m,  "
            f"D_eff={r['D_eff']:.6f},  "
            f"P_hit={r['P_hit']:.3e}"
        )


    # ------------------------------------------------------------
    # GUP classification (NOT-IN-D proxy)
    # ------------------------------------------------------------
    gup_pairs = detect_gup_pairs(df_all, nodes)
    info(f"GUP count = {len(gup_pairs)}")



    eff_df["is_gup"] = eff_df.apply(
        lambda r: (str(r[START_COL]), str(r[END_COL])) in gup_pairs,
        axis=1
    )

    # save full
    span_tag = f"{SPAN_START.replace(':','')}_{SPAN_END.replace(':','')}"
    out_csv = os.path.join(output_dir, f"eff_h3_all_{span_tag}.csv")
    eff_df.to_csv(out_csv, index=False)
    info(f"wrote: {out_csv}")

    # ------------------------------------------------------------
    # Split eff_df into GUP / non-GUP
    # ------------------------------------------------------------
    eff_gup     = eff_df[eff_df["is_gup"] == True].copy()
    eff_non_gup = eff_df[eff_df["is_gup"] == False].copy()

    gup_all_csv = os.path.join(output_dir, f"eff_h3_gup_all_{span_tag}.csv")
    eff_gup.to_csv(gup_all_csv, index=False)
    info(f"wrote: {gup_all_csv}")

    non_gup_all_csv = os.path.join(output_dir, f"eff_h3_non_gup_all_{span_tag}.csv")
    eff_non_gup.to_csv(non_gup_all_csv, index=False)
    info(f"wrote: {non_gup_all_csv}")


    # percentile band
    vals = eff_df["D_eff"].to_numpy(float)
    lo = np.nanpercentile(vals, LOW_Q)
    hi = np.nanpercentile(vals, HIGH_Q)
    band = eff_df[(eff_df["D_eff"] >= lo) & (eff_df["D_eff"] <= hi)].copy()

    band_csv = os.path.join(output_dir,
                            f"eff_h3_band_q{LOW_Q}_to_q{HIGH_Q}_{span_tag}.csv")
    band.to_csv(band_csv, index=False)
    info(f"wrote: {band_csv}")


    # ------------------------------------------------------------
    # Percentile band for GUP
    # ------------------------------------------------------------
    if not eff_gup.empty:
        vals_g = eff_gup["D_eff"].to_numpy(float)
        lo_g = np.nanpercentile(vals_g, LOW_Q)
        hi_g = np.nanpercentile(vals_g, HIGH_Q)
        band_gup = eff_gup[(eff_gup["D_eff"] >= lo_g) & (eff_gup["D_eff"] <= hi_g)].copy()
    else:
        band_gup = eff_gup.iloc[0:0].copy()

    gup_band_csv = os.path.join(
        output_dir, f"eff_h3_gup_band_q{LOW_Q}_to_q{HIGH_Q}_{span_tag}.csv"
    )
    band_gup.to_csv(gup_band_csv, index=False)
    info(f"wrote: {gup_band_csv}")

    # ------------------------------------------------------------
    # Percentile band for NON-GUP
    # ------------------------------------------------------------
    if not eff_non_gup.empty:
        vals_ng = eff_non_gup["D_eff"].to_numpy(float)
        lo_ng = np.nanpercentile(vals_ng, LOW_Q)
        hi_ng = np.nanpercentile(vals_ng, HIGH_Q)
        band_non_gup = eff_non_gup[
            (eff_non_gup["D_eff"] >= lo_ng) & (eff_non_gup["D_eff"] <= hi_ng)
        ].copy()
    else:
        band_non_gup = eff_non_gup.iloc[0:0].copy()

    non_gup_band_csv = os.path.join(
        output_dir, f"eff_h3_non_gup_band_q{LOW_Q}_to_q{HIGH_Q}_{span_tag}.csv"
    )
    band_non_gup.to_csv(non_gup_band_csv, index=False)
    info(f"wrote: {non_gup_band_csv}")

    # ================================
    # Manifest / node metadata / metro
    # ================================
    manifest = load_manifest(MANIFEST_PATH)
    node_meta = build_node_meta(manifest, nodes)
    metro_pairs = build_metro_pairs(manifest)

    overlay_edges, background_edges = build_regular_edge_pairs(
        manifest,
        include_background=INCLUDE_BACKGROUND_EDGES
    )

    # normalize
    overlay_edges = {(o.lower(), d.lower()) for (o, d) in overlay_edges}
    background_edges = {(o.lower(), d.lower()) for (o, d) in background_edges}

    # --- KEY FIX: normalize to lowercase ---
    normalized_meta = {k.lower(): v for k, v in node_meta.items()}
    node_meta = normalized_meta

    metro_pairs = {(o.lower(), d.lower()) for (o, d) in metro_pairs}

    # ================================
    # Render map
    # ================================
    map_path = os.path.join(
        output_dir,
        f"map_h3_ed_q{LOW_Q}_to_{HIGH_Q}_{span_tag}.html"
    )

    make_ed_map(
        band,
        node_meta,
        metro_pairs,
        overlay_edges,
        background_edges,
        map_path,
        f"Effective Distance ({LOW_Q}–{HIGH_Q}%)"
        f"{SPAN_START} → {SPAN_END}<br>"
    )

    # ------------------------------------------------------------
    # GUP map
    # ------------------------------------------------------------
    if not band_gup.empty:
        gup_map_path = os.path.join(
            output_dir,
            f"map_h3_gup_q{LOW_Q}_to_{HIGH_Q}_{span_tag}.html"
        )
        make_ed_map(
            band_gup,
            node_meta,
            metro_pairs,
            overlay_edges,
            background_edges,
            gup_map_path,
        f"Effective Distance - GUP({LOW_Q}–{HIGH_Q}%) - "
        f"{SPAN_START} → {SPAN_END}<br>"
        )

    # ------------------------------------------------------------
    # non-GUP map
    # ------------------------------------------------------------
    if not band_non_gup.empty:
        non_gup_map_path = os.path.join(
            output_dir,
            f"map_h3_non_gup_q{LOW_Q}_to_{HIGH_Q}_{span_tag}.html"
        )
        make_ed_map(
            band_non_gup,
            node_meta,
            metro_pairs,
            overlay_edges,
            background_edges,
            non_gup_map_path,
            f"H3 Effective Distance — NON-GUP ({LOW_Q}–{HIGH_Q}%)"
        )

    print("Rows in eff_df:", len(eff_df))
    print("Finite D_eff:", np.isfinite(eff_df["D_eff"]).sum())
    print("Min D_eff:", eff_df["D_eff"].min())
    print("Max D_eff:", eff_df["D_eff"].max())

    print("P_hit > 0:", (eff_df["P_hit"] > 0).sum())
    print("P_hit >= 1e-6:", (eff_df["P_hit"] >= 1e-6).sum())


    if DO_PATH_ENUM:   # or just remove this guard
        result = enumerate_first_hitting_paths_h3(
            od_pair=od_pair,
            packs=packs,
            nodes=nodes,
            node_index=node_index,
            write_csv=True
        )
    banner("DONE")


    max_row = None
    band_valid = band[np.isfinite(band["D_eff"])]
    if not band_valid.empty:
        max_row = band_valid.loc[band_valid["D_eff"].idxmax()]

    max_edge = None
    if max_row is not None:
        max_edge = (
            str(max_row[START_COL]),
            str(max_row[END_COL])
        )

    if max_edge is not None:
        overlay_map_path = os.path.join(
            output_dir,
            f"map_h3_overlay_max_edge_{span_tag}.html"
        )

        make_overlay_endpoint_map(
            node_meta=node_meta,
            overlay_edges=overlay_edges,
            background_edges=background_edges,
            metro_pairs=metro_pairs,
            max_edge=max_edge,
            html_path=overlay_map_path,
            title=(
                "Graph Overlay — Max D_eff Endpoints<br>"
                # f"{max_edge[0]} → {max_edge[1]}"
            ),
        )

if __name__ == "__main__":
    main()
