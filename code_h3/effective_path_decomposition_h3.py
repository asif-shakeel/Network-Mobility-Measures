#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
effective_path_decomposition_h3.py
------------------------------------------

H3-native effective path decomposition and visualization.

This script explains *why* a large effective distance occurs by
enumerating and visualizing the dominant first-hitting paths that
contribute to D_elapsed(o,d).

It combines:

    • D_elapsed(o,d), P_hit(o,d)   from effective_distance_pep_h3
    • Geometric regression        D_direct = α + β·geo_dist
    • Exact first-hit path enumeration via:
            enumerate_first_hitting_paths_h3()

For selected OD pairs, the script decomposes the effective distance
into its most probable contributing trajectories and renders them
on a layered interactive map.

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

3) Effective-distance engine
    effective_distance_pep_h3.py
    (imported as h3ed)

----------------------------------------------------------------------
COMPUTATION
----------------------------------------------------------------------

1) Subset OD to time window [SPAN_START, SPAN_END)
2) Compute LSCC of H3 OD graph
3) Build time-sliced Markov packs M(t), D(t)
4) Compute:

        D_elapsed(i,j)
        P_hit(i,j)

5) Fit regression:

        D_direct = α + β · geo_dist(o,d)

6) For each OD in OD_PAIRS:

    a) Compute:
           D_elapsed, D_direct, D_eff, P_hit

    b) Enumerate *all* first-hitting paths:
           enumerate_first_hitting_paths_h3()

    c) Rank paths by:
           • probability (default), or
           • distance

    d) Keep top-K (MAX_PATHS_PER_OD)

    e) Identify:
           • endpoints
           • intermediate bottleneck nodes

----------------------------------------------------------------------
OUTPUTS
----------------------------------------------------------------------

All outputs are written to:

    {RUN_DIR}/effective_path_decomposition/

1) Interactive Folium map

    path_decomposition_h3__{SPAN_START}_to_{SPAN_END}.html

    The map contains toggleable layers:

        • All H3 tiles
        • Hubs (from manifest)
        • Metro links
        • Corridor / feeder overlay edges
        • Background adjacency edges
        • OD edges (thick, directed)
        • Top-K first-hitting paths (dashed, colored)
        • Arrowheads showing direction

    Each OD edge popup shows:
        D_elapsed, D_direct, D_eff, P_hit

    Each path tooltip shows:
        path probability, total distance, number of steps.

2) Path CSVs (one per OD, map-consistent, top-K only)

    paths_{origin_h3}_{dest_h3}.csv

    Columns:
        origin, destination,
        rank,
        probability,
        distance_m,
        steps,
        path   (H3 → H3 → … → H3)

----------------------------------------------------------------------
KEY FEATURES
----------------------------------------------------------------------

• Fully H3-native (no geohash, no reprojection)
• Exact first-hit path enumeration
• Consistent with effective_distance_pep_h3
• Top-K path filtering
• Structural bottleneck discovery
• Visual explanation of extreme D_eff edges

----------------------------------------------------------------------
PIPELINE CONTEXT
----------------------------------------------------------------------

1) graph_builder_h3.py
       → synthetic corridor network

2) pep_generator_h3.py
       → synthetic OD flows + Markov kernels

3) effective_distance_pep_h3.py
       → D_elapsed, D_direct, D_eff, GUPs, maps

4) effective_path_decomposition_h3.py
       → interpretable path-level explanation

----------------------------------------------------------------------
AUTHOR
----------------------------------------------------------------------

Asif Shakeel  
ashakeel@ucsd.edu
"""


import os
import time
import math
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import folium

# ============================================================
# Import working ED engine
# ============================================================
import effective_distance_pep_h3 as h3ed


# ============================================================
# USER CONFIG
# ============================================================
COUNTRY     = "USA" # "Mexico"
REGION_NAME = "Atlanta, Georgia" # "Mexico City, Mexico"
H3_RESOLUTION = 6
NEIGHBOR_TOPOLOGY = "h3"
GRAPH_MODE =     "corridors"   # "corridors" | "grid"
FEEDER_MODE = "single"
EDGE_DIRECTION_MODE = "potential"
INCLUDE_BACKGROUND_EDGES = False
H3_OVERLAY_FLAG = 0
INIT_X_MODE = "periodic_fixed_point"    #   "periodic_fixed_point"   # or "flat"

BASE_DIR   = "/Users/asif/Documents/nm24"
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
RUNS_DIR   = os.path.join(OUTPUTS_DIR, "runs")



START_COL = "start_h3"
END_COL   = "end_h3"
DATE_COL  = "date"
TIME_COL  = "time"
COUNT_COL = "trip_count"
DIST_COL  = "mean_length_m"

TIME_RES_MIN = 30

# Time window for path-decomposition
SPAN_START = "2025-06-01 06:00:00"
SPAN_END   = "2025-06-01 08:00:00"

PEP_SPAN_START   = "2025-06-01 00:00:00"
PEP_SPAN_END     = "2025-06-30 00:00:00"


SPAN_START_TS = pd.to_datetime(PEP_SPAN_START)
SPAN_END_TS   = pd.to_datetime(PEP_SPAN_END)

# ============================================================
# PATH FILTERING CONFIG
# ============================================================

MAX_PATHS_PER_OD = 8        # K
PATH_RANK_BY = "prob"       # "prob" | "distance"

# Node colors (CONSISTENT WITH ED MAPS)
ENDPOINT_NODE_COLOR = "#00897b"      # teal
INTERMEDIATE_NODE_COLOR = "#2E86C1" #"#d81b60"           # red/magenta
HUB_NODE_COLOR =  "#f9a825"  # amber
REG_NODE_COLOR = "#aaaaaa"

ENDPOINT_NODE_RADIUS = 8
INTERMEDIATE_NODE_RADIUS = 8


# OD pairs to decompose
OD_PAIRS = [
    ('8644c1a77ffffff',	'8644c1a57ffffff') #("8649864b7ffffff",	"86499596fffffff"),
    # add more here
]
    


# Map style
FOLIUM_TILES = "cartodbpositron"

# Path drawing
PATH_COLORS = [
     "#D35400","#2FBF71", "#E4572E", "#4D8BFF", "#F1C40F", "#9B59B6", "#17A589",
    "#E67E22", "#2E86C1", "#C0392B", "#27AE60", "#7F8C8D",
]
PATH_EDGE_WEIGHT = 2
PATH_DASH = "6,6"
PATH_ARROW_SCALE = 1.4

# OD edge style
OD_COLOR = "#002060"
OD_WEIGHT = 4
OD_ARROW_SCALE = 2.0

# Arrow geometry
ARROW_FRAC = 0.10
ARROW_MIN_LEN_M = 30
ARROW_MAX_LEN_M = 5000
ARROW_TIP_ANGLE = 22.0



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

OUTPUT_DIR=os.path.join(RUN_DIR,"effective_path_decomposition")
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ============================================================
# Small Helpers
# ============================================================

def info(msg: str):
    print(f"[INFO] {msg}", flush=True)


def _mercator_xy(lat, lon, R=6378137.0):
    x = R * math.radians(lon)
    lat_c = max(min(lat, 85.05112878), -85.05112878)
    y = R * math.log(math.tan(math.pi/4 + math.radians(lat_c)/2))
    return x, y


def _mercator_inv(x, y, R=6378137.0):
    lat = math.degrees(math.atan(math.sinh(y / R)))
    lon = math.degrees(x / R)
    return lat, lon


def arrow_triangle(lat1, lon1, lat2, lon2, scale=1.0):
    x1, y1 = _mercator_xy(lat1, lon1)
    x2, y2 = _mercator_xy(lat2, lon2)

    vx = x2 - x1
    vy = y2 - y1
    seg = math.hypot(vx, vy)
    if seg < 1e-6:
        return None

    L = seg * ARROW_FRAC
    L = max(ARROW_MIN_LEN_M, min(ARROW_MAX_LEN_M, L))
    L *= scale

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
        _mercator_inv(rightx, righty),
    ]


# ============================================================
# Core WRAPPER: compute packs + elapsed + regression
# ============================================================

def build_context():
    """
    Build window, LSCC, packs, nodes, regression parameters.
    Uses the existing ED engine.
    """
    info("Loading full PEP OD...")

    # Rebuild OD path exactly the same way ED script does
    od_path = os.path.join(
        RUN_DIR,
        "csvs/pep_od.csv"
    )

    df_all = h3ed.load_pep(od_path)

    # normalize
    df_all[h3ed.START_COL] = df_all[h3ed.START_COL].astype(str).str.lower()
    df_all[h3ed.END_COL]   = df_all[h3ed.END_COL].astype(str).str.lower()

    # Window
    df_win0 = h3ed.subset_window(df_all, SPAN_START, SPAN_END)

    # LSCC
    nodes_lscc = h3ed.compute_lscc_nodes(df_win0)
    nodes_lscc = [h.lower() for h in nodes_lscc]

    df_win = df_win0[
        df_win0[h3ed.START_COL].isin(nodes_lscc) &
        df_win0[h3ed.END_COL].isin(nodes_lscc)
    ].copy()

    nodes = sorted(nodes_lscc)
    slices, node_index, packs = h3ed.build_packs(df_win, nodes)
    N = len(nodes)

    # reachable origins
    union = df_win.groupby(h3ed.START_COL)[h3ed.COUNT_COL].sum()
    origins = [node_index[o] for o in union.index if o in node_index]

    info("Computing elapsed distances...")
    X_elapsed, P_hit = h3ed.batched_elapsed_all_origins(packs, N, origins)

    info("Fitting regression (α,β)...")
    alpha, beta, rmse = h3ed.fit_region_regression(df_all)

    # node metadata + metro
    manifest = h3ed.load_manifest(MANIFEST_PATH)
    node_meta = h3ed.build_node_meta(manifest, nodes)
    metro = h3ed.build_metro_pairs(manifest)
    node_meta = {k.lower(): v for k,v in node_meta.items()}
    metro = {(o.lower(), d.lower()) for (o,d) in metro}

    regular_overlay, regular_background = \
        h3ed.build_regular_edge_pairs(
            manifest,
            include_background=INCLUDE_BACKGROUND_EDGES
        )

    return dict(
        df_all=df_all,
        df_win=df_win,
        nodes=nodes,
        node_index=node_index,
        slices=slices,
        packs=packs,
        X_elapsed=X_elapsed,
        P_hit=P_hit,
        alpha=alpha,
        beta=beta,
        node_meta=node_meta,
        metro_pairs=metro,
        regular_overlay_edges=regular_overlay,
        regular_background_edges=regular_background,
    )


# ============================================================
# Build Path-Decomposition Map
# ============================================================

def build_map(ctx, od_pairs: List[Tuple[str,str]], out_html: str):

    nodes       = ctx["nodes"]
    node_meta   = ctx["node_meta"]
    node_index  = ctx["node_index"]
    packs       = ctx["packs"]
    X_elapsed   = ctx["X_elapsed"]
    P_hit       = ctx["P_hit"]
    alpha       = ctx["alpha"]
    beta        = ctx["beta"]
    metro_pairs = ctx["metro_pairs"]
    regular_overlay = ctx["regular_overlay_edges"]
    regular_background = ctx["regular_background_edges"]

    # map center
    lats = [meta["lat"] for meta in node_meta.values()]
    lons = [meta["lon"] for meta in node_meta.values()]
    mlat = float(np.mean(lats))
    mlon = float(np.mean(lons))

    m = folium.Map(location=(mlat, mlon), zoom_start=11, tiles=FOLIUM_TILES)

    # ------------------------------------------------------------
    # Small unobtrusive map title
    # ------------------------------------------------------------
    title = (
        f"Effective Path Decomposition <br>"
        # f"OD: {od_pairs[0][0]} → {od_pairs[0][1]} · "
        # f"{SPAN_START} → {SPAN_END}<br>"
        # f"Cutoff: top-{MAX_PATHS_PER_OD} by {PATH_RANK_BY}"
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

    # ------------------------------------------------------------
    # Determine endpoint + intermediate nodes (top-K consistent)
    # ------------------------------------------------------------
    endpoint_nodes = set()
    intermediate_nodes = set()

    for (o, d) in od_pairs:
        o, d = o.lower(), d.lower()
        endpoint_nodes |= {o, d}

        res = h3ed.enumerate_first_hitting_paths_h3(
            (o, d), packs, nodes, node_index, write_csv=False
        )

        paths = res["paths_geo"]

        # rank + top-K (matches map)
        paths = sorted(
            paths,
            key=lambda p: float(p["prob"]),
            reverse=(PATH_RANK_BY == "prob"),
        )[:MAX_PATHS_PER_OD]

        for p in paths:
            for n in p["path_geo"]:
                if n not in (o, d):
                    intermediate_nodes.add(n)

    # safety: endpoints win
    intermediate_nodes -= endpoint_nodes

    # Layers
    g_metro = folium.FeatureGroup("Metro", show=True)
    g_od    = folium.FeatureGroup("OD edges", show=True)
    g_bg = folium.FeatureGroup("Background edges", show=False)
    g_overlay = folium.FeatureGroup("Overlay edges", show=True)

    # Draw nodes
    # ------------------------------------------------------------
    # Draw nodes with role-based coloring
    # ------------------------------------------------------------
    for h, meta in node_meta.items():
        lat, lon = meta["lat"], meta["lon"]

        if h in endpoint_nodes:
            color = ENDPOINT_NODE_COLOR
            radius = ENDPOINT_NODE_RADIUS
            label = f"OD endpoint<br>{h}"

        elif h in intermediate_nodes:
            color = INTERMEDIATE_NODE_COLOR
            radius = INTERMEDIATE_NODE_RADIUS
            label = f"Path intermediate<br>{h}"

        elif meta.get("is_hub", False):
            color = HUB_NODE_COLOR
            radius = 6
            label = f"Hub<br>{h}"

        else:
            color = REG_NODE_COLOR
            radius = 2
            label = h

        folium.CircleMarker(
            location=(lat, lon),
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            tooltip=label,
        ).add_to(m)


    # Metro edges
    for o, d in metro_pairs:
        if o in node_meta and d in node_meta:
            lo = node_meta[o]
            ld = node_meta[d]
            folium.PolyLine(
                [(lo["lat"], lo["lon"]), (ld["lat"], ld["lon"])],
                color="#777", weight=3, opacity=0.9
            ).add_to(g_metro)


    # Overlay (corridor/feeder) edges
    for o, d in regular_overlay:
        if o in node_meta and d in node_meta:
            lo = node_meta[o]
            ld = node_meta[d]
            folium.PolyLine(
                [(lo["lat"], lo["lon"]), (ld["lat"], ld["lon"])],
                color="#1f78b4",
                weight=2,
                opacity=0.6,
            ).add_to(g_overlay)

    # Background adjacency edges
    for o, d in regular_background:
        if o in node_meta and d in node_meta:
            lo = node_meta[o]
            ld = node_meta[d]
            folium.PolyLine(
                [(lo["lat"], lo["lon"]), (ld["lat"], ld["lon"])],
                color="#bbbbbb",
                weight=1,
                opacity=0.4,
            ).add_to(g_bg)


    g_metro.add_to(m)
    g_od.add_to(m)
    g_bg.add_to(m)
    g_overlay.add_to(m)


    # Draw each OD
    for (o, d) in od_pairs:

        if (o not in node_index) or (d not in node_index):
            info(f"OD {o}->{d} not in LSCC — skipping.")
            continue

        i = node_index[d]
        j = node_index[o]

        D_elapsed = float(X_elapsed[i, j])
        P_hit_od  = float(P_hit[i, j])
        D_direct  = h3ed.predict_direct_distance(o, d, alpha, beta)
        D_eff     = D_elapsed / D_direct if D_direct > 0 else float('nan')

        info(f"[OD {o}->{d}] D_elapsed={D_elapsed:.1f}m  D_direct={D_direct:.1f}m  D_eff={D_eff:.3f}")

        # --- Draw OD edge ---
        lo = node_meta[o]
        ld = node_meta[d]

        popup_text = (
            f"<b>{o} → {d}</b><br>"
            f"D_elapsed = {D_elapsed:.1f} m<br>"
            f"D_direct  = {D_direct:.1f} m<br>"
            f"D_eff     = {D_eff:.3f}<br>"
            f"P_hit     = {P_hit_od:.3e}"
        )

        line = folium.PolyLine(
            [(lo["lat"], lo["lon"]), (ld["lat"], ld["lon"])],
            color=OD_COLOR, weight=OD_WEIGHT, opacity=0.95
        )
        line.add_child(folium.Popup(popup_text, max_width=260))
        line.add_to(g_od)

        tri = arrow_triangle(lo["lat"], lo["lon"], ld["lat"], ld["lon"], scale=OD_ARROW_SCALE)
        if tri:
            folium.Polygon(
                tri, color=OD_COLOR, fill=True, fill_color=OD_COLOR, fill_opacity=0.95
            ).add_to(g_od)

        # --- Enumerate paths ---
        res = h3ed.enumerate_first_hitting_paths_h3(
            (o, d), packs, nodes, node_index, write_csv=False
        )

        P_sum = res["P_sum"]
        paths_geo = res["paths_geo"]

        # -------------------------------
        # TOP-K PATH FILTERING
        # -------------------------------
        if PATH_RANK_BY == "prob":
            paths_geo = sorted(
                paths_geo,
                key=lambda p: float(p.get("prob", 0.0)),
                reverse=True,
            )
        elif PATH_RANK_BY == "distance":
            paths_geo = sorted(
                paths_geo,
                key=lambda p: float(p.get("dist", float("inf"))),
            )
        else:
            raise ValueError(f"Unknown PATH_RANK_BY={PATH_RANK_BY}")

        paths_geo = paths_geo[:MAX_PATHS_PER_OD]

        info(
            f"[{o}->{d}] showing {len(paths_geo)} / "
            f"{len(res['paths_geo'])} paths | "
            f"P_sum={P_sum:.6f}"
        )

        # =========================================================
        # WRITE PATH CSV (POST-FILTER, MAP-CONSISTENT)
        # =========================================================
        csv_rows = []
        for rank, p in enumerate(paths_geo, start=1):
            csv_rows.append({
                "origin": o,
                "destination": d,
                "rank": rank,
                "probability": float(p["prob"]),
                "distance_m": float(p["dist"]),
                "steps": int(p["t"]),
                "path": "->".join(p["path_geo"]),
            })

        if csv_rows:
            df_paths = pd.DataFrame(csv_rows)
            csv_path = os.path.join(
                OUTPUT_DIR,
                f"paths_{o}_{d}.csv"
            )
            df_paths.to_csv(csv_path, index=False)
            info(f"[PATH CSV] wrote: {csv_path}")


        # Draw each path group
        for k, p in enumerate(paths_geo):
            color = PATH_COLORS[k % len(PATH_COLORS)]
            gh_path = p["path_geo"]
            prob    = float(p["prob"])
            dist_m  = float(p["dist"])
            steps   = int(p["t"])

            tooltip = (
                f"Path {k+1} ({o}->{d})<br>"
                f"P_path={prob:.3e}<br>"
                f"dist={dist_m:.1f} m<br>"
                f"steps={steps}"
            )

            for u, v in zip(gh_path[:-1], gh_path[1:]):
                if u not in node_meta or v not in node_meta:
                    continue

                lu = node_meta[u]
                lv = node_meta[v]

                seg = folium.PolyLine(
                    [(lu["lat"], lu["lon"]), (lv["lat"], lv["lon"])],
                    color=color, weight=PATH_EDGE_WEIGHT,
                    opacity=0.95, dash_array=PATH_DASH,
                )
                seg.add_child(folium.Tooltip(tooltip))
                # seg.add_to(fg)
                seg.add_to(m)

                tri2 = arrow_triangle(lu["lat"], lu["lon"], lv["lat"], lv["lon"], scale=PATH_ARROW_SCALE)
                if tri2:
                    folium.Polygon(
                        tri2,
                        color=color, fill=True, fill_color=color, fill_opacity=0.95
                    ).add_to(m) #.add_to(fg)

            # fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_html)
    info(f"[MAP] wrote: {out_html}")


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    info("Building context...")
    ctx = build_context()

    out_html = os.path.join(
        OUTPUT_DIR,
        f"path_decomposition_h3__{SPAN_START.replace(':','-')}_to_{SPAN_END.replace(':','-')}.html"
    )

    build_map(ctx, OD_PAIRS, out_html)

    info(f"TOTAL RUNTIME = {time.time() - t0:.2f} sec")


if __name__ == "__main__":
    main()
