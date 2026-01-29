#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
time_elapsed_netflows_geohash.py
------------------------------------------

Time–elapsed mobility propagation from aggregated NetMob OD data
on a geohash tiling (single city / single region).

This script constructs a time-ordered linear mobility operator
directly from observed OD matrices and computes a directed
net effective flow field, following the methodology described in:

    Network-Level Measures of Mobility from Aggregated Origin-Destination Data  
    https://arxiv.org/abs/2502.04162

----------------------------------------------------------------------
INPUT
----------------------------------------------------------------------

Aggregated OD CSV from the NetMob 2024 Data Challenge.

Required columns:
    start_geohash, end_geohash,
    date, time,
    trip_count, mean_length_m

An optional manifest file may be present:

    corridors_outputs_geohash/
        corridors_manifest_{FILE_TAG}.json

The manifest is used only to obtain geographic coordinates for
visualization. If missing, geohash centroids are decoded directly.

----------------------------------------------------------------------
COMPUTATION
----------------------------------------------------------------------

Let F(t) be the empirical OD flow matrix at time bin t, where

    F(t)[d,o] = number of trips from o → d.

Define the column-stochastic transition operator

    M(t) = F(t) · diag(1 / Σ_d F(t)[d,o]).

Form the time-ordered product over a window [t₁, t₂):

    A = M(t₂−1) · M(t₂−2) · ... · M(t₁).

Inject the observed outflow at the first bin:

    n_start(o) = Σ_d F(t₁)[d,o].

The elapsed flow field is

    C = A · diag(n_start),

so that C[d,o] is the expected number of trips that start at o
at time t₁ and arrive at d after temporal propagation.

To expose directional structure, compute the net flow matrix

    C_net = max(C − Cᵀ, 0),

with the diagonal optionally removed.

----------------------------------------------------------------------
OUTPUT
----------------------------------------------------------------------

Interactive HTML map:

    *_netflows_p{P}_YYYYMMDD_HHMM_to_YYYYMMDD_HHMM.html

showing the strongest directed effective flows, with color indicating
magnitude and arrows indicating direction.

----------------------------------------------------------------------
INTERPRETATION
----------------------------------------------------------------------

For theoretical background and interpretation, see:

    Network-Level Measures of Mobility from Aggregated Origin-Destination Data  
    https://arxiv.org/abs/2502.04162

----------------------------------------------------------------------
AUTHOR
----------------------------------------------------------------------

Asif Shakeel  
ashakeel@ucsd.edu
"""



import os, html, json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from scipy import sparse

# Optional mapping
try:
    import folium
    import branca.colormap as cmb
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False

import sys

def progress(msg):
    sys.stdout.write("\r" + msg)
    sys.stdout.flush()

# =========================
# CONFIG
# =========================

BASE_DIR = "/Users/asif/Documents/nm24/outputs/nm_outputs"
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_OD_FILE = "od_mx_agg5_3h_MexicoCity_20190601T000000_20191001T000000.csv"   # in DATA_DIR
OD_INPUT_PATH = os.path.join(DATA_DIR, INPUT_OD_FILE)

OUTPUT_DIR = os.path.join(BASE_DIR, "timeElapsed_flows")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REGION_NAME = "Mexico City, Mexico"

# Geohash resolution (used for tags / decode)
GEOHASH_RESOLUTION = 5

# Time window and binning
TIME_RES_MIN = 180
SPAN_START   = "2019-06-05 15:00:00"
SPAN_END     = "2019-06-05 21:00:00"

# Column names in OD file
START_COL = "start_geohash"
END_COL   = "end_geohash"
DATE_COL  = "date"
TIME_COL  = "time"
COUNT_COL = "trip_count"       # will be normalized into "trip_count"
DIST_COL  = "mean_length_m"    # not used in the core calculation

# Thresholds
NETFLOWS_PERCENTILE = 0.75     # your PERCENTILE
MIN_COUNT = 1.0
DROP_DIAGONAL = True

# Mean edge flow configs (kept, but can be turned off if not needed)
RUN_MEAN_EDGE_IN_OUT = True
MEAN_EDGE_PERCENTILE = 0.00
MEAN_NEUTRAL_TOL     = 1e-9

RUN_MEAN_EDGE_NETFLOW = True
RUN_MEAN_NODE_ACCUM   = True

# Map config
MAKE_MAPS = True
FOLIUM_TILES = "cartodbpositron"
SHOW_ALL_CELLS = True
ALL_CELLS_MARKER_RADIUS = 2.5
ALL_CELLS_MARKER_OPACITY = 0.9

# --- NEW: Map UI placement controls ---
# Colorbar wrapper position: choose LEFT or RIGHT anchoring.
COLORBAR_BOTTOM_PX = 100
COLORBAR_LEFT_PX   = None   # set to int to anchor from left
COLORBAR_RIGHT_PX  = 200     # set to None if using left

# LayerControl position override: choose LEFT or RIGHT anchoring.
LAYERCTRL_TOP_PX   = 50
LAYERCTRL_LEFT_PX  = None   # set to int to anchor from left
LAYERCTRL_RIGHT_PX = 200    # set to None if using left
# --- END NEW ---

# Arrow styling
ARROW_MIN_LEN_M = 300.0
ARROW_MAX_LEN_M = 1200.0
ARROW_FRAC_OF_SEG = 0.50
ARROW_TIP_ANGLE   = 28.0

RUN_NETFLOWS = True
SHOW_NODE_INTENSITY = True

# Graph / manifest tags (kept from original logic)
NEIGHBOR_TOPOLOGY = "4"
FEEDER_MODE       = "single"
EDGE_DIRECTION_MODE = "potential"
INCLUDE_BACKGROUND_EDGES = False

MANIFEST_NODES: Dict[str, Tuple[float, float]] = {}
MANIFEST_HUBS: List[str] = []
MANIFEST_METRO_EDGES: List[Tuple[str, str]] = []

def _sanitize_region(r: str) -> str:
    core = r.split(",")[0]
    return "".join(ch for ch in core if ch.isalnum())

REGION_TAG = _sanitize_region(REGION_NAME)
TAG_TOPOLOGY = f"top-{NEIGHBOR_TOPOLOGY}"
TAG_FEEDER   = f"fdr-{FEEDER_MODE}"
TAG_EDGEDIR  = f"edge-{('pot' if EDGE_DIRECTION_MODE=='potential' else 'geo')}"
TAG_OVERLAY  = f"ovr-{(1 if INCLUDE_BACKGROUND_EDGES else 0)}"
FILE_TAG     = f"mode-geohash_{TAG_TOPOLOGY}_fdr-{FEEDER_MODE}_{TAG_EDGEDIR}_{TAG_OVERLAY}"

def _city_tag() -> str:
    return (REGION_NAME.split(',')[0]).replace(' ', '') or "Region"

def _top_tag() -> str:
    return f"gh-{NEIGHBOR_TOPOLOGY}-res-{GEOHASH_RESOLUTION}"

def _fdr_tag() -> str:
    return f"fdr-{str(FEEDER_MODE).lower()}"

def _edge_tag():
    if EDGE_DIRECTION_MODE == "potential":
        return "edge-pot"
    elif EDGE_DIRECTION_MODE == "geometric":
        return "edge-geom"
    else:
        return f"edge-{EDGE_DIRECTION_MODE}"

def _overlay_tag() -> str:
    return "ovr-1" if INCLUDE_BACKGROUND_EDGES else "ovr-0"

city_tag = _city_tag()
top_tag  = _top_tag()
fdr_tag  = _fdr_tag()
edge_tag = _edge_tag()
overlay_tag = _overlay_tag()

file_tag = f"{REGION_TAG}_{FILE_TAG}"
CORRIDOR_MANIFEST_FILENAME  = f"corridors_manifest_{file_tag}.json"
CORRIDOR_MANIFEST_PATH      = os.path.join(
    BASE_DIR, "corridors_outputs_geohash", CORRIDOR_MANIFEST_FILENAME
)

# =========================
# Manifest loading (optional)
# =========================

def _load_manifest(path: str) -> Tuple[Dict[str, Tuple[float, float]], List[str], List[Tuple[str, str]]]:
    if not os.path.exists(path):
        print(f"[MANIFEST] Not found: {path}")
        return {}, [], []

    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception as e:
        print(f"[MANIFEST] Error reading {path}: {e}")
        return {}, [], []

    nodes_obj = doc.get("nodes")
    nodes_out: Dict[str, Tuple[float, float]] = {}
    if isinstance(nodes_obj, dict):
        for gh_, meta in nodes_obj.items():
            try:
                lat = float(meta["lat"])
                lon = float(meta["lon"])
            except Exception:
                continue
            key = str(gh_).strip().lower()
            nodes_out[key] = (lat, lon)

    hubs_out: List[str] = []
    hubs_obj = doc.get("hubs", {}).get("geohash", {})
    if isinstance(hubs_obj, dict):
        hubs_out = [str(gh_).strip().lower() for gh_ in hubs_obj.keys()]
    print(f"[MANIFEST] Loaded {len(nodes_out)} node coords, {len(hubs_out)} hubs from {path}")

    metro_edges: List[Tuple[str, str]] = []
    edges_obj = doc.get("edges", {})
    am_edges = edges_obj.get("AM", [])
    if isinstance(am_edges, list):
        for rec in am_edges:
            try:
                if int(rec.get("is_metro", 0)) != 1:
                    continue
                s = str(rec["start_geohash"]).strip().lower()
                d = str(rec["end_geohash"]).strip().lower()
            except Exception:
                continue
            metro_edges.append((s, d))
    print(f"[MANIFEST] Loaded {len(metro_edges)} metro edges from manifest")

    return nodes_out, hubs_out, metro_edges

MANIFEST_NODES, MANIFEST_HUBS, MANIFEST_METRO_EDGES = _load_manifest(CORRIDOR_MANIFEST_PATH)

# =========================
# Geohash decode (fallback)
# =========================

_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_BITS   = np.array([16, 8, 4, 2, 1])

def geohash_decode(g: str) -> Tuple[float, float]:
    g = str(g).strip().lower()
    if not g:
        raise ValueError("Empty geohash")
    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]
    even = True
    for ch in g:
        cd = _BASE32.find(ch)
        if cd == -1:
            raise ValueError(f"Invalid geohash char: {ch!r}")
        for mask in _BITS:
            bit = 1 if (cd & mask) else 0
            if even:
                mid = (lon_interval[0] + lon_interval[1]) / 2.0
                if bit:
                    lon_interval[0] = mid
                else:
                    lon_interval[1] = mid
            else:
                mid = (lat_interval[0] + lat_interval[1]) / 2.0
                if bit:
                    lat_interval[0] = mid
                else:
                    lat_interval[1] = mid
            even = not even
    lat = (lat_interval[0] + lat_interval[1]) / 2.0
    lon = (lon_interval[0] + lon_interval[1]) / 2.0
    return lat, lon

def resolve_latlon(gh: str) -> Tuple[float, float]:
    key = str(gh).strip().lower()
    if MANIFEST_NODES:
        coord = MANIFEST_NODES.get(key)
        if coord is not None:
            return coord
    return geohash_decode(key)

# =========================
# Time parsing
# =========================

def ensure_bin_start(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # normalize count column into "trip_count"
    if "trip_count" not in d.columns and COUNT_COL in d.columns:
        d = d.rename(columns={COUNT_COL: "trip_count"})
    if "trip_count" not in d.columns and "count" in d.columns:
        d = d.rename(columns={"count": "trip_count"})
    if "trip_count" not in d.columns:
        raise ValueError("Missing 'trip_count' (or alias) column.")

    d["trip_count"] = pd.to_numeric(d["trip_count"], errors="coerce").fillna(0).astype("int64")

    d[START_COL] = d[START_COL].astype(str).str.lower().str.strip()
    d[END_COL]   = d[END_COL].astype(str).str.lower().str.strip()

    if DATE_COL not in d.columns or TIME_COL not in d.columns:
        raise ValueError(f"Expected '{DATE_COL}' and '{TIME_COL}' in OD CSV.")

    time_str = d[TIME_COL].astype(str).str.strip()
    combo = d[DATE_COL].astype(str) + " " + time_str
    dt = pd.to_datetime(combo, format="%Y%m%d %H:%M:%S", errors="coerce")
    if dt.isna().any():
        dt = pd.to_datetime(combo, errors="coerce")

    if hasattr(dt, "dt") and getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)

    if dt.isna().any():
        bad = d.loc[dt.isna(), [DATE_COL, TIME_COL]].head(8)
        raise ValueError(f"Could not parse some timestamps; examples:\n{bad}")

    d["bin_start"] = dt
    d["bin_end"]   = dt + pd.to_timedelta(TIME_RES_MIN, unit="m")
    return d

def crop_to_span(df: pd.DataFrame) -> pd.DataFrame:
    d = ensure_bin_start(df)
    t1 = pd.to_datetime(SPAN_START)
    t2 = pd.to_datetime(SPAN_END)
    last_needed_start = t2 - pd.to_timedelta(TIME_RES_MIN, unit="m")
    return d[(d["bin_start"] >= t1) & (d["bin_start"] <= last_needed_start)].copy()

# =========================
# Per-bin matrices
# =========================

@dataclass
class PerBin:
    bin_start: pd.Timestamp
    F: sparse.csr_matrix  # (dst, src)
    M: sparse.csr_matrix  # column-stochastic

def build_per_bin(df: pd.DataFrame) -> Tuple[List[PerBin], pd.Index]:
    cells = pd.Index(sorted(pd.unique(pd.concat([df[START_COL], df[END_COL]], ignore_index=True))))
    idx = {g: i for i, g in enumerate(cells)}
    N = len(cells)
    out: List[PerBin] = []
    for t, g in df.groupby("bin_start", sort=True):
        rows = g[END_COL].map(idx).to_numpy()
        cols = g[START_COL].map(idx).to_numpy()
        data = g["trip_count"].to_numpy(dtype=np.int64)
        F = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
        colsum = np.asarray(F.sum(axis=0)).ravel()
        inv = np.zeros_like(colsum, dtype=float)
        nz = colsum > 0
        inv[nz] = 1.0 / colsum[nz]
        M = F @ sparse.diags(inv)
        out.append(PerBin(bin_start=pd.Timestamp(t), F=F.tocsr(), M=M.tocsr()))
    return out, cells

# =========================
# Legacy elapsed (single injection at t1)
# =========================

def legacy_elapsed_counts(per_bin: List[PerBin], i1: int, i2_excl: int) -> np.ndarray:
    N = per_bin[0].F.shape[0]
    A = sparse.identity(N, format="csr")
    for t in range(i1, i2_excl):
        A = per_bin[t].M @ A
    F1 = per_bin[i1].F
    n_start = np.asarray(F1.sum(axis=0)).ravel().astype(float)
    C = A.toarray() * n_start[np.newaxis, :]
    return C

def locate_span(per_bin: List[PerBin]) -> Tuple[int, int]:
    t1 = pd.to_datetime(SPAN_START)
    t2 = pd.to_datetime(SPAN_END)
    t_last = t2 - pd.to_timedelta(TIME_RES_MIN, unit="m")
    index = {pb.bin_start: i for i, pb in enumerate(per_bin)}
    if t1 not in index or t_last not in index:
        raise RuntimeError("Span not aligned to bins")
    return index[t1], index[t_last] + 1

# =========================
# Convert matrix to edges
# =========================

def counts_to_edges(C: np.ndarray,
                    cells: pd.Index,
                    percentile: float,
                    min_count: float) -> pd.DataFrame:
    ii, jj = np.nonzero(C)
    if ii.size == 0:
        return pd.DataFrame(columns=[
            "origin","destination","flow",
            "src_lat","src_lon","dst_lat","dst_lon"
        ])
    vals = C[ii, jj].astype(float)
    elig = vals[vals > 0]
    thr = float(np.quantile(elig, percentile)) if (percentile > 0 and elig.size > 0) else 0.0
    keep = (vals > 0) & (vals >= max(thr, min_count))
    ii, jj, vals = ii[keep], jj[keep], vals[keep]
    if vals.size == 0:
        return pd.DataFrame(columns=[
            "origin","destination","flow",
            "src_lat","src_lon","dst_lat","dst_lon"
        ])

    df = pd.DataFrame({"i": ii, "j": jj, "flow": vals})
    df["origin"]      = cells[df["j"].values]
    df["destination"] = cells[df["i"].values]

    src = np.array([resolve_latlon(g) for g in df["origin"]])
    dst = np.array([resolve_latlon(g) for g in df["destination"]])
    df["src_lat"], df["src_lon"] = src[:, 0], src[:, 1]
    df["dst_lat"], df["dst_lon"] = dst[:, 0], dst[:, 1]

    df.sort_values("flow", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# =========================
# Folium helpers & map export
# =========================

def _mercator_xy(lat, lon, R=6378137.0):
    from math import log, tan, pi, radians
    x = R * radians(lon)
    lat = max(min(lat, 85.05112878), -85.05112878)
    y = R * log(tan(pi/4.0 + radians(lat)/2.0))
    return x, y

def _mercator_inv(x, y, R=6378137.0):
    from math import atan, sinh, degrees
    lat = degrees(atan(sinh(y / R)))
    lon = degrees(x / R)
    return lat, lon

def _arrow_triangle(lat1, lon1, lat2, lon2,
                    min_len_m=ARROW_MIN_LEN_M,
                    max_len_m=ARROW_MAX_LEN_M,
                    frac_of_seg=ARROW_FRAC_OF_SEG,
                    tip_angle_deg=ARROW_TIP_ANGLE):
    from math import radians, tan
    x1, y1 = _mercator_xy(lat1, lon1)
    x2, y2 = _mercator_xy(lat2, lon2)
    vx, vy = x2 - x1, y2 - y1
    seg = (vx*vx + vy*vy)**0.5
    if seg < 1e-6:
        return None
    L = max(min_len_m, min(max_len_m, seg*frac_of_seg))
    ux, uy = vx / seg, vy / seg
    bx, by = x2 - L*ux, y2 - L*uy
    px, py = -uy, ux
    half_base = L * tan(radians(tip_angle_deg)/2.0)
    leftx,  lefty  = bx + half_base*px, by + half_base*py
    rightx, righty = bx - half_base*px, by - half_base*py
    tip     = _mercator_inv(x2, y2)
    left    = _mercator_inv(leftx, lefty)
    right   = _mercator_inv(rightx, righty)
    return [tip, left, right]

def add_title(m, text: str):
    m.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
      z-index:9999;background:rgba(255,255,255,.95);padding:6px 10px;border-radius:8px;
      box-shadow:0 1px 6px rgba(0,0,0,.15);font-family:sans-serif;font-size:13px;font-weight:700;color:#222;">
      {html.escape(text)}
    </div>"""))

# --- NEW: Flexible UI helpers (colorbar + LayerControl) ---
def add_colorbar(m, cmap_obj):
    if not HAS_FOLIUM:
        return
    if COLORBAR_LEFT_PX is not None:
        horiz = f"left:{int(COLORBAR_LEFT_PX)}px;"
    else:
        horiz = f"right:{int(COLORBAR_RIGHT_PX)}px;"

    m.get_root().html.add_child(
        folium.Element(
            f"""
            <div style="position:fixed;bottom:{int(COLORBAR_BOTTOM_PX)}px;{horiz}z-index:9999;">
                {cmap_obj._repr_html_()}
            </div>
            """
        )
    )

def move_layer_control(m):
    if not HAS_FOLIUM:
        return
    if LAYERCTRL_LEFT_PX is not None:
        horiz = f"left:{int(LAYERCTRL_LEFT_PX)}px;"
    else:
        horiz = f"right:{int(LAYERCTRL_RIGHT_PX)}px;"

    m.get_root().html.add_child(
        folium.Element(
            f"""
            <style>
            .leaflet-control-layers {{
                position: fixed !important;
                top: {int(LAYERCTRL_TOP_PX)}px !important;
                {horiz}
                z-index: 9999;
            }}
            </style>
            """
        )
    )
# --- END NEW ---

def export_edges_map(
    edges: pd.DataFrame,
    all_cells: Optional[pd.Index],
    title: str,
    outfile: str,
):
    if not HAS_FOLIUM:
        print("[MAP] Folium not available; skipping:", title)
        return

    latlons: List[Tuple[float, float]] = []
    if edges is not None and not edges.empty:
        latlons += edges[["src_lat", "src_lon"]].to_numpy().tolist()
        latlons += edges[["dst_lat", "dst_lon"]].to_numpy().tolist()
    if all_cells is not None and len(all_cells) > 0:
        latlons += [resolve_latlon(g) for g in all_cells]

    if latlons:
        lats = [p[0] for p in latlons]
        lons = [p[1] for p in latlons]
        center_lat, center_lon = float(np.mean(lats)), float(np.mean(lons))
    else:
        # fallback near Mexico City
        center_lat, center_lon = 19.4326, -99.1332

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles=FOLIUM_TILES)
    add_title(m, title)

    if SHOW_ALL_CELLS and all_cells is not None and len(all_cells) > 0:
        fg_cells = folium.FeatureGroup(name="All cells", show=False)
        for gh in all_cells:
            lat, lon = resolve_latlon(gh)
            folium.CircleMarker(
                location=(lat, lon),
                radius=ALL_CELLS_MARKER_RADIUS,
                color="#3a3a3a",
                fill=True, fill_color="#3a3a3a",
                fill_opacity=0.6, opacity=0.6,
                tooltip=gh,
            ).add_to(fg_cells)
        fg_cells.add_to(m)

    if edges is not None and not edges.empty:
        vals = edges["flow"].astype(float)

        vmin_raw = float(vals.min())
        vmax_raw = float(vals.max())

        vmin = int(np.floor(vmin_raw))
        vmax = int(np.ceil(vmax_raw))
        if vmax <= vmin:
            vmax = vmin + 1

        ticks = np.linspace(vmin, vmax, 6)
        ticks = [int(round(t)) for t in ticks]
        for k in range(1, len(ticks)):
            if ticks[k] <= ticks[k-1]:
                ticks[k] = ticks[k-1] + 1

        cmap_edges = cmb.LinearColormap(
            ["magenta", "navy"],
            vmin=vmin,
            vmax=vmax
        ).to_step(len(ticks))
        cmap_edges.index = ticks
        cmap_edges.caption = "Flow"


        fg_edges = folium.FeatureGroup(name="Edges", show=True)
        edges_sorted = edges.sort_values("flow", ascending=True)

        for r in edges_sorted.itertuples(index=False):
            c = cmap_edges(float(r.flow))
            folium.PolyLine(
                [[r.src_lat, r.src_lon], [r.dst_lat, r.dst_lon]],
                color=c, weight=1.6, opacity=0.75,
                tooltip=f"{r.origin} → {r.destination}: {float(r.flow):.3f}",
            ).add_to(fg_edges)
            tri = _arrow_triangle(r.src_lat, r.src_lon, r.dst_lat, r.dst_lon)
            if tri is not None:
                folium.Polygon(tri, color=c, fill=True, fill_color=c, fill_opacity=0.85).add_to(fg_edges)
        fg_edges.add_to(m)

        # --- CHANGED: was hard-coded right:20px; now configurable ---
        add_colorbar(m, cmap_edges)

    # --- NEW: configurable placement for LayerControl (tick legend) ---
    move_layer_control(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(outfile)
    print(f"[MAP] wrote {outfile}")

# =========================
# Main
# =========================

def main():
    print("[SETUP] OUTPUT_DIR:", OUTPUT_DIR)
    print("[MANIFEST] entries:", len(MANIFEST_NODES))

    print(f"[READ] {OD_INPUT_PATH}")
    raw = pd.read_csv(OD_INPUT_PATH, dtype={START_COL: str, END_COL: str})

    od = crop_to_span(raw)
    if od.empty:
        raise RuntimeError("No rows in span.")

    per_bin, cells = build_per_bin(od)
    i1, i2 = locate_span(per_bin)

    t1 = pd.to_datetime(SPAN_START)
    t2 = pd.to_datetime(SPAN_END)

    from_slug = t1.strftime("%Y%m%d_%H%M")
    to_slug   = t2.strftime("%Y%m%d_%H%M")

    if RUN_NETFLOWS:
        C = legacy_elapsed_counts(per_bin, i1, i2)
        Cnet = C - C.T
        if DROP_DIAGONAL:
            np.fill_diagonal(Cnet, 0.0)
        Cnet = np.maximum(Cnet, 0.0)

        edges_net = counts_to_edges(Cnet, cells,
                                    percentile=NETFLOWS_PERCENTILE,
                                    min_count=MIN_COUNT)

        if MAKE_MAPS:
            html_out = os.path.join(
                OUTPUT_DIR,
                f"{os.path.splitext(INPUT_OD_FILE)[0]}_netflows_p{int(round(NETFLOWS_PERCENTILE*100))}"
                f"_{from_slug}_to_{to_slug}.html"
            )
            title = f"Net flows — Top {int(round(100 - NETFLOWS_PERCENTILE * 100))}% — {t1:%Y-%m-%d %H:%M} → {t2:%Y-%m-%d %H:%M}"
            export_edges_map(edges_net, cells, title, html_out)

    print("[DONE]")

if __name__ == "__main__":
    main()
