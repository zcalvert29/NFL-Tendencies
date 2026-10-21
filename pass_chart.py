"""
Pass chart: every throw drawn where it was caught (or thrown to) on a perspective field, ported from the Dowling
app's pass_chart.py.

nflfastR gives each throw its air yards (exact depth) and a direction (left / middle / right), so:
  * depth = air yards, with a hair of jitter so stacked throws separate;
  * side  = a random spot inside that third of the field (the same spot every time the chart is drawn).
Sacks, scrambles, and throws with no target depth (spikes, some throwaways) aren't drawn but count in the stat line.

Markers: completion = filled green, touchdown = filled blue, incomplete = hollow ring, interception = filled red.
No Streamlit code here: st.html(pass_chart_html(...)).
"""
from __future__ import annotations

import base64
import html
import zlib

import numpy as np
import pandas as pd

FIELD_W = 160 / 3                          # 53 1/3 yards
HASH_X = (FIELD_W / 2 - 3.083, FIELD_W / 2 + 3.083)   # NFL hashes: 18'6" apart, centered
Y_MIN, Y_MAX = -10, 50
GREEN, RED, BLUE = "#1D9E75", "#E24B4A", "#185FA5"
OUTCOME_COLOR = {"Complete": GREEN, "Touchdown": BLUE, "Interception": RED}
FIELD_FILL, SIDELINE_FILL = "rgba(128,128,128,.07)", "rgba(128,128,128,.22)"
LINE, TICK = "rgba(128,128,128,.38)", "rgba(128,128,128,.32)"
THIRD = {"Left": 0, "Middle": 1, "Right": 2}


def throws(team_d: pd.DataFrame) -> pd.DataFrame:
    """Pass attempts (sacks and scrambles left out) with chart positions."""
    d = team_d[(team_d["PASS"] == 1) & (team_d["sack"] != 1) & (team_d["qb_scramble"] != 1)].copy()
    d["is_int"] = d["interception"] == 1
    d["is_comp"] = d["complete_pass"] == 1
    d["is_td"] = d["is_comp"] & (d["touchdown"] == 1)
    d["outcome"] = np.select([d["is_int"], d["is_td"], d["is_comp"]], ["Interception", "Touchdown", "Complete"],
                             "Incomplete")
    xs, ys = [], []
    for gid, pid, side, depth in zip(d["game_id"], d["play_id"], d["Pass Direction"], d["air_yards"]):
        if side not in THIRD or pd.isna(depth):
            xs.append(np.nan)
            ys.append(np.nan)
            continue
        rng = np.random.default_rng(zlib.crc32(f"{gid}|{pid}".encode()))
        lo = THIRD[side] * FIELD_W / 3
        xs.append(lo + rng.uniform(2.2, FIELD_W / 3 - 2.2))
        y = depth + rng.uniform(-0.6, 0.6)
        if depth >= 0:
            y = max(y, 0.0)
        ys.append(float(np.clip(y, Y_MIN + 1, Y_MAX - 1)))
    d["x"], d["y"] = xs, ys
    return d


def stat_line(t: pd.DataFrame) -> dict:
    air = t["air_yards"].dropna()
    return {"comp": int(t["is_comp"].sum()), "att": len(t),
            "yards": int(t.loc[t["is_comp"], "yards_gained"].fillna(0).sum()), "td": int(t["is_td"].sum()),
            "int": int(t["is_int"].sum()), "adot": float(air.mean()) if len(air) else float("nan"),
            "plotted": int(t["x"].notna().sum())}


VB_W, VB_H = 760, 470
_NEAR_W = 600
_FIELD_TOP, _FIELD_BOT = 16, 452


def _project(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    v = np.clip((y - Y_MIN) / (Y_MAX - Y_MIN), 0, 1)
    scale = 1.0 - 0.36 * v
    frac = 1 - (1 - v) ** 1.18
    px = VB_W / 2 + (x - FIELD_W / 2) / FIELD_W * _NEAR_W * scale
    py = _FIELD_BOT - frac * (_FIELD_BOT - _FIELD_TOP)
    return px, py, scale


def _seg(x0, y0, x1, y1, color, width):
    (a, c), (b, d), _ = _project([x0, x1], [y0, y1])
    return f'<line x1="{a:.1f}" y1="{b:.1f}" x2="{c:.1f}" y2="{d:.1f}" stroke="{color}" stroke-width="{width}"/>'


def _poly(pts, fill):
    px, py, _ = _project([p[0] for p in pts], [p[1] for p in pts])
    return f'<polygon points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in zip(px, py))}" fill="{fill}"/>'


def _field_svg() -> str:
    out = [_poly([(-2.2, Y_MIN), (0, Y_MIN), (0, Y_MAX), (-2.2, Y_MAX)], SIDELINE_FILL),
           _poly([(FIELD_W, Y_MIN), (FIELD_W + 2.2, Y_MIN), (FIELD_W + 2.2, Y_MAX), (FIELD_W, Y_MAX)], SIDELINE_FILL),
           _poly([(0, Y_MIN), (FIELD_W, Y_MIN), (FIELD_W, Y_MAX), (0, Y_MAX)], FIELD_FILL)]
    for y in range(Y_MIN, Y_MAX + 1, 5):
        if y != 0:
            out.append(_seg(0, y, FIELD_W, y, LINE, 1.4 if y % 10 == 0 else 0.8))
        if y > 0 and y % 10 == 0:
            for x, anchor in ((-3.4, "end"), (FIELD_W + 3.4, "start")):
                px, py, s = _project(x, y)
                out.append(f'<text x="{float(px):.1f}" y="{float(py):.1f}" text-anchor="{anchor}" '
                           f'dominant-baseline="central" font-size="{12 * (0.8 + 0.2 * float(s)):.1f}" '
                           f'font-weight="600" fill="currentColor" opacity=".7">+{y}</text>')
    for y in range(Y_MIN + 1, Y_MAX):
        if y % 5 == 0:
            continue
        out.append(_seg(0, y, 0.9, y, TICK, 0.8))
        out.append(_seg(FIELD_W - 0.9, y, FIELD_W, y, TICK, 0.8))
        for xc in HASH_X:
            out.append(_seg(xc - 0.55, y, xc + 0.55, y, TICK, 0.9))
    # Dashed lines where the left / middle / right thirds meet (the pass directions).
    for xc in (FIELD_W / 3, 2 * FIELD_W / 3):
        out.append(_seg(xc, Y_MIN, xc, Y_MAX, TICK, 1).replace("/>", ' stroke-dasharray="4 5"/>'))
    out.append(_seg(-2.2, 0, FIELD_W + 2.2, 0, BLUE, 3))
    for x, anchor in ((-3.4, "end"), (FIELD_W + 3.4, "start")):
        px, py, _ = _project(x, 0)
        out.append(f'<text x="{float(px):.1f}" y="{float(py):.1f}" text-anchor="{anchor}" dominant-baseline="central" '
                   f'font-size="12" font-weight="600" fill="currentColor" opacity=".7">LOS</text>')
    for i, name in enumerate(("Left", "Middle", "Right")):
        px, py, _ = _project((i + 0.5) * FIELD_W / 3, Y_MIN)
        out.append(f'<text x="{float(px):.1f}" y="{float(py) + 13:.1f}" text-anchor="middle" font-size="11" '
                   f'fill="currentColor" opacity=".55">{name}</text>')
    return "".join(out)


def _throws_svg(t: pd.DataFrame, ink: str, bg: str) -> str:
    t = t[t["x"].notna()].sort_values("y", ascending=False)
    n = len(t)
    base_r = 9.0 if n <= 40 else 7.5 if n <= 90 else 6.0 if n <= 250 else 4.5
    px, py, s = _project(t["x"].to_numpy(), t["y"].to_numpy())
    out = []
    for o, x, y, sc in zip(t["outcome"], px, py, s):
        rad = base_r * (0.7 + 0.3 * sc)
        if o == "Incomplete":
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rad:.1f}" fill="{bg}" fill-opacity=".55" '
                       f'stroke="{ink}" stroke-width="{1.4 if n > 250 else 2}" stroke-opacity=".85"/>')
        else:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rad:.1f}" fill="{OUTCOME_COLOR[o]}" '
                       f'stroke="{bg}" stroke-width="1.2"/>')
    return "".join(out)


def _legend_html() -> str:
    dot = lambda style: (f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;'
                         f'box-sizing:border-box;{style}"></span>')
    items = [(dot(f"background:{GREEN}"), "Complete"), (dot(f"background:{BLUE}"), "Touchdown"),
             (dot("border:2px solid currentColor;opacity:.85"), "Incomplete"), (dot(f"background:{RED}"), "Interception")]
    return ('<div style="display:flex;flex-wrap:wrap;gap:16px;font-size:12px;margin-top:6px">'
            + "".join(f'<span style="display:inline-flex;align-items:center;gap:6px">{d}{lab}</span>' for d, lab in items)
            + "</div>")


def pass_chart_html(team_d: pd.DataFrame, team: str, league_d: pd.DataFrame | None = None, ink: str = "#31333F",
                    bg: str = "#FFFFFF") -> str:
    t = throws(team_d)
    if t.empty:
        return ""
    s = stat_line(t)
    lg = stat_line(throws(league_d)) if league_d is not None and len(league_d) else None
    muted = "opacity:.65"
    pct = f"{s['comp'] / s['att']:.0%}" if s["att"] else "—"
    adot = "—" if np.isnan(s["adot"]) else f"{s['adot']:.1f}"
    nfl = {"Comp %": f"NFL {lg['comp'] / lg['att']:.0%}" if lg and lg["att"] else "",
           "Avg depth": f"NFL {lg['adot']:.1f}" if lg and not np.isnan(lg["adot"]) else ""}
    kpis = [("Comp / Att", f"{s['comp']}/{s['att']}"), ("Comp %", pct), ("Yards", f"{s['yards']:,}"),
            ("TD-INT", f"{s['td']}-{s['int']}"), ("Avg depth", f"{adot} yds")]
    kpi_html = "".join(
        f'<div style="background:rgba(128,128,128,.08);border-radius:8px;padding:6px 10px">'
        f'<div style="font-size:12px;{muted}">{lab}</div>'
        f'<div style="font-size:20px;font-weight:500;font-variant-numeric:tabular-nums">{val}</div>'
        f'<div style="font-size:11px;{muted}">{nfl.get(lab, "")}</div></div>' for lab, val in kpis)
    notes = ["Depth is air yards; side is nflfastR's pass direction, so a dot's spot within its third is approximate."]
    if s["plotted"] < s["att"]:
        notes.append(f"{s['att'] - s['plotted']} throw(s) without a direction or depth are counted but not drawn.")
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VB_W} {VB_H}" '
           f'font-family="Source Sans Pro, -apple-system, Segoe UI, Helvetica, Arial, sans-serif" color="{ink}">'
           f'{_field_svg().replace("currentColor", ink)}{_throws_svg(t, ink, bg)}</svg>')
    b64 = base64.b64encode(svg.encode()).decode("ascii")
    return (f'<div style="max-width:820px;font-family:inherit;color:{ink}">'
            f'<div style="display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin-bottom:10px">'
            f'{kpi_html}</div>'
            f'<img src="data:image/svg+xml;base64,{b64}" alt="{html.escape(team)} pass chart" '
            f'style="width:100%;height:auto;display:block">{_legend_html()}'
            f'<div style="font-size:12px;{muted};margin-top:6px">{" ".join(notes)}</div></div>')
