"""
Pass game tab: pass-zone heatmap (each zone next to the NFL's numbers in the same zone), the pass chart, and the
pass tables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import common as cm
import data as dt
import pass_chart as pc

_pct = lambda v: f"{v:.0%}"
_epa = lambda v: f"{v:+.2f}"
ZONE_METRICS = {   # label -> (short label, column, formatter)
    "Share of throws": ("Share", "share", lambda v: f"{v:.1%}"),
    "Completion %": ("Comp", "comp", _pct),
    "EPA per play": ("EPA", "epa", _epa),
    "Success rate": ("Succ", "success", _pct),
    "Explosive rate": ("Expl", "explosive", _pct),
}
LOW_N = 10
# Blue (below) -> gray -> red (above) for volume; red (worse for the offense) -> gray -> green (better) otherwise.
SHARE_STOPS = [("#185FA5", "#E6F1FB"), ("#378ADD", "#042C53"), ("#B5D4F4", "#042C53"), ("#F1EFE8", "#2C2C2A"),
               ("#F7C1C1", "#501313"), ("#E24B4A", "#501313"), ("#A32D2D", "#FCEBEB")]
GOOD_BAD_STOPS = [("#A32D2D", "#FCEBEB"), ("#E24B4A", "#501313"), ("#F7C1C1", "#501313"), ("#F1EFE8", "#2C2C2A"),
                  ("#9FE1CB", "#04342C"), ("#1D9E75", "#04342C"), ("#0F6E56", "#E1F5EE")]
EMPTY = ("rgba(128,128,128,0.12)", "inherit")
ZONES = [(depth, side) for depth in dt.DEPTHS for side in dt.PASS_DIRECTIONS]


def _attempts(d: pd.DataFrame) -> pd.DataFrame:
    return d[(d["PASS"] == 1) & d["Pass Direction"].notna() & d["Depth"].notna()]


def zone_stats(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    g = d.groupby(["Depth", "Pass Direction"])
    s = pd.DataFrame({"att": g.size(), "comp": g["completion"].mean(), "epa": g["epa"].mean(),
                      "success": g["success"].mean(), "explosive": g["explosive"].mean()}).reindex(ZONES)
    s["att"] = s["att"].fillna(0).astype(int)
    s["share"] = s["att"] / max(s["att"].sum(), 1)
    overall = {"att": len(d), "comp": d["completion"].mean(), "epa": d["epa"].mean(), "success": d["success"].mean(),
               "explosive": d["explosive"].mean()}
    return s, overall


def _color(value, ref, spread, share: bool, n: int):
    if pd.isna(value) or pd.isna(ref) or not spread:
        return EMPTY
    t = float(np.clip((value - ref) / spread, -1, 1))
    step = 0 if abs(t) < 0.2 else 1 if abs(t) < 0.5 else 2 if abs(t) < 0.8 else 3
    if n < LOW_N and not share:
        step = min(step, 1)   # a few throws can't paint a zone dark
    return (SHARE_STOPS if share else GOOD_BAD_STOPS)[3 + int(np.sign(t)) * step]


def pass_zone_html(team_d: pd.DataFrame, league_d: pd.DataFrame, metric: str) -> str:
    d, lg = _attempts(team_d), _attempts(league_d)
    if d.empty:
        return ""
    short, col, fn = ZONE_METRICS[metric]
    s, overall = zone_stats(d)
    ls, lov = zone_stats(lg)
    ref = ls[col]                                   # each zone vs the NFL in the same zone
    sized = s[s["att"] >= LOW_N]
    spread = (sized[col] - ref[sized.index]).abs().max() if not sized.empty else None
    muted = "opacity:.65"
    nfl_line = {"comp": lov["comp"], "epa": lov["epa"], "success": lov["success"], "explosive": lov["explosive"]}
    kpis = [("Attempts", str(overall["att"]), ""), ("Comp %", _pct(overall["comp"]), f"NFL {_pct(nfl_line['comp'])}"),
            ("EPA / play", _epa(overall["epa"]), f"NFL {_epa(nfl_line['epa'])}"),
            ("Success", _pct(overall["success"]), f"NFL {_pct(nfl_line['success'])}"),
            ("Explosive", _pct(overall["explosive"]), f"NFL {_pct(nfl_line['explosive'])}")]
    kpi_html = "".join(f'<div style="background:rgba(128,128,128,.08);border-radius:8px;padding:6px 10px">'
                       f'<div style="font-size:12px;{muted}">{l}</div>'
                       f'<div style="font-size:20px;font-weight:500;font-variant-numeric:tabular-nums">{v}</div>'
                       f'<div style="font-size:11px;{muted}">{n}</div></div>' for l, v, n in kpis)

    def share_of(zones):
        return f'{s.loc[zones, "share"].sum():.0%}'

    cells = ['<div></div>'] + [
        f'<div style="text-align:center;font-size:13px">{side}<br><span style="font-size:12px;{muted}">'
        f'{share_of([(dp, side) for dp in dt.DEPTHS])} of throws · NFL {ls.loc[[(dp, side) for dp in dt.DEPTHS], "share"].sum():.0%}'
        f'</span></div>' for side in dt.PASS_DIRECTIONS]
    for depth in dt.DEPTHS:
        row_z = [(depth, side) for side in dt.PASS_DIRECTIONS]
        cells.append(f'<div style="display:flex;flex-direction:column;justify-content:center;align-items:flex-end;'
                     f'text-align:right;font-size:12px;padding-right:6px;{muted}">{depth}<span>{share_of(row_z)}</span>'
                     f'<span>NFL {ls.loc[row_z, "share"].sum():.0%}</span></div>')
        for z in row_z:
            r, n = s.loc[z], int(s.at[z, "att"])
            bg, fg = _color(r[col], ref[z], spread, col == "share", n) if n else EMPTY
            flag = " · low n" if 0 < n < LOW_N else ""
            others = "".join(f"<span>{sh} {'–' if pd.isna(r[c]) else f(r[c])}</span>"
                             for m, (sh, c, f) in ZONE_METRICS.items() if m != metric)
            nfl = "" if pd.isna(ref[z]) else f"NFL {fn(ref[z])}"
            cells.append(
                f'<div style="background:{bg};color:{fg};border-radius:8px;padding:9px 11px;min-height:112px;'
                f'display:flex;flex-direction:column;justify-content:space-between">'
                f'<div style="display:flex;justify-content:space-between;font-size:12px;opacity:.85">'
                f'<span>{n} att{flag}</span><span>{nfl}</span></div>'
                f'<div style="font-size:24px;font-weight:500;font-variant-numeric:tabular-nums">'
                f'{"–" if pd.isna(r[col]) else fn(r[col])}</div>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:1px 8px;font-size:12px;'
                f'font-variant-numeric:tabular-nums">{others}</div></div>')
    swatches = "".join(f'<span style="width:22px;height:12px;border-radius:2px;background:{b};'
                       f'border:0.5px solid rgba(128,128,128,.4)"></span>'
                       for b, _ in (SHARE_STOPS if col == "share" else GOOD_BAD_STOPS))
    low_txt, high_txt = (("Thrown here less than the NFL", "More than the NFL") if col == "share"
                         else ("Worse than the NFL here", "Better than the NFL here"))
    return (f'<div style="max-width:860px;font-family:inherit">'
            f'<div style="display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin-bottom:12px">{kpi_html}</div>'
            f'<div style="display:grid;grid-template-columns:76px repeat(3,minmax(0,1fr));gap:4px">{"".join(cells)}</div>'
            '<div style="display:grid;grid-template-columns:76px 1fr;gap:4px;margin-top:4px">'
            f'<div style="font-size:12px;text-align:right;padding-right:6px;{muted}">LOS</div>'
            '<div style="border-top:2px solid rgba(128,128,128,.6);text-align:center;font-size:12px;padding-top:4px">QB</div></div>'
            f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-size:12px;margin-top:8px">'
            f'<span style="{muted}">{low_txt}</span>{swatches}<span style="{muted}">{high_txt}</span></div></div>')


PASS_MEASURES = ["Plays", "Share", "Comp %", "NFL Comp %", "Success Rate", "NFL Success Rate", "EPA per Play",
                 "NFL EPA per Play", "Explosive Rate"]


def render(team_d: pd.DataFrame, league_d: pd.DataFrame, team: str) -> None:
    passes, lg_passes = team_d[team_d["PASS"] == 1], league_d[league_d["PASS"] == 1]
    st.markdown(f"**{team} pass zones**")
    if passes.empty:
        st.info("No passes match the current filters.")
        return
    metric = st.radio("Shade by", list(ZONE_METRICS), horizontal=True, index=2, key="pz_metric")
    st.html(pass_zone_html(team_d, league_d, metric))
    st.caption(f"Each zone is shaded against the NFL's number in the same zone (shown top right of the zone); share "
               f"of throws is compared to the NFL's share. Zones with fewer than {LOW_N} throws get at most a light "
               f"shade. Depth = air yards; side = nflfastR's pass direction. Sacks and scrambles aren't throws.")

    st.markdown(f"**{team} pass chart**")
    chart = pc.pass_chart_html(team_d, team, league_d, ink=cm.theme_ink(), bg=cm.theme_bg())
    if chart:
        st.html(chart)
    st.caption("Pick one week in the sidebar to see a single game.")

    zones = cm.crosstab(_attempts(passes).assign(Zone=lambda d: d["Depth"] + " " + d["Pass Direction"].str.lower()),
                        "Zone", PASS_MEASURES, _attempts(lg_passes).assign(
                            Zone=lambda d: d["Depth"] + " " + d["Pass Direction"].str.lower()), sort_by_count=True)
    cm.render_takeaways(zones, "zone")
    cm.render_usage_scatter(zones, f"{team} pass zones: EPA vs success", noun="Zone", min_plays=5,
                            nfl_epa=lg_passes["epa"].mean())
    # Full width, not side by side: half-width tables cut off the NFL columns.
    cm.show_table(f"{team} passes by depth", cm.crosstab(passes, "Depth", PASS_MEASURES, lg_passes,
                                                          order=dt.DEPTHS[::-1]))
    cm.show_table(f"{team} dropbacks by formation", cm.crosstab(passes, "Formation", PASS_MEASURES, lg_passes,
                                                                 sort_by_count=True))
    if cm.charting_note(passes, "Coverage", "coverage"):
        cm.show_table(f"{team} vs coverage", cm.crosstab(passes, "Coverage", PASS_MEASURES, lg_passes,
                                                          sort_by_count=True))
        cm.show_table(f"{team} vs man and zone", cm.crosstab(passes, "Man/Zone", PASS_MEASURES, lg_passes))
    tgt = passes[passes["receiver_player_name"].notna()].assign(Target=lambda d: d["receiver_player_name"])
    cm.show_table(f"{team} targets", cm.crosstab(tgt, "Target", ["Plays", "Share", "Comp %", "Success Rate",
                                                                 "EPA per Play", "Explosive Rate"],
                                                 sort_by_count=True).head(10).rename(columns={"Plays": "Targets"}))
    cm.show_plays(f"{team} best passes", passes.nlargest(8, "epa"))
