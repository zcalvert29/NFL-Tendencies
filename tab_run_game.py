"""
Run game tab: the Dowling run-gap diagram drawn on nflfastR's seven run lanes (end / tackle / guard each side,
plus middle), each lane next to the NFL's EPA in the same lane, then the run tables.
"""
from __future__ import annotations

import base64

import pandas as pd
import streamlit as st

import common as cm
import data as dt

# lane -> (bend x, arrow tip x, short label). Bends sit just behind the line at y=275; arrows end at y=110.
# Each lane points at the lineman it's run over (end = outside the tackle), 90 px apart so the labels have room.
CX = 380                                    # center / RB x
LANES = {
    "Left end": (150, 75, "L end"), "Left tackle": (200, 200, "L tackle"), "Left guard": (290, 290, "L guard"),
    "Middle": (CX, CX, "Middle"),
    "Right guard": (470, 470, "R guard"), "Right tackle": (560, 560, "R tackle"), "Right end": (610, 685, "R end"),
}
LINEMEN = [("LT", 200), ("LG", 290), ("C", CX), ("RG", 470), ("RT", 560)]
NONE_C, LOW_C = "rgba(128,128,128,.45)", "#9A9890"
GAP_MIN_CARRIES = 8
EPA_MARGIN = 0.05      # a lane is green/red when its EPA is this far above/below the NFL's in the same lane


def lane_stats(runs: pd.DataFrame) -> pd.DataFrame:
    g = runs[runs["Run Lane"].notna()].groupby("Run Lane")
    return pd.DataFrame({"n": g.size(), "epa": g["epa"].mean(), "ypc": g["yards_gained"].mean(),
                         "success": g["success"].mean()}).reindex(dt.RUN_LANES)


def run_gaps_svg(runs: pd.DataFrame, lg_runs: pd.DataFrame, ink: str = "#31333F") -> str:
    font = 'font-family="Source Sans Pro, Segoe UI, Helvetica, Arial, sans-serif"'
    num = f'{font} font-size="14" font-weight="600"'
    small = f'{font} font-size="11" fill="{ink}" fill-opacity=".6"'
    lab = f'{font} font-size="12" fill="{ink}" fill-opacity=".7"'
    td = int(runs["touchdown"].sum())
    stats = [(f"{len(runs)}", "carries", ""), (f"{runs['yards_gained'].sum():.0f}", "yards", ""),
             (f"{runs['yards_gained'].mean():.1f}", "yds / carry", f"NFL {lg_runs['yards_gained'].mean():.1f}"),
             (f"{td}", "rush TD", ""), (f"{runs['explosive'].mean():.0%}", "explosive (10+)", f"NFL {lg_runs['explosive'].mean():.0%}"),
             (f"{runs['epa'].mean():+.2f}", "EPA / rush", f"NFL {lg_runs['epa'].mean():+.2f}")]
    parts = []
    for i, (value, label, nfl) in enumerate(stats):
        x = 85 + i * 118
        parts.append(f'<text x="{x}" y="24" text-anchor="middle" fill="{ink}" {num}>{value}</text>'
                     f'<text x="{x}" y="41" text-anchor="middle" {lab}>{label}</text>'
                     f'<text x="{x}" y="55" text-anchor="middle" {small}>{nfl}</text>')
    parts.append(f'<line x1="30" y1="64" x2="730" y2="64" stroke="{ink}" stroke-opacity=".2"/>')

    s, ls = lane_stats(runs), lane_stats(lg_runs)
    max_n = max(s["n"].max() if s["n"].notna().any() else 1, 1)
    for lane, (bend_x, tip_x, name) in LANES.items():
        n = 0 if pd.isna(s.at[lane, "n"]) else int(s.at[lane, "n"])
        epa, nfl = s.at[lane, "epa"], ls.at[lane, "epa"]
        if n == 0:
            color = NONE_C
        elif n < GAP_MIN_CARRIES or pd.isna(nfl):
            color = LOW_C
        elif epa >= nfl + EPA_MARGIN:
            color = cm.GREEN
        elif epa <= nfl - EPA_MARGIN:
            color = cm.RED
        else:
            color = "#888780"
        dash = ' stroke-dasharray="6 6"' if n == 0 else ""
        width = 3 + 7 * n / max_n
        parts.append(f'<path d="M{CX} 308 L{bend_x} 275 L{tip_x} 110" fill="none" stroke="{color}" '
                     f'stroke-width="{width:.1f}" stroke-linecap="round" stroke-linejoin="round"{dash}/>')
        dx, dy = tip_x - bend_x, 110 - 275
        length = (dx * dx + dy * dy) ** 0.5
        ux, uy = dx / length, dy / length
        bx, by = tip_x - 14 * ux, 110 - 14 * uy
        px, py = -uy * 8, ux * 8
        parts.append(f'<polygon points="{tip_x + 2 * ux:.1f},{110 + 2 * uy:.1f} {bx + px:.1f},{by + py:.1f} '
                     f'{bx - px:.1f},{by - py:.1f}" fill="{color}"/>')
        value = "–" if n == 0 else f"{epa:+.2f}"
        parts.append(f'<text x="{tip_x}" y="84" text-anchor="middle" fill="{color}" {num}>{value}</text>'
                     f'<text x="{tip_x}" y="98" text-anchor="middle" {small}>NFL {nfl:+.2f}</text>')
        share = f"{n / s['n'].sum():.0%}" if s["n"].sum() else ""
        parts.append(f'<text x="{tip_x}" y="388" text-anchor="middle" fill="{color}" {num}>{n}</text>'
                     f'<text x="{tip_x}" y="404" text-anchor="middle" {lab}>{name}</text>'
                     f'<text x="{tip_x}" y="418" text-anchor="middle" {small}>{share}</text>')
    for name, x in LINEMEN:
        parts.append(f'<circle cx="{x}" cy="240" r="21" fill="{cm.BLUE}"/>'
                     f'<text x="{x}" y="240" text-anchor="middle" dominant-baseline="central" fill="#E6F1FB" {num}>{name}</text>')
    parts.append(f'<circle cx="{CX}" cy="330" r="22" fill="{cm.RUN_C}"/><text x="{CX}" y="330" text-anchor="middle" '
                 f'dominant-baseline="central" fill="#FAEEDA" {num}>RB</text>')
    parts.append(f'<line x1="30" y1="364" x2="730" y2="364" stroke="{ink}" stroke-opacity=".2"/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="430" viewBox="0 0 760 430">'
           f'{"".join(parts)}</svg>')
    b64 = base64.b64encode(svg.encode()).decode("ascii")   # st.html strips inline <svg>, so embed it as an image
    return (f'<img src="data:image/svg+xml;base64,{b64}" alt="EPA per rush and carries by run lane" '
            f'style="width:100%;max-width:860px;height:auto">')


RUN_MEASURES = ["Plays", "Share", "Avg Yards Gained", "Success Rate", "NFL Success Rate", "EPA per Play",
                "NFL EPA per Play", "Explosive Rate"]


def render(team_d: pd.DataFrame, league_d: pd.DataFrame, team: str) -> None:
    runs = team_d[team_d["PASS"] == 0]
    lg_runs = league_d[league_d["PASS"] == 0]
    st.markdown(f"**{team} run game by lane**")
    if runs.empty:
        st.info("No runs match the current filters.")
        return
    st.html(run_gaps_svg(runs, lg_runs, cm.theme_ink()))
    missing = int(runs["Run Lane"].isna().sum())
    st.caption(f"Arrow number = EPA per carry in that lane; under it, the NFL's EPA per carry in the same lane. Green/red "
               f"= at least {EPA_MARGIN:.2f} better/worse than the NFL there; gray = about the same, or fewer than "
               f"{GAP_MIN_CARRIES} carries. Thicker arrows = more carries. Lanes are nflfastR's run location and gap "
               f"(end = outside the tackle)." + (f" {missing} runs without a direction (mostly QB runs) are in the "
                                                  f"totals but not the arrows." if missing else ""))

    lanes = cm.crosstab(runs, "Run Lane", RUN_MEASURES, lg_runs, order=dt.RUN_LANES)
    cm.render_takeaways(lanes, "run lane")
    cm.show_table(f"{team} runs by lane", lanes)
    cm.render_usage_scatter(lanes, f"{team} run lanes: EPA vs success", noun="Run lane", min_plays=5,
                            nfl_epa=lg_runs["epa"].mean())
    # Full width, not side by side: half-width tables cut off the NFL columns.
    cm.show_table(f"{team} runs by formation", cm.crosstab(runs, "Formation", RUN_MEASURES, lg_runs, sort_by_count=True))
    cm.show_table(f"{team} runs by distance to go",
                  cm.crosstab(runs, "Distance", RUN_MEASURES, lg_runs, order=dt.DISTANCE_ORDER))
    if cm.charting_note(runs, "Personnel", "personnel"):
        cm.show_table(f"{team} runs by personnel", cm.crosstab(runs, "Personnel", RUN_MEASURES, lg_runs,
                                                               sort_by_count=True))
        cm.show_table(f"{team} runs vs men in the box", cm.crosstab(runs, "Box", RUN_MEASURES, lg_runs,
                                                                    order=["5 or fewer", "6", "7", "8+"]))
    carriers = cm.crosstab(runs.assign(Rusher=runs["rusher_player_name"]), "Rusher",
                           ["Plays", "Share", "Avg Yards Gained", "Success Rate", "EPA per Play", "Explosive Rate"],
                           sort_by_count=True)
    cm.show_table(f"{team} ball carriers", carriers.head(8))
    cm.show_plays(f"{team} best runs", runs.nlargest(8, "epa"))
