"""
Pictures: pass-zone heatmap, run-lane diagram, weekly trends, league scatter, percentile profiles, and win
probability charts.
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import data as dt
import tables as tb

# (background, text) from bad to good for the offense; index 3 is "about average".
GOOD_BAD_STOPS = [
    ("#A32D2D", "#FCEBEB"), ("#E24B4A", "#501313"), ("#F7C1C1", "#501313"),
    ("#F1EFE8", "#2C2C2A"),
    ("#9FE1CB", "#04342C"), ("#1D9E75", "#04342C"), ("#0F6E56", "#E1F5EE"),
]
# Volume (share of throws / carries): blue below an even split, red above.
SHARE_STOPS = [
    ("#185FA5", "#E6F1FB"), ("#378ADD", "#042C53"), ("#B5D4F4", "#042C53"),
    ("#F1EFE8", "#2C2C2A"),
    ("#F7C1C1", "#501313"), ("#E24B4A", "#501313"), ("#A32D2D", "#FCEBEB"),
]
EMPTY = ("rgba(128,128,128,0.12)", "inherit")
ZONE_METRICS = {   # label -> (column, formatter)
    "Share of throws": ("share", lambda v: f"{v:.0%}"),
    "EPA per play": ("epa", lambda v: f"{v:+.2f}"),
    "Success rate": ("success", lambda v: f"{v:.0%}"),
    "Completion %": ("comp", lambda v: f"{v:.0%}"),
    "Explosive rate": ("explosive", lambda v: f"{v:.0%}"),
}
LOW_N = 15


def _fmt(fn, v) -> str:
    return "–" if pd.isna(v) else fn(v)


def _color(value, ref, spread, good_high=True, share=False, n=None):
    """Small samples (under LOW_N) get at most the lightest shade, so one 4-throw zone can't shout."""
    if pd.isna(value) or pd.isna(ref) or not spread:
        return EMPTY
    t = float(np.clip((value - ref) / spread, -1, 1))
    a = abs(t)
    step = 0 if a < 0.2 else 1 if a < 0.5 else 2 if a < 0.8 else 3
    if n is not None and n < LOW_N and not share:
        step = min(step, 1)
    if share:
        return SHARE_STOPS[3 + int(np.sign(t)) * step]
    return GOOD_BAD_STOPS[3 + int(np.sign(t)) * (1 if good_high else -1) * step]


def _group_stats(d: pd.DataFrame, key: str) -> pd.DataFrame:
    g = d.groupby(key)
    s = pd.DataFrame({"n": g.size(), "epa": g["epa"].mean(), "success": g["success"].mean(),
                      "comp": g["completion"].mean(), "explosive": g["explosive"].mean(),
                      "ypp": g["yards_gained"].mean()})
    s["share"] = s["n"] / s["n"].sum()
    return s


# ---- Pass zones --------------------------------------------------------------------------------------------
def render_pass_zones(d: pd.DataFrame, title: str, key: str, good_high: bool = True, league: pd.DataFrame | None = None):
    """
    Where the throws go: nflfastR's pass direction (left / middle / right) by air yards. Colors compare each
    zone to the league's numbers in the same zone when `league` is given, otherwise to the team's own average.
    """
    st.markdown(f"**{title}**")
    d = d[d["is_pass"] & d["Pass Zone"].notna()]
    if d.empty:
        st.info("No charted throws match the current filters.")
        return
    metric = st.segmented_control("Color by", list(ZONE_METRICS), default="EPA per play", key=key) or "EPA per play"
    col, fn = ZONE_METRICS[metric]
    stats = _group_stats(d, "Pass Zone")
    ref_stats = _group_stats(league[league["is_pass"] & league["Pass Zone"].notna()], "Pass Zone") \
        if league is not None and not league.empty else None
    overall = {"epa": d["epa"].mean(), "success": d["success"].mean(), "comp": d["completion"].mean(),
               "explosive": d["explosive"].mean(), "share": 1 / 12}
    zones = [f"{r} {c}" for r in dt.AIR_ROWS for c in dt.PASS_DIRECTIONS]
    stats = stats.reindex(zones)
    stats["n"] = stats["n"].fillna(0).astype(int)
    if ref_stats is not None:
        ref_stats = ref_stats.reindex(zones)
        refs = ref_stats[col]
    else:
        refs = pd.Series(overall[col], index=zones)
    sized = stats[stats["n"] >= LOW_N]
    spread = (sized[col] - refs[sized.index]).abs().max() if not sized.empty else None

    muted = "opacity:.65"
    cells = ['<div></div>'] + [
        f'<div style="text-align:center;font-size:13px">{c}<br><span style="font-size:12px;{muted}">'
        f'{stats.loc[[f"{r} {c}" for r in dt.AIR_ROWS], "share"].sum():.0%} of throws</span></div>'
        for c in dt.PASS_DIRECTIONS]
    for r in dt.AIR_ROWS:
        cells.append(f'<div style="display:flex;flex-direction:column;justify-content:center;text-align:right;'
                     f'font-size:12px;padding-right:6px;{muted}">{r}<span>'
                     f'{stats.loc[[f"{r} {c}" for c in dt.PASS_DIRECTIONS], "share"].sum():.0%}</span></div>')
        for c in dt.PASS_DIRECTIONS:
            z = f"{r} {c}"
            row = stats.loc[z]
            n = int(row["n"])
            bg, fg = _color(row[col], refs[z], spread, good_high, share=(col == "share"), n=n) if n >= 3 else EMPTY
            vs = ""
            if ref_stats is not None and col != "share" and not pd.isna(row[col]):
                vs = f'<span style="font-size:11px;opacity:.8">NFL {fn(refs[z])}</span>'
            flag = " · low n" if 0 < n < LOW_N else ""
            others = "".join(f"<span>{lbl.split()[0]} {_fmt(f, row[c2])}</span>"
                             for lbl, (c2, f) in ZONE_METRICS.items() if c2 != col)
            cells.append(
                f'<div style="background:{bg};color:{fg};border-radius:8px;padding:8px 10px;min-height:104px;'
                f'display:flex;flex-direction:column;justify-content:space-between">'
                f'<div style="display:flex;justify-content:space-between;font-size:12px;opacity:.85">'
                f'<span>{n} att{flag}</span>{vs}</div>'
                f'<div style="font-size:22px;font-weight:500;font-variant-numeric:tabular-nums">{_fmt(fn, row[col])}</div>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:1px 8px;font-size:11px;'
                f'font-variant-numeric:tabular-nums">{others}</div></div>')
    st.html(
        '<div style="max-width:820px;font-family:inherit">'
        f'<div style="display:grid;grid-template-columns:70px repeat(3,minmax(0,1fr));gap:4px">{"".join(cells)}</div>'
        '<div style="display:grid;grid-template-columns:70px 1fr;gap:4px;margin-top:4px">'
        f'<div style="font-size:12px;text-align:right;padding-right:6px;{muted}">LOS</div>'
        '<div style="border-top:2px solid rgba(128,128,128,.6);text-align:center;font-size:12px;padding-top:4px">QB</div>'
        '</div></div>')
    vs_txt = "the league's number in the same zone" if ref_stats is not None else "the team's own average"
    side = "offense" if good_high else "defense"
    st.caption(f"{len(d):,} throws with a direction and air yards (sacks and throwaways without a target are left "
               f"out). {'Share is compared to an even split.' if col == 'share' else f'Green = better for the {side} than {vs_txt}.'}")


# ---- Run lanes ---------------------------------------------------------------------------------------------
def render_run_lanes(d: pd.DataFrame, title: str, good_high: bool = True, league: pd.DataFrame | None = None):
    """Seven lanes (ends, tackles, guards, middle) from nflfastR's run location + gap."""
    st.markdown(f"**{title}**")
    d = d[d["is_run"] & d["Run Lane"].notna()]
    if d.empty:
        st.info("No charted runs match the current filters.")
        return
    stats = _group_stats(d, "Run Lane").reindex(dt.RUN_LANES)
    stats["n"] = stats["n"].fillna(0).astype(int)
    if league is not None and not league.empty:
        ref = _group_stats(league[league["is_run"] & league["Run Lane"].notna()], "Run Lane").reindex(dt.RUN_LANES)["epa"]
    else:
        ref = pd.Series(d["epa"].mean(), index=dt.RUN_LANES)
    sized = stats[stats["n"] >= LOW_N]
    spread = (sized["epa"] - ref[sized.index]).abs().max() if not sized.empty else None
    lanes = []
    for lane in dt.RUN_LANES:
        r = stats.loc[lane]
        n = int(r["n"])
        bg, fg = _color(r["epa"], ref[lane], spread, good_high, n=n) if n >= 3 else EMPTY
        lanes.append(
            f'<div style="background:{bg};color:{fg};border-radius:8px;padding:8px 6px;text-align:center;'
            f'font-variant-numeric:tabular-nums">'
            f'<div style="font-size:12px;opacity:.85">{lane.replace("Left ", "L ").replace("Right ", "R ")}</div>'
            f'<div style="font-size:20px;font-weight:500">{_fmt(lambda v: f"{v:+.2f}", r["epa"])}</div>'
            f'<div style="font-size:11px">{n} car · {r["share"]:.0%}</div>'
            f'<div style="font-size:11px">{_fmt(lambda v: f"{v:.1f}", r["ypp"])} ypc · {_fmt(lambda v: f"{v:.0%}", r["success"])} succ</div>'
            f'<div style="font-size:10px;opacity:.75">NFL {_fmt(lambda v: f"{v:+.2f}", ref[lane])}</div></div>')
    line = "".join(f'<div style="text-align:center;font-size:12px;font-weight:500;padding:4px 0;'
                   f'border-radius:6px;background:rgba(128,128,128,.15)">{p}</div>'
                   for p in ["TE", "LT", "LG", "C", "RG", "RT", "TE"])
    st.html('<div style="max-width:900px">'
            f'<div style="display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:4px">{"".join(lanes)}</div>'
            f'<div style="display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:4px;margin-top:6px">{line}</div>'
            '</div>')
    st.caption("EPA per carry by lane (nflfastR's run location and gap; scrambles left out). End = outside the "
               "tackle. Colored against the league's EPA in the same lane; \"NFL\" is that number.")


# ---- Trends -------------------------------------------------------------------------------------------------
def weekly_trend(team_d: pd.DataFrame, league_d: pd.DataFrame, metric: str, title: str, fmt: str, color: str):
    """One line per week for the team, with the league's weekly average as a dashed line."""
    st.markdown(f"**{title}**")
    if team_d.empty:
        st.info("No plays match the current filters.")
        return
    f = tb.METRICS[metric]
    key = ["Season", "Week"]
    t = team_d.groupby(key).apply(f, include_groups=False).rename("Team").reset_index()
    opp = team_d.groupby(key)["Opponent"].first().reset_index()
    t = t.merge(opp, on=key)
    lg = league_d.groupby(key).apply(f, include_groups=False).rename("NFL average").reset_index()
    t = t.merge(lg, on=key, how="left")
    t["Game"] = t["Season"].astype(str) + " W" + t["Week"].astype(str).str.zfill(2)
    long = t.melt(id_vars=["Game", "Opponent"], value_vars=["Team", "NFL average"], var_name="Series", value_name=metric)
    chart = alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("Game:N", sort=sorted(t["Game"]), title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y(f"{metric}:Q", axis=alt.Axis(format=fmt), title=metric),
        color=alt.Color("Series:N", scale=alt.Scale(domain=["Team", "NFL average"], range=[color, "#888780"]),
                        legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.StrokeDash("Series:N", scale=alt.Scale(domain=["Team", "NFL average"], range=[[1, 0], [5, 4]]),
                                  legend=None),
        tooltip=["Game", "Opponent", "Series", alt.Tooltip(f"{metric}:Q", format=fmt)],
    )
    st.altair_chart(chart.properties(height=280), width="stretch")


def league_scatter(teams: pd.DataFrame, x: str, y: str, x_title: str, y_title: str, highlight: list[str],
                   colors: dict, fmt: str = "+.2f", flip_x: bool = False, flip_y: bool = False):
    """Every team as a labeled dot (team colors), with league-average lines."""
    data = teams.reset_index().rename(columns={"index": "Team"})
    data["color"] = data["Team"].map(colors).fillna("#888780")
    data["size"] = np.where(data["Team"].isin(highlight), 260, 90)
    data["opacity"] = np.where(data["Team"].isin(highlight) | (len(highlight) == 0), 1.0, 0.45)
    base = alt.Chart(data).encode(
        x=alt.X(f"{x}:Q", title=x_title, axis=alt.Axis(format=fmt), scale=alt.Scale(zero=False, reverse=flip_x)),
        y=alt.Y(f"{y}:Q", title=y_title, axis=alt.Axis(format=fmt), scale=alt.Scale(zero=False, reverse=flip_y)),
        tooltip=["Team", alt.Tooltip(f"{x}:Q", format=fmt, title=x_title), alt.Tooltip(f"{y}:Q", format=fmt, title=y_title)],
    )
    pts = base.mark_circle(stroke="white", strokeWidth=1).encode(
        color=alt.Color("color:N", scale=None), size=alt.Size("size:Q", scale=None, legend=None),
        opacity=alt.Opacity("opacity:Q", scale=None))
    labels = base.mark_text(dy=-12, fontSize=11, fontWeight="bold").encode(text="Team", opacity=alt.Opacity("opacity:Q", scale=None))
    rule = lambda **enc: alt.Chart(pd.DataFrame({k: [v] for k, v in enc.items()})).mark_rule(
        strokeDash=[5, 4], color="#888780").encode(**{k: f"{k}:Q" for k in enc})
    st.altair_chart(alt.layer(rule(x=data[x].mean()), rule(y=data[y].mean()), pts, labels).properties(height=520),
                    width="stretch")


# ---- Team profile (percentile bars) -------------------------------------------------------------------------
def percentile_bars(rows: pd.DataFrame, title: str, color: str):
    """rows: index = metric label, columns Value (formatted text) and Pct (0-100, higher = better)."""
    st.markdown(f"**{title}**")
    data = rows.reset_index().rename(columns={"index": "Metric"})
    data["Band"] = pd.cut(data["Pct"], [-1, 33, 66, 101], labels=["Bottom third", "Middle", "Top third"]).astype(str)
    order = list(data["Metric"])
    base = alt.Chart(data).encode(y=alt.Y("Metric:N", sort=order, title=None, axis=alt.Axis(labelLimit=200)))
    bg = alt.Chart(data).mark_bar(color="rgba(128,128,128,0.15)").encode(
        y=alt.Y("Metric:N", sort=order), x=alt.X("full:Q", scale=alt.Scale(domain=[0, 100]))
    ).transform_calculate(full="100")
    bars = base.mark_bar(color=color).encode(
        x=alt.X("Pct:Q", scale=alt.Scale(domain=[0, 100]), title="League percentile (100 = best)"),
        tooltip=["Metric", "Value", alt.Tooltip("Pct:Q", format=".0f", title="Percentile"), "Rank"])
    text = base.mark_text(align="left", dx=4, fontSize=11).encode(x="Pct:Q", text="Value:N")
    mid = alt.Chart(pd.DataFrame({"x": [50]})).mark_rule(strokeDash=[4, 3], color="#888780").encode(x="x:Q")
    st.altair_chart(alt.layer(bg, bars, text, mid).properties(height=26 * len(data) + 40), width="stretch")


# ---- Win probability ----------------------------------------------------------------------------------------
def wp_chart(g: pd.DataFrame, home: str, away: str, home_color: str, away_color: str):
    """Home win probability through the game (nflfastR's wp), with scoring plays marked."""
    g = g[g["home_wp"].notna() & g["play_type"].notna()].copy()
    if g.empty:
        st.info("No win probability for this game yet.")
        return
    g["Minute"] = (3600 - g["game_seconds_remaining"]) / 60
    g["Home WP"] = g["home_wp"]
    g["Play"] = g["desc"].str.slice(0, 140)
    g["Score"] = g["away_team"] + " " + g["total_away_score"].astype(int).astype(str) + " – " + \
        g["home_team"] + " " + g["total_home_score"].astype(int).astype(str)
    # Several plays can share a clock time (penalties, timeouts), so keep the play order rather than sorting by x.
    g["order"] = range(len(g))
    area = alt.Chart(g).mark_line(color=home_color, strokeWidth=2).encode(
        x=alt.X("Minute:Q", scale=alt.Scale(domain=[0, max(60, g["Minute"].max())]),
                axis=alt.Axis(values=[0, 15, 30, 45, 60], title="Game minute")),
        y=alt.Y("Home WP:Q", scale=alt.Scale(domain=[0, 1]), stack=None, axis=alt.Axis(format=".0%", title=f"{home} win probability")),
        order="order:Q",
        tooltip=[alt.Tooltip("Minute:Q", format=".1f"), alt.Tooltip("Home WP:Q", format=".0%"), "Score", "Play"])
    mid = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[4, 3], color="#888780").encode(y="y:Q")
    scores = g[g["touchdown"].eq(1) | g["field_goal_result"].eq("made")]
    pts = alt.Chart(scores).mark_point(filled=True, size=70, color=away_color).encode(
        x="Minute:Q", y="Home WP:Q", tooltip=["Score", "Play"])
    st.altair_chart(alt.layer(area, mid, pts).properties(height=320), width="stretch")
    st.caption(f"Above 50% = {home} favored. Dots are scores. From nflfastR's win probability model.")
