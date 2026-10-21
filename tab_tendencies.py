"""
Tendencies tab: the Dowling Scout Opposing Offense > Tendencies tab for an NFL offense, with pass rate over expected
(PROE) and the league's numbers in the same situation added to every view.
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import common as cm
import data as dt

TELL_SHARE = 0.75   # down & distance boxes: 75%+ one way on TELL_MIN_N+ plays gets a badge


# ---- Down & distance grid ------------------------------------------------------------------------------------
def dd_grid_html(team_d: pd.DataFrame, league_d: pd.DataFrame) -> str:
    """Downs x distance: run/pass split bar (with the NFL's split marked), PROE, top formation, tell badge."""
    d = team_d[team_d["DN"].notna() & team_d["Distance"].notna()]
    if d.empty:
        return ""
    lg = league_d[league_d["DN"].notna() & league_d["Distance"].notna()]
    nfl_pass = lg.groupby(["DN", "Distance"])["PASS"].mean()
    muted = "opacity:.65"
    cells = ['<div></div>'] + [
        f'<div style="font-size:12px;text-align:center;{muted}">{lab.split(" (")[0]}<br>{lab.split("(")[1].rstrip(")")}</div>'
        for lab in dt.DISTANCE_ORDER]
    for dn in [x for x in (1, 2, 3, 4) if (d["DN"] == x).any()]:
        cells.append(f'<div style="font-weight:600;font-size:14px;display:flex;align-items:center">{dt.DOWN_NAMES[dn]}</div>')
        for dist in dt.DISTANCE_ORDER:
            g = d[(d["DN"] == dn) & (d["Distance"] == dist)]
            if g.empty:
                cells.append(f'<div style="{cm.CARD};font-size:12px;{muted}">No plays</div>')
                continue
            n, pr = len(g), g["PASS"].mean()
            nfl = nfl_pass.get((dn, dist), np.nan)
            lead = f"Pass {pr:.0%}" if pr >= 0.5 else f"Run {1 - pr:.0%}"
            k = round(max(pr, 1 - pr) * n)
            flag = cm.badge(k, n) if max(pr, 1 - pr) >= TELL_SHARE else ""
            low = " · low n" if n < cm.TELL_MIN_N else ""
            forms = g["Formation"].value_counts()
            form_txt = f"{forms.index[0]} · {forms.iat[0] / n:.0%}" if len(forms) else "–"
            pers = g["Personnel"].value_counts()
            if len(pers):
                form_txt += f" · {pers.index[0]} pers. {pers.iat[0] / n:.0%}"
            cells.append(
                f'<div style="{cm.CARD};font-size:12px">'
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:6px">'
                f'<div style="font-weight:600;font-size:13px">{lead}{flag}</div>'
                f'<div style="{muted};white-space:nowrap">{n} plays{low}</div></div>'
                f'{cm.split_bar(1 - pr, None if pd.isna(nfl) else 1 - nfl)}'
                f'<div style="display:flex;justify-content:space-between;gap:6px">'
                f'<span style="{muted}">NFL {nfl:.0%} pass</span>{cm.proe_text(g["pass_oe"].mean() / 100)}</div>'
                f'<div style="{muted};margin-top:2px">{form_txt}</div></div>')
    return (f'<div style="max-width:980px;font-family:inherit">{cm.LEGEND}'
            f'<div style="display:grid;grid-template-columns:48px repeat(4,minmax(0,1fr));gap:6px">'
            f'{"".join(cells)}</div></div>')


def render_dd_grid(team_d, league_d, team):
    st.markdown(f"**{team} down & distance tendencies**")
    html = dd_grid_html(team_d, league_d)
    if not html:
        st.info("No plays match the current filters.")
        return
    st.html(html)
    st.caption(f"Bottom line = most-used formation (and personnel, when charted) and its share of the box. Badge = "
               f"{TELL_SHARE:.0%}+ one way on {cm.TELL_MIN_N}+ plays, graded by how sure the split is. PROE = how much "
               f"more often they throw than an average NFL team would in the same spots.")


# ---- PROE charts ----------------------------------------------------------------------------------------------
SITUATIONS = [   # (label, rule): game situations for the second PROE chart
    ("Neutral (within 7, not 2-min)", lambda d: d["neutral"]),
    ("Leading by 8+", lambda d: d["score_differential"] >= 8),
    ("Trailing by 8+", lambda d: d["score_differential"] <= -8),
    ("1st & 10", lambda d: (d["DN"] == 1) & (d["ydstogo"] == 10)),
    ("2nd & 7+", lambda d: (d["DN"] == 2) & (d["ydstogo"] >= 7)),
    ("2nd & 1-3", lambda d: (d["DN"] == 2) & (d["ydstogo"] <= 3)),
    ("3rd & 1-2", lambda d: (d["DN"] == 3) & (d["ydstogo"] <= 2)),
    ("Red zone", lambda d: d["yardline_100"] <= 20),
    ("Goal to go", lambda d: d["ydstogo"] >= d["yardline_100"]),
    ("Backed up (own 10 and in)", lambda d: d["yardline_100"] >= 90),
    ("1st half", lambda d: d["qtr"] <= 2),
    ("2nd half", lambda d: d["qtr"] >= 3),
    ("Last 2 min of a half", lambda d: d["half_seconds_remaining"] <= 120),
]


def proe_frame(team_d: pd.DataFrame, league_d: pd.DataFrame, groups: list[tuple[str, callable]]) -> pd.DataFrame:
    rows = []
    for label, rule in groups:
        t, lg = team_d[rule(team_d).fillna(False)], league_d[rule(league_d).fillna(False)]
        if t.empty:
            continue
        rows.append({"Situation": label, "Plays": len(t), "Pass rate": t["PASS"].mean(),
                     "Expected": t["xpass"].mean(), "PROE": t["pass_oe"].mean() / 100,
                     "NFL pass rate": lg["PASS"].mean() if len(lg) else np.nan})
    return pd.DataFrame(rows)


def proe_chart(f: pd.DataFrame, title: str, min_plays: int = 5) -> None:
    """One row per situation: their pass rate (purple), what an average team would do in their exact spots
    (hollow ring; the gap is PROE), and the NFL's actual pass rate in that situation (gray tick)."""
    st.markdown(f"**{title}**")
    f = f[f["Plays"] >= min_plays]
    if f.empty:
        st.info("Not enough plays match the current filters.")
        return
    order = list(f["Situation"])
    f = f.assign(label=f["PROE"].map(cm.pct_points),
                 lo=f[["Pass rate", "Expected"]].min(axis=1), hi=f[["Pass rate", "Expected"]].max(axis=1))
    y = alt.Y("Situation:N", sort=order, title=None, axis=alt.Axis(labelLimit=220))
    tip = ["Situation", "Plays", alt.Tooltip("Pass rate:Q", format=".0%"), alt.Tooltip("Expected:Q", format=".0%"),
           alt.Tooltip("PROE:Q", format="+.0%"), alt.Tooltip("NFL pass rate:Q", format=".0%")]
    base = alt.Chart(f).encode(y=y, tooltip=tip)
    gap = base.mark_rule(strokeWidth=3, opacity=0.35, color=cm.PASS_C).encode(x="lo:Q", x2="hi:Q")
    nfl = base.mark_tick(color="#888780", thickness=2, size=16).encode(x="NFL pass rate:Q")
    exp = base.mark_point(shape="circle", size=110, filled=False, color=cm.PASS_C, strokeWidth=2).encode(x="Expected:Q")
    act = base.mark_circle(size=120, color=cm.PASS_C, opacity=1).encode(
        x=alt.X("Pass rate:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%", title="Pass rate")))
    text = base.mark_text(align="left", dx=12, fontSize=12, fontWeight="bold", color=cm.PASS_C).encode(
        x=alt.X("hi:Q"), text="label:N")
    st.altair_chart(alt.layer(gap, nfl, exp, act, text).properties(height=34 * len(f) + 30), width="stretch")
    st.caption("Filled dot = their pass rate · ring = expected pass rate for an average NFL team in the same spots "
               "(nflfastR's xpass) · the label is PROE, the gap between them · gray tick = what the NFL actually did "
               "in that situation with the current filters.")


def dd_groups(team_d: pd.DataFrame) -> list[tuple[str, callable]]:
    out = []
    for dn in (1, 2, 3, 4):
        for dist in dt.DISTANCE_ORDER:
            out.append((f"{dt.DOWN_NAMES[dn]} & {dist.split(' (')[0].lower()}",
                        lambda d, dn=dn, dist=dist: (d["DN"] == dn) & (d["Distance"] == dist)))
    return out


# ---- Tables --------------------------------------------------------------------------------------------------
TEND_MEASURES = ["Run Rate", "Pass Rate", "NFL Pass Rate", "Exp. Pass Rate", "PROE", "Plays"]
LOOK_MEASURES = ["Run Rate", "Pass Rate", "NFL Pass Rate", "PROE", "Success Rate", "EPA per Play", "NFL EPA per Play",
                 "Plays"]


def render_tables(team_d, league_d, team):
    t = cm.crosstab(team_d, ["DN", "Distance"], TEND_MEASURES, league_d)
    t.index = t.index.set_levels([dt.DOWN_NAMES[int(x)] for x in t.index.levels[0]], level=0).rename(["Down", "Distance"])
    cm.show_table(f"{team} offensive tendencies", t)
    forms = cm.crosstab(team_d, "Formation", LOOK_MEASURES, league_d, sort_by_count=True)
    cm.show_table(f"{team} formation tendencies", forms)
    if cm.charting_note(team_d, "Personnel", "personnel"):
        pers = cm.crosstab(team_d, "Personnel", LOOK_MEASURES, league_d, sort_by_count=True)
        cm.show_table(f"{team} personnel tendencies", pers)
        both = cm.crosstab(team_d, ["Personnel", "Formation"], LOOK_MEASURES, league_d, sort_by_count=True)
        with st.expander("Personnel × formation"):
            cm.show_table(f"{team} personnel and formation together", both)


def render_down_table(team_d, league_d, team, down):
    d, lg = team_d[team_d["DN"] == down], league_d[league_d["DN"] == down]
    t = cm.crosstab(d, "Distance", ["Pass Rate", "NFL Pass Rate", "PROE", "Conversion Rate", "NFL Conversion Rate",
                                    "Conversions", "Plays"], lg, order=dt.DISTANCE_ORDER)
    cm.show_table(f"{team} offense {dt.DOWN_NAMES[down]} downs", t, gray_low_n=False)


def render_down_calls(team_d, team, down):
    d = team_d[team_d["DN"] == down].sort_values(["Season", "Week", "play_id"], kind="stable")
    label = dt.DOWN_NAMES[down]
    st.markdown(f"**{team} {label} down calls and results**")
    if d.empty:
        st.info(f"No {label} down plays match the current filters.")
        return
    t = cm.plays_table(d)
    t.insert(0, "Converted", np.where(d["first_down"] == 1, "Yes", "No"))
    with st.expander(f"All {len(d)} {label} down plays"):
        st.dataframe(t.style.format({"EPA": "{:+.2f}"}, na_rep=""), width="stretch", hide_index=True)
    st.caption(f"Converted {int(d['first_down'].sum())} of {len(d)}. Every {label} down run or pass in the selected "
               "games, oldest first.")


def render(team_d: pd.DataFrame, league_d: pd.DataFrame, team: str) -> None:
    render_dd_grid(team_d, league_d, team)
    c1, c2 = st.columns(2)
    with c1:
        proe_chart(proe_frame(team_d, league_d, dd_groups(team_d)), f"{team} pass rate over expected by down & distance")
    with c2:
        proe_chart(proe_frame(team_d, league_d, SITUATIONS), f"{team} pass rate over expected by game situation")
    render_tables(team_d, league_d, team)
    render_down_table(team_d, league_d, team, 3)
    render_down_table(team_d, league_d, team, 4)
    render_down_calls(team_d, team, 3)
    render_down_calls(team_d, team, 4)
    cm.show_plays(f"{team} best 10 plays (most likely to come back)", team_d.nlargest(10, "epa"))
