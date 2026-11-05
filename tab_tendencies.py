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

# ---- Down & distance: the boxes that stand out, then the full grid ----------------------------------------------
def dd_cells(team_d: pd.DataFrame, league_d: pd.DataFrame) -> pd.DataFrame:
    """One row per down & distance box: plays, pass rate, expected pass rate, PROE, NFL pass rate, top look."""
    d = team_d[team_d["DN"].notna() & team_d["Distance"].notna()]
    lg = league_d[league_d["DN"].notna() & league_d["Distance"].notna()]
    nfl_pass = lg.groupby(["DN", "Distance"], observed=True)["PASS"].mean()
    rows = []
    for (dn, dist), g in d.groupby(["DN", "Distance"], observed=True):
        x = g[g["xpass"].notna()]   # nflfastR has no xpass on a handful of snaps
        forms = g["Formation"].value_counts(normalize=True)
        pers = g["Personnel"].value_counts(normalize=True)
        look = (f"{forms.index[0]} {forms.iat[0]:.0%}" if len(forms) else "") + (
            f", {pers.index[0]} personnel {pers.iat[0]:.0%}" if len(pers) else "")
        rows.append({"DN": int(dn), "Distance": dist, "n": len(g), "pass_rate": g["PASS"].mean(),
                     "xpass": x["xpass"].mean() if len(x) else np.nan,
                     "proe": x["pass_oe"].mean() / 100 if len(x) else np.nan,
                     "nfl": nfl_pass.get((dn, dist), np.nan), "look": look})
    return pd.DataFrame(rows)


def _name(dn: int, dist: str) -> str:
    return f"{dt.DOWN_NAMES[dn]} & {dist.split(' (')[0].lower()}"


def _hover(r) -> str:
    expected = "" if pd.isna(r.xpass) else f" · expected for an average team {r.xpass:.0%}"
    proe = "" if pd.isna(r.proe) else f"\nPROE {cm.pct_points(r.proe)}"
    return (f"{_name(r.DN, r.Distance)} ({r.Distance.split('(')[1].rstrip(')')}): {r.n} plays\n"
            f"Pass {r.pass_rate:.0%}{expected} · NFL actual {r.nfl:.0%}{proe}" + (f"\nMost used: {r.look}" if r.look else ""))


def callouts_html(cells: pd.DataFrame) -> str:
    """A scouting-report sentence for each box that clearly differs from an average team (enough plays, |PROE| at
    least CHIP_MIN_PROE), biggest first."""
    if cells["proe"].isna().all():
        return f'<div style="{cm.CARD};font-size:14px">{cm.XPASS_NOTE} The grid shows their pass rates.</div>'
    notable = cells[(cells["n"] >= cm.TELL_MIN_N) & (cells["proe"].abs() >= cm.CHIP_MIN_PROE)]
    notable = notable.reindex(notable["proe"].abs().sort_values(ascending=False).index).head(6)
    if notable.empty:
        return (f'<div style="{cm.CARD};font-size:14px">Nothing stands out: in every down & distance with '
                f'{cm.TELL_MIN_N}+ plays they call it within {cm.CHIP_MIN_PROE:.0%} of an average team.</div>')
    rows = []
    for r in notable.itertuples():
        # Shown as whole percents that add up: their rate − expected = the PROE on the chip.
        their, gap = round(r.pass_rate * 100), int(round(r.proe * 100))
        if r.proe > 0:
            what = f"They throw <b>{their}%</b> of the time; an average team would throw {their - gap}% in those spots."
        else:
            what = (f"They run <b>{100 - their}%</b> of the time; an average team would run {100 - their + gap}% in "
                    f"those spots.")
        rows.append(f'<div title="{_hover(r)}" style="{cm.CARD};display:grid;'
                    f'grid-template-columns:130px 1fr auto;gap:12px;align-items:center;font-size:14px">'
                    f'<b>{_name(r.DN, r.Distance)}</b><span>{what} '
                    f'<span style="opacity:.6;font-size:12px">{r.n} plays</span></span>{cm.proe_chip(r.proe)}</div>')
    return f'<div style="display:grid;gap:6px;max-width:980px">{"".join(rows)}</div>'


def dd_grid_html(cells: pd.DataFrame) -> str:
    """Downs x distance: pass %, the run/pass bar, a PROE chip when the box clearly differs from an average team,
    and the play count bottom right. Everything else is in the box's hover."""
    if cells.empty:
        return ""
    muted = "opacity:.65"
    out = ['<div></div>'] + [
        f'<div style="font-size:12px;text-align:center;{muted}">{lab.split(" (")[0]}<br>{lab.split("(")[1].rstrip(")")}</div>'
        for lab in dt.DISTANCE_ORDER]
    for dn in [x for x in (1, 2, 3, 4) if (cells["DN"] == x).any()]:
        out.append(f'<div style="font-weight:600;font-size:14px;display:flex;align-items:center">{dt.DOWN_NAMES[dn]}</div>')
        for dist in dt.DISTANCE_ORDER:
            m = cells[(cells["DN"] == dn) & (cells["Distance"] == dist)]
            if m.empty:
                out.append(f'<div style="border:1px solid rgba(128,128,128,.25);border-radius:8px;padding:8px 10px;'
                           f'font-size:12px;{muted}">No plays</div>')
                continue
            r = next(m.itertuples())
            small = r.n < cm.TELL_MIN_N
            chip = cm.proe_chip(r.proe) if not small and abs(r.proe) >= cm.CHIP_MIN_PROE else ""
            out.append(
                f'<div title="{_hover(r)}" style="border:1px solid rgba(128,128,128,.25);'
                f'border-radius:8px;padding:8px 10px;font-size:12px{";opacity:.45" if small else ""}">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;gap:6px;min-height:20px">'
                f'<span><b style="font-size:15px">{r.pass_rate:.0%}</b> <span style="{muted}">pass</span></span>{chip}</div>'
                f'{cm.split_bar(1 - r.pass_rate)}'
                f'<div style="text-align:right;{muted}">{r.n} plays</div></div>')
    legend = (f'<div style="display:flex;flex-wrap:wrap;gap:14px;font-size:12px;margin:2px 0 8px;align-items:center">'
              f'<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:{cm.RUN_C};'
              f'margin-right:4px"></span>Run</span>'
              f'<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:{cm.PASS_C};'
              f'margin-right:4px"></span>Pass</span>'
              f'<span style="opacity:.7">chip = PROE, shown when it is {cm.CHIP_MIN_PROE:.0%} or more either way · faded = '
              f'under {cm.TELL_MIN_N} plays · hover a box for the details</span></div>')
    return (f'<div style="max-width:980px;font-family:inherit">{legend}'
            f'<div style="display:grid;grid-template-columns:48px repeat(4,minmax(0,1fr));gap:6px">'
            f'{"".join(out)}</div></div>')


def render_dd_grid(team_d, league_d, team):
    st.markdown(f"**{team} down & distance tendencies**")
    cells = dd_cells(team_d, league_d)
    if cells.empty:
        st.info("No plays match the current filters.")
        return
    st.html(callouts_html(cells))
    st.html(dd_grid_html(cells))
    st.caption("PROE = pass rate over expected: how much more often they throw than an average NFL team would in the "
               "same spots (down, distance, field position, score, time). Hover a box for the expected and NFL pass "
               "rates and the most-used formation.")


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


# ---- Cheat sheet and "lately" ---------------------------------------------------------------------------------
CHEAT_MIN_PLAYS = 20      # a look needs this many plays to make the cheat sheet
CHEAT_TOP = 6
LATELY_GAMES = 4          # "lately" = their last this-many games in the selection
LATELY_FLAG = 0.08        # a split that moved this much (8 points) gets flagged...
LATELY_MIN_PLAYS = 15     # ...if the last games have this many plays in it


def _looks(d: pd.DataFrame) -> dict[str, pd.Series]:
    """The pre-snap looks a coordinator scouts, as one label per play (missing where the season has no charting)."""
    dd = d["DN"].map(lambda n: dt.DOWN_NAMES.get(int(n)) if pd.notna(n) else None) + " & " + \
        d["Distance"].astype(str).str.split(" (", regex=False).str[0].str.lower()
    last = np.where(d["first_of_drive"], "Start of a drive", "After a " + d["prev_type"].fillna("").str.lower())
    looks = {"Down & distance": dd, "Formation": d["Formation"], "Personnel": d["Personnel"].map(
                 lambda v: f"{v} personnel" if isinstance(v, str) else None),
             "Motion": d["Motion"].map({"Motion": "With motion", "No motion": "No motion"}),
             "Backfield": d["Backfield"], "Last call": pd.Series(last, index=d.index),
             "Field": pd.Series(np.where(d["yardline_100"] <= 20, "Red zone", None), index=d.index)}
    return {k: v for k, v in looks.items() if v.notna().any()}


def cheat_sheet(team_d: pd.DataFrame, league_d: pd.DataFrame) -> pd.DataFrame:
    """Their strongest tells across every look: where their pass rate is furthest from the NFL's in the same look
    (CHEAT_MIN_PLAYS+ plays), with how often those plays worked."""
    tl, ll = _looks(team_d), _looks(league_d)
    rows = []
    for dim, labels in tl.items():
        nfl = league_d.groupby(ll[dim])["PASS"].mean()
        for look, g in team_d.groupby(labels):
            if len(g) < CHEAT_MIN_PLAYS or look in ("", "After a ") or look not in nfl.index:
                continue
            rows.append({"Look type": dim, "Look": look, "Plays": len(g), "Pass rate": g["PASS"].mean(),
                         "NFL": nfl[look], "Worked": g["success"].mean()})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["Gap"] = t["Pass rate"] - t["NFL"]
    return t.reindex(t["Gap"].abs().sort_values(ascending=False).index).head(CHEAT_TOP).reset_index(drop=True)


def cheat_html(t: pd.DataFrame) -> str:
    cards = []
    for _, r in t.iterrows():
        passy = r["Gap"] > 0
        color = cm.PASS_C if passy else cm.RUN_C
        rate, nfl = (r["Pass rate"], r["NFL"]) if passy else (1 - r["Pass rate"], 1 - r["NFL"])
        cards.append(f'<div style="{cm.CARD};display:flex;flex-direction:column;gap:3px">'
                     f'<span style="font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;'
                     f'opacity:.65">{r["Look type"]}</span><span style="font-weight:600">{r["Look"]}</span>'
                     f'<span style="font-size:26px;font-weight:800;line-height:1.1;color:{color}">'
                     f'{"PASS" if passy else "RUN"} {rate:.0%}</span>'
                     f'<span style="font-size:12px;opacity:.7">NFL {nfl:.0%} · {r["Plays"]} plays · worked '
                     f'{r["Worked"]:.0%} of the time</span></div>')
    return ('<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:8px;'
            f'max-width:980px">{"".join(cards)}</div>')


LATELY_SPLITS = [("Overall", lambda d: d), ("1st down", lambda d: d[d["DN"] == 1]),
                 ("2nd down", lambda d: d[d["DN"] == 2]), ("3rd down", lambda d: d[d["DN"] == 3]),
                 ("Red zone", lambda d: d[d["yardline_100"] <= 20]), ("Shotgun", lambda d: d[d["shotgun"] == 1]),
                 ("Under center", lambda d: d[d["shotgun"] == 0])]


def lately(team_d: pd.DataFrame, league_d: pd.DataFrame) -> pd.DataFrame:
    """Pass rate over expected (or, before 2006, pass rate minus the NFL's) in each split: the whole selection vs
    their last LATELY_GAMES games, with a flag where it moved LATELY_FLAG+."""
    games = team_d.drop_duplicates("game_id").sort_values(["Season", "Week"])["game_id"]
    if len(games) < LATELY_GAMES + 2:
        return pd.DataFrame()
    recent_ids = set(games.tail(LATELY_GAMES))
    use_proe = cm.has_xpass(team_d)

    def lean(d: pd.DataFrame, lg: pd.DataFrame) -> float:
        if use_proe:
            return d["pass_oe"].mean() / 100
        return d["PASS"].mean() - lg["PASS"].mean()
    rows = []
    for label, pick in LATELY_SPLITS:
        whole, lg = pick(team_d), pick(league_d)
        recent = whole[whole["game_id"].isin(recent_ids)]
        if len(whole) < LATELY_MIN_PLAYS or recent.empty:
            continue
        rows.append({"Split": label, "Season": lean(whole, lg), "Lately": lean(recent, lg), "Recent plays": len(recent)})
    t = pd.DataFrame(rows)
    if len(t):
        t["Flag"] = ((t["Lately"] - t["Season"]).abs() >= LATELY_FLAG) & (t["Recent plays"] >= LATELY_MIN_PLAYS)
    return t


def lately_chart(t: pd.DataFrame, measure: str) -> alt.LayerChart:
    order = list(t["Split"])
    y = alt.Y("Split:N", sort=order, title=None, axis=alt.Axis(labelFontSize=13))
    tip = ["Split", alt.Tooltip("Season:Q", format="+.0%", title="Whole selection"),
           alt.Tooltip("Lately:Q", format="+.0%", title=f"Last {LATELY_GAMES} games"),
           alt.Tooltip("Recent plays:Q", title="Plays in the last games")]
    lo = float(min(t["Season"].min(), t["Lately"].min(), 0)) - 0.05
    hi = float(max(t["Season"].max(), t["Lately"].max(), 0)) + 0.05
    x = alt.X("Season:Q", title=measure, axis=alt.Axis(format="+.0%"), scale=alt.Scale(domain=[lo, hi]))
    base = alt.Chart(t).encode(y=y, tooltip=tip)
    zero = alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(color="#888780").encode(x="v:Q")
    line = base.mark_rule(strokeWidth=3).encode(x=x, x2="Lately:Q", color=alt.condition(
        "datum.Flag", alt.value("#C8321E"), alt.value("#888780")))
    ring = base.mark_point(size=110, filled=False, strokeWidth=2.5, color=cm.theme_ink()).encode(x="Season:Q")
    dot = base.mark_circle(size=130, opacity=1).encode(x="Lately:Q", color=alt.condition(
        "datum.Flag", alt.value("#C8321E"), alt.value(cm.theme_ink())))
    return alt.layer(zero, line, ring, dot).properties(height=36 * len(t) + 20)


def render_cheat_and_lately(team_d: pd.DataFrame, league_d: pd.DataFrame, team: str) -> None:
    t = cheat_sheet(team_d, league_d)
    st.markdown(f"**{team} cheat sheet: the {CHEAT_TOP} strongest tells**")
    if t.empty:
        st.info(f"No look has {CHEAT_MIN_PLAYS}+ plays with these filters.")
    else:
        st.html(cheat_html(t))
        st.caption(f"From every look on this page (down & distance, formation, personnel, motion, backfield, last call, "
                   f"red zone), the {CHEAT_TOP} where their pass rate is furthest from the NFL's in the same look, "
                   f"{CHEAT_MIN_PLAYS}+ plays each. Worked = share of those plays with positive EPA.")
    lt = lately(team_d, league_d)
    if lt.empty:
        return
    measure = "Pass rate over expected" if cm.has_xpass(team_d) else "Pass rate vs the NFL"
    moved = lt[lt["Flag"]]
    st.markdown(f"**Lately: their last {LATELY_GAMES} games vs the whole selection**")
    if len(moved):
        r = moved.reindex((moved["Lately"] - moved["Season"]).abs().sort_values(ascending=False).index).iloc[0]
        st.warning(f"**{r['Split']} has changed:** {r['Season']:+.0%} for the whole selection, {r['Lately']:+.0%} in "
                   f"the last {LATELY_GAMES} games ({'more pass-heavy' if r['Lately'] > r['Season'] else 'more run-heavy'}"
                   "). Check the recent games before trusting the tables below for this spot.")
    st.altair_chart(lately_chart(lt, measure), width="stretch")
    st.caption(f"Ring = the whole selection; dot = their last {LATELY_GAMES} games. Red = moved "
               f"{LATELY_FLAG * 100:.0f}+ points on {LATELY_MIN_PLAYS}+ recent plays. "
               + ("Pass rate over expected: how much more often they throw than an average team in the same spots."
                  if cm.has_xpass(team_d) else "Before 2006: their pass rate minus the NFL's in the same split."))


def render(team_d: pd.DataFrame, league_d: pd.DataFrame, team: str) -> None:
    render_cheat_and_lately(team_d, league_d, team)
    render_dd_grid(team_d, league_d, team)
    if cm.has_xpass(team_d):
        c1, c2 = st.columns(2)
        with c1:
            proe_chart(proe_frame(team_d, league_d, dd_groups(team_d)),
                       f"{team} pass rate over expected by down & distance")
        with c2:
            proe_chart(proe_frame(team_d, league_d, SITUATIONS), f"{team} pass rate over expected by game situation")
    render_tables(team_d, league_d, team)
    render_down_table(team_d, league_d, team, 3)
    render_down_table(team_d, league_d, team, 4)
    render_down_calls(team_d, team, 3)
    render_down_calls(team_d, team, 4)
    cm.show_plays(f"{team} best 10 plays (most likely to come back)", team_d.nlargest(10, "epa"))
