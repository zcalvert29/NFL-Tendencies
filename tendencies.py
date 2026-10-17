"""
Tendency views: down & distance grid, tells (where a team is predictable), tendency tree, sequencing, field
position, red zone, and 3rd/4th downs. Every tendency is compared both to the team's own average and to the
league in the same situation, with a two-proportion z-test so small samples don't get flagged as tells.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import data as dt
import tables as tb

Z_FLAG = 1.96          # |z| at or above this = unlikely to be noise (95%)
TELL_MIN_N = 20        # rows with fewer plays aren't flagged as tells
TELL_MIN_GAP = 0.12    # ...or when the pass rate is within 12 points of the comparison


def two_prop_z(k1, n1, k2, n2) -> float:
    if n1 == 0 or n2 == 0:
        return 0.0
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return 0.0 if se == 0 else (k1 / n1 - k2 / n2) / se


def _pass_color(rate, ref):
    """Blue = run-heavy vs the reference, red = pass-heavy, gray = about the same."""
    if pd.isna(rate) or pd.isna(ref):
        return ("rgba(128,128,128,0.12)", "inherit")
    diff = rate - ref
    stops = [("#185FA5", "#E6F1FB"), ("#378ADD", "#042C53"), ("#B5D4F4", "#042C53"), ("#F1EFE8", "#2C2C2A"),
             ("#F7C1C1", "#501313"), ("#E24B4A", "#501313"), ("#A32D2D", "#FCEBEB")]
    a = abs(diff)
    step = 0 if a < 0.05 else 1 if a < 0.12 else 2 if a < 0.2 else 3
    return stops[3 + int(np.sign(diff)) * step]


def _legend(left="More runs than the NFL", right="More passes than the NFL") -> str:
    sw = "".join(f'<span style="width:20px;height:11px;border-radius:2px;background:{c}"></span>'
                 for c in ["#185FA5", "#378ADD", "#B5D4F4", "#F1EFE8", "#F7C1C1", "#E24B4A", "#A32D2D"])
    return (f'<div style="display:flex;gap:6px;align-items:center;font-size:12px;margin-top:6px">'
            f'<span style="opacity:.65">{left}</span>{sw}<span style="opacity:.65">{right}</span></div>')


# ---- Down & distance ---------------------------------------------------------------------------------------
def render_down_distance(team_d: pd.DataFrame, league_d: pd.DataFrame, title: str):
    """Pass rate by down and distance, colored against the league in the same cell."""
    st.markdown(f"**{title}**")
    t = team_d[team_d["down"].notna() & team_d["Distance"].notna()]
    if t.empty:
        st.info("No plays match the current filters.")
        return
    lg = league_d[league_d["down"].notna() & league_d["Distance"].notna()]
    lg_rate = lg.groupby(["down", "Distance"])["is_pass"].mean()
    cells = ['<div></div>'] + [f'<div style="font-size:12px;text-align:center;opacity:.8">{c}</div>'
                               for c in dt.DISTANCE_ORDER]
    for down in [1, 2, 3, 4]:
        cells.append(f'<div style="font-size:13px;font-weight:500;display:flex;align-items:center">'
                     f'{["1st", "2nd", "3rd", "4th"][down - 1]} down</div>')
        for dist in dt.DISTANCE_ORDER:
            c = t[(t["down"] == down) & (t["Distance"] == dist)]
            n = len(c)
            ref = lg_rate.get((down, dist), np.nan)
            if n == 0:
                cells.append('<div style="background:rgba(128,128,128,0.08);border-radius:8px;min-height:84px"></div>')
                continue
            rate = c["is_pass"].mean()
            bg, fg = _pass_color(rate, ref) if n >= 5 else ("rgba(128,128,128,0.12)", "inherit")
            proe = c["pass_oe"].mean()
            proe_txt = "" if pd.isna(proe) else f" · PROE {proe:+.0f}%"
            cells.append(
                f'<div style="background:{bg};color:{fg};border-radius:8px;padding:7px 9px;min-height:84px;'
                f'font-variant-numeric:tabular-nums">'
                f'<div style="font-size:20px;font-weight:500">{rate:.0%} <span style="font-size:12px;font-weight:400">pass</span></div>'
                f'<div style="font-size:11px">NFL {ref:.0%}{proe_txt}</div>'
                f'<div style="font-size:11px">{n} plays · EPA {c["epa"].mean():+.2f} · {c["success"].mean():.0%} succ</div></div>')
    st.html(f'<div style="max-width:900px"><div style="display:grid;grid-template-columns:80px repeat(4,minmax(0,1fr));'
            f'gap:4px">{"".join(cells)}</div>{_legend()}</div>')
    st.caption("PROE = pass rate over expected (nflfastR's model also accounts for field position, score, and "
               "time). Cells under 5 plays aren't colored.")


# ---- Tells -------------------------------------------------------------------------------------------------
TELL_DIMENSIONS = {
    "Personnel": ["Personnel"],
    "Formation": ["Formation"],
    "Personnel + formation": ["Personnel", "Formation"],
    "Down & distance": ["down", "Distance"],
    "Field zone": ["Field Zone"],
    "Score": ["Score"],
    "Quarter": ["Quarter"],
    "Previous play": ["Prev Result"],
    "No huddle": ["no_huddle"],
}


def _label(dim: str, key) -> str:
    key = key if isinstance(key, tuple) else (key,)
    if dim == "Down & distance":
        return f"{['1st', '2nd', '3rd', '4th'][int(key[0]) - 1]} & {str(key[1]).split(' (')[0].lower()}"
    if dim == "Personnel":
        return f"{key[0]} personnel"
    if dim == "Personnel + formation":
        return f"{key[0]} personnel, {key[1].lower()}"
    if dim == "Previous play":
        return f"After a {key[0].lower()}"
    if dim == "No huddle":
        return "No huddle" if key[0] == 1 else "Huddled"
    return " · ".join(map(str, key))


def tells_frame(team_d: pd.DataFrame, league_d: pd.DataFrame) -> pd.DataFrame:
    """Every situation where the team's run/pass split is far from what they usually do and from the league."""
    team_d, league_d = team_d[team_d["scrimmage"]], league_d[league_d["scrimmage"]]
    base = team_d["is_pass"].mean()
    rows = []
    for dim, cols in TELL_DIMENSIONS.items():
        t = team_d.dropna(subset=cols)
        if t.empty:
            continue
        lg = league_d.dropna(subset=cols).groupby(cols, observed=True)["is_pass"].agg(["sum", "count"])
        for key, c in t.groupby(cols, observed=True):
            n, k = len(c), int(c["is_pass"].sum())
            if n < TELL_MIN_N:
                continue
            key = key[0] if isinstance(key, tuple) and len(key) == 1 else key
            rate = k / n
            lk, ln = lg.loc[key] if key in lg.index else (np.nan, np.nan)
            rest = team_d[~team_d.index.isin(c.index)]
            z_self = two_prop_z(k, n, int(rest["is_pass"].sum()), len(rest))
            z_lg = two_prop_z(k, n, lk, ln) if ln == ln and ln else 0.0
            rows.append({"Dimension": dim, "Situation": _label(dim, key), "Plays": n, "Pass Rate": rate,
                         "Team usual": base, "NFL here": lk / ln if ln else np.nan,
                         "z vs usual": z_self, "z vs NFL": z_lg, "EPA per Play": c["epa"].mean(),
                         "Run EPA": c.loc[c["is_run"], "epa"].mean(), "Pass EPA": c.loc[c["is_pass"], "epa"].mean()})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["Lean"] = np.where(out["Pass Rate"] >= 0.5, "Pass", "Run")
    out["Strength"] = out[["z vs usual", "z vs NFL"]].abs().min(axis=1)
    return out.sort_values("Strength", ascending=False)


def render_tells(team_d: pd.DataFrame, league_d: pd.DataFrame, team: str, top: int = 8):
    """Cards for the strongest tells, then the full table in an expander."""
    t = tells_frame(team_d, league_d)
    st.markdown(f"**Where {team} tips its hand**")
    if t.empty:
        st.info("Not enough plays match the current filters.")
        return
    gap_self = (t["Pass Rate"] - t["Team usual"]).abs()
    gap_lg = (t["Pass Rate"] - t["NFL here"]).abs()
    strong = t[(t["Strength"] >= Z_FLAG) & (gap_self >= TELL_MIN_GAP) & (gap_lg >= TELL_MIN_GAP)]
    if strong.empty:
        st.caption("No strong tells with the current filters: nothing is far from both their usual split and the "
                   "league's at a sample size that rules out noise.")
    else:
        cols = st.columns(min(4, len(strong.head(top))))
        for i, (_, r) in enumerate(strong.head(top).iterrows()):
            lean = r["Pass Rate"] if r["Lean"] == "Pass" else 1 - r["Pass Rate"]
            with cols[i % len(cols)].container(border=True):
                st.caption(r["Dimension"])
                st.markdown(f"**{r['Situation']}**")
                st.markdown(f"{lean:.0%} {r['Lean'].lower()} · {int(r['Plays'])} plays")
                st.caption(f"Usually {r['Team usual']:.0%} pass · NFL here {r['NFL here']:.0%} pass")
    with st.expander("Every situation"):
        show = t.drop(columns=["Strength", "Lean"]).set_index(["Dimension", "Situation"])
        st.dataframe(show.style.format({"Plays": "{:,.0f}", "Pass Rate": "{:.0%}", "Team usual": "{:.0%}",
                                        "NFL here": "{:.0%}", "z vs usual": "{:+.1f}", "z vs NFL": "{:+.1f}",
                                        "EPA per Play": "{:+.2f}", "Run EPA": "{:+.2f}", "Pass EPA": "{:+.2f}"},
                                       na_rep=""), width="stretch")
    st.caption(f"A tell = a pass rate at least {TELL_MIN_GAP:.0%} away from both the team's usual split and the "
               f"league's in the same situation, with |z| ≥ {Z_FLAG} on both two-proportion z-tests and "
               f"{TELL_MIN_N}+ plays.")


# ---- Tendency tree -----------------------------------------------------------------------------------------
def _split_bar(pass_share: float) -> str:
    return (f'<div style="display:flex;height:6px;border-radius:3px;overflow:hidden;width:120px">'
            f'<div style="width:{(1 - pass_share) * 100:.0f}%;background:#378ADD"></div>'
            f'<div style="width:{pass_share * 100:.0f}%;background:#E24B4A"></div></div>')


def render_tendency_tree(team_d: pd.DataFrame, league_d: pd.DataFrame, title: str, levels=("Personnel", "Formation"),
                         min_plays: int = 15):
    """Personnel → formation → what they run out of it (pass zone or run lane), with the run/pass split."""
    st.markdown(f"**{title}**")
    d = team_d[team_d["scrimmage"]].dropna(subset=list(levels))
    if d.empty:
        tb.coverage_note(team_d[team_d["scrimmage"]], levels[0], levels[0].lower())
        return
    lg = league_d[league_d["scrimmage"]].dropna(subset=list(levels))
    total = len(d)
    rows = []

    def node(depth, label, c, lg_c):
        n = len(c)
        rate = c["is_pass"].mean()
        ref = lg_c["is_pass"].mean() if len(lg_c) else np.nan
        z = two_prop_z(int(c["is_pass"].sum()), n, int(lg_c["is_pass"].sum()), len(lg_c)) if len(lg_c) else 0
        flag = ""
        if n >= TELL_MIN_N and abs(z) >= Z_FLAG and abs(rate - ref) >= TELL_MIN_GAP:
            flag = (f'<span style="font-size:11px;padding:1px 7px;border-radius:999px;background:'
                    f'{"#FCEBEB;color:#791F1F" if rate > ref else "#E6F1FB;color:#042C53"}">'
                    f'{"pass" if rate > ref else "run"}-heavy vs NFL</span>')
        runs, passes = c[c["is_run"]], c[c["is_pass"]]
        top_run = runs["Run Lane"].value_counts().head(2)
        top_pass = passes["Pass Zone"].value_counts().head(2)
        favs = ", ".join([f"{k.lower()} run ({v})" for k, v in top_run.items()] +
                         [f"{k.lower()} pass ({v})" for k, v in top_pass.items()])
        rows.append(
            f'<div style="display:grid;grid-template-columns:minmax(160px,1.4fr) 70px 130px 90px 70px minmax(0,2fr);'
            f'gap:8px;align-items:center;padding:5px 0;border-bottom:0.5px solid rgba(128,128,128,.2);'
            f'font-size:13px;font-variant-numeric:tabular-nums;{"opacity:.55" if n < TELL_MIN_N else ""}">'
            f'<div style="font-weight:{600 if depth == 0 else 400};padding-left:{depth * 22}px">{label} {flag}</div>'
            f'<div>{n} · {n / total:.0%}</div>'
            f'<div>{_split_bar(rate)}<span style="font-size:11px">{rate:.0%} pass · NFL {ref:.0%}</span></div>'
            f'<div>EPA {c["epa"].mean():+.2f}</div><div>{c["success"].mean():.0%} succ</div>'
            f'<div style="font-size:11px;opacity:.8">{favs}</div></div>')

    for k1, c1 in sorted(d.groupby(levels[0], observed=True), key=lambda kv: -len(kv[1])):
        if len(c1) < min_plays:
            continue
        node(0, f"{k1}{' personnel' if levels[0] == 'Personnel' else ''}", c1, lg[lg[levels[0]] == k1])
        if len(levels) > 1:
            for k2, c2 in sorted(c1.groupby(levels[1], observed=True), key=lambda kv: -len(kv[1])):
                if len(c2) >= max(5, min_plays // 3):
                    node(1, k2, c2, lg[(lg[levels[0]] == k1) & (lg[levels[1]] == k2)])
    header = ('<div style="display:grid;grid-template-columns:minmax(160px,1.4fr) 70px 130px 90px 70px minmax(0,2fr);'
              'gap:8px;font-size:11px;opacity:.6;padding-bottom:4px"><div></div><div>Plays · share</div>'
              '<div>Run / pass</div><div>EPA</div><div>Success</div><div>Favorite calls</div></div>')
    st.html(f'<div style="max-width:1100px">{header}{"".join(rows)}</div>')
    st.caption(f"Blue = run, red = pass. Rows under {TELL_MIN_N} plays are faded; groups under {min_plays} plays are "
               f"left off. \"NFL\" = the league's pass rate from the same personnel/formation.")


# ---- Sequencing --------------------------------------------------------------------------------------------
def render_sequencing(team_d: pd.DataFrame, league_d: pd.DataFrame, team: str):
    st.markdown(f"**What {team} calls after...**")
    d = team_d[team_d["scrimmage"] & team_d["Prev Result"].notna()]
    if len(d) < 20:
        st.info("Not enough plays match the current filters.")
        return
    lg = league_d[league_d["scrimmage"] & league_d["Prev Result"].notna()]
    a, b = d[d["Prev Play"] == "Run"], d[d["Prev Play"] == "Pass"]
    if len(a) >= 20 and len(b) >= 20:
        pa, pb = a["is_pass"].mean(), b["is_pass"].mean()
        z = two_prop_z(int(b["is_pass"].sum()), len(b), int(a["is_pass"].sum()), len(a))
        verdict = "a real pattern at this sample size" if abs(z) >= Z_FLAG else "could still be noise"
        st.html(f'<div style="font-size:14px;margin-bottom:6px">Throws <b>{pb:.0%}</b> after a pass vs <b>{pa:.0%}</b> '
                f'after a run ({len(b)} and {len(a)} plays): {verdict}.</div>')
    order = ["Run that worked", "Run that didn't", "Pass that worked", "Pass that didn't"]
    t = tb.crosstab(d, "Prev Result", ["Plays", "Pass Rate", "EPA per Play", "Success Rate"], order=order)
    t["NFL pass rate"] = lg.groupby("Prev Result")["is_pass"].mean().reindex(t.index)
    t.index.name = "Previous snap"
    tb.show_table("", t)
    st.caption("Previous snap = the run or pass right before, in the same drive. \"Worked\" = positive EPA.")


# ---- Field position & red zone -----------------------------------------------------------------------------
def render_field_position(team_d: pd.DataFrame, league_d: pd.DataFrame, title: str, good_high: bool = True):
    t = tb.crosstab(team_d[team_d["scrimmage"]], "Field Zone", ["Plays", "Pass Rate", "PROE"] + tb.PLAY_RESULTS[1:],
                    order=dt.FIELD_ZONES)
    t["NFL pass rate"] = league_d[league_d["scrimmage"]].groupby("Field Zone")["is_pass"].mean().reindex(t.index)
    t["NFL EPA"] = league_d[league_d["scrimmage"]].groupby("Field Zone")["epa"].mean().reindex(t.index)
    tb.show_table(title, t, good_high=good_high)


def drives_frame(games: pd.DataFrame, team: str, side: str = "offense") -> pd.DataFrame:
    """One row per possession for the team (side='offense') or its opponents (side='defense')."""
    col = "posteam" if side == "offense" else "defteam"
    d = games[(games[col] == team) & games["fixed_drive"].notna()]
    if d.empty:
        return pd.DataFrame()
    snaps = d[d["scrimmage"]]
    g = d.groupby(["game_id", "fixed_drive"])
    out = pd.DataFrame({
        "result": g["fixed_drive_result"].first(),
        "start": 100 - g["yardline_100"].max(),
        "plays": g["drive_play_count"].max(),
    })
    out["closest"] = snaps.groupby(["game_id", "fixed_drive"])["yardline_100"].min().reindex(out.index)
    out["points"] = out["result"].map({"Touchdown": 7, "Field goal": 3}).fillna(0)
    out["rz_trip"] = out["closest"] <= 20
    out["Season"] = g["Season"].first()
    return out.reset_index()


def drive_kpis(dr: pd.DataFrame) -> dict:
    if dr.empty:
        return {}
    rz = dr[dr["rz_trip"]]
    return {
        "Drives": len(dr), "Points / drive": dr["points"].mean(), "Scoring rate": (dr["points"] > 0).mean(),
        "TD rate": (dr["result"] == "Touchdown").mean(), "Turnover rate": dr["result"].isin(["Turnover", "Opp touchdown"]).mean(),
        "Red zone trips": len(rz), "RZ TD rate": (rz["result"] == "Touchdown").mean() if len(rz) else np.nan,
        "Points / RZ trip": rz["points"].mean() if len(rz) else np.nan, "Avg start": dr["start"].mean(),
    }


KPI_FMT = {"Drives": "{:,.0f}", "Points / drive": "{:.2f}", "Scoring rate": "{:.0%}", "TD rate": "{:.0%}",
           "Turnover rate": "{:.0%}", "Red zone trips": "{:,.0f}", "RZ TD rate": "{:.0%}", "Points / RZ trip": "{:.2f}",
           "Avg start": "own {:.0f}"}


def render_kpis(k: dict, league: dict | None = None, keys=None, good_high: bool = True):
    keys = keys or list(k)
    cols = st.columns(len(keys))
    for col, key in zip(cols, keys):
        v = k.get(key, np.nan)
        val = "–" if pd.isna(v) else KPI_FMT.get(key, "{:.2f}").format(v)
        delta = None
        if league and key in league and not pd.isna(v) and not pd.isna(league[key]) and key not in ("Drives", "Red zone trips"):
            diff = v - league[key]
            delta = (f"{diff:+.0%}" if "{:.0%}" == KPI_FMT.get(key) else f"{diff:+.2f}") + " vs NFL"
        inverse = (key == "Turnover rate") == good_high
        col.metric(key, val, delta, delta_color="inverse" if inverse else "normal")


def render_red_zone(team_d: pd.DataFrame, league_d: pd.DataFrame, games: pd.DataFrame, league_games: pd.DataFrame,
                    team: str, side: str, good_high: bool = True):
    dr = drives_frame(games, team, side)
    if dr.empty:
        st.info("No drives match the current filters.")
        return
    lg_dr = pd.concat([drives_frame(league_games, t, "offense") for t in league_games["posteam"].dropna().unique()])
    render_kpis(drive_kpis(dr), drive_kpis(lg_dr), ["Red zone trips", "RZ TD rate", "Points / RZ trip", "Points / drive",
                                                     "Scoring rate"], good_high=good_high)
    rz = team_d[team_d["scrimmage"] & team_d["red_zone"]]
    lg_rz = league_d[league_d["scrimmage"] & league_d["red_zone"]]
    c1, c2 = st.columns(2)
    with c1:
        t = tb.crosstab(rz, ["down"], ["Plays", "Pass Rate"] + tb.PLAY_RESULTS[1:])
        t["NFL pass rate"] = lg_rz.groupby("down")["is_pass"].mean().reindex(t.index)
        t.index = [["1st", "2nd", "3rd", "4th"][int(i) - 1] for i in t.index]
        tb.show_table("Red zone by down", t, good_high=good_high)
    with c2:
        t = tb.crosstab(rz, "Field Zone", ["Plays", "Pass Rate", "Success Rate", "EPA per Play"])
        t["NFL pass rate"] = lg_rz.groupby("Field Zone")["is_pass"].mean().reindex(t.index)
        tb.show_table("Red zone vs inside the 10", t, good_high=good_high)
    c1, c2 = st.columns(2)
    with c1:
        t = tb.crosstab(rz, "Personnel", ["Plays", "Share", "Pass Rate", "Success Rate", "EPA per Play"], sort_by_count=True)
        tb.show_table("Red zone personnel", t, good_high=good_high)
    with c2:
        t = tb.crosstab(rz[rz["is_pass"]], "Pass Zone", ["Plays", "Share", "Comp %", "Success Rate", "EPA per Play"],
                        sort_by_count=True)
        tb.show_table("Red zone throws", t, good_high=good_high)
    st.caption("Red zone trip = a drive with a snap at the 20 or closer. Points count a touchdown as 7.")


# ---- 3rd & 4th downs ---------------------------------------------------------------------------------------
def render_third_downs(team_d: pd.DataFrame, league_d: pd.DataFrame, title: str, good_high: bool = True):
    d = team_d[team_d["scrimmage"] & (team_d["down"] == 3)]
    lg = league_d[league_d["scrimmage"] & (league_d["down"] == 3)]
    t = tb.crosstab(d, "Distance", ["Plays", "Pass Rate", "Conversion Rate", "Success Rate", "EPA per Play"],
                    order=dt.DISTANCE_ORDER)
    t["NFL conv rate"] = lg.groupby("Distance")["first_down"].mean().reindex(t.index)
    t["NFL pass rate"] = lg.groupby("Distance")["is_pass"].mean().reindex(t.index)
    tb.show_table(title, t, good_high=good_high)


def go_rate_frame(d: pd.DataFrame) -> pd.DataFrame:
    """4th downs (2-pt tries and plays in the last 2 minutes of the 4th left out): go = ran a run or pass play."""
    f = d[(d["down"] == 4) & ~((d["qtr"] == 4) & d["two_minute"])
          & d["play_type"].isin(["run", "pass", "punt", "field_goal"]) & (d["qb_kneel"] != 1)].copy()
    f["went"] = f["play_type"].isin(["run", "pass"])
    f["To go"] = pd.cut(f["ydstogo"], [0, 1, 3, 6, 100], labels=["4th & 1", "4th & 2-3", "4th & 4-6", "4th & 7+"]).astype(object)
    f["Where"] = pd.cut(f["yardline_100"], [0, 35, 50, 100], labels=["Opp 35 and in", "Opp 36-50", "Own side"]).astype(object)
    return f


def render_fourth_downs(team_d: pd.DataFrame, league_d: pd.DataFrame, team: str):
    st.markdown(f"**{team} 4th-down aggressiveness**")
    f, lg = go_rate_frame(team_d), go_rate_frame(league_d)
    if f.empty:
        st.info("No 4th downs match the current filters.")
        return
    # Expected go rate: what the league does in the same to-go x field bucket, weighted by this team's mix.
    lg_rate = lg.groupby(["To go", "Where"])["went"].mean()
    f["NFL go"] = [lg_rate.get((a, b), np.nan) for a, b in zip(f["To go"], f["Where"])]
    went, exp = f["went"].mean(), f["NFL go"].mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("4th downs", f"{len(f)}")
    c2.metric("Went for it", f"{went:.0%}", f"{went - exp:+.0%} vs NFL",
              help="Compared with how often the league goes for it from the same distance and field position.")
    attempts = f[f["went"]]
    c3.metric("Converted", "–" if attempts.empty else f"{attempts['first_down'].mean():.0%}",
              None if attempts.empty or lg[lg['went']].empty else
              f"{attempts['first_down'].mean() - lg[lg['went']]['first_down'].mean():+.0%} vs NFL")
    c4.metric("EPA on attempts", "–" if attempts.empty else f"{attempts['epa'].mean():+.2f}")
    grid = f.groupby(["To go", "Where"])["went"].agg(["mean", "count"])
    cells = ['<div></div>'] + [f'<div style="font-size:12px;text-align:center;opacity:.8">{w}</div>'
                               for w in ["Own side", "Opp 36-50", "Opp 35 and in"]]
    for tg in ["4th & 1", "4th & 2-3", "4th & 4-6", "4th & 7+"]:
        cells.append(f'<div style="font-size:13px;display:flex;align-items:center">{tg}</div>')
        for w in ["Own side", "Opp 36-50", "Opp 35 and in"]:
            n = int(grid["count"].get((tg, w), 0))
            r = grid["mean"].get((tg, w), np.nan)
            ref = lg_rate.get((tg, w), np.nan)
            bg, fg = _pass_color(r, ref) if n >= 3 else ("rgba(128,128,128,0.10)", "inherit")
            body = "" if not n else (f'<div style="font-size:18px;font-weight:500">{r:.0%}</div>'
                                     f'<div style="font-size:11px">{n} tries · NFL {ref:.0%}</div>')
            cells.append(f'<div style="background:{bg};color:{fg};border-radius:8px;padding:6px 9px;min-height:58px;'
                         f'font-variant-numeric:tabular-nums">{body}</div>')
    st.html(f'<div style="max-width:700px"><div style="display:grid;grid-template-columns:90px repeat(3,minmax(0,1fr));'
            f'gap:4px">{"".join(cells)}</div>'
            f'{_legend("Goes for it less than the NFL", "Goes for it more than the NFL")}</div>')
    st.caption("Go rate = share of 4th downs where they ran a play (fakes count as going). Last 2 minutes of the 4th "
               "quarter left out, since those are forced.")
    log = f.sort_values(["Season", "Week", "play_id"], ascending=False)[
        ["Season", "Week", "Opponent", "qtr", "ydstogo", "yardline_100", "score_differential", "play_type", "first_down",
         "epa", "wpa", "desc"]].rename(columns={"qtr": "Qtr", "ydstogo": "To go", "yardline_100": "Yds to goal",
                                                 "score_differential": "Score diff", "play_type": "Call",
                                                 "first_down": "1st down", "epa": "EPA", "wpa": "WPA", "desc": "Play"})
    with st.expander(f"Every 4th down ({len(log)})"):
        st.dataframe(log.style.format({"EPA": "{:+.2f}", "WPA": "{:+.1%}", "To go": "{:.0f}", "Yds to goal": "{:.0f}",
                                       "Score diff": "{:+.0f}", "1st down": "{:.0f}"}, na_rep=""),
                     width="stretch", hide_index=True)
