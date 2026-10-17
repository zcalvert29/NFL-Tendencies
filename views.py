"""
Page bodies. Each takes a Ctx: the selected team, the filtered plays for the whole league (so every number can
be compared to the NFL in the same filters), and team colors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

import charts as ch
import data as dt
import tables as tb
import tendencies as td


@dataclass
class Ctx:
    team: str
    opponent: str
    plays: pd.DataFrame      # every sidebar filter applied, all teams, every play type
    games: pd.DataFrame      # only the game-level filters (season, season type, weeks): whole games, for drives
    teams: pd.DataFrame      # nflverse team names / colors / logos, indexed by abbreviation
    seasons: list[int]

    @property
    def snaps(self) -> pd.DataFrame:
        return self.plays[self.plays["scrimmage"]]

    def offense(self, team: str | None = None) -> pd.DataFrame:
        s = self.snaps
        return s[s["posteam"] == (team or self.team)]

    def defense(self, team: str | None = None) -> pd.DataFrame:
        s = self.snaps
        return s[s["defteam"] == (team or self.team)]

    def color(self, team: str | None = None, which: str = "team_color") -> str:
        t = team or self.team
        return self.teams[which].get(t, "#378ADD") if t in self.teams.index else "#378ADD"

    def name(self, team: str | None = None) -> str:
        t = team or self.team
        return self.teams["team_nick"].get(t, t) if t in self.teams.index else t


def team_banner(ctx: Ctx, team: str | None = None, side: str = ""):
    t = team or ctx.team
    logo = ctx.teams["team_logo_espn"].get(t, "") if t in ctx.teams.index else ""
    full = ctx.teams["team_name"].get(t, t) if t in ctx.teams.index else t
    st.html(f'<div style="display:flex;align-items:center;gap:12px;margin:-6px 0 6px">'
            f'<img src="{logo}" style="height:46px" onerror="this.style.display=\'none\'">'
            f'<div><div style="font-size:20px;font-weight:600">{full}</div>'
            f'<div style="font-size:13px;opacity:.65">{side}</div></div></div>')


# ---- Team-level numbers (used by Home, profiles, matchup, season comparison) -------------------------------
def team_summary(snaps: pd.DataFrame, by: str = "posteam") -> pd.DataFrame:
    """One row per team: offense numbers (by='posteam') or what defenses allowed (by='defteam')."""
    g = snaps.groupby(by)
    p, r = snaps[snaps["is_pass"]].groupby(by), snaps[snaps["is_run"]].groupby(by)
    out = pd.DataFrame({
        "Plays": g.size(), "EPA/play": g["epa"].mean(), "Success": g["success"].mean(),
        "Explosive": g["explosive"].mean(), "Pass rate": g["is_pass"].mean(),
        "PROE": snaps[snaps["neutral"]].groupby(by)["pass_oe"].mean() / 100,
        "Pass EPA": p["epa"].mean(), "Rush EPA": r["epa"].mean(), "Pass success": p["success"].mean(),
        "Rush success": r["success"].mean(), "Yds/play": g["yards_gained"].mean(), "Sack rate": p["sack"].mean(),
        "Turnover rate": g["turnover"].mean(), "CPOE": p["cpoe"].mean(), "aDOT": p["air_yards"].mean(),
        "Pressure rate": p["pressure"].mean(), "Blitz rate": p["blitz"].mean(),
        "3rd down conv": snaps[snaps["down"] == 3].groupby(by)["first_down"].mean(),
    })
    out.index.name = None
    return out


SUMMARY_FMT = {"Plays": "{:,.0f}", "EPA/play": "{:+.3f}", "Success": "{:.1%}", "Explosive": "{:.1%}",
               "Pass rate": "{:.0%}", "PROE": "{:+.1%}", "Pass EPA": "{:+.3f}", "Rush EPA": "{:+.3f}",
               "Pass success": "{:.1%}", "Rush success": "{:.1%}", "Yds/play": "{:.2f}", "Sack rate": "{:.1%}",
               "Turnover rate": "{:.1%}", "CPOE": "{:+.1f}", "aDOT": "{:.1f}", "Pressure rate": "{:.0%}",
               "Blitz rate": "{:.0%}", "3rd down conv": "{:.0%}", "Points/drive": "{:.2f}", "RZ TD rate": "{:.0%}"}
# For the offense, higher is better except these. A defense's view flips everything.
LOWER_BETTER = {"Sack rate", "Turnover rate", "Pressure rate"}
NEUTRAL = {"Plays", "Pass rate", "PROE", "aDOT", "Blitz rate"}   # style, not good or bad


def drive_summary(games: pd.DataFrame, side: str) -> pd.DataFrame:
    rows = {}
    for t in sorted(games["posteam"].dropna().unique()):
        k = td.drive_kpis(td.drives_frame(games, t, side))
        rows[t] = {"Points/drive": k.get("Points / drive"), "RZ TD rate": k.get("RZ TD rate")}
    return pd.DataFrame(rows).T


@st.cache_data(show_spinner=False, max_entries=20)
def league_tables(plays: pd.DataFrame, games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    snaps = plays[plays["scrimmage"]]
    off = team_summary(snaps, "posteam").join(drive_summary(games, "offense"))
    deff = team_summary(snaps, "defteam").join(drive_summary(games, "defense"))
    return off, deff


def percentile(table: pd.DataFrame, col: str, value: float, higher_better: bool) -> float:
    s = table[col].dropna()
    if s.empty or pd.isna(value):
        return np.nan
    pct = (s < value).mean() * 100 + (s == value).mean() * 50
    return pct if higher_better else 100 - pct


def rank(table: pd.DataFrame, col: str, team: str, higher_better: bool) -> str:
    s = table[col].dropna().rank(ascending=not higher_better, method="min")
    return f"{int(s[team])} of {len(s)}" if team in s else "–"


# ---- Home ------------------------------------------------------------------------------------------------
def home(ctx: Ctx):
    off, deff = league_tables(ctx.plays, ctx.games)
    st.caption("Every team's offense and defense in the selected filters. Pick a team in the sidebar to dig in.")
    c1, c2 = st.columns(2)
    colors = ctx.teams["team_color"].to_dict()
    both = pd.DataFrame({"Offense EPA/play": off["EPA/play"], "Defense EPA/play allowed": deff["EPA/play"]}).dropna()
    with c1:
        st.markdown("**Offense vs defense**")
        ch.league_scatter(both, "Offense EPA/play", "Defense EPA/play allowed", "Offense EPA/play (→ better)",
                          "Defense EPA/play allowed (↑ better)", [ctx.team], colors, flip_y=True)
    with c2:
        st.markdown("**Pass vs run efficiency (offense)**")
        pr = off[["Pass EPA", "Rush EPA"]].dropna()
        ch.league_scatter(pr, "Rush EPA", "Pass EPA", "Rush EPA/play", "Pass EPA/play", [ctx.team], colors)
    st.markdown("**League table**")
    side = st.segmented_control("Side", ["Offense", "Defense"], default="Offense", key="home_side") or "Offense"
    t = off if side == "Offense" else deff
    cols = ["Plays", "EPA/play", "Success", "Explosive", "Pass EPA", "Rush EPA", "Points/drive", "RZ TD rate",
            "3rd down conv", "Pass rate", "PROE", "Sack rate", "Turnover rate"]
    t = t[cols].sort_values("EPA/play", ascending=side == "Defense")
    good_high = side == "Offense"

    def shade(col):
        if col.name in NEUTRAL:
            return [""] * len(col)
        up = (col.name not in LOWER_BETTER) == good_high
        q_hi, q_lo = col.quantile(0.75), col.quantile(0.25)
        return [tb.GOOD_STYLE if (v >= q_hi if up else v <= q_lo) else tb.BAD_STYLE if (v <= q_lo if up else v >= q_hi)
                else "" for v in col]

    st.dataframe(t.style.format(SUMMARY_FMT, na_rep="").apply(shade, axis=0), width="stretch", height=36 * 33)
    st.caption("Green = top quarter of the league, red = bottom quarter. PROE is from neutral situations (score within "
               f"{dt.NEUTRAL_MARGIN}, outside the last 2 minutes of a half). Defense = what it allowed.")


# ---- Offense ---------------------------------------------------------------------------------------------
def kpi_row(team_d: pd.DataFrame, league_d: pd.DataFrame, good_high: bool = True):
    items = [("EPA / play", "epa", "{:+.2f}"), ("Success", "success", "{:.0%}"), ("Explosive", "explosive", "{:.1%}"),
             ("Yards / play", "yards_gained", "{:.1f}"), ("Pass rate", "is_pass", "{:.0%}")]
    cols = st.columns(len(items) + 1)
    for col, (label, c, f) in zip(cols, items):
        v, ref = team_d[c].mean(), league_d[c].mean()
        delta = None if pd.isna(v) else (f"{(v - ref):+.0%}" if "%" in f else f"{v - ref:+.2f}") + " vs NFL"
        col.metric(label, "–" if pd.isna(v) else f.format(v), delta,
                   delta_color="off" if label == "Pass rate" else "normal" if good_high else "inverse")
    cols[-1].metric("Plays", f"{len(team_d):,}")


def offense_overview(ctx: Ctx):
    team_banner(ctx, side="Offense")
    o, lg = ctx.offense(), ctx.snaps
    if o.empty:
        st.info("No plays match the current filters.")
        return
    kpi_row(o, lg)
    pers = tb.crosstab(o, "Personnel", ["Plays", "Share", "Pass Rate"] + tb.PLAY_RESULTS[1:], sort_by_count=True)
    if not pers.empty:
        tb.render_takeaways(pers, "personnel group")
    by_down = tb.crosstab(o, "down", ["Plays", "Pass Rate", "PROE"] + tb.PLAY_RESULTS[1:])
    by_down["NFL pass rate"] = lg.groupby("down")["is_pass"].mean().reindex(by_down.index)
    by_down["NFL EPA"] = lg.groupby("down")["epa"].mean().reindex(by_down.index)
    by_down.index = [["1st", "2nd", "3rd", "4th"][int(i) - 1] for i in by_down.index]
    by_down.index.name = "Down"
    tb.show_table("By down", by_down)
    c1, c2 = st.columns(2)
    with c1:
        if tb.coverage_note(o, "Personnel", "personnel"):
            pers["NFL pass rate"] = lg.groupby("Personnel")["is_pass"].mean().reindex(pers.index)
            tb.show_table("Personnel", pers)
    with c2:
        forms = tb.crosstab(o, "Formation", ["Plays", "Share", "Pass Rate"] + tb.PLAY_RESULTS[1:], sort_by_count=True)
        forms["NFL pass rate"] = lg.groupby("Formation")["is_pass"].mean().reindex(forms.index)
        tb.show_table("Formation", forms)
    pf = tb.crosstab(o, ["Personnel", "Formation"], ["Plays", "Share", "Pass Rate"] + tb.PLAY_RESULTS[1:], sort_by_count=True)
    if not pf.empty:
        tb.render_usage_scatter(pf, f"{ctx.team} personnel + formation: EPA vs success", noun="Look", min_plays=15)
    st.markdown("**Game situation**")
    c1, c2 = st.columns(2)
    with c1:
        sc = tb.crosstab(o, "Score", ["Plays", "Pass Rate", "PROE", "Success Rate", "EPA per Play"],
                         order=["Down 9+", "Down 1-8", "Tied", "Up 1-8", "Up 9+"])
        sc["NFL pass rate"] = lg.groupby("Score")["is_pass"].mean().reindex(sc.index)
        tb.show_table("By score", sc)
    with c2:
        q = tb.crosstab(o, "Quarter", ["Plays", "Pass Rate", "PROE", "Success Rate", "EPA per Play"])
        q["NFL pass rate"] = lg.groupby("Quarter")["is_pass"].mean().reindex(q.index)
        tb.show_table("By quarter", q)


def offense_run_game(ctx: Ctx):
    team_banner(ctx, side="Run game")
    o, lg = ctx.offense(), ctx.snaps
    runs, lg_runs = o[o["is_run"]], lg[lg["is_run"]]
    ch.render_run_lanes(o, f"{ctx.team} runs by lane", league=lg)
    c1, c2 = st.columns(2)
    with c1:
        t = tb.crosstab(runs, "Personnel", tb.PLAY_RESULTS, sort_by_count=True)
        if tb.coverage_note(runs, "Personnel", "personnel"):
            tb.render_takeaways(t, "personnel group")
            tb.show_table("Runs by personnel", t)
    with c2:
        t = tb.crosstab(runs, "Formation", tb.PLAY_RESULTS, sort_by_count=True)
        tb.show_table("Runs by formation", t)
    c1, c2 = st.columns(2)
    with c1:
        if tb.coverage_note(runs, "Box", "men-in-the-box"):
            t = tb.crosstab(runs, "Box", ["Plays", "Share"] + tb.PLAY_RESULTS[1:], order=["5 or fewer", "6", "7", "8+"])
            t["NFL EPA"] = lg_runs.groupby("Box")["epa"].mean().reindex(t.index)
            tb.show_table("Runs vs men in the box", t)
    with c2:
        t = tb.crosstab(runs, "Distance", tb.PLAY_RESULTS, order=dt.DISTANCE_ORDER)
        tb.show_table("Runs by distance to go", t)
    rushers = tb.crosstab(runs.assign(Rusher=runs["rusher_player_name"]), "Rusher",
                          ["Plays", "Yards per Play", "Success Rate", "EPA per Play", "Explosive Rate"], sort_by_count=True)
    tb.show_table("Ball carriers", rushers.head(10))


def offense_pass_game(ctx: Ctx):
    team_banner(ctx, side="Pass game")
    o, lg = ctx.offense(), ctx.snaps
    passes, lg_passes = o[o["is_pass"]], lg[lg["is_pass"]]
    ch.render_pass_zones(o, f"{ctx.team} pass zones", key="pz_off", league=lg)
    c1, c2 = st.columns(2)
    with c1:
        t = tb.crosstab(passes, "Air Yards", ["Plays", "Share", "Comp %", "Success Rate", "EPA per Play"],
                        order=dt.AIR_ROWS[::-1])
        t["NFL EPA"] = lg_passes.groupby("Air Yards")["epa"].mean().reindex(t.index)
        tb.show_table("By depth", t)
    with c2:
        t = tb.crosstab(passes, "Formation", ["Plays", "Comp %", "aDOT", "Sack Rate", "Success Rate", "EPA per Play"],
                        sort_by_count=True)
        tb.show_table("Dropbacks by formation", t)
    st.markdown("**Against the defense's call**")
    if tb.coverage_note(passes, "Coverage", "coverage"):
        c1, c2 = st.columns(2)
        with c1:
            t = tb.crosstab(passes, "Coverage", ["Plays", "Share", "Comp %", "aDOT", "Success Rate", "EPA per Play"],
                            sort_by_count=True)
            t["NFL EPA"] = lg_passes.groupby("Coverage")["epa"].mean().reindex(t.index)
            tb.render_takeaways(t, "coverage")
            tb.show_table("Vs coverage", t)
        with c2:
            t = tb.crosstab(passes, "Man/Zone", ["Plays", "Share", "Comp %", "Success Rate", "EPA per Play"])
            t["NFL EPA"] = lg_passes.groupby("Man/Zone")["epa"].mean().reindex(t.index)
            tb.show_table("Vs man and zone", t)
            t = tb.crosstab(passes.assign(Pressured=passes["pressure"].map({1.0: "Pressured", 0.0: "Clean"})),
                            "Pressured", ["Plays", "Share", "Comp %", "Sack Rate", "EPA per Play"])
            tb.show_table("Clean pocket vs pressure", t)
        t = tb.crosstab(passes, "Rushers", ["Plays", "Share", "Pressure Rate", "Sack Rate", "Success Rate", "EPA per Play"],
                        order=["3 or fewer", "4", "5", "6+"])
        t["NFL EPA"] = lg_passes.groupby("Rushers")["epa"].mean().reindex(t.index)
        tb.show_table("Vs number of pass rushers", t)
    tgt = passes[passes["receiver_player_name"].notna()]
    t = tb.crosstab(tgt.assign(Target=tgt["receiver_player_name"]), "Target",
                    ["Plays", "Share", "Comp %", "aDOT", "Success Rate", "EPA per Play"], sort_by_count=True)
    t = t.rename(columns={"Plays": "Targets"})
    tb.show_table("Targets", t.head(12))
    qbs = tb.crosstab(passes.assign(Passer=passes["passer_player_name"]), "Passer",
                      ["Plays", "Comp %", "CPOE", "aDOT", "Sack Rate", "EPA per Play"], sort_by_count=True)
    tb.show_table("Passers", qbs.rename(columns={"Plays": "Dropbacks"}).head(5))


def weekly_trends(ctx: Ctx, side: str = "offense"):
    team_banner(ctx, side="Weekly trends")
    d = ctx.offense() if side == "offense" else ctx.defense()
    color = ctx.color()
    c1, c2 = st.columns(2)
    with c1:
        ch.weekly_trend(d, ctx.snaps, "EPA per Play", "EPA per play by game", "+.2f", color)
        ch.weekly_trend(d[d["is_pass"]], ctx.snaps[ctx.snaps["is_pass"]], "EPA per Play", "Pass EPA by game", "+.2f", color)
    with c2:
        ch.weekly_trend(d, ctx.snaps, "Success Rate", "Success rate by game", ".0%", color)
        ch.weekly_trend(d[d["is_run"]], ctx.snaps[ctx.snaps["is_run"]], "EPA per Play", "Rush EPA by game", "+.2f", color)
    if side == "offense":
        n = ctx.snaps[ctx.snaps["neutral"]]
        ch.weekly_trend(d[d["neutral"]], n, "PROE", "Pass rate over expected by game (neutral situations)", "+.0%", color)
    st.caption("Uses the down/distance/situation filters; the Week filter is ignored here so every game shows. "
               "Defense pages show what the defense allowed, so lower is better there.")


# ---- Defense ---------------------------------------------------------------------------------------------
def defense_overview(ctx: Ctx):
    team_banner(ctx, side="Defense (what it allowed)")
    d, lg = ctx.defense(), ctx.snaps
    if d.empty:
        st.info("No plays match the current filters.")
        return
    kpi_row(d, lg, good_high=False)
    by_down = tb.crosstab(d, "down", ["Plays", "Pass Rate"] + tb.PLAY_RESULTS[1:])
    by_down["NFL EPA"] = lg.groupby("down")["epa"].mean().reindex(by_down.index)
    by_down.index = [["1st", "2nd", "3rd", "4th"][int(i) - 1] for i in by_down.index]
    tb.show_table("By down", by_down, good_high=False)
    c1, c2 = st.columns(2)
    with c1:
        if tb.coverage_note(d, "Personnel", "personnel"):
            t = tb.crosstab(d, "Personnel", ["Plays", "Share", "Pass Rate"] + tb.PLAY_RESULTS[1:], sort_by_count=True)
            tb.render_takeaways(t, "personnel group faced", good_high=False)
            tb.show_table("Vs offensive personnel", t, good_high=False)
    with c2:
        if tb.coverage_note(d, "Def Package", "defensive personnel"):
            t = tb.crosstab(d, "Def Package", ["Plays", "Share", "Pass Rate"] + tb.PLAY_RESULTS[1:], sort_by_count=True)
            t["NFL share"] = lg["Def Package"].value_counts(normalize=True).reindex(t.index)
            tb.show_table("Defensive package", t, good_high=False)
            pk = d.dropna(subset=["Personnel", "Def Package"])
            if not pk.empty:
                mix = pd.crosstab(pk["Personnel"], pk["Def Package"], normalize="index")
                top = pk["Personnel"].value_counts().head(5).index
                tb.show_mix("Package they answer each personnel with", mix.loc[top])
    tb.render_usage_scatter(
        tb.crosstab(d, ["Personnel", "Formation"], ["Plays", "Success Rate", "EPA per Play"], sort_by_count=True),
        "Offensive looks faced: EPA vs success allowed", good_high=False, noun="Look", min_plays=15)


def defense_pass(ctx: Ctx):
    team_banner(ctx, side="Pass defense")
    d, lg = ctx.defense(), ctx.snaps
    passes, lg_passes = d[d["is_pass"]], lg[lg["is_pass"]]
    ch.render_pass_zones(d, f"Throws against {ctx.team}", key="pz_def", good_high=False, league=lg)
    if not tb.coverage_note(passes, "Coverage", "coverage"):
        return
    c1, c2 = st.columns(2)
    with c1:
        t = tb.crosstab(passes, "Coverage", ["Plays", "Share", "Comp %", "Success Rate", "EPA per Play"], sort_by_count=True)
        t["NFL share"] = lg_passes["Coverage"].value_counts(normalize=True).reindex(t.index)
        tb.render_takeaways(t, "coverage", good_high=False)
        tb.show_table("Coverage usage and results", t, good_high=False)
    with c2:
        t = tb.crosstab(passes, "Rushers", ["Plays", "Share", "Pressure Rate", "Sack Rate", "Success Rate", "EPA per Play"],
                        order=["3 or fewer", "4", "5", "6+"])
        t["NFL share"] = lg_passes["Rushers"].value_counts(normalize=True).reindex(t.index)
        tb.show_table("Pass rush: how many they send", t, good_high=False)
    st.markdown("**Coverage and blitz tendencies by situation**")
    c1, c2 = st.columns(2)
    with c1:
        cov = passes.dropna(subset=["Coverage", "down"])
        if not cov.empty:
            mix = pd.crosstab([cov["down"].astype(int).map({1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}), cov["Distance"]],
                              cov["Coverage"], normalize="index")
            top = cov["Coverage"].value_counts().head(6).index
            tb.show_mix("Coverage mix by down & distance", mix[top])
    with c2:
        t = tb.crosstab(passes, ["down", "Distance"], ["Plays", "Blitz Rate", "Pressure Rate", "Sack Rate", "EPA per Play"])
        t["NFL blitz rate"] = lg_passes.groupby(["down", "Distance"], observed=True)["blitz"].mean().reindex(t.index)
        tb.show_table("Blitz and pressure by down & distance", t, good_high=False)
    t = tb.crosstab(passes, ["Def Package", "Man/Zone"], ["Plays", "Share", "Blitz Rate", "Comp %", "EPA per Play"],
                    sort_by_count=True)
    tb.show_table("Package × man/zone", t, good_high=False)


def defense_run(ctx: Ctx):
    team_banner(ctx, side="Run defense")
    d, lg = ctx.defense(), ctx.snaps
    runs, lg_runs = d[d["is_run"]], lg[lg["is_run"]]
    ch.render_run_lanes(d, f"Runs against {ctx.team} by lane", good_high=False, league=lg)
    c1, c2 = st.columns(2)
    with c1:
        if tb.coverage_note(runs, "Box", "men-in-the-box"):
            t = tb.crosstab(runs, "Box", ["Plays", "Share"] + tb.PLAY_RESULTS[1:], order=["5 or fewer", "6", "7", "8+"])
            t["NFL share"] = lg_runs["Box"].value_counts(normalize=True).reindex(t.index)
            tb.show_table("Box count on runs", t, good_high=False)
    with c2:
        t = tb.crosstab(runs, "Personnel", tb.PLAY_RESULTS, sort_by_count=True)
        tb.show_table("Runs allowed by offensive personnel", t, good_high=False)
    t = tb.crosstab(runs, "Distance", tb.PLAY_RESULTS, order=dt.DISTANCE_ORDER)
    tb.show_table("Runs allowed by distance to go", t, good_high=False)


# ---- Scouting ------------------------------------------------------------------------------------------
def tendencies(ctx: Ctx):
    team_banner(ctx, side="Offensive tendencies")
    o, lg = ctx.offense(), ctx.snaps
    if o.empty:
        st.info("No plays match the current filters.")
        return
    tabs = st.tabs(["Tells", "Down & distance", "Tendency tree", "Sequencing", "Field position", "3rd & 4th downs"])
    with tabs[0]:
        td.render_tells(o, lg, ctx.team)
    with tabs[1]:
        td.render_down_distance(o, lg, f"{ctx.team} run/pass by down & distance")
    with tabs[2]:
        levels = st.segmented_control("Branch by", ["Personnel → formation", "Formation → personnel",
                                                     "Down → distance"], default="Personnel → formation",
                                      key="tree_levels") or "Personnel → formation"
        lv = {"Personnel → formation": ("Personnel", "Formation"), "Formation → personnel": ("Formation", "Personnel"),
              "Down → distance": ("down", "Distance")}[levels]
        td.render_tendency_tree(o, lg, f"{ctx.team} tendency tree", levels=lv)
    with tabs[3]:
        td.render_sequencing(o, lg, ctx.team)
        first = tb.crosstab(o[o["Drive Snap"] == 1], "down", ["Plays", "Pass Rate", "Success Rate", "EPA per Play"])
        first["NFL pass rate"] = lg[lg["Drive Snap"] == 1].groupby("down")["is_pass"].mean().reindex(first.index)
        first.index = ["First snap of a drive"] * len(first)
        tb.show_table("Opening snap of each drive", first)
    with tabs[4]:
        td.render_field_position(o, lg, f"{ctx.team} by field position")
    with tabs[5]:
        td.render_third_downs(o, lg, f"{ctx.team} on 3rd down")
        td.render_fourth_downs(ctx.plays[ctx.plays["posteam"] == ctx.team], ctx.plays, ctx.team)


def red_zone(ctx: Ctx, side: str):
    team_banner(ctx, side="Red zone offense" if side == "offense" else "Red zone defense")
    d = ctx.offense() if side == "offense" else ctx.defense()
    td.render_red_zone(d, ctx.snaps, ctx.games, ctx.games, ctx.team, side, good_high=side == "offense")


PROFILE_ROWS = ["EPA/play", "Success", "Explosive", "Pass EPA", "Rush EPA", "Pass success", "Rush success",
                "Points/drive", "RZ TD rate", "3rd down conv", "Sack rate", "Turnover rate", "CPOE", "Pressure rate"]


def _profile(table: pd.DataFrame, team: str, offense: bool) -> pd.DataFrame:
    rows = {}
    for m in PROFILE_ROWS:
        if m not in table or team not in table.index or pd.isna(table.loc[team, m]):
            continue
        hb = (m not in LOWER_BETTER) == offense
        v = table.loc[team, m]
        rows[m] = {"Value": SUMMARY_FMT[m].format(v), "Pct": percentile(table, m, v, hb), "Rank": rank(table, m, team, hb)}
    return pd.DataFrame(rows).T.astype({"Pct": float}) if rows else pd.DataFrame()


def team_profiles(ctx: Ctx):
    off, deff = league_tables(ctx.plays, ctx.games)
    teams = [ctx.team] + ([ctx.opponent] if ctx.opponent != ctx.team else [])
    for col, t in zip(st.columns(len(teams)), teams):
        with col:
            team_banner(ctx, t, "League percentiles")
            p = _profile(off, t, True)
            if not p.empty:
                ch.percentile_bars(p, "Offense", ctx.color(t))
            p = _profile(deff, t, False)
            if not p.empty:
                ch.percentile_bars(p, "Defense (100 = allowed the least)", ctx.color(t, "team_color2"))
    st.caption("Percentile among all 32 teams in the same filters (100 = best). Pressure rate on defense = pressure "
               "created. Hover a bar for the rank.")


def matchup(ctx: Ctx):
    a, b = ctx.team, ctx.opponent
    if a == b:
        st.info("Pick a different opponent in the sidebar.")
        return
    off, deff = league_tables(ctx.plays, ctx.games)
    st.caption("Each unit's league rank in the same filters. The side with the better rank has the edge.")
    metrics = ["EPA/play", "Success", "Explosive", "Pass EPA", "Rush EPA", "Points/drive", "RZ TD rate", "3rd down conv",
               "Sack rate", "Turnover rate"]
    for o_team, d_team in [(a, b), (b, a)]:
        st.subheader(f"{o_team} offense vs {d_team} defense")
        rows = []
        for m in metrics:
            hb = m not in LOWER_BETTER
            ov = off.loc[o_team, m] if o_team in off.index else np.nan
            dv = deff.loc[d_team, m] if d_team in deff.index else np.nan
            op, dp = percentile(off, m, ov, hb), percentile(deff, m, dv, not hb)
            edge = "–" if pd.isna(op) or pd.isna(dp) or abs(op - dp) < 15 else (f"{o_team} O" if op > dp else f"{d_team} D")
            rows.append({"Metric": m, f"{o_team} O": SUMMARY_FMT[m].format(ov) if ov == ov else "–",
                         "O rank": rank(off, m, o_team, hb), f"{d_team} D allowed": SUMMARY_FMT[m].format(dv) if dv == dv else "–",
                         "D rank": rank(deff, m, d_team, not hb), "Edge": edge})
        t = pd.DataFrame(rows).set_index("Metric")
        st.dataframe(t.style.apply(lambda c: [tb.GOOD_STYLE if v == f"{o_team} O" else tb.BAD_STYLE if v == f"{d_team} D"
                                              else "" for v in c], subset=["Edge"]), width="stretch")
        c1, c2 = st.columns(2)
        o, dd = ctx.offense(o_team), ctx.defense(d_team)
        with c1:
            t = tb.crosstab(o, "Personnel", ["Plays", "Share", "Pass Rate", "EPA per Play"], sort_by_count=True).head(5)
            t2 = tb.crosstab(dd, "Personnel", ["Plays", "EPA per Play"]).rename(
                columns={"Plays": f"{d_team} D faced", "EPA per Play": f"{d_team} D EPA allowed"})
            t = t.join(t2)
            tb.show_table(f"{o_team}'s personnel vs how {d_team} defends it", t)
        with c2:
            if dd["Coverage"].notna().any():
                t = tb.crosstab(dd[dd["is_pass"]], "Coverage", ["Plays", "Share", "EPA per Play"], sort_by_count=True).head(6)
                t2 = tb.crosstab(o[o["is_pass"]], "Coverage", ["Plays", "EPA per Play"]).rename(
                    columns={"Plays": f"{o_team} throws", "EPA per Play": f"{o_team} EPA vs it"})
                t = t.join(t2)
                tb.show_table(f"{d_team}'s coverages vs how {o_team} attacks them", t, good_high=False)


def season_comparison(ctx: Ctx, all_plays: pd.DataFrame):
    team_banner(ctx, side="Season comparison")
    seasons = sorted(all_plays["Season"].unique())
    if len(seasons) < 2:
        st.info("Load at least two seasons (sidebar → Seasons loaded) to compare.")
        return
    c1, c2 = st.columns(2)
    s1 = c1.selectbox("Season A", seasons, index=len(seasons) - 2, key="sc_a")
    s2 = c2.selectbox("Season B", seasons, index=len(seasons) - 1, key="sc_b")
    snaps = all_plays[all_plays["scrimmage"]]
    rows = {}
    for s in (s1, s2):
        sd = snaps[snaps["Season"] == s]
        g = all_plays[all_plays["Season"] == s]
        o = team_summary(sd, "posteam").join(drive_summary(g, "offense"))
        de = team_summary(sd, "defteam").join(drive_summary(g, "defense"))
        rows[s] = (o, de)
    for side, idx, off_side in [("Offense", 0, True), ("Defense (allowed)", 1, False)]:
        out = []
        for m in PROFILE_ROWS + ["Pass rate", "PROE"]:
            a, b = rows[s1][idx], rows[s2][idx]
            if ctx.team not in a.index or ctx.team not in b.index or m not in a:
                continue
            va, vb = a.loc[ctx.team, m], b.loc[ctx.team, m]
            hb = (m not in LOWER_BETTER) == off_side
            out.append({"Metric": m, str(s1): SUMMARY_FMT[m].format(va) if va == va else "–",
                        f"{s1} rank": rank(a, m, ctx.team, hb), str(s2): SUMMARY_FMT[m].format(vb) if vb == vb else "–",
                        f"{s2} rank": rank(b, m, ctx.team, hb),
                        "Change": "–" if pd.isna(va) or pd.isna(vb) else SUMMARY_FMT[m].replace(":", ":+").replace("++", "+").format(vb - va),
                        "_better": None if m in NEUTRAL or pd.isna(va) or pd.isna(vb) or va == vb else (vb > va) == hb})
        t = pd.DataFrame(out).set_index("Metric")
        better = t.pop("_better")
        st.subheader(side)
        st.dataframe(t.style.apply(lambda c: [tb.GOOD_STYLE if b is True else tb.BAD_STYLE if b is False else ""
                                              for b in better], subset=["Change"]), width="stretch")
    st.caption("Whole seasons (the Season and Week filters don't apply here; the others do). Green = better in "
               f"{s2}. Ranks are among all 32 teams that season.")


# ---- Game review ---------------------------------------------------------------------------------------
def game_label(g: pd.DataFrame) -> str:
    r = g.iloc[0]
    wk = f"W{int(r['week'])}" if r["season_type"] == "REG" else "Playoffs"
    return f"{r['season']} {wk}: {r['away_team']} {int(r['away_score'])} @ {r['home_team']} {int(r['home_score'])}"


def game_recap(ctx: Ctx, all_plays: pd.DataFrame):
    games = all_plays[(all_plays["home_team"] == ctx.team) | (all_plays["away_team"] == ctx.team)]
    if games.empty:
        st.info("No games for this team in the loaded seasons.")
        return
    ids = games.sort_values("game_date", ascending=False)["game_id"].unique()
    labels = {gid: game_label(games[games["game_id"] == gid]) for gid in ids}
    gid = st.selectbox("Game", ids, format_func=labels.get, key="g_game")
    g = all_plays[all_plays["game_id"] == gid].sort_values("play_id")
    home, away = g["home_team"].iloc[0], g["away_team"].iloc[0]
    c1, c2, c3 = st.columns([1, 1, 2])
    for col, t, s in [(c1, away, g["away_score"].iloc[0]), (c2, home, g["home_score"].iloc[0])]:
        with col:
            team_banner(ctx, t, f"{'Home' if t == home else 'Away'} · {int(s)}")
    ch.wp_chart(g, home, away, ctx.color(home), ctx.color(away))
    snaps = g[g["scrimmage"]]
    box = team_summary(snaps, "posteam")[["Plays", "EPA/play", "Success", "Explosive", "Pass rate", "Pass EPA", "Rush EPA",
                                          "Yds/play", "Turnover rate", "3rd down conv"]]
    box["Total EPA"] = snaps.groupby("posteam")["epa"].sum()
    st.markdown("**Head to head**")
    st.dataframe(box.style.format({**SUMMARY_FMT, "Total EPA": "{:+.1f}"}, na_rep=""), width="stretch")
    c1, c2 = st.columns(2)
    for col, t in [(c1, away), (c2, home)]:
        with col:
            dr = td.drives_frame(g, t, "offense")
            if not dr.empty:
                st.markdown(f"**{t} drives**")
                show = dr.assign(Start=dr["start"].map(lambda s: f"own {s:.0f}" if s <= 50 else f"opp {100 - s:.0f}"))
                st.dataframe(show[["fixed_drive", "Start", "plays", "result"]].rename(
                    columns={"fixed_drive": "Drive", "plays": "Plays", "result": "Result"}), hide_index=True, width="stretch")
    st.markdown("**Biggest swings**")
    big = g.dropna(subset=["wpa"]).assign(swing=lambda x: x["wpa"].abs()).nlargest(10, "swing")
    st.dataframe(big[["qtr", "posteam", "down", "ydstogo", "yardline_100", "epa", "wpa", "desc"]].rename(
        columns={"qtr": "Qtr", "posteam": "Offense", "down": "Down", "ydstogo": "To go", "yardline_100": "Yds to goal",
                 "epa": "EPA", "wpa": "WPA (offense)", "desc": "Play"}).style.format(
        {"EPA": "{:+.2f}", "WPA (offense)": "{:+.1%}", "Down": "{:.0f}", "To go": "{:.0f}", "Yds to goal": "{:.0f}"},
        na_rep=""), hide_index=True, width="stretch")
    with st.expander("Every play"):
        st.dataframe(g[["qtr", "posteam", "down", "ydstogo", "yardline_100", "Formation", "Personnel", "Play Type",
                        "yards_gained", "epa", "wp", "desc"]], hide_index=True, width="stretch")


# ---- Tools ---------------------------------------------------------------------------------------------
def play_finder(ctx: Ctx):
    d = ctx.plays[ctx.plays["scrimmage"]]
    c1, c2, c3, c4 = st.columns(4)
    side = c1.radio("Team is on", ["Offense", "Defense"], horizontal=True, key="pf_side")
    d = d[d["posteam" if side == "Offense" else "defteam"] == ctx.team]
    pt = c2.multiselect("Play type", ["Run", "Pass"], default=["Run", "Pass"], key="pf_type")
    d = d[d["Play Type"].isin(pt)]
    pers = c3.multiselect("Personnel", sorted(d["Personnel"].dropna().unique()), key="pf_pers")
    if pers:
        d = d[d["Personnel"].isin(pers)]
    form = c4.multiselect("Formation", sorted(d["Formation"].dropna().unique()), key="pf_form")
    if form:
        d = d[d["Formation"].isin(form)]
    c1, c2, c3 = st.columns(3)
    cov = c1.multiselect("Coverage", sorted(d["Coverage"].dropna().unique()), key="pf_cov")
    if cov:
        d = d[d["Coverage"].isin(cov)]
    fz = c2.multiselect("Field zone", dt.FIELD_ZONES, key="pf_fz")
    if fz:
        d = d[d["Field Zone"].isin(fz)]
    text = c3.text_input("Description contains", key="pf_text", placeholder="e.g. Kelce, play action, deep left")
    if text:
        d = d[d["desc"].str.contains(text, case=False, na=False, regex=False)]
    st.caption(f"{len(d):,} plays · EPA {d['epa'].mean():+.2f} · {d['success'].mean():.0%} success" if len(d) else "No plays.")
    st.dataframe(d.sort_values(["Season", "Week", "play_id"], ascending=False)[
        ["Season", "Week", "posteam", "defteam", "qtr", "down", "ydstogo", "yardline_100", "Personnel", "Formation",
         "Coverage", "Play Type", "yards_gained", "epa", "desc"]].rename(columns={
            "posteam": "Offense", "defteam": "Defense", "qtr": "Qtr", "down": "Down", "ydstogo": "To go",
            "yardline_100": "Yds to goal", "yards_gained": "Yards", "epa": "EPA", "desc": "Play"}),
        hide_index=True, width="stretch", height=600)
    st.download_button("Download these plays (CSV)", d.drop(columns=["desc"]).to_csv(index=False).encode(),
                       file_name=f"{ctx.team}_plays.csv", mime="text/csv")


def glossary():
    st.markdown(f"""
**Where the data comes from:** [nflfastR](https://www.nflfastr.com/) play-by-play from the
[nflverse data releases](https://github.com/nflverse/nflverse-data/releases), updated through the season.
Personnel, formation, coverage, men in the box, pass rushers, and pressure come from nflverse's FTN/NGS
participation data, which is published after each season, so the season in progress only has nflfastR's
shotgun / under center flag for formation and no personnel or coverage yet.

**EPA (expected points added):** how many points a play was worth, from nflfastR's expected points model (down,
distance, field position, time). Above 0 is good for the offense.

**Success:** a play with positive EPA (nflfastR's definition, not the 40/70/100% yardage rule).

**Explosive:** a run of 10+ yards or a pass of 20+ yards.

**PROE (pass rate over expected):** how much more often a team passes than nflfastR's `xpass` model expects an
average team to in the same down, distance, field position, score, and time. The cleanest way to compare play
calling across teams.

**CPOE:** completion % over expected, from nflfastR's completion probability model.

**Personnel:** RBs (including FBs) then TEs: "11" = 1 RB, 1 TE, 3 WR. "+OL" = an extra offensive lineman.

**Defensive package:** Base = 4 DBs, Nickel = 5, Dime = 6, Quarter = 7+.

**Pass zones:** nflfastR's pass direction (left / middle / right) by air yards at the catch point or target.
**Run lanes:** nflfastR's run location and gap: end = outside the tackle.

**Neutral situations:** score within {dt.NEUTRAL_MARGIN} and not in the last 2 minutes of either half.
**Garbage time removed:** plays where the offense's win probability is between 10% and 90%.

**Comparisons to the NFL** always use the same sidebar filters across all 32 teams, so a "vs NFL" number is apples
to apples.

**Colors on offense tables:** green = clearly above the table's average. Red only for results that are bad on their
own: negative EPA, under 4.5 yards per play, or under a 40% success rate. **On defense tables**, green = good for the
defense. Gray rows have fewer than {tb.LOW_N_ROWS} plays.

**Tells:** a situation where a team's pass rate is at least {td.TELL_MIN_GAP:.0%} away from both its own usual split
and the league's in the same situation, and a two-proportion z-test says it's unlikely to be noise
(|z| ≥ {td.Z_FLAG}) on both comparisons.
""")
