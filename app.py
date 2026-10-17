"""
NFL Tendencies: a scouting and tendency dashboard on nflfastR play-by-play, built like the Dowling Catholic
scouting app (same tables, colors, takeaway cards, tendency views), with every number compared to the league.

Run with:  streamlit run app.py

    app.py          entrypoint: navigation, sidebar filters, page layouts
    data.py         downloads nflfastR pbp + participation from nflverse and builds the app's columns
    tables.py       crosstabs, table coloring, takeaway cards, EPA vs success scatter
    charts.py       pass zones, run lanes, weekly trends, league scatter, percentile bars, win probability
    tendencies.py   tells, down & distance grid, tendency tree, sequencing, field position, red zone, 3rd/4th downs
    views.py        page bodies
"""
import pandas as pd
import streamlit as st

import data as dt
import tables as tb
import views as vw

st.set_page_config(page_title="NFL Tendencies", page_icon="🏈", layout="wide")


@st.cache_data(show_spinner="Loading nflfastR play-by-play…", ttl=dt.REFRESH_HOURS * 3600)
def load(seasons: tuple[int, ...]) -> pd.DataFrame:
    return dt.load_seasons(list(seasons))


@st.cache_data(show_spinner=False)
def load_teams() -> pd.DataFrame:
    return dt.load_teams()


SITUATIONS = {
    "All plays": lambda d: d,
    "Neutral (score within 10, not 2-min)": lambda d: d[d["neutral"]],
    "Garbage time removed (WP 10-90%)": lambda d: d[d["wp"].between(0.10, 0.90)],
    "Early downs (1st & 2nd)": lambda d: d[d["down"].isin([1, 2])],
    "Two-minute (end of either half)": lambda d: d[d["two_minute"]],
    "First half": lambda d: d[d["qtr"] <= 2],
    "Second half & OT": lambda d: d[d["qtr"] >= 3],
    "Red zone": lambda d: d[d["red_zone"]],
    "Leading": lambda d: d[d["score_differential"] > 0],
    "Trailing": lambda d: d[d["score_differential"] < 0],
}

# Keep filter picks when visiting a page without the filters and coming back.
for _k in list(st.session_state.keys()):
    if _k.startswith("f_"):
        st.session_state[_k] = st.session_state[_k]


def page_header(title: str):
    st.title(title)
    chips = active_filter_chips()
    st.html('<div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-size:12px">'
            '<span style="opacity:.6;margin-right:2px">Showing</span>'
            + "".join(f'<span style="padding:3px 10px;border-radius:999px;border:0.5px solid rgba(128,128,128,.4);'
                      f'background:rgba(128,128,128,.08)">{c}</span>' for c in chips) + "</div>")


# ---- Pages -------------------------------------------------------------------------------------------------
def _page(title, body):
    def run():
        page_header(title)
        body(ctx)
    run.__name__ = title.lower().replace(" ", "_").replace("&", "and")
    return run


def home_page():
    page_header("NFL Tendencies")
    vw.home(ctx)


def offense_trends():
    page_header("Offense Weekly Trends")
    vw.weekly_trends(ctx_all_weeks, "offense")


def defense_trends():
    page_header("Defense Weekly Trends")
    vw.weekly_trends(ctx_all_weeks, "defense")


def season_comparison_page():
    page_header("Season Comparison")
    vw.season_comparison(ctx, filtered(all_plays, use_season=False, use_week=False))


def game_recap_page():
    st.title("Game Recap")
    vw.game_recap(ctx, all_plays)


def how_to_read():
    st.title("How to Read This App")
    vw.glossary()


HOME = st.Page(home_page, title="League Overview", icon=":material/home:", url_path="home", default=True)
PAGES = {
    "Offense": [
        st.Page(_page("Offense Overview", vw.offense_overview), title="Offense Overview", url_path="offense"),
        st.Page(_page("Run Game", vw.offense_run_game), title="Run Game", url_path="run-game"),
        st.Page(_page("Pass Game", vw.offense_pass_game), title="Pass Game", url_path="pass-game"),
        st.Page(_page("Offense Red Zone", lambda c: vw.red_zone(c, "offense")), title="Offense Red Zone",
                url_path="red-zone-offense"),
        st.Page(offense_trends, title="Offense Weekly Trends", url_path="offense-trends"),
    ],
    "Defense": [
        st.Page(_page("Defense Overview", vw.defense_overview), title="Defense Overview", url_path="defense"),
        st.Page(_page("Pass Defense", vw.defense_pass), title="Pass Defense & Coverage", url_path="pass-defense"),
        st.Page(_page("Run Defense", vw.defense_run), title="Run Defense", url_path="run-defense"),
        st.Page(_page("Defense Red Zone", lambda c: vw.red_zone(c, "defense")), title="Defense Red Zone",
                url_path="red-zone-defense"),
        st.Page(defense_trends, title="Defense Weekly Trends", url_path="defense-trends"),
    ],
    "Scouting": [
        st.Page(_page("Tendencies", vw.tendencies), title="Tendencies & Tells", url_path="tendencies"),
        st.Page(_page("Matchup", vw.matchup), title="Matchup", url_path="matchup"),
        st.Page(_page("Team Profiles", vw.team_profiles), title="Team Profiles", url_path="team-profiles"),
        st.Page(season_comparison_page, title="Season Comparison", url_path="season-comparison"),
    ],
    "Game Review": [
        st.Page(game_recap_page, title="Game Recap", url_path="game-recap"),
    ],
    "Tools": [
        st.Page(_page("Play Finder", vw.play_finder), title="Play Finder", icon=":material/search:", url_path="play-finder"),
        st.Page(how_to_read, title="How to Read This", icon=":material/help:", url_path="how-to-read"),
    ],
}
pg = st.navigation({"": [HOME], **PAGES}, expanded=True)
OPPONENT_PAGES = {"matchup", "team-profiles"}


# ---- Sidebar -----------------------------------------------------------------------------------------------
def dropdown_multiselect(label, options, key, fmt=str, default=None):
    """Compact dropdown: a button that opens a checklist, with a one-line summary (same as the Dowling app)."""
    keys = {o: f"{key}_{o}" for o in options}
    for o, k in keys.items():
        st.session_state.setdefault(k, default is None or o in default)

    def set_all(value):
        for k in keys.values():
            st.session_state[k] = value

    with st.popover(label, width="stretch"):
        c1, c2 = st.columns(2)
        c1.button("Select all", key=f"btn_{key}_all", on_click=set_all, args=(True,), width="stretch")
        c2.button("Clear", key=f"btn_{key}_none", on_click=set_all, args=(False,), width="stretch")
        selected = [o for o in options if st.checkbox(fmt(o), key=keys[o])]
    summary = "All" if len(selected) == len(options) else "None selected" if not selected else ", ".join(map(fmt, selected))
    st.caption(f"{label}: {summary}")
    return selected


def _summary(selected, options, fmt=str, all_text="All"):
    if set(selected) == set(options):
        return all_text
    return ", ".join(fmt(o) for o in selected) if selected else "None"


def active_filter_chips() -> list[str]:
    chips = [_summary(sel_seasons, loaded, str, "All loaded seasons"), season_type]
    if week_range != (1, max_week):
        chips.append(f"Weeks {week_range[0]}-{week_range[1]}")
    chips += [_summary(sel_downs, [1, 2, 3, 4], lambda d: f"{d}{['st', 'nd', 'rd', 'th'][d - 1]}", "All downs"),
              _summary(sel_dist, dt.DISTANCE_ORDER, lambda s: s.split(" (")[0], "All distances")]
    if situation != "All plays":
        chips.append(situation)
    if pg.url_path not in ("home", "how-to-read"):
        chips.append(f"Team: {team}")
    if pg.url_path in OPPONENT_PAGES:
        chips.append(f"Opponent: {opponent}")
    return chips


def filtered(d: pd.DataFrame, use_season=True, use_week=True, use_play=True) -> pd.DataFrame:
    if use_season:
        d = d[d["Season"].isin(sel_seasons)]
    if season_type != "Regular season + playoffs":
        d = d[d["Season Type"] == season_type]
    if use_week:
        d = d[d["Week"].between(*week_range) | (d["season_type"] == "POST")]
    if not use_play:
        return d
    if set(sel_downs) != {1, 2, 3, 4}:
        d = d[d["down"].isin(sel_downs)]
    if set(sel_dist) != set(dt.DISTANCE_ORDER):
        d = d[d["Distance"].isin(sel_dist)]
    return SITUATIONS[situation](d)


teams_info = load_teams()
all_seasons = dt.available_seasons()
latest = all_seasons[-1]
with st.sidebar:
    st.header("Data")
    st.session_state.setdefault("f_loaded", all_seasons[-3:])
    loaded = sorted(st.multiselect("Seasons loaded", all_seasons[::-1], key="f_loaded",
                                   help="Seasons to download and keep in memory. More seasons = slower first load."))
    if not loaded:
        st.warning("Pick at least one season.")
        st.stop()

try:
    all_plays = load(tuple(loaded))
except Exception as e:  # first run with no internet, or nflverse hasn't published a season yet
    st.error(f"Couldn't load nflfastR data: {e}")
    st.stop()
# A season that hasn't started yet has no plays; drop it from the choices.
loaded = sorted(all_plays["Season"].unique().tolist())
TEAMS = sorted(all_plays["posteam"].dropna().unique())
max_week = int(all_plays.loc[all_plays["season_type"] == "REG", "Week"].max())

with st.sidebar:
    st.header("Filters")
    st.session_state.setdefault("f_team", "KC" if "KC" in TEAMS else TEAMS[0])
    team = st.selectbox("Team", TEAMS, key="f_team",
                        format_func=lambda t: f"{t} · {teams_info['team_name'].get(t, t)}" if t in teams_info.index else t)
    if pg.url_path in OPPONENT_PAGES:
        st.session_state.setdefault("f_opp", next(t for t in TEAMS if t != team))
        opponent = st.selectbox("Opponent", TEAMS, key="f_opp")
    else:
        opponent = st.session_state.get("f_opp", team)
    sel_seasons = dropdown_multiselect("Season", loaded, key="f_season", default=[loaded[-1]])
    st.session_state.setdefault("f_stype", "Regular season + playoffs")
    season_type = st.selectbox("Season type", ["Regular season", "Playoffs", "Regular season + playoffs"], key="f_stype")
    st.session_state.setdefault("f_weeks", (1, max_week))
    if st.session_state["f_weeks"][1] > max_week:
        st.session_state["f_weeks"] = (1, max_week)
    week_range = st.slider("Weeks (regular season)", 1, max_week, key="f_weeks")
    sel_downs = dropdown_multiselect("Down", [1, 2, 3, 4], key="f_down",
                                     fmt=lambda d: f"{d}{['st', 'nd', 'rd', 'th'][d - 1]}")
    sel_dist = dropdown_multiselect("Distance", dt.DISTANCE_ORDER, key="f_dist")
    situation = st.selectbox("Situation", list(SITUATIONS), key="f_sit")
    st.session_state.setdefault("f_minplays", 1)
    tb.MIN_PLAYS = st.slider("Hide table rows with fewer than … plays", 1, 50, key="f_minplays")

    def _reset():
        for k in list(st.session_state.keys()):
            if k.startswith(("f_down_", "f_dist_")):
                st.session_state[k] = True
            if k.startswith("f_season_"):
                st.session_state[k] = k == f"f_season_{loaded[-1]}"
        st.session_state.update(f_sit="All plays", f_minplays=1, f_weeks=(1, max_week), f_stype="Regular season + playoffs")

    st.button("Reset filters", on_click=_reset, width="stretch")
    last = all_plays["game_date"].max()
    st.caption(f"nflfastR data through {last} · {all_plays['game_id'].nunique():,} games loaded")

if not sel_seasons:
    st.warning("Pick at least one season in the sidebar.")
    st.stop()

plays = filtered(all_plays)
games = filtered(all_plays, use_play=False)
ctx = vw.Ctx(team, opponent, plays, games, teams_info, sel_seasons)
ctx_all_weeks = vw.Ctx(team, opponent, filtered(all_plays, use_week=False), games, teams_info, sel_seasons)

pg.run()
