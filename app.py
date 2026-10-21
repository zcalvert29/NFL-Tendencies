"""
NFL Tendencies: the Dowling Catholic app's Scout Opposing Offense page for any NFL offense, built on nflfastR
play-by-play, with pass rate over expected (PROE) and the league's numbers in the same situation added throughout.

Run with:  streamlit run app.py

    app.py              page, sidebar filters, tabs
    data.py             downloads nflfastR pbp + participation from nflverse and builds the app's columns
    common.py           metrics, NFL comparison columns, table colors, tendency badges, split bars
    tab_tendencies.py   down & distance grid, PROE charts, tendency tables, 3rd/4th downs, best plays
    tab_sequencing.py   what they call after each kind of snap
    tab_red_zone.py     red zone trips and calls
    tab_run_game.py     run-lane diagram and run tables
    tab_pass_game.py    pass zones, pass chart (pass_chart.py), and pass tables
"""
import pandas as pd
import streamlit as st

import common as cm
import data as dt
import tab_pass_game
import tab_red_zone
import tab_run_game
import tab_sequencing
import tab_tendencies

st.set_page_config(page_title="NFL Tendencies", page_icon="🏈", layout="wide")

SITUATIONS = {
    "All plays": None,
    "Neutral (within 7, not last 2 min of a half)": lambda d: d["neutral"],
    "Red zone (inside the 20)": lambda d: d["yardline_100"] <= 20,
    "Goal to go": lambda d: d["ydstogo"] >= d["yardline_100"],
    "Backed up (own 10 and in)": lambda d: d["yardline_100"] >= 90,
    "3rd & short (1-3 yds)": lambda d: (d["DN"] == 3) & (d["ydstogo"] <= 3),
    "Short yardage (3rd/4th & 1-2)": lambda d: d["DN"].isin([3, 4]) & (d["ydstogo"] <= 2),
    "Long yardage (2nd/3rd & 8+)": lambda d: d["DN"].isin([2, 3]) & (d["ydstogo"] >= 8),
    "Two-minute (last 2 min of a half)": lambda d: d["half_seconds_remaining"] <= 120,
}

DEFINITIONS = f"""
**Pass rate over expected (PROE):** how much more often a team throws than an average NFL team would in the exact
same spots (down, distance, field position, score, time left, timeouts), from nflfastR's `xpass` model. +8% = they
throw 8 points more often than expected; negative = run-heavy for the situation.

**NFL columns and markers:** the league's numbers for the same situation, with the same sidebar filters, across all
32 teams. "NFL Pass Rate" on a 3rd & long row is the league's pass rate on 3rd & long. The black tick on a run/pass
bar is the league's split in that situation.

**Success:** a play with positive EPA (nflfastR's definition). **EPA:** expected points added per play.
**Explosive:** a run of 10+ yards or a pass of 20+ yards.

**Colors:** from the scouted offense's side. When a table has the NFL column for a number, green/red = clearly
better/worse than the NFL in that same row. Otherwise the Dowling offense rule: green = clearly above the table's
average, red only for results bad on their own (negative EPA or under a 40% success rate).
Rows with fewer than {cm.COLOR_MIN_PLAYS} plays aren't colored; gray rows have fewer than {cm.LOW_N_ROWS} plays.

**Tell / lean / small sample:** how much to trust a {cm.MIN_TELL_SHARE:.0%}+ split one way, from the lowest share
consistent with the plays seen (Wilson lower bound, 90%): tell = still {cm.TENDENCY_GRADES[0][0]:.0%}+ one way,
lean = {cm.TENDENCY_GRADES[1][0]:.0%}+.

**Formation and personnel:** nflverse's FTN charting, published after each season. The season in progress has
only nflfastR's shotgun / under center flag and no personnel, box count, or coverage yet.
"""


@st.cache_data(show_spinner="Loading nflfastR play-by-play…", ttl=dt.REFRESH_HOURS * 3600)
def load(seasons: tuple[int, ...]) -> pd.DataFrame:
    return dt.load_seasons(list(seasons))


@st.cache_data(show_spinner=False)
def teams() -> pd.DataFrame:
    try:
        return dt.load_teams()
    except Exception:
        return pd.DataFrame(columns=["team_name", "team_logo_espn"])


def dropdown_multiselect(label, options, key, fmt=str, default=None):
    """Compact dropdown: a button that opens a checklist, with a one-line summary (as in the Dowling app)."""
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
    summary = ("All" if len(selected) == len(options) else "None selected" if not selected
               else ", ".join(fmt(o) for o in selected))
    st.caption(f"{label}: {summary}")
    return selected


def _summary(selected, options, fmt=str, all_text="All"):
    if set(selected) == set(options):
        return all_text
    return ", ".join(fmt(o) for o in selected) if selected else "None"


# ---- Sidebar -------------------------------------------------------------------------------------------------
seasons_all = dt.available_seasons()
team_info = teams()
with st.sidebar:
    st.header("Filters")
    st.session_state.setdefault("f_seasons", [seasons_all[-1]])
    sel_seasons = sorted(st.multiselect("Season", seasons_all[::-1], key="f_seasons"))
if not sel_seasons:
    st.warning("Pick at least one season in the sidebar.")
    st.stop()
try:
    plays = load(tuple(sel_seasons))
except Exception as e:   # offline on first run, or a season nflverse hasn't published yet
    st.error(f"Couldn't load nflfastR data: {e}")
    st.stop()
if plays.empty:
    st.warning("No plays for that season yet.")
    st.stop()

TEAMS = sorted(plays["posteam"].dropna().unique())
WEEKS = sorted(plays["Week"].unique())
with st.sidebar:
    st.session_state.setdefault("f_team", "KC" if "KC" in TEAMS else TEAMS[0])
    team = st.selectbox("Offense to scout", TEAMS, key="f_team",
                        format_func=lambda t: f"{t} · {team_info['team_name'].get(t, '')}".rstrip(" ·"))
    st.session_state.setdefault("f_stype", "Regular season + playoffs")
    season_type = st.selectbox("Season type", ["Regular season", "Playoffs", "Regular season + playoffs"], key="f_stype")
    sel_weeks = dropdown_multiselect("Week", WEEKS, key="f_week", fmt=lambda w: f"Week {w}" if w <= 18 else
                                     {19: "Wild Card", 20: "Divisional", 21: "Conference", 22: "Super Bowl"}.get(w, f"Week {w}"))
    sel_downs = dropdown_multiselect("Down", [1, 2, 3, 4], key="f_down", fmt=lambda d: dt.DOWN_NAMES[d])
    sel_dist = dropdown_multiselect("Distance", dt.DISTANCE_ORDER, key="f_dist")
    st.session_state.setdefault("f_sit", "All plays")
    situation = st.selectbox("Situation", list(SITUATIONS), key="f_sit")
    st.session_state.setdefault("f_minplays", 1)
    cm.MIN_PLAYS = st.slider("Hide table rows with fewer than … plays", 1, 30, key="f_minplays")

    def _reset():
        for k in list(st.session_state.keys()):
            if k.startswith(("f_down_", "f_dist_", "f_week_")):
                st.session_state[k] = True
        st.session_state.update(f_sit="All plays", f_minplays=1, f_stype="Regular season + playoffs")

    st.button("Reset filters", on_click=_reset, width="stretch")
    st.caption(f"nflfastR data through {plays['game_date'].max()}")


def game_filters(d: pd.DataFrame) -> pd.DataFrame:
    """Filters that pick whole games (season type, week): red zone trips need whole drives."""
    if season_type != "Regular season + playoffs":
        d = d[d["season_type"] == ("REG" if season_type == "Regular season" else "POST")]
    if set(sel_weeks) != set(WEEKS):
        d = d[d["Week"].isin(sel_weeks)]
    return d


def play_filters(d: pd.DataFrame) -> pd.DataFrame:
    """Every sidebar filter. A filter with everything selected is skipped."""
    d = game_filters(d)
    if set(sel_downs) != {1, 2, 3, 4}:
        d = d[d["DN"].isin(sel_downs)]
    if set(sel_dist) != set(dt.DISTANCE_ORDER):
        d = d[d["Distance"].isin(sel_dist)]
    rule = SITUATIONS[situation]
    return d[rule(d)] if rule else d


league_games = game_filters(plays)
league = play_filters(plays)
team_d = league[league["posteam"] == team]
team_games = league_games[league_games["posteam"] == team]

# ---- Page ------------------------------------------------------------------------------------------------------
st.title("Tendencies")
logo = team_info["team_logo_espn"].get(team, "") if "team_logo_espn" in team_info else ""
chips = [_summary(sel_seasons, seasons_all, str), season_type, _summary(sel_weeks, WEEKS, lambda w: f"W{w}", "All weeks"),
         _summary(sel_downs, [1, 2, 3, 4], lambda d: dt.DOWN_NAMES[d], "All downs"),
         _summary(sel_dist, dt.DISTANCE_ORDER, lambda s: s.split(" (")[0], "All distances")]
if situation != "All plays":
    chips.append(situation)
if cm.MIN_PLAYS > 1:
    chips.append(f"Rows with {cm.MIN_PLAYS}+ plays")
st.html(f'<div style="display:flex;align-items:center;gap:12px;margin:-4px 0 4px">'
        f'<img src="{logo}" style="height:44px" onerror="this.style.display=\'none\'">'
        f'<div><div style="font-size:20px;font-weight:600">{team_info["team_name"].get(team, team) if len(team_info) else team} offense</div>'
        f'<div style="font-size:13px;opacity:.65">{len(team_d):,} runs and passes · NFL comparisons use '
        f'{len(league):,} plays from all 32 teams with the same filters</div></div></div>'
        '<div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-size:12px">'
        '<span style="opacity:.6;margin-right:2px">Showing</span>'
        + "".join(f'<span style="padding:3px 10px;border-radius:999px;border:0.5px solid rgba(128,128,128,.4);'
                  f'background:rgba(128,128,128,.08)">{c}</span>' for c in chips) + "</div>")
with st.expander("What do these numbers mean?"):
    st.markdown(DEFINITIONS)

if team_d.empty:
    st.info(f"No {team} plays match the current filters.")
    st.stop()

tendencies, sequencing, red_zone, run_game, pass_game = st.tabs(
    ["Tendencies", "Sequencing", "Red zone", "Run game", "Pass game"])
with tendencies:
    tab_tendencies.render(team_d, league, team)
with sequencing:
    tab_sequencing.render(team_d, league, team)
with red_zone:
    tab_red_zone.render(team_d, league, team_games, league_games, team)
with run_game:
    tab_run_game.render(team_d, league, team)
with pass_game:
    tab_pass_game.render(team_d, league, team)
