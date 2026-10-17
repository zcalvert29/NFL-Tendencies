"""
Loads nflfastR play-by-play (via the nflverse data releases) and turns it into the app's columns.

Every season is downloaded once and kept in data/ as a slimmed-down parquet file, so the app only hits GitHub
for a season it hasn't seen yet (and re-downloads the current season once it's older than REFRESH_HOURS, so new
weeks show up). FTN/NGS participation data (personnel, formation, coverage, men in the box, pressure) is joined
on when nflverse has published it for that season; for seasons without it those columns are left blank and the
formation falls back to nflfastR's shotgun / under center flag.

Run `python data.py 2024 2025 2026` to pre-download seasons from the command line.
"""
from __future__ import annotations

import io
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"
PBP_URL = RELEASES + "/pbp/play_by_play_{season}.parquet"
PARTICIPATION_URL = RELEASES + "/pbp_participation/pbp_participation_{season}.parquet"
TEAMS_URL = RELEASES + "/teams/teams_colors_logos.csv"

DATA_DIR = Path(__file__).parent / "data"
FIRST_SEASON = 2016          # participation data starts in 2016; earlier pbp works but has no personnel
REFRESH_HOURS = 12           # the season in progress is re-downloaded once its file is older than this

# nflfastR columns the app keeps (everything else in the 372-column file is dropped on download).
PBP_COLUMNS = [
    "game_id", "play_id", "season", "season_type", "week", "game_date", "home_team", "away_team",
    "posteam", "defteam", "posteam_type", "qtr", "down", "ydstogo", "yardline_100", "goal_to_go",
    "half_seconds_remaining", "game_seconds_remaining", "score_differential", "posteam_score", "defteam_score",
    "play_type", "pass", "rush", "qb_dropback", "qb_scramble", "qb_kneel", "qb_spike", "shotgun", "no_huddle",
    "pass_location", "pass_length", "air_yards", "yards_after_catch", "run_location", "run_gap",
    "yards_gained", "complete_pass", "incomplete_pass", "interception", "sack", "fumble_lost", "touchdown",
    "pass_touchdown", "rush_touchdown", "first_down", "third_down_converted", "third_down_failed",
    "fourth_down_converted", "fourth_down_failed", "penalty", "epa", "wpa", "wp", "home_wp", "vegas_wp",
    "success", "xpass", "pass_oe", "cpoe", "drive", "fixed_drive", "fixed_drive_result", "drive_play_count",
    "drive_start_yard_line", "passer_player_name", "rusher_player_name", "receiver_player_name",
    "field_goal_result", "kick_distance", "total_home_score", "total_away_score", "home_score", "away_score",
    "desc",
]
PARTICIPATION_COLUMNS = [
    "nflverse_game_id", "play_id", "offense_formation", "offense_personnel", "defense_personnel",
    "defenders_in_box", "number_of_pass_rushers", "was_pressure", "defense_man_zone_type", "defense_coverage_type",
    "time_to_throw",
]

# ---- Buckets (same names and cut points as the Dowling app) -------------------------------------------------
DISTANCE_ORDER = ["Short (1-3 yds)", "Medium (4-6 yds)", "Long (7-10 yds)", "Extra Long (11+ yds)"]
FIELD_ZONES = ["Own 1-20", "Own 21-49", "Opp 50-21", "Red zone (20-11)", "Inside the 10"]
AIR_ROWS = ["20+ yds", "10–19 yds", "0–9 yds", "Behind LOS"]
PASS_DIRECTIONS = ["Left", "Middle", "Right"]
RUN_LANES = ["Left end", "Left tackle", "Left guard", "Middle", "Right guard", "Right tackle", "Right end"]
COVERAGE_NAMES = {
    "COVER_0": "Cover 0", "COVER_1": "Cover 1", "COVER_2": "Cover 2", "2_MAN": "2 Man", "COVER_3": "Cover 3",
    "COVER_4": "Cover 4", "COVER_6": "Cover 6", "COVER_9": "Cover 9", "PREVENT": "Prevent", "BLOWN": "Blown",
    "SCREEN": "Screen", "COMBO": "Combo", "BOMB": "Bomb",
}
NEUTRAL_MARGIN = 10   # "neutral" situations: score within this many points, outside the last 2 min of a half


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "NFL-Tendencies"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _current_season() -> int:
    """NFL seasons start in September; January-August games belong to the previous season."""
    t = time.gmtime()
    return t.tm_year if t.tm_mon >= 9 else t.tm_year - 1


def _stale(path: Path, season: int) -> bool:
    if not path.exists():
        return True
    return season >= _current_season() and time.time() - path.stat().st_mtime > REFRESH_HOURS * 3600


def season_file(season: int) -> Path:
    return DATA_DIR / f"pbp_{season}.parquet"


def fetch_season(season: int, force: bool = False) -> Path:
    """Download one season's pbp (+ participation when published), slim it, and save it to data/."""
    path = season_file(season)
    if not force and not _stale(path, season):
        return path
    DATA_DIR.mkdir(exist_ok=True)
    try:
        pbp = pd.read_parquet(io.BytesIO(_download(PBP_URL.format(season=season))))
    except (urllib.error.URLError, OSError):
        if path.exists():   # offline or GitHub hiccup: keep using the copy we have
            return path
        raise
    pbp = pbp[[c for c in PBP_COLUMNS if c in pbp.columns]]
    try:
        part = pd.read_parquet(io.BytesIO(_download(PARTICIPATION_URL.format(season=season))))
        part = part[[c for c in PARTICIPATION_COLUMNS if c in part.columns]]
        part = part.rename(columns={"nflverse_game_id": "game_id"}).drop_duplicates(["game_id", "play_id"])
        pbp = pbp.merge(part, on=["game_id", "play_id"], how="left")
    except (urllib.error.URLError, OSError, ValueError):
        pass   # not published yet for this season
    for c in PARTICIPATION_COLUMNS[2:]:
        if c not in pbp.columns:
            pbp[c] = np.nan
    pbp.to_parquet(path, index=False)
    return path


def available_seasons() -> list[int]:
    return list(range(FIRST_SEASON, _current_season() + 1))


def load_teams() -> pd.DataFrame:
    """Team names, colors, and logos (cached in data/ too)."""
    path = DATA_DIR / "teams.csv"
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        path.write_bytes(_download(TEAMS_URL))
    t = pd.read_csv(path)
    # The nflverse file also lists old franchise codes (OAK, SD, STL); the app only uses current ones.
    return t[~t["team_abbr"].isin(["OAK", "SD", "STL", "LAR"])].set_index("team_abbr")


# ---- Curation ----------------------------------------------------------------------------------------------
def personnel_code(s) -> str | float:
    """'1 C, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 3 WR' -> '11' (RBs incl. FBs, then TEs). Extra OL/DL show as '+OL'."""
    if not isinstance(s, str) or not s:
        return np.nan
    counts = {}
    for part in s.split(","):
        bits = part.strip().split(" ", 1)
        if len(bits) == 2 and bits[0].isdigit():
            counts[bits[1]] = counts.get(bits[1], 0) + int(bits[0])
    rb = counts.get("RB", 0) + counts.get("FB", 0)
    te = counts.get("TE", 0)
    ol = counts.get("C", 0) + counts.get("G", 0) + counts.get("T", 0) + counts.get("OL", 0)
    code = f"{rb}{te}"
    return code + " +OL" if ol > 5 else code


def defense_package(s) -> str | float:
    """Count the DBs: 4 or fewer = Base, 5 = Nickel, 6 = Dime, 7+ = Quarter."""
    if not isinstance(s, str) or not s:
        return np.nan
    db = 0
    for part in s.split(","):
        bits = part.strip().split(" ", 1)
        if len(bits) == 2 and bits[0].isdigit() and bits[1] in ("CB", "FS", "SS", "S", "DB"):
            db += int(bits[0])
    return "Base (4 DB)" if db <= 4 else "Nickel (5 DB)" if db == 5 else "Dime (6 DB)" if db == 6 else "Quarter (7+ DB)"


def _bucket(values, edges, labels):
    return pd.cut(values, edges, labels=labels, right=True).astype(object)


def curate(pbp: pd.DataFrame) -> pd.DataFrame:
    """Add the app's columns. Keeps every play (special teams included); pages pick what they need."""
    d = pbp.copy()
    d["play_type"] = d["play_type"].astype(object)
    d = d[d["posteam"].notna()]
    # Scrambles count as passes (the call was a pass), as in nflfastR's own pass/rush flags.
    d["is_pass"] = (d["pass"] == 1)
    d["is_run"] = (d["rush"] == 1) & ~d["is_pass"]
    d["Play Type"] = np.select([d["is_pass"], d["is_run"]], ["Pass", "Run"], default=None)
    # A penalty with "no_play" still has pass/rush set by nflfastR when the call is clear from the description.
    d["scrimmage"] = d["Play Type"].notna() & (d["qb_kneel"] != 1) & (d["qb_spike"] != 1)

    d["Distance"] = _bucket(d["ydstogo"], [0, 3, 6, 10, 200], DISTANCE_ORDER)
    d["Field Zone"] = _bucket(d["yardline_100"], [0, 10, 20, 50, 80, 100], FIELD_ZONES[::-1])
    d["red_zone"] = d["yardline_100"] <= 20
    d["explosive"] = np.where(d["is_pass"], d["yards_gained"] >= 20, d["yards_gained"] >= 10).astype(float)
    d["two_minute"] = d["half_seconds_remaining"] <= 120
    d["neutral"] = (d["score_differential"].abs() <= NEUTRAL_MARGIN) & ~d["two_minute"]
    d["Quarter"] = d["qtr"].map({1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}).fillna("OT")
    d["Score"] = np.select(
        [d["score_differential"] <= -9, d["score_differential"] < 0, d["score_differential"] == 0,
         d["score_differential"] <= 8],
        ["Down 9+", "Down 1-8", "Tied", "Up 1-8"], default="Up 9+")

    # Formation: FTN formation when published, else shotgun / under center from the play description.
    ftn = d["offense_formation"].astype(object).str.title().replace({"Under Center": "Under center", "": np.nan})
    d["Formation"] = ftn.where(ftn.notna(), np.where(d["shotgun"] == 1, "Shotgun", "Under center"))
    d["Formation"] = d["Formation"].replace({"Jumbo": "Jumbo", "Wildcat": "Wildcat", "I_Form": "I-form"})
    d["Personnel"] = d["offense_personnel"].map(personnel_code)
    d["Def Package"] = d["defense_personnel"].map(defense_package)
    d["Box"] = pd.to_numeric(d["defenders_in_box"], errors="coerce").where(lambda s: s > 0)
    d["Box"] = d["Box"].map(lambda b: np.nan if pd.isna(b) else "5 or fewer" if b <= 5 else "8+" if b >= 8 else f"{int(b)}")
    rushers = pd.to_numeric(d["number_of_pass_rushers"], errors="coerce").where(lambda s: s > 0)
    d["Rushers"] = rushers.map(lambda r: np.nan if pd.isna(r) else "3 or fewer" if r <= 3 else "6+" if r >= 6 else f"{int(r)}")
    d["blitz"] = (rushers >= 5).astype(float).where(rushers.notna())
    d["pressure"] = d["was_pressure"].map({True: 1.0, False: 0.0}).where(d["is_pass"])
    d["Coverage"] = d["defense_coverage_type"].map(COVERAGE_NAMES)
    d["Man/Zone"] = d["defense_man_zone_type"].map({"MAN_COVERAGE": "Man", "ZONE_COVERAGE": "Zone"})

    # Pass zones: nflfastR's direction (left/middle/right) by air yards.
    d["Pass Direction"] = d["pass_location"].astype(object).str.title()
    d["Air Yards"] = _bucket(d["air_yards"], [-100, -0.5, 9, 19, 100], AIR_ROWS[::-1])
    d["Pass Zone"] = np.where(d["Pass Direction"].notna() & d["Air Yards"].notna(),
                              d["Air Yards"].astype(str) + " " + d["Pass Direction"].astype(str), None)
    # Run lanes: nflfastR gives a direction (left/middle/right) and, off the middle, a gap (end/tackle/guard).
    loc, gap = d["run_location"].astype(object), d["run_gap"].astype(object)
    d["Run Lane"] = np.where(loc == "middle", "Middle",
                             np.where(loc.notna() & gap.notna(), loc.str.title() + " " + gap, None))
    d["Run Direction"] = loc.str.title()
    d["completion"] = d["complete_pass"].where(d["is_pass"] & (d["sack"] != 1) & (d["qb_scramble"] != 1))
    d["turnover"] = ((d["interception"] == 1) | (d["fumble_lost"] == 1)).astype(float)
    d["Week"] = d["week"].astype(int)
    d["Season"] = d["season"].astype(int)
    d["Season Type"] = d["season_type"].map({"REG": "Regular season", "POST": "Playoffs"})
    d["Opponent"] = d["defteam"]

    # Sequencing: the run or pass right before, in the same drive (penalties and kicks skipped).
    s = d[d["scrimmage"]].sort_values(["game_id", "play_id"])
    grp = s.groupby(["game_id", "fixed_drive"], sort=False)
    prev_type, prev_success = grp["Play Type"].shift(), grp["success"].shift()
    d["Prev Play"] = prev_type.reindex(d.index)
    d["Prev Result"] = (prev_type + np.where(prev_success == 1, " that worked", " that didn't")).where(
        prev_type.notna()).reindex(d.index)
    d["Drive Snap"] = (grp.cumcount() + 1).reindex(d.index)
    return d


def load_seasons(seasons: list[int]) -> pd.DataFrame:
    frames = [curate(pd.read_parquet(fetch_season(s))) for s in seasons]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    for arg in sys.argv[1:] or [str(_current_season())]:
        p = fetch_season(int(arg), force=True)
        print(f"{arg}: {len(pd.read_parquet(p)):,} plays -> {p}")
