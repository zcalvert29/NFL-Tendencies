"""
Loads nflfastR play-by-play (from the nflverse data releases) and builds the columns the Tendencies page uses.

Each season is downloaded once and kept in data/ as a slimmed-down parquet file. The season in progress is
re-downloaded once its copy is older than REFRESH_HOURS, so new weeks show up. nflverse's FTN/NGS participation
data (formation, personnel, men in the box, coverage) is joined on when it has been published for that season; it
comes out after the season, so the season in progress only has nflfastR's shotgun / under center flag.

Run `python data.py 2025 2026` to download seasons from the command line.
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
FIRST_SEASON = 2016          # participation charting starts here
REFRESH_HOURS = 12

PBP_COLUMNS = [
    "game_id", "play_id", "season", "season_type", "week", "game_date", "home_team", "away_team",
    "posteam", "defteam", "qtr", "down", "ydstogo", "yardline_100", "half_seconds_remaining",
    "score_differential", "play_type", "pass", "rush", "qb_scramble", "qb_kneel", "qb_spike", "shotgun",
    "no_huddle", "pass_location", "air_yards", "run_location", "run_gap", "yards_gained", "complete_pass",
    "interception", "sack", "touchdown", "first_down", "fumble_lost", "epa", "success", "xpass", "pass_oe",
    "fixed_drive", "fixed_drive_result", "passer_player_name", "rusher_player_name", "receiver_player_name", "desc",
]
PARTICIPATION_COLUMNS = ["offense_formation", "offense_personnel", "defenders_in_box", "defense_coverage_type",
                         "defense_man_zone_type"]

# Same buckets as the Dowling app.
DISTANCE_ORDER = ["Short (1-3 yds)", "Medium (4-6 yds)", "Long (7-10 yds)", "Extra Long (11+ yds)"]
DOWN_NAMES = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
NEUTRAL_MARGIN = 7           # score within this many points, and not the last 2 minutes of a half
COVERAGE_NAMES = {"COVER_0": "Cover 0", "COVER_1": "Cover 1", "COVER_2": "Cover 2", "2_MAN": "2 Man",
                  "COVER_3": "Cover 3", "COVER_4": "Cover 4", "COVER_6": "Cover 6", "COVER_9": "Cover 9",
                  "PREVENT": "Prevent"}
# nflfastR run location + gap -> lane, left to right.
RUN_LANES = ["Left end", "Left tackle", "Left guard", "Middle", "Right guard", "Right tackle", "Right end"]
PASS_DIRECTIONS = ["Left", "Middle", "Right"]
DEPTHS = ["20+ yds", "10–19 yds", "0–9 yds", "Behind LOS"]    # top to bottom, as drawn


# ---- Download ----------------------------------------------------------------------------------------------
def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "NFL-Tendencies"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def current_season() -> int:
    t = time.gmtime()
    return t.tm_year if t.tm_mon >= 9 else t.tm_year - 1


def available_seasons() -> list[int]:
    return list(range(FIRST_SEASON, current_season() + 1))


def fetch_season(season: int, force: bool = False) -> Path:
    """Download one season (+ participation when published), keep only the columns used, save to data/."""
    path = DATA_DIR / f"pbp_{season}.parquet"
    fresh = path.exists() and (season < current_season() or time.time() - path.stat().st_mtime < REFRESH_HOURS * 3600)
    if fresh and not force:
        return path
    DATA_DIR.mkdir(exist_ok=True)
    try:
        pbp = pd.read_parquet(io.BytesIO(_download(PBP_URL.format(season=season))))
    except (urllib.error.URLError, OSError):
        if path.exists():   # offline: keep using the copy we have
            return path
        raise
    pbp = pbp[[c for c in PBP_COLUMNS if c in pbp.columns]]
    try:
        part = pd.read_parquet(io.BytesIO(_download(PARTICIPATION_URL.format(season=season))))
        part = part.rename(columns={"nflverse_game_id": "game_id"})[["game_id", "play_id"] + PARTICIPATION_COLUMNS]
        pbp = pbp.merge(part.drop_duplicates(["game_id", "play_id"]), on=["game_id", "play_id"], how="left")
    except (urllib.error.URLError, OSError, KeyError, ValueError):
        pass   # not published yet
    for c in PARTICIPATION_COLUMNS:
        if c not in pbp.columns:
            pbp[c] = np.nan
    pbp.to_parquet(path, index=False)
    return path


LOGO_PX = 64     # logos are cached at this size: plenty for chart markers, small enough to embed


def team_logos(teams: pd.DataFrame) -> dict[str, str]:
    """abbr -> logo as a data URI (PNG, LOGO_PX square), downloaded once from ESPN's CDN into data/logos/.

    Charts embed the image itself instead of linking to ESPN: Vega draws on a canvas, which needs CORS headers on
    linked images, so a hotlinked logo can silently fail to show. A logo that can't be downloaded is left out (the
    chart still shows the team's colored dot)."""
    import base64
    from PIL import Image

    folder = DATA_DIR / "logos"
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for abbr, url in teams.get("team_logo_espn", pd.Series(dtype=str)).dropna().items():
        path = folder / f"{abbr}.png"
        if not path.exists():
            try:
                img = Image.open(io.BytesIO(_download(url))).convert("RGBA")
                img.thumbnail((LOGO_PX, LOGO_PX))
                img.save(path, optimize=True)
            except (urllib.error.URLError, OSError, ValueError):
                continue
        out[abbr] = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    return out


def load_teams() -> pd.DataFrame:
    path = DATA_DIR / "teams.csv"
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        path.write_bytes(_download(TEAMS_URL))
    return pd.read_csv(path).set_index("team_abbr")


# ---- Curation ----------------------------------------------------------------------------------------------
def personnel_code(s) -> str | float:
    """'1 C, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 3 WR' -> '11' (RBs incl. FBs, then TEs). An extra lineman adds ' +OL'."""
    if not isinstance(s, str) or not s:
        return np.nan
    counts: dict[str, int] = {}
    for part in s.split(","):
        bits = part.strip().split(" ", 1)
        if len(bits) == 2 and bits[0].isdigit():
            counts[bits[1]] = counts.get(bits[1], 0) + int(bits[0])
    rb = counts.get("RB", 0) + counts.get("FB", 0)
    ol = sum(counts.get(p, 0) for p in ("C", "G", "T", "OL"))
    code = f"{rb}{counts.get('TE', 0)}"
    return code + " +OL" if ol > 5 else code


def _cut(values, edges, labels):
    return pd.cut(values, edges, labels=labels).astype(object)


def curate(pbp: pd.DataFrame) -> pd.DataFrame:
    """Every run and pass (scrimmage snaps only: kneels, spikes, and kicks dropped), with the app's columns."""
    d = pbp[pbp["posteam"].notna() & ((pbp["pass"] == 1) | (pbp["rush"] == 1))
            & (pbp["qb_kneel"] != 1) & (pbp["qb_spike"] != 1)].copy()
    # Scrambles and sacks count as passes: the call was a pass (nflfastR's pass flag).
    d["PASS"] = (d["pass"] == 1).astype(int)
    d["PLAY TYPE"] = np.where(d["PASS"] == 1, "Pass", "Run")
    d["DN"] = d["down"]
    d["Distance"] = pd.Categorical(_cut(d["ydstogo"], [0, 3, 6, 10, 200], DISTANCE_ORDER), DISTANCE_ORDER, ordered=True)
    d["explosive"] = np.where(d["PASS"] == 1, d["yards_gained"] >= 20, d["yards_gained"] >= 10).astype(float)
    d["completion"] = d["complete_pass"].where((d["PASS"] == 1) & (d["sack"] != 1) & (d["qb_scramble"] != 1))
    d["neutral"] = (d["score_differential"].abs() <= NEUTRAL_MARGIN) & (d["half_seconds_remaining"] > 120)
    d["Season"] = d["season"].astype(int)
    d["Week"] = d["week"].astype(int)

    # Formation: FTN charting when published, else shotgun / under center from the play description.
    ftn = d["offense_formation"].astype(object).replace({"": np.nan}).str.replace("_", " ").str.title()
    ftn = ftn.replace({"Under Center": "Under center", "I Form": "I-form"})
    d["Formation"] = ftn.where(ftn.notna(), np.where(d["shotgun"] == 1, "Shotgun", "Under center"))
    d["Personnel"] = d["offense_personnel"].map(personnel_code)
    box = pd.to_numeric(d["defenders_in_box"], errors="coerce").where(lambda b: b > 0)
    d["Box"] = box.map(lambda b: np.nan if pd.isna(b) else "5 or fewer" if b <= 5 else "8+" if b >= 8 else str(int(b)))
    d["Coverage"] = d["defense_coverage_type"].map(COVERAGE_NAMES)
    d["Man/Zone"] = d["defense_man_zone_type"].map({"MAN_COVERAGE": "Man", "ZONE_COVERAGE": "Zone"})

    loc, gap = d["run_location"].astype(object), d["run_gap"].astype(object)
    d["Run Lane"] = np.where(loc == "middle", "Middle",
                             np.where(loc.notna() & gap.notna(), loc.str.title() + " " + gap, None))
    d.loc[d["PASS"] == 1, "Run Lane"] = None
    d["Pass Direction"] = d["pass_location"].astype(object).str.title().where(d["PASS"] == 1)
    d["Depth"] = _cut(d["air_yards"], [-100, -1, 9, 19, 100], DEPTHS[::-1])

    # Sequencing: the snap before, in the same drive (built here so the sidebar filters can't hide it).
    d = d.sort_values(["game_id", "play_id"])
    g = d.groupby(["game_id", "fixed_drive"], sort=False)
    d["prev_type"] = g["PLAY TYPE"].shift()
    d["prev_gain"] = g["yards_gained"].shift()
    d["prev_complete"] = g["completion"].shift()
    d["prev_success"] = g["success"].shift()
    d["first_of_drive"] = g.cumcount() == 0
    return d


def load_seasons(seasons: list[int]) -> pd.DataFrame:
    frames = [curate(pd.read_parquet(fetch_season(s))) for s in seasons]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    for arg in sys.argv[1:] or [str(current_season())]:
        p = fetch_season(int(arg), force=True)
        print(f"{arg}: {len(pd.read_parquet(p)):,} plays -> {p}")
