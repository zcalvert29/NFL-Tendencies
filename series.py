"""
3rd & 4th down series: judge a 3rd-down call by the whole series, not just the snap. A series is won when the 3rd-down
play gets a first down (or a touchdown), or the 3rd down comes up short, the offense goes for it on 4th, and that play
gets one. Every 3rd-down run or pass is followed to the next 4th-down run, pass, punt, or field goal in the same drive
(penalties and timeouts in between are skipped). No Streamlit code here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import data as dt
import unpredictability as up

MAX_MARGIN = 16         # same snaps as the other pages: within this score margin...
ZONES = ["Own side", "Opp 31-50", "Opp 30 and in"]
ZONE_NAMES = {"Own side": "Own side of the field", "Opp 31-50": "Opponent's 31-50", "Opp 30 and in": "Opponent's 30 and in"}
OUTCOMES = ["Converted on 3rd", "Converted on 4th", "Went for it, stopped", "Kicked", "Turned it over on 3rd"]


def third_downs(raw: pd.DataFrame, callers: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every regular-season 3rd-down run or pass (outside the last 2 minutes of a half, score within MAX_MARGIN) with
    how its series ended. `raw` = one season of nflfastR play-by-play with every play, kicks included."""
    raw = raw[raw["season_type"] == "REG"].sort_values(["game_id", "play_id"])
    snap = raw["play_type"].isin(["pass", "run"]) & (raw["qb_kneel"] != 1) & (raw["qb_spike"] != 1)
    third = raw[snap & (raw["down"] == 3) & (raw["half_seconds_remaining"] > 120)
                & (raw["score_differential"].abs() <= MAX_MARGIN)]
    fourth = raw[(raw["down"] == 4) & raw["play_type"].isin(["pass", "run", "punt", "field_goal"])]
    fourth = fourth[["game_id", "fixed_drive", "play_id", "play_type", "ydstogo", "first_down", "touchdown",
                     "interception", "fumble_lost"]].rename(columns=lambda c: "f_" + c)
    m = third.merge(fourth, left_on=["game_id", "fixed_drive"], right_on=["f_game_id", "f_fixed_drive"], how="left")
    m = (m[m["f_play_id"].isna() | (m["f_play_id"] > m["play_id"])]
         .sort_values("f_play_id").drop_duplicates(["game_id", "play_id"]))

    def offense_td(td, interception, fumble):
        return (td == 1) & (interception != 1) & (fumble != 1)

    m["conv3"] = ((m["first_down"] == 1) | offense_td(m["touchdown"], m["interception"], m["fumble_lost"])).astype(int)
    m["turnover3"] = ((m["interception"] == 1) | (m["fumble_lost"] == 1)).astype(int)
    short = (m["conv3"] == 0) & (m["turnover3"] == 0)
    m["went"] = (short & m["f_play_type"].isin(["pass", "run"])).astype(int)
    m["kicked"] = (short & ~m["f_play_type"].isin(["pass", "run"])).astype(int)   # punt, field goal, or half ended
    m["conv4"] = ((m["went"] == 1) & ((m["f_first_down"] == 1) | offense_td(
        m["f_touchdown"], m["f_interception"], m["f_fumble_lost"]))).astype(int)
    m["series"] = ((m["conv3"] == 1) | (m["conv4"] == 1)).astype(int)
    m["togo4"] = m["f_ydstogo"].where(short)
    m["Call"] = np.where((m["rush"] == 1) & (m["qb_scramble"] != 1), "Run", "Pass")   # scrambles and sacks = passes
    m["Zone"] = pd.cut(m["yardline_100"], [0, 30, 50, 100], labels=ZONES[::-1]).astype(str)
    m["Season"], m["Week"] = m["season"].astype(int), m["week"].astype(int)
    m["caller"] = up.assign_callers(m, callers if callers is not None else up.load_callers())
    return m[["Season", "Week", "game_id", "play_id", "posteam", "caller", "ydstogo", "yardline_100", "Zone", "Call", "epa",
              "conv3", "turnover3", "went", "kicked", "conv4", "series", "togo4", "desc"]].reset_index(drop=True)


def summary(g: pd.DataFrame) -> pd.Series:
    """Rates for a group of 3rd-down calls: on 3rd, the series, and what happened after a 3rd down came up short."""
    short = g[(g["conv3"] == 0) & (g["turnover3"] == 0)]
    went = g[g["went"] == 1]
    return pd.Series({
        "Plays": len(g), "Converted on 3rd": g["conv3"].mean(), "Won the series": g["series"].mean(),
        "Came up short": len(short), "Went for it": len(went),
        "Go-for-it rate": len(went) / len(short) if len(short) else np.nan,
        "Converted on 4th": went["conv4"].mean() if len(went) else np.nan,
        "Left 4th & 1-2": (short["togo4"] <= 2).mean() if len(short) else np.nan,
        "Typical 4th & ...": short["togo4"].median() if len(short) else np.nan,
        "EPA/play": g["epa"].mean()})


def outcome_shares(g: pd.DataFrame) -> pd.Series:
    """How the series ended, as shares of all the calls (they add up to 1)."""
    n = len(g)
    if n == 0:
        return pd.Series(np.nan, index=OUTCOMES)
    c4 = g["conv4"].sum()
    return pd.Series([g["conv3"].sum() / n, c4 / n, (g["went"].sum() - c4) / n, g["kicked"].sum() / n,
                      g["turnover3"].sum() / n], index=OUTCOMES)


def by(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """summary() for each group (e.g. ["Call"], ["ydstogo", "Call"])."""
    return d.groupby(cols).apply(summary).reset_index() if len(d) else pd.DataFrame()


def caller_table(d: pd.DataFrame, min_plays: int = 15) -> pd.DataFrame:
    """One row per play caller (team + caller, one season): his calls, run share, and series results."""
    rows = []
    for (season, team, caller), g in d.groupby(["Season", "posteam", "caller"]):
        if len(g) < min_plays:
            continue
        runs = g[g["Call"] == "Run"]
        rows.append({"Season": season, "Team": team, "Play caller": caller, **summary(g).to_dict(),
                     "Run share": (g["Call"] == "Run").mean(), "Runs": len(runs),
                     "Runs: won the series": runs["series"].mean() if len(runs) else np.nan})
    return pd.DataFrame(rows)


def load(seasons: list[int]) -> pd.DataFrame:
    """third_downs() for each season, from the cached play-by-play (kicks included)."""
    callers_csv = up.load_callers()
    return pd.concat([third_downs(pd.read_parquet(dt.fetch_season(s)), callers_csv) for s in seasons],
                     ignore_index=True)
