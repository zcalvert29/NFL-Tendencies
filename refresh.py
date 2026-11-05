"""
Weekly refresh (run by .github/workflows/refresh.yml every Tuesday at 5:30 AM Central, after Monday Night Football):
download the latest play-by-play for the current season, then precompute the slow whole-season tables (the film
index and the balance score for every season since 1999, each with its outcomes, all snaps and neutral situations
only) into precomputed/, which is committed so the app's outcome-trend and coach-history sections open instantly.

    python refresh.py              # every season (download what's missing, re-download the current season)
    python refresh.py 2024 2025    # just these seasons (the rest of precomputed/ is kept)
"""
from __future__ import annotations

import json
import sys
import time

import pandas as pd

import data as dt
import unpredictability as up


def season_tables(season: int) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    dt.fetch_season(season, force=season == dt.current_season())
    plays = dt.load_seasons([season])
    games = plays["game_id"].nunique()
    d, clues = up.score(plays, season, None, by="caller")
    film, balance = [], []
    for neutral in (False, True):
        film.append(up.film_season_table(d, clues, season, neutral).assign(Neutral=neutral, _season_games=games))
        balance.append(up.balance_season_table(plays, season, neutral).assign(Neutral=neutral, _season_games=games))
    return film, balance


def main(seasons: list[int]) -> None:
    film, balance = [], []
    for season in seasons:
        t = time.time()
        f, b = season_tables(season)
        film += f
        balance += b
        print(f"{season}: {len(f[0])} play callers (film), {len(b[0])} (balance) in {time.time() - t:.0f}s", flush=True)
    dt.PRECOMPUTED_DIR.mkdir(exist_ok=True)
    for kind, frames in (("film", film), ("balance", balance)):
        new = pd.concat(frames, ignore_index=True)
        old = dt.load_precomputed(kind)
        if len(old):
            new = pd.concat([old[~old["Season"].isin(seasons)], new], ignore_index=True)
        new.sort_values(["Season", "Neutral", "Team"]).to_parquet(dt.PRECOMPUTED_DIR / f"{kind}.parquet", index=False)
    current = dt.current_season()
    weeks = dt.load_seasons([current])["Week"].max() if current in seasons else None
    meta = {"refreshed_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime()), "current_season": current,
            "current_season_through_week": None if weeks is None or pd.isna(weeks) else int(weeks)}
    (dt.PRECOMPUTED_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(meta)


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or dt.available_seasons())
