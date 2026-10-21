"""
Play-calling identity: how far each offense's play calling is from what an average NFL team would do in the same
spots, measured three ways, and which teams are unorthodox and which are NFL average.

Every measure compares each snap with the league in that exact spot, then averages over the team's snaps:

  * Lean (PROE): pass (1/0) minus the chance an average team throws here: nflfastR's xpass (down, distance, field
    position, score, time, timeouts), re-centered on what the rest of the league actually did in the same cell
    that season.
    xpass was fit on earlier seasons and drifts: the league now runs more on 3rd/4th & 1-2 than it expects (league
    short-yardage PROE is -2% to -8% a season), which would make every team look run-heavy there.
  * Shotgun over expected: shotgun (1/0) minus the rest of the league's shotgun rate in the same cell (down ×
    distance × field zone × score state, same season; the team itself is left out of its own baseline).
  * Spread personnel over expected: 3+ WR on the field (RBs + TEs ≤ 2, no extra lineman) minus the league's rate in
    the same cell. Only for seasons with FTN participation charting (not the season in progress).

Each measure gets:
  * a z-score with game-clustered standard errors (snaps in the same game aren't independent: a game plan repeats),
    so |z| < 2 means "not detectably different from an average team";
  * an empirical-Bayes shrunk value: noisy measures from fewer snaps are pulled toward zero by how much of the
    spread between teams is real (between-team variance minus the average sampling variance);
  * a standardized value: the shrunk value in units of the real between-team spread that season.

Unorthodoxy index = root mean square of the standardized values over the measures available that season: 0 = plays
it exactly like an average NFL team, 1 = typically one team-to-team spread away from average on each measure.

Situational splits (early downs, 1st & 10, 2nd & long, short yardage, red zone) are each team's PROE there minus the
league's PROE there, listed for exploration but kept out of the index: they're noisier (see reliability()).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_MARGIN = 16                 # snaps with the score within this many points...
MIN_HALF_SECONDS = 120          # ...and not in the last 2 minutes of a half (game script takes over there)
Z_DETECT = 2.0                  # |z| at or above this = detectably different from an average team...
MATERIAL = 0.5                  # ...and at least this many between-team SDs away = "different" on that measure
UNORTHODOX = 1.0                # index at or above this = unorthodox

MEASURES = {   # key -> (label, per-snap value column, expected column)
    "lean": ("Lean (PROE)", "PASS", "exp_pass"),
    "shotgun": ("Shotgun over expected", "shotgun", "exp_shotgun"),
    "spread": ("3+ WR personnel over expected", "spread", "exp_spread"),
}
SPLITS = {   # situational PROE vs the league in the same situation, for exploration
    "Early-down PROE": lambda d: d["DN"].isin([1, 2]),
    "1st & 10 PROE": lambda d: (d["DN"] == 1) & (d["ydstogo"] == 10),
    "2nd & long PROE": lambda d: (d["DN"] == 2) & (d["ydstogo"] >= 7),
    "Short-yardage PROE": lambda d: d["DN"].isin([3, 4]) & (d["ydstogo"] <= 2),
    "Red zone PROE": lambda d: d["yardline_100"] <= 20,
}


def personnel_counts(code: str) -> tuple[float, float, float]:
    """'11' -> (RBs 1, TEs 1, extra OL 0); '12 +OL' -> (1, 2, 1)."""
    if not isinstance(code, str) or len(code) < 2 or not code[:2].isdigit():
        return np.nan, np.nan, np.nan
    return float(code[0]), float(code[1]), float("+OL" in code)


def _rest_of_league_mean(d: pd.DataFrame, values: pd.Series) -> pd.Series:
    """For each snap: the mean of `values` over every other team's snaps in the same season and cell."""
    v = values.astype(float)
    ok = v.notna()
    keys = [d["Season"], d["cell"]]
    cell_sum = v.where(ok, 0).groupby(keys).transform("sum")
    cell_n = ok.astype(int).groupby(keys).transform("sum")
    team_sum = v.where(ok, 0).groupby(keys + [d["posteam"]]).transform("sum")
    team_n = ok.astype(int).groupby(keys + [d["posteam"]]).transform("sum")
    rest_n = cell_n - team_n
    return ((cell_sum - team_sum) / rest_n.where(rest_n > 0)).where(ok)


def sample(plays: pd.DataFrame) -> pd.DataFrame:
    """Regular-season snaps where the call is a real choice, with the cell and per-snap measure columns."""
    d = plays[(plays["season_type"] == "REG") & plays["xpass"].notna() & plays["DN"].notna()
              & (plays["half_seconds_remaining"] > MIN_HALF_SECONDS)
              & (plays["score_differential"].abs() <= MAX_MARGIN)].copy()
    fz = pd.cut(d["yardline_100"], [0, 20, 60, 100], labels=["red zone", "midfield", "own side"]).astype(str)
    score = np.select([d["score_differential"] < -3, d["score_differential"] > 3], ["behind", "ahead"], "close")
    d["cell"] = d["DN"].astype(int).astype(str) + "|" + d["Distance"].astype(str) + "|" + fz + "|" + score
    d["shotgun"] = d["shotgun"].astype(float)
    counts = d["Personnel"].map(personnel_counts)
    rb, te, extra_ol = counts.str[0], counts.str[1], counts.str[2]
    d["spread"] = ((rb + te <= 2) & (extra_ol == 0)).astype(float).where(rb.notna())
    d["two_te"] = (te >= 2).astype(float).where(rb.notna())
    d["two_rb"] = (rb >= 2).astype(float).where(rb.notna())
    # Expected = the rest of the league in the same cell, same season (leave-one-out, so a team with an extreme
    # style doesn't drag its own baseline toward itself).
    for col in ("shotgun", "spread", "two_te", "two_rb"):
        d[f"exp_{col}"] = _rest_of_league_mean(d, d[col])
    d["exp_pass"] = (d["xpass"] + _rest_of_league_mean(d, d["PASS"] - d["xpass"])).clip(0.01, 0.99)
    return d


def _measure(g: pd.DataFrame, val: str, exp: str) -> tuple[float, float, float, int]:
    """(effect, z, standard error, snaps) for one team-season and one measure. The standard error is clustered by
    game: the spread of each game's total difference around what the season average predicts for that game."""
    m = g[["game_id", val, exp]].dropna()
    n = len(m)
    if n == 0 or m["game_id"].nunique() < 3:
        return np.nan, np.nan, np.nan, n
    m = m.assign(diff=m[val] - m[exp])
    effect = m["diff"].mean()
    per_game = m.groupby("game_id")["diff"].agg(["sum", "size"])
    k = len(per_game)
    var = ((per_game["sum"] - per_game["size"] * effect) ** 2).sum() * k / (k - 1)
    se = np.sqrt(var) / n
    return effect, effect / se if se > 0 else np.nan, se, n


def _tvd(g: pd.DataFrame, league_shares: pd.DataFrame) -> float:
    """Personnel mix distance: share of snaps that would have to change personnel group to match what the league
    uses in the same cells (total variation distance, 0-100%)."""
    g = g[g["Personnel"].notna()]
    if g.empty:
        return np.nan
    observed = g["Personnel"].value_counts(normalize=True)
    expected = league_shares.reindex(g["cell"]).mean()
    both = pd.concat([observed, expected], axis=1).fillna(0)
    return 0.5 * (both.iloc[:, 0] - both.iloc[:, 1]).abs().sum()


def _shrink(effect: pd.Series, se: pd.Series) -> tuple[pd.Series, pd.Series, float]:
    """Empirical Bayes: (shrunk effect, its standardized value, between-team true SD)."""
    ok = effect.notna() & se.notna()
    if ok.sum() < 3:
        return effect * np.nan, effect * np.nan, np.nan
    mean = effect[ok].mean()
    true_var = max(effect[ok].var() - (se[ok] ** 2).mean(), 1e-9)
    rel = true_var / (true_var + se ** 2)
    shrunk = mean + rel * (effect - mean)
    return shrunk, shrunk / np.sqrt(true_var), float(np.sqrt(true_var))


def season_table(plays: pd.DataFrame, season: int) -> pd.DataFrame:
    """One row per team for one season: every measure, situational split, and the index. Columns for measures the
    season has no data for (personnel before charting is published) are left out."""
    d = sample(plays[plays["Season"] == season])
    if d.empty:
        return pd.DataFrame()
    has_personnel = d["Personnel"].notna().mean() > 0.5
    keys = [k for k in MEASURES if k != "spread" or has_personnel]
    rows = {}
    shares = (d[d["Personnel"].notna()].groupby("cell")["Personnel"].value_counts(normalize=True).unstack(fill_value=0)
              if has_personnel else None)
    league_split = {name: d.loc[rule(d), "pass_oe"].mean() for name, rule in SPLITS.items()}
    for team, g in d.groupby("posteam"):
        r = {"Snaps": len(g), "Pass rate": g["PASS"].mean(), "Expected pass rate": g["exp_pass"].mean()}
        for k in keys:
            label, val, exp = MEASURES[k]
            r[f"{k}_effect"], r[f"{k}_z"], r[f"{k}_se"], _ = _measure(g, val, exp)
        r["Shotgun rate"], r["Expected shotgun rate"] = g["shotgun"].mean(), g["exp_shotgun"].mean()
        if has_personnel:
            r["3+ WR rate"], r["Expected 3+ WR rate"] = g["spread"].mean(), g["exp_spread"].mean()
            for col, name in (("two_te", "2+ TE"), ("two_rb", "2+ RB")):
                r[f"{name} rate"] = g[col].mean()
                r[f"{name} over expected"] = (g[col] - g[f"exp_{col}"]).mean()
            r["Personnel mix distance"] = _tvd(g, shares)
            top = g["Personnel"].value_counts(normalize=True)
            r["Top personnel"] = f"{top.index[0]} ({top.iat[0]:.0%})" if len(top) else ""
        for name, rule in SPLITS.items():
            s = g[rule(g)]
            r[name] = (s["pass_oe"].mean() - league_split[name]) / 100 if len(s) else np.nan
        rows[team] = r
    t = pd.DataFrame.from_dict(rows, orient="index")
    std_cols = []
    for k in keys:
        t[f"{k}_shrunk"], t[f"{k}_std"], _ = _shrink(t[f"{k}_effect"].astype(float), t[f"{k}_se"].astype(float))
        std_cols.append(f"{k}_std")
    t["Unorthodoxy index"] = np.sqrt((t[std_cols].astype(float) ** 2).mean(axis=1))
    different = pd.DataFrame({k: (t[f"{k}_z"].astype(float).abs() >= Z_DETECT)
                              & (t[f"{k}_std"].astype(float).abs() >= MATERIAL) for k in keys})
    t["Different on"] = different.apply(
        lambda row: ", ".join(_direction(k, t.at[row.name, f"{k}_effect"]) for k, v in row.items() if v) or "nothing",
        axis=1)
    t["Rank"] = t["Unorthodoxy index"].rank(ascending=False, method="min").astype(int)
    t["Profile"] = np.select([t["Unorthodoxy index"] >= UNORTHODOX, ~different.any(axis=1)],
                             ["Unorthodox", "NFL average"], "Distinct")
    t.attrs["measures"] = keys
    return t.sort_values("Unorthodoxy index", ascending=False)


def _direction(key: str, effect: float) -> str:
    up = effect > 0
    return {"lean": "pass-heavy" if up else "run-heavy", "shotgun": "more shotgun" if up else "more under center",
            "spread": "more 3+ WR" if up else "heavier personnel"}[key]


def reliability(plays: pd.DataFrame, season: int) -> pd.Series:
    """Split-half reliability of each measure and split (odd vs even weeks, Spearman-Brown to a full season):
    how much of the team-to-team spread is a real trait rather than noise. 1 = all trait, 0 = all noise."""
    d = sample(plays[plays["Season"] == season])
    out = {}
    halves = [d[d["week"] % 2 == p] for p in (0, 1)]
    for k, (label, val, exp) in MEASURES.items():
        if d[val].notna().mean() < 0.5:
            continue
        a, b = [(h[val] - h[exp]).groupby(h["posteam"]).mean() for h in halves]
        out[label] = a.corr(b)
    for name, rule in SPLITS.items():
        a, b = [h[rule(h)].groupby("posteam")["pass_oe"].mean() for h in halves]
        out[name] = a.corr(b)
    r = pd.Series(out)
    return (2 * r / (1 + r)).clip(lower=0)
