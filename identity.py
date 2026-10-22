"""
Play-calling identity: how far each offense's play calling is from what an average NFL team would do in the same
spots, and which teams are unorthodox and which are NFL average.

Every measure compares each snap with the rest of the league in that exact spot (the team is left out of its own
baseline, so an extreme team can't drag "average" toward itself), then averages over the team's snaps:

  * Lean (PROE): pass (1/0) minus the chance an average team throws here: nflfastR's xpass (down, distance, field
    position, score, time, timeouts), re-centered on what the rest of the league actually did in the same cell
    that season. xpass was fit on earlier seasons and drifts: the league now runs more on 3rd/4th & 1-2 than it
    expects (league short-yardage PROE is -2% to -8% a season), which would make every team look run-heavy there.
  * Shotgun over expected: shotgun (1/0) minus the rest of the league's shotgun rate in the same cell (down ×
    distance × field zone × score state, same season).
  * 3+ WR personnel over expected: 3+ WR on the field (RBs + TEs ≤ 2, no extra lineman) minus the rest of the
    league's rate in the same cell. Only for seasons with FTN participation charting (not the season in progress).
  * Situational play calling: for each situation in SITUATIONS (3rd & short, red zone, after a run, first 15
    snaps...), how much more or less the team throws there than the rest of the league does there, *beyond its
    overall lean*. A pass-heavy team that's pass-heavy everywhere scores 0 here; one that's run-heavy overall but
    throws on 3rd & 1 scores high. Each situation gets its own z-score and shrinkage; the component is the root mean
    square of the situations' standardized values.

Each measure gets:
  * a z-score with game-clustered standard errors (snaps in the same game aren't independent: a game plan repeats),
    so |z| < 2 means "not detectably different from an average team". Situations are many tests at once, so a
    situation needs a Bonferroni-adjusted z (about 3.0 for 13 situations);
  * an empirical-Bayes shrunk value: noisy values from fewer snaps are pulled toward the league by how much of the
    spread between teams is real (between-team variance minus the average sampling variance);
  * a standardized value: the shrunk value's distance from the league, in units of the real between-team spread.

Unorthodoxy index = root mean square of the standardized values over the measures used: 0 = plays it exactly like an
average NFL team, 1 = typically one team-to-team spread away from average on each measure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd

Z_DETECT = 2.0                  # |z| at or above this = detectably different from an average team...
MATERIAL = 0.5                  # ...and at least this many between-team SDs away = "different" on that measure
UNORTHODOX = 1.0                # index at or above this = unorthodox
FIRST_SNAPS = 15                # "scripted" = a team's first this-many snaps of the game

MEASURES = {   # key -> (label, per-snap value column, expected column)
    "lean": ("Lean (PROE)", "PASS", "exp_pass"),
    "shotgun": ("Shotgun over expected", "shotgun", "exp_shotgun"),
    "spread": ("3+ WR personnel over expected", "spread", "exp_spread"),
}
SITUATIONAL = "situational"
COMPONENTS = {**{k: v[0] for k, v in MEASURES.items()}, SITUATIONAL: "Situational play calling"}

SITUATIONS = {   # name -> rule; each team's pass rate there vs the rest of the league there, beyond its lean
    "1st & 10": lambda d: (d["DN"] == 1) & (d["ydstogo"] == 10),
    "2nd & short (1-3)": lambda d: (d["DN"] == 2) & (d["ydstogo"] <= 3),
    "2nd & long (7+)": lambda d: (d["DN"] == 2) & (d["ydstogo"] >= 7),
    "3rd/4th & 1-2": lambda d: d["DN"].isin([3, 4]) & (d["ydstogo"] <= 2),
    "3rd & medium (3-6)": lambda d: (d["DN"] == 3) & d["ydstogo"].between(3, 6),
    "Red zone": lambda d: d["yardline_100"] <= 20,
    "Goal to go": lambda d: d["ydstogo"] >= d["yardline_100"],
    "Backed up (own 10 and in)": lambda d: d["yardline_100"] >= 90,
    "Leading by 4+": lambda d: d["score_differential"] >= 4,
    "Trailing by 4+": lambda d: d["score_differential"] <= -4,
    "After a run": lambda d: d["prev_type"] == "Run",
    "After a pass": lambda d: d["prev_type"] == "Pass",
    f"First {FIRST_SNAPS} snaps": lambda d: d["game_snap"] <= FIRST_SNAPS,
}
PHRASES = {   # "throws more ..." / "runs more ..."
    "1st & 10": "on 1st & 10", "2nd & short (1-3)": "on 2nd & short", "2nd & long (7+)": "on 2nd & long",
    "3rd/4th & 1-2": "on 3rd/4th & 1-2", "3rd & medium (3-6)": "on 3rd & medium", "Red zone": "in the red zone",
    "Goal to go": "on goal to go", "Backed up (own 10 and in)": "when backed up", "Leading by 4+": "when leading",
    "Trailing by 4+": "when trailing", "After a run": "after a run", "After a pass": "after a pass",
    f"First {FIRST_SNAPS} snaps": f"in the first {FIRST_SNAPS} snaps",
}


@dataclass(frozen=True)
class Settings:
    """Which snaps count and what goes in the index. Frozen so it can key a cache."""
    weeks: tuple[int, int] | None = None         # regular-season week range, None = all
    max_margin: int = 16                         # score within this many points...
    drop_two_minute: bool = True                 # ...and not in the last 2 minutes of a half
    components: tuple[str, ...] = ("lean", "shotgun", "spread", SITUATIONAL)
    situations: tuple[str, ...] = field(default_factory=lambda: tuple(SITUATIONS))


def z_situation(n_situations: int) -> float:
    """Bonferroni: the |z| a single situation needs so that, over all of them, a fluke flag stays near 5%."""
    return NormalDist().inv_cdf(1 - 0.025 / max(n_situations, 1))


def personnel_counts(code: str) -> tuple[float, float, float]:
    """'11' -> (RBs 1, TEs 1, extra OL 0); '12 +OL' -> (1, 2, 1)."""
    if not isinstance(code, str) or len(code) < 2 or not code[:2].isdigit():
        return np.nan, np.nan, np.nan
    return float(code[0]), float(code[1]), float("+OL" in code)


def _rest_of_league_mean(d: pd.DataFrame, values: pd.Series, by: list[pd.Series] | None = None) -> pd.Series:
    """For each snap: the mean of `values` over every other team's snaps in the same group (default: season and
    cell)."""
    v = values.astype(float)
    ok = v.notna()
    keys = by if by is not None else [d["Season"], d["cell"]]
    cell_sum = v.where(ok, 0).groupby(keys).transform("sum")
    cell_n = ok.astype(int).groupby(keys).transform("sum")
    team_sum = v.where(ok, 0).groupby(keys + [d["posteam"]]).transform("sum")
    team_n = ok.astype(int).groupby(keys + [d["posteam"]]).transform("sum")
    rest_n = cell_n - team_n
    return ((cell_sum - team_sum) / rest_n.where(rest_n > 0)).where(ok)


def sample(plays: pd.DataFrame, settings: Settings = Settings()) -> pd.DataFrame:
    """Regular-season snaps where the call is a real choice, with the cell and per-snap measure columns."""
    d = plays[(plays["season_type"] == "REG") & plays["xpass"].notna() & plays["DN"].notna()].copy()
    # A team's snap number in the game, counted before any filtering, so "first 15 snaps" means the opening script.
    d["game_snap"] = d.sort_values(["game_id", "play_id"]).groupby(["game_id", "posteam"]).cumcount() + 1
    keep = d["score_differential"].abs() <= settings.max_margin
    if settings.drop_two_minute:
        keep &= d["half_seconds_remaining"] > 120
    if settings.weeks is not None:
        keep &= d["week"].between(*settings.weeks)
    d = d[keep]
    fz = pd.cut(d["yardline_100"], [0, 20, 60, 100], labels=["red zone", "midfield", "own side"]).astype(str)
    score = np.select([d["score_differential"] < -3, d["score_differential"] > 3], ["behind", "ahead"], "close")
    d["cell"] = d["DN"].astype(int).astype(str) + "|" + d["Distance"].astype(str) + "|" + fz + "|" + score
    d["shotgun"] = d["shotgun"].astype(float)
    counts = d["Personnel"].map(personnel_counts)
    rb, te, extra_ol = counts.str[0], counts.str[1], counts.str[2]
    d["spread"] = ((rb + te <= 2) & (extra_ol == 0)).astype(float).where(rb.notna())
    d["two_te"] = (te >= 2).astype(float).where(rb.notna())
    d["two_rb"] = (rb >= 2).astype(float).where(rb.notna())
    for col in ("shotgun", "spread", "two_te", "two_rb"):
        d[f"exp_{col}"] = _rest_of_league_mean(d, d[col])
    d["exp_pass"] = (d["xpass"] + _rest_of_league_mean(d, d["PASS"] - d["xpass"])).clip(0.01, 0.99)
    d["pass_resid"] = d["PASS"] - d["exp_pass"]
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
    """Empirical Bayes: (shrunk effect, standardized distance from the league, between-team true SD). When almost
    none of the spread between teams is real, everyone shrinks to the league and the standardized values go to 0."""
    ok = effect.notna() & se.notna()
    if ok.sum() < 3:
        return effect * np.nan, effect * np.nan, np.nan
    mean = effect[ok].mean()
    true_var = max(effect[ok].var() - (se[ok] ** 2).mean(), 1e-9)
    rel = true_var / (true_var + se ** 2)
    shrunk = mean + rel * (effect - mean)
    return shrunk, (shrunk - mean) / np.sqrt(true_var), float(np.sqrt(true_var))


def _direction(key: str, effect: float) -> str:
    up = effect > 0
    return {"lean": "pass-heavy" if up else "run-heavy", "shotgun": "more shotgun" if up else "more under center",
            "spread": "more 3+ WR" if up else "heavier personnel"}[key]


def season_table(plays: pd.DataFrame, season: int, settings: Settings = Settings()) -> pd.DataFrame:
    """One row per team for one season: every measure, situation, and the index. Measures the season has no data
    for (personnel before charting is published) are left out of the index and their columns dropped."""
    d = sample(plays[plays["Season"] == season], settings)
    if d.empty or d["posteam"].nunique() < 3:
        return pd.DataFrame()
    has_personnel = d["Personnel"].notna().mean() > 0.5
    keys = [k for k in MEASURES if k in settings.components and (k != "spread" or has_personnel)]
    situations = [s for s in settings.situations if s in SITUATIONS] if SITUATIONAL in settings.components else []
    shares = (d[d["Personnel"].notna()].groupby("cell")["Personnel"].value_counts(normalize=True).unstack(fill_value=0)
              if has_personnel else None)
    # Situational residuals: each snap's pass residual minus the rest of the league's in the same situation.
    sit_cols = {}
    for i, name in enumerate(situations):
        inside = SITUATIONS[name](d).fillna(False).astype(bool)
        col = f"sit{i}"
        resid = d["pass_resid"].where(inside)
        d[col] = resid - _rest_of_league_mean(d, resid, by=[d["Season"]])
        sit_cols[name] = col
    rows = {}
    for team, g in d.groupby("posteam"):
        lean_all = g["pass_resid"].mean()
        r = {"Snaps": len(g), "Games": g["game_id"].nunique(), "EPA/play": g["epa"].mean(),
             "Success rate": g["success"].mean(), "Pass rate": g["PASS"].mean(),
             "Expected pass rate": g["exp_pass"].mean()}
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
        for name, col in sit_cols.items():
            # "vs NFL" (raw, for the table) and the part beyond the team's own lean (for the index).
            r[f"{name}|vs_nfl"], r[f"{name}|z_nfl"] = _measure(g.assign(_zero=0.0), col, "_zero")[:2]
            r[f"{name}|n"] = int(g[col].notna().sum())
            r[f"{name}|effect"], r[f"{name}|z"], r[f"{name}|se"], _ = _measure(
                g.assign(_lean=lean_all), col, "_lean")
        rows[team] = r
    t = pd.DataFrame.from_dict(rows, orient="index")

    std_cols = []
    for k in keys:
        t[f"{k}_shrunk"], t[f"{k}_std"], _ = _shrink(t[f"{k}_effect"], t[f"{k}_se"])
        std_cols.append(f"{k}_std")
    different = pd.DataFrame({k: (t[f"{k}_z"].abs() >= Z_DETECT) & (t[f"{k}_std"].abs() >= MATERIAL) for k in keys},
                             index=t.index)
    labels = {k: [] for k in t.index}
    for k in keys:
        for team in t.index[different[k]]:
            labels[team].append(_direction(k, t.at[team, f"{k}_effect"]))
    if situations:
        z_sit = z_situation(len(situations))
        sit_std = []
        flags = pd.Series(False, index=t.index)
        for name in situations:
            _, t[f"{name}|std"], _ = _shrink(t[f"{name}|effect"], t[f"{name}|se"])
            sit_std.append(f"{name}|std")
            hit = (t[f"{name}|z"].abs() >= z_sit) & (t[f"{name}|std"].abs() >= MATERIAL)
            flags |= hit.fillna(False)
            for team in t.index[hit.fillna(False)]:
                more = "throws" if t.at[team, f"{name}|effect"] > 0 else "runs"
                labels[team].append(f"{more} more {PHRASES[name]}")
        t["situational_std"] = np.sqrt((t[sit_std] ** 2).mean(axis=1))
        biggest = t[sit_std].abs().apply(
            lambda row: row.dropna().idxmax().replace("|std", "") if row.notna().any() else np.nan, axis=1)
        t["Most distinctive situation"] = [
            "" if pd.isna(b) else (f"{b}: {'throws' if t.at[team, f'{b}|effect'] > 0 else 'runs'} "
                                   f"{abs(t.at[team, f'{b}|effect']) * 100:.0f} pts more than their usual lean "
                                   f"({t.at[team, f'{b}|vs_nfl']:+.0%} pass vs NFL)")
            for team, b in zip(t.index, biggest)]
        std_cols.append("situational_std")
        different[SITUATIONAL] = flags
    t["Unorthodoxy index"] = np.sqrt((t[std_cols] ** 2).mean(axis=1))
    t["Different on"] = [", ".join(labels[team]) or "nothing" for team in t.index]
    t["Rank"] = t["Unorthodoxy index"].rank(ascending=False, method="min").astype("Int64")
    # Fewer than 3 games: no game-clustered standard error, so nothing can be measured yet.
    t["Profile"] = np.select([t["Unorthodoxy index"].isna(), t["Unorthodoxy index"] >= UNORTHODOX,
                              ~different.any(axis=1)], ["Too few games", "Unorthodox", "NFL average"], "Distinct")
    t.attrs["measures"] = keys + ([SITUATIONAL] if situations else [])
    t.attrs["situations"] = situations
    return t.sort_values("Unorthodoxy index", ascending=False)


def reliability(plays: pd.DataFrame, season: int, settings: Settings = Settings()) -> pd.Series:
    """Split-half reliability (odd vs even weeks, Spearman-Brown to a full sample): how much of the team-to-team
    spread in each measure and situation is a real trait rather than noise. 1 = all trait, 0 = all noise."""
    halves = []
    for parity in (0, 1):
        p = plays[(plays["Season"] == season) & (plays["week"] % 2 == parity)]
        halves.append(season_table(p, season, settings))
    if any(h.empty for h in halves):
        return pd.Series(dtype=float)
    a, b = halves

    def corr(x: pd.Series, y: pd.Series) -> float:
        both = pd.concat([x, y], axis=1).dropna()
        return both.iloc[:, 0].corr(both.iloc[:, 1]) if len(both) >= 8 else np.nan   # too few teams to say

    out = {}
    for k in [m for m in a.attrs["measures"] if m in MEASURES]:
        out[MEASURES[k][0]] = corr(a[f"{k}_effect"], b[f"{k}_effect"])
    for name in a.attrs["situations"]:
        out[f"{name} (beyond lean)"] = corr(a[f"{name}|effect"], b[f"{name}|effect"])
    r = pd.Series(out)
    return (2 * r / (1 + r)).clip(lower=0)
