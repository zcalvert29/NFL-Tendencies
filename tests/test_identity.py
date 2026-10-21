import numpy as np
import pandas as pd
import pytest

import data as dt
import identity as idn


def league(personnel: bool = True, seed: int = 0) -> pd.DataFrame:
    """A fake 32-team season: every team calls it like the league, except PASSY (throws 20 points more often than
    expected), GUNS (shotgun every snap) and HEAVY (never 3+ WR). 16 games of 60 snaps each."""
    rng = np.random.default_rng(seed)
    rows = []
    teams = ["PASSY", "GUNS", "HEAVY"] + [f"AVG{i}" for i in range(29)]
    for team in teams:
        for game in range(16):
            n = 60
            dn = rng.integers(1, 4, n)
            togo = rng.integers(1, 15, n)
            xpass = np.clip(0.35 + 0.15 * (dn - 1) + 0.01 * togo, 0.05, 0.95)
            p = np.clip(xpass + (0.20 if team == "PASSY" else 0), 0, 1)
            gun = np.ones(n) if team == "GUNS" else (rng.random(n) < 0.6).astype(float)
            pers = np.where(rng.random(n) < (0 if team == "HEAVY" else 0.6), "11", "12")
            rows.append(pd.DataFrame({
                "posteam": team, "game_id": f"{team}{game}", "week": game + 1, "Season": 2025, "season_type": "REG",
                "DN": dn.astype(float), "ydstogo": togo, "yardline_100": rng.integers(5, 95, n),
                "Distance": pd.cut(togo, [0, 3, 6, 10, 200], labels=dt.DISTANCE_ORDER).astype(str),
                "half_seconds_remaining": 900, "score_differential": 0, "xpass": xpass,
                "PASS": (rng.random(n) < p).astype(int), "shotgun": gun,
                "Personnel": pers if personnel else np.nan}))
    d = pd.concat(rows, ignore_index=True)
    d["pass_oe"] = (d["PASS"] - d["xpass"]) * 100
    return d


def test_personnel_counts():
    assert idn.personnel_counts("11") == (1, 1, 0)
    assert idn.personnel_counts("21") == (2, 1, 0)
    assert idn.personnel_counts("12 +OL") == (1, 2, 1)
    assert all(np.isnan(x) for x in idn.personnel_counts(np.nan))


def test_measure_effect_and_clustered_z():
    g = pd.DataFrame({"game_id": np.repeat(["a", "b", "c", "d"], 50), "v": 1.0, "e": 0.6})
    effect, z, se, n = idn._measure(g, "v", "e")
    assert effect == pytest.approx(0.4) and n == 200
    assert np.isnan(z) or z > 5        # every game identical: no game-to-game noise at all
    rng = np.random.default_rng(1)
    g2 = pd.DataFrame({"game_id": np.repeat(list("abcdefghij"), 60), "v": (rng.random(600) < 0.5).astype(float),
                       "e": 0.5})
    effect, z, se, n = idn._measure(g2, "v", "e")
    assert abs(z) < 2.5                # a coin flip shouldn't look different from a coin flip


def test_shrink_pulls_noisy_values_in():
    effect = pd.Series([0.10, 0.10, -0.05, 0.0, 0.02])
    se = pd.Series([0.01, 0.20, 0.01, 0.01, 0.01])      # same effect, but the 2nd is very noisy
    shrunk, std, sd = idn._shrink(effect, se)
    assert abs(shrunk[1] - effect.mean()) < abs(shrunk[0] - effect.mean())


def test_season_table_finds_the_odd_teams():
    t = idn.season_table(league(), 2025)
    assert set(t.attrs["measures"]) == {"lean", "shotgun", "spread"}
    assert "pass-heavy" in t.at["PASSY", "Different on"]
    assert "more shotgun" in t.at["GUNS", "Different on"]
    assert "heavier personnel" in t.at["HEAVY", "Different on"]
    assert set(t.index[:3]) == {"PASSY", "GUNS", "HEAVY"}           # the three oddballs top the index
    avg = t.loc[[f"AVG{i}" for i in range(29)], "Profile"]
    assert not (avg == "Unorthodox").any()                          # no normal team is called unorthodox...
    assert (avg == "NFL average").mean() >= 0.7                     # ...and most read as average (some flag by chance)
    assert abs(t["lean_effect"].mean()) < 0.01                      # lean is re-centered on the league


def test_personnel_columns_dropped_without_charting():
    t = idn.season_table(league(personnel=False), 2025)
    assert t.attrs["measures"] == ["lean", "shotgun"]
    assert not any(c.startswith(("spread", "3+ WR", "2+ ", "Personnel", "Top personnel")) for c in t.columns)


def test_z_scores_are_calibrated_for_average_teams():
    """With no oddballs, team z-scores should look like noise: SD near 1 (t with 15 df ≈ 1.07)."""
    zs = []
    for seed in range(3):
        d = idn.sample(league(seed=seed).query("posteam.str.startswith('AVG')", engine="python"))
        zs += [idn._measure(g, "PASS", "exp_pass")[1] for _, g in d.groupby("posteam")]
    assert 0.75 < np.std(zs) < 1.4
