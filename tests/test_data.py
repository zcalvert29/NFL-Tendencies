import numpy as np
import pandas as pd
import pytest

import data as dt
import tendencies as td


@pytest.mark.parametrize("raw, code", [
    ("1 C, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 3 WR", "11"),
    ("1 C, 2 G, 1 QB, 1 RB, 2 T, 2 TE, 2 WR", "12"),
    ("1 C, 1 FB, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 2 WR", "21"),
    ("1 C, 2 G, 1 QB, 2 T, 5 WR", "00"),
    ("1 C, 1 G, 1 QB, 1 RB, 3 T, 1 TE, 3 WR", "11"),
    ("2 C, 2 G, 1 QB, 1 RB, 2 T, 2 TE, 1 WR", "12 +OL"),
    ("", np.nan),
    (None, np.nan),
])
def test_personnel_code(raw, code):
    out = dt.personnel_code(raw)
    assert (pd.isna(out) and pd.isna(code)) or out == code


@pytest.mark.parametrize("raw, pkg", [
    ("2 CB, 2 DE, 2 DT, 1 FS, 1 MLB, 2 OLB, 1 SS", "Base (4 DB)"),
    ("3 CB, 2 DE, 2 DT, 1 FS, 1 MLB, 1 OLB, 1 SS", "Nickel (5 DB)"),
    ("4 CB, 2 DE, 2 DT, 1 FS, 1 OLB, 1 SS", "Dime (6 DB)"),
    ("5 CB, 1 DE, 2 DT, 1 FS, 1 OLB, 1 SS", "Quarter (7+ DB)"),
])
def test_defense_package(raw, pkg):
    assert dt.defense_package(raw) == pkg


def _plays(**cols):
    n = len(next(iter(cols.values())))
    base = {c: [np.nan] * n for c in dt.PBP_COLUMNS + dt.PARTICIPATION_COLUMNS[2:]}
    base.update(game_id=["g"] * n, play_id=list(range(1, n + 1)), season=[2025] * n, season_type=["REG"] * n,
                week=[1] * n, posteam=["KC"] * n, defteam=["BUF"] * n, pass_=None)
    base.pop("pass_")
    base.update(cols)
    return pd.DataFrame(base)


def test_curate_buckets_and_flags():
    d = dt.curate(_plays(
        **{"pass": [1, 0, 1, 0], "rush": [0, 1, 0, 1]},
        ydstogo=[2, 5, 9, 15], yardline_100=[95, 60, 35, 5], yards_gained=[25, 12, 9, 3],
        score_differential=[0, -14, 3, 7], half_seconds_remaining=[900, 900, 60, 900],
        shotgun=[1, 0, 1, 0], air_yards=[22, np.nan, -2, np.nan], pass_location=["left", None, "middle", None],
        run_location=[None, "middle", None, "left"], run_gap=[None, None, None, "tackle"],
        qb_kneel=[0] * 4, qb_spike=[0] * 4, fixed_drive=[1, 1, 1, 1], success=[1, 0, 1, 1], qtr=[1, 2, 2, 3],
    ))
    assert list(d["Play Type"]) == ["Pass", "Run", "Pass", "Run"]
    assert list(d["Distance"]) == dt.DISTANCE_ORDER
    assert list(d["Field Zone"]) == ["Own 1-20", "Own 21-49", "Opp 50-21", "Inside the 10"]
    assert list(d["explosive"]) == [1.0, 1.0, 0.0, 0.0]          # pass 20+, run 10+
    assert list(d["neutral"]) == [True, False, False, True]      # down 14, and a 2-minute play, aren't neutral
    assert list(d["Formation"]) == ["Shotgun", "Under center", "Shotgun", "Under center"]
    assert d["Pass Zone"].iloc[0] == "20+ yds Left" and d["Pass Zone"].iloc[2] == "Behind LOS Middle"
    assert list(d["Run Lane"].iloc[[1, 3]]) == ["Middle", "Left tackle"]
    assert pd.isna(d["Prev Play"].iloc[0])
    assert list(d["Prev Result"].iloc[1:]) == ["Pass that worked", "Run that didn't", "Pass that worked"]


def test_two_prop_z():
    assert td.two_prop_z(50, 100, 50, 100) == 0
    assert td.two_prop_z(80, 100, 50, 100) > 1.96
    assert td.two_prop_z(1, 0, 5, 10) == 0


def test_tells_flags_only_real_patterns():
    rng = np.random.default_rng(0)
    n = 400
    league = pd.DataFrame({"scrimmage": True, "is_pass": rng.random(n * 10) < 0.6,
                           "Formation": rng.choice(["Shotgun", "Under center"], n * 10)})
    team = pd.DataFrame({"scrimmage": True, "Formation": ["Under center"] * 100 + ["Shotgun"] * 300})
    team["is_pass"] = [False] * 95 + [True] * 5 + [True] * 210 + [False] * 90   # 5% pass from under center
    for c in td.TELL_DIMENSIONS.values():
        for col in c:
            for frame in (league, team):
                if col not in frame:
                    frame[col] = np.nan
    for frame in (league, team):
        frame["epa"] = 0.0
        frame["is_run"] = ~frame["is_pass"]
    t = td.tells_frame(team, league)
    top = t.iloc[0]
    assert top["Situation"] == "Under center" and top["Pass Rate"] == pytest.approx(0.05)
