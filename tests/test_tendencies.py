import numpy as np
import pandas as pd
import pytest

import common as cm
import data as dt
import tab_red_zone
import tab_sequencing


@pytest.mark.parametrize("raw, code", [
    ("1 C, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 3 WR", "11"),
    ("1 C, 2 G, 1 QB, 1 RB, 2 T, 2 TE, 2 WR", "12"),
    ("1 C, 1 FB, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 2 WR", "21"),
    ("1 C, 2 G, 1 QB, 2 T, 5 WR", "00"),
    ("2 C, 2 G, 1 QB, 1 RB, 2 T, 2 TE, 1 WR", "12 +OL"),
    ("", np.nan),
    (None, np.nan),
])
def test_personnel_code(raw, code):
    out = dt.personnel_code(raw)
    assert (pd.isna(out) and pd.isna(code)) or out == code


def raw_plays(**cols) -> pd.DataFrame:
    """A minimal nflfastR-shaped frame: one drive of len(any column) snaps, KC on offense."""
    n = len(next(iter(cols.values())))
    base = {c: [np.nan] * n for c in dt.PBP_COLUMNS + dt.PARTICIPATION_COLUMNS}
    base.update(game_id=["g1"] * n, play_id=list(range(1, n + 1)), season=[2025] * n, season_type=["REG"] * n,
                week=[1] * n, posteam=["KC"] * n, defteam=["BUF"] * n, qb_kneel=[0] * n, qb_spike=[0] * n,
                fixed_drive=[1] * n, qtr=[1] * n, half_seconds_remaining=[900] * n, score_differential=[0] * n)
    base.update(cols)
    return pd.DataFrame(base)


def test_curate_columns_and_sequencing():
    d = dt.curate(raw_plays(**{"pass": [1, 0, 1, 0, 0], "rush": [0, 1, 0, 1, 0]},
                            down=[1, 2, 3, 1, 1], ydstogo=[2, 5, 9, 15, 10], yardline_100=[75] * 5,
                            yards_gained=[25, 12, 0, 3, 0], complete_pass=[1, 0, 0, 0, 0], sack=[0] * 5,
                            qb_scramble=[0] * 5, shotgun=[1, 0, 1, 0, 0], air_yards=[22, np.nan, -3, np.nan, np.nan],
                            pass_location=["left", None, "middle", None, None],
                            run_location=[None, "middle", None, "left", None], run_gap=[None, None, None, "tackle", None],
                            success=[1, 1, 0, 0, 0]))
    assert len(d) == 4                                         # the 5th snap is neither a run nor a pass
    assert list(d["PLAY TYPE"]) == ["Pass", "Run", "Pass", "Run"]
    assert list(d["Distance"]) == dt.DISTANCE_ORDER
    assert list(d["explosive"]) == [1.0, 1.0, 0.0, 0.0]      # pass 20+, run 10+
    assert list(d["Formation"]) == ["Shotgun", "Under center", "Shotgun", "Under center"]
    assert list(d["Run Lane"].iloc[[1, 3]]) == ["Middle", "Left tackle"]
    assert list(d["Depth"].iloc[[0, 2]]) == ["20+ yds", "Behind LOS"]
    assert list(d["first_of_drive"]) == [True, False, False, False]
    assert list(d["prev_type"].iloc[1:]) == ["Pass", "Run", "Pass"]
    assert list(d["prev_complete"].iloc[1:2]) == [1.0]


def frame(rows: list[dict]) -> pd.DataFrame:
    d = pd.DataFrame(rows)
    d["play_id"] = range(len(d))
    return d


def test_crosstab_nfl_columns_use_same_rows():
    team = frame([{"DN": 3, "PASS": 1, "pass_oe": 20.0}, {"DN": 3, "PASS": 1, "pass_oe": 10.0},
                  {"DN": 1, "PASS": 0, "pass_oe": -30.0}])
    league = frame([{"DN": 3, "PASS": p, "pass_oe": 0.0} for p in (1, 1, 1, 0)] +
                   [{"DN": 1, "PASS": p, "pass_oe": 0.0} for p in (1, 0)])
    t = cm.crosstab(team, "DN", ["Pass Rate", "NFL Pass Rate", "PROE", "Plays"], league)
    assert t.loc[3, "Pass Rate"] == 1.0 and t.loc[3, "NFL Pass Rate"] == 0.75
    assert t.loc[1, "NFL Pass Rate"] == 0.5
    assert t.loc[3, "PROE"] == pytest.approx(0.15)          # nflfastR's pass_oe is in percentage points
    assert t.loc[3, "Plays"] == 2


def test_style_compares_to_nfl_and_skips_tiny_rows():
    t = pd.DataFrame({"EPA per Play": [0.30, -0.05, 0.50], "NFL EPA per Play": [0.0, 0.10, 0.0],
                      "Plays": [50, 50, 2]}, index=["a", "b", "c"])
    styles = cm.style_table(t, gray_low_n=False)._compute().ctx
    css = lambda r: " ".join(f"{k}:{v}" for k, v in styles.get((r, 0), []))
    assert "#E1F5EE" in css(0)      # clearly better than the NFL in that row
    assert "#FCEBEB" in css(1)      # clearly worse
    assert css(2) == ""             # 2 plays: never colored


def test_tendency_grades():
    assert cm.tendency_grade(29, 35) == "Solid"
    assert cm.tendency_grade(6, 6) == "Likely"
    assert cm.tendency_grade(5, 6) == "Small sample"
    assert cm.badge(6, 8) == ""                             # under TELL_MIN_N plays
    assert "tell" in cm.badge(29, 35)


def test_two_prop_z():
    assert tab_sequencing.two_prop_z(50, 100, 50, 100) == 0
    assert tab_sequencing.two_prop_z(80, 100, 50, 100) > 1.96
    assert tab_sequencing.two_prop_z(1, 0, 5, 10) == 0


def test_red_zone_trips():
    snaps = pd.DataFrame({"game_id": ["g"] * 5, "fixed_drive": [1, 1, 2, 2, 3], "posteam": ["KC"] * 5,
                          "yardline_100": [40, 15, 25, 8, 60],
                          "fixed_drive_result": ["Touchdown", "Touchdown", "Field goal", "Field goal", "Punt"]})
    t = tab_red_zone.trips(snaps)
    assert list(t["fixed_drive"]) == [1, 2]                  # drive 3 never got inside the 20
    assert list(t["points"]) == [7, 3]
