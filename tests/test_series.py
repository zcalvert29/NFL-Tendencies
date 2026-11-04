import numpy as np
import pandas as pd

import series as se

NO_CALLERS = pd.DataFrame(columns=["season", "team", "play_caller", "role", "first_week", "last_week"])


def play(drive, pid, down, togo, kind, **kw):
    row = {"game_id": "G1", "play_id": pid, "season": 2025, "season_type": "REG", "week": 3, "posteam": "AAA",
           "fixed_drive": drive, "down": down, "ydstogo": togo, "yardline_100": 45, "play_type": kind,
           "half_seconds_remaining": 900, "score_differential": 0, "qb_kneel": 0, "qb_spike": 0,
           "rush": int(kind == "run"), "qb_scramble": 0, "first_down": 0, "touchdown": 0, "interception": 0,
           "fumble_lost": 0, "epa": 0.0, "desc": ""}
    row.update(kw)
    return row


def raw():
    return pd.DataFrame([
        # Drive 1: 3rd & 4 run gains 2, a penalty, then 4th & 2 run converts -> series won on 4th.
        play(1, 10, 3, 4, "run"), play(1, 11, 4, 2, "no_play"), play(1, 12, 4, 2, "run", first_down=1),
        # Drive 2: 3rd & 5 pass incomplete, punt -> kicked.
        play(2, 20, 3, 5, "pass"), play(2, 21, 4, 5, "punt", rush=0),
        # Drive 3: 3rd & 6 pass converts.
        play(3, 30, 3, 6, "pass", first_down=1),
        # Drive 4: 3rd & 4 pass intercepted -> turnover, no 4th down.
        play(4, 40, 3, 4, "pass", interception=1),
        # Drive 5: 3rd & 4 run stopped, go for it on 4th & 3 and fail.
        play(5, 50, 3, 4, "run"), play(5, 51, 4, 3, "pass"),
        # Last 2 minutes of a half: left out.
        play(6, 60, 3, 4, "run", half_seconds_remaining=90),
    ])


def test_series_outcomes():
    d = se.third_downs(raw(), NO_CALLERS).set_index("play_id")
    assert 60 not in d.index
    assert d.loc[10, ["conv3", "went", "conv4", "series"]].tolist() == [0, 1, 1, 1]
    assert d.loc[10, "togo4"] == 2 and d.loc[10, "Call"] == "Run"
    assert d.loc[20, ["conv3", "went", "kicked", "series"]].tolist() == [0, 0, 1, 0]
    assert d.loc[30, ["conv3", "series"]].tolist() == [1, 1]
    assert d.loc[40, ["turnover3", "went", "kicked", "series"]].tolist() == [1, 0, 0, 0]
    assert d.loc[50, ["went", "conv4", "series"]].tolist() == [1, 0, 0]


def test_summary_and_outcome_shares():
    d = se.third_downs(raw(), NO_CALLERS)
    runs = se.summary(d[d["Call"] == "Run"])
    assert runs["Plays"] == 2 and runs["Won the series"] == 0.5 and runs["Go-for-it rate"] == 1.0
    assert runs["Converted on 4th"] == 0.5 and runs["Left 4th & 1-2"] == 0.5
    shares = se.outcome_shares(d)
    assert np.isclose(shares.sum(), 1.0)
    assert shares["Converted on 3rd"] == 0.2 and shares["Turned it over on 3rd"] == 0.2
    assert shares["Converted on 4th"] == 0.2 and shares["Went for it, stopped"] == 0.2 and shares["Kicked"] == 0.2


def test_caller_table_uses_play_caller_list():
    callers = pd.DataFrame({"season": [2025], "team": ["AAA"], "play_caller": ["Pat Coach"], "role": ["OC"],
                            "first_week": [1], "last_week": [22]})
    d = se.third_downs(raw(), callers)
    t = se.caller_table(d, min_plays=1)
    assert t.loc[0, "Play caller"] == "Pat Coach" and t.loc[0, "Plays"] == 5
    assert t.loc[0, "Run share"] == 0.4 and t.loc[0, "Runs: won the series"] == 0.5
