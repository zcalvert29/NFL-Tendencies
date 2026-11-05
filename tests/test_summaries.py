"""The plain-language summaries added on top of the models: the defense's guess-right rate (Balance Score), the
three-group verdicts and down mix (Surprise Plays), the cheat sheet and "lately" check (Tendencies), and the weekly
precomputed tables."""
import numpy as np
import pandas as pd

import data as dt
import surprise as sp
import tab_tendencies as tt
import unpredictability as up
from test_surprise import scored
from test_unpredictability import NO_CALLERS, league


def test_guess_right_and_league_guess():
    rows = pd.DataFrame({"posteam": ["A"] * 4 + ["B"] * 4, "caller": ["A coach"] * 4 + ["B coach"] * 4,
                         "dd": "1st & long", "PASS": [1, 1, 1, 0, 0, 0, 0, 0]})
    # League: 3 passes in 8, so a defense guesses run: right on 1 of A's 4 snaps and all 4 of B's.
    assert np.isclose(up.league_guess(rows), (1 + 4) / 8)
    t = up.balance_table(up.sample(league(), 2025, callers=NO_CALLERS)).set_index("Team")
    # TELL throws every shotgun snap, so guessing his more common call by down & distance is right more often.
    assert t["Guess right"].between(0.5, 1).all()
    assert t.loc["TELL", "Guess right"] > t.drop("TELL")["Guess right"].median()


def surprise_snaps():
    d = scored()
    d["game_id"] = d["posteam"] + (np.arange(len(d)) // 200).astype(str)
    d["DN"] = np.arange(len(d)) % 4 + 1
    d["ydstogo"] = np.arange(len(d)) % 11 + 1
    return sp.snaps(d, 2025)


def test_three_groups_and_verdicts():
    d = surprise_snaps()
    g = sp.three_groups(d, "EPA/play")
    assert set(g["Group"]) == set(sp.GROUPS) and g["Plays"].sum() == len(d)
    assert np.allclose(g["Value"], g["Own"] + g["Diff"])
    # Runs given away lose 0.3 EPA in the fake data; passes don't care.
    run, pas = sp.verdict(g, "Run"), sp.verdict(g, "Pass")
    assert run["head"] == "Stopped more" and run["clear"] and run["diff"] < -0.15
    assert pas["head"] == "No clear cost" and not pas["clear"]
    pg = sp.per_game(d, "EPA/play")
    assert 0 < pg["given_per_game"] < pg["runs_per_game"]
    assert np.isclose(pg["per_season"], pg["given_per_game"] * pg["diff"] * 17)


def test_obvious_mix():
    d = surprise_snaps()
    d["League confidence"] = np.where(np.arange(len(d)) % 3 == 0, 0.9, 0.5)
    share, sure, toss = sp.obvious_mix(d, "Pass")
    assert np.isclose(share.sum(), 1) and set(share.index) <= {"3rd/4th & 7+", "3rd/4th & 1-6", "2nd down",
                                                               "1st down"}
    assert 0 <= sure <= 1 and 0 <= toss <= 1


def tendency_league():
    d = league()
    d["Season"], d["success"], d["shotgun"] = 2025, (d["epa"] > 0).astype(float), (d["Formation"] == "Shotgun") * 1
    return d


def test_cheat_sheet_finds_the_tell():
    d = tendency_league()
    t = tt.cheat_sheet(d[d["posteam"] == "TELL"], d)
    top = t.iloc[0]
    # TELL throws every shotgun snap and runs every under-center snap: those are his two biggest tells.
    assert set(t["Look"].head(2)) == {"Shotgun", "Under center"} and abs(top["Gap"]) > 0.25
    assert (t["Plays"] >= tt.CHEAT_MIN_PLAYS).all() and len(t) <= tt.CHEAT_TOP


def test_lately_flags_a_recent_change():
    d = tendency_league()
    recent = (d["posteam"] == "AVG0") & (d["Week"] > 10 - tt.LATELY_GAMES) & (d["DN"] == 2)
    d.loc[recent, "PASS"] = 0                               # started running every 2nd down
    d["pass_oe"] = (d["PASS"] - d["xpass"]) * 100
    t = tt.lately(d[d["posteam"] == "AVG0"], d).set_index("Split")
    assert t.loc["2nd down", "Flag"] and t.loc["2nd down", "Lately"] < t.loc["2nd down", "Season"]
    assert not t.loc["Shotgun", "Flag"] or t.loc["Shotgun", "Lately"] < t.loc["Shotgun", "Season"]


def test_precomputed_missing_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(dt, "PRECOMPUTED_DIR", tmp_path)
    assert dt.load_precomputed("film").empty
    pd.DataFrame({"Season": [2025], "Neutral": [False]}).to_parquet(tmp_path / "film.parquet")
    assert len(dt.load_precomputed("film")) == 1
