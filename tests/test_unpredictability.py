import numpy as np
import pandas as pd

import data as dt
import unpredictability as up


def league(seed: int = 0, n_games: int = 10) -> pd.DataFrame:
    """A fake 32-team season. Every team throws 70% from shotgun and 30% from under center, except TELL, which
    throws every shotgun snap and runs every under-center snap: a defense with his film knows every call."""
    rng = np.random.default_rng(seed)
    rows = []
    for team in ["TELL"] + [f"AVG{i}" for i in range(31)]:
        for game in range(n_games):
            n = 60
            gun = rng.random(n) < 0.6
            p = np.where(gun, 1.0, 0.0) if team == "TELL" else np.where(gun, 0.7, 0.3)
            dn = rng.integers(1, 4, n)
            togo = rng.integers(1, 15, n)
            rows.append(pd.DataFrame({
                "posteam": team, "game_id": f"{team}{game}", "Week": game + 1, "Season": 2025, "season_type": "REG",
                "DN": dn.astype(float), "ydstogo": togo, "yardline_100": rng.integers(5, 95, n),
                "Distance": pd.cut(togo, [0, 3, 6, 10, 200], labels=dt.DISTANCE_ORDER).astype(str),
                "half_seconds_remaining": 900, "score_differential": 0, "xpass": 0.54,
                "PASS": (rng.random(n) < p).astype(int), "Formation": np.where(gun, "Shotgun", "Under center"),
                "Personnel": np.nan, "Motion": np.where(rng.random(n) < 0.5, "Motion", "No motion"),
                "Backfield": "1 back", "epa": rng.normal(0, 1, n), "first_of_drive": np.arange(n) % 6 == 0}))
    d = pd.concat(rows, ignore_index=True)
    d["pass_oe"] = (d["PASS"] - d["xpass"]) * 100
    d["prev_type"] = d.groupby("game_id")["PASS"].shift().map({1: "Pass", 0: "Run"})
    return d


NO_CALLERS = pd.DataFrame(columns=["season", "team", "play_caller", "role", "first_week", "last_week"])


def test_fit_logit_recovers_an_effect():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(20000), rng.random(20000) < 0.5]).astype(float)
    y = (rng.random(20000) < 1 / (1 + np.exp(-(-0.5 + 1.0 * X[:, 1])))).astype(float)
    b = up.fit_logit(X, y, np.zeros(20000), lam=1e-6)
    assert np.allclose(b, [-0.5, 1.0], atol=0.08)


def test_tell_team_is_the_most_readable():
    d, clues = up.score(league(), 2025, callers=NO_CALLERS)
    assert "personnel" not in clues and "formation" in clues
    t = up.caller_table(d, clues).set_index("Team")
    assert t["Unpredictability index"].idxmin() == "TELL"
    assert t.loc["TELL", "Profile"] == "Readable"
    assert t.loc["TELL", "Biggest tell"] == "Formation"
    assert t.loc["TELL", "Film accuracy"] > 0.95
    # Average teams have nothing for the film to find: their edge is about 0 (it can't be learned, only overfit).
    assert t.drop("TELL")["Edge"].abs().max() < 0.03
    looks = up.look_table(d, "TELL", "TELL play caller", clues)
    top = looks.nlargest(2, "Edge share")
    assert set(top["Look"]) == {"Shotgun", "Under center"}


def test_walk_forward_only_uses_earlier_games():
    base = league()
    d1, clues = up.score(base, 2025, callers=NO_CALLERS)
    first = d1[d1["posteam"] == "TELL"].groupby("Week")["scored"].any()
    assert not first.loc[: up.MIN_GAMES].any() and first.loc[up.MIN_GAMES + 1:].all()
    # Changing TELL's last game can't change what the film scout predicted for any earlier game.
    flipped = base.copy()
    last = (flipped["posteam"] == "TELL") & (flipped["Week"] == flipped["Week"].max())
    flipped.loc[last, "PASS"] = 1 - flipped.loc[last, "PASS"]
    d2, _ = up.score(flipped, 2025, callers=NO_CALLERS)
    early = (d1["posteam"] == "TELL") & d1["scored"] & (d1["Week"] < d1["Week"].max())
    assert np.allclose(d1.loc[early, "p_film"], d2.loc[early, "p_film"])


def test_play_caller_change_starts_new_film():
    callers = pd.DataFrame({"season": [2025, 2025], "team": ["TELL", "TELL"], "play_caller": ["Old", "New"],
                            "role": ["OC", "HC"], "first_week": [1, 6], "last_week": [5, 22]})
    d, clues = up.score(league(), 2025, callers=callers)
    tell = d[d["posteam"] == "TELL"]
    assert set(tell["caller"]) == {"Old", "New"}
    scored_weeks = sorted(tell.loc[tell["scored"], "Week"].unique())
    # Old: weeks 1-5, scored from his 4th game; New: weeks 6-10, scored from week 9.
    assert scored_weeks == [4, 5, 9, 10]
    assert set(up.caller_table(d, clues)["Play caller"]) >= {"Old", "New"}


def test_play_caller_list_is_complete():
    c = up.load_callers()
    for season, g in c.groupby("season"):
        assert g["team"].nunique() == 32, season
        for team, tg in g.groupby("team"):
            weeks = sorted(w for r in tg.itertuples() for w in range(r.first_week, r.last_week + 1))
            assert weeks == list(range(1, 23)), (season, team)


def test_balance_score():
    d = up.sample(league(), 2025, callers=NO_CALLERS)
    t = up.balance_table(d).set_index("Team")
    # 3rd/4th & 7+ are left out; everything else counts.
    assert not set(up.BALANCE_SKIP) & set(up.BALANCE_CELLS) and len(up.BALANCE_CELLS) == 12
    assert t.loc["TELL", "Snaps"] == d[(d["posteam"] == "TELL") & ~d["dd"].isin(up.BALANCE_SKIP)].shape[0]
    # Every AVG team throws about 54% in every cell, so it scores close to 100 once chance is taken out. TELL throws
    # every shotgun snap (64% overall): a little lopsided by down & distance, but nowhere near how readable it is,
    # since formation is what gives it away and the balance score never sees formation.
    avg = t.drop("TELL")["Balance score"]
    assert avg.between(95, 102).all()
    assert 85 < t.loc["TELL", "Balance score"] < avg.min()
    # The small-sample correction only ever raises the score.
    assert (t["Balance score"] >= t["Raw balance score"]).all()


def test_balance_correction_removes_chance():
    rows = pd.DataFrame({"posteam": "A", "caller": "A play caller", "dd": "1st & long", "PASS": [1, 1, 1, 0]})
    c = up.balance_cells(rows)
    # 3 of 4: raw gap (0.75 - 0.5)^2 = 0.0625, chance part 0.75 * 0.25 / 3 = 0.0625, so no tendency left yet.
    assert np.isclose(c["Gap"].iloc[0], 0.0)


def test_tiers_and_phrases():
    assert [up.tier(i) for i in (14, 40, 43, 50, 57, 60, 64)] == [
        "Open Book", "Open Book", "Readable", "Average", "Mixes It Up", "Mixes It Up", "Poker Face"]
    assert up.look_name("Personnel", "11") == "11 personnel"
    assert up.look_phrase("Personnel", "11") == "In 11 personnel"
    assert up.look_phrase("Down & distance", "3rd & short") == "On 3rd & short"
    assert up.look_phrase("Formation", "Under center") == "From under center"
    assert up.look_phrase("Last call", "After a run") == "After a run"


def test_plain_language_summary_for_a_tell_team():
    d, clues = up.score(league(), 2025, callers=NO_CALLERS)
    t = up.caller_table(d, clues).set_index("Team", drop=False)
    r = t.loc["TELL"]
    assert up.tier(r["Unpredictability index"]) == "Open Book"
    assert r["Plays given away"] > 5 > t.drop("TELL")["Plays given away"].max()
    looks = up.look_table(d, "TELL", "TELL play caller", clues)
    found = {x["look"]: x for x in up.tells(looks)}
    assert {"Shotgun", "Under center"} <= set(found)
    assert found["Shotgun"]["call"] == "PASS" and found["Shotgun"]["rate"] == 1.0
    assert found["Under center"]["call"] == "RUN"
    text = up.verdict(r, up.tells(looks), up.misleads(looks))
    assert text.startswith("TELL ") and "formation" in text
    assert up.certainty(t)["TELL"] == "Clear"
