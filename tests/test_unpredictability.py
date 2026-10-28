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
