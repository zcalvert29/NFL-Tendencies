import numpy as np
import pandas as pd

import surprise as sp


def scored(seed=0, n=4000):
    """Fake film-scored snaps for two offenses. GOOD passes better than BAD on every play. For both, runs where the
    film scout was surer than the league scout (a tell) lose 0.3 EPA; passes don't care."""
    rng = np.random.default_rng(seed)
    team = np.where(np.arange(n) % 2 == 0, "GOOD", "BAD")
    is_pass = rng.random(n) < 0.55
    p_league = rng.uniform(0.3, 0.7, n)
    tell = rng.normal(0, 0.05, n)
    p_film = np.clip(np.where(is_pass, p_league + tell, p_league - tell), 0.01, 0.99)
    epa = rng.normal(0, 0.2, n) + np.where(is_pass, np.where(team == "GOOD", 0.3, 0.0), -0.3 * (tell > 0.03))
    return pd.DataFrame({"scored": True, "posteam": team, "caller": team + " coach", "PASS": is_pass.astype(int),
                         "p_film": p_film, "p_league": p_league, "epa": epa, "success": (epa > 0).astype(float),
                         "explosive": (epa > 0.4).astype(float)})


def test_confidence_and_tell():
    d = sp.snaps(scored(), 2025)
    passes = d[d["Call"] == "Pass"]
    assert np.allclose(passes["Film confidence"], passes["Film confidence"].clip(0, 1))
    assert np.allclose(d["Tell"], d["Film confidence"] - d["League confidence"])
    # "vs own" = against that offense's average for that call, so each group averages 0.
    assert np.allclose(d.groupby(sp.OFFENSE)["epa vs own"].mean(), 0)


def test_readable_runs_do_worse_passes_dont():
    d = sp.snaps(scored(), 2025)
    run = sp.tell_trend(d, "Run", "EPA/play", within=True)
    pas = sp.tell_trend(d, "Pass", "EPA/play", within=True)
    assert run["slope"] < 0 and run["p"] < 0.01
    assert pas["p"] > 0.05
    assert sp.tell_sentence(run, "Run", "EPA/play", True).startswith("**Runs: readable ones did worse.**")
    assert "his usual run" in sp.tell_sentence(run, "Run", "EPA/play", True)
    assert sp.tell_sentence(pas, "Pass", "EPA/play", True).startswith("**Passes: no clear effect.**")
    t = sp.by_tell(d, "EPA/play", within=True).set_index(["Call", "Tell group"])
    assert t.loc[("Run", "His film gave it away (8+ pts)"), "Value"] < t.loc[("Run", "Same as league norms"), "Value"]


def test_caller_table():
    d = sp.snaps(scored(), 2025)
    t = sp.caller_table(d, "EPA/play", min_plays=15).set_index("Team")
    assert set(t.index) == {"GOOD", "BAD"}
    assert (t["Runs: given away vs others"] < -0.2).all()
    assert t["Share of runs given away"].between(0.1, 0.4).all()
