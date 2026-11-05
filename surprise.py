"""
Surprise plays: does a call the defense could see coming do worse? Play by play, from the Unpredictability Index's
two scouts (unpredictability.py):

  * league confidence: how sure a scout who knows only league norms was of the call that actually came;
  * film confidence: how sure the film scout (league norms + this play caller's earlier games) was;
  * tell = film confidence - league confidence: how much his own habits gave this call away (positive) or hid it
    (negative), beyond what any team's call there would show.

Calls are compared within run and within pass (a pass gains more than a run on average, whatever the scouts thought),
and, by default, within the same offense: each play against that offense's average run or pass that season, so a
team with a great quarterback (whose passes are both good and predictable) doesn't pass for "predictable passes work".
No Streamlit code here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as cm

PLAY_OUTCOMES = {"EPA/play": ("epa", "+.3f"), "Success rate": ("success", ".1%"), "Explosive rate": ("explosive", ".1%")}
CONF_BINS = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
CONF_LABELS = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
TELL_BINS = [-1, -0.08, -0.03, 0.03, 0.08, 1]
TELL_LABELS = ["His film hid it (8+ pts)", "Hid it a little", "Same as league norms", "Gave it away a little",
               "His film gave it away (8+ pts)"]
GAVE_AWAY = 0.03          # tell above this = his habits made the call more predictable than league norms
OFFENSE = ["Season", "posteam", "caller", "Call"]


def snaps(scored: pd.DataFrame, season: int) -> pd.DataFrame:
    """The film-scored snaps (a caller's 4th game on) with each scout's confidence in the call that came, the tell,
    and every outcome also measured against the same offense's average for that call that season ("<col> vs own")."""
    d = scored[scored["scored"]].copy()
    d["Season"] = season
    is_pass = d["PASS"] == 1
    d["Call"] = np.where(is_pass, "Pass", "Run")
    d["Film confidence"] = np.where(is_pass, d["p_film"], 1 - d["p_film"])
    d["League confidence"] = np.where(is_pass, d["p_league"], 1 - d["p_league"])
    d["Tell"] = d["Film confidence"] - d["League confidence"]
    d["Tell group"] = pd.cut(d["Tell"], TELL_BINS, labels=TELL_LABELS)
    for col in ["epa", "success", "explosive", "Tell"]:
        d[f"{col} vs own"] = d[col] - d.groupby(OFFENSE)[col].transform("mean")
    return d


def by_confidence(d: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Outcome by how sure each scout was of the call, for runs and passes."""
    col = PLAY_OUTCOMES[outcome][0]
    rows = []
    for scout in ("Film confidence", "League confidence"):
        g = d.groupby(["Call", pd.cut(d[scout], CONF_BINS, labels=CONF_LABELS, include_lowest=True)], observed=True)
        t = g[col].agg(["mean", "size"]).reset_index().rename(columns={scout: "Confidence", "mean": "Value",
                                                                        "size": "Plays"})
        rows.append(t.assign(Scout="His film" if scout == "Film confidence" else "League norms"))
    return pd.concat(rows, ignore_index=True)


def by_tell(d: pd.DataFrame, outcome: str, within: bool) -> pd.DataFrame:
    """Outcome by how much his habits gave the call away, for runs and passes (vs his own average if within)."""
    col = PLAY_OUTCOMES[outcome][0] + (" vs own" if within else "")
    g = d.groupby(["Call", "Tell group"], observed=True)[col].agg(["mean", "size", "sem"]).reset_index()
    return g.rename(columns={"mean": "Value", "size": "Plays", "sem": "SE"})


def tell_trend(d: pd.DataFrame, call: str, outcome: str, within: bool) -> dict:
    """Least-squares outcome against the tell (in percentage points), one call type."""
    g = d[d["Call"] == call]
    col = PLAY_OUTCOMES[outcome][0]
    x = g["Tell vs own"] if within else g["Tell"]
    return cm.fit_trend(x * 100, g[f"{col} vs own"] if within else g[col])


def plural(call: str) -> str:
    return "passes" if call == "Pass" else "runs"


def tell_sentence(t: dict, call: str, outcome: str, within: bool) -> str:
    """'**Runs: readable ones did worse.** For every 10 points his habits made a run more predictable ...'"""
    noun = plural(call)
    if t["n"] < 100 or pd.isna(t["r"]):
        return f"**{noun.capitalize()}:** too few plays to tell."
    effect = t["slope"] * 10
    vs = f"his usual {call.lower()}" if within else f"other {noun}"
    more = "less" if effect < 0 else "more"
    if outcome == "EPA/play":
        did = f"gained {abs(effect):.3f} EPA per play {more} than {vs}"
    else:
        verb = "succeeded" if outcome == "Success rate" else "went for an explosive gain"
        did = f"{verb} {abs(effect) * 100:.1f} percentage points {more} often than {vs}"
    p = "p < 0.01" if t["p"] < 0.01 else f"p = {t['p']:.2f}"
    how = (f"for every 10 points his own habits made a {call.lower()} more predictable than league norms, it {did} "
           f"({p}, {t['n']:,} {noun})")
    if t["p"] >= 0.05:
        return (f"**{noun.capitalize()}: no clear effect.** Being readable didn't clearly cost his {noun}: {how}, "
                "within what chance produces.")
    verdict = "readable ones did worse" if effect < 0 else "readable ones did better"
    return f"**{noun.capitalize()}: {verdict}.** {how[0].upper() + how[1:]}."


def caller_table(d: pd.DataFrame, outcome: str, min_plays: int = 15) -> pd.DataFrame:
    """Per play caller (seasons pooled): how his calls did when his habits gave them away vs his other calls, against
    his own average for that call each season, for runs and passes."""
    col = PLAY_OUTCOMES[outcome][0] + " vs own"
    rows = []
    for (team, caller), g in d.groupby(["posteam", "caller"]):
        row = {"Team": team, "Play caller": caller, "Seasons": ", ".join(str(s) for s in sorted(g["Season"].unique())),
               "Snaps": len(g)}
        for call in ("Run", "Pass"):
            c, noun = g[g["Call"] == call], plural(call)
            gave = c["Tell"] > GAVE_AWAY
            row[f"{noun.capitalize()} given away"] = int(gave.sum())
            row[f"Share of {noun} given away"] = gave.mean() if len(c) else np.nan
            row[f"{noun.capitalize()}: given away vs others"] = (c.loc[gave, col].mean() - c.loc[~gave, col].mean()
                                                     if gave.sum() >= min_plays and (~gave).sum() >= min_plays else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# ---- Plain-language summaries ----------------------------------------------------------------------------------
GROUPS = ["Hid it", "Usual", "Gave it away"]


def group3(tell: pd.Series) -> pd.Series:
    """Hid it / Usual / Gave it away: his habits made the call less or more predictable than league norms by
    GAVE_AWAY or more."""
    return pd.Series(np.select([tell < -GAVE_AWAY, tell > GAVE_AWAY], ["Hid it", "Gave it away"], "Usual"),
                     index=tell.index)


def three_groups(d: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Per call and group: plays, the raw outcome, the same offenses' usual outcome for that call ("Own"), and the
    difference with its standard error."""
    col = PLAY_OUTCOMES[outcome][0]
    g = d.assign(Group=group3(d["Tell"]))
    rows = []
    for (call, grp), h in g.groupby(["Call", "Group"]):
        rows.append({"Call": call, "Group": grp, "Plays": len(h), "Value": h[col].mean(),
                     "Own": (h[col] - h[f"{col} vs own"]).mean(), "Diff": h[f"{col} vs own"].mean(),
                     "SE": h[f"{col} vs own"].sem()})
    return pd.DataFrame(rows)


def verdict(groups: pd.DataFrame, call: str) -> dict:
    """The given-away group for one call: its difference from usual, whether it's clear (beyond 2 standard errors),
    and a one-or-two-word headline."""
    r = groups[(groups["Call"] == call) & (groups["Group"] == "Gave it away")]
    if r.empty:
        return {"head": "Too few plays", "diff": np.nan, "clear": False, "value": np.nan, "own": np.nan, "z": np.nan}
    r = r.iloc[0]
    z = r["Diff"] / r["SE"] if r["SE"] > 0 else 0.0
    clear = abs(z) >= 2
    head = ("Stopped more" if r["Diff"] < 0 else "Did better") if clear else "No clear cost"
    return {"head": head, "diff": r["Diff"], "clear": clear, "value": r["Value"], "own": r["Own"], "z": z}


def per_game(d: pd.DataFrame, outcome: str) -> dict:
    """How many of a team's runs a game its habits gave away, and what the given-away difference adds up to over a
    game and a 17-game season."""
    team_games = d.drop_duplicates(["game_id", "posteam"]).shape[0]
    runs = d[d["Call"] == "Run"]
    given = runs[group3(runs["Tell"]) == "Gave it away"]
    col = PLAY_OUTCOMES[outcome][0]
    diff = given[f"{col} vs own"].mean() if len(given) else np.nan
    per = len(given) / team_games if team_games else np.nan
    return {"given_per_game": per, "runs_per_game": len(runs) / team_games if team_games else np.nan,
            "diff": diff, "per_season": per * diff * 17 if pd.notna(diff) else np.nan}


def obvious_mix(d: pd.DataFrame, call: str, sure: float = 0.8) -> tuple[pd.Series, float, float]:
    """For calls an average defense was `sure`+ confident of: the share by situation, their success rate, and the
    success rate of toss-ups (league confidence 40-60%) of the same call."""
    c = d[d["Call"] == call]
    s = c[c["League confidence"] >= sure]
    if call == "Pass":
        cat = np.select([(s["DN"] >= 3) & (s["ydstogo"] >= 7), s["DN"] >= 3, s["DN"] == 2],
                        ["3rd/4th & 7+", "3rd/4th & 1-6", "2nd down"], "1st down")
    else:
        cat = np.select([(s["DN"] >= 3) & (s["ydstogo"] <= 2), s["ydstogo"] <= 2],
                        ["3rd/4th & 1-2", "1st/2nd & 1-2"], "Other spots")
    toss = c[c["League confidence"].between(0.4, 0.6)]
    return (pd.Series(cat, index=s.index).value_counts(normalize=True), float(s["success"].mean()),
            float(toss["success"].mean()))
