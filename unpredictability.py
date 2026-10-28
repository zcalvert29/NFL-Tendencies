"""
Unpredictability index: how well a defense that has studied an offense's film could call run or pass before the
snap, beyond what it would guess knowing only how the NFL plays.

Two scouts predict every snap (pass = 1, run = 0) from what a defense sees before the snap:

  * Generic scout: knows the league. nflfastR's xpass (down, distance, field position, score, time, timeouts) plus
    league-wide effects for each clue (formation, personnel, motion, backfield, what was called last snap), fit on
    the other 31 teams that season. It never sees this offense.
  * Film scout: starts from the generic scout and learns this play caller's own tendencies from his earlier games:
    his overall lean (how much more he throws than the generic scout expects, in any look) and
    his habits by look: a logistic model with the generic scout's guess as an offset and one ridge-penalized effect
    per clue value
    (so a look seen 6 times barely moves the guess and one seen 200 times can). Walk-forward: a game is predicted
    only from that caller's earlier games, and only from his 4th game on (MIN_GAMES of film first). A mid-season
    change of play caller starts a new film study.

Readability (the "edge") = how many bits per snap the film scout saves over the generic scout (log loss, out of
sample): what studying this play caller buys a defense that already knows the league. It splits in two: the lean
edge (a film scout that knows only his overall lean) and the look edge (the rest: tendencies by down & distance,
formation, personnel, motion...). Most of the reliable difference between callers is lean; the look edge is smaller
and noisier, which is why it's shown but not the index on its own. Overfit
can't help the film scout: a tendency that doesn't repeat costs it bits, so readability can be negative (the film
misleads). Unpredictability index = a T-score of readability, flipped: 50 = an average NFL play caller, 60 = one
standard deviation harder to read than average, 40 = one easier.

"League tells" is the other half of the story: how much the generic scout's clues alone tell it beyond xpass for this
offense (a heavy under-center team gives its runs away the way everyone does). It's shown for context but not in the
index: it barely repeats from one half of a season to the other, while the film edge does (split-half r 0.5-0.85).

The same fit with the lean and one clue at a time (scored against the lean alone) says which clues give a caller away (single-clue edges overlap: shotgun and
backfield count often tell the same story). Each look's share of its clue's edge (that clue's film scout's savings on
snaps from that look) says where a caller is readable (positive) or where his film misleads (negative).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import data as dt

MIN_GAMES = 3          # games of a caller's film before his snaps are scored
LAMBDA = 150.0         # ridge penalty on each film-scout team effect (chosen on 2023-25: larger = steadier, smaller =
                       # bigger edges; a single look needs roughly 4 * LAMBDA snaps to move its guess halfway alone)
LEAN_LAMBDA = LAMBDA   # penalty on the caller's overall lean (a lighter one made the index noisier on 2023-25)
LEAGUE_LAMBDA = 1.0    # light penalty on the generic scout's league effects (keeps rare looks finite)
MIN_SNAPS = 150        # scored snaps a caller needs to get an index
MAX_MARGIN = 16        # only snaps within this score margin (blowouts are one-sided by necessity)
RARE_SHARE = 0.01      # personnel groupings under this share of league snaps are pooled as "Other"
CALLERS_PATH = Path(__file__).parent / "play_callers.csv"

CLUES = {   # key -> (label, columns)
    "downdist": ("Down & distance", ["dd"]),
    "field": ("Field position", ["Field"]),
    "score": ("Score", ["Score"]),
    "sequence": ("Last call", ["Last call"]),
    "formation": ("Formation", ["Formation"]),
    "personnel": ("Personnel", ["Personnel"]),
    "motion": ("Motion", ["Motion"]),
    "backfield": ("Backfield", ["Backfield"]),
}
COLUMN_CLUE = {c: k for k, (_, cols) in CLUES.items() for c in cols}


# ---- Play callers -------------------------------------------------------------------------------------------
def load_callers(path: Path = CALLERS_PATH) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"season": int, "first_week": int, "last_week": int})


def assign_callers(d: pd.DataFrame, callers: pd.DataFrame) -> pd.Series:
    """The play caller for each snap (from play_callers.csv; "<TEAM> play caller" where the list has no row)."""
    out = pd.Series(d["posteam"] + " play caller", index=d.index, dtype=object)
    for r in callers.itertuples():
        m = (d["Season"] == r.season) & (d["posteam"] == r.team) & d["Week"].between(r.first_week, r.last_week)
        out[m] = r.play_caller
    return out


# ---- Sample ---------------------------------------------------------------------------------------------------
def sample(plays: pd.DataFrame, season: int, weeks: tuple[int, int] | None = None,
           callers: pd.DataFrame | None = None) -> pd.DataFrame:
    """Regular-season snaps with a down, outside the last 2 minutes of a half, within MAX_MARGIN, with the clues."""
    d = plays[(plays["Season"] == season) & (plays["season_type"] == "REG") & plays["DN"].notna()
              & (plays["half_seconds_remaining"] > 120) & (plays["score_differential"].abs() <= MAX_MARGIN)].copy()
    if weeks:
        d = d[d["Week"].between(*weeks)]
    d["dd"] = (d["DN"].astype(int).map(dt.DOWN_NAMES) + " & "
               + d["Distance"].astype(str).str.split(" (", regex=False).str[0].str.lower())
    d["Field"] = pd.cut(d["yardline_100"], [0, 20, 60, 100], labels=["Red zone", "Midfield", "Own end"]).astype(object)
    d["Score"] = np.select([d["score_differential"] < -3, d["score_differential"] > 3], ["Trailing 4+", "Leading 4+"],
                           "Within 3")
    d["Last call"] = np.where(d["first_of_drive"], "Start of drive", "After a " + d["prev_type"].fillna("").str.lower())
    common = d["Personnel"].value_counts(normalize=True)
    d["Personnel"] = d["Personnel"].where(d["Personnel"].isin(common[common >= RARE_SHARE].index) | d["Personnel"].isna(),
                                          "Other")
    x = d["xpass"].astype(float)
    d["xpass"] = x.fillna(d.groupby("dd")["PASS"].transform("mean")).clip(0.02, 0.98)
    d["caller"] = assign_callers(d, callers if callers is not None else load_callers())
    return d


def available_clues(d: pd.DataFrame) -> list[str]:
    """Clues this season has (personnel isn't charted for the season in progress; motion and backfield need FTN)."""
    return [k for k, (_, cols) in CLUES.items() if all(d[c].notna().mean() > 0.5 for c in cols)]


# ---- Logistic ridge with an offset -------------------------------------------------------------------------------
def _expit(x):
    return 1 / (1 + np.exp(-x))


def fit_logit(X: np.ndarray, y: np.ndarray, offset: np.ndarray, lam, beta: np.ndarray | None = None,
              iters: int = 30) -> np.ndarray:
    """argmax of log likelihood(y | offset + X b) - sum(lam/2 b^2), by Newton's method (IRLS). lam is one penalty
    or one per column."""
    b = np.zeros(X.shape[1]) if beta is None else beta.copy()
    lam = np.broadcast_to(np.asarray(lam, float), b.shape)
    eye = np.diag(lam)
    for _ in range(iters):
        p = _expit(offset + X @ b)
        step = np.linalg.solve((X.T * (p * (1 - p))) @ X + eye, X.T @ (y - p) - lam * b)
        b += step
        if np.abs(step).max() < 1e-6:
            break
    return b


def _bits(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log2(p) + (1 - y) * np.log2(1 - p))


def design(d: pd.DataFrame, clues: list[str]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """One-hot columns for the clues' values (a missing value = no column on), their names ("column=value"), and
    the clue each belongs to."""
    cols = [c for k in clues for c in CLUES[k][1]]
    X = pd.get_dummies(d[cols].astype(object), prefix=cols, prefix_sep="=", dtype=float)
    return X.to_numpy(), list(X.columns), np.array([COLUMN_CLUE[n.split("=")[0]] for n in X.columns])


# ---- The two scouts -------------------------------------------------------------------------------------------
def generic_scout(d: pd.DataFrame, X: np.ndarray) -> np.ndarray:
    """Logit of the generic scout's P(pass) for every snap, fit on the other 31 teams (so a look only one team uses
    can't teach the league about that team)."""
    y, off = d["PASS"].to_numpy(float), np.log(d["xpass"] / (1 - d["xpass"])).to_numpy()
    team = d["posteam"].to_numpy()
    out = np.zeros(len(d))
    beta = None
    for t in np.unique(team):
        rest = team != t
        beta = fit_logit(X[rest], y[rest], off[rest], LEAGUE_LAMBDA, beta)
        out[~rest] = off[~rest] + X[~rest] @ beta
    return out


def film_scout(d: pd.DataFrame, X: np.ndarray, groups: np.ndarray, league_logit: np.ndarray, clues: list[str],
               by: str = "caller") -> pd.DataFrame:
    """Walk-forward scouts for every snap from a stint's (MIN_GAMES+1)th game on. Returns, per scored snap, P(pass)
    from the NFL scout plus the caller's overall lean so far ("p_lean"), plus every clue ("p_film"), and plus each
    clue alone ("p_film_<clue>")."""
    y = d["PASS"].to_numpy(float)
    X = np.column_stack([np.ones(len(X)), X])            # column 0 = the caller's lean
    groups = np.concatenate([["lean"], groups])
    lam = np.where(groups == "lean", LEAN_LAMBDA, LAMBDA)
    stint = (d["posteam"] + "|" + d[by]).to_numpy()
    game_order = d.groupby(["posteam", by])["Week"].rank(method="dense").to_numpy()
    rows = []
    for s in np.unique(stint):
        idx = np.flatnonzero(stint == s)
        g = game_order[idx]
        sets = ({"lean": groups == "lean", "film": np.ones(X.shape[1], bool)}
                | {f"film_{k}": (groups == k) | (groups == "lean") for k in clues})
        betas: dict[str, np.ndarray | None] = {k: None for k in sets}
        for k_game in range(MIN_GAMES + 1, int(g.max()) + 1):
            train, test = idx[g < k_game], idx[g == k_game]
            if len(test) == 0:
                continue
            out = {"i": test, "p_league": _expit(league_logit[test])}
            for name, cols in sets.items():
                b = fit_logit(X[np.ix_(train, cols)], y[train], league_logit[train], lam[cols], betas[name])
                betas[name] = b
                out[f"p_{name}"] = _expit(league_logit[test] + X[np.ix_(test, cols)] @ b)
            rows.append(pd.DataFrame(out))
    if not rows:
        return pd.DataFrame()
    f = pd.concat(rows).set_index("i").sort_index()
    f.index = d.index[f.index]
    return f


def score(plays: pd.DataFrame, season: int, weeks: tuple[int, int] | None = None, by: str = "caller",
          callers: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Every snap of the season with the clues, plus the scouts' guesses and per-snap savings in bits on scored
    snaps (NaN before a stint has MIN_GAMES of film). Also returns the clues used."""
    d = sample(plays, season, weeks, callers)
    if by == "team":
        d["caller"] = d["posteam"]
    clues = available_clues(d)
    if d.empty or not clues:
        return d, clues
    X, _, groups = design(d, clues)
    league_logit = generic_scout(d, X)
    d["p_league"] = _expit(league_logit)
    f = film_scout(d, X, groups, league_logit, clues)
    d = d.join(f.drop(columns="p_league"))
    y = d["PASS"].to_numpy(float)
    d["scored"] = d["p_film"].notna()
    d["bits_league"] = np.where(d["scored"], _bits(y, d["p_league"].to_numpy()), np.nan)
    d["bits_xpass"] = np.where(d["scored"], _bits(y, d["xpass"].to_numpy()), np.nan)
    d["bits_lean"] = np.where(d["scored"], _bits(y, d["p_lean"].fillna(0.5).to_numpy()), np.nan)
    d["gain_lean"] = d["bits_league"] - d["bits_lean"]
    d["gain_film"] = d["bits_league"] - _bits(y, d["p_film"].fillna(0.5).to_numpy())       # the whole edge
    for k in clues:                                                                         # each clue, beyond lean
        d[f"gain_film_{k}"] = d["bits_lean"] - _bits(y, d[f"p_film_{k}"].fillna(0.5).to_numpy())  # NaN unless scored
    return d, clues


# ---- Tables ---------------------------------------------------------------------------------------------------
def _clustered_se(gain: pd.Series, game: pd.Series) -> float:
    """SE of a mean per-snap gain with snaps in the same game treated as one draw (a game plan repeats)."""
    g = pd.DataFrame({"x": gain, "g": game}).dropna()
    n, games = len(g), g["g"].nunique()
    if games < 2:
        return np.nan
    resid = (g["x"] - g["x"].mean()).groupby(g["g"]).sum()
    return float(np.sqrt((resid ** 2).sum() * games / (games - 1)) / n)


def caller_table(d: pd.DataFrame, clues: list[str]) -> pd.DataFrame:
    """One row per stint (team + play caller): readability, accuracy, each clue's edge, and the index."""
    rows = []
    for (team, caller), g in d.groupby(["posteam", "caller"], sort=False):
        s = g[g["scored"]]
        if s.empty:
            continue
        y = s["PASS"]
        row = {"Team": team, "Play caller": caller, "Weeks": f"{g['Week'].min()}-{g['Week'].max()}",
               "Games": g["game_id"].nunique(), "Snaps": len(g), "Scored snaps": len(s),
               "Pass rate": g["PASS"].mean(), "PROE": g["pass_oe"].mean() / 100, "EPA/play": g["epa"].mean(),
               "Generic bits": s["bits_league"].mean(), "Edge": s["gain_film"].mean(),
               "Lean edge": s["gain_lean"].mean(), "League tells": (s["bits_xpass"] - s["bits_league"]).mean(),
               "Lean": (s["PASS"] - s["p_league"]).mean(),
               "Edge SE": _clustered_se(s["gain_film"], s["game_id"]),
               "Generic accuracy": ((s["p_league"] > 0.5) == (y == 1)).mean(),
               "Film accuracy": ((s["p_film"] > 0.5) == (y == 1)).mean()}
        row["Uncertainty removed"] = row["Edge"] / row["Generic bits"]
        # The film edge in plays: the share of the guessing his film removes, times his scored snaps per game. 2.7 =
        # studying him is worth about 2.7 calls a game a defense would know for sure.
        row["Plays given away"] = row["Uncertainty removed"] * len(s) / s["game_id"].nunique()
        for k in clues:
            row[CLUES[k][0]] = s[f"gain_film_{k}"].mean()
        rows.append(row)
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["Look edge"] = t["Edge"] - t["Lean edge"]
    t["Edge z"] = t["Edge"] / t["Edge SE"]
    ok = t["Scored snaps"] >= MIN_SNAPS
    mu, sd = t.loc[ok, "Edge"].mean(), t.loc[ok, "Edge"].std()
    t["Unpredictability index"] = np.where(ok, 50 - 10 * (t["Edge"] - mu) / (sd if sd > 0 else 1), np.nan)
    t["Profile"] = np.select([~ok, t["Unpredictability index"] >= 60, t["Unpredictability index"] <= 40],
                             ["Too few snaps", "Hard to read", "Readable"], "NFL average")
    t["Biggest tell"] = [_biggest_clue(r, clues) for _, r in t.iterrows()]
    return t.sort_values("Unpredictability index", ascending=False, na_position="last").reset_index(drop=True)


def _biggest_clue(r: pd.Series, clues: list[str]) -> str:
    edges = {CLUES[k][0]: r[CLUES[k][0]] for k in clues}
    best = max(edges, key=lambda k: edges[k] if pd.notna(edges[k]) else -np.inf)
    return best if pd.notna(edges[best]) and edges[best] > 0.002 else "None"


def look_table(d: pd.DataFrame, team: str, caller: str, clues: list[str]) -> pd.DataFrame:
    """Every look (clue value) the stint showed on scored snaps: how often it passed, what the generic scout
    expected, what the film said about that clue going in, and that look's share of the clue's edge (mbits per snap
    of the whole stint; positive = the film read them there, negative = the film misled there). Each look is scored
    with its own clue's film scout, so a look on every snap (1 back, say) can't take credit for another clue's tell;
    within a clue the looks' shares add up to that clue's edge."""
    s = d[(d["posteam"] == team) & (d["caller"] == caller) & d["scored"]]
    if s.empty:
        return pd.DataFrame()
    rows = []
    for k in clues:
        label, (col,) = CLUES[k]
        for look, g in s.groupby(col):
            rows.append({"Clue": label, "Look": look, "Snaps": len(g), "Share": len(g) / len(s),
                         "Pass rate": g["PASS"].mean(), "NFL scout expected": g["p_league"].mean(),
                         "Their film said": g[f"p_film_{k}"].mean(),
                         "Edge share": g[f"gain_film_{k}"].sum() / len(s) * 1000})
    return pd.DataFrame(rows)


def tell_sentence(r: pd.Series) -> str:
    """'3rd & short: throws 81% (an average team here would throw 52%), 37 snaps.'"""
    diff = r["Pass rate"] - r["NFL scout expected"]
    verb = "throws" if diff > 0 else "runs"
    rate = r["Pass rate"] if diff > 0 else 1 - r["Pass rate"]
    exp = r["NFL scout expected"] if diff > 0 else 1 - r["NFL scout expected"]
    return f"{r['Look']}: {verb} {rate:.0%} (an average team would {verb[:-1]} {exp:.0%}), {int(r['Snaps'])} snaps"


def reliability(d: pd.DataFrame) -> float:
    """Split-half reliability of the edge: correlation across stints of odd-game vs even-game edges, stepped up to
    the full sample (Spearman-Brown)."""
    s = d[d["scored"]]
    if s.empty:
        return np.nan
    odd = s.groupby(["posteam", "caller"])["Week"].rank(method="dense") % 2
    h = s.assign(half=odd).groupby(["posteam", "caller", "half"])["gain_film"].mean().unstack()
    n = s.groupby(["posteam", "caller"]).size()
    h = h[n.reindex(h.index) >= MIN_SNAPS].dropna()
    if len(h) < 5:
        return np.nan
    r = h.iloc[:, 0].corr(h.iloc[:, 1])
    return 2 * r / (1 + r)


# ---- Balance score ------------------------------------------------------------------------------------------------
# A second, simpler view: how close to 50/50 a play caller calls each down & distance. For every cell, the squared gap
# from 50/50, minus the part of it a small sample adds by chance; averaged over cells weighted by snaps; flipped so
# 100 = a coin flip in every cell and 0 = the same call every time. Obvious passing downs (3rd/4th & 7+) are left out:
# every team throws there, so they say nothing about the caller and only punish offenses that face more of them.
BALANCE_SKIP = ("3rd & long", "3rd & extra long", "4th & long", "4th & extra long")
BALANCE_CELLS = [f"{dn} & {dist.split(' (')[0].lower()}" for dn in ("1st", "2nd", "3rd", "4th")
                 for dist in dt.DISTANCE_ORDER]
BALANCE_CELLS = [c for c in BALANCE_CELLS if c not in BALANCE_SKIP]


def balance_cells(d: pd.DataFrame) -> pd.DataFrame:
    """Per stint (team + caller) and decision-down cell: snaps, pass rate, and the noise-corrected squared gap from
    50/50, (p - 0.5)^2 - p(1 - p)/(n - 1). The correction is the expected chance part of the gap, so a 4-snap cell
    that went 3-1 isn't scored as a lopsided tendency."""
    c = (d[d["dd"].isin(BALANCE_CELLS)].groupby(["posteam", "caller", "dd"])["PASS"]
         .agg(["mean", "size"]).rename(columns={"mean": "Pass rate", "size": "Snaps"}).reset_index())
    p, n = c["Pass rate"], c["Snaps"]
    c["Gap"] = (p - 0.5) ** 2 - np.where(n > 1, p * (1 - p) / (n - 1).clip(lower=1), 0.0)
    return c


def balance_table(d: pd.DataFrame) -> pd.DataFrame:
    """One row per stint: the balance score, its raw (uncorrected) version, and the most and least balanced
    decision-down cells (8+ snaps)."""
    c = balance_cells(d)
    rows = []
    for (team, caller), g in c.groupby(["posteam", "caller"], sort=False):
        n = g["Snaps"].sum()
        raw = (g["Snaps"] * (g["Pass rate"] - 0.5) ** 2).sum() / n
        big = g[g["Snaps"] >= 8]
        worst = big.loc[(big["Pass rate"] - 0.5).abs().idxmax()] if len(big) else None
        best = big.loc[(big["Pass rate"] - 0.5).abs().idxmin()] if len(big) else None
        rows.append({"Team": team, "Play caller": caller, "Snaps": int(n),
                     "Balance score": 100 * (1 - 4 * (g["Snaps"] * g["Gap"]).sum() / n),
                     "Raw balance score": 100 * (1 - 4 * raw),
                     "Pass rate": (g["Snaps"] * g["Pass rate"]).sum() / n,
                     "Most lopsided": "" if worst is None else f"{worst['dd']}: {worst['Pass rate']:.0%} pass",
                     "Most balanced": "" if best is None else f"{best['dd']}: {best['Pass rate']:.0%} pass"})
    t = pd.DataFrame(rows)
    return t.sort_values("Balance score", ascending=False).reset_index(drop=True) if len(t) else t



# ---- Plain-language summary ("At a glance" view) ----------------------------------------------------------------
TIERS = [(40, "Open Book"), (46, "Readable"), (54, "Average"), (60, "Mixes It Up"), (np.inf, "Poker Face")]
TIER_HELP = {"Open Book": "His film clearly helps a defense.", "Readable": "His film helps a bit more than most.",
             "Average": "About what a typical coach gives away.", "Mixes It Up": "His film barely helps a defense.",
             "Poker Face": "His film doesn't help, or it fools you."}
TELL_MIN_GAP = 0.07       # a tell must differ from the average team by this much...
TELL_MIN_SNAPS = 20       # ...on at least this many snaps
CLUE_WORDS = {"Personnel": "his personnel groupings", "Formation": "his formation", "Motion": "motion",
              "Backfield": "how many backs he lines up", "Down & distance": "down and distance",
              "Field position": "field position", "Score": "the score", "Last call": "what he called the snap before"}


def tier(index: float) -> str:
    return next(name for top, name in TIERS if index <= top)


def certainty(t: pd.DataFrame) -> pd.Series:
    """How sure each caller's tier is: his edge vs the average caller's, in standard errors."""
    z = (t["Edge"] - t["Edge"].mean()).abs() / t["Edge SE"]
    return pd.Series(np.select([z > 2, z > 1], ["Clear", "Probably"], "Too close to call"), index=t.index)


def look_name(clue: str, look: str) -> str:
    """'11' -> '11 personnel', 'Motion' -> 'With motion', 'Leading 4+' -> 'Leading by 4+'."""
    if clue == "Personnel":
        return f"{look} personnel"
    if clue == "Motion":
        return "With motion" if look == "Motion" else "No motion"
    if clue == "Score":
        return {"Leading 4+": "Leading by 4+", "Trailing 4+": "Trailing by 4+", "Within 3": "Within 3 points"}.get(look, look)
    if clue == "Backfield" and look == "Empty":
        return "Empty backfield"
    return str(look)


def look_phrase(clue: str, look: str) -> str:
    """The look as a sentence opener: 'In 11 personnel', 'From shotgun', 'On 3rd & short', 'After a run'."""
    look = str(look)
    if clue == "Personnel":
        return f"In {look} personnel"
    if clue == "Formation":
        return "From under center" if look == "Under center" else f"From {look.lower() if look != 'I-form' else 'the I-form'}"
    if clue == "Motion":
        return "With motion" if look == "Motion" else "Without motion"
    if clue == "Backfield":
        return {"Empty": "From an empty backfield", "1 back": "With 1 back", "2+ backs": "With 2+ backs"}.get(look, look)
    if clue == "Down & distance":
        return f"On {look}"
    if clue == "Field position":
        return {"Red zone": "In the red zone", "Midfield": "Between the 20s", "Own end": "In his own end"}.get(look, look)
    if clue == "Score":
        return {"Leading 4+": "When leading by 4+", "Trailing 4+": "When trailing by 4+",
                "Within 3": "In a one-score-or-less game"}.get(look, look)
    if clue == "Last call":
        return look[0].upper() + look[1:]
    return look


def tells(looks: pd.DataFrame, n: int = 3) -> list[dict]:
    """His biggest tells from look_table: looks where his film read him and he clearly differs from an average team.
    Each: look, call (PASS/RUN, whichever he does more than average), rate (how often he made that call), league."""
    lt = looks[(looks["Snaps"] >= TELL_MIN_SNAPS) & (looks["Edge share"] > 0.3)
               & ((looks["Pass rate"] - looks["NFL scout expected"]).abs() >= TELL_MIN_GAP)]
    out = []
    for _, x in lt.nlargest(n, "Edge share").iterrows():
        passy = x["Pass rate"] >= x["NFL scout expected"]
        out.append({"look": look_name(x["Clue"], x["Look"]), "phrase": look_phrase(x["Clue"], x["Look"]), "call": "PASS" if passy else "RUN",
                    "rate": x["Pass rate"] if passy else 1 - x["Pass rate"],
                    "league": x["NFL scout expected"] if passy else 1 - x["NFL scout expected"], "snaps": int(x["Snaps"])})
    return out


def misleads(looks: pd.DataFrame, n: int = 2) -> list[dict]:
    """Looks where his earlier film pointed one way and his next games went another (by 5+ points)."""
    lt = looks[(looks["Snaps"] >= TELL_MIN_SNAPS) & (looks["Edge share"] < -0.3)
               & ((looks["Pass rate"] - looks["Their film said"]).abs() >= 0.05)]
    return [{"look": look_name(x["Clue"], x["Look"]), "phrase": look_phrase(x["Clue"], x["Look"]), "film": x["Their film said"], "actual": x["Pass rate"],
             "snaps": int(x["Snaps"])} for _, x in lt.nsmallest(n, "Edge share").iterrows()]


def verdict(r: pd.Series, tl: list[dict], ml: list[dict]) -> str:
    """One or two plain sentences on why a caller lands in his tier."""
    last = str(r["Play caller"]).split()[-1] if not str(r["Play caller"]).endswith(" play caller") else r["Team"]
    tell = ""
    if tl:
        t = tl[0]
        verb = "throws" if t["call"] == "PASS" else "runs"
        tell = (f" {t['phrase']}, he {verb} {t['rate']:.0%} of the time; an average team would {verb[:-1]} "
                f"{t['league']:.0%}.")
    name = tier(r["Unpredictability index"])
    if name in ("Open Book", "Readable"):
        if r["Lean edge"] >= r["Look edge"] and abs(r["Lean"]) >= 0.04:
            word = "pass" if r["Lean"] > 0 else "run"
            return (f"{last} {'throws' if r['Lean'] > 0 else 'runs'} more than an average team would in almost every "
                    f"situation, by about {abs(r['Lean']) * 100:.0f} percentage points. A defense that leans {word} "
                    f"against him is right more than it should be.{tell}")
        how = f", most of all {CLUE_WORDS.get(r['Biggest tell'], str(r['Biggest tell']).lower())}" \
            if r["Biggest tell"] != "None" else ""
        return f"{last} tips his calls with what he shows before the snap{how}.{tell}"
    if name == "Average":
        return (f"{last} calls it close to how the league does. His film gives a defense a little, but nothing that "
                f"stands out." + (f" The one thing to watch: {tell[1].lower()}{tell[2:]}" if tell else ""))
    miss = ""
    if ml:
        m = ml[0]
        phrase = m["phrase"][0].lower() + m["phrase"][1:]
        miss = (f" His earlier games said he'd throw {m['film']:.0%} of the time {phrase}; in the games that "
                f"followed he threw {m['actual']:.0%}.")
    if name == "Mixes It Up":
        return f"{last}'s film barely helps. The habits he showed in earlier games mostly didn't hold up.{miss}"
    return f"{last}'s film works against you: what he did in earlier games isn't what he did next.{miss}"
