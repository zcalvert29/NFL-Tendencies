"""
Sequencing tab: what the offense calls after each kind of snap (the Dowling app's "What they call after..." view),
with PROE and the NFL's pass rate after the same kind of snap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import common as cm

SEQUENCE_ROWS = [
    ("First snap of a drive", lambda d: d["first_of_drive"]),
    ("After a run", lambda d: d["prev_type"] == "Run"),
    ("After a pass", lambda d: d["prev_type"] == "Pass"),
    ("After a completion", lambda d: (d["prev_type"] == "Pass") & (d["prev_complete"] == 1)),
    ("After an incompletion", lambda d: (d["prev_type"] == "Pass") & (d["prev_complete"] == 0)),
    ("After a gain of 10+", lambda d: d["prev_gain"] >= 10),
    ("After no gain or a loss", lambda d: d["prev_type"].notna() & (d["prev_gain"] <= 0)),
    ("After a play that worked (EPA > 0)", lambda d: d["prev_success"] == 1),
    ("After a play that didn't", lambda d: d["prev_success"] == 0),
    ("New set of downs (not drive start)", lambda d: (d["DN"] == 1) & ~d["first_of_drive"]),
]
COLS = "minmax(170px,1.4fr) 56px minmax(150px,2fr) minmax(120px,1fr) 80px 64px 64px"


def two_prop_z(k1, n1, k2, n2) -> float:
    if not n1 or not n2:
        return 0.0
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) if 0 < p < 1 else 0
    return (k1 / n1 - k2 / n2) / se if se else 0.0


def sequence_table(team_d: pd.DataFrame, league_d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, rule in SEQUENCE_ROWS:
        s = team_d[rule(team_d).fillna(False).astype(bool)]
        lg = league_d[rule(league_d).fillna(False).astype(bool)]
        if s.empty:
            continue
        rows.append({"After": name, "Plays": len(s), "Pass": s["PASS"].mean(), "NFL pass": lg["PASS"].mean(),
                     "PROE": s["pass_oe"].mean() / 100, "Success": s["success"].mean(), "EPA": s["epa"].mean()})
    return pd.DataFrame(rows)


def sequence_html(t: pd.DataFrame, base_pass: float) -> str:
    rows = []
    for _, r in t.iterrows():
        n, pr = int(r["Plays"]), r["Pass"]
        k = round(max(pr, 1 - pr) * n)
        diff = pr - base_pass
        diff_txt = (f'<span style="opacity:.65">{diff * 100:+.0f} pts pass vs usual</span>' if abs(diff) >= 0.05
                    else '<span style="opacity:.4">about usual</span>')
        rows.append(
            f'<div style="display:grid;grid-template-columns:{COLS};gap:10px;align-items:center;padding:6px 0;'
            f'border-top:0.5px solid rgba(128,128,128,.25);font-size:13px{";opacity:.55" if n < cm.TELL_MIN_N else ""}">'
            f'<div>{r["After"]}</div><div style="text-align:right;opacity:.7">{n}</div>'
            f'<div>{cm.split_bar(1 - pr)}<div style="font-size:11px;opacity:.75">'
            f'Run {1 - pr:.0%} · Pass {pr:.0%} · NFL {r["NFL pass"]:.0%}{cm.badge(k, n)}</div></div>'
            f'<div style="font-size:12px">{diff_txt}</div>'
            f'<div style="font-size:12px">{cm.proe_text(r["PROE"])}</div>'
            f'<div style="text-align:right">{r["Success"]:.0%}</div><div style="text-align:right">{r["EPA"]:+.2f}</div></div>')
    head = (f'<div style="display:grid;grid-template-columns:{COLS};gap:10px;font-size:11px;opacity:.6;padding-bottom:4px">'
            '<div>Previous snap</div><div style="text-align:right">Plays</div><div>Next play</div>'
            '<div>vs. their usual pass rate</div><div>vs. expected</div><div style="text-align:right">Success</div>'
            '<div style="text-align:right">EPA</div></div>')
    return cm.LEGEND + f'<div style="overflow-x:auto"><div style="min-width:760px">{head}{"".join(rows)}</div></div>'


def render(team_d: pd.DataFrame, league_d: pd.DataFrame, team: str) -> None:
    st.markdown(f"**What {team} calls after...**")
    if len(team_d) < 10:
        st.info("Not enough plays match the current filters.")
        return
    t = sequence_table(team_d, league_d)
    base = team_d["PASS"].mean()
    a, b = team_d[team_d["prev_type"] == "Run"], team_d[team_d["prev_type"] == "Pass"]
    if len(a) >= 10 and len(b) >= 10:
        pa, pb = a["PASS"].mean(), b["PASS"].mean()
        la, lb = league_d.loc[league_d["prev_type"] == "Run", "PASS"].mean(), league_d.loc[league_d["prev_type"] == "Pass", "PASS"].mean()
        z = two_prop_z(b["PASS"].sum(), len(b), a["PASS"].sum(), len(a))
        verdict = "a real pattern at this sample size" if abs(z) >= 1.645 else "could still be noise at this sample size"
        st.html(f'<div style="font-size:14px;margin-bottom:6px">Throws <b>{pb:.0%}</b> after a pass vs <b>{pa:.0%}</b> '
                f'after a run ({len(b)} and {len(a)} plays): {verdict}. The NFL throws {lb:.0%} after a pass and '
                f'{la:.0%} after a run in the same filters.</div>')
    st.html(sequence_html(t, base))
    st.caption(f"\"Usual\" = {base:.0%} pass across all these plays. Previous snap = the run or pass right before, in "
               f"the same drive (penalties and kicks skipped). PROE = pass rate over expected on the next play. Faded "
               f"rows have under {cm.TELL_MIN_N} plays.")

    # Run/pass after run/pass, by down: where the pattern actually shows up.
    seq = team_d[team_d["prev_type"].notna() & team_d["DN"].isin([1, 2, 3])].assign(
        Previous=lambda d: "After a " + d["prev_type"].str.lower(),
        Down=lambda d: d["DN"].map({1: "1st", 2: "2nd", 3: "3rd"}))
    lg = league_d[league_d["prev_type"].notna() & league_d["DN"].isin([1, 2, 3])].assign(
        Previous=lambda d: "After a " + d["prev_type"].str.lower(),
        Down=lambda d: d["DN"].map({1: "1st", 2: "2nd", 3: "3rd"}))
    tbl = cm.crosstab(seq, ["Down", "Previous"], ["Pass Rate", "NFL Pass Rate", "PROE", "Success Rate", "EPA per Play",
                                                  "Plays"], lg)
    cm.show_table(f"{team} by down after a run or a pass", tbl)
