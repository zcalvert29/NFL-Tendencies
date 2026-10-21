"""
Red zone tab: trips and how they ended, then how the offense calls it as the field shrinks, each next to the NFL's
numbers in the same zone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import common as cm

RZ_BANDS = [("Red zone (11–20)", 11, 20), ("Inside the 10 (6–10)", 6, 10), ("Goal line (1–5)", 1, 5)]
RZ_RESULTS = [("Touchdown", "#1D9E75"), ("Field goal", "#E8B923"), ("Missed field goal", "#E24B4A"),
              ("Turnover on downs", "#E24B4A"), ("Turnover", "#A32D2D"), ("Opp touchdown", "#501313"),
              ("End of half", "#888780"), ("Safety", "#501313")]
POINTS = {"Touchdown": 7, "Field goal": 3}
RZ_MEASURES = ["Pass Rate", "NFL Pass Rate", "PROE", "Success Rate", "NFL Success Rate", "EPA per Play",
               "NFL EPA per Play", "TD Rate", "Plays"]


def trips(snaps: pd.DataFrame) -> pd.DataFrame:
    """One row per drive with a snap at the 20 or closer: how it ended and the points (a touchdown counts 7)."""
    g = snaps.groupby(["game_id", "fixed_drive"])
    out = pd.DataFrame({"closest": g["yardline_100"].min(), "result": g["fixed_drive_result"].first(),
                        "posteam": g["posteam"].first()})
    out = out[out["closest"] <= 20].reset_index()
    out["points"] = out["result"].map(POINTS).fillna(0)
    return out


def _kpis(t: pd.DataFrame, games: int) -> dict:
    return {"Trips": len(t), "Touchdown rate": (t["result"] == "Touchdown").mean(), "Scoring rate": (t["points"] > 0).mean(),
            "Points / trip": t["points"].mean(), "Trips per game": len(t) / games if games else np.nan}


def render_trips(team_games: pd.DataFrame, league_games: pd.DataFrame, team: str) -> None:
    st.markdown(f"**{team} red zone trips**")
    t = trips(team_games)
    if t.empty:
        st.info("No drives reached the red zone in the selected games.")
        return
    k = _kpis(t, team_games["game_id"].nunique())
    lt = trips(league_games)
    lk = _kpis(lt, league_games.groupby("posteam")["game_id"].nunique().sum())
    fmt = {"Trips": "{:.0f}", "Touchdown rate": "{:.0%}", "Scoring rate": "{:.0%}", "Points / trip": "{:.1f}",
           "Trips per game": "{:.1f}"}
    for col, (name, v) in zip(st.columns(5), k.items()):
        delta = None
        if name != "Trips" and not pd.isna(lk[name]):
            diff = v - lk[name]
            delta = (f"{diff * 100:+.0f} pts" if fmt[name] == "{:.0%}" else f"{diff:+.1f}") + " vs NFL"
        col.metric(name, fmt[name].format(v), delta,
                   help={"Trips": "Drives with a snap at or inside the opponent's 20.",
                         "Points / trip": "Touchdowns counted as 7."}.get(name))
    res = t["result"].value_counts()
    known = [r for r, _ in RZ_RESULTS if r in res]
    other = int(res.drop(known).sum())
    pieces = [(r, c, res[r]) for r, c in RZ_RESULTS if r in res] + ([("Other", "#B4B2A9", other)] if other else [])
    n = len(t)
    seg = "".join(f'<span title="{r}: {k}" style="width:{k / n * 100:.1f}%;background:{c}"></span>' for r, c, k in pieces)
    leg = "".join(f'<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:{c};'
                  f'margin-right:4px"></span>{r} {k}</span>' for r, c, k in pieces)
    st.html(f'<div style="display:flex;height:14px;border-radius:4px;overflow:hidden;margin:4px 0 6px">{seg}</div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:12px;font-size:12px">{leg}</div>')
    st.caption(f"\"vs NFL\" = the league's red zone numbers over the same weeks ({lk['Touchdown rate']:.0%} touchdowns, "
               f"{lk['Points / trip']:.1f} points a trip).")


def _band_table(rz: pd.DataFrame, lg_rz: pd.DataFrame) -> pd.DataFrame:
    def band(d):
        return pd.cut(d["yardline_100"], [0, 5, 10, 20], labels=[b[0] for b in RZ_BANDS][::-1]).astype(object)
    return cm.crosstab(rz.assign(Zone=band(rz)), "Zone", RZ_MEASURES, lg_rz.assign(Zone=band(lg_rz)),
                       order=[b[0] for b in RZ_BANDS])


def _call(d: pd.DataFrame) -> pd.Series:
    """'Run left end', 'Pass middle', ...: nflfastR has no play calls, so direction stands in (as in the Dowling
    app when OFF PLAY isn't tagged)."""
    run = ("Run " + d["Run Lane"].str.lower()).fillna("Run (no direction, mostly QB runs)")
    pas = ("Pass " + d["Pass Direction"].str.lower()).fillna("Sack, scramble, or no target")
    return pd.Series(np.where(d["PASS"] == 1, pas, run), index=d.index)


def render(team_d, league_d, team_games, league_games, team) -> None:
    render_trips(team_games, league_games, team)
    rz, lg_rz = team_d[team_d["yardline_100"] <= 20], league_d[league_d["yardline_100"] <= 20]
    if rz.empty:
        return
    cm.show_table(f"{team}: how it's called inside the 20", _band_table(rz, lg_rz), gray_low_n=False)
    cm.show_table("Formations inside the 20", cm.crosstab(rz, "Formation", RZ_MEASURES, lg_rz, sort_by_count=True),
                  gray_low_n=False)
    if rz["Personnel"].notna().any():
        cm.show_table("Personnel inside the 20", cm.crosstab(rz, "Personnel", RZ_MEASURES, lg_rz, sort_by_count=True),
                      gray_low_n=False)
    for where, lo, hi in (("between the 11 and 20 yard line", 11, 20), ("inside the 10 yard line", 1, 10)):
        b, lb = rz[rz["yardline_100"].between(lo, hi)], lg_rz[lg_rz["yardline_100"].between(lo, hi)]
        title = f"Run/pass by direction {where}"
        if b.empty:
            st.markdown(f"**{title}**")
            st.info("No plays here with the current filters.")
            continue
        t = cm.crosstab(b.assign(Call=_call(b)), "Call", ["Share", "Success Rate", "NFL Success Rate", "EPA per Play",
                                                           "NFL EPA per Play", "TD Rate", "Plays"],
                        lb.assign(Call=_call(lb)), sort_by_count=True)
        cm.show_table(title, t, gray_low_n=False)
    st.caption("Trips use whole drives from the selected seasons and weeks; the zone, formation, and direction tables "
               "follow every sidebar filter. TD Rate = share of snaps that scored right there. NFL columns = every "
               "team's plays in the same zone.")
