"""
Shared pieces for the Tendencies page: metrics, crosstabs with NFL comparison columns, table coloring, tendency
badges, and the run/pass split bar. Colors and rules follow the Dowling app's Scout Opposing Offense page, from the
scouted team's side: green = it works for them, red = it doesn't.

League-wide situational averages: any measure can be asked for as "NFL <measure>", which is the same measure for
every team's plays in the same row (same sidebar filters), so "NFL Pass Rate" next to a 3rd & long row is the
league's pass rate on 3rd & long.
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

RUN_C, PASS_C = "#BA7517", "#534AB7"
GREEN, RED, BLUE = "#1D9E75", "#E24B4A", "#185FA5"
GOOD_STYLE = "background-color:#E1F5EE;color:#085041"
BAD_STYLE = "background-color:#FCEBEB;color:#791F1F"
LOW_N_STYLE = "color:#9A9890;background-color:transparent"
CARD = "background:rgba(128,128,128,.08);border-radius:8px;padding:8px 10px"

METRICS = {
    "Plays": lambda g: g["play_id"].count(),
    "Run Rate": lambda g: 1 - g["PASS"].mean(),
    "Pass Rate": lambda g: g["PASS"].mean(),
    "Exp. Pass Rate": lambda g: g["xpass"].mean(),
    "PROE": lambda g: g["pass_oe"].mean() / 100,
    "Avg Yards Gained": lambda g: g["yards_gained"].mean(),
    "Success Rate": lambda g: g["success"].mean(),
    "EPA per Play": lambda g: g["epa"].mean(),
    "Explosive Rate": lambda g: g["explosive"].mean(),
    "Comp %": lambda g: g["completion"].mean(),
    "Conversion Rate": lambda g: g["first_down"].mean(),
    "Conversions": lambda g: g["first_down"].sum(),
    "TD Rate": lambda g: g["touchdown"].mean(),
    "Share": lambda g: g["play_id"].count(),     # made a share of the table in crosstab()
}
PCT = "{:.0%}"


def pct_points(v) -> str:
    """+7% / -12% / +0% (never "-0%")."""
    return "" if pd.isna(v) else f"{int(round(v * 100)) + 0:+d}%"


FORMATS = {"Plays": "{:,.0f}", "Conversions": "{:,.0f}", "Avg Yards Gained": "{:.1f}", "EPA per Play": "{:+.2f}",
           "PROE": pct_points, **{m: PCT for m in ["Run Rate", "Pass Rate", "Exp. Pass Rate", "Success Rate",
                                                   "Explosive Rate", "Comp %", "Conversion Rate", "TD Rate",
                                                   "Share"]}}
METRIC_HELP = {
    "Plays": "Number of plays in this row.",
    "Run Rate": "Share of plays that were designed runs.",
    "Pass Rate": "Share of plays that were passes (dropbacks, including sacks and scrambles).",
    "Exp. Pass Rate": "How often nflfastR's model expects an average NFL team to pass in these exact situations "
                      "(down, distance, field position, score, time, timeouts).",
    "PROE": "Pass rate over expected: pass rate minus the expected pass rate, play by play. +10% = they throw 10 "
            "points more often than an average team would in the same spots.",
    "Avg Yards Gained": "Average yards gained per play.",
    "Success Rate": "Share of plays with positive EPA (nflfastR's definition).",
    "EPA per Play": "Expected points added per play. Above 0 is good for the offense.",
    "Explosive Rate": "Share of plays that were a run of 10+ yards or a pass of 20+ yards.",
    "Comp %": "Completions per pass attempt (sacks and scrambles left out).",
    "Conversion Rate": "Share of plays that gained a first down or scored.",
    "Conversions": "Plays that gained a first down or scored.",
    "TD Rate": "Share of plays that scored a touchdown on the play.",
    "Share": "Share of the table's plays in this row.",
}
for _m in list(METRIC_HELP):
    METRIC_HELP[f"NFL {_m}"] = f"League-wide, same situation and filters: {METRIC_HELP[_m][0].lower()}{METRIC_HELP[_m][1:]}"

# A cell is colored only when it's at least this far from its comparison (the NFL in the same row when the
# table has that column, else the table's own average).
COLOR_THRESHOLDS = {"Avg Yards Gained": 1.0, "Success Rate": 0.05, "EPA per Play": 0.10, "Explosive Rate": 0.04,
                    "Comp %": 0.06, "Conversion Rate": 0.08}
# Results bad on their own: red when a table has no NFL column to compare with (the Dowling offense rule, minus its
# 4.5 yards-a-play line, which would paint nearly every NFL run red).
RED_BELOW = {"EPA per Play": 0.0, "Success Rate": 0.40}
PLAY_RESULTS = ["Avg Yards Gained", "Success Rate", "EPA per Play", "Explosive Rate"]

MIN_PLAYS = 1          # set from the sidebar: tables hide rows with fewer plays
LOW_N_ROWS = 10        # rows with fewer plays are grayed out
COLOR_MIN_PLAYS = 5    # rows with fewer plays keep their numbers but never get green/red
TELL_MIN_N = 10        # a split needs this many plays before it can be called a tendency
MIN_TELL_SHARE = 0.70  # ...and at least this share one way
TENDENCY_GRADES = [(0.70, "Solid"), (0.55, "Likely")]
BADGE = {"Solid": ("tell", "#FCEBEB", "#791F1F"), "Likely": ("lean", "#FAEEDA", "#854F0B"),
         "Small sample": ("small sample", "rgba(128,128,128,.15)", "inherit")}


def theme_ink() -> str:
    """Neutral text color matching Streamlit's light/dark theme (for the SVG drawings)."""
    theme = getattr(getattr(st.context, "theme", None), "type", None)
    return "#FAFAFA" if theme == "dark" else "#31333F"


def theme_bg() -> str:
    theme = getattr(getattr(st.context, "theme", None), "type", None)
    return "#0E1117" if theme == "dark" else "#FFFFFF"


def fmt(col: str) -> str | None:
    return FORMATS.get(col[4:] if col.startswith("NFL ") else col)


# ---- Tables ------------------------------------------------------------------------------------------------
def crosstab(d: pd.DataFrame, rows, measures, league: pd.DataFrame | None = None, sort_by_count: bool = False,
             order=None) -> pd.DataFrame:
    """
    Rows = dimension(s), columns = measures. "NFL <measure>" columns come from `league` grouped the same way.
    The table's overall values go in out.attrs["overall"].
    """
    rows = [rows] if isinstance(rows, str) else list(rows)
    d = d.dropna(subset=rows)
    g = d.groupby(rows, observed=True, sort=True)
    out = pd.DataFrame(index=g.size().index)
    lg = league.dropna(subset=rows).groupby(rows, observed=True) if league is not None else None
    for m in measures:
        if m.startswith("NFL "):
            out[m] = METRICS[m[4:]](lg).reindex(out.index) if lg is not None else np.nan
        else:
            out[m] = METRICS[m](g)
    if "Share" in out and out["Share"].sum():
        out["Share"] = out["Share"] / out["Share"].sum()
    if order is not None:
        out = out.reindex([o for o in order if o in out.index])
    elif sort_by_count:
        out = out.loc[g.size().sort_values(ascending=False, kind="stable").index]
    out.attrs["overall"] = {m: float(METRICS[m](d)) for m in measures if m in METRICS and m != "Share"} if len(d) else {}
    return out


def style_table(table: pd.DataFrame, gray_low_n: bool = True):
    """
    Color from the offense's side. When the table has the NFL column for a metric, each cell is compared to the NFL
    in the same situation: green = clearly better, red = clearly worse. Without one, the Dowling offense rule: green
    = clearly above the table's average, red only for results bad on their own (RED_BELOW). Rows with fewer than
    COLOR_MIN_PLAYS plays are never colored.
    """
    overall = table.attrs.get("overall", {})
    formats = {c: fmt(c) for c in table.columns if fmt(c)}
    few = (table["Plays"] < COLOR_MIN_PLAYS).to_numpy() if "Plays" in table else np.zeros(len(table), bool)

    def color_col(col):
        thr = COLOR_THRESHOLDS.get(col.name)
        if thr is None:
            return [""] * len(col)
        has_nfl = f"NFL {col.name}" in table
        refs = table[f"NFL {col.name}"] if has_nfl else pd.Series(overall.get(col.name, np.nan), index=table.index)
        out = []
        for v, ref, small in zip(col, refs, few):
            if pd.isna(v) or small:
                out.append("")
            elif has_nfl:
                out.append("" if pd.isna(ref) else GOOD_STYLE if v >= ref + thr else BAD_STYLE if v <= ref - thr else "")
            elif col.name in RED_BELOW and v < RED_BELOW[col.name]:
                out.append(BAD_STYLE)
            elif not pd.isna(ref) and v >= ref + thr:
                out.append(GOOD_STYLE)
            else:
                out.append("")
        return out

    sty = table.style.format(formats, na_rep="").apply(color_col, axis=0)
    if gray_low_n and "Plays" in table.columns:
        low = (table["Plays"] < LOW_N_ROWS).to_numpy()
        sty = sty.apply(lambda row: [LOW_N_STYLE if low[table.index.get_loc(row.name)] else "" for _ in row], axis=1)
    return sty


def show_table(title: str, table: pd.DataFrame, caption: str | None = None, gray_low_n: bool = True) -> None:
    st.markdown(f"**{title}**")
    overall = table.attrs.get("overall", {})
    if "Plays" in table.columns and MIN_PLAYS > 1:
        table = table[table["Plays"] >= MIN_PLAYS]
        table.attrs["overall"] = overall
    if table.empty:
        st.info("No plays match the current filters.")
        return
    config = {c: st.column_config.Column(help=METRIC_HELP[c]) for c in table.columns if c in METRIC_HELP}
    st.dataframe(style_table(table, gray_low_n), width="stretch", column_config=config)
    if caption:
        st.caption(caption)


def charting_note(d: pd.DataFrame, col: str, label: str) -> bool:
    """Explain missing FTN charting columns. Returns False when nothing here has the column."""
    have = int(d[col].notna().sum()) if len(d) else 0
    if have == 0:
        st.info(f"No {label} data for these plays yet. nflverse publishes FTN participation charting (formation, "
                f"personnel, box count, coverage) after the season, so the season in progress doesn't have it.")
        return False
    if have < 0.9 * len(d):
        st.caption(f"{have:,} of {len(d):,} plays here have {label} data, so this only covers those plays.")
    return True


# ---- Tendency badges and split bars ------------------------------------------------------------------------
def wilson_lower(k: float, n: float, z: float = 1.645) -> float:
    if n <= 0:
        return 0.0
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((center - margin) / denom)


def tendency_grade(k: float, n: float) -> str:
    """How much to trust k of n plays going the majority way (Wilson lower bound, 90%)."""
    lb = wilson_lower(k, n)
    return next((name for cut, name in TENDENCY_GRADES if lb >= cut), "Small sample")


def badge(k: int, n: int) -> str:
    if n < TELL_MIN_N or k / n < MIN_TELL_SHARE:
        return ""
    label, bg, fg = BADGE[tendency_grade(k, n)]
    return (f'<span style="font-size:11px;padding:1px 7px;border-radius:999px;background:{bg};color:{fg};'
            f'margin-left:6px;white-space:nowrap">{label}</span>')


def split_bar(run_share: float, height: int = 8) -> str:
    """Run (orange) / pass (purple) bar."""
    return (f'<div style="display:flex;height:{height}px;border-radius:4px;overflow:hidden;margin:6px 0">'
            f'<span style="width:{run_share * 100:.0f}%;background:{RUN_C}"></span>'
            f'<span style="width:{(1 - run_share) * 100:.0f}%;background:{PASS_C}"></span></div>')


LEGEND = (f'<div style="display:flex;flex-wrap:wrap;gap:14px;font-size:12px;margin:2px 0 8px;align-items:center">'
          f'<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:{RUN_C};'
          f'margin-right:4px"></span>Run</span>'
          f'<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:{PASS_C};'
          f'margin-right:4px"></span>Pass</span>'
          f'<span style="opacity:.7">tell / lean / small sample = how much to trust a {MIN_TELL_SHARE:.0%}+ split'
          f'</span></div>')


def proe_text(v: float) -> str:
    if pd.isna(v):
        return ""
    color = PASS_C if v > 0.05 else RUN_C if v < -0.05 else "inherit"
    return f'<span style="color:{color};font-weight:600">PROE {pct_points(v)}</span>'


CHIP_MIN_PROE = 0.05   # a PROE chip shows only when the team is at least this far from an average team...


def proe_chip(v: float) -> str:
    """'PROE +14%' as a colored pill: purple = throws more than expected, orange = runs more."""
    if pd.isna(v):
        return ""
    return (f'<span style="font-size:11px;font-weight:700;padding:1px 8px;border-radius:999px;color:#fff;'
            f'white-space:nowrap;background:{PASS_C if v > 0 else RUN_C}">PROE {pct_points(v)}</span>')


# ---- Plays ---------------------------------------------------------------------------------------------------
def _spot(y100) -> str:
    if pd.isna(y100):
        return ""
    y100 = int(y100)
    return "Midfield" if y100 == 50 else f"Own {100 - y100}" if y100 > 50 else f"Opp {y100}"


def plays_table(d: pd.DataFrame) -> pd.DataFrame:
    """One row per play, outcome first (as in the Dowling app), then a readable situation."""
    out = pd.DataFrame(index=d.index)
    out["EPA"] = d["epa"]
    out["Yards"] = [str(int(x)) if pd.notna(x) else "" for x in d["yards_gained"]]
    down = [f"{DOWN(n)} & {int(x)}" if pd.notna(n) else "" for n, x in zip(d["DN"], d["ydstogo"])]
    out["Situation"] = [f"Q{int(q)} · {dd} · {_spot(y)}" for q, dd, y in zip(d["qtr"], down, d["yardline_100"])]
    out["Type"] = d["PLAY TYPE"]
    out["Formation"] = d["Formation"]
    if d["Personnel"].notna().any():
        out["Personnel"] = d["Personnel"].fillna("")
    out["Week"] = d["Season"].astype(str) + " W" + d["Week"].astype(str)
    out["Opponent"] = d["defteam"]
    out["Play"] = d["desc"]
    return out


def DOWN(n) -> str:  # noqa: N802 - reads like the label it makes
    return {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(int(n), "") if pd.notna(n) else ""


def show_plays(title: str, d: pd.DataFrame, color: bool = True) -> None:
    st.markdown(f"**{title}**")
    if d.empty:
        st.info("No plays.")
        return
    t = plays_table(d)
    sty = t.style.format({"EPA": "{:+.2f}"}, na_rep="")
    if color:
        sty = sty.apply(lambda c: [(GOOD_STYLE if x >= 0 else BAD_STYLE) if pd.notna(x) else "" for x in c], subset=["EPA"])
    st.dataframe(sty, width="stretch", hide_index=True)


# ---- Takeaways and the EPA vs success chart (same as the Dowling app) --------------------------------------
TAKEAWAY_MIN_PLAYS = 10


def render_takeaways(table: pd.DataFrame, noun: str) -> None:
    """Three cards: most used, best, and struggling (best and struggling need TAKEAWAY_MIN_PLAYS+ plays)."""
    if table.empty or not {"Plays", "EPA per Play"} <= set(table.columns):
        return
    t = table.dropna(subset=["EPA per Play"])
    if t.empty:
        return
    name = lambda idx: " · ".join(map(str, idx)) if isinstance(idx, tuple) else str(idx)
    top = t["Plays"].idxmax()
    cards = [("Most used", name(top), f"{t.loc[top, 'Plays'] / table['Plays'].sum():.0%} of plays · "
                                      f"{t.loc[top, 'EPA per Play']:+.2f} EPA")]
    sized = t[t["Plays"] >= TAKEAWAY_MIN_PLAYS]
    if not sized.empty:
        best, worst = sized["EPA per Play"].idxmax(), sized["EPA per Play"].idxmin()
        cards.append((f"Best ({TAKEAWAY_MIN_PLAYS}+ plays)", name(best),
                      f"{sized.loc[best, 'EPA per Play']:+.2f} EPA on {int(sized.loc[best, 'Plays'])} plays"))
        avg = table.attrs.get("overall", {}).get("EPA per Play", t["EPA per Play"].mean())
        if worst != best and sized.loc[worst, "EPA per Play"] < avg:
            cards.append(("Struggling", name(worst),
                          f"{sized.loc[worst, 'EPA per Play']:+.2f} EPA on {int(sized.loc[worst, 'Plays'])} plays"))
    else:
        cards.append((f"Best ({TAKEAWAY_MIN_PLAYS}+ plays)", "–", f"No {noun} has {TAKEAWAY_MIN_PLAYS}+ plays yet"))
    for col, (label, value, sub) in zip(st.columns(len(cards)), cards):
        with col.container(border=True):
            st.caption(label)
            st.markdown(f"**{value}**")
            st.caption(sub)


def render_usage_scatter(table: pd.DataFrame, title: str, noun: str, min_plays: int = 5,
                         nfl_epa: float | None = None) -> None:
    """Success rate (x) vs EPA per play (y), dot size = plays, dashed lines = this team's averages. A gray dotted
    line marks the NFL's EPA per play over the same plays, when given."""
    if table.empty or not {"Plays", "EPA per Play", "Success Rate"} <= set(table.columns):
        return
    data = table.reset_index()
    if isinstance(table.index, pd.MultiIndex):
        data[noun] = data[list(table.index.names)].astype(str).agg(" · ".join, axis=1)
    else:
        data = data.rename(columns={data.columns[0]: noun})
    data = data.dropna(subset=["EPA per Play", "Success Rate"])
    data = data[data["Plays"] >= min_plays]
    if data.empty:
        return
    overall = table.attrs.get("overall", {})
    avg_epa, avg_sr = overall.get("EPA per Play", 0.0), overall.get("Success Rate", 0.45)
    # Red = negative EPA only: a success-rate floor would paint every deep-ball row red however well it works.
    cats = ["Clearly above their average", "In between", "Negative EPA"]
    data["vs_avg"] = np.select([data["EPA per Play"] < 0, data["EPA per Play"] >= avg_epa + 0.10], [cats[2], cats[0]],
                               default=cats[1])
    data["label"] = np.where(data["Plays"] >= TAKEAWAY_MIN_PLAYS, data[noun], "")
    st.markdown(f"**{title}**")
    tooltip = [alt.Tooltip(f"{noun}:N"), alt.Tooltip("EPA per Play:Q", format="+.2f"),
               alt.Tooltip("Success Rate:Q", format=".0%"), alt.Tooltip("Plays:Q")]
    base = alt.Chart(data).encode(
        x=alt.X("Success Rate:Q", scale=alt.Scale(zero=False, padding=20), axis=alt.Axis(format=".0%"),
                title="Success rate"),
        y=alt.Y("EPA per Play:Q", scale=alt.Scale(zero=False, padding=20), axis=alt.Axis(format="+.2f"),
                title="EPA per play"))
    pts = base.mark_circle(stroke="white", strokeWidth=1).encode(
        size=alt.Size("Plays:Q", scale=alt.Scale(range=[40, 900]), legend=alt.Legend(orient="right")),
        color=alt.Color("vs_avg:N", title=None, scale=alt.Scale(domain=cats, range=[GREEN, "#B4B2A9", RED]),
                        legend=alt.Legend(orient="top")), tooltip=tooltip)
    labels = base.mark_text(align="left", dx=10, fontSize=12).encode(text="label", tooltip=tooltip)

    def rule(dash, **enc):
        return alt.Chart(pd.DataFrame({k: [v] for k, v in enc.items()})).mark_rule(
            strokeDash=dash, color="#888780").encode(
            **{k: alt.X(f"{k}:Q", title="Success rate") if k == "x" else alt.Y(f"{k}:Q", title="EPA per play")
               for k in enc})

    layers = [rule([5, 4], y=avg_epa), rule([5, 4], x=avg_sr)]
    if nfl_epa is not None and not pd.isna(nfl_epa):
        layers.append(rule([1, 3], y=nfl_epa))
    st.altair_chart(alt.layer(*layers, pts, labels).properties(height=360), width="stretch")
    nfl_txt = f" Dotted line: the NFL's EPA per play on the same kind of plays ({nfl_epa:+.2f})." if nfl_epa is not None else ""
    st.caption(f"Dashed lines: this team's average EPA ({avg_epa:+.2f}) and success rate ({avg_sr:.0%}).{nfl_txt} "
               f"Bigger dots = more plays; rows with {TAKEAWAY_MIN_PLAYS}+ plays are labeled.")
