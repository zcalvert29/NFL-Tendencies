"""
Crosstab tables, their coloring, takeaway cards, and the EPA vs success "usage" scatter. Same rules as the
Dowling app: green = clearly better than the table's average, red only for results that are bad on their own
(offense view), gray rows = small samples.
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ---- Metrics ---------------------------------------------------------------------------------------------
METRICS = {
    "Plays": lambda g: g["play_id"].count(),
    "Share": lambda g: g["play_id"].count(),           # turned into a share of the table in crosstab()
    "Pass Rate": lambda g: g["is_pass"].mean(),
    "Run Rate": lambda g: g["is_run"].mean(),
    "PROE": lambda g: g["pass_oe"].mean() / 100,
    "Yards per Play": lambda g: g["yards_gained"].mean(),
    "Success Rate": lambda g: g["success"].mean(),
    "EPA per Play": lambda g: g["epa"].mean(),
    "Explosive Rate": lambda g: g["explosive"].mean(),
    "Comp %": lambda g: g["completion"].mean(),
    "CPOE": lambda g: g["cpoe"].mean(),
    "aDOT": lambda g: g["air_yards"].mean(),
    "Sack Rate": lambda g: g["sack"].mean(),
    "Pressure Rate": lambda g: g["pressure"].mean(),
    "Blitz Rate": lambda g: g["blitz"].mean(),
    "Turnover Rate": lambda g: g["turnover"].mean(),
    "1st Down Rate": lambda g: g["first_down"].mean(),
    "Conversion Rate": lambda g: g["first_down"].mean(),
}

PCT = "{:.0%}"
FORMATS = {
    "Plays": "{:,.0f}", "Yards per Play": "{:.1f}", "EPA per Play": "{:+.2f}", "aDOT": "{:.1f}",
    "CPOE": "{:+.1f}", "PROE": "{:+.0%}", "Share": "{:.0%}",
    **{m: PCT for m in ["Pass Rate", "Run Rate", "Success Rate", "Explosive Rate", "Comp %", "Sack Rate",
                        "Pressure Rate", "Blitz Rate", "Turnover Rate", "1st Down Rate", "Conversion Rate",
                        "NFL pass rate", "NFL conv rate", "NFL success"]},
    "NFL EPA": "{:+.2f}",
}
METRIC_HELP = {
    "Plays": "Number of plays in this row.",
    "Share": "Share of the table's plays in this row.",
    "Pass Rate": "Share of plays that were passes (dropbacks, including sacks and scrambles).",
    "Run Rate": "Share of plays that were designed runs.",
    "PROE": "Pass rate over expected: how much more often they pass than nflfastR's model expects an average "
            "team to in the same down, distance, field position, score, and time.",
    "Yards per Play": "Average yards gained per play.",
    "Success Rate": "Share of plays with positive EPA (nflfastR's definition).",
    "EPA per Play": "Expected points added: how many points the average play was worth, based on down, distance, "
                    "field position, and time. Above 0 is good for the offense.",
    "Explosive Rate": "Share of plays that were a run of 10+ yards or a pass of 20+ yards.",
    "Comp %": "Completions per pass attempt (sacks and scrambles left out).",
    "CPOE": "Completion % over expected, in percentage points: completions vs how hard the throws were.",
    "aDOT": "Average depth of target: air yards per throw.",
    "Sack Rate": "Sacks per dropback.",
    "Pressure Rate": "Share of dropbacks where the QB was pressured (FTN charting).",
    "Blitz Rate": "Share of dropbacks with 5+ pass rushers (FTN charting).",
    "Turnover Rate": "Interceptions and lost fumbles per play.",
    "1st Down Rate": "Share of plays that gained a first down.",
    "Conversion Rate": "Share of plays that picked up the first down (or scored).",
    "NFL pass rate": "The league's pass rate in the same situation.",
    "NFL conv rate": "The league's conversion rate in the same situation.",
    "NFL EPA": "The league's EPA per play in the same situation.",
}

# A cell is colored only when it's at least this far from the table's average.
COLOR_THRESHOLDS = {
    "Yards per Play": 1.0, "Success Rate": 0.05, "EPA per Play": 0.10, "Explosive Rate": 0.03,
    "Comp %": 0.06, "CPOE": 3.0, "1st Down Rate": 0.06, "Conversion Rate": 0.08, "Sack Rate": 0.03, "Turnover Rate": 0.015,
    "Pressure Rate": 0.06,
}
# Metrics where a high number is bad for the offense.
LOW_IS_GOOD = {"Sack Rate", "Turnover Rate", "Pressure Rate"}
GOOD_STYLE = "background-color:#E1F5EE;color:#085041"
BAD_STYLE = "background-color:#FCEBEB;color:#791F1F"
OFFENSE_RED_BELOW = {"EPA per Play": 0.0, "Yards per Play": 4.5, "Success Rate": 0.40}
LOW_N_STYLE = "color:#9A9890;background-color:transparent"

PLAY_RESULTS = ["Plays", "Yards per Play", "Success Rate", "EPA per Play", "Explosive Rate"]
MIN_PLAYS = 1        # set from the sidebar: tables hide rows with fewer plays
LOW_N_ROWS = 20      # NFL samples are bigger than high school ones, so the gray-out line is higher
TAKEAWAY_MIN_PLAYS = 25


def crosstab(df: pd.DataFrame, rows, measures, sort_by_count: bool = False, order=None) -> pd.DataFrame:
    """Rows = dimension(s), columns = measures. The table's overall values go in out.attrs["overall"]."""
    rows = [rows] if isinstance(rows, str) else list(rows)
    df = df.dropna(subset=rows)
    g = df.groupby(rows, observed=True, sort=True)
    out = pd.DataFrame({m: METRICS[m](g) for m in measures})
    if "Share" in out:
        out["Share"] = out["Share"] / out["Share"].sum() if out["Share"].sum() else np.nan
    if order is not None:
        out = out.reindex([o for o in order if o in out.index])
    elif sort_by_count:
        out = out.loc[g.size().sort_values(ascending=False, kind="stable").index]
    out.attrs["overall"] = {m: float(METRICS[m](df)) for m in measures if m != "Share"} if len(df) else {}
    return out


def column_format(col: str) -> str | None:
    """Format for a column: its METRICS format, else a guess from the name (for comparison columns)."""
    if col in FORMATS:
        return FORMATS[col]
    low = col.lower()
    if "epa" in low:
        return "{:+.2f}"
    if any(w in low for w in ("rate", "share", "%")):
        return "{:.0%}"
    if any(w in low for w in ("faced", "throws", "targets", "dropbacks")):
        return "{:,.0f}"
    return None


def style_table(table: pd.DataFrame, good_high: bool = True, overall: dict | None = None, gray_low_n: bool = True):
    """
    Color cells green (good for the team being studied) or red (bad) when they're clearly away from the
    table's average. good_high=False flips the colors, for a defense's view of the offense it faced.
    """
    overall = table.attrs.get("overall", {}) if overall is None else overall
    fmt = {c: column_format(c) for c in table.columns if column_format(c)}

    def color_col(col):
        thr, ref = COLOR_THRESHOLDS.get(col.name), overall.get(col.name)
        if thr is None or ref is None or pd.isna(ref):
            return [""] * len(col)
        up_good = good_high != (col.name in LOW_IS_GOOD)
        styles = []
        for v in col:
            if pd.isna(v):
                styles.append("")
            elif good_high and col.name in OFFENSE_RED_BELOW:
                # Offense rule: red only for results bad on their own; green at/above the cutoff.
                if v < OFFENSE_RED_BELOW[col.name]:
                    styles.append(BAD_STYLE)
                elif v >= ref + thr:
                    styles.append(GOOD_STYLE)
                else:
                    styles.append("")
            elif abs(v - ref) < thr:
                styles.append("")
            else:
                styles.append(GOOD_STYLE if (v > ref) == up_good else BAD_STYLE)
        return styles

    sty = table.style.format(fmt, na_rep="").apply(color_col, axis=0)
    if gray_low_n and "Plays" in table.columns:
        low = (table["Plays"] < LOW_N_ROWS).to_numpy()
        sty = sty.apply(lambda row: [LOW_N_STYLE if low[table.index.get_loc(row.name)] else "" for _ in row], axis=1)
    return sty


def show_table(title: str, table: pd.DataFrame, good_high: bool = True, caption: str | None = None) -> None:
    st.markdown(f"**{title}**")
    overall = table.attrs.get("overall", {})
    if "Plays" in table.columns and MIN_PLAYS > 1:
        table = table[table["Plays"] >= MIN_PLAYS]
        table.attrs["overall"] = overall
    if table.empty:
        st.info("No plays match the current filters.")
        return
    config = {c: st.column_config.Column(help=METRIC_HELP[c]) for c in table.columns if c in METRIC_HELP}
    st.dataframe(style_table(table, good_high, overall), width="stretch", column_config=config)
    if caption:
        st.caption(caption)


def coverage_note(d: pd.DataFrame, col: str, label: str) -> bool:
    """Caption when a charting column is missing; returns False when there's nothing to show."""
    if d.empty:
        return False
    have = int(d[col].notna().sum())
    if have == 0:
        st.info(f"No {label} data for these plays. nflverse publishes FTN/NGS participation charting "
                f"(personnel, formation, coverage, box, pressure) after the season, so the season in progress "
                f"won't have it yet.")
        return False
    if have / len(d) < 0.9:
        st.caption(f"{have:,} of {len(d):,} plays here have {label} data, so this only covers those plays.")
    return True


# ---- Takeaway cards ---------------------------------------------------------------------------------------
def render_takeaways(table: pd.DataFrame, noun: str, good_high: bool = True) -> None:
    """Three plain-language cards: most used, best, and struggling (TAKEAWAY_MIN_PLAYS+ plays for the last two)."""
    if table.empty or not {"Plays", "EPA per Play"} <= set(table.columns):
        return
    t = table.dropna(subset=["EPA per Play"])
    if t.empty:
        return
    total = table["Plays"].sum()
    avg = table.attrs.get("overall", {}).get("EPA per Play", t["EPA per Play"].mean())
    name = lambda idx: " · ".join(map(str, idx)) if isinstance(idx, tuple) else str(idx)
    top = t["Plays"].idxmax()
    cards = [("Most used", name(top), f"{t.loc[top, 'Plays'] / total:.0%} of plays · {t.loc[top, 'EPA per Play']:+.2f} EPA")]
    sized = t[t["Plays"] >= TAKEAWAY_MIN_PLAYS]
    if not sized.empty:
        best = sized["EPA per Play"].idxmax() if good_high else sized["EPA per Play"].idxmin()
        worst = sized["EPA per Play"].idxmin() if good_high else sized["EPA per Play"].idxmax()
        cards.append((f"Best ({TAKEAWAY_MIN_PLAYS}+ plays)", name(best),
                      f"{sized.loc[best, 'EPA per Play']:+.2f} EPA on {int(sized.loc[best, 'Plays'])} plays"))
        worst_epa = sized.loc[worst, "EPA per Play"]
        if (worst_epa < avg) == good_high and worst != best:
            sr = sized.loc[worst, "Success Rate"] if "Success Rate" in sized else np.nan
            sr_txt = "" if pd.isna(sr) else f" · {sr:.0%} success"
            cards.append(("Struggling", name(worst), f"{worst_epa:+.2f} EPA{sr_txt} on {int(sized.loc[worst, 'Plays'])} plays"))
    else:
        cards.append((f"Best ({TAKEAWAY_MIN_PLAYS}+ plays)", "–", f"No {noun} has {TAKEAWAY_MIN_PLAYS}+ plays yet"))
    for col, (label, value, sub) in zip(st.columns(len(cards)), cards):
        with col.container(border=True):
            st.caption(label)
            st.markdown(f"**{value}**")
            st.caption(sub)


# ---- EPA vs success scatter -------------------------------------------------------------------------------
def render_usage_scatter(table: pd.DataFrame, title: str, good_high: bool = True, noun: str = "Formation",
                         min_plays: int = 10) -> None:
    """Success rate (x) vs EPA per play (y) per row, dot size = plays, dashed lines at the averages."""
    need = {"Plays", "EPA per Play", "Success Rate"}
    if table.empty or not need <= set(table.columns):
        return
    data = table.reset_index()
    if isinstance(table.index, pd.MultiIndex):
        data[noun] = data[list(table.index.names)].astype(str).agg(" · ".join, axis=1)
    else:
        data = data.rename(columns={data.columns[0]: noun})
    data = data.dropna(subset=["EPA per Play", "Success Rate"])
    data[noun] = data[noun].astype(str)
    hidden = int((data["Plays"] < min_plays).sum())
    data = data[data["Plays"] >= min_plays]
    if data.empty:
        return
    overall = table.attrs.get("overall", {})
    avg_epa = overall.get("EPA per Play", data["EPA per Play"].mean())
    avg_sr = overall.get("Success Rate", data["Success Rate"].mean())
    if good_high:
        cats = ["Above average", "In between", "Negative EPA"]
        colors = ["#1D9E75", "#B4B2A9", "#E24B4A"]
        data["vs_avg"] = np.select([data["EPA per Play"] < 0, data["EPA per Play"] >= avg_epa + 0.1],
                                   [cats[2], cats[0]], default=cats[1])
    else:
        cats, colors = ["Good for the defense", "Bad for the defense"], ["#1D9E75", "#E24B4A"]
        data["vs_avg"] = np.where(data["EPA per Play"] < avg_epa, cats[0], cats[1])
    data["Shown label"] = np.where(data["Plays"] >= TAKEAWAY_MIN_PLAYS, data[noun], "")

    st.markdown(f"**{title}**")
    y_lo, y_hi = data["EPA per Play"].min(), data["EPA per Play"].max()
    pad = max((y_hi - y_lo) * 0.12, 0.1)
    y_dom = [min(y_lo, avg_epa) - pad, max(y_hi, avg_epa) + pad]
    x_dom = [max(0.0, min(data["Success Rate"].min(), avg_sr) - 0.05),
             min(1.0, max(data["Success Rate"].max(), avg_sr) + 0.05)]
    base = alt.Chart(data).encode(
        x=alt.X("Success Rate:Q", scale=alt.Scale(domain=x_dom), title="Success rate", axis=alt.Axis(format=".0%")),
        y=alt.Y("EPA per Play:Q", scale=alt.Scale(domain=y_dom), title="EPA per play", axis=alt.Axis(format="+.2f")),
    )
    tooltip = [alt.Tooltip(f"{noun}:N"), alt.Tooltip("EPA per Play:Q", format="+.2f"),
               alt.Tooltip("Success Rate:Q", format=".0%"), alt.Tooltip("Plays:Q")]
    points = base.mark_circle(stroke="white", strokeWidth=1).encode(
        size=alt.Size("Plays:Q", scale=alt.Scale(range=[40, 900]), legend=alt.Legend(title="Plays", orient="right")),
        color=alt.Color("vs_avg:N", title=None, scale=alt.Scale(domain=cats, range=colors),
                        legend=alt.Legend(orient="top")),
        tooltip=tooltip,
    )
    labels = base.mark_text(align="left", dx=10, fontSize=12).encode(text="Shown label", tooltip=tooltip)
    rule = lambda **enc: alt.Chart(pd.DataFrame({k: [v] for k, v in enc.items()})).mark_rule(
        strokeDash=[5, 4], color="#888780").encode(**{k: f"{k}:Q" for k in enc})
    st.altair_chart(alt.layer(rule(y=avg_epa), rule(x=avg_sr), points, labels).properties(height=380), width="stretch")
    hidden_txt = f" {hidden} with fewer than {min_plays} plays left off." if hidden else ""
    st.caption(f"Dashed lines: average EPA ({avg_epa:+.2f}) and success rate ({avg_sr:.0%}). Bigger dots = more "
               f"plays; rows with {TAKEAWAY_MIN_PLAYS}+ plays are labeled.{hidden_txt}")


def bar_chart(data: pd.DataFrame, x: str, y: str, title: str, y_format: str, x_sort=None, color=None) -> None:
    st.markdown(f"**{title}**")
    if data.empty:
        st.info("No plays match the current filters.")
        return
    base = alt.Chart(data).encode(
        x=alt.X(f"{x}:N", sort=x_sort, axis=alt.Axis(labelAngle=0), title=None),
        y=alt.Y(f"{y}:Q", axis=alt.Axis(format=y_format)),
        tooltip=[alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{y}:Q", format=y_format)],
    )
    bars = base.mark_bar(color=color) if color else base.mark_bar()
    labels = base.mark_text(dy=-8).encode(text=alt.Text(f"{y}:Q", format=y_format))
    st.altair_chart((bars + labels).properties(height=280), width="stretch")


def show_mix(title: str, mix: pd.DataFrame) -> None:
    """A table of row shares (e.g. coverage mix by down), each cell shaded by its share."""
    st.markdown(f"**{title}**")
    if mix.empty:
        st.info("No plays match the current filters.")
        return
    shade = lambda v: "" if pd.isna(v) else f"background-color:rgba(55,138,221,{min(float(v), 1) * 0.85:.2f})"
    st.dataframe(mix.style.format("{:.0%}", na_rep="").map(shade), width="stretch")
