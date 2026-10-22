"""
Play-Calling Identity page: which offenses call it unlike an average NFL team in the same spots, and which are NFL
average. The measures live in identity.py; this page draws them. Every chart uses team colors and logos.
"""
from dataclasses import replace

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import common as cm
import data as dt
import identity as idn

LABELS = {"lean": "Lean (PROE)", "shotgun": "Shotgun over expected", "spread": "3+ WR over expected",
          idn.SITUATIONAL: "Situational play calling"}
PROFILES = ["Unorthodox", "Distinct", "NFL average"]
LOGO = 26          # logo size on the bar chart, px


@st.cache_data(show_spinner="Loading nflfastR play-by-play…", ttl=dt.REFRESH_HOURS * 3600)
def plays(seasons: tuple[int, ...]) -> pd.DataFrame:
    return dt.load_seasons(list(seasons))


@st.cache_data(show_spinner="Measuring every offense…", ttl=dt.REFRESH_HOURS * 3600)
def tables(season: int, settings: idn.Settings) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.Series]:
    """(this season's table, last season's full-season table or None, this season's reliabilities)."""
    seasons = tuple(s for s in (season - 1, season) if s >= dt.FIRST_SEASON)
    p = plays(seasons)
    prior = idn.season_table(p, season - 1, replace(settings, weeks=None)) if season - 1 in seasons else None
    return idn.season_table(p, season, settings), prior, idn.reliability(p, season, settings)


@st.cache_data(show_spinner=False)
def team_info() -> pd.DataFrame:
    """Team names, colors, divisions, and logos (embedded data URIs; see data.team_logos)."""
    try:
        info = dt.load_teams()
    except Exception:
        return pd.DataFrame(columns=["team_name", "team_color", "team_color2", "team_conf", "team_division", "logo"])
    return info.assign(logo=pd.Series(dt.team_logos(info)))


def _luminance(hex_color: str) -> float:
    h = str(hex_color).lstrip("#")
    if len(h) != 6:
        return 0.5
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def team_frame(t: pd.DataFrame, info: pd.DataFrame, highlight: list[str]) -> pd.DataFrame:
    """The table with the columns every chart needs: Team, color (readable on this theme), logo, opacity."""
    d = t.reset_index().rename(columns={"index": "Team"})
    dark = cm.theme_ink() == "#FAFAFA"
    color, color2 = d["Team"].map(info.get("team_color", {})), d["Team"].map(info.get("team_color2", {}))
    d["color"] = [c2 if dark and _luminance(c) < 0.06 and isinstance(c2, str) else c if isinstance(c, str) else "#888780"
                  for c, c2 in zip(color, color2)]
    d["logo"] = d["Team"].map(info.get("logo", {})).fillna("")
    d["name"] = d["Team"].map(info.get("team_name", {})).fillna(d["Team"])
    d["opacity"] = np.where(d["Team"].isin(highlight) | (len(highlight) == 0), 1.0, 0.25)
    return d


# ---- Charts --------------------------------------------------------------------------------------------------
def index_chart(d: pd.DataFrame, prior: pd.DataFrame | None) -> alt.Chart:
    """Bars in each team's primary color with the team's logo on top. Last season's index is in the tooltip."""
    d = d.assign(**{"Last season": d["Team"].map(prior["Unorthodoxy index"]) if prior is not None else np.nan})
    order = list(d.sort_values("Unorthodoxy index", ascending=False)["Team"])
    top = float(d["Unorthodoxy index"].max() * 1.18)
    tip = [alt.Tooltip("name:N", title="Team"), "Profile", alt.Tooltip("Unorthodoxy index:Q", format=".2f"),
           alt.Tooltip("Last season:Q", format=".2f"), "Different on"]
    x = alt.X("Team:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=10))
    y = alt.Y("Unorthodoxy index:Q", title="Unorthodoxy index", scale=alt.Scale(domain=[0, top]))
    base = alt.Chart(d).encode(x=x, tooltip=tip)
    bars = base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        y=y, color=alt.Color("color:N", scale=None), opacity=alt.Opacity("opacity:Q", scale=None))
    logos = base.mark_image(width=LOGO, height=LOGO, baseline="bottom").encode(
        y="Unorthodoxy index:Q", url="logo:N", opacity=alt.Opacity("opacity:Q", scale=None))
    return alt.layer(bars, logos).properties(height=340)


def logo_scatter(d: pd.DataFrame, x: str, y: str, x_title: str, y_title: str, x_fmt: str, y_fmt: str,
                 tip_extra: list | None = None, box: tuple[float, float] | None = None, zero_lines: bool = True,
                 trend: bool = False, height: int = 460) -> alt.LayerChart:
    """One team per point, drawn as its logo (a dot in the team's color if the logo couldn't be downloaded).
    box = (half-width x, half-height y) of a gray noise box around 0; trend = least-squares line."""
    d = d.dropna(subset=[x, y])
    pad_x = (d[x].max() - d[x].min()) * 0.08 or 0.01
    pad_y = (d[y].max() - d[y].min()) * 0.08 or 0.01
    xs = alt.Scale(domain=[d[x].min() - pad_x, d[x].max() + pad_x])
    ys = alt.Scale(domain=[d[y].min() - pad_y, d[y].max() + pad_y])
    if zero_lines:   # symmetric around 0 so "average" sits in the middle
        lx, ly = max(abs(d[x]).max() + pad_x, box[0] if box else 0), max(abs(d[y]).max() + pad_y, box[1] if box else 0)
        xs, ys = alt.Scale(domain=[-lx, lx]), alt.Scale(domain=[-ly, ly])
    tip = [alt.Tooltip("name:N", title="Team"), "Profile", alt.Tooltip("Unorthodoxy index:Q", format=".2f"),
           alt.Tooltip(f"{x}:Q", title=x_title, format=x_fmt), alt.Tooltip(f"{y}:Q", title=y_title, format=y_fmt)]
    tip += tip_extra or []
    base = alt.Chart(d).encode(
        x=alt.X(f"{x}:Q", title=x_title, scale=xs, axis=alt.Axis(format=x_fmt, tickCount=8)),
        y=alt.Y(f"{y}:Q", title=y_title, scale=ys, axis=alt.Axis(format=y_fmt, tickCount=8)), tooltip=tip)
    layers = []
    if box:
        layers.append(alt.Chart(pd.DataFrame({"a": [-box[0]], "b": [box[0]], "c": [-box[1]], "e": [box[1]]}))
                      .mark_rect(color="#888780", opacity=0.12).encode(x="a:Q", x2="b:Q", y="c:Q", y2="e:Q"))
    if zero_lines:
        z = alt.Chart(pd.DataFrame({"v": [0]}))
        layers += [z.mark_rule(color="#888780", strokeDash=[4, 4]).encode(x="v:Q"),
                   z.mark_rule(color="#888780", strokeDash=[4, 4]).encode(y="v:Q")]
    if trend:
        layers.append(base.transform_regression(x, y).mark_line(color="#888780", strokeDash=[6, 4], strokeWidth=2)
                      .encode(tooltip=alt.value(None)))
    layers += [
        base.transform_filter("datum.logo == ''").mark_circle(size=500).encode(
            color=alt.Color("color:N", scale=None), opacity=alt.Opacity("opacity:Q", scale=None)),
        base.mark_image(width=38, height=38).encode(url="logo:N", opacity=alt.Opacity("opacity:Q", scale=None)),
    ]
    return alt.layer(*layers).properties(height=height)


def proe_chart(d: pd.DataFrame) -> alt.Chart:
    """PROE per team: bars in the team's color, up = throws more than expected, down = runs more; the logo sits at
    the end of each bar."""
    d = d.assign(PROE=d["Pass rate"] - d["Expected pass rate"])
    order = list(d.sort_values("PROE", ascending=False)["Team"])
    lim = float(d["PROE"].abs().max() * 1.3) or 0.05
    tip = [alt.Tooltip("name:N", title="Team"), alt.Tooltip("PROE:Q", format="+.1%"),
           alt.Tooltip("Pass rate:Q", format=".1%"), alt.Tooltip("Expected pass rate:Q", format=".1%"),
           alt.Tooltip("nflfastR PROE:Q", format="+.1%", title="nflfastR's raw PROE"), "Snaps"]
    x = alt.X("Team:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=10, orient="bottom"))
    y = alt.Y("PROE:Q", title="Pass rate over expected", scale=alt.Scale(domain=[-lim, lim]),
              axis=alt.Axis(format="+.0%"))
    base = alt.Chart(d).encode(x=x, tooltip=tip)
    bars = base.mark_bar(cornerRadius=3).encode(
        y=y, color=alt.Color("color:N", scale=None), opacity=alt.Opacity("opacity:Q", scale=None))
    up = base.transform_filter("datum.PROE >= 0").mark_image(width=LOGO, height=LOGO, baseline="bottom").encode(
        y="PROE:Q", url="logo:N", opacity=alt.Opacity("opacity:Q", scale=None))
    down = base.transform_filter("datum.PROE < 0").mark_image(width=LOGO, height=LOGO, baseline="top").encode(
        y="PROE:Q", url="logo:N", opacity=alt.Opacity("opacity:Q", scale=None))
    zero = alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(color="#888780").encode(y="v:Q")
    return alt.layer(bars, zero, up, down).properties(height=360)


# ---- Table ----------------------------------------------------------------------------------------------------
def display_table(t: pd.DataFrame, prior: pd.DataFrame | None, info: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Every value, readable column names. Measures the season doesn't have aren't in t, so they're not here."""
    out = pd.DataFrame(index=t.index)
    out["Team name"] = t.index.map(info.get("team_name", {}))
    out["Rank"] = t["Rank"]
    out["Profile"] = t["Profile"]
    out["Unorthodoxy index"] = t["Unorthodoxy index"]
    if prior is not None:
        out["Last season rank"] = t.index.map(prior["Rank"]).astype("Int64")
        out["Last season index"] = t.index.map(prior["Unorthodoxy index"])
    out["Different on"] = t["Different on"]
    if "Most distinctive situation" in t:
        out["Most distinctive situation"] = t["Most distinctive situation"]
    out["EPA/play"], out["Success rate"] = t["EPA/play"], t["Success rate"]
    out["Snaps"], out["Games"] = t["Snaps"], t["Games"]
    out["Pass rate"], out["Expected pass rate"] = t["Pass rate"], t["Expected pass rate"]
    measures = t.attrs["measures"]
    if "lean" in measures:
        out["PROE"], out["PROE std"] = t["lean_effect"], t["lean_std"]
    out["Shotgun rate"], out["Expected shotgun rate"] = t["Shotgun rate"], t["Expected shotgun rate"]
    if "shotgun" in measures:
        out["Shotgun OE"], out["Shotgun std"] = t["shotgun_effect"], t["shotgun_std"]
    if "3+ WR rate" in t:
        out["3+ WR rate"], out["Expected 3+ WR rate"] = t["3+ WR rate"], t["Expected 3+ WR rate"]
        if "spread" in measures:
            out["3+ WR OE"], out["3+ WR std"] = t["spread_effect"], t["spread_std"]
        for name in ("2+ TE", "2+ RB"):
            out[f"{name} rate"], out[f"{name} OE"] = t[f"{name} rate"], t[f"{name} over expected"]
        out["Personnel mix distance"], out["Top personnel"] = t["Personnel mix distance"], t["Top personnel"]
    if idn.SITUATIONAL in measures:
        out["Situational std"] = t["situational_std"]
        for s in t.attrs["situations"]:
            out[f"{s}: vs NFL"] = t[f"{s}|vs_nfl"]
            out[f"{s}: beyond lean"] = t[f"{s}|effect"]
    out.index.name = "Team"
    fmt = {c: "{:.1%}" for c in out.columns if "rate" in c.lower() or c == "Personnel mix distance"}
    fmt.update({c: "{:+.1%}" for c in out.columns if c.endswith((" OE", ": vs NFL", ": beyond lean")) or c == "PROE"})
    fmt.update({c: "{:+.2f}" for c in out.columns if c.endswith(" std") and c != "Situational std"})
    fmt.update({"Unorthodoxy index": "{:.2f}", "Last season index": "{:.2f}", "Situational std": "{:.2f}",
                "Snaps": "{:,.0f}", "Games": "{:.0f}", "EPA/play": "{:+.3f}", "Success rate": "{:.1%}"})
    return out, fmt


HELP = {
    "Unorthodoxy index": "Root mean square of the Std values: 0 = like an average NFL team, 1 = one typical "
                         "team-to-team spread away on each measure.",
    "Different on": f"Measures where the team is detectably (|z| ≥ {idn.Z_DETECT:.0f}; situations need the "
                    f"Bonferroni-adjusted z) and materially (|Std| ≥ {idn.MATERIAL}) different from an average team.",
    "Most distinctive situation": "The situation where the team departs most from its own overall lean.",
    "EPA/play": "Offensive EPA per play over the same snaps the index uses.",
    "PROE": "Pass rate over expected (vs the rest of the league in the same spots).",
    "Shotgun OE": "Shotgun rate minus the rest of the league's rate in the same down, distance, field zone, and score.",
    "3+ WR OE": "3+ WR personnel rate minus the rest of the league's rate in the same spots.",
    "2+ TE OE": "Share of snaps with 2+ tight ends minus the league's rate in the same spots.",
    "2+ RB OE": "Share of snaps with 2+ backs (usually a fullback) minus the league's rate in the same spots.",
    "Personnel mix distance": "Share of snaps that would have to change personnel group to match what the league "
                              "uses in the same spots (total variation distance).",
    "Situational std": "Root mean square of the situations' Std values: how unusual the situational calling is "
                       "beyond the team's overall lean.",
}


def color_profile(col):
    colors = {"Unorthodox": cm.PASS_C, "Distinct": "#378ADD", "NFL average": "#888780"}
    return [f"color:{colors.get(v, 'inherit')};font-weight:600" for v in col]


def shade_std(col):
    out = []
    for v in col:
        a = 0 if pd.isna(v) else min(abs(float(v)) / 2.5, 1)
        out.append(f"background-color:rgba(83,74,183,{a * 0.45:.2f})" if a >= 0.2 else "")
    return out


DEFINITIONS = f"""
Every snap is compared with what **the rest of the league** does in that exact spot that season (the team itself is
left out of its own baseline). The sidebar picks which snaps count (weeks, score window, last 2 minutes of a half)
and what goes into the index.

- **Lean (PROE):** pass rate over expected. nflfastR's `xpass` gives the chance an average team throws on each snap;
  it was fit on earlier seasons and drifts (today's NFL runs more on 3rd/4th & 1-2 than it expects), so it's
  re-centered on what the rest of the league actually did in the same spots.
- **Shotgun over expected:** shotgun rate minus the rest of the league's in the same down, distance, field zone, and
  score state.
- **3+ WR over expected:** 3+ WR personnel rate minus the rest of the league's in the same spots. Needs FTN charting,
  which nflverse publishes after the season, so the season in progress leaves it out.
- **Situational play calling:** in each situation (1st & 10, 2nd & long, 3rd/4th & 1-2, red zone, after a run, first
  {idn.FIRST_SNAPS} snaps, ...), how much more or less the team throws than the rest of the league does there,
  **beyond its own overall lean**. A pass-heavy team that's pass-heavy everywhere scores 0; a run-heavy team that
  throws on 3rd & 1 scores high. The component is the root mean square of the situations' Std values.

For each measure: **z** = standard errors from average, clustered by game (a game plan repeats within a game);
|z| ≥ {idn.Z_DETECT:.0f} = detectable. Situations are many tests at once, so each needs a Bonferroni-adjusted z (about
{idn.z_situation(len(idn.SITUATIONS)):.1f} with all {len(idn.SITUATIONS)} on). **Std** = the value shrunk toward the
league by how noisy it is (empirical Bayes), in units of the real team-to-team spread; noisy situations shrink
toward 0, so they barely move the index.

**Unorthodoxy index** = root mean square of the Std values.

**How to read the index**
- **The unit is "a typical team-to-team gap."** 1.0 means that, across the measures, the offense sits about one
  typical gap between NFL offenses away from what an average team would do in the same spots. 2025 Baltimore (1.33):
  lean −1.51 (run-heavy), shotgun +0.17, personnel −2.13 (heavy), situational 0.46; squared, averaged, and square-rooted,
  that's 1.33.
- **It has no direction and no grade.** Pass-heavy and run-heavy both count as unusual, and unusual isn't good or
  bad. Use the bars, scatters, and table to see *which way* a team is different, and the EPA chart for whether it works.
- **It never reaches 0 in practice.** The squares can't cancel and some noise survives the shrinkage, so even the most
  average offenses land around 0.3-0.4. Read it against the league: across 2021-25 the median team is about 0.77, the
  top quarter 1.07+, the top 10% 1.3+, and the most extreme 1.84 (2024 New Orleans).
- **It's relative to that season.** Each measure is scaled by that season's real spread between teams, so compare
  ranks across seasons rather than raw values, and expect the season in progress (fewer games, no personnel yet) to
  run a little differently.

**Profiles:** *Unorthodox* = index ≥
{idn.UNORTHODOX:.1f}; *NFL average* = not detectably *and* materially (|Std| ≥ {idn.MATERIAL}) different on anything;
*Distinct* = in between. Early in a season many teams read as NFL average simply because there isn't enough data yet.
"""

# ---- Page ----------------------------------------------------------------------------------------------------
seasons = dt.available_seasons()
info = team_info()
with st.sidebar:
    st.header("Play-Calling Identity")
    st.session_state.setdefault("id_season", seasons[-1])
    season = st.selectbox("Season", seasons[::-1], key="id_season")
    season_plays = plays((season,))
    reg_weeks = season_plays.loc[season_plays["season_type"] == "REG", "Week"]
    if reg_weeks.empty:
        st.info("No regular-season plays for this season yet.")
        st.stop()
    max_week = int(reg_weeks.max())
    weeks = st.slider("Weeks", 1, max_week, (1, max_week), key=f"id_weeks_{season}",
                      help="Measure only these regular-season weeks, e.g. the last 4 to catch a change in play caller.")

    st.subheader("Which snaps count")
    margin = st.slider("Score within … points", 3, 30, 16, key="id_margin",
                       help="Leave out snaps in lopsided games, where the scoreboard calls the plays.")
    drop_2min = st.checkbox("Leave out the last 2 minutes of each half", True, key="id_2min")

    st.subheader("What goes in the index")
    has_personnel = season_plays["Personnel"].notna().mean() > 0.5
    available = [k for k in LABELS if k != "spread" or has_personnel]
    components = st.multiselect("Measures", available, default=available, format_func=LABELS.get, key=f"id_comp_{season}")
    situations = list(idn.SITUATIONS)
    if idn.SITUATIONAL in components:
        with st.expander("Situations"):
            situations = st.multiselect("Situations in the situational measure", list(idn.SITUATIONS),
                                        default=list(idn.SITUATIONS), key="id_sits", label_visibility="collapsed")

    st.subheader("Which teams to show")
    conf = st.segmented_control("Conference", ["AFC", "NFC"], selection_mode="multi", default=["AFC", "NFC"],
                                key="id_conf")
    divisions = sorted(info["team_division"].dropna().unique()) if "team_division" in info else []
    divisions = [x for x in divisions if x.split()[0] in (conf or ["AFC", "NFC"])]
    sel_div = st.multiselect("Division", divisions, default=divisions, key="id_div")
    profiles = st.segmented_control("Profile", PROFILES, selection_mode="multi", default=PROFILES, key="id_prof")
    highlight = st.multiselect("Highlight teams", sorted(season_plays["posteam"].dropna().unique()), key="id_hl",
                               help="Fades every other team in the charts.")
if not components:
    st.warning("Pick at least one measure for the index.")
    st.stop()
if idn.SITUATIONAL in components and not situations:
    components = [c for c in components if c != idn.SITUATIONAL]

settings = idn.Settings(weeks=None if weeks == (1, max_week) else weeks, max_margin=margin,
                        drop_two_minute=drop_2min, components=tuple(components), situations=tuple(situations))

st.title("Play-Calling Identity")
st.caption("Which offenses call it unlike an average NFL team in the same spots, and which are NFL average.")
with st.expander("How this is measured"):
    st.markdown(DEFINITIONS)
try:
    t, prior, rel = tables(season, settings)
except Exception as e:
    st.error(f"Couldn't load nflfastR data: {e}")
    st.stop()
if t.empty or t["Unorthodoxy index"].isna().all():
    st.info("Not enough games with these filters: each team needs at least 3 games to measure. Widen the weeks.")
    st.stop()
t = t[t["Unorthodoxy index"].notna()]

measures = t.attrs["measures"]
if not has_personnel:
    st.info("No personnel charting for this season yet (nflverse publishes it after the season), so the personnel "
            "measure and columns are left out.")
d_all = team_frame(t, info, highlight)
shown = d_all["Team"].map(info.get("team_division", {})).isin(sel_div) & d_all["Profile"].isin(profiles or PROFILES)
d = d_all[shown]
if d.empty:
    st.info("No teams match the sidebar's team filters.")
    st.stop()
week_txt = "all weeks" if settings.weeks is None else f"weeks {weeks[0]}-{weeks[1]}"
counts = t["Profile"].value_counts()
c = st.columns(5)
c[0].metric("Unorthodox", int(counts.get("Unorthodox", 0)), help=f"Index ≥ {idn.UNORTHODOX:.1f}")
c[1].metric("Distinct", int(counts.get("Distinct", 0)))
c[2].metric("NFL average", int(counts.get("NFL average", 0)))
c[3].metric("Snaps per team", f"{int(t['Snaps'].median()):,}", help=f"Median snaps per team used ({week_txt}).")
c[4].metric("Index uses", f"{len(measures)} measures", help=", ".join(LABELS[m] for m in measures))

st.markdown(f"**Unorthodoxy index, {season} ({week_txt})**")
st.altair_chart(index_chart(d, prior), width="stretch")
st.caption(f"Bars in each team's primary color, logo on top. {idn.UNORTHODOX:.1f} or higher = unorthodox. Hover a bar "
           f"for what makes the team different" + (" and its index last season." if prior is not None else "."))

st.markdown("**Each team's score on each measure**")
score_cols = {m: f"{m}_std" for m in measures if m in idn.MEASURES}
if idn.SITUATIONAL in measures:
    score_cols[idn.SITUATIONAL] = "situational_std"
scores = pd.DataFrame({"Logo": d["logo"].to_numpy(), "Team": d["name"].to_numpy(),
                       "Unorthodoxy index": d["Unorthodoxy index"].to_numpy(),
                       **{LABELS[m]: d[c].to_numpy() for m, c in score_cols.items()}}, index=d["Team"].to_numpy())
# Which measure contributes most to the index (the largest score, either direction).
scores["Driven by"] = scores[[LABELS[m] for m in score_cols]].abs().idxmax(axis=1)
score_fmt = {LABELS[m]: "{:.2f}" if m == idn.SITUATIONAL else "{:+.2f}" for m in score_cols}
st.dataframe(
    scores.style.format({**score_fmt, "Unorthodoxy index": "{:.2f}"}, na_rep="")
    .apply(shade_std, subset=list(score_fmt)),
    width="stretch", hide_index=True, height=36 * (len(scores) + 1) + 2,
    column_config={"Logo": st.column_config.ImageColumn("", width="small")})


def _extremes(col: str) -> str:
    """'BAL −1.51 … LA +1.68': the lowest and highest score shown, for the writeup."""
    s = scores[LABELS[col]].dropna()
    if s.empty:
        return ""
    lo, hi = s.idxmin(), s.idxmax()
    return f" This season's range: {lo} {s[lo]:+.2f} to {hi} {s[hi]:+.2f}."


notes = {
    "lean": "**Lean (PROE)** — how much more often the team throws than an average NFL team would in the same spots. "
            "Positive = pass-heavy, negative = run-heavy. ±1 is about one typical gap between NFL offenses; beyond ±1.5 "
            "is among the most lopsided play callers in the league.",
    "shotgun": "**Shotgun over expected** — how often the team snaps from shotgun compared with what the league does in "
               "the same down, distance, field position, and score. Positive = more shotgun (spread, quick-game "
               "offenses), negative = more under center (play action, wide zone, power run games).",
    "spread": "**3+ WR over expected** — how often the team has three or more receivers on the field compared with the "
              "league in the same spots. Positive = lighter, spread personnel (lots of 11 or 10); negative = heavier "
              "personnel (12, 13, 21, fullbacks, extra linemen).",
    idn.SITUATIONAL: "**Situational play calling** — how unusually the team calls specific situations (1st & 10, 2nd & "
                     "long, 3rd/4th & 1-2, red zone, after a run, its first 15 snaps, ...) once its overall lean is "
                     "taken out. It has no sign: 0 = calls each situation the way its overall tendency predicts, "
                     "higher = more situation-specific habits. Even average teams score about 0.4-0.6 here because some "
                     "noise survives; above ~0.8 is worth a look, and the table further down shows which situation.",
}
st.markdown("\n".join(f"- {notes[m]}" + (_extremes(m) if m in idn.MEASURES else "") for m in score_cols))
st.caption("Scores are in units of a typical gap between NFL offenses this season, after pulling small samples toward "
           "the league average. Purple shading = how far from average, either direction. The unorthodoxy index is the "
           "square root of the average squared score; \"Driven by\" is the measure with the largest score.")

st.markdown("**Offensive EPA per play vs unorthodoxy**")
r = d["Unorthodoxy index"].corr(d["EPA/play"])
st.altair_chart(logo_scatter(d, "Unorthodoxy index", "EPA/play", "Unorthodoxy index →", "Offensive EPA per play →",
                             ".2f", "+.2f", tip_extra=["Different on"], zero_lines=False, trend=True),
                width="stretch")
st.caption(f"Does calling it differently pay off? Correlation across these teams: r = {r:+.2f} (dashed line = "
           f"least-squares fit). EPA per play is over the same snaps the index uses. Right = more unorthodox, up = more "
           f"efficient.")

pairs = [("lean", "shotgun"), ("lean", "spread"), ("lean", idn.SITUATIONAL)]
pairs = [(x, y) for x, y in pairs if x in measures and y in measures]
if pairs:
    st.markdown("**What makes each offense different**")
    words = {"lean": ("run-heavy", "pass-heavy"), "shotgun": ("more under center", "more shotgun"),
             "spread": ("heavier personnel", "more 3+ WR"), idn.SITUATIONAL: ("", "")}
    for x, y in pairs:   # full width each: 32 logos need the room
        st.markdown(f"*{LABELS[x]} vs {LABELS[y][0].lower() + LABELS[y][1:]}*")
        if y == idn.SITUATIONAL:
            dd = d.assign(xv=d[f"{x}_effect"], yv=d["situational_std"])
            chart = logo_scatter(dd, "xv", "yv", f"{LABELS[x]} →", "Situational play calling →", "+.0%", ".2f",
                                 tip_extra=["Most distinctive situation"], zero_lines=False, height=440)
            note = (f"Left = {words[x][0]}, right = {words[x][1]}. Up = calls situations more unlike the NFL, "
                    f"beyond its overall lean.")
        else:
            dd = d.assign(xv=d[f"{x}_effect"], yv=d[f"{y}_effect"])
            box = (2 * t[f"{x}_se"].median(), 2 * t[f"{y}_se"].median())
            chart = logo_scatter(dd, "xv", "yv", f"{LABELS[x]} →", f"{LABELS[y]} →", "+.0%", "+.0%", box=box,
                                 height=440)
            note = (f"Left = {words[x][0]}, right = {words[x][1]}; down = {words[y][0]}, up = {words[y][1]}. Gray "
                    f"box = noise band for a typical team (±2 standard errors).")
        st.altair_chart(chart, width="stretch")
        st.caption(note)

st.markdown("**Pass rate over expected**")
st.altair_chart(proe_chart(d), width="stretch")
st.caption("How much more often each team throws than an average NFL team would in the same spots (down, distance, "
           "field position, score, time, timeouts), in percentage points. Up = pass-heavy, down = run-heavy. Expected "
           "= nflfastR's xpass, re-centered on what the rest of the league actually did in the same spots this season, "
           "so these can differ by a point or two from raw nflfastR PROE on public sites (hover for both).")

st.markdown("**Every team, every value**")
table, fmt = display_table(t.loc[d["Team"]], prior, info)
std_cols = [c for c in table.columns if c.endswith(" std")]
sty = table.style.format(fmt, na_rep="").apply(color_profile, subset=["Profile"]).apply(shade_std, subset=std_cols)
st.dataframe(sty, width="stretch", height=36 * (len(table) + 1) + 2,
             column_config={k: st.column_config.Column(help=v) for k, v in HELP.items() if k in table.columns})
st.caption("Purple shading on the Std columns = how far from average, either direction. OE = over expected (the "
           "team's rate minus the rest of the league's in the same spots). For each situation: vs NFL = pass rate "
           "there minus the league's; beyond lean = that minus the team's overall PROE (what the index uses). Sort any "
           "column by clicking its header; download with the icon at the table's top right.")

st.markdown("**How much of each measure is a real team trait**")
if rel.empty:
    st.caption("Not enough weeks to split the sample in half.")
else:
    rt = rel.rename("Reliability").to_frame()
    st.dataframe(rt.style.format({"Reliability": "{:.2f}"}, na_rep="too few teams to tell"), width="stretch")
    st.caption("Split-half reliability: correlation between each team's value in odd and even weeks, stepped up to the "
               "full sample (Spearman-Brown). 1 = all real team-to-team difference, 0 = all noise. Low-reliability "
               "situations get shrunk hard, so they move the index little.")
