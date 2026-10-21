"""
Play-Calling Identity page: which offenses call it unlike an average NFL team in the same spots, and which are NFL
average. The measures live in identity.py; this page draws them.
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import common as cm
import data as dt
import identity as idn

PROFILE_COLORS = {"Unorthodox": cm.PASS_C, "Distinct": "#85B7EB", "NFL average": "#888780"}
LABELS = {"lean": "Lean (PROE)", "shotgun": "Shotgun over expected", "spread": "3+ WR over expected"}


@st.cache_data(show_spinner="Loading nflfastR play-by-play…", ttl=dt.REFRESH_HOURS * 3600)
def tables(season: int) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.Series]:
    """(this season's table, last season's table or None, this season's reliabilities)."""
    seasons = [s for s in (season - 1, season) if s >= dt.FIRST_SEASON]
    plays = dt.load_seasons(seasons)
    prior = idn.season_table(plays, season - 1) if season - 1 in seasons else None
    return idn.season_table(plays, season), prior, idn.reliability(plays, season)


DEFINITIONS = f"""
Every snap is compared with what an average NFL team does **in that exact spot**, then averaged over the team's
snaps. Only regular-season snaps with the score within {idn.MAX_MARGIN} and outside the last 2 minutes of a half are
used, where the call is a real choice rather than forced by the clock or the scoreboard.

- **Lean (PROE):** pass rate over expected. nflfastR's `xpass` gives the chance an average team throws on each snap
  (down, distance, field position, score, time, timeouts). It was fit on earlier seasons and drifts (today's NFL
  runs more on 3rd/4th & 1-2 than it expects), so it's re-centered on what the league actually did in the same
  spots that season. PROE = pass rate minus that.
- **Shotgun over expected:** shotgun rate minus the league's shotgun rate in the same down, distance, field zone, and
  score state that season.
- **3+ WR over expected:** how often 3+ wide receivers are on the field (RBs + TEs ≤ 2, no extra lineman) minus the
  league's rate in the same spots. Needs FTN personnel charting, which nflverse publishes after the season, so the
  season in progress leaves it out.

For each measure:
- **z:** how many standard errors from an average team, with standard errors clustered by game (a game plan
  repeats within a game, so snaps aren't independent). |z| ≥ {idn.Z_DETECT:.0f} = detectably different.
- **Std:** the value shrunk toward the league by how noisy it is (empirical Bayes), in units of the real spread
  between teams that season. ±1 = one typical team-to-team difference.

**Unorthodoxy index** = root mean square of the Std values: 0 = calls it like an average NFL team, 1 = typically
one team-to-team spread away on each measure. **Profiles:** *Unorthodox* = index ≥ {idn.UNORTHODOX:.1f}.
*NFL average* = not different on any measure, where different means detectable (|z| ≥ {idn.Z_DETECT:.0f}) *and*
material (|Std| ≥ {idn.MATERIAL}). *Distinct* = different on at least one measure, but the index is under
{idn.UNORTHODOX:.1f}. Early in a season most teams read as NFL average simply because there aren't enough games yet
to tell.

**Situational PROE** (early downs, 1st & 10, 2nd & long, short yardage, red zone) is each team's PROE there minus
the league's PROE there, so 0 = throws as often as the NFL does in that situation. It's in the table for
exploration but kept out of the index: per team-season it's noisier. See the reliability table.
"""


def scatter(t: pd.DataFrame, x: str, y: str) -> alt.Chart:
    """One dot per team: x and y = raw measure values; the gray box = within noise (±2 typical standard errors)."""
    d = t.reset_index().rename(columns={"index": "Team"})
    d["x"], d["y"] = d[f"{x}_effect"].astype(float), d[f"{y}_effect"].astype(float)
    bx, by = 2 * d[f"{x}_se"].astype(float).median(), 2 * d[f"{y}_se"].astype(float).median()
    lim_x = max(d["x"].abs().max(), bx) * 1.15
    lim_y = max(d["y"].abs().max(), by) * 1.15
    box = alt.Chart(pd.DataFrame({"x": [-bx], "x2": [bx], "y": [-by], "y2": [by]})).mark_rect(
        color="#888780", opacity=0.12).encode(x="x:Q", x2="x2:Q", y="y:Q", y2="y2:Q")
    axes = alt.Chart(pd.DataFrame({"v": [0]}))
    vx = axes.mark_rule(color="#888780", strokeDash=[4, 4]).encode(x=alt.X("v:Q"))
    vy = axes.mark_rule(color="#888780", strokeDash=[4, 4]).encode(y=alt.Y("v:Q"))
    tip = ["Team", "Profile", alt.Tooltip("Unorthodoxy index:Q", format=".2f"),
           alt.Tooltip("x:Q", format="+.1%", title=LABELS[x]), alt.Tooltip(f"{x}_z:Q", format="+.1f", title=f"{LABELS[x]} z"),
           alt.Tooltip("y:Q", format="+.1%", title=LABELS[y]), alt.Tooltip(f"{y}_z:Q", format="+.1f", title=f"{LABELS[y]} z"),
           "Different on"]
    base = alt.Chart(d).encode(
        x=alt.X("x:Q", title=f"{LABELS[x]} →", scale=alt.Scale(domain=[-lim_x, lim_x]), axis=alt.Axis(format="+.0%")),
        y=alt.Y("y:Q", title=f"{LABELS[y]} →", scale=alt.Scale(domain=[-lim_y, lim_y]), axis=alt.Axis(format="+.0%")),
        tooltip=tip)
    dots = base.mark_circle(size=260, opacity=0.9, stroke="white", strokeWidth=1).encode(
        color=alt.Color("Profile:N", scale=alt.Scale(domain=list(PROFILE_COLORS), range=list(PROFILE_COLORS.values())),
                        legend=alt.Legend(orient="top", title=None)))
    text = base.mark_text(fontSize=11, fontWeight="bold", dy=-14).encode(text="Team:N")
    return alt.layer(box, vx, vy, dots, text).properties(height=480)


def corner_note(x: str, y: str) -> str:
    words = {"lean": ("run-heavy", "pass-heavy"), "shotgun": ("more under center", "more shotgun"),
             "spread": ("heavier personnel", "more 3+ WR")}
    return (f"Left = {words[x][0]}, right = {words[x][1]}; down = {words[y][0]}, up = {words[y][1]}, each compared "
            f"with an average NFL team in the same spots. The gray box is the noise band for a typical team (±2 "
            f"standard errors): a dot inside it can't be told apart from average on these two measures.")


def index_chart(t: pd.DataFrame, prior: pd.DataFrame | None) -> alt.Chart:
    d = t.reset_index().rename(columns={"index": "Team"})
    d["Last season"] = d["Team"].map(prior["Unorthodoxy index"]) if prior is not None else np.nan
    order = list(d.sort_values("Unorthodoxy index", ascending=False)["Team"])
    base = alt.Chart(d).encode(x=alt.X("Team:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)))
    bars = base.mark_bar().encode(
        y=alt.Y("Unorthodoxy index:Q", title="Unorthodoxy index"),
        color=alt.Color("Profile:N", scale=alt.Scale(domain=list(PROFILE_COLORS), range=list(PROFILE_COLORS.values())),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Team", "Profile", alt.Tooltip("Unorthodoxy index:Q", format=".2f"),
                 alt.Tooltip("Last season:Q", format=".2f"), "Different on"])
    layers = [bars, alt.Chart(pd.DataFrame({"y": [idn.UNORTHODOX]})).mark_rule(strokeDash=[5, 4], color="#888780")
              .encode(y="y:Q")]
    if d["Last season"].notna().any():
        layers.append(base.mark_tick(color="currentColor", thickness=2, size=14).encode(y="Last season:Q"))
    return alt.layer(*layers).properties(height=300)


def display_table(t: pd.DataFrame, prior: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    """Every value, readable column names. Measures the season doesn't have aren't in t, so they're not here."""
    out = pd.DataFrame(index=t.index)
    out["Rank"] = t["Rank"]
    out["Profile"] = t["Profile"]
    out["Unorthodoxy index"] = t["Unorthodoxy index"].astype(float)
    if prior is not None:
        out["Last season rank"] = t.index.map(prior["Rank"]).astype("Int64")
        out["Last season index"] = t.index.map(prior["Unorthodoxy index"]).astype(float)
    out["Different on"] = t["Different on"]
    out["Snaps"] = t["Snaps"].astype(int)
    out["Pass rate"], out["Expected pass rate"] = t["Pass rate"], t["Expected pass rate"]
    out["PROE"], out["PROE z"], out["PROE std"] = t["lean_effect"], t["lean_z"], t["lean_std"]
    out["Shotgun rate"], out["Expected shotgun rate"] = t["Shotgun rate"], t["Expected shotgun rate"]
    out["Shotgun OE"], out["Shotgun z"], out["Shotgun std"] = t["shotgun_effect"], t["shotgun_z"], t["shotgun_std"]
    if "spread" in t.attrs["measures"]:
        out["3+ WR rate"], out["Expected 3+ WR rate"] = t["3+ WR rate"], t["Expected 3+ WR rate"]
        out["3+ WR OE"], out["3+ WR z"], out["3+ WR std"] = t["spread_effect"], t["spread_z"], t["spread_std"]
        for name in ("2+ TE", "2+ RB"):
            out[f"{name} rate"], out[f"{name} OE"] = t[f"{name} rate"], t[f"{name} over expected"]
        out["Personnel mix distance"] = t["Personnel mix distance"]
        out["Top personnel"] = t["Top personnel"]
    for name in idn.SPLITS:
        out[name] = t[name]
    out.index.name = "Team"
    pct = "{:.1%}"
    fmt = {c: pct for c in out.columns if "rate" in c.lower() or c in ("Personnel mix distance",)}
    fmt.update({c: "{:+.1%}" for c in out.columns if c.endswith(" OE") or c == "PROE" or c in idn.SPLITS})
    fmt.update({c: "{:+.1f}" for c in out.columns if c.endswith(" z")})
    fmt.update({c: "{:+.2f}" for c in out.columns if c.endswith(" std")})
    fmt.update({"Unorthodoxy index": "{:.2f}", "Last season index": "{:.2f}", "Snaps": "{:,.0f}"})
    return out, fmt


HELP = {
    "Unorthodoxy index": "Root mean square of the Std values: 0 = like an average NFL team, 1 = one typical "
                         "team-to-team spread away on each measure.",
    "Different on": f"Measures where the team is detectably (|z| ≥ {idn.Z_DETECT:.0f}) and materially "
                    f"(|Std| ≥ {idn.MATERIAL}) different from an average team.",
    "PROE": "Pass rate over expected (pass rate minus nflfastR's xpass).",
    "Shotgun OE": "Shotgun rate minus the league's rate in the same down, distance, field zone, and score state.",
    "3+ WR OE": "3+ WR personnel rate minus the league's rate in the same spots.",
    "2+ TE OE": "Share of snaps with 2+ tight ends minus the league's rate in the same spots.",
    "2+ RB OE": "Share of snaps with 2+ backs (usually a fullback) minus the league's rate in the same spots.",
    "Personnel mix distance": "Share of snaps that would have to change personnel group to match what the league "
                              "uses in the same spots (total variation distance).",
}


def color_profile(col):
    return [f"color:{PROFILE_COLORS.get(v, 'inherit')};font-weight:600" for v in col]


def shade_std(col):
    """Purple = far from average (either direction), so the eye finds what drives the index."""
    out = []
    for v in col:
        a = 0 if pd.isna(v) else min(abs(float(v)) / 2.5, 1)
        out.append(f"background-color:rgba(83,74,183,{a * 0.45:.2f})" if a >= 0.2 else "")
    return out


# ---- Page --------------------------------------------------------------------------------------------------
seasons = dt.available_seasons()
with st.sidebar:
    st.header("Play-Calling Identity")
    st.session_state.setdefault("id_season", seasons[-1])
    season = st.selectbox("Season", seasons[::-1], key="id_season")

st.title("Play-Calling Identity")
st.caption("Which offenses call it unlike an average NFL team in the same spots, and which are NFL average.")
with st.expander("How this is measured"):
    st.markdown(DEFINITIONS)
try:
    t, prior, rel = tables(season)
except Exception as e:
    st.error(f"Couldn't load nflfastR data: {e}")
    st.stop()
if t.empty:
    st.info("No regular-season plays for this season yet.")
    st.stop()

measures = t.attrs["measures"]
if "spread" not in measures:
    st.info("No personnel charting for this season yet (nflverse publishes it after the season), so the index uses "
            "lean and shotgun only, and the personnel columns are left off.")
counts = t["Profile"].value_counts()
c = st.columns(4)
c[0].metric("Unorthodox", int(counts.get("Unorthodox", 0)), help=f"Index ≥ {idn.UNORTHODOX:.1f}")
c[1].metric("Distinct", int(counts.get("Distinct", 0)))
c[2].metric("NFL average", int(counts.get("NFL average", 0)))
c[3].metric("Snaps per team", f"{int(t['Snaps'].median()):,}",
            help=f"Median snaps per team used (score within {idn.MAX_MARGIN}, not the last 2 minutes of a half).")

st.markdown("**Unorthodoxy index**")
st.altair_chart(index_chart(t, prior), width="stretch")
st.caption(f"Bars = this season's index, colored by profile; dashed line = the unorthodox line ({idn.UNORTHODOX:.1f})."
           + (" Black ticks = the same team's index last season." if prior is not None else ""))

pairs = [("lean", "shotgun", "Lean vs shotgun")] + ([("lean", "spread", "Lean vs personnel")] if "spread" in measures else [])
for col, (x, y, title) in zip(st.columns(len(pairs)), pairs):
    with col:
        st.markdown(f"**{title}**")
        st.altair_chart(scatter(t, x, y), width="stretch")
        st.caption(corner_note(x, y))

st.markdown("**Every team, every value**")
table, fmt = display_table(t, prior)
std_cols = [c for c in table.columns if c.endswith(" std")]
sty = (table.style.format(fmt, na_rep="").apply(color_profile, subset=["Profile"]).apply(shade_std, subset=std_cols))
st.dataframe(sty, width="stretch", height=36 * (len(table) + 1) + 2,
             column_config={k: st.column_config.Column(help=v) for k, v in HELP.items() if k in table.columns})
st.caption("Purple shading on the Std columns = how far from average, either direction. OE = over expected (the "
           "team's rate minus an average team's in the same spots). Situational PROE columns are the team's PROE in "
           "that situation minus the NFL's. Sort any column by clicking its header.")

st.markdown("**How much of each measure is a real team trait**")
r = rel.rename("Reliability").to_frame()
r["In the index"] = ["Yes" if k in [idn.MEASURES[m][0] for m in measures] else "No (exploration only)" for k in r.index]
st.dataframe(r.style.format({"Reliability": "{:.2f}"}), width="stretch")
st.caption("Split-half reliability: correlation between each team's value in odd and even weeks, stepped up to a "
           "full season (Spearman-Brown). 1 = all real team-to-team difference, 0 = all noise.")
