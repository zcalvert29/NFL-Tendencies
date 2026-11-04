"""
Surprise Plays page: play by play, does a call the defense could see coming do worse? The logic lives in surprise.py;
the scouts come from the Unpredictability Index (unpredictability.py).
"""
import altair as alt
import pandas as pd
import streamlit as st

import common as cm
import data as dt
import surprise as sp
import ui

CALL_COLORS = {"Pass": cm.PASS_C, "Run": cm.RUN_C}


@st.cache_data(show_spinner="Scoring every snap…", ttl=dt.REFRESH_HOURS * 3600)
def all_snaps(seasons: tuple[int, ...]) -> pd.DataFrame:
    keep = ["Season", "posteam", "caller", "Call", "epa", "success", "explosive", "Film confidence",
            "League confidence", "Tell", "Tell group", "epa vs own", "success vs own", "explosive vs own", "Tell vs own"]
    return pd.concat([sp.snaps(ui.scored(s, None, "caller")[0], s)[keep] for s in seasons], ignore_index=True)


def tell_chart(t: pd.DataFrame, outcome: str, within: bool) -> alt.LayerChart:
    fmt = sp.PLAY_OUTCOMES[outcome][1]
    t = t.assign(lo=t["Value"] - 2 * t["SE"], hi=t["Value"] + 2 * t["SE"])
    x = alt.X("Tell group:N", sort=sp.TELL_LABELS, title=None, axis=alt.Axis(labelAngle=0, labelLimit=160))
    color = alt.Color("Call:N", scale=alt.Scale(domain=list(CALL_COLORS), range=list(CALL_COLORS.values())),
                      legend=alt.Legend(title=None, orient="top"))
    off = alt.XOffset("Call:N", sort=["Run", "Pass"])
    y_title = f"{outcome} vs his usual {'call' if within else ''}".strip() if within else outcome
    tip = ["Call", "Tell group", alt.Tooltip("Value:Q", format=fmt, title=outcome), alt.Tooltip("Plays:Q", format=",")]
    bars = alt.Chart(t).mark_bar().encode(x=x, xOffset=off, color=color, tooltip=tip,
                                          y=alt.Y("Value:Q", title=y_title, axis=alt.Axis(format=fmt)))
    err = alt.Chart(t).mark_rule(color=cm.theme_ink(), opacity=0.6).encode(x=x, xOffset=off, y="lo:Q", y2="hi:Q")
    zero = alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(color="#888780").encode(y="v:Q")
    return alt.layer(bars, err, zero).properties(height=340)


def confidence_chart(c: pd.DataFrame, outcome: str) -> alt.Chart:
    fmt = sp.PLAY_OUTCOMES[outcome][1]
    return alt.Chart(c).mark_line(point=True).encode(
        x=alt.X("Confidence:N", sort=sp.CONF_LABELS, title="How sure the scout was of the call that came",
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Value:Q", title=outcome, axis=alt.Axis(format=fmt)),
        color=alt.Color("Call:N", scale=alt.Scale(domain=list(CALL_COLORS), range=list(CALL_COLORS.values())),
                        legend=alt.Legend(title=None, orient="top")),
        strokeDash=alt.StrokeDash("Scout:N", scale=alt.Scale(domain=["His film", "League norms"], range=[[1, 0], [5, 4]]),
                                  legend=alt.Legend(title=None, orient="top")),
        tooltip=["Call", "Scout", "Confidence", alt.Tooltip("Value:Q", format=fmt, title=outcome),
                 alt.Tooltip("Plays:Q", format=",")],
    ).properties(height=320)


# ---- Page ----------------------------------------------------------------------------------------------------
seasons = dt.available_seasons()
info = ui.team_info()
last_full = seasons[-2] if len(seasons) > 1 else seasons[-1]
with st.sidebar:
    st.header("Surprise Plays")
    span = st.slider("Seasons", seasons[0], seasons[-1], (max(seasons[0], last_full - 4), last_full), key="sp_span",
                     help="Every film-scored snap (each play caller's 4th game on) in these regular seasons.")
    outcome = st.selectbox("Did the play work? Measured by", list(sp.PLAY_OUTCOMES), key="sp_outcome")
    within = st.toggle("Compare each play with that offense's usual run or pass", True, key="sp_within",
                       help="On: each play against the same offense's average run (or pass) that season, so a great "
                            "quarterback's predictable passes don't count as 'predictable passes work'. Off: raw.")
    min_plays = st.slider("Play callers: minimum given-away plays", 5, 60, 15, 5, key="sp_min")

years = f"{span[0]}" if span[0] == span[1] else f"{span[0]}-{span[1]}"
st.title("Surprise Plays")
st.markdown("Does a call the defense could see coming do worse? For every snap, the Unpredictability Index's two scouts "
            "say how sure they were of the call that came: one knows only **league norms** (down, distance, score, "
            "formation, personnel, motion...), the other also studied **his earlier games**. Where his film was surer "
            "than league norms, his own habits **gave the call away**; where it was less sure, they **hid it**.")
try:
    d = all_snaps(tuple(range(span[0], span[1] + 1)))
except Exception as e:
    st.error(f"Couldn't load nflfastR data: {e}")
    st.stop()
if d.empty:
    st.info("No film-scored snaps in these seasons yet.")
    st.stop()

st.subheader("When his own habits gave the call away")
for call in ("Run", "Pass"):
    st.markdown(sp.tell_sentence(sp.tell_trend(d, call, outcome, within), call, outcome, within))
st.altair_chart(tell_chart(sp.by_tell(d, outcome, within), outcome, within), width="stretch")
st.caption(f"{len(d):,} snaps, {years}. Groups by how much his film moved the scout's confidence in the call that came, "
           "compared with league norms alone. " + ("Bars = the play against that offense's average run or pass that "
           "season (0 = his usual)." if within else "Bars = the raw average.") + " Lines = ±2 standard errors: groups "
           "whose lines cross 0 aren't clearly different from usual. Most plays sit in the middle group: a habit has to "
           "show up again and again before the film scout trusts it.")

st.subheader("Calls the defense saw coming")
st.altair_chart(confidence_chart(sp.by_confidence(d, outcome), outcome), width="stretch")
st.caption("Raw averages by how sure each scout was. Solid = his film; dashed = league norms. The two lines nearly "
           "match, so this mostly shows the situation, not the coach: passes everyone sees coming are 3rd & long "
           "throws, which work less often for every team. The section above takes the league part out.")

st.subheader("Play callers")
ct = sp.caller_table(d, outcome, min_plays)
if ct.empty:
    st.info("No play callers here.")
else:
    ct = ct.sort_values("Runs: given away vs others", na_position="last")
    fmt = sp.PLAY_OUTCOMES[outcome][1]
    diff = (lambda v: "–" if pd.isna(v) else f"{v:+.3f}") if "%" not in fmt else (
        lambda v: "–" if pd.isna(v) else f"{v * 100:+.1f} pts")
    show = pd.DataFrame({"Logo": ct["Team"].map(info.get("logo", {})).fillna(""), "Play caller": ct["Play caller"],
                         "Team": ct["Team"], "Seasons": ct["Seasons"], "Snaps": ct["Snaps"],
                         "Runs given away": ct["Share of runs given away"] * 100,
                         # As text: Streamlit draws a missing number as "None".
                         "Readable runs vs his others": ct["Runs: given away vs others"].map(diff),
                         "Passes given away": ct["Share of passes given away"] * 100,
                         "Readable passes vs his others": ct["Passes: given away vs others"].map(diff)})
    pct = st.column_config.ProgressColumn
    st.dataframe(show,
                 width="stretch", hide_index=True, height=min(36 * (len(show) + 1) + 2, 760),
                 column_config={"Logo": st.column_config.ImageColumn("", width="small"),
                                "Runs given away": pct("Runs given away", format="%.0f%%", min_value=0, max_value=60,
                                                       help="Share of his runs where his habits made the call more "
                                                            "predictable than league norms (by 3+ points)."),
                                "Passes given away": pct("Passes given away", format="%.0f%%", min_value=0,
                                                         max_value=60),
                                "Readable runs vs his others": st.column_config.Column(
                                    help=f"{outcome} on his given-away runs minus his other runs, each against his "
                                         f"average run that season. Blank = under {min_plays} of either."),
                                "Readable passes vs his others": st.column_config.Column(
                                    help=f"Same for passes. Blank = under {min_plays} of either.")})
    st.caption(f"Play callers {years}, seasons pooled, most-hurt readable runs first. Most coaches have only dozens of "
               "given-away plays, so single numbers swing a lot; read the league-wide result above first.")

with st.expander("Read with care"):
    st.markdown(
        "- **The scouts only know what's visible before the snap.** A defense also reads splits, alignment, and the "
        "quarterback; a call the scouts thought was 50/50 may not have been a surprise on the field.\n"
        "- **Readable isn't chosen at random.** Coaches lean on what works; a readable call can still be his best call. "
        "Comparing within the same offense takes out the team's overall quality, not every reason for a call.\n"
        "- **Small effects, many plays.** With tens of thousands of snaps even a tiny difference can be more than chance; "
        "check the size (EPA per play or percentage points), not just whether it's significant.\n"
        "- **Counted:** regular season, outside the last 2 minutes of each half, score within 16, each play caller's "
        "4th game on (the film scout needs film first).")
