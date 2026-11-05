"""
Surprise Plays page: play by play, does a call the defense could see coming do worse? The logic lives in surprise.py;
the scouts come from the Unpredictability Index (unpredictability.py).
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import common as cm
import data as dt
import surprise as sp
import ui

CALL_COLORS = {"Pass": cm.PASS_C, "Run": cm.RUN_C}


@st.cache_data(show_spinner="Scoring every snap…", ttl=dt.REFRESH_HOURS * 3600)
def all_snaps(seasons: tuple[int, ...]) -> pd.DataFrame:
    keep = ["Season", "game_id", "posteam", "caller", "Call", "DN", "ydstogo", "epa", "success", "explosive",
            "Film confidence",
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


# ---- Plain-language pieces ------------------------------------------------------------------------------------
CSS = """<style>
.sp-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
.sp-card{border:1px solid rgba(128,128,128,.3);border-radius:10px;padding:16px;border-top:6px solid;display:flex;flex-direction:column;gap:6px}
.sp-tag{font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;opacity:.7}
.sp-head{font-size:28px;font-weight:800;text-transform:uppercase;line-height:1.05}
.sp-num{font-size:40px;font-weight:800;line-height:1;font-variant-numeric:tabular-nums}
.sp-sub{font-size:14px;opacity:.75}
.sp-callout{border:1px solid rgba(128,128,128,.3);border-left:6px solid;border-radius:10px;padding:16px;display:flex;flex-direction:column;gap:10px}
.sp-q{font-size:18px;font-weight:700}
.sp-mix{display:flex;height:30px;border-radius:6px;overflow:hidden}
.sp-mix span{display:flex;align-items:center;justify-content:center;color:#fff;font-size:12px;font-weight:600;white-space:nowrap;overflow:hidden}
.sp-mixlbl{font-size:14px;font-weight:600;margin-bottom:4px}
</style>"""


def size_text(v: float, outcome: str) -> str:
    """A difference in the outcome's own units: '-1.9 pts' (rates) or '-0.006 EPA'."""
    return f"{v * 100:+.1f} pts" if "%" in sp.PLAY_OUTCOMES[outcome][1] else f"{v:+.3f} EPA"


def level_text(v: float, outcome: str) -> str:
    return f"{v:.1%}" if "%" in sp.PLAY_OUTCOMES[outcome][1] else f"{v:+.3f}"


def verdicts_html(groups: pd.DataFrame, outcome: str) -> str:
    cards = []
    for call, color in (("Run", cm.RUN_C), ("Pass", cm.PASS_C)):
        v = sp.verdict(groups, call)
        noun = sp.plural(call)
        if pd.isna(v["diff"]):
            sub = "Too few plays here."
        else:
            sure = (f"Clear: about {abs(v['z']):.0f}× its margin of error." if v["clear"]
                    else "Within its margin of error.")
            sub = (f"{outcome} vs that offense's usual {call.lower()} ({level_text(v['value'], outcome)} vs "
                   f"{level_text(v['own'], outcome)}). {sure}")
        cards.append(f'<div class="sp-card" style="border-top-color:{color}"><span class="sp-tag">{noun.capitalize()} his '
                     f'habits gave away</span><span class="sp-head" style="color:{color}">{v["head"]}</span>'
                     f'<span class="sp-num">{"–" if pd.isna(v["diff"]) else size_text(v["diff"], outcome)}</span>'
                     f'<span class="sp-sub">{sub}</span></div>')
    return CSS + f'<div class="sp-row">{"".join(cards)}</div>'


def three_group_chart(g: pd.DataFrame, outcome: str) -> alt.LayerChart:
    fmt = sp.PLAY_OUTCOMES[outcome][1]
    g = g.assign(Calls=g["Call"].map(sp.plural).str.capitalize(),
                 label=g["Value"].map(lambda v: level_text(v, outcome)))
    x = alt.X("Group:N", sort=sp.GROUPS, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=13))
    color = alt.Color("Call:N", scale=alt.Scale(domain=["Run", "Pass"], range=[cm.RUN_C, cm.PASS_C]), legend=None)
    tip = ["Calls", "Group", alt.Tooltip("Value:Q", format=fmt, title=outcome),
           alt.Tooltip("Own:Q", format=fmt, title="That offense's usual"), alt.Tooltip("Plays:Q", format=",")]
    lo = float(min(g["Value"].min(), g["Own"].min()))
    span = float(max(g["Value"].max(), g["Own"].max())) - lo
    scale = alt.Scale(domain=[lo - span * 0.6, lo + span * 1.35], zero=False)
    base = alt.Chart(g).encode(x=x, tooltip=tip)
    floor = lo - span * 0.6
    bars = base.transform_calculate(floor=str(floor)).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        y=alt.Y("Value:Q", title=outcome, axis=alt.Axis(format=fmt), scale=scale), y2="floor:Q", color=color,
        opacity=alt.condition("datum.Group == 'Usual'", alt.value(0.55), alt.value(1)))
    tick = base.mark_tick(color=cm.theme_ink(), thickness=3, size=60).encode(y="Own:Q")
    text = base.mark_text(dy=-10, fontWeight="bold", color=cm.theme_ink()).encode(y="Value:Q", text="label:N")
    return alt.layer(bars, tick, text).properties(height=300, width=260).facet(
        column=alt.Column("Calls:N", sort=["Runs", "Passes"], title=None,
                          header=alt.Header(labelFontSize=15, labelFontWeight="bold")))


def per_game_sentence(pg: dict, outcome: str) -> str:
    if pd.isna(pg["diff"]) or pd.isna(pg["given_per_game"]):
        return ""
    base = (f"About {pg['given_per_game']:.0f} of a team's {pg['runs_per_game']:.0f} runs a game are ones its own "
            "habits gave away.")
    if outcome == "Success rate":
        lost = -pg["per_season"]
        what = ("one fewer successful run" if lost > 0 else "one more successful run")
        every = 17 / abs(pg["per_season"]) if pg["per_season"] else np.nan
        return (f"{base} They work {abs(pg['diff']) * 100:.1f} points {'less' if pg['diff'] < 0 else 'more'} often "
                f"than usual: about {what} every {every:.0f} games, or {abs(pg['per_season']):.1f} a season.")
    if outcome == "EPA/play":
        return (f"{base} They gain {abs(pg['diff']):.3f} EPA {'less' if pg['diff'] < 0 else 'more'} than usual: about "
                f"{abs(pg['per_season']):.1f} expected points a season.")
    return (f"{base} They go for an explosive gain {abs(pg['diff']) * 100:.1f} points "
            f"{'less' if pg['diff'] < 0 else 'more'} often than usual.")


def obvious_callout_html(d: pd.DataFrame) -> str:
    def mix_bar(call: str, colors: list[str]) -> tuple[str, float, float]:
        mix, success, toss = sp.obvious_mix(d, call)
        parts = "".join(f'<span title="{k}: {v:.0%}" style="width:{v * 100:.1f}%;background:{colors[i % len(colors)]};'
                        f'color:{"#16212B" if i > 1 else "#fff"}">{f"{k} {v:.0%}" if v >= 0.1 else ""}</span>'
                        for i, (k, v) in enumerate(mix.items()))
        return f'<div class="sp-mix">{parts}</div>', success, toss
    pbar, ps, pt = mix_bar("Pass", [cm.PASS_C, "#7A74C9", "#A9A4DE", "#C9C6EA"])
    rbar, rs, _ = mix_bar("Run", [cm.RUN_C, "#D3A15C", "#E6C899"])
    film_gap = abs((d["Film confidence"] - d["League confidence"]).mean()) * 100
    return CSS + (
        f'<div class="sp-callout" style="border-left-color:{cm.PASS_C}"><span class="sp-q">Does his film change this? '
        f'No.</span><span>Passes an average defense would see coming work {ps:.0%} of the time, against {pt:.0%} for '
        f'toss-ups, because many are 3rd & long. Runs everyone sees coming work <b>more</b> often ({rs:.0%}): they\'re '
        f'mostly short yardage. His own film moves the scout\'s confidence by about {film_gap:.1f} points on average, so '
        f'this is the situation, not the play caller. The section above takes the situation out.</span>'
        f'<div><div class="sp-mixlbl">Passes an average defense was 80%+ sure of (they worked {ps:.0%})</div>{pbar}</div>'
        f'<div><div class="sp-mixlbl">Runs an average defense was 80%+ sure of (they worked {rs:.0%})</div>{rbar}</div>'
        '</div>')


# ---- Page ----------------------------------------------------------------------------------------------------
seasons = dt.available_seasons()
info = ui.team_info()
last_full = seasons[-2] if len(seasons) > 1 else seasons[-1]
with st.sidebar:
    st.header("Surprise Plays")
    span = st.slider("Seasons", seasons[0], seasons[-1], (max(seasons[0], last_full - 4), last_full), key="sp_span",
                     help="Every film-scored snap (each play caller's 4th game on) in these regular seasons.")
    outcome = st.selectbox("Did the play work? Measured by", list(sp.PLAY_OUTCOMES), index=1, key="sp_outcome",
                           help="Success rate (the play gained ground for its down) is the plainest; EPA per play "
                                "weighs big plays more.")
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
groups = sp.three_groups(d, outcome)
st.html(verdicts_html(groups, outcome))
st.altair_chart(three_group_chart(groups, outcome), width="stretch")
pg = sp.per_game(d, outcome)
st.caption(per_game_sentence(pg, outcome) + f" {len(d):,} snaps, {years}. \"Gave it away\" = his own habits made the "
           f"call {sp.GAVE_AWAY * 100:.0f}+ points more predictable than league norms; \"hid it\" = {sp.GAVE_AWAY * 100:.0f}+ "
           "points less. Black tick = the same offenses' usual run or pass that season.")
with st.expander("Show the math"):
    for call in ("Run", "Pass"):
        st.markdown(sp.tell_sentence(sp.tell_trend(d, call, outcome, within), call, outcome, within))
    st.altair_chart(tell_chart(sp.by_tell(d, outcome, within), outcome, within), width="stretch")
    st.caption("Five groups by how much his film moved the scout's confidence in the call that came, compared with "
               "league norms alone. " + ("Bars = the play against that offense's average run or pass that season "
               "(0 = his usual)." if within else "Bars = the raw average.") + " Lines = ±2 standard errors: groups whose "
               "lines cross 0 aren't clearly different from usual. The sentences fit a line through every play.")

st.subheader("Calls the defense saw coming")
st.html(obvious_callout_html(d))
with st.expander("Show the chart"):
    st.altair_chart(confidence_chart(sp.by_confidence(d, outcome), outcome), width="stretch")
    st.caption("Raw averages by how sure each scout was. Solid = his film; dashed = league norms. The two lines nearly "
               "match: this is the situation, not the coach.")

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
