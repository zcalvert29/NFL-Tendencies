"""
3rd and Medium page: on 3rd down, do runs or passes win more series once the 4th-down play after a short 3rd down
counts? The series logic lives in series.py; this page draws it.
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import common as cm
import data as dt
import series as se

CALL_COLORS = {"Pass": cm.PASS_C, "Run": cm.RUN_C}
OUTCOME_COLORS = dict(zip(se.OUTCOMES, ["#2F8C86", "#8CC5BF", "#C8321E", "#A9B1B5", "#7A1F12"]))
CSS = """<style>
.sr-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}
.sr-card{border:1px solid rgba(128,128,128,.3);border-radius:12px;padding:16px;border-top:6px solid;display:flex;flex-direction:column;gap:12px}
.sr-tag{font-size:28px;font-weight:800;text-transform:uppercase;line-height:1}
.sr-sub{opacity:.7;font-size:14px}
.sr-m{display:grid;grid-template-columns:150px 1fr 52px;gap:10px;align-items:center;font-size:15px}
.sr-v{font-size:24px;font-weight:800;text-align:right;font-variant-numeric:tabular-nums}
.sr-track{height:14px;background:rgba(128,128,128,.18);border-radius:7px;position:relative;overflow:hidden}
.sr-track i{position:absolute;top:0;bottom:0;left:0;border-radius:7px}
.sr-why{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}
.sr-stat{border:1px solid rgba(128,128,128,.3);border-radius:10px;padding:12px 14px;display:flex;flex-direction:column;gap:6px}
.sr-lbl{font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;opacity:.7}
.sr-pair{display:flex;gap:18px;align-items:baseline}
.sr-pair span{font-size:34px;font-weight:800;line-height:1;font-variant-numeric:tabular-nums}
.sr-pair small{display:block;font-size:11px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;opacity:.7;margin-top:3px}
.sr-note{font-size:13px;opacity:.75}
.sr-ba{display:flex;align-items:baseline;gap:10px;font-weight:800}
.sr-a{font-size:26px;opacity:.6}
.sr-arrow{font-size:18px;opacity:.6}
.sr-b{font-size:38px}
</style>"""


@st.cache_data(show_spinner="Following every 3rd down to the end of its series…", ttl=dt.REFRESH_HOURS * 3600)
def third_downs(seasons: tuple[int, ...]) -> pd.DataFrame:
    return se.load(list(seasons))


@st.cache_data(show_spinner=False)
def team_info() -> pd.DataFrame:
    try:
        info = dt.load_teams()
    except Exception:
        return pd.DataFrame(columns=["team_name", "team_color", "logo"])
    return info.assign(logo=pd.Series(dt.team_logos(info)))


def pct(v) -> str:
    return "–" if pd.isna(v) else f"{v:.0%}"


def calls_html(s: pd.DataFrame, span: str) -> str:
    cards = []
    for _, r in s.iterrows():
        c = CALL_COLORS[r["Call"]]
        cards.append(
            f'<div class="sr-card" style="border-top-color:{c}"><div><div class="sr-tag" style="color:{c}">{r["Call"]}'
            f'</div><div class="sr-sub">{int(r["Plays"]):,} calls, {span}</div></div>'
            f'<div class="sr-m"><span>Moved the chains on 3rd</span><div class="sr-track"><i style="width:'
            f'{r["Converted on 3rd"] * 100:.0f}%;background:{c};opacity:.45"></i></div>'
            f'<span class="sr-v">{pct(r["Converted on 3rd"])}</span></div>'
            f'<div class="sr-m"><span><b>Won the series</b> (3rd + 4th)</span><div class="sr-track"><i style="width:'
            f'{r["Won the series"] * 100:.0f}%;background:{c}"></i></div>'
            f'<span class="sr-v" style="color:{c}">{pct(r["Won the series"])}</span></div></div>')
    return CSS + f'<div class="sr-row">{"".join(cards)}</div>'


def why_html(s: pd.DataFrame) -> str:
    r = s.set_index("Call")
    get = lambda call, col: r[col].get(call, np.nan)   # noqa: E731 - a short lookup

    def box(label, col, fmt, note):
        return (f'<div class="sr-stat"><span class="sr-lbl">{label}</span><div class="sr-pair">'
                + "".join(f'<div><span style="color:{CALL_COLORS[c]}">{fmt(get(c, col))}</span><small>{c}</small></div>'
                          for c in ("Run", "Pass")) + f'</div><div class="sr-note">{note}</div></div>')
    num = lambda v: "–" if pd.isna(v) else f"{v:.0f}"   # noqa: E731
    return CSS + '<div class="sr-why">' + "".join([
        box("Short 3rd leaves 4th & 1-2", "Left 4th & 1-2", pct,
            "A stopped run still gains a few yards. A stopped pass is usually an incompletion: 4th & the same distance."),
        box("Typical 4th & ...", "Typical 4th & ...", num, "Median yards to go on 4th after the 3rd down came up short."),
        box("Went for it on 4th", "Go-for-it rate", pct, "After a short 3rd down."),
        box("Converted when going for it", "Converted on 4th", pct, "On the 4th-down play itself."),
    ]) + "</div>"


def flow_chart(d: pd.DataFrame) -> alt.LayerChart:
    """One stacked bar per call: how its series ended, as shares of the calls."""
    rows = []
    for call, g in d.groupby("Call"):
        left = 0.0
        for i, (o, v) in enumerate(se.outcome_shares(g).items()):
            rows.append({"Call": call, "Outcome": o, "Share": v, "k": i, "start": left, "end": left + v,
                         "mid": left + v / 2})
            left += v
    f = pd.DataFrame(rows)
    y = alt.Y("Call:N", title=None, sort=["Pass", "Run"], axis=alt.Axis(labelFontSize=15, labelFontWeight="bold"))
    bars = alt.Chart(f).mark_bar().encode(
        y=y, x=alt.X("start:Q", title="Share of the calls", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1])),
        x2="end:Q", tooltip=["Call", "Outcome", alt.Tooltip("Share:Q", format=".1%")],
        color=alt.Color("Outcome:N", sort=se.OUTCOMES, legend=alt.Legend(title=None, orient="bottom", columns=5),
                        scale=alt.Scale(domain=list(OUTCOME_COLORS), range=list(OUTCOME_COLORS.values()))))
    text = alt.Chart(f[f["Share"] >= 0.06]).mark_text(color="white", fontWeight="bold").encode(
        y=y, x="mid:Q", text=alt.Text("Share:Q", format=".0%"))
    return alt.layer(bars, text).properties(height=170)


def split_chart(s: pd.DataFrame, group: str, order: list[str], title: str) -> alt.Chart:
    """For each group and call: converted on 3rd (ring) -> won the series (dot); the line is what 4th down added."""
    s = s.assign(label=s[group]).dropna(subset=["Converted on 3rd"])
    y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelOverlap=False, labelLimit=200))
    off = alt.YOffset("Call:N", sort=["Pass", "Run"])
    color = alt.Color("Call:N", scale=alt.Scale(domain=list(CALL_COLORS), range=list(CALL_COLORS.values())),
                      legend=alt.Legend(title=None, orient="top"))
    tip = ["label:N", "Call", alt.Tooltip("Plays:Q", format=","), alt.Tooltip("Converted on 3rd:Q", format=".0%"),
           alt.Tooltip("Won the series:Q", format=".0%"), alt.Tooltip("Go-for-it rate:Q", format=".0%")]
    base = alt.Chart(s).encode(y=y, yOffset=off, color=color, tooltip=tip)
    x = alt.X("Converted on 3rd:Q", axis=alt.Axis(format=".0%", title=title), scale=alt.Scale(zero=False))
    return alt.layer(
        base.mark_rule(strokeWidth=3).encode(x=x, x2="Won the series:Q"),
        base.mark_point(filled=False, size=90, strokeWidth=2.5).encode(x="Converted on 3rd:Q"),
        base.mark_circle(size=110, opacity=1).encode(x="Won the series:Q"),
    ).properties(height=70 * len(order) + 30)


ERAS = [(1999, 2005), (2006, 2012), (2013, 2019), (2020, None)]   # None = through the latest season


def era_table(d: pd.DataFrame, last: int) -> pd.DataFrame:
    """summary() for each era and call, with the era's label."""
    rows = []
    for first, end in ERAS:
        end = end or last
        g = d[d["Season"].between(first, end)]
        for call, h in g.groupby("Call"):
            rows.append({"Era": f"{first}-{end}", "Call": call, **se.summary(h).to_dict()})
    return pd.DataFrame(rows)


def era_tiles_html(e: pd.DataFrame) -> str:
    """Three before-and-after numbers: first era vs latest era."""
    first, last = e["Era"].iloc[0], e["Era"].iloc[-1]
    get = lambda era, call, col: e[(e["Era"] == era) & (e["Call"] == call)][col].iat[0]   # noqa: E731

    def tile(label, a, b, color, sub):
        return (f'<div class="sr-stat"><span class="sr-lbl">{label}</span><div class="sr-ba"><span class="sr-a">{a}'
                f'</span><span class="sr-arrow">→</span><span class="sr-b" style="color:{color}">{b}</span></div>'
                f'<div class="sr-note">{sub}</div></div>')
    return CSS + '<div class="sr-why">' + "".join([
        tile("Runs won the series", pct(get(first, "Run", "Won the series")), pct(get(last, "Run", "Won the series")),
             cm.RUN_C, f"{first} → {last}"),
        tile("Passes won the series", pct(get(first, "Pass", "Won the series")),
             pct(get(last, "Pass", "Won the series")), cm.PASS_C, f"{first} → {last}"),
        tile("Went for it after a short run", pct(get(first, "Run", "Go-for-it rate")),
             pct(get(last, "Run", "Go-for-it rate")), "#2F8C86", "The reason: teams now use 4th down"),
    ]) + "</div>"


def era_chart(e: pd.DataFrame) -> alt.LayerChart:
    """Run vs pass series won, by era; the go-for-it rate after a short run under each era."""
    go = e[e["Call"] == "Run"].set_index("Era")["Go-for-it rate"]
    e = e.assign(label=e["Era"] + "|went for it after a short run: " + e["Era"].map(go).map(pct))
    order = list(dict.fromkeys(e["label"]))
    x = alt.X("label:N", sort=order, title=None,
              axis=alt.Axis(labelAngle=0, labelExpr="split(datum.label, '|')", labelFontSize=12, labelPadding=8,
                            labelLimit=320))
    base = alt.Chart(e).encode(x=x, xOffset=alt.XOffset("Call:N", sort=["Run", "Pass"]),
                               tooltip=["Era", "Call", alt.Tooltip("Won the series:Q", format=".0%"),
                                        alt.Tooltip("Plays:Q", format=","), alt.Tooltip("Go-for-it rate:Q", format=".0%")])
    lo = max(0.0, round(float(e["Won the series"].min()) - 0.1, 1))   # bars start at the axis floor, not 0
    base = base.transform_calculate(floor=str(lo))
    bars = base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        y=alt.Y("Won the series:Q", title="Won the series", axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domain=[lo, min(1.0, float(e["Won the series"].max()) + 0.06)])), y2="floor:Q",
        color=alt.Color("Call:N", scale=alt.Scale(domain=["Run", "Pass"], range=[cm.RUN_C, cm.PASS_C]),
                        legend=alt.Legend(title=None, orient="top")))
    text = base.mark_text(dy=-8, fontWeight="bold", color=cm.theme_ink()).encode(
        y="Won the series:Q", text=alt.Text("Won the series:Q", format=".0%"))
    return alt.layer(bars, text).properties(height=340)


# ---- Page ----------------------------------------------------------------------------------------------------
seasons = dt.available_seasons()
info = team_info()
last_full = seasons[-2] if len(seasons) > 1 else seasons[-1]
with st.sidebar:
    st.header("3rd and Medium")
    span = st.slider("Seasons", seasons[0], seasons[-1], (max(seasons[0], last_full - 4), last_full), key="sr_span",
                     help="The league numbers use every regular season in this range.")
    togo = st.slider("3rd & … to go", 1, 15, (4, 6), key="sr_togo", help="Default: 3rd & medium (4-6 yards).")
    zones = st.multiselect("Field position", se.ZONES, default=se.ZONES, format_func=se.ZONE_NAMES.get, key="sr_zones")
    min_plays = st.slider("Play callers: minimum snaps", 5, 60, 15, 5, key="sr_min")

dist = f"3rd & {togo[0]}" if togo[0] == togo[1] else f"3rd & {togo[0]}-{togo[1]}"
years = f"{span[0]}" if span[0] == span[1] else f"{span[0]}-{span[1]}"
st.title(f"{dist}: one down or two?")
st.markdown(f"Most coaches judge a {dist} call by whether it moves the chains. Here every call is judged by the whole "
            "**series**: did the 3rd-down play, plus the 4th-down play after it if the team went for it, get a first "
            "down?")
try:
    all_d = third_downs(tuple(range(span[0], span[1] + 1)))
except Exception as e:
    st.error(f"Couldn't load nflfastR data: {e}")
    st.stop()
d = all_d[all_d["ydstogo"].between(*togo) & all_d["Zone"].isin(zones or se.ZONES)]
if d.empty:
    st.info("No 3rd downs match these filters.")
    st.stop()
league = se.by(d, ["Call"])
st.html(calls_html(league, years))
if league["Plays"].min() < 100:
    st.warning(f"Only {int(league['Plays'].min())} {league.loc[league['Plays'].idxmin(), 'Call'].lower()} calls here, so "
               "these rates can swing several points either way. Widen the seasons or the distance.")

st.subheader("How the series plays out")
st.altair_chart(flow_chart(d), width="stretch")
st.caption("Share of all the calls. Kicked = punted or tried a field goal on 4th (or the half ended).")

st.subheader("Why")
st.html(why_html(league))

st.subheader("By distance and field position")
c1, c2 = st.columns(2)
with c1:
    by_dist = se.by(d, ["ydstogo", "Call"]).assign(ydstogo=lambda x: "3rd & " + x["ydstogo"].astype(int).astype(str))
    order = [f"3rd & {n}" for n in range(togo[0], togo[1] + 1)]
    st.altair_chart(split_chart(by_dist, "ydstogo", order, "Converted on 3rd → won the series"), width="stretch")
with c2:
    by_zone = se.by(d, ["Zone", "Call"]).assign(Zone=lambda x: x["Zone"].map(se.ZONE_NAMES))
    st.altair_chart(split_chart(by_zone, "Zone", [se.ZONE_NAMES[z] for z in se.ZONES],
                                "Converted on 3rd → won the series"), width="stretch")
st.caption("Ring = converted on 3rd; dot = won the series; the line between them is what 4th down added. Hover for the "
           "number of calls and how often teams went for it.")

st.subheader("Over the years")
everything = third_downs(tuple(seasons))
eras = era_table(everything[everything["ydstogo"].between(*togo) & everything["Zone"].isin(zones or se.ZONES)],
                 seasons[-1])
if eras["Era"].nunique() == len(ERAS) and eras.groupby("Era")["Call"].nunique().min() == 2:
    st.html(era_tiles_html(eras))
    st.altair_chart(era_chart(eras), width="stretch")
    st.caption(f"Every regular season since {seasons[0]}, whatever the seasons picked in the sidebar ({dist}, the "
               "field positions picked). As teams went for it on 4th more often, runs started winning more series: a "
               "short run usually leaves 4th & short. Runs did worse in 2006-2012, before that change.")

st.subheader("Play callers")
caller_seasons = sorted(d["Season"].unique())[::-1]
pick = st.selectbox("Season", caller_seasons, key="sr_caller_season")
ct = se.caller_table(d[d["Season"] == pick], min_plays=min_plays)
if ct.empty:
    st.info(f"No play caller has {min_plays}+ of these snaps in {pick}. Lower the minimum in the sidebar.")
else:
    ct = ct.sort_values("Run share", ascending=False)
    w = lambda g, col: (g[col] * g["Plays"]).sum() / g["Plays"].sum()   # noqa: E731
    hi, lo = ct[ct["Run share"] >= 0.12], ct[ct["Run share"] < 0.12]
    if len(hi) and len(lo):
        st.markdown(f"In {pick}, the {len(hi)} play callers who ran 12%+ of the time won {w(hi, 'Won the series'):.0%} "
                    f"of these series and went for it on {w(hi, 'Go-for-it rate'):.0%} of short 3rd downs; the other "
                    f"{len(lo)} won {w(lo, 'Won the series'):.0%} and went for it on {w(lo, 'Go-for-it rate'):.0%}.")
    unlisted = ct["Play caller"].str.endswith(" play caller").all()
    show = pd.DataFrame({
        "Logo": ct["Team"].map(info.get("logo", {})).fillna(""), "Play caller": ct["Team"] if unlisted else ct["Play caller"],
        "Team": ct["Team"], "Snaps": ct["Plays"].astype(int), "Run share": ct["Run share"] * 100,
        "Moved the chains on 3rd": ct["Converted on 3rd"], "Won the series": ct["Won the series"],
        "Went for it on 4th": ct["Go-for-it rate"], "Runs": ct["Runs"].astype(int),
        # As text: Streamlit draws a missing number as "None", and a coach with no runs has no rate.
        "Runs: won the series": ct["Runs: won the series"].map(lambda v: "–" if pd.isna(v) else f"{v:.0%}")})
    st.dataframe(
        show.style.format({c: "{:.0%}" for c in ["Moved the chains on 3rd", "Won the series", "Went for it on 4th"]},
                          na_rep="–"),
        width="stretch", hide_index=True, height=36 * (len(show) + 1) + 2,
        column_config={"Logo": st.column_config.ImageColumn("", width="small"),
                       "Run share": st.column_config.ProgressColumn("Run share", format="%.0f%%", min_value=0,
                                                                     max_value=40),
                       "Went for it on 4th": st.column_config.Column(
                           help="How often he went for it on 4th after the 3rd down came up short."),
                       "Runs: won the series": st.column_config.Column(
                           help="Series won on his 3rd-down runs only. Often a handful of plays.")})
    st.caption(f"Play callers with {min_plays}+ of these snaps in {pick}, most runs first. One season gives a coach "
               "only a few dozen of these snaps, so one or two plays move his numbers a lot.")

with st.expander("Read with care"):
    st.markdown(
        "- **The run's edge is mostly the plan, not the play.** Coaches who run here have usually already decided to "
        "use 4th down, and they go for it far more often after a short run than others do after a short pass. A pass by "
        "a coach who punts can never win the series on 4th, so this measures the two-down plan as much as the call.\n"
        "- **Runs are rare here.** A season has a few hundred 3rd & medium runs across the league against thousands of "
        "passes; check the call counts at the top.\n"
        f"- **Counted:** regular-season 3rd downs outside the last 2 minutes of each half, score within {se.MAX_MARGIN}. "
        "Scrambles and sacks count as passes (the call was a pass). Each 3rd down is followed to the next 4th-down run, "
        "pass, punt, or field goal in the same drive; penalties and timeouts in between are skipped.")
