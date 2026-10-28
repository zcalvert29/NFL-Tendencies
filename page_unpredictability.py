"""
Unpredictability Index page: which play callers a defense can read from their film, and why. The model lives in
unpredictability.py; this page draws it. Every chart uses team colors and logos.
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import common as cm
import data as dt
import unpredictability as up

LOGO = 26          # logo size on the bar chart, px
PROFILES = {"Hard to read": cm.PASS_C, "NFL average": "#888780", "Readable": "#D85A30", "Too few snaps": "#B4B2A9"}
LOOK_MIN_EDGE = 0.3  # a look's edge share (mbits per snap) must pass this to count as a tell or a miss


@st.cache_data(show_spinner="Loading nflfastR play-by-play…", ttl=dt.REFRESH_HOURS * 3600)
def plays(season: int) -> pd.DataFrame:
    return dt.load_seasons([season])


@st.cache_data(show_spinner="Scouting every play caller's film…", ttl=dt.REFRESH_HOURS * 3600)
def scored(season: int, weeks: tuple[int, int] | None, by: str) -> tuple[pd.DataFrame, list[str]]:
    d, clues = up.score(plays(season), season, weeks, by=by)
    keep = ["posteam", "caller", "Week", "game_id", "PASS", "pass_oe", "epa", "scored", "xpass"]
    keep += [c for c in d.columns if c.startswith(("p_", "gain_", "bits_"))]
    keep += [c for k in clues for c in up.CLUES[k][1]]
    return d[keep], clues


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


def with_team(t: pd.DataFrame, info: pd.DataFrame, highlight: list[str], by: str) -> pd.DataFrame:
    """The caller table plus what every chart needs: color (readable on this theme), logo, label, opacity."""
    d = t.copy()
    dark = cm.theme_ink() == "#FAFAFA"
    color, color2 = d["Team"].map(info.get("team_color", {})), d["Team"].map(info.get("team_color2", {}))
    d["color"] = [c2 if dark and _luminance(c) < 0.06 and isinstance(c2, str) else c if isinstance(c, str) else "#888780"
                  for c, c2 in zip(color, color2)]
    d["logo"] = d["Team"].map(info.get("logo", {})).fillna("")
    d["Team name"] = d["Team"].map(info.get("team_name", {})).fillna(d["Team"])
    last = d["Play caller"].str.split().str[-1]
    unlisted = d["Play caller"].str.endswith(" play caller")      # no row in play_callers.csv for this team-season
    d["label"] = np.where((by == "team") | unlisted, d["Team"], last + " (" + d["Team"] + ")")
    d["opacity"] = np.where(d["Team"].isin(highlight) | (len(highlight) == 0), 1.0, 0.25)
    d["Film edge"] = d["Edge"] * 100                      # bits per 100 snaps
    d["Lean part"], d["Look part"] = d["Lean edge"] * 100, d["Look edge"] * 100
    d["Accuracy gain"] = d["Film accuracy"] - d["Generic accuracy"]
    return d


# ---- Charts --------------------------------------------------------------------------------------------------
TIP = [alt.Tooltip("Play caller:N"), alt.Tooltip("Team name:N", title="Team"), "Weeks:N", "Profile:N",
       alt.Tooltip("Unpredictability index:Q", format=".0f"),
       alt.Tooltip("Film edge:Q", format="+.2f", title="Film edge (bits / 100 snaps)"),
       alt.Tooltip("Lean part:Q", format="+.2f", title="  from his overall lean"),
       alt.Tooltip("Look part:Q", format="+.2f", title="  from situations & looks"),
       alt.Tooltip("Lean:Q", format="+.1%", title="Lean (pass rate vs NFL scout)"),
       alt.Tooltip("Uncertainty removed:Q", format="+.1%", title="Uncertainty film removes"),
       alt.Tooltip("Generic accuracy:Q", format=".1%", title="NFL-scout accuracy"),
       alt.Tooltip("Film accuracy:Q", format=".1%", title="Film-scout accuracy"),
       "Biggest tell:N", alt.Tooltip("Scored snaps:Q", format=",")]


def index_chart(d: pd.DataFrame) -> alt.LayerChart:
    """Bars from 50 (an average play caller): up = harder to read, down = easier. Logo at the end of each bar."""
    order = list(d.sort_values("Unpredictability index", ascending=False)["label"])
    lo, hi = float(d["Unpredictability index"].min()), float(d["Unpredictability index"].max())
    pad = (hi - lo) * 0.12 + 2
    d = d.assign(base=50.0)
    x = alt.X("label:N", sort=order, title=None, axis=alt.Axis(labelAngle=-50, labelFontSize=10, labelLimit=120,
                                                                labelOverlap=False))
    y = alt.Y("Unpredictability index:Q", title="Unpredictability index (50 = NFL average)",
              scale=alt.Scale(domain=[min(lo - pad, 45), max(hi + pad, 55)]))
    base = alt.Chart(d).encode(x=x, tooltip=TIP)
    bars = base.mark_bar(cornerRadius=3).encode(y=y, y2="base:Q", color=alt.Color("color:N", scale=None),
                                                opacity=alt.Opacity("opacity:Q", scale=None))
    up_ = base.transform_filter("datum['Unpredictability index'] >= 50").mark_image(
        width=LOGO, height=LOGO, baseline="bottom").encode(y="Unpredictability index:Q", url="logo:N",
                                                           opacity=alt.Opacity("opacity:Q", scale=None))
    down = base.transform_filter("datum['Unpredictability index'] < 50").mark_image(
        width=LOGO, height=LOGO, baseline="top").encode(y="Unpredictability index:Q", url="logo:N",
                                                        opacity=alt.Opacity("opacity:Q", scale=None))
    rules = alt.Chart(pd.DataFrame({"v": [40, 50, 60]})).mark_rule(color="#888780").encode(
        y="v:Q", strokeDash=alt.condition("datum.v == 50", alt.value([1, 0]), alt.value([4, 4])))
    return alt.layer(rules, bars, up_, down).properties(height=400)


def clue_heatmap(d: pd.DataFrame, clues: list[str]) -> alt.Chart:
    """Callers (most readable at the top) x clues: how much each clue alone lets the film scout read them."""
    labels = [up.CLUES[k][0] for k in clues]
    long = d.melt(id_vars=["label", "Play caller", "Team name", "Unpredictability index"], value_vars=labels,
                  var_name="Clue", value_name="edge")
    long["edge"] *= 100
    order = list(d.sort_values("Unpredictability index")["label"])
    lim = float(np.nanpercentile(long["edge"].abs(), 95)) or 0.5
    rect = alt.Chart(long).mark_rect(cornerRadius=2).encode(
        x=alt.X("Clue:N", sort=labels, title=None, axis=alt.Axis(labelAngle=0, orient="top", labelLimit=160,
                                                                   labelOverlap=False, labelFontSize=11)),
        y=alt.Y("label:N", sort=order, title=None),
        color=alt.Color("edge:Q", title="Bits / 100 snaps",
                        scale=alt.Scale(scheme="redblue", reverse=True, domain=[-lim, lim], clamp=True)),
        tooltip=["Play caller", alt.Tooltip("Team name:N", title="Team"), "Clue",
                 alt.Tooltip("edge:Q", format="+.2f", title="Film edge from this clue alone (bits / 100 snaps)")])
    text = alt.Chart(long).mark_text(fontSize=10).encode(
        x=alt.X("Clue:N", sort=labels), y=alt.Y("label:N", sort=order), text=alt.Text("edge:Q", format="+.1f"),
        color=alt.condition(f"abs(datum.edge) > {lim * 0.6}", alt.value("white"), alt.value(cm.theme_ink())))
    return alt.layer(rect, text).properties(height=max(22 * len(d), 200))


def logo_scatter(d: pd.DataFrame, x: str, y: str, x_title: str, y_title: str, x_fmt: str, y_fmt: str,
                 trend: bool = False, ref: tuple[float | None, float | None] = (None, None)) -> alt.LayerChart:
    """One play caller per point, drawn as the team logo (a dot in the team's color if the logo is missing)."""
    d = d.dropna(subset=[x, y])
    # Name only the points at the edges (the middle of the cloud overlaps); hover names the rest.
    rank = lambda s: s.rank(pct=True)
    d = d.assign(show_label=(rank(d[x]) - 0.5).abs().ge(0.4) | (rank(d[y]) - 0.5).abs().ge(0.4))
    pad_x, pad_y = (d[x].max() - d[x].min()) * 0.08 or 0.01, (d[y].max() - d[y].min()) * 0.08 or 0.01
    base = alt.Chart(d).encode(
        x=alt.X(f"{x}:Q", title=x_title, scale=alt.Scale(domain=[d[x].min() - pad_x, d[x].max() + pad_x]),
                axis=alt.Axis(format=x_fmt, tickCount=8)),
        y=alt.Y(f"{y}:Q", title=y_title, scale=alt.Scale(domain=[d[y].min() - pad_y, d[y].max() + pad_y]),
                axis=alt.Axis(format=y_fmt, tickCount=8)), tooltip=TIP)
    layers = []
    for axis, v in zip("xy", ref):
        if v is not None:
            layers.append(alt.Chart(pd.DataFrame({"v": [v]})).mark_rule(color="#888780", strokeDash=[4, 4])
                          .encode(**{axis: alt.X("v:Q", scale=alt.Scale(zero=False)) if axis == "x" else
                                     alt.Y("v:Q", scale=alt.Scale(zero=False))}))
    if trend:
        layers.append(base.transform_regression(x, y).mark_line(color="#888780", strokeDash=[6, 4], strokeWidth=2)
                      .encode(tooltip=alt.value(None)))
    layers += [
        base.transform_filter("datum.logo == ''").mark_circle(size=500).encode(
            color=alt.Color("color:N", scale=None), opacity=alt.Opacity("opacity:Q", scale=None)),
        base.mark_image(width=36, height=36).encode(url="logo:N", opacity=alt.Opacity("opacity:Q", scale=None)),
        base.transform_filter("datum.show_label").mark_text(dy=24, fontSize=10, color=cm.theme_ink()).encode(
            text="label:N", opacity=alt.Opacity("opacity:Q", scale=None)),
    ]
    return alt.layer(*layers).properties(height=460)


def history_chart(h: pd.DataFrame) -> alt.Chart:
    """Each play caller's index in each season he called plays (dots joined by a line)."""
    base = alt.Chart(h).encode(
        x=alt.X("Season:O", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Unpredictability index:Q", title="Unpredictability index", scale=alt.Scale(zero=False)),
        detail="Play caller:N", tooltip=["Play caller", "Team", "Season", alt.Tooltip("Unpredictability index:Q",
                                                                                      format=".0f"), "Profile"])
    line = base.mark_line(color="#888780", opacity=0.5)
    dots = base.mark_circle(size=90, opacity=1).encode(color=alt.Color("color:N", scale=None))
    rule = alt.Chart(pd.DataFrame({"v": [50]})).mark_rule(color="#888780", strokeDash=[4, 4]).encode(y="v:Q")
    return alt.layer(rule, line, dots).properties(height=380)


# ---- Tables ---------------------------------------------------------------------------------------------------
def comparison_table(d: pd.DataFrame, clues: list[str]) -> tuple[pd.DataFrame, dict]:
    out = pd.DataFrame({"Logo": d["logo"], "Play caller": d["Play caller"], "Team": d["Team"], "Weeks": d["Weeks"],
                        "Unpredictability index": d["Unpredictability index"], "Profile": d["Profile"],
                        "Film edge": d["Film edge"], "± (2 SE)": 2 * d["Edge SE"] * 100,
                        "From lean": d["Lean part"], "From looks": d["Look part"], "Lean": d["Lean"],
                        "Uncertainty removed": d["Uncertainty removed"],
                        "NFL-scout accuracy": d["Generic accuracy"], "Film-scout accuracy": d["Film accuracy"],
                        "Biggest tell": d["Biggest tell"],
                        **{up.CLUES[k][0]: d[up.CLUES[k][0]] * 100 for k in clues},
                        "League tells": d["League tells"] * 100, "Pass rate": d["Pass rate"], "PROE": d["PROE"],
                        "EPA/play": d["EPA/play"], "Games": d["Games"], "Scored snaps": d["Scored snaps"]})
    fmt = {"Unpredictability index": "{:.0f}", "Film edge": "{:+.2f}", "± (2 SE)": "{:.2f}", "From lean": "{:+.2f}",
           "From looks": "{:+.2f}", "Lean": "{:+.1%}",
           "Uncertainty removed": "{:+.1%}", "NFL-scout accuracy": "{:.1%}", "Film-scout accuracy": "{:.1%}",
           "League tells": "{:+.1f}", "Pass rate": "{:.1%}", "PROE": "{:+.1%}", "EPA/play": "{:+.3f}",
           "Scored snaps": "{:,.0f}", **{up.CLUES[k][0]: "{:+.2f}" for k in clues}}
    return out, fmt


HELP = {
    "Unpredictability index": "50 = an average NFL play caller this season; every 10 points = one standard deviation. "
                              "Higher = his own film tells a defense less.",
    "Film edge": "Bits per 100 snaps a defense saves by studying this caller's earlier games, beyond knowing how the NFL "
                 "calls the same down, distance, score, and look. Out of sample: tendencies that don't repeat cost bits.",
    "± (2 SE)": "Two standard errors on the film edge (clustered by game). An edge inside ± this is not clearly "
                "different from 0.",
    "From lean": "The part of the film edge from his overall lean alone: knowing from his earlier games that he throws "
                 "more (or less) than the NFL scout expects, in any look.",
    "From looks": "The rest of the film edge: his habits by down & distance, field position, score, formation, "
                  "personnel, motion, backfield, and last call, beyond his lean.",
    "Lean": "How much more often he threw than the NFL scout expected (situation and look), on scored snaps.",
    "Uncertainty removed": "Film edge as a share of the uncertainty the NFL-only scout has left.",
    "NFL-scout accuracy": "How often a guess from league norms alone (situation + look) gets run/pass right.",
    "Film-scout accuracy": "How often the same guess, adjusted by this caller's own earlier film, gets it right.",
    "Biggest tell": "The clue that adds the most to his lean (None = no clue adds anything).",
    "League tells": "Bits per 100 snaps the look (formation, personnel, motion, backfield, last call) tells any scout "
                    "beyond down, distance, and score, the way it does for every NFL team. Not in the index.",
    "PROE": "Pass rate over expected (nflfastR).",
}


def color_profile(col):
    return [f"color:{PROFILES.get(v, 'inherit')};font-weight:600" for v in col]


def shade_edge(col):
    lim = max(float(np.nanpercentile(np.abs(col.astype(float)), 95)), 1e-9)
    out = []
    for v in col:
        a = 0 if pd.isna(v) else min(abs(float(v)) / lim, 1)
        rgb = "216,90,48" if v > 0 else "83,74,183"
        out.append(f"background-color:rgba({rgb},{a * 0.45:.2f})" if a >= 0.15 else "")
    return out


def shade_index(col):
    """Orange below 50 (readable), purple above (hard to read), stronger the farther from 50."""
    out = []
    for v in col:
        a = 0 if pd.isna(v) else min(abs(float(v) - 50) / 25, 1)
        rgb = "216,90,48" if not pd.isna(v) and v < 50 else "83,74,183"
        out.append(f"background-color:rgba({rgb},{a * 0.5:.2f})" if a >= 0.15 else "")
    return out


def look_display(lt: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"Clue": lt["Clue"], "Look": lt["Look"], "Snaps": lt["Snaps"], "Share of snaps": lt["Share"],
                         "Pass rate": lt["Pass rate"], "NFL scout expected": lt["NFL scout expected"],
                         "Their film said": lt["Their film said"], "Edge (bits / 100 snaps)": lt["Edge share"] / 10})


LOOK_FMT = {"Share of snaps": "{:.0%}", "Pass rate": "{:.0%}", "NFL scout expected": "{:.0%}",
            "Their film said": "{:.0%}", "Edge (bits / 100 snaps)": "{:+.2f}"}


def why_summary(r: pd.Series, lt: pd.DataFrame, clues: list[str]) -> str:
    """Two or three sentences on why this caller is readable or not."""
    name = r["Play caller"]
    lift = r["Film accuracy"] - r["Generic accuracy"]
    if r["Profile"] == "Readable":
        head = (f"**{name} is readable.** His earlier games let a defense call run/pass right {r['Film accuracy']:.0%} "
                f"of the time ({lift * 100:+.1f} points vs league norms alone), and the film removes {r['Uncertainty removed']:.1%} "
                f"of the uncertainty that's left after down, distance, and look.")
    elif r["Profile"] == "Hard to read":
        head = (f"**{name} is hard to read.** Studying his earlier games doesn't help a defense: a guess from his film is "
                f"right {r['Film accuracy']:.0%} of the time ({lift * 100:+.1f} points vs league norms alone). What he did before "
                f"in a look isn't what he does there next.")
    else:
        head = (f"**{name} is about as readable as an average NFL play caller.** His film moves a defense's call "
                f"accuracy {lift * 100:+.1f} points over league norms ({r['Film accuracy']:.0%}).")
    lean_txt = (f"Overall he threw {abs(r['Lean']):.0%} {'more' if r['Lean'] > 0 else 'less'} often than the NFL scout "
                f"expected; that lean is worth {r['Lean part']:+.2f} of his {r['Film edge']:+.2f} bits per 100 snaps, "
                f"his habits by situation and look {r['Look part']:+.2f}.")
    edges = {up.CLUES[k][0]: r[up.CLUES[k][0]] for k in clues}
    pos = [k for k, v in sorted(edges.items(), key=lambda kv: -kv[1]) if v > 0.002]
    neg = [k for k, v in sorted(edges.items(), key=lambda kv: kv[1]) if v < -0.002]
    parts = [head, lean_txt]
    if pos:
        parts.append(f"Beyond his lean, the clues that give him away most: {', '.join(c.lower() for c in pos[:2])}.")
    if neg:
        parts.append(f"His film misleads on {', '.join(c.lower() for c in neg[:2])}: he breaks his own tendencies "
                     "there.")
    return " ".join(parts)


DEFINITIONS = f"""
Two scouts predict run or pass on every snap from what a defense sees **before** the snap:

- **NFL scout:** knows how the league calls it, nothing about this offense. nflfastR's `xpass` (down, distance, field
  position, score, time, timeouts) plus the league's tendency in each look: formation, personnel, motion,
  backfield count, and whether the last snap was a run or a pass. Fit on the other 31 teams.
- **Film scout:** starts from the NFL scout and learns this play caller's own tendencies from **his earlier games
  only** (game 4 is predicted from games 1-3, game 5 from 1-4, ...): his overall lean (pass-heavy or run-heavy in
  any look) and his habits by situation and look. Each look's deviation is pulled hard toward 0
  until the film shows it again and again, so a habit seen a few times barely counts. A play caller who takes over
  mid-season starts a fresh film study.

**Film edge** = how much better the film scout predicts than the NFL scout, in bits per 100 snaps (log loss: a
confident right guess earns a lot, a confident wrong one costs a lot). It's measured on games the film scout hadn't
seen, so a tendency that doesn't repeat costs it: the edge can be negative (the film misleads).

**Unpredictability index** = the film edge as a T-score, flipped: **50** = an average NFL play caller this season,
**60+ = hard to read** (one standard deviation less readable), **40 and below = readable**. Play callers need
{up.MIN_SNAPS} scored snaps for an index.

**Snaps used:** regular season, outside the last 2 minutes of each half, score within {up.MAX_MARGIN}. Personnel
comes from nflverse's participation charting, published after each season, so the season in progress has no
personnel clue. Motion and backfield count come from FTN charting (2022 on).

**From lean / from looks:** a film scout that knows only his lean gets the first part; the rest is his habits by
situation and look. Most of the reliable difference between play callers is lean; habits are smaller and noisier.

**Per clue:** a film scout with his lean and one clue, against one with only his lean. Clues overlap (shotgun, backfield count, and personnel
often tell the same story), so they don't add up to the full edge.

**Per look:** every value of every clue (*Empty*, *12 personnel*, *3rd & short*, *After a run*...) with how often he
threw from it, what the NFL scout expected, what his film said going in, and that look's share of what its clue adds
to his lean.
Positive = the film read him there. Negative = his film said one thing and he did another.
"""

# ---- Balance score view --------------------------------------------------------------------------------------
BALANCE_COLORS = {"Coin Flip": "#2F8C86", "Balanced": "#5A9E6F", "Average": "#8A949A", "Leans": "#D9822B",
                  "Lopsided": "#C8321E"}
DOWN_COLORS = {"1st": "#3B5EA8", "2nd": "#7FA2E0", "3rd": "#D9822B", "4th": "#C8321E"}
BALANCE_DEFINITIONS = f"""
**Balance score** = how close to 50/50 a play caller is in each down & distance, on decision downs only.

1. For each down & distance, take his pass rate there and square its gap from 50%.
2. Take out the part of that gap a small sample creates by chance, p(1 − p)/(n − 1): a cell with 4 snaps that went
   3-1 isn't a tendency yet.
3. Average over the cells, weighted by how many snaps he ran in each.
4. Flip it: **100 × (1 − 4 × that average)**. 100 = a coin flip in every down & distance; 0 = the same call every time.

**Typical call** says the same thing in football terms: the run/pass split of his typical decision down. A score of
87.5 is a 68/32 call (the typical gap from 50/50 is √((1 − score/100) / 4)). **Tiers** by typical call: Coin Flip
(62/38 or closer), Balanced (to 65/35), Average (to 68/32), Leans (to 71/29), Lopsided (beyond). With the formation
split on, every caller looks about 5 points more lopsided, so the tiers move up 5 points.

**Points lost** per down: how much each down & distance takes off his score. They add up to 100 minus his score, so
they show where his balance goes.

**Counted:** every 1st and 2nd down, 3rd & short and medium, 4th & short. **Left out:** 3rd & long and extra long,
4th & medium and longer. Every team throws on 3rd & 7+, and 4th & 4+ is a handful of snaps a season, nearly all
passes. 3rd & medium stays on purpose: most coaches throw there, and the ones who don't are playing it as the first of
two downs, ready to go for it on 4th.

**Split by shotgun / under center** (sidebar): scores each down & distance separately from shotgun and from under
center, which is closer to what a defense sees before the snap.

**Snaps used:** regular season, outside the last 2 minutes of each half, score within {up.MAX_MARGIN} (same as the film
index). **Reliability:** each coach's score in his odd vs even games, correlated across coaches (1 = all real
differences, 0 = all noise).
"""


@st.cache_data(show_spinner="Measuring balance…", ttl=dt.REFRESH_HOURS * 3600)
def balance(season: int, weeks: tuple[int, int] | None, by: str, by_formation: bool = False):
    """(caller table, per-cell table, split-half reliability, the league's 3rd & medium pass rate)."""
    d = up.sample(plays(season), season, weeks)
    if by == "team":
        d["caller"] = d["posteam"]
    cells = up.balance_cells(d, by_formation)
    third = cells[cells["dd"] == "3rd & medium"]
    league_third = float((third["Pass rate"] * third["Snaps"]).sum() / third["Snaps"].sum()) if len(third) else np.nan
    return (up.balance_table(d, by_formation), cells, up.balance_reliability(d, by_formation), league_third)


def balance_anchors_html(g: pd.DataFrame) -> str:
    def one(r, label):
        return (f'<div class="gl-anchor">{badge_html(r, 52)}<div style="min-width:0">'
                f'<div class="gl-lbl" style="color:{BALANCE_COLORS[r["Balance tier"]]}">{label}</div>'
                f'<div class="gl-nm">{r["Play caller"]}</div><div class="gl-sub">{r["Team name"]} · typical down is a '
                f'{up.split_text(r["Typical split"])} call</div></div></div>')
    s = g.sort_values("Balance score", ascending=False)
    return GLANCE_CSS + f'<div class="gl-row">{one(s.iloc[0], "Most balanced")}{one(s.iloc[-1], "Most lopsided")}</div>'


def balance_board_html(g: pd.DataFrame) -> str:
    cols = []
    for _, name in up.BALANCE_TIERS:
        color = BALANCE_COLORS[name]
        rows = g[g["Balance tier"] == name].sort_values("Balance score", ascending=False)
        chips = "".join(f'<div class="gl-chip">{badge_html(r)}<span>{r["Play caller"]}</span>'
                        f'<span class="q">{up.split_text(r["Typical split"])}</span></div>'
                        for _, r in rows.iterrows()) or '<div class="gl-def">Nobody this season</div>'
        cols.append(f'<div class="gl-col" style="border-top-color:{color}"><h4 style="color:{color}">{name}</h4>'
                    f'<div class="gl-def">{up.BALANCE_TIER_HELP[name]}</div>{chips}</div>')
    return GLANCE_CSS + f'<div class="gl-board">{"".join(cols)}</div>'


def points_lost_chart(cells: pd.DataFrame, g: pd.DataFrame) -> alt.Chart:
    """One bar per coach: the points each down & distance takes off his score, colored by down."""
    c = cells.merge(g[["Team", "Play caller", "label", "Balance score"]], left_on=["posteam", "caller"],
                    right_on=["Team", "Play caller"])
    c = c[c["Cost"] > 0].assign(Down=lambda x: x["dd"].str[:3])
    order = list(g.sort_values("Balance score", ascending=False)["label"])
    cell_order = [x for x in up.BALANCE_CELLS]
    c["k"] = c["dd"].map({v: i for i, v in enumerate(cell_order)})
    return alt.Chart(c).mark_bar(stroke=cm.theme_bg(), strokeWidth=0.5).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelOverlap=False)),
        x=alt.X("sum(Cost):Q", title="Balance points lost (100 − his score)"),
        color=alt.Color("Down:N", scale=alt.Scale(domain=list(DOWN_COLORS), range=list(DOWN_COLORS.values())),
                        legend=alt.Legend(title="Down", orient="top")),
        order=alt.Order("k:Q"),
        tooltip=["Play caller", alt.Tooltip("cell:N", title="Down & distance"), alt.Tooltip("Pass rate:Q", format=".0%"),
                 "Snaps", alt.Tooltip("Cost:Q", format=".1f", title="Points lost")],
    ).properties(height=19 * len(order) + 30)


def third_medium_chart(tm: pd.DataFrame, league: float) -> alt.LayerChart:
    """3rd & medium pass rate per coach; orange = plays it as a two-down set."""
    lo = min(0.5, float(tm["Pass rate"].min()) - 0.05)
    tm = tm.assign(two=tm["Pass rate"] <= league - up.THIRD_MEDIUM_GAP, lo=lo)   # bars start at the chart's edge
    order = list(tm.sort_values("Pass rate")["label"])
    base = alt.Chart(tm).encode(y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelOverlap=False)),
                                tooltip=["Play caller", alt.Tooltip("Pass rate:Q", format=".0%"), "Snaps"])
    bars = base.mark_bar(cornerRadius=2).encode(
        x=alt.X("Pass rate:Q", title="Pass rate on 3rd & medium (4-6 to go)", axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domain=[lo, 1])), x2="lo:Q",
        color=alt.condition("datum.two", alt.value("#D9822B"), alt.value("#8A949A")))
    text = base.mark_text(align="left", dx=3, fontSize=10, color=cm.theme_ink()).encode(
        x="Pass rate:Q", text=alt.Text("Pass rate:Q", format=".0%"))
    rule = alt.Chart(pd.DataFrame({"v": [league]})).mark_rule(color=cm.theme_ink(), strokeDash=[4, 4]).encode(x="v:Q")
    return alt.layer(bars, rule, text).properties(height=19 * len(order) + 10)


def dd_grid_html(cells: pd.DataFrame, by_formation: bool) -> str:
    """His decision downs in the Tendencies tab's down & distance grid: the run/pass bar, PROE, plays, and the
    points each one costs his balance. Downs that aren't counted are grayed out."""
    dists = [x.split(" (")[0] for x in dt.DISTANCE_ORDER]
    head = '<div></div>' + "".join(f'<div style="font-size:12px;text-align:center;opacity:.65">{x}</div>' for x in dists)
    out = [head]
    for dn in ("1st", "2nd", "3rd", "4th"):
        out.append(f'<div style="font-weight:600;font-size:14px;display:flex;align-items:center">{dn}</div>')
        for dist in dists:
            key = f"{dn} & {dist.lower()}"
            g = cells[cells["dd"] == key]
            if key not in up.BALANCE_CELLS:
                out.append(f'<div style="{cm.CARD};font-size:12px;opacity:.45">Not counted</div>')
                continue
            if g.empty:
                out.append(f'<div style="{cm.CARD};font-size:12px;opacity:.65">No plays</div>')
                continue
            n = int(g["Snaps"].sum())
            pr = float((g["Pass rate"] * g["Snaps"]).sum() / n)
            proe = float((g["PROE"] * g["Snaps"]).sum() / n)
            cost = float(g["Cost"].sum())
            lead = f"Pass {pr:.0%}" if pr >= 0.5 else f"Run {1 - pr:.0%}"
            chip_c = cm.PASS_C if proe > 0 else cm.RUN_C
            chip = (f'<span style="font-size:11px;font-weight:600;padding:1px 7px;border-radius:999px;'
                    f'border:1px solid {chip_c};color:{chip_c}">PROE {proe * 100:+.0f}%</span>'
                    if abs(proe) >= 0.05 and n >= 5 else '<span></span>')
            form = ""
            if by_formation and len(g) > 1:
                form = '<div style="font-size:11px;opacity:.7">' + " · ".join(
                    f'{"Gun" if f == "Shotgun" else "UC"} {p:.0%}' for f, p in zip(g["Formation"], g["Pass rate"])) + "</div>"
            cost_txt = f"−{cost:.1f} pts" if cost >= 0.05 else "0 pts"
            out.append(
                f'<div style="{cm.CARD};font-size:12px;{"opacity:.6" if n < 8 else ""}">'
                f'<div style="display:flex;justify-content:space-between;gap:6px;align-items:baseline">'
                f'<b style="font-size:13px">{lead}</b><span style="opacity:.75">{cost_txt}</span></div>'
                f'{cm.split_bar(1 - pr)}{form}'
                f'<div style="display:flex;justify-content:space-between;align-items:center;gap:6px;margin-top:3px">'
                f'{chip}<span style="opacity:.65">{n} plays</span></div></div>')
    legend = ('<div style="display:flex;gap:14px;font-size:12px;margin:2px 0 8px">' + "".join(
        f'<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:{c};'
        f'margin-right:4px"></span>{n}</span>' for n, c in (("Run", cm.RUN_C), ("Pass", cm.PASS_C))) + "</div>")
    return (f'<div style="max-width:980px">{legend}<div style="display:grid;grid-template-columns:40px '
            f'repeat(4,minmax(0,1fr));gap:6px">{"".join(out)}</div></div>')


def history_html(caller: str, by_formation: bool, seasons: list[int]) -> str:
    """'2024: 66/34 Average → 2025: 61/39 Coin Flip → …' for every listed season he called plays (full seasons)."""
    items = []
    for s in seasons:
        try:
            t, _, _, _ = balance(s, None, "caller", by_formation)
        except Exception:
            continue
        row = t[t["Play caller"] == caller]
        if len(row):
            r = row.sort_values("Snaps").iloc[-1]
            items.append(f'<span><b>{s}</b> {up.split_text(r["Typical split"])} '
                         f'<span style="color:{BALANCE_COLORS[r["Balance tier"]]};font-weight:600">{r["Balance tier"]}</span>'
                         f' <span style="opacity:.6">({r["Team"]})</span></span>')
    if len(items) < 2:
        return ""
    return ('<div style="display:flex;flex-wrap:wrap;gap:10px;align-items:center;font-size:14px">'
            '<span style="opacity:.7">Season by season:</span>' + ' <span style="opacity:.4">→</span> '.join(items) + "</div>")


def callout_cards(both: pd.DataFrame) -> str:
    """Coaches where balance and the film index disagree."""
    a = both[both["Balance tier"].isin(["Coin Flip", "Balanced"]) & both["Tier"].isin(["Open Book", "Readable"])]
    b = both[both["Balance tier"].isin(["Leans", "Lopsided"]) & both["Tier"].isin(["Mixes It Up", "Poker Face"])]
    cards = []
    for _, r in a.sort_values("Unpredictability index").iterrows():
        cards.append((r, "Balanced, but readable",
                      f"His typical down is a {up.split_text(r['Typical split'])} call, yet his film gives away "
                      f"{r['Plays given away']:.1f} plays a game: his lean and his looks tell you what down & distance "
                      "doesn't."))
    for _, r in b.sort_values("Unpredictability index", ascending=False).iterrows():
        cards.append((r, "Lopsided, but hard to read",
                      f"His typical down is a {up.split_text(r['Typical split'])} call, yet his film doesn't help a "
                      "defense: his habits by down change from game to game, so last month's tendencies don't hold."))
    if not cards:
        return ""
    html = "".join(f'<div class="gl-tell"><div class="gl-head" style="gap:10px">{badge_html(r, 34)}<div>'
                   f'<div class="gl-look">{kind}</div><div style="font-weight:700;font-size:17px">{r["Play caller"]}</div>'
                   f'</div></div><div class="gl-cmp" style="font-size:14px;opacity:.85">{text}</div></div>'
                   for r, kind, text in cards)
    return GLANCE_CSS + f'<div class="gl-tells">{html}</div>'


def balance_chart(d: pd.DataFrame) -> alt.LayerChart:
    """Balance score per play caller, team colors, logo at the end of each bar; dashed line = average caller."""
    order = list(d.sort_values("Balance score", ascending=False)["label"])
    lo = float(min(d["Balance score"].min() - 3, 75))
    tip = [alt.Tooltip("Play caller:N"), alt.Tooltip("Team name:N", title="Team"),
           alt.Tooltip("Balance score:Q", format=".1f"), "Typical call:N", "Balance tier:N",
           alt.Tooltip("Raw balance score:Q", format=".1f", title="Without the small-sample correction"),
           "Most lopsided:N", "Most balanced:N", alt.Tooltip("Snaps:Q", format=",")]
    base = alt.Chart(d.assign(base=lo)).encode(
        x=alt.X("label:N", sort=order, title=None, axis=alt.Axis(labelAngle=-50, labelFontSize=10, labelLimit=120,
                                                                labelOverlap=False)), tooltip=tip)
    y = alt.Y("Balance score:Q", title="Balance score (100 = 50/50 in every decision down)",
              scale=alt.Scale(domain=[lo, 100]))
    bars = base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        y=y, y2="base:Q", color=alt.Color("color:N", scale=None), opacity=alt.Opacity("opacity:Q", scale=None))
    logos = base.mark_image(width=LOGO, height=LOGO, baseline="bottom").encode(
        y="Balance score:Q", url="logo:N", opacity=alt.Opacity("opacity:Q", scale=None))
    mean = alt.Chart(pd.DataFrame({"v": [d["Balance score"].mean()]})).mark_rule(
        color="#888780", strokeDash=[4, 4]).encode(y="v:Q")
    return alt.layer(bars, mean, logos).properties(height=380)


def cost_heatmap(cells: pd.DataFrame, d: pd.DataFrame) -> alt.LayerChart:
    """Each coach x decision down, colored by the points it costs his balance (text = pass rate)."""
    c = cells.merge(d[["Team", "Play caller", "label"]], left_on=["posteam", "caller"], right_on=["Team", "Play caller"])
    c = (c.assign(passes=c["Pass rate"] * c["Snaps"]).groupby(["label", "Play caller", "dd"])
         .agg(passes=("passes", "sum"), Snaps=("Snaps", "sum"), Cost=("Cost", "sum")).reset_index())
    c["Pass rate"] = c["passes"] / c["Snaps"]
    order = list(d.sort_values("Balance score", ascending=False)["label"])
    x = alt.X("dd:N", sort=up.BALANCE_CELLS, title=None,
              axis=alt.Axis(orient="top", labelAngle=-35, labelOverlap=False, labelLimit=140))
    y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelOverlap=False))
    tip = ["Play caller", alt.Tooltip("dd:N", title="Down & distance"), alt.Tooltip("Pass rate:Q", format=".0%"),
           "Snaps", alt.Tooltip("Cost:Q", format=".1f", title="Points lost")]
    hi = float(max(c["Cost"].quantile(0.97), 1))
    rect = alt.Chart(c).mark_rect().encode(
        x=x, y=y, tooltip=tip,
        color=alt.Color("Cost:Q", title="Points lost", scale=alt.Scale(scheme="oranges", domain=[0, hi], clamp=True)))
    text = alt.Chart(c).mark_text(fontSize=9).encode(
        x=x, y=y, text=alt.Text("Pass rate:Q", format=".0%"),
        color=alt.condition(f"datum.Cost > {hi * 0.6}", alt.value("white"), alt.value(cm.theme_ink())))
    return alt.layer(rect, text).properties(height=19 * len(order) + 10)


def render_balance(season: int, weeks: tuple[int, int] | None, by: str, min_snaps: int, highlight: list[str],
                   by_formation: bool, listed_seasons: list[int]):
    st.title("How balanced is he?")
    st.markdown("On every decision down (all 1st and 2nd downs, 3rd & short and medium, 4th & short), how close is "
                "he to a 50/50 run/pass call? We sum it up as his **typical call**: 60/40 is hard to call, 75/25 is "
                "easy." + (" **Split by shotgun / under center is on:** each down is scored separately from each "
                           "formation." if by_formation else ""))
    with st.expander("How this is measured"):
        st.markdown(BALANCE_DEFINITIONS)
    try:
        t, cells, rel, league_third = balance(season, weeks, by, by_formation)
    except Exception as e:
        st.error(f"Couldn't load nflfastR data: {e}")
        st.stop()
    if t.empty:
        st.info("No snaps match these filters.")
        st.stop()
    d = with_team(t.assign(Edge=np.nan, **{"Lean edge": np.nan, "Look edge": np.nan, "Film accuracy": np.nan,
                                           "Generic accuracy": np.nan}), info, highlight, by)
    d = d[d["Snaps"] >= min_snaps]
    if d.empty:
        st.info("No play callers with that many snaps. Lower the minimum.")
        st.stop()
    d["Typical call"] = d["Typical split"].map(up.split_text)
    if pd.isna(rel) or rel < 0.3:
        st.warning("It's early: there aren't enough snaps yet to tell most coaches apart (their scores in odd and even "
                   "games barely agree). Treat these tiers as a first look.")

    st.html(balance_anchors_html(d))
    st.subheader("Every play caller, by tier")
    st.html(balance_board_html(d))
    st.caption("The number next to each name is his typical call: the run/pass split of his typical decision down.")

    st.subheader("Where each coach loses balance")
    st.altair_chart(points_lost_chart(cells, d), width="stretch")
    st.caption("The longer the bar, the easier he is to call by down & distance. Each piece is one down & distance; "
               "hover for his pass rate there. Most balanced at the top.")

    st.subheader("3rd & medium: one down or two?")
    tm = up.third_medium(cells).merge(d[["Team", "Play caller", "label"]], left_on=["posteam", "caller"],
                                      right_on=["Team", "Play caller"])
    tm = tm[tm["Snaps"] >= 8]
    if len(tm):
        st.altair_chart(third_medium_chart(tm, league_third), width="stretch")
        st.caption(f"Dashed line = the league ({league_third:.0%} pass on 3rd & 4-6). Orange = {up.THIRD_MEDIUM_GAP:.0%} "
                   "or more below it: he runs often enough on 3rd & medium that he's playing it as the first of two "
                   "downs, ready to go for it on 4th. Coaches with 8+ snaps there.")

    st.subheader("Balanced isn't the same as hard to read")
    try:
        f = with_team(up.caller_table(*scored(season, weeks, by)), info, highlight, by)
        f["Tier"] = f["Unpredictability index"].map(lambda v: up.tier(v) if pd.notna(v) else None)
        both = f.merge(d[["Team", "Play caller", "Balance score", "Typical split", "Balance tier"]],
                       on=["Team", "Play caller"])
        both = both[both["Unpredictability index"].notna()]
    except Exception:
        both = pd.DataFrame()
    if len(both) >= 5:
        cards = callout_cards(both)
        if cards:
            st.html(cards)
        r = both["Balance score"].corr(both["Unpredictability index"])
        st.altair_chart(logo_scatter(both, "Balance score", "Unpredictability index", "More balanced by down & distance →",
                                     "Harder to read on film (unpredictability index) →", ".0f", ".0f",
                                     ref=(float(both["Balance score"].mean()), 50.0)), width="stretch")
        st.caption(f"r = {r:+.2f}: the two barely agree. Bottom right = balanced by down, yet readable (his lean, "
                   "formation, or personnel gives it away). Top left = lopsided by down, yet hard to read (his habits "
                   "don't hold from one game to the next).")
    else:
        st.caption("Not enough film yet to compare with the film index.")

    st.subheader("Scouting card")
    order = d.sort_values("Balance score", ascending=False)
    labels = {f"{r['Play caller']} ({r['Team']}) · {r['Typical call']} · {r['Balance tier']}": i
              for i, r in order.iterrows()}
    pick = st.selectbox("Play caller", list(labels), key=f"up_bal_pick_{season}",
                        help="Sorted from most balanced to most lopsided.")
    r = d.loc[labels[pick]]
    mine = cells[(cells["posteam"] == r["Team"]) & (cells["caller"] == r["Play caller"])]
    color = BALANCE_COLORS[r["Balance tier"]]
    st.html(GLANCE_CSS + f'<div class="gl-card"><div class="gl-head">{badge_html(r, 56)}<div style="min-width:0">'
            f'<div class="gl-nm" style="font-size:30px">{r["Play caller"]}</div><div class="gl-sub">{r["Team name"]} · '
            f'{season} · {int(r["Snaps"])} decision-down snaps</div></div>'
            f'<span class="gl-pill" style="background:{color}">{r["Balance tier"]}</span></div>'
            f'<p class="gl-verdict">{up.balance_verdict(r, mine, league_third)}</p>'
            f'<div class="gl-stat"><span class="gl-n" style="color:{color}">{r["Typical call"]}</span>'
            f'<span>his typical decision down. The average coach: '
            f'{up.split_text(up.typical_split(d["Balance score"].mean()))}.</span></div>'
            f'{history_html(r["Play caller"], by_formation, listed_seasons) if by == "caller" else ""}</div>')
    st.markdown(f"**{r['Play caller']}'s decision downs**")
    st.html(dd_grid_html(mine, by_formation))
    st.caption("Same grid as the Tendencies tab: the bar is his run/pass split, the chip is PROE (how much more or less "
               "he throws than an average team in the same spots, shown at ±5% or more), and the top right is how many "
               "points that down takes off his balance. Faded = under 8 plays." +
               (" Gun / UC = his pass rate from shotgun and from under center." if by_formation else ""))

    with st.expander("Show the math"):
        st.markdown(f"**Balance score, {season}**")
        st.altair_chart(balance_chart(d), width="stretch")
        st.markdown("**Points lost on every decision down** (text = pass rate)")
        st.altair_chart(cost_heatmap(cells, d), width="stretch")
        show = pd.DataFrame({"Logo": d["logo"], "Play caller": d["Play caller"], "Team": d["Team"],
                             "Typical call": d["Typical call"], "Tier": d["Balance tier"],
                             "Balance score": d["Balance score"], "Raw balance score": d["Raw balance score"],
                             "Pass rate": d["Pass rate"], "Most lopsided": d["Most lopsided"],
                             "Most balanced": d["Most balanced"], "Snaps": d["Snaps"]})
        st.dataframe(show.style.format({"Balance score": "{:.1f}", "Raw balance score": "{:.1f}",
                                        "Pass rate": "{:.0%}", "Snaps": "{:,}"})
                     .apply(shade_index_balance, subset=["Balance score"]),
                     width="stretch", hide_index=True, height=36 * (len(show) + 1) + 2,
                     column_config={"Logo": st.column_config.ImageColumn("", width="small"),
                                    "Raw balance score": st.column_config.Column(
                                        help="The same score without taking out small-sample chance."),
                                    "Most lopsided": st.column_config.Column(
                                        help="The down & distance (8+ snaps) that costs him the most points."),
                                    "Pass rate": st.column_config.Column(help="Pass rate on decision downs."),
                                    "Snaps": st.column_config.Column(help="Decision-down snaps.")})
        st.caption("Reliability (odd vs even games): " + ("–" if pd.isna(rel) else f"{rel:.2f}") +
                   ". 1 = all real differences between coaches, 0 = all noise.")


def shade_index_balance(col):
    """Orange below the average caller, purple above, stronger the farther from it."""
    mu, sd = col.mean(), col.std() or 1
    out = []
    for v in col:
        z = (v - mu) / sd
        a = min(abs(z) / 2, 1)
        rgb = "83,74,183" if z > 0 else "216,90,48"
        out.append(f"background-color:rgba({rgb},{a * 0.45:.2f})" if a >= 0.15 else "")
    return out


# ---- At a glance view -----------------------------------------------------------------------------------------
TIER_COLORS = {"Open Book": "#C8321E", "Readable": "#D9822B", "Average": "#8A949A", "Mixes It Up": "#2F8C86",
               "Poker Face": "#3B5EA8"}
GLANCE_CSS = """<style>
.gl-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.gl-anchor{border:1px solid rgba(128,128,128,.3);border-radius:10px;padding:14px 16px;display:flex;gap:14px;align-items:center}
.gl-lbl{font-size:12px;font-weight:700;letter-spacing:.1em;text-transform:uppercase}
.gl-nm{font-size:24px;font-weight:700;line-height:1.15}
.gl-sub{opacity:.7;font-size:14px}
.gl-board{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}
@media (max-width:900px){.gl-board{grid-template-columns:1fr 1fr}}
@media (max-width:500px){.gl-board{grid-template-columns:1fr}}
.gl-col{background:rgba(128,128,128,.07);border-radius:10px;padding:12px;border-top:6px solid;min-width:0}
.gl-col h4{margin:0 0 2px;font-size:18px;text-transform:uppercase;letter-spacing:.03em;padding:0}
.gl-def{font-size:13px;opacity:.7;line-height:1.35;margin-bottom:8px}
.gl-chip{display:flex;align-items:center;gap:8px;padding:3px 0;font-size:14px}
.gl-chip .q{margin-left:auto;opacity:.6;font-family:monospace}
.gl-badge{flex:none;display:inline-grid;place-items:center;border-radius:50%;font-weight:700;border:2px solid}
.gl-card{border:1px solid rgba(128,128,128,.3);border-radius:12px;padding:20px;display:flex;flex-direction:column;gap:18px}
.gl-head{display:flex;gap:14px;align-items:center;flex-wrap:wrap}
.gl-pill{font-size:13px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;padding:3px 12px;border-radius:999px;color:#fff}
.gl-sure{font-size:12px;opacity:.75;border:1px solid rgba(128,128,128,.4);border-radius:999px;padding:2px 10px}
.gl-verdict{font-size:19px;line-height:1.5;max-width:62ch;margin:0}
.gl-stat{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}
.gl-n{font-size:56px;font-weight:800;line-height:1;font-variant-numeric:tabular-nums}
.gl-tells{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
.gl-tell{border:1px solid rgba(128,128,128,.3);border-radius:10px;padding:12px 14px;display:flex;flex-direction:column;gap:6px}
.gl-look{font-size:13px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;opacity:.7}
.gl-big{font-size:30px;font-weight:800;line-height:1}
.gl-big small{font-size:16px;margin-left:4px}
.gl-bar{position:relative;height:10px;background:rgba(128,128,128,.18);border-radius:5px}
.gl-bar i{position:absolute;top:0;bottom:0;left:0;border-radius:5px}
.gl-bar u{position:absolute;top:-4px;bottom:-4px;width:3px;background:currentColor;border-radius:2px}
.gl-cmp{font-size:13px;opacity:.75}
.gl-h{font-size:18px;font-weight:700;margin:0}
</style>"""


def badge_html(r: pd.Series, size: int = 26) -> str:
    """The team logo, or a badge in team colors when the logo couldn't be downloaded."""
    if r.get("logo"):
        return f'<img src="{r["logo"]}" width="{size}" height="{size}" alt="{r["Team"]}" style="flex:none">'
    fg = "#16212B" if _luminance(r["color"]) > 0.5 else "#FFFFFF"
    return (f'<span class="gl-badge" style="width:{size}px;height:{size}px;font-size:{max(9, size // 3)}px;'
            f'background:{r["color"]};color:{fg};border-color:{r["color"]}">{r["Team"]}</span>')


def anchors_html(g: pd.DataFrame) -> str:
    def one(r, label):
        why = (f"film gives away ~{r['Plays given away']:.1f} plays a game" if r["Plays given away"] > 0
               else "his film misleads a defense")
        return (f'<div class="gl-anchor">{badge_html(r, 52)}<div style="min-width:0">'
                f'<div class="gl-lbl" style="color:{TIER_COLORS[r["Tier"]]}">{label}</div>'
                f'<div class="gl-nm">{r["Play caller"]}</div><div class="gl-sub">{r["Team name"]} · {why}</div></div></div>')
    s = g.sort_values("Unpredictability index")
    return GLANCE_CSS + f'<div class="gl-row">{one(s.iloc[0], "Easiest to read")}{one(s.iloc[-1], "Hardest to read")}</div>'


def board_html(g: pd.DataFrame) -> str:
    cols = []
    for _, name in up.TIERS:
        color = TIER_COLORS[name]
        rows = g[g["Tier"] == name].sort_values("Unpredictability index")
        chips = "".join(f'<div class="gl-chip">{badge_html(r)}<span>{r["Play caller"]}</span>'
                        f'{"<span class=q title=&quot;Too close to call&quot;>~</span>" if r["Sure"] == "Too close to call" else ""}'
                        f'</div>' for _, r in rows.iterrows()) or '<div class="gl-def">Nobody this season</div>'
        cols.append(f'<div class="gl-col" style="border-top-color:{color}"><h4 style="color:{color}">{name}</h4>'
                    f'<div class="gl-def">{up.TIER_HELP[name]}</div>{chips}</div>')
    return GLANCE_CSS + f'<div class="gl-board">{"".join(cols)}</div>'


def four_box(g: pd.DataFrame) -> alt.LayerChart:
    """Lean part (across) vs look part (up), with the four corners named."""
    x, y = "Lean part", "Look part"
    pad_x, pad_y = (g[x].max() - g[x].min()) * 0.08 or 0.01, (g[y].max() - g[y].min()) * 0.08 or 0.01
    x0, x1, y0, y1 = g[x].min() - pad_x, g[x].max() + pad_x, g[y].min() - pad_y, g[y].max() + pad_y
    chart = logo_scatter(g, x, y, "His overall pass/run lean gives it away →", "His looks give it away →", ".1f", ".1f",
                         ref=(float(g[x].mean()), float(g[y].mean())))
    corners = pd.DataFrame({"x": [x1, x0, x1, x0], "y": [y1, y1, y0, y0], "align": ["right", "left", "right", "left"],
                            "base": ["top", "top", "bottom", "bottom"],
                            "t": ["TIPS BOTH", "TELLS IN HIS LOOKS", "KNOWN LEAN", "NOTHING TO GO ON"]})
    labels = [alt.Chart(corners.iloc[[i]]).mark_text(align=r.align, baseline=r.base, dx=-8 if r.align == "right" else 8,
                                                     dy=8 if r.base == "top" else -8, fontSize=13, fontWeight="bold",
                                                     opacity=0.45, color=cm.theme_ink())
              .encode(x=alt.X("x:Q", scale=alt.Scale(zero=False)), y=alt.Y("y:Q", scale=alt.Scale(zero=False)), text="t:N")
              for i, r in enumerate(corners.itertuples())]
    return alt.layer(chart, *labels)


def card_html(r: pd.Series, tl: list[dict], ml: list[dict], avg_given: float, season: int) -> str:
    color = TIER_COLORS[r["Tier"]]
    g = r["Plays given away"]
    stat = ("plays a game his film gives away to a defense that studied it." if g > 0 else
            "plays a game: studying his film makes a defense's guesses slightly worse.")
    head = (f'<div class="gl-head">{badge_html(r, 56)}<div style="min-width:0"><div class="gl-nm" style="font-size:30px">'
            f'{r["Play caller"]}</div><div class="gl-sub">{r["Team name"]} · {season} · weeks {r["Weeks"]}</div></div>'
            f'<span class="gl-pill" style="background:{color}">{r["Tier"]}</span>'
            f'<span class="gl-sure">{"Clear result" if r["Sure"] == "Clear" else r["Sure"]}</span></div>')
    body = (f'<p class="gl-verdict">{up.verdict(r, tl, ml)}</p>'
            f'<div class="gl-stat"><span class="gl-n" style="color:{color}">{g:.1f}</span>'
            f'<span>{stat} The average coach: {avg_given:.1f}.</span></div>')
    if tl:
        cards = "".join(
            f'<div class="gl-tell"><span class="gl-look">{t["look"]}</span>'
            f'<span class="gl-big" style="color:{"#3B5EA8" if t["call"] == "PASS" else "#D9822B"}">{t["rate"]:.0%}'
            f'<small>{t["call"]}</small></span><div class="gl-bar"><i style="width:{t["rate"] * 100:.0f}%;'
            f'background:{"#3B5EA8" if t["call"] == "PASS" else "#D9822B"}"></i>'
            f'<u style="left:calc({t["league"] * 100:.0f}% - 1px)"></u></div>'
            f'<span class="gl-cmp">Average team: {t["league"]:.0%} · {t["snaps"]} plays</span></div>' for t in tl)
        body += (f'<div><p class="gl-h">What gives him away</p></div><div class="gl-tells">{cards}</div>'
                 '<div class="gl-cmp">Bar = how often he made that call; mark = an average team in the same spots.</div>')
    else:
        body += '<div><p class="gl-h">What gives him away</p><div class="gl-cmp">No single look gives him away by 7 points or more.</div></div>'
    if ml:
        cards = "".join(
            f'<div class="gl-tell"><span class="gl-look">{m["look"]}</span><span class="gl-cmp">Earlier games said</span>'
            f'<span class="gl-big" style="font-size:24px">pass {m["film"]:.0%}</span>'
            f'<span class="gl-cmp">The next games: <b>pass {m["actual"]:.0%}</b> · {m["snaps"]} plays</span></div>'
            for m in ml)
        body += f'<div><p class="gl-h">Where his film fools you</p></div><div class="gl-tells">{cards}</div>'
    return GLANCE_CSS + f'<div class="gl-card">{head}{body}</div>'


def render_glance(ranked: pd.DataFrame, d_snaps: pd.DataFrame, clues: list[str], season: int, rel: float) -> None:
    g = ranked.copy()
    g["Tier"] = g["Unpredictability index"].map(up.tier)
    g["Sure"] = up.certainty(g)
    if pd.isna(rel) or rel < 0.3:
        st.warning("It's early: there isn't enough film yet to tell most coaches apart. Treat these tiers as a first "
                   "look; they settle down over a full season.")
    st.html(anchors_html(g))

    st.subheader("Every play caller, by tier")
    st.html(board_html(g))
    st.caption("~ = too close to call: his result is within the normal ups and downs, so he could sit a tier either "
               "side. Pick any coach below for his scouting card.")

    st.subheader("What gives him away?")
    st.markdown("Two ways a coach tips his hand. **Right:** he leans pass (or run) all the time, so a defense can just "
                "play the odds. **Up:** his formations, personnel, motion, or the down tell you what's coming.")
    st.altair_chart(four_box(g), width="stretch")
    st.caption("Dashed lines = the average play caller. Hover a logo for his tier.")

    st.subheader("Scouting card")
    order = g.sort_values("Unpredictability index")
    labels = {f"{r['Play caller']} ({r['Team']}) · {r['Tier']}": i for i, r in order.iterrows()}
    pick = st.selectbox("Play caller", list(labels), key=f"up_glance_{season}",
                        help="Sorted from easiest to hardest to read.")
    r = g.loc[labels[pick]]
    looks = up.look_table(d_snaps, r["Team"], r["Play caller"], clues)
    st.html(card_html(r, up.tells(looks), up.misleads(looks), float(g["Plays given away"].mean()), season))
    with st.expander("Show the math"):
        math = pd.DataFrame({"": [
            "Unpredictability index (50 = average)", "Film edge, bits per 100 snaps", "…from his overall lean",
            "…from his looks", "Lean (pass rate vs expected)", "Right guesses: league scout → film scout",
            "Scored snaps", "Offensive EPA per play"], "Value": [
            f"{r['Unpredictability index']:.0f}", f"{r['Film edge']:+.2f} ± {2 * r['Edge SE'] * 100:.2f}",
            f"{r['Lean part']:+.2f}", f"{r['Look part']:+.2f}", f"{r['Lean']:+.1%}",
            f"{r['Generic accuracy']:.0%} → {r['Film accuracy']:.0%}", f"{int(r['Scored snaps']):,}",
            f"{r['EPA/play']:+.3f}"]})
        st.dataframe(math, hide_index=True, width="content")
        st.caption("Switch View to **Show the math** in the sidebar for every coach's numbers and the charts behind them.")
    st.caption("**How it's measured.** Two computer scouts guess every snap from what a defense sees before it: down "
               "and distance, field position, score, formation, personnel, motion, backs in the backfield, and the "
               "last call. One knows only how the whole NFL plays; the other also studies this coach's earlier games "
               "(never later ones). Plays a game his film gives away = how much of the guessing studying him removes, "
               "counted in plays. Tiers by index: 40 or less = Open Book, 40-46 Readable, 46-54 Average, 54-60 Mixes It "
               "Up, over 60 Poker Face.")


# ---- Page ----------------------------------------------------------------------------------------------------
seasons = dt.available_seasons()
info = team_info()
callers_csv = up.load_callers()
with st.sidebar:
    st.header("Unpredictability Index")
    view = st.segmented_control("View", ["At a glance", "Show the math", "Balance score"], default="At a glance",
                                key="up_view",
                                help="At a glance: tiers, what gives each coach away, and a scouting card, in plain "
                                     "words. Show the math: every number behind them. Balance score: how close to "
                                     "50/50 he calls each down & distance.") or "At a glance"
    st.session_state.setdefault("up_season", seasons[-1])
    season = st.selectbox("Season", seasons[::-1], key="up_season")
    season_plays = plays(season)
    reg_weeks = season_plays.loc[season_plays["season_type"] == "REG", "Week"]
    if reg_weeks.empty:
        st.info("No regular-season plays for this season yet.")
        st.stop()
    max_week = int(reg_weeks.max())
    weeks = st.slider("Weeks", 1, max_week, (1, max_week), key=f"up_weeks_{season}",
                      help=f"Film from these weeks only. A caller's first {up.MIN_GAMES} games in the range are film "
                           "study, not scored.")
    by_label = st.segmented_control("Compare", ["Play callers", "Teams"], default="Play callers", key="up_by",
                                    help="Play callers: a mid-season change of play caller starts a new film study. "
                                         "Teams: one film study per team for the whole range.")
    by = "team" if by_label == "Teams" else "caller"
    min_snaps = st.slider("Minimum scored snaps to show" if view != "Balance score" else "Minimum snaps to show",
                          0, 600, up.MIN_SNAPS, 25, key="up_min",
                          help=f"Index needs {up.MIN_SNAPS}+; lower shows short stints without an index.")
    highlight = st.multiselect("Highlight teams", sorted(season_plays["posteam"].dropna().unique()), key="up_hl",
                               help="Fades every other team in the charts.")
    by_formation = view == "Balance score" and st.toggle(
        "Split by shotgun / under center", False, key="up_bal_form",
        help="Score each down & distance separately from shotgun and from under center, closer to what a defense "
             "sees before the snap.")

week_range = None if weeks == (1, max_week) else weeks
if view == "Balance score":
    render_balance(season, week_range, by, min_snaps, highlight, by_formation,
                   sorted(s for s in callers_csv["season"].unique() if s in seasons))
    st.stop()

if view == "At a glance":
    st.title("Can you read his calls?")
    st.markdown("We had a computer study every play caller's earlier games, then guess run or pass on his next ones. "
                "If studying his film helps the guess, he's readable. If it doesn't, or it leads you the wrong way, "
                "he's hard to read.")
else:
    st.title("Unpredictability Index")
    st.caption("How much a defense learns by studying a play caller's film: which coaches give their calls away, "
               "which don't, and the looks that tell on them.")
    with st.expander("How this is measured"):
        st.markdown(DEFINITIONS)
try:
    d_snaps, clues = scored(season, week_range, by)
except Exception as e:
    st.error(f"Couldn't load nflfastR data: {e}")
    st.stop()
t = up.caller_table(d_snaps, clues)
if t.empty or t["Unpredictability index"].isna().all():
    st.info(f"Not enough film yet: a play caller needs {up.MIN_GAMES} games of film plus {up.MIN_SNAPS} scored snaps. "
            "Widen the weeks.")
    st.stop()
if by == "caller" and not (callers_csv["season"] == season).any():
    st.info(f"The play-caller list doesn't cover {season} yet, so each team is one play caller here. Add rows to "
            "play_callers.csv to split teams by play caller.")
if "personnel" not in clues and view != "At a glance":
    st.info("No personnel charting for this season yet (nflverse publishes it after the season), so personnel isn't a "
            "clue. Motion and backfield count are." if {"motion", "backfield"} & set(clues) else
            "No personnel charting for this season, so personnel isn't a clue.")

d_all = with_team(t, info, highlight, by)
d = d_all[d_all["Scored snaps"] >= min_snaps]
ranked = d[d["Unpredictability index"].notna()]
rel = up.reliability(d_snaps)
if view == "At a glance":
    render_glance(ranked, d_snaps, clues, season, rel)
    st.stop()

counts = ranked["Profile"].value_counts()
c = st.columns(5)
c[0].metric("Hard to read", int(counts.get("Hard to read", 0)), help="Index 60 or higher")
c[1].metric("NFL average", int(counts.get("NFL average", 0)))
c[2].metric("Readable", int(counts.get("Readable", 0)), help="Index 40 or lower")
c[3].metric("Film edge, average caller", f"{ranked['Film edge'].mean():+.2f}",
            help="Bits per 100 snaps a defense gains by studying an average play caller's film.")
c[4].metric("Reliability", "–" if pd.isna(rel) else f"{rel:.2f}",
            help="Split-half: correlation of each caller's film edge in odd vs even games, stepped up to the full "
                 "sample. 1 = all real differences between callers, 0 = all noise.")

if pd.isna(rel) or rel < 0.3:
    st.warning("Early in a season (or over a short week range) most of the gap between play callers is noise: the "
               "edges in odd and even games barely agree (reliability "
               f"{'–' if pd.isna(rel) else f'{rel:.2f}'}). Read the index as a first look, not a verdict.")
st.markdown(f"**Unpredictability index, {season}**")
st.altair_chart(index_chart(ranked), width="stretch")
st.caption("Bars start at 50, an average NFL play caller. Up = his film tells a defense less than the average "
           "caller's; down = more. Dashed lines at 40 and 60 mark readable and hard to read. Hover for the numbers.")

st.markdown("**Every play caller, side by side**")
table, fmt = comparison_table(d, clues)
clue_cols = [up.CLUES[k][0] for k in clues]
sty = (table.style.format(fmt, na_rep="").apply(color_profile, subset=["Profile"])
       .apply(shade_edge, subset=["Film edge"] + clue_cols))
st.dataframe(sty, width="stretch", hide_index=True, height=36 * (len(table) + 1) + 2,
             column_config={"Logo": st.column_config.ImageColumn("", width="small"),
                            **{k: st.column_config.Column(help=v) for k, v in HELP.items()},
                            **{k: st.column_config.Column(help=f"Film edge from {k.lower()} alone (bits / 100 snaps).")
                               for k in clue_cols}})
st.caption("Orange = his film gives him away (positive edge); purple = his film misleads (negative). Film edge = from "
           "lean + from looks. Clue columns: what that clue adds to his lean, in bits per 100 snaps (clues overlap, "
           "so they don't add up to From looks). Sort by clicking a header.")

st.markdown("**Which clues give each play caller away**")
st.altair_chart(clue_heatmap(ranked, clues), width="stretch")
st.caption("What each clue adds to his lean, bits per 100 snaps (a film scout that knows his lean and that one clue, vs "
           "one that knows only his lean). Red = studying that clue on his film helps a defense; blue = his film on that "
           "clue misleads. Most readable callers at the top.")

st.markdown("**Readable from his lean, or from his habits?**")
st.altair_chart(logo_scatter(ranked, "Lean part", "Look part", "From his overall lean (bits / 100 snaps) →",
                             "From his situations & looks (bits / 100 snaps) →", "+.1f", "+.1f", ref=(0.0, 0.0)),
                width="stretch")
st.caption("Right = his overall pass/run lean gives him away (he's clearly pass- or run-heavy and stays that way). Up = "
           "beyond that, his calls by down & distance, formation, personnel, and motion repeat. Left or down of the "
           "dashed lines = his film on that misleads: the lean or the habits changed from game to game. Lean is most of "
           "the gap between callers; habits are smaller and noisier.")

st.markdown("**Offensive EPA per play vs unpredictability**")
r = ranked["Unpredictability index"].corr(ranked["EPA/play"])
st.altair_chart(logo_scatter(ranked, "Unpredictability index", "EPA/play", "Unpredictability index →",
                             "Offensive EPA per play →", ".0f", "+.2f", trend=True, ref=(50.0, None)), width="stretch")
st.caption(f"Does being hard to read pay off? Correlation across these play callers: r = {r:+.2f} (dashed line = "
           "least-squares fit). EPA per play is over the same snaps the index uses.")

# ---- Why ------------------------------------------------------------------------------------------------------
st.subheader("Why a play caller is readable or hard to read")
options = list(d.sort_values("Unpredictability index", na_position="last")["label"])
pick = st.selectbox("Play caller", options, key=f"up_pick_{season}_{by}",
                    help="Sorted from most readable to hardest to read.")
r = d[d["label"] == pick].iloc[0]
lt = up.look_table(d_snaps, r["Team"], r["Play caller"], clues)
lt = lt[lt["Snaps"] >= 5]
c = st.columns(4)
c[0].metric("Unpredictability index", "–" if pd.isna(r["Unpredictability index"]) else
            f"{r['Unpredictability index']:.0f}", r["Profile"], delta_color="off")
c[1].metric("Film edge (bits / 100 snaps)", f"{r['Film edge']:+.2f}", f"± {2 * r['Edge SE'] * 100:.2f}",
            delta_color="off")
c[2].metric("Film-scout accuracy", f"{r['Film accuracy']:.1%}", f"{r['Accuracy gain'] * 100:+.1f} pts vs NFL scout")
c[3].metric("Scored snaps", f"{int(r['Scored snaps']):,}", f"weeks {r['Weeks']}", delta_color="off")
st.markdown(why_summary(r, lt, clues))
if lt.empty:
    st.info("No looks with 5+ scored snaps.")
else:
    tells = lt[lt["Edge share"] > LOOK_MIN_EDGE].nlargest(8, "Edge share")
    misled = lt[lt["Edge share"] < -LOOK_MIN_EDGE].nsmallest(8, "Edge share")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Why he's readable: tells his film caught**")
        if tells.empty:
            st.caption("No look where his film helped.")
        else:
            st.markdown("\n".join(f"- {up.tell_sentence(x)}" for _, x in tells.head(3).iterrows()))
            st.dataframe(look_display(tells).style.format(LOOK_FMT), width="stretch", hide_index=True)
    with c2:
        st.markdown("**Why he's hard to read: where his film misled**")
        if misled.empty:
            st.caption("No look where his film misled.")
        else:
            st.markdown("\n".join(f"- {x['Look']}: film said {x['Their film said']:.0%} pass, he threw "
                                  f"{x['Pass rate']:.0%} ({int(x['Snaps'])} snaps)" for _, x in misled.head(3).iterrows()))
            st.dataframe(look_display(misled).style.format(LOOK_FMT), width="stretch", hide_index=True)
    st.caption("Edge = that look's share of its clue's film edge, in bits per 100 of all his scored snaps; each look "
               "is scored with its own clue alone, so a clue's looks add up to that clue's column in the table above. "
               "NFL scout expected = the league's pass rate in the same situations and looks. Their film said = the "
               "film scout's guess (from that clue) going into each game, averaged.")
    with st.expander("Every look"):
        st.dataframe(look_display(lt.sort_values(["Clue", "Snaps"], ascending=[True, False])).style.format(LOOK_FMT)
                     .apply(shade_edge, subset=["Edge (bits / 100 snaps)"]), width="stretch", hide_index=True)

# ---- Across seasons ---------------------------------------------------------------------------------------------
listed = sorted(s for s in callers_csv["season"].unique() if s in seasons)
if by == "caller" and len(listed) > 1:
    st.subheader("Play callers across seasons")
    frames = []
    for s in listed:
        ds, cl = scored(s, None, "caller")
        ts = up.caller_table(ds, cl).assign(Season=s)
        frames.append(ts[ts["Unpredictability index"].notna()])
    h = pd.concat(frames, ignore_index=True)
    h = with_team(h, info, highlight, "caller")
    repeat = h["Play caller"].map(h["Play caller"].value_counts()) > 1
    st.altair_chart(history_chart(h[repeat]), width="stretch")
    st.caption(f"Full regular seasons {listed[0]}-{listed[-1]} (the season in progress so far), play callers with "
               "more than one season. Dot color = the team he called plays for that season. Each season is scored "
               "against that season's NFL, so 50 is always that year's average.")
    pivot = h.pivot_table(index="Play caller", columns="Season", values="Unpredictability index", aggfunc="mean")
    teams = h.groupby("Play caller")["Team"].agg(lambda x: " → ".join(dict.fromkeys(x)))
    pivot.insert(0, "Teams", teams)
    pivot["Seasons"] = pivot[listed].notna().sum(axis=1)
    pivot["Average"] = pivot[listed].mean(axis=1)
    pivot = pivot.sort_values("Average")
    pivot.columns = [str(c) for c in pivot.columns]
    yrs = [str(s) for s in listed]
    pivot[yrs + ["Average"]] = pivot[yrs + ["Average"]].astype(float)
    # Streamlit draws a missing number as "None" whatever the Styler says, so the numbers go in as text.
    nums = pivot[yrs + ["Average"]]
    shown = pivot.copy()
    shown[yrs + ["Average"]] = nums.map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
    st.dataframe(shown.style.apply(lambda col: shade_index(nums[col.name]), subset=yrs + ["Average"]),
                 width="stretch", height=min(36 * (len(pivot) + 1) + 2, 720))
    st.caption("Lower = more readable (orange), higher = harder to read (purple). Sorted most readable first.")

with st.expander(f"Play-caller list, {season}"):
    lst = callers_csv[callers_csv["season"] == season]
    if lst.empty:
        st.caption("No play-caller list for this season yet.")
    else:
        st.dataframe(lst.drop(columns="season").rename(columns=str.capitalize).rename(
            columns={"Play_caller": "Play caller", "First_week": "First week", "Last_week": "Last week"}),
            width="stretch", hide_index=True)
    st.caption("Who called the offensive plays, by week (play_callers.csv). Week 22 = through the playoffs. "
               "Medium confidence = reported but not confirmed by the team, or a shared or changing role.")
