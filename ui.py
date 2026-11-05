"""
Shared pieces for the play-caller pages (Unpredictability Index, Balance Score): cached data, team colors and logos,
the logo scatter, badges and card styles, and the EPA/play trend section.
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import common as cm
import data as dt
import unpredictability as up

LOGO = 26          # logo size on bar charts, px


@st.cache_data(show_spinner="Loading nflfastR play-by-play…", ttl=dt.REFRESH_HOURS * 3600)
def plays(season: int) -> pd.DataFrame:
    return dt.load_seasons([season])


@st.cache_data(show_spinner="Scouting every play caller's film…", ttl=dt.REFRESH_HOURS * 3600)
def scored(season: int, weeks: tuple[int, int] | None, by: str) -> tuple[pd.DataFrame, list[str]]:
    d, clues = up.score(plays(season), season, weeks, by=by)
    keep = ["posteam", "caller", "Week", "game_id", "PASS", "pass_oe", "epa", "scored", "xpass", "success",
            "explosive", "first_down", "DN", "ydstogo", "qtr", "score_differential", "fixed_drive",
            "fixed_drive_result"]
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
    """A caller table plus what every chart needs: color (readable on this theme), logo, label, opacity."""
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
    if "Edge" in d:                                       # film index tables
        d["Film edge"] = d["Edge"] * 100                  # bits per 100 snaps
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


def logo_scatter(d: pd.DataFrame, x: str, y: str, x_title: str, y_title: str, x_fmt: str, y_fmt: str,
                 trend: bool = False, ref: tuple[float | None, float | None] = (None, None), tip: list | None = None,
                 logos: bool = True) -> alt.LayerChart:
    """One play caller per point, drawn as the team logo (a dot in the team's color if the logo is missing, or
    always when logos=False, for charts with hundreds of points)."""
    d = d.dropna(subset=[x, y])
    # Name only the points at the edges (the middle of the cloud overlaps); hover names the rest.
    rank = lambda s: s.rank(pct=True)
    d = d.assign(show_label=(rank(d[x]) - 0.5).abs().ge(0.4) | (rank(d[y]) - 0.5).abs().ge(0.4))
    pad_x, pad_y = (d[x].max() - d[x].min()) * 0.08 or 0.01, (d[y].max() - d[y].min()) * 0.08 or 0.01
    base = alt.Chart(d).encode(
        x=alt.X(f"{x}:Q", title=x_title, scale=alt.Scale(domain=[d[x].min() - pad_x, d[x].max() + pad_x]),
                axis=alt.Axis(format=x_fmt, tickCount=8)),
        y=alt.Y(f"{y}:Q", title=y_title, scale=alt.Scale(domain=[d[y].min() - pad_y, d[y].max() + pad_y]),
                axis=alt.Axis(format=y_fmt, tickCount=8)), tooltip=tip or TIP)
    layers = []
    for axis, v in zip("xy", ref):
        if v is not None:
            layers.append(alt.Chart(pd.DataFrame({"v": [v]})).mark_rule(color="#888780", strokeDash=[4, 4])
                          .encode(**{axis: alt.X("v:Q", scale=alt.Scale(zero=False)) if axis == "x" else
                                     alt.Y("v:Q", scale=alt.Scale(zero=False))}))
    if trend:
        layers.append(base.transform_regression(x, y).mark_line(color="#888780", strokeDash=[6, 4], strokeWidth=2)
                      .encode(tooltip=alt.value(None)))
    if not logos:
        layers.append(base.mark_circle(size=55).encode(color=alt.Color("color:N", scale=None),
                                                       opacity=alt.value(0.75)))
        return alt.layer(*layers).properties(height=460)
    layers += [
        base.transform_filter("datum.logo == ''").mark_circle(size=500).encode(
            color=alt.Color("color:N", scale=None), opacity=alt.Opacity("opacity:Q", scale=None)),
        base.mark_image(width=36, height=36).encode(url="logo:N", opacity=alt.Opacity("opacity:Q", scale=None)),
        base.transform_filter("datum.show_label").mark_text(dy=24, fontSize=10, color=cm.theme_ink()).encode(
            text="label:N", opacity=alt.Opacity("opacity:Q", scale=None)),
    ]
    return alt.layer(*layers).properties(height=460)


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



# ---- Outcomes: does the score go with a better offense? ------------------------------------------------------
SPANS = ["This season", "Last 5 seasons", "1999 to this season"]


@st.cache_data(show_spinner=False, ttl=dt.REFRESH_HOURS * 3600)
def _precomputed(kind: str) -> pd.DataFrame:
    return dt.load_precomputed(kind)


def _from_precomputed(kind: str, season: int, neutral: bool) -> pd.DataFrame | None:
    """The weekly job's rows for this season. A finished season's never change; the current season's are used only if
    they cover every game the app has loaded (a newer download since Tuesday's refresh means computing live)."""
    t = _precomputed(kind)
    if t.empty:
        return None
    rows = t[(t["Season"] == season) & (t["Neutral"] == neutral)]
    if rows.empty or (season >= dt.current_season()
                      and rows["_season_games"].iloc[0] < plays(season)["game_id"].nunique()):
        return None
    # seasons are stacked in one file, so drop the columns this season doesn't have (e.g. Motion before 2022)
    return rows.drop(columns=["Neutral", "_season_games"]).dropna(axis=1, how="all").reset_index(drop=True)


@st.cache_data(show_spinner="Scoring a season…", ttl=dt.REFRESH_HOURS * 3600)
def film_season(season: int, neutral: bool) -> pd.DataFrame:
    """One season's film index (whole season, by play caller) with every outcome over the same scored snaps."""
    pre = _from_precomputed("film", season, neutral)
    if pre is not None:
        return pre
    d, clues = scored(season, None, "caller")
    return up.film_season_table(d, clues, season, neutral)


@st.cache_data(show_spinner="Measuring a season…", ttl=dt.REFRESH_HOURS * 3600)
def balance_season(season: int, neutral: bool) -> pd.DataFrame:
    """One season's balance table (whole season, by play caller) with every outcome over the same decision-down
    snaps."""
    pre = _from_precomputed("balance", season, neutral)
    if pre is not None:
        return pre
    return up.balance_season_table(plays(season), season, neutral)


def stacked(loader, seasons: range | list[int], neutral: bool) -> pd.DataFrame:
    frames = [loader(s, neutral) for s in seasons]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _vs_season(d: pd.DataFrame, col: str) -> pd.Series:
    """The outcome minus that season's average across play callers (outcomes drift up over the years)."""
    return d[col] - d.groupby("Season")[col].transform("mean")


def outcome_trend_section(loader, x: str, x_title: str, x_fmt: str, more: str, per: float, unit: str,
                          info: pd.DataFrame, highlight: list[str], season: int, snaps_note: str, key: str) -> None:
    """An outcome against a score, one point per play caller, with a least-squares line and a plain verdict on
    whether a trend exists. `loader(season, neutral)` returns the score table with up.outcomes() columns."""
    seasons = dt.available_seasons()
    c1, c2, c3 = st.columns([2, 1.4, 1.4])
    span = c1.segmented_control("Play callers from", SPANS, default=SPANS[0], key=f"{key}_span") or SPANS[0]
    outcome = c2.selectbox("Offense measured by", list(up.OUTCOMES), key=f"{key}_outcome",
                           help="\n".join(f"{k}: {v[1]}" for k, v in up.OUTCOMES.items()))
    neutral = c3.toggle("Neutral game script only", False, key=f"{key}_neutral",
                        help=f"Count only snaps within {up.NEUTRAL_MARGIN} points and outside the 4th quarter, so "
                             "teams protecting or chasing a lead don't skew the offense's numbers.")
    first = {"This season": season, "Last 5 seasons": max(seasons[0], season - 4)}.get(span, seasons[0])
    d = stacked(loader, range(first, season + 1), neutral)
    if d.empty:
        st.info("No play callers with enough snaps here yet.")
        return
    d = with_team(d, info, highlight, "caller").dropna(subset=[x, outcome])
    fmt = up.OUTCOMES[outcome][0]
    if span == "This season":
        d["y"], y_title, logos = d[outcome], f"{outcome} →", True
    else:
        d["y"], y_title, logos = _vs_season(d, outcome), f"{outcome} vs that season's average →", False
        d["label"] = d["label"] + " " + d["Season"].astype(str)
    tip = [alt.Tooltip("Play caller:N"), alt.Tooltip("Team name:N", title="Team"), alt.Tooltip("Season:O"),
           alt.Tooltip(f"{x}:Q", format=x_fmt, title=x), alt.Tooltip(f"{outcome}:Q", format=fmt),
           alt.Tooltip("Outcome snaps:Q", format=",", title="Snaps")]
    y_fmt = "+.1%" if "%" in fmt else "+.2f"
    st.altair_chart(logo_scatter(d, x, "y", x_title, y_title, x_fmt, y_fmt, trend=True, tip=tip, logos=logos),
                    width="stretch")
    st.markdown(cm.trend_sentence(cm.fit_trend(d[x], d["y"]), more=more, per=per, unit=unit, outcome=outcome))
    st.caption(f"{len(d)} play callers{'' if span == 'This season' else ' (one point per play caller per season)'}, "
               f"whole regular seasons. {outcome} is over {snaps_note}" +
               (f", within {up.NEUTRAL_MARGIN} points and outside the 4th quarter" if neutral else "") +
               ". Dashed line = least-squares fit. r = correlation (-1 to 1); p = the chance of a slope this steep if "
               "there were no real link (under 0.05 = a clear trend). A trend is a link, not proof of cause: a good "
               "offense can be the reason a coach can afford to be predictable, or not.")


def coach_history_section(loader, x: str, x_title: str, x_fmt: str, more: str, per: float, unit: str,
                          info: pd.DataFrame, key: str) -> None:
    """A coach over the years: his score and his offense (vs each season's average) season by season, and, across
    every coach with 3+ seasons, whether his offense does better in the seasons he scores higher than his own
    norm. Comparing a coach with himself takes out what doesn't change: his scheme, and often his quarterback."""
    seasons = dt.available_seasons()
    c1, c2 = st.columns([2, 1.4])
    outcome = c2.selectbox("Offense measured by", list(up.OUTCOMES), key=f"{key}_outcome")
    d = stacked(loader, seasons, False)
    if d.empty:
        return
    d = with_team(d, info, [], "caller")
    d = d[~d["Play caller"].str.endswith(" play caller")]
    d["y"] = _vs_season(d, outcome)
    counts = d.groupby("Play caller")["Season"].nunique()
    regulars = counts[counts >= 3].sort_values(ascending=False)
    if regulars.empty:
        st.info("No play caller has 3+ seasons with enough snaps yet.")
        return
    names = sorted(regulars.index)
    default = "Andy Reid" if "Andy Reid" in names else regulars.index[0]
    who = c1.selectbox("Play caller", names, index=names.index(default), key=f"{key}_who",
                       help="Play callers with 3+ seasons on the list (1999 on).")
    me = d[d["Play caller"] == who].sort_values("Season")
    fmt = up.OUTCOMES[outcome][0]
    base = alt.Chart(me).encode(x=alt.X("Season:O", title=None, axis=alt.Axis(labelAngle=0)),
                                tooltip=["Season", "Team", alt.Tooltip(f"{x}:Q", format=x_fmt),
                                         alt.Tooltip(f"{outcome}:Q", format=fmt)])
    top = alt.layer(base.mark_line(color="#888780"), base.mark_circle(size=120, opacity=1).encode(
        color=alt.Color("color:N", scale=None)), ).encode(
        y=alt.Y(f"{x}:Q", title=x, scale=alt.Scale(zero=False))).properties(height=200)
    zero = alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(color="#888780", strokeDash=[4, 4]).encode(y="v:Q")
    bottom = alt.layer(zero, base.mark_bar(width=18, opacity=0.85).encode(
        y=alt.Y("y:Q", title=f"{outcome} vs league", axis=alt.Axis(format="+.0%" if "%" in fmt else "+.2f")),
        color=alt.Color("color:N", scale=None))).properties(height=180)
    st.altair_chart(alt.vconcat(top, bottom), width="stretch")
    st.caption(f"Top: his {x.lower()} each season (dot = the team he called plays for). Bottom: his offense's "
               f"{outcome if outcome == 'EPA/play' else outcome.lower()} that season against the average play caller that "
               "season.")

    reg = d[d["Play caller"].isin(regulars.index)].copy()
    for col in (x, "y"):
        reg[f"{col} vs own"] = reg[col] - reg.groupby("Play caller")[col].transform("mean")
    t = cm.fit_trend(reg[f"{x} vs own"], reg["y vs own"])
    st.markdown("###### Within the same coach")
    st.markdown(cm.trend_sentence(t, more=f"{more} than usual", per=per, unit=unit, outcome=outcome, noun="coaches"))
    st.caption(f"{len(reg)} seasons from {reg['Play caller'].nunique()} play callers with 3+ seasons. Each season is "
               "compared with the same coach's average, so what stays the same for him (scheme, often the "
               "quarterback) drops out.")
