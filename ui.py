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



# ---- EPA/play trend ------------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Scoring every season…", ttl=dt.REFRESH_HOURS * 3600)
def film_seasons(seasons: tuple[int, ...]) -> pd.DataFrame:
    """The film index table for each season (whole season, by play caller), stacked."""
    out = []
    for s in seasons:
        t = up.caller_table(*scored(s, None, "caller"))
        out.append(t[t["Unpredictability index"].notna()].assign(Season=s))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


@st.cache_data(show_spinner="Measuring every season…", ttl=dt.REFRESH_HOURS * 3600)
def balance_seasons(seasons: tuple[int, ...], min_snaps: int = up.MIN_SNAPS) -> pd.DataFrame:
    """The balance table for each season (whole season, by play caller), stacked."""
    out = []
    for s in seasons:
        t = up.balance_table(up.sample(plays(s), s))
        out.append(t[t["Snaps"] >= min_snaps].assign(Season=s))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def epa_trend_section(this: pd.DataFrame, pooled_loader, x: str, x_title: str, x_fmt: str, more: str,
                      per: float, unit: str, info: pd.DataFrame, highlight: list[str], season: int,
                      key: str) -> None:
    """EPA/play against a score, one point per play caller, with a least-squares line and a plain verdict on whether
    a trend exists. `pooled_loader(seasons)` returns the same table for several seasons; pooled EPA/play is
    measured against each season's average, since the league's EPA/play rises over the years."""
    seasons = dt.available_seasons()
    span = st.segmented_control("Play callers from", ["This season", "Last 5 seasons", "1999 to this season"],
                                default="This season", key=key) or "This season"
    if span == "This season":
        d = this.dropna(subset=[x, "EPA/play"]).assign(y=lambda t: t["EPA/play"])
        y_title, logos = "Offensive EPA per play →", True
    else:
        first = seasons[0] if span == "1999 to this season" else max(seasons[0], season - 4)
        p = pooled_loader(tuple(range(first, season + 1)))
        p = p.assign(y=p["EPA/play"] - p.groupby("Season")["EPA/play"].transform("mean"))
        d = with_team(p, info, highlight, "caller").dropna(subset=[x, "y"])
        d["label"] = d["label"] + " " + d["Season"].astype(str)
        y_title, logos = "EPA per play vs that season's average →", False
    tip = [alt.Tooltip("Play caller:N"), alt.Tooltip("Team name:N", title="Team"),
           *([alt.Tooltip("Season:O")] if "Season" in d else []), alt.Tooltip(f"{x}:Q", format=x_fmt, title=x),
           alt.Tooltip("EPA/play:Q", format="+.3f")]
    st.altair_chart(logo_scatter(d, x, "y", x_title, y_title, x_fmt, "+.2f", trend=True, tip=tip, logos=logos),
                    width="stretch")
    t = cm.fit_trend(d[x], d["y"])
    st.markdown(cm.trend_sentence(t, more=more, per=per, unit=unit))
    st.caption(f"{t['n']} play callers{'' if span == 'This season' else ' (one point per play caller per season)'}. "
               "Dashed line = least-squares fit. r = correlation (-1 to 1); p = the chance of a slope this steep if "
               "there were no real link (under 0.05 = a clear trend). A trend here is a link, not proof of cause: a "
               "good offense can be the reason a coach can afford to be predictable, or not.")
