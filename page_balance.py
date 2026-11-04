"""
Balance Score page: how close to 50/50 each play caller calls run and pass on every decision down, where his balance
goes, and whether balance goes with a better offense. The scores live in unpredictability.py (balance_*).
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import common as cm
import data as dt
import ui
import unpredictability as up

# ---- Balance score --------------------------------------------------------------------------------------
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
    d = up.sample(ui.plays(season), season, weeks)
    if by == "team":
        d["caller"] = d["posteam"]
    cells = up.balance_cells(d, by_formation)
    third = cells[cells["dd"] == "3rd & medium"]
    league_third = float((third["Pass rate"] * third["Snaps"]).sum() / third["Snaps"].sum()) if len(third) else np.nan
    return (up.balance_table(d, by_formation), cells, up.balance_reliability(d, by_formation), league_third)


def balance_anchors_html(g: pd.DataFrame) -> str:
    def one(r, label):
        return (f'<div class="gl-anchor">{ui.badge_html(r, 52)}<div style="min-width:0">'
                f'<div class="gl-lbl" style="color:{BALANCE_COLORS[r["Balance tier"]]}">{label}</div>'
                f'<div class="gl-nm">{r["Play caller"]}</div><div class="gl-sub">{r["Team name"]} · typical down is a '
                f'{up.split_text(r["Typical split"])} call</div></div></div>')
    s = g.sort_values("Balance score", ascending=False)
    return ui.GLANCE_CSS + f'<div class="gl-row">{one(s.iloc[0], "Most balanced")}{one(s.iloc[-1], "Most lopsided")}</div>'


def balance_board_html(g: pd.DataFrame) -> str:
    cols = []
    for _, name in up.BALANCE_TIERS:
        color = BALANCE_COLORS[name]
        rows = g[g["Balance tier"] == name].sort_values("Balance score", ascending=False)
        chips = "".join(f'<div class="gl-chip">{ui.badge_html(r)}<span>{r["Play caller"]}</span>'
                        f'<span class="q">{up.split_text(r["Typical split"])}</span></div>'
                        for _, r in rows.iterrows()) or '<div class="gl-def">Nobody this season</div>'
        cols.append(f'<div class="gl-col" style="border-top-color:{color}"><h4 style="color:{color}">{name}</h4>'
                    f'<div class="gl-def">{up.BALANCE_TIER_HELP[name]}</div>{chips}</div>')
    return ui.GLANCE_CSS + f'<div class="gl-board">{"".join(cols)}</div>'


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
    html = "".join(f'<div class="gl-tell"><div class="gl-head" style="gap:10px">{ui.badge_html(r, 34)}<div>'
                   f'<div class="gl-look">{kind}</div><div style="font-weight:700;font-size:17px">{r["Play caller"]}</div>'
                   f'</div></div><div class="gl-cmp" style="font-size:14px;opacity:.85">{text}</div></div>'
                   for r, kind, text in cards)
    return ui.GLANCE_CSS + f'<div class="gl-tells">{html}</div>'


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
    logos = base.mark_image(width=ui.LOGO, height=ui.LOGO, baseline="bottom").encode(
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
    d = ui.with_team(t.assign(Edge=np.nan, **{"Lean edge": np.nan, "Look edge": np.nan, "Film accuracy": np.nan,
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
        f = ui.with_team(up.caller_table(*ui.scored(season, weeks, by)), info, highlight, by)
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
        st.altair_chart(ui.logo_scatter(both, "Balance score", "Unpredictability index", "More balanced by down & distance →",
                                     "Harder to read on film (unpredictability index) →", ".0f", ".0f",
                                     ref=(float(both["Balance score"].mean()), 50.0)), width="stretch")
        st.caption(f"r = {r:+.2f}: the two barely agree. Bottom right = balanced by down, yet readable (his lean, "
                   "formation, or personnel gives it away). Top left = lopsided by down, yet hard to read (his habits "
                   "don't hold from one game to the next).")
    else:
        st.caption("Not enough film yet to compare with the film index.")

    st.subheader("Does balance pay off?")
    st.markdown("Offensive EPA per play against the balance score: is a more 50/50 play caller a more efficient one?")
    ui.epa_trend_section(d, ui.balance_seasons, "Balance score", "More balanced by down & distance →", ".0f",
                         more="more balanced by down & distance", per=5, unit="balance points", info=info,
                         highlight=highlight, season=season, key="bal_epa_span")
    if by_formation:
        st.caption("Pooled seasons use the plain balance score (no formation split).")

    st.subheader("Scouting card")
    order = d.sort_values("Balance score", ascending=False)
    labels = {f"{r['Play caller']} ({r['Team']}) · {r['Typical call']} · {r['Balance tier']}": i
              for i, r in order.iterrows()}
    pick = st.selectbox("Play caller", list(labels), key=f"up_bal_pick_{season}",
                        help="Sorted from most balanced to most lopsided.")
    r = d.loc[labels[pick]]
    mine = cells[(cells["posteam"] == r["Team"]) & (cells["caller"] == r["Play caller"])]
    color = BALANCE_COLORS[r["Balance tier"]]
    st.html(ui.GLANCE_CSS + f'<div class="gl-card"><div class="gl-head">{ui.badge_html(r, 56)}<div style="min-width:0">'
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


# ---- Page ----------------------------------------------------------------------------------------------------
seasons = dt.available_seasons()
info = ui.team_info()
callers_csv = up.load_callers()
with st.sidebar:
    st.header("Balance Score")
    st.session_state.setdefault("bal_season", seasons[-1])
    season = st.selectbox("Season", seasons[::-1], key="bal_season")
    season_plays = ui.plays(season)
    reg_weeks = season_plays.loc[season_plays["season_type"] == "REG", "Week"]
    if reg_weeks.empty:
        st.info("No regular-season plays for this season yet.")
        st.stop()
    max_week = int(reg_weeks.max())
    weeks = st.slider("Weeks", 1, max_week, (1, max_week), key=f"bal_weeks_{season}")
    by_label = st.segmented_control("Compare", ["Play callers", "Teams"], default="Play callers", key="bal_by",
                                    help="Play callers: a mid-season change of play caller is scored separately. "
                                         "Teams: one score per team.")
    by = "team" if by_label == "Teams" else "caller"
    min_snaps = st.slider("Minimum snaps to show", 0, 600, up.MIN_SNAPS, 25, key="bal_min")
    highlight = st.multiselect("Highlight teams", sorted(season_plays["posteam"].dropna().unique()), key="bal_hl",
                               help="Fades every other team in the charts.")
    by_formation = st.toggle("Split by shotgun / under center", False, key="bal_form",
                             help="Score each down & distance separately from shotgun and from under center, closer "
                                  "to what a defense sees before the snap.")

render_balance(season, None if weeks == (1, max_week) else weeks, by, min_snaps, highlight, by_formation,
               sorted(s for s in callers_csv["season"].unique() if s in seasons))
