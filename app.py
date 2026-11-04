"""
NFL Tendencies: scouting and play-calling views built on nflfastR play-by-play.

Run with:  streamlit run app.py

    app.py              navigation between the pages
    page_tendencies.py  Tendencies: the Dowling app's Scout Opposing Offense page for any NFL offense
    page_balance.py     Balance Score: how close to 50/50 each play caller is on every decision down
    page_unpredictability.py  Unpredictability Index: how much a defense learns from each play caller's film
    page_series.py      3rd & 4th Down: do runs or passes win more series once the 4th-down play counts
    data.py             downloads nflfastR pbp + participation from nflverse and builds the app's columns
    ui.py               shared pieces for the play-caller pages (team colors and logos, charts, EPA trend)
    series.py           3rd-down series outcomes (no Streamlit code)
    unpredictability.py the unpredictability model (no Streamlit code); play_callers.csv = who called plays, by week
    common.py           metrics, NFL comparison columns, table colors, tendency badges, split bars
    tab_*.py            the Tendencies page's tabs; pass_chart.py draws the pass chart
"""
import streamlit as st

st.set_page_config(page_title="NFL Tendencies", page_icon="🏈", layout="wide")

pg = st.navigation([
    st.Page("page_tendencies.py", title="Tendencies", icon=":material/target:", url_path="tendencies", default=True),
    st.Page("page_balance.py", title="Balance Score", icon=":material/balance:", url_path="balance"),
    st.Page("page_unpredictability.py", title="Unpredictability Index", icon=":material/shuffle:",
            url_path="unpredictability"),
    st.Page("page_series.py", title="3rd & 4th Down", icon=":material/flag:", url_path="third-down"),
])
pg.run()
