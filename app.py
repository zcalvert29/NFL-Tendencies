"""
NFL Tendencies: scouting and play-calling views built on nflfastR play-by-play.

Run with:  streamlit run app.py

    app.py              navigation between the pages
    page_tendencies.py  Tendencies: the Dowling app's Scout Opposing Offense page for any NFL offense
    page_identity.py    Play-Calling Identity: which offenses call it unlike an average NFL team, and which are average
    data.py             downloads nflfastR pbp + participation from nflverse and builds the app's columns
    identity.py         the play-calling identity measures (no Streamlit code)
    common.py           metrics, NFL comparison columns, table colors, tendency badges, split bars
    tab_*.py            the Tendencies page's tabs; pass_chart.py draws the pass chart
"""
import streamlit as st

st.set_page_config(page_title="NFL Tendencies", page_icon="🏈", layout="wide")

pg = st.navigation([
    st.Page("page_tendencies.py", title="Tendencies", icon=":material/target:", url_path="tendencies", default=True),
    st.Page("page_identity.py", title="Play-Calling Identity", icon=":material/scatter_plot:", url_path="identity"),
])
pg.run()
