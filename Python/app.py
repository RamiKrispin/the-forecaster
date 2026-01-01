import streamlit as st
import plotly.express as px
from streamlit_plotly_events import plotly_events

df = px.data.iris()

fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")

selected = plotly_events(
    fig,
    click_event=True,
    hover_event=False,
    select_event=False,
)

st.write("Clicked point:", selected)
