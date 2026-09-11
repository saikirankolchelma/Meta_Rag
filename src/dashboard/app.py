"""Operational dashboard over request_logs: queries per route, latency, and
estimated cost savings vs. an all-GPT-4o baseline.

Run with: streamlit run src/dashboard/app.py
"""

import warnings

import pandas as pd
import plotly.express as px
import psycopg2
import streamlit as st

# pandas warns that psycopg2 connections aren't a tested DBAPI2 path for
# read_sql; it works fine here and SQLAlchemy would be overkill for one query.
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")

from src.utils.config import POSTGRES_URL

# Reference categorical palette (dataviz skill default) — fixed hue order,
# one slot per route, never reassigned by filtering.
ROUTE_COLORS = {
    "naive": "#2a78d6",   # blue
    "parent": "#eb6834",  # orange
    "hyde": "#1baf7a",    # aqua
}
ROUTE_ORDER = ["naive", "parent", "hyde"]

# Blended per-1M-token GPT-4o pricing, matching dispatcher.py's cost estimate,
# used only to compute the hypothetical "if every query used gpt-4o" baseline.
GPT4O_BLENDED_COST_PER_QUERY = 0.006  # rough average observed cost of a hyde-tier call

st.set_page_config(page_title="Meta-RAG Dashboard", layout="wide")
st.title("Meta-RAG: Adaptive Gateway Dashboard")


@st.cache_data(ttl=30)
def load_logs() -> pd.DataFrame:
    conn = psycopg2.connect(POSTGRES_URL)
    df = pd.read_sql(
        "SELECT created_at, route, complexity_score, latency_ms, cost_usd, cache_hit "
        "FROM request_logs ORDER BY created_at",
        conn,
    )
    conn.close()
    return df


df = load_logs()

if df.empty:
    st.info("No requests logged yet. Send some queries to the gateway (`POST /chat`) to populate this dashboard.")
    st.stop()

total_requests = len(df)
total_cost = df["cost_usd"].sum()
cache_hit_rate = df["cache_hit"].mean()
baseline_cost = total_requests * GPT4O_BLENDED_COST_PER_QUERY
savings_pct = (1 - total_cost / baseline_cost) * 100 if baseline_cost > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total requests", total_requests)
col2.metric("Actual cost", f"${total_cost:.4f}")
col3.metric("Cost vs. all-GPT-4o baseline", f"${baseline_cost:.4f}", delta=f"-{savings_pct:.0f}%")
col4.metric("Cache hit rate", f"{cache_hit_rate:.0%}")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Queries per route")
    counts = df["route"].value_counts().reindex(ROUTE_ORDER).fillna(0).reset_index()
    counts.columns = ["route", "count"]
    fig = px.bar(counts, x="route", y="count", color="route", color_discrete_map=ROUTE_COLORS)
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Requests")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Average latency per route (ms)")
    latency = df.groupby("route")["latency_ms"].mean().reindex(ROUTE_ORDER).fillna(0).reset_index()
    fig = px.bar(latency, x="route", y="latency_ms", color="route", color_discrete_map=ROUTE_COLORS)
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="ms")
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Recent requests")
st.dataframe(df.sort_values("created_at", ascending=False).head(50), use_container_width=True)
