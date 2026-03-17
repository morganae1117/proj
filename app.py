import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Gen AI Downtime Prototype", layout="wide")

st.title("Gen AI Downtime and Maintenance Prototype")
st.write("This prototype combines eMaint maintenance records and coffee downtime records to identify repeated downtime patterns and support engineering review.")

# -------------------------------
# LOAD DATA
# -------------------------------
# Option 1:
# Put your CSV files in the same folder as app.py and keep these file names exactly the same.
df1 = pd.read_csv("Emaint Data.csv")
df2 = pd.read_csv("Coffee Downtime and Maintenance Data.csv")

# If you put them in a data folder instead, use:
# df1 = pd.read_csv("data/Emaint Data.csv")
# df2 = pd.read_csv("data/Coffee Downtime and Maintenance Data.csv")

# -------------------------------
# CLEAN COLUMN NAMES
# -------------------------------
df1.columns = df1.columns.str.strip().str.lower().str.replace(" ", "_")
df2.columns = df2.columns.str.strip().str.lower().str.replace(" ", "_")

# -------------------------------
# BUILD TEXT FIELDS
# -------------------------------
def make_text_emaint(row):
    return f"""
    Asset ID: {row.get('asset_id', '')}
    Equipment Description: {row.get('equipment_description', '')}
    Line No: {row.get('line_no', '')}
    Work Order Type: {row.get('wo_type', '')}
    Failure Type: {row.get('failure_type', '')}
    Downtime: {row.get('downtime', '')}
    Date: {row.get('wo_date', '')}
    """.strip()

df1["combined_text"] = df1.apply(make_text_emaint, axis=1)

def make_text_coffee(row):
    return f"""
    Plant: {row.get('plantname', '')}
    Line: {row.get('linename', '')}
    Shift: {row.get('shiftname', '')}
    Order Number: {row.get('activeordernumber', '')}
    Shift Start Date: {row.get('shiftstartdate', '')}
    Material: {row.get('materialdescr', '')}
    Uptime: {row.get('uptime', '')}
    Total Downtime: {row.get('totaldowntime', '')}
    Unplanned Downtime: {row.get('unplanneddowntime', '')}
    Planned Downtime: {row.get('planneddowntime', '')}
    Other Downtime: {row.get('otherdowntime', '')}
    Changeover: {row.get('changeover', '')}
    Quantity In: {row.get('qtyin', '')}
    Quantity Out: {row.get('qtyout', '')}
    Quantity Processed: {row.get('qtyprocessed', '')}
    Quantity Rejected: {row.get('qtyrejected', '')}
    Audit Status: {row.get('auditstatus', '')}
    Data Source: {row.get('datasource', '')}
    """.strip()

df2["combined_text"] = df2.apply(make_text_coffee, axis=1)

# -------------------------------
# TREND PREP
# -------------------------------
df1["wo_date"] = pd.to_datetime(df1["wo_date"], errors="coerce")
df1["downtime"] = pd.to_numeric(df1["downtime"], errors="coerce")

df1_trend = df1.dropna(subset=["wo_date"]).copy()
df1_trend["month"] = df1_trend["wo_date"].dt.to_period("M").astype(str)

# -------------------------------
# SUMMARIES
# -------------------------------
failure_trend = (
    df1_trend.groupby(["month", "failure_type"])
    .size()
    .unstack(fill_value=0)
    .sort_index()
)

wo_trend = (
    df1_trend.groupby(["month", "wo_type"])
    .size()
    .unstack(fill_value=0)
    .sort_index()
)

failure_summary = (
    df1_trend.groupby("failure_type")
    .agg(
        work_order_count=("wo_no.", "count"),
        total_downtime_hours=("downtime", "sum"),
        avg_downtime_hours=("downtime", "mean")
    )
    .sort_values(by="work_order_count", ascending=False)
)

wo_summary = (
    df1_trend.groupby("wo_type")
    .agg(
        work_order_count=("wo_no.", "count"),
        total_downtime_hours=("downtime", "sum"),
        avg_downtime_hours=("downtime", "mean")
    )
    .sort_values(by="work_order_count", ascending=False)
)

equipment_summary = (
    df1_trend.groupby(["asset_id", "equipment_description", "line_no"])
    .agg(
        work_order_count=("wo_no.", "count"),
        total_downtime_hours=("downtime", "sum"),
        avg_downtime_hours=("downtime", "mean")
    )
    .sort_values(by="work_order_count", ascending=False)
)

monthly_downtime = (
    df1_trend.groupby("month")
    .agg(
        total_work_orders=("wo_no.", "count"),
        total_downtime_hours=("downtime", "sum"),
        avg_downtime_hours=("downtime", "mean")
    )
    .sort_index()
)

# -------------------------------
# SOURCE TAGS
# -------------------------------
df1["source"] = "emaint"
df2["source"] = "coffee"

combined_df = pd.concat([
    df1[["source", "combined_text"]],
    df2[["source", "combined_text"]]
], ignore_index=True)

emaint_df = combined_df[combined_df["source"] == "emaint"].copy()
coffee_df = combined_df[combined_df["source"] == "coffee"].copy()

vectorizer_emaint = TfidfVectorizer(stop_words="english")
X_emaint = vectorizer_emaint.fit_transform(emaint_df["combined_text"].fillna(""))

vectorizer_coffee = TfidfVectorizer(stop_words="english")
X_coffee = vectorizer_coffee.fit_transform(coffee_df["combined_text"].fillna(""))

def retrieve_both_sources(query, top_k_each=5):
    query_vec_emaint = vectorizer_emaint.transform([query])
    similarities_emaint = cosine_similarity(query_vec_emaint, X_emaint).flatten()
    top_indices_emaint = similarities_emaint.argsort()[-top_k_each:][::-1]
    top_emaint = emaint_df.iloc[top_indices_emaint].assign(score=similarities_emaint[top_indices_emaint])

    query_vec_coffee = vectorizer_coffee.transform([query])
    similarities_coffee = cosine_similarity(query_vec_coffee, X_coffee).flatten()
    top_indices_coffee = similarities_coffee.argsort()[-top_k_each:][::-1]
    top_coffee = coffee_df.iloc[top_indices_coffee].assign(score=similarities_coffee[top_indices_coffee])

    return top_emaint, top_coffee

# -------------------------------
# STREAMLIT LAYOUT
# -------------------------------
st.subheader("Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("eMaint Rows", len(df1))
col2.metric("Coffee Rows", len(df2))
col3.metric("Combined Rows", len(combined_df))

st.subheader("User Query")
query = st.text_input(
    "Enter a question",
    value="What patterns are causing repeated downtime and what maintenance actions should be considered?"
)

top_k = st.slider("Top matches per source", min_value=1, max_value=10, value=5)

if st.button("Run Retrieval"):
    top_emaint, top_coffee = retrieve_both_sources(query, top_k_each=top_k)

    st.subheader("Top eMaint Matches")
    st.dataframe(top_emaint[["source", "score", "combined_text"]], use_container_width=True)

    st.subheader("Top Coffee Matches")
    st.dataframe(top_coffee[["source", "score", "combined_text"]], use_container_width=True)

    st.subheader("Prototype Findings")
    st.markdown("""
The retrieved eMaint records suggest recurring corrective maintenance activity associated with process-related failures.
This indicates that some assets and failure categories may be appearing more than once and may deserve closer review.

The retrieved coffee downtime records suggest repeated line-level downtime events, including cases with very low uptime,
limited production output, and relatively high downtime. This points to possible recurring interruptions in production activity.

Overall, the prototype suggests possible repeated downtime patterns linked to maintenance history and production-line events.
These results support further review of repeat failures, affected equipment, and higher-downtime operating periods.
""")

st.subheader("Failure Type Trend by Month")
st.dataframe(failure_trend, use_container_width=True)

st.subheader("Work Order Type Trend by Month")
st.dataframe(wo_trend, use_container_width=True)

st.subheader("Failure Type Summary")
st.dataframe(failure_summary, use_container_width=True)

st.subheader("Work Order Type Summary")
st.dataframe(wo_summary, use_container_width=True)

st.subheader("Repeated Issues by Asset / Equipment / Line")
st.dataframe(equipment_summary.head(15), use_container_width=True)

st.subheader("Monthly Downtime Summary")
st.dataframe(monthly_downtime, use_container_width=True)
