import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="GenAI Maintenance Prototype", layout="wide")

st.title("GenAI Maintenance & Downtime Prototype")
st.write(
    "This prototype retrieves relevant records from eMaint and coffee downtime data "
    "to support engineering review of repeated downtime patterns."
)

EMAINT_FILE = "Emaint Data.csv"
COFFEE_FILE = "Coffee Downtime and Maintenance Data.csv"


@st.cache_data
def load_data():
    df1 = pd.read_csv(EMAINT_FILE)
    df2 = pd.read_csv(COFFEE_FILE)

    df1.columns = df1.columns.str.strip().str.lower().str.replace(" ", "_")
    df2.columns = df2.columns.str.strip().str.lower().str.replace(" ", "_")

    def make_text_emaint(row):
        return f"""
Asset ID: {row.get('asset_id', '')}
Equipment: {row.get('equipment_description', '')}
Work Order Type: {row.get('wo_type', '')}
Failure Type: {row.get('failure_type', '')}
Downtime: {row.get('downtime', '')}
Date: {row.get('wo_date', '')}
""".strip()

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

    df1["combined_text"] = df1.apply(make_text_emaint, axis=1)
    df2["combined_text"] = df2.apply(make_text_coffee, axis=1)

    df1["source"] = "emaint"
    df2["source"] = "coffee"

    combined_df = pd.concat(
        [df1[["source", "combined_text"]], df2[["source", "combined_text"]]],
        ignore_index=True
    )

    return df1, df2, combined_df


@st.cache_resource
def build_retrievers(combined_df):
    emaint_df = combined_df[combined_df["source"] == "emaint"].copy()
    coffee_df = combined_df[combined_df["source"] == "coffee"].copy()

    vectorizer_emaint = TfidfVectorizer(stop_words="english")
    X_emaint = vectorizer_emaint.fit_transform(emaint_df["combined_text"].fillna(""))

    vectorizer_coffee = TfidfVectorizer(stop_words="english")
    X_coffee = vectorizer_coffee.fit_transform(coffee_df["combined_text"].fillna(""))

    return emaint_df, coffee_df, vectorizer_emaint, X_emaint, vectorizer_coffee, X_coffee


def retrieve_both_sources(query, top_k_each, emaint_df, coffee_df, vectorizer_emaint, X_emaint, vectorizer_coffee, X_coffee):
    query_vec_emaint = vectorizer_emaint.transform([query])
    similarities_emaint = cosine_similarity(query_vec_emaint, X_emaint).flatten()
    top_indices_emaint = similarities_emaint.argsort()[-top_k_each:][::-1]
    top_emaint = emaint_df.iloc[top_indices_emaint].assign(score=similarities_emaint[top_indices_emaint])

    query_vec_coffee = vectorizer_coffee.transform([query])
    similarities_coffee = cosine_similarity(query_vec_coffee, X_coffee).flatten()
    top_indices_coffee = similarities_coffee.argsort()[-top_k_each:][::-1]
    top_coffee = coffee_df.iloc[top_indices_coffee].assign(score=similarities_coffee[top_indices_coffee])

    return top_emaint, top_coffee


def summarize_results(top_emaint, top_coffee):
    return """
Prototype interpretation:
- The eMaint results highlight repeated corrective maintenance and process-failure related events.
- The coffee results highlight major downtime periods, including zero-uptime / zero-output situations on specific lines.
- Together, the retrieved records provide evidence for investigating repeat-failure assets and severe production interruptions.
- Recommended next step: review recurring assets and compare maintenance history against repeated downtime/zero-throughput periods.
""".strip()


try:
    df1, df2, combined_df = load_data()
    emaint_df, coffee_df, vectorizer_emaint, X_emaint, vectorizer_coffee, X_coffee = build_retrievers(combined_df)

    st.success("Data loaded successfully.")
    st.write("Source counts:")
    st.write(combined_df["source"].value_counts())

    query = st.text_area(
        "Enter a maintenance or downtime question:",
        value="What patterns are causing repeated downtime and what maintenance actions should be considered?"
    )

    top_k_each = st.slider("Top records from each source", min_value=1, max_value=10, value=5)

    if st.button("Run Prototype"):
        top_emaint, top_coffee = retrieve_both_sources(
            query,
            top_k_each,
            emaint_df,
            coffee_df,
            vectorizer_emaint,
            X_emaint,
            vectorizer_coffee,
            X_coffee
        )

        st.subheader("Prototype Findings")
        st.write(summarize_results(top_emaint, top_coffee))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("eMaint Results")
            for i in range(len(top_emaint)):
                with st.expander(f"eMaint Match {i+1} | Score: {top_emaint.iloc[i]['score']:.4f}"):
                    st.text(top_emaint.iloc[i]["combined_text"])

        with col2:
            st.subheader("Coffee Results")
            for i in range(len(top_coffee)):
                with st.expander(f"Coffee Match {i+1} | Score: {top_coffee.iloc[i]['score']:.4f}"):
                    st.text(top_coffee.iloc[i]["combined_text"])

        st.subheader("Limitations")
        st.write(
            "This prototype is retrieval-based. It identifies relevant patterns for engineering review, "
            "but it does not prove final root cause. Some records also have sparse or missing descriptive fields."
        )

except FileNotFoundError as e:
    st.error("Could not find one or both CSV files.")
    st.write("Make sure these files are in the same GitHub repo as app.py:")
    st.code(EMAINT_FILE)
    st.code(COFFEE_FILE)
    st.write(f"Missing file detail: {e}")
except Exception as e:
    st.error("An error occurred while running the app.")
    st.exception(e)
