import pandas as pd
import streamlit as st
import ast
import altair as alt
from pathlib import Path

# Create a brand list
brands = ['Banana Boat', 'Hawaiian Tropic', 'Neutrogena', 'Coppertone',
          'Stream2Sea', 'Raw Elements', 'Thinksport', 'Badger Balm', 'Sun Bum']

# Brands with fewer documents than this are flagged: their mean score is less reliable
MIN_DOCS = 10

#Set page to wide layout for better visualisation
st.set_page_config(layout="wide")

# Set title and description
st.header("Reef-safe Sunscreen & Greenwashing: the conversation, brand by brand")
st.caption(f"Demo dashboard: sentiment of online discussion about {len(brands)} sunscreen brands, anchored to the 2025 ACCC v Edgewell case.")

# Function to check if a brand is mentioned in the entities
def mentions_brand(entities_or_str, brand):
    try:
        if isinstance(entities_or_str, list):
            entities = entities_or_str
        else:
            entities = ast.literal_eval(entities_or_str)
    except (ValueError, SyntaxError, TypeError):
        return False
    for entity_text, entity_label in entities:
        if brand.lower() in entity_text.lower():
            return True
    return False

# Cache: the CSV is read once, not on every interaction
@st.cache_data
def load_data():
    root = Path(__file__).resolve().parent.parent
    return pd.read_csv(root/"data"/ "processed"/"df_sunscreen_dashboard.csv")

df = load_data()
n_synthetic = df["is_synthetic"].sum()
n_real = len(df) - n_synthetic

# Short introduction for readers who don't know the case
st.markdown("""
A 2025 court case (ACCC v Edgewell Personal Care) centred on Banana Boat and Hawaiian Tropic sunscreens, marketed in Australia as reef-friendly. The Australian consumer regulator (ACCC) alleges they contained ingredients that harm reefs or risk harming them.

We analysed online posts about these claims and measured the sentiment of the conversation around each brand: positive, neutral or negative.

Select a brand to see how it is discussed; the table gives the overview for all brands.
""")

# Display a warning about the dataset composition
st.warning(f"The dataset contains {n_real} real documents and {n_synthetic} synthetic (AI-generated) documents. The results are demonstrative and do not represent real public opinion.")

# Create columns for brand selection and summary table
col1, col2 = st.columns([1,1])
with col1:
    st.write("### Brand selection and sentiment distribution")
    # Create a selectbox for brand selection
    selected_brand = st.selectbox("Select a brand:", brands)
    mask = df["entities"].apply(mentions_brand, args=(selected_brand,))
    brand_df = df[mask]

    # Display the sentiment distribution for the selected brand
    # Reindex: always show three bars, even when a class has no documents
    shares = brand_df["roberta_label"].value_counts(normalize=True).reindex(["negative", "neutral", "positive"], fill_value=0)
    st.write(f"Based on {len(brand_df)} documents mentioning {selected_brand}.")
    chart_df = shares.reset_index()
    chart_df.columns = ["sentiment", "share"]
    # domain and range are paired by position: negative -> coral, neutral -> grey, positive -> teal
    chart = alt.Chart(chart_df).mark_bar(size=150).encode(
        x=alt.X("sentiment", sort=["negative", "neutral", "positive"], axis=alt.Axis(labelAngle=0, title="Sentiment")),
        y=alt.Y("share", axis=alt.Axis(format=".0%", title="Share of documents")),
        tooltip=[
        alt.Tooltip("sentiment", title="Sentiment"),
        alt.Tooltip("share", format=".0%", title="Share"),],
        color=alt.Color(
            "sentiment",
            scale=alt.Scale(domain=["negative", "neutral", "positive"], range=["#ff7f50", "#b0b0b0", "#008080"]),
            legend=None,
        ),
    ).properties(height=420)
    st.altair_chart(chart, use_container_width=True)
    st.caption("The bars show the share of documents associated with a sentiment: negative, neutral, or positive. This shows how the brand is discussed, not proof that it engages in greenwashing.")

with col2:
    st.write("### Summary Table")
    # Display a summary table for all brands
    rows = []
    for brand in brands:
        mask_brand = df["entities"].apply(mentions_brand, args=(brand,))
        rows.append({
            "brand": brand,
            "n_docs": mask_brand.sum(),
            "mean_score": df.loc[mask_brand, "roberta_score_numeric"].mean(),
        })

    summary = pd.DataFrame(rows)
    summary["note"] = " "
    summary.loc[summary["n_docs"] < MIN_DOCS, "note"] = "Few documents"
    st.dataframe(
    summary,
    hide_index=True,
    column_config={
        "brand": st.column_config.TextColumn("Brand"),
        "n_docs": st.column_config.NumberColumn("Number of documents"),
        "mean_score": st.column_config.NumberColumn("Mean sentiment score (-1 to 1)", format="%.2f"),
        "note": st.column_config.TextColumn("Note", width="medium"),
    },
    )
    st.caption("Mean sentiment score ranges from -1 (all documents negative) to +1 (all documents positive). Each document is scored as a whole, so a reef-safe brand may score lower than how it is actually discussed, for instance if mentioned as a trusted alternative in a document with overall negative sentiment.")
    st.caption(f"Brands marked \"Few documents\" are based on fewer than {MIN_DOCS} documents, so their average can change a lot with a single document.")

# Filter the dataframe for documents mentioning ACCC, Edgewell, or Federal Court
accc_pattern = "ACCC|Edgewell|Federal Court"
df_accc = df[df["Text"].str.contains(accc_pattern, case=False, na=False)]

# Create columns for VADER vs RoBERTa comparison and ACCC mentions
col3, col4 = st.columns([1,1])
with col3:
    st.write("### VADER vs RoBERTa comparison")
    st.write(f"Based on {len(df_accc)} documents mentioning the ACCC case.")
    # Count how many ACCC documents each model puts in each sentiment class.
    # Reindex fixes the order and shows 0 for a class with no documents.
    comparison = pd.DataFrame({
    "VADER": df_accc["sentiment_label"].value_counts(),
    "RoBERTa": df_accc["roberta_label"].value_counts(),
    }).reindex(["negative", "neutral", "positive"], fill_value=0)
    with st.expander("Show counts as a table"):
        st.dataframe(comparison)
    # Unpivot the table from wide to long (one row per model and sentiment), so Altair can use "model" as a column
    comparison_long = comparison.rename_axis("sentiment").reset_index().melt(
    id_vars="sentiment", var_name="model", value_name="documents"
    )
    # Grouped bars: one pair (VADER, RoBERTa) for each sentiment class.
    # Counts, not percentages: 75 documents are too few for shares to be meaningful.
    # The sort lists keep the same order as the table and the other chart.
    # The colour scale pins VADER to light blue and RoBERTa to dark blue.
    chart_cmp = alt.Chart(comparison_long).mark_bar().encode(
    x=alt.X("sentiment", sort=["negative", "neutral", "positive"], axis=alt.Axis(labelAngle=0, title="Sentiment")),
    xOffset=alt.XOffset("model", sort=["VADER", "RoBERTa"]),
    y=alt.Y("documents", axis=alt.Axis(title="Number of documents")),
    color=alt.Color("model", title="Model", sort=["VADER", "RoBERTa"], scale=alt.Scale(domain=["VADER", "RoBERTa"], range=["#83c9ff", "#0068c9"]), legend=alt.Legend(orient="top", direction="horizontal")),
    tooltip=["model", "sentiment", "documents"],
    ).properties(height=420)
    st.altair_chart(chart_cmp, use_container_width=True)
    # Reading note: the explanation is a hypothesis, not verified on labelled data
    st.caption("VADER scores text with a word list and fixed rules, so it tends to miss sarcasm and irony and can read sarcastic or ironic comments as positive. RoBERTa takes the context of the whole text into account and is better equipped to handle this. The gap between the two on these 75 documents is consistent with that explanation. This explanation has not been checked against hand-labelled data.")
with col4:
    st.write("### ACCC mentions")
    # For each brand, count in how many of the 75 ACCC documents it is named (through the spaCy entities).
    accc_brand_counts = pd.Series({
    brand: df_accc["entities"].apply(mentions_brand, args=(brand,)).sum()
    for brand in brands
    })
    # Turn the counts into a table with two named columns (Brand, Number of documents), so Altair can use them
    accc_table = accc_brand_counts.rename_axis("Brand").reset_index(name="Number of documents")
    # Horizontal bars, sorted from the brand with most documents to the one with fewest.
    # Counts, not percentages: 75 documents are too few for shares to be meaningful.
    # tickMinStep=1 keeps the axis on whole numbers; the purple is not used in the other charts,
    # so it cannot be read as a sentiment or a model.
    chart_accc = alt.Chart(accc_table).mark_bar(color="#b07aa1").encode(
        x=alt.X("Number of documents", axis=alt.Axis(title="Number of documents", tickMinStep=1)),
    y=alt.Y("Brand", sort="-x", axis=alt.Axis(title=None)),
    tooltip=["Brand", "Number of documents"],
    ).properties(height=420)
    st.altair_chart(chart_accc, use_container_width=True)
    # Reading notes: what a missing bar means, and the status of the case.
    st.caption("Number of documents, among the 75 mentioning the ACCC case, in which each brand is named. No bar means the brand is not named in any of them.")
    st.caption("The case concerns only Banana Boat and Hawaiian Tropic. The ACCC's claims are allegations, and this dashboard does not report the outcome of the case.")