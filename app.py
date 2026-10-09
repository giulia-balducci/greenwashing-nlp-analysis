import pandas as pd
import streamlit as st
import ast
import altair as alt

# Create a brand list
brands = ['Banana Boat', 'Hawaiian Tropic', 'Neutrogena', 'Coppertone',
          'Stream2Sea', 'Raw Elements', 'Thinksport', 'Badger Balm', 'Sun Bum']

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
    return pd.read_csv("data/processed/df_sunscreen_dashboard.csv")

df = load_data()
n_synthetic = df["is_synthetic"].sum()
n_real = len(df) - n_synthetic

# Short introduction for readers who don't know the case
st.markdown("""
A 2025 court case (ACCC v Edgewell Personal Care) centred on Banana Boat and Hawaiian Tropic sunscreens, which were labelled reef-friendly while allegedly containing oxybenzone and other chemical UV filters linked to coral bleaching.

We analysed online posts about these claims and measured the sentiment of the conversation around each brand: positive, neutral or negative.

Select a brand to see how it is discussed; the table gives the overview for all brands.
""")

# Display a warning about the dataset composition
st.warning(f"The dataset contains {n_real} real documents and {n_synthetic} synthetic (AI-generated) documents. The results are demonstrative and do not represent real public opinion.")

# Create columns for brand selection and summary table
col1, col2 = st.columns([3,2])
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
    chart = alt.Chart(chart_df).mark_bar().encode(
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
    )
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
    st.dataframe(
    summary,
    hide_index=True,
    column_config={
        "brand": st.column_config.TextColumn("Brand"),
        "n_docs": st.column_config.NumberColumn("Number of documents"),
        "mean_score": st.column_config.NumberColumn("Mean sentiment score (-1 to 1)", format="%.2f"),
    },
    )
    st.caption("Mean sentiment score ranges from -1 (all documents negative) to +1 (all documents positive). Each document is scored as a whole, so a reef-safe brand may score lower than how it is actually discussed, for instance if mentioned as a trusted alternative in a document with overall negative sentiment.")
    