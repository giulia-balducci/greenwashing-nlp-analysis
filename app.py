import pandas as pd
import streamlit as st
import ast
import altair as alt

# Create a brand list
brands = ['Banana Boat', 'Hawaiian Tropic', 'Neutrogena', 'Coppertone',
          'Stream2Sea', 'Raw Elements', 'Thinksport', 'Badger Balm', 'Sun Bum']

# Set title and description
st.title("Reef-safe Sunscreen & Greenwashing: the conversation, brand by brand")
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

# Display a warning about the dataset composition
st.warning(f"The dataset contains {n_real} real documents and {n_synthetic} synthetic (AI-generated) documents. The results are demonstrative and do not represent real public opinion.")

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
    x=alt.X("sentiment", sort=["negative", "neutral", "positive"]),
    y="share",
    color=alt.Color(
        "sentiment",
        scale=alt.Scale(domain=["negative", "neutral", "positive"], range=["#ff7f50", "#d3d3d3", "#008080"]),
        legend=None,
    ),
)
st.altair_chart(chart, use_container_width=True)

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
st.dataframe(summary, hide_index=True, column_config={"mean_score": st.column_config.NumberColumn(format="%.2f")})
