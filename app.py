import pandas as pd
import streamlit as st
import ast
import altair as alt

st.title("Greenwashing NLP Analysis - Brand Sentiment and Mention Frequency Dashboard")

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

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/df_sunscreen_dashboard.csv")

df = load_data()
n_synthetic = df["is_synthetic"].sum()
n_real = len(df) - n_synthetic

st.warning(f"The dataset contains {n_real} real documents and {n_synthetic} synthetic (AI-generated) documents. The results are demonstrative and do not represent real public opinion.")

brands = ['Banana Boat', 'Hawaiian Tropic', 'Neutrogena', 'Coppertone',
          'Stream2Sea', 'Raw Elements', 'Thinksport', 'Badger Balm', 'Sun Bum']

selected_brand = st.selectbox("Select a brand:", brands)
mask = df["entities"].apply(mentions_brand, args=(selected_brand,))
brand_df = df[mask]

shares = brand_df["roberta_label"].value_counts(normalize=True).reindex(["negative", "neutral", "positive"], fill_value=0)
st.write(f"Documents mentioning {selected_brand}: {len(brand_df)}")
chart_df = shares.reset_index()
chart_df.columns = ["sentiment", "share"]
chart_df["color"] = chart_df["sentiment"].map({
    "negative": "#ff7f50",
    "neutral": "#d3d3d3",
    "positive": "#008080",
})
st.bar_chart(chart_df, x="sentiment", y="share", color="color")
