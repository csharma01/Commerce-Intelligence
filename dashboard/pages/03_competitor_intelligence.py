import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Competitor Intelligence: Commerce Intelligence", layout="wide")

@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

st.title("Competitor Intelligence")

df_gap = load_data("outputs/competitor_gap.csv")
df_prices = load_data("data/external/competitor_prices.csv")

if df_gap.empty:
    st.error("🚨 Missing data: outputs/competitor_gap.csv not found.")
    st.stop()

# --- TOP METRIC ---
high_gap_threshold = 500
high_gap_cats = len(df_gap[df_gap['Gap %'] > high_gap_threshold])
st.metric("High Margin Opportunities", f"{high_gap_cats} of {len(df_gap)} categories show >500% gap vs Cox & Cox")

st.write("---")

# --- MAIN GAP TABLE ---
st.subheader("Price Gap Analysis")
st.write("Benchmark wholesale prices against premium retailer Cox & Cox.")

df_display_gap = df_gap.copy()
for col in df_display_gap.select_dtypes(include='object').columns:
    df_display_gap[col] = df_display_gap[col].astype(str)

st.dataframe(
    df_display_gap.style.background_gradient(subset=['Gap %'], cmap='Greens').format({
        'Our Wholesale £': '£{:.2f}',
        'Implied Retail £': '£{:.2f}',
        'Cox&Cox £': '£{:.2f}',
        'Gap %': '{:.1f}%',
        'Headroom vs Implied': '£{:.2f}'
    }),
    use_container_width=True
)

st.write("---")

# --- BAR CHART ---
st.subheader("Gap % by Category")
fig_gap = px.bar(
    df_gap,
    x='Gap %',
    y='Category',
    orientation='h',
    text='Gap %',
    color='Gap %',
    color_continuous_scale='Greens',
    labels={'Gap %': 'Price Gap (%)'},
    title="Wholesale-to-Retail Price Gap by Category"
)
fig_gap.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig_gap.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
st.plotly_chart(fig_gap, use_container_width=True)

st.write("---")

# --- SAMPLE PRODUCTS ---
st.subheader("Sample Competitor Products")
if not df_prices.empty:
    all_categories = sorted(df_prices['our_category'].unique())
    selected_category = st.selectbox("Select a category to view actual products:", options=all_categories)
    
    cat_samples = df_prices[df_prices['our_category'] == selected_category].head(5)
    if not cat_samples.empty:
        df_display_samples = cat_samples[['product_name', 'price_gbp', 'source', 'product_url']].copy()
        for col in df_display_samples.select_dtypes(include='object').columns:
            df_display_samples[col] = df_display_samples[col].astype(str)
            
        st.dataframe(
            df_display_samples.style.format({'price_gbp': '£{:.2f}'}),
            use_container_width=True
        )
    else:
        st.info("No products found for this category.")
else:
    st.info("Competitor products data (data/external/competitor_prices.csv) unavailable.")

# --- INSIGHT TEXT ---
st.write("---")
st.markdown("""
### Strategic Insight
The **Gap %** represents the difference between our wholesale cost and Cox & Cox retail price.
- **Why it matters:** A large gap indicates significant pricing headroom.
- **Implied Retail GBP:** Represents our wholesale price with a standard 2.5x markup.
- **Cox & Cox Comparison:** Our wholesale prices are extremely competitive. Most categories offer headroom even after applying standard retail markups.
""")
