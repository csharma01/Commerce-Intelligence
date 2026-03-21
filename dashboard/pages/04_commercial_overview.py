import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Commercial Overview: Commerce Intelligence", layout="wide")

@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

st.title("Commercial Overview")
st.subheader("Commerce Intelligence: UK Giftware Wholesale Intelligence Platform")

# Load required datasets
df_elasticity = load_data("outputs/elasticity_results.csv")
df_risk = load_data("outputs/stock_risk_report.csv")
df_gap = load_data("outputs/competitor_gap.csv")

if not df_risk.empty:
    df_risk = df_risk.round({
        'Weekly_Forecast': 0,
        'Current_Inventory': 0,
        'Suggested_Reorder_Qty': 0,
        'Days_Until_Stockout': 1
    })

# --- SECTION 1: EXECUTIVE SUMMARY ---
st.markdown("### Executive Summary")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total SKUs analysed", "2,480")

with col2:
    if not df_elasticity.empty:
        total_revenue = df_elasticity['total_revenue'].sum()
        st.metric("Total Revenue Analysed", f"£{total_revenue/1_000_000:.1f}M")
    else:
        st.metric("Total Revenue Analysed", "N/A")

with col3:
    st.metric("Models beating baseline", "80%", help="50 LightGBM models evaluated against naive persistence")

with col4:
    st.metric("Competitor products benchmarked", "288", help="Data source: Cox & Cox")

with col5:
    if not df_gap.empty:
        high_gap_count = len(df_gap[df_gap['Gap %'] > 500])
        st.metric("Categories with >500% retail gap", high_gap_count)
    else:
        st.metric("Categories with >500% retail gap", "N/A")

st.write("---")

# --- SECTION 2: METHODOLOGY & ASSUMPTIONS ---
st.markdown("### Methodology, Assumptions & Key Decisions")
st.write(
    "This platform was built on the UCI Online Retail II dataset "
    "(776,842 transactions, Dec 2009 - Nov 2011). Several important "
    "analytical decisions were made during development. These are "
    "documented here for transparency."
)

with st.expander("Why the forecasting model shows poor mean accuracy despite strong median performance"):
    st.write(
        "The dataset contains a mix of retail and wholesale buyers. A single bulk order of thousands of units "
        "creates an extreme outlier that inflates the mean absolute error while the median performance, "
        "which reflects typical weekly demand, remains strong. The median improvement over naive baseline is 17.1%. "
        "For a wholesale business, bulk orders should be treated as manual exceptions requiring human oversight, "
        "not model predictions. The model is evaluated on median performance."
    )

with st.expander("How zero-sales weeks were handled in demand forecasting"):
    st.write(
        "Retail transaction data does not record weeks where nothing was sold: those weeks simply have no rows. "
        "Standard lag features built on this data would reference weeks from months ago as if they were last week. "
        "A complete week grid was constructed for all 50 SKUs, filling missing weeks with zero demand. "
        "This allowed lag and rolling features to correctly reflect true historical demand patterns. "
        "Switching to a Tweedie distribution objective (designed for zero-inflated data with long tails) "
        "further improved median model performance from 9.7% to 17.1% over the naive baseline."
    )

with st.expander("Why price elasticity was separated by customer segment"):
    st.write(
        "The dataset mixes retail customers (buying 1-5 units) and wholesale buyers (buying 50-5,000 units). "
        "Aggregating both segments into a single regression produces artificially extreme elasticity coefficients: "
        "the model interprets a bulk order at a lower price tier as evidence of extreme price sensitivity. "
        "Separating WHOLESALE and RETAIL transactions produces distinct demand curves for each segment. "
        "RETAIL elasticity results (135 HIGH priority SKUs) are more reliable for consumer pricing strategy. "
        "WHOLESALE elasticity is directionally useful but limited by the dataset's lack of genuine price change events."
    )

with st.expander("The purpose of competitor price benchmarking"):
    st.write(
        "Price elasticity measures how customers respond to price changes within your own historical data. "
        "Competitor pricing answers a different question: what is the market willing to pay? "
        "Cox & Cox, a premium UK home and giftware retailer, was selected as the benchmark because its "
        "product categories directly overlap with this dataset. Prices were scraped live using BeautifulSoup. "
        "Products over £150 were excluded from the Storage category to remove non-comparable furniture items. "
        "The resulting gap analysis shows that in all 12 categories, retail prices are 3-27x higher than wholesale "
        "supply prices, confirming that wholesale customers operate with substantial retail margin. "
        "This creates pricing headroom for measured wholesale price increases."
    )

with st.expander("Data limitations and what a production version would require"):
    st.write(
        "Several limitations affect the reliability of these results. "
        "First, the dataset ends in November 2011: seasonal patterns may have shifted in the years since. "
        "Second, genuine price change events are rare in this dataset; most price variation reflects different "
        "customer tiers buying at different prices on the same day, not deliberate pricing decisions. "
        "A production elasticity system would require a dedicated pricing experiment log. "
        "Third, the demand forecasting models were trained on 80 weeks of data per SKU: more history would improve accuracy. "
        "Fourth, competitor prices reflect a single scrape date and should be refreshed regularly to remain actionable."
    )

st.write("---")

# --- SECTION 3: CATEGORY DEEP DIVE ---
st.markdown("### Category Deep Dive")
if not df_elasticity.empty:
    all_categories = sorted(df_elasticity['Category'].unique())
    selected_cat = st.selectbox("Select Category", options=all_categories)
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        cat_elasticity = df_elasticity[df_elasticity['Category'] == selected_cat]['elasticity_coefficient'].mean()
        st.metric("Avg Category Elasticity", f"{cat_elasticity:.2f}")
    
    with col_b:
        if not df_gap.empty:
            cat_gap = df_gap[df_gap['Category'] == selected_cat]
            if not cat_gap.empty:
                st.metric("Cox & Cox Avg Price", f"£{cat_gap['Cox&Cox £'].values[0]:.2f}")
                st.metric("Price Gap", f"{cat_gap['Gap %'].values[0]:.1f}%")
            else:
                st.info("Category gap data unavailable")
        else:
            st.info("Gap analysis data unavailable")
            
    with col_c:
        if not df_risk.empty:
            cat_skus = df_elasticity[df_elasticity['Category'] == selected_cat]['StockCode'].unique()
            cat_risks = df_risk[df_risk['StockCode'].isin(cat_skus)]
            if not cat_risks.empty:
                st.write("Top Risks in Category:")
                st.table(cat_risks[['StockCode', 'Risk_Level', 'Days_Until_Stockout']])
            else:
                st.success("✅ No stock risks for this category.")
        else:
            st.info("Stock risk report missing")
