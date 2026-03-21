import streamlit as st
import pandas as pd
import os

# Page configuration
st.set_page_config(
    page_title="Commerce Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CACHED DATA LOADING ---
@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# --- SIDEBAR KPIs ---
def show_sidebar_kpis():
    st.sidebar.title("Key Performance Indicators")
    
    # 1. Total SKUs analysed
    df_elasticity = load_data("outputs/elasticity_results.csv")
    if not df_elasticity.empty:
        total_skus = len(df_elasticity['StockCode'].unique())
        st.sidebar.metric("Total SKUs Analysed", f"{total_skus:,}")
    else:
        st.sidebar.info("Total SKUs: Data unavailable")

    # 2. HIGH priority pricing opportunities
    df_high_priority = load_data("outputs/high_priority_skus.csv")
    if not df_high_priority.empty:
        high_priority_count = len(df_high_priority)
        st.sidebar.metric("HIGH Priority Opps", f"{high_priority_count}")
    else:
        st.sidebar.info("HIGH Priority: Data unavailable")

    # 3. Stock-out risks today (CRITICAL + HIGH)
    df_stock_risk = load_data("outputs/stock_risk_report.csv")
    if not df_stock_risk.empty:
        urgent_risks = len(df_stock_risk[df_stock_risk['Risk_Level'].isin(['CRITICAL', 'HIGH'])])
        st.sidebar.metric("🚨 Urgent Stock-out Risks", f"{urgent_risks}", delta_color="inverse")
    else:
        st.sidebar.info("Stock Risks: Data unavailable")

    # 4. Categories with competitor gap >500%
    df_comp_gap = load_data("outputs/competitor_gap.csv")
    if not df_comp_gap.empty:
        high_gap_cats = len(df_comp_gap[df_comp_gap['Gap %'] > 500])
        st.sidebar.metric("High Margin Categories (>500%)", f"{high_gap_cats}")
    else:
        st.sidebar.info("Competitor Gaps: Data unavailable")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Navigation")
    st.sidebar.info(
        """
        Explore the intelligence modules:
        - Pricing: Elasticity and opportunities.
        - Forecasting: Demand and stock risks.
        - Competitor: Cox & Cox price gaps.
        - Overview: Unified action dashboard.
        """
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.success("✅ Elasticity")
        st.success("✅ Forecasting")
    with col2:
        st.success("✅ Scraper")
        st.success("✅ API")

# --- MAIN PAGE CONTENT ---
def main():
    st.title("Commerce Intelligence")
    st.markdown("### Precision Commercial Intelligence for Modern E-commerce")
    
    show_sidebar_kpis()
    
    st.write("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Pricing Intelligence")
        st.write("Optimise margins by identifying inelastic products with high revenue potential.")
        if st.button("Go to Pricing", width='stretch'):
            st.switch_page("pages/01_pricing_intelligence.py")
            
    with col2:
        st.subheader("Demand Forecasting")
        st.write("Prevent stock-outs with Tweedie LightGBM models designed for intermittent demand.")
        if st.button("Go to Forecasting", width='stretch'):
            st.switch_page("pages/02_demand_forecasting.py")
            
    with col3:
        st.subheader("Competitor Intelligence")
        st.write("Benchmark wholesale prices against Cox & Cox retail prices to find pricing headroom.")
        if st.button("Go to Competitor Info", width='stretch'):
            st.switch_page("pages/03_competitor_intelligence.py")

    st.write("---")
    st.info("Select a module from the sidebar or above to begin your analysis.")

if __name__ == "__main__":
    main()
