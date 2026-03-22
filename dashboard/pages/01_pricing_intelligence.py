import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Pricing Intelligence: Commerce Intelligence", layout="wide")

@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

st.title("Pricing Intelligence")

df_results = load_data("outputs/elasticity_results.csv")
df_high_priority = load_data("outputs/high_priority_skus.csv")

if df_results.empty:
    st.error("🚨 Missing data: outputs/elasticity_results.csv not found.")
    st.stop()

# --- TOP KPI ROW ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total SKUs Analysed", f"{len(df_results['StockCode'].unique()):,}")
with col2:
    st.metric("HIGH Priority Count", f"{len(df_high_priority) if not df_high_priority.empty else 0}")
with col3:
    if not df_high_priority.empty:
        total_opp = df_high_priority['estimated_annual_opportunity'].sum()
        st.metric("Total Opportunity GBP", f"£{total_opp:,.0f}")
    else:
        st.metric("Total Opportunity GBP", "£0")

st.write("---")

# --- FILTERS ---
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    segments = df_results['customer_type'].unique()
    segment_filter = st.selectbox("Customer Segment", options=segments)
with col_f2:
    all_categories = sorted(df_results['Category'].unique().tolist())
    category_filter = st.multiselect(
        'Category Filter',
        options=all_categories,
        default=all_categories
    )
with col_f3:
    significant_only = st.checkbox("Show statistically significant only", value=True)

# Apply filters
filtered_df = df_results[
    (df_results['customer_type'] == segment_filter) &
    (df_results['Category'].isin(category_filter))
]

if significant_only:
    filtered_df = filtered_df[filtered_df['is_significant'] == True]

# --- MAIN TABLE: HIGH PRIORITY ---
st.subheader("High Priority Opportunities")
hp_filtered = df_high_priority[
    (df_high_priority['customer_type'] == segment_filter) &
    (df_high_priority['Category'].isin(category_filter))
] if not df_high_priority.empty else pd.DataFrame()

if not hp_filtered.empty:
    display_cols = [
        'Description', 'Category', 'customer_type', 
        'elasticity_coefficient', 'mean_price', 'estimated_annual_opportunity'
    ]
    df_display = hp_filtered[display_cols].copy()
    for col in df_display.select_dtypes(include='object').columns:
        df_display[col] = df_display[col].astype(str)
        
    st.dataframe(
        df_display.style.background_gradient(
            subset=['estimated_annual_opportunity'], cmap='YlOrRd'
        ).format({
            'elasticity_coefficient': '{:.2f}',
            'mean_price': '£{:.2f}',
            'estimated_annual_opportunity': '£{:.2f}'
        }),
        use_container_width=True
    )
    
    csv = hp_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Filtered Results (CSV)",
        csv,
        "high_priority_skus_filtered.csv",
        "text/csv",
        key='download-csv'
    )
else:
    st.info("No HIGH priority SKUs found for the current filters.")

st.write("---")

# --- SCATTER PLOT ---
st.subheader("Elasticity vs Revenue")
if len(filtered_df) == 0:
    st.info("No data matches the current filters.")
else:
    fig = px.scatter(
        filtered_df,
        x='elasticity_coefficient',
        y='total_revenue',
        color='commercial_priority',
        hover_name='Description',
        labels={
            'elasticity_coefficient': 'Elasticity Coefficient',
            'total_revenue': 'Total Revenue (GBP)',
            'commercial_priority': 'Priority Level'
        },
        color_discrete_map={'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'blue'},
        title=f"Elasticity vs Revenue for {segment_filter}"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Guide:**
- **Elastic (<-1.0):** Volume is sensitive to price. Be cautious with increases.
- **Inelastic (>-1.0):** Volume is less sensitive to price. Potential for margin improvement.
""")
