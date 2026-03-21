import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Demand Forecasting: Commerce Intelligence", layout="wide")

@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

st.title("Demand Forecasting")

df_risk = load_data("outputs/stock_risk_report.csv")
df_performance = load_data("outputs/model_performance.csv")

if df_risk.empty:
    st.error("🚨 Missing data: outputs/stock_risk_report.csv not found.")
    st.stop()

# Round columns for display
df_risk = df_risk.round({
    'Weekly_Forecast': 0,
    'Current_Inventory': 0,
    'Suggested_Reorder_Qty': 0,
    'Days_Until_Stockout': 1
})

# --- ALERT ROW ---
critical_count = len(df_risk[df_risk['Risk_Level'] == 'CRITICAL'])
high_count = len(df_risk[df_risk['Risk_Level'] == 'HIGH'])

if critical_count > 0:
    st.error(f"🚨 CRITICAL ALERT: {critical_count} SKUs predicted to stock out in less than 7 days!")
if high_count > 0:
    st.warning(f"⚠️ HIGH RISK: {high_count} SKUs predicted to stock out in 7-14 days.")

st.write("---")

# --- RISK REPORT TABLE ---
st.subheader("Stock Risk Report")
st.write("SKUs sorted by estimated days until stockout.")

# Color coding for Risk_Level
def color_risk(val):
    color = 'white'
    if val == 'CRITICAL': color = '#ff4b4b'
    elif val == 'HIGH': color = '#ffa500'
    elif val == 'MEDIUM': color = '#f9d71c'
    elif val == 'LOW': color = '#00cc96'
    return f'background-color: {color}; color: black; font-weight: bold'

st.dataframe(
    df_risk.style.map(color_risk, subset=['Risk_Level']).format({
        'Weekly_Forecast': '{:.0f}',
        'Current_Inventory': '{:.0f}',
        'Suggested_Reorder_Qty': '{:.0f}',
        'Days_Until_Stockout': '{:.1f}'
    }),
    use_container_width=True
)

st.write("---")

# --- CHARTS ROW ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Performance")
    if not df_performance.empty:
        fig_perf = px.histogram(
            df_performance, 
            x='Improvement_Over_Naive',
            nbins=20,
            labels={'Improvement_Over_Naive': 'Improvement over Naive Baseline'},
            color_discrete_sequence=['#00cc96'],
            title="Distribution of Model Improvements (%)"
        )
        fig_perf.update_layout(bargap=0.1)
        st.plotly_chart(fig_perf, use_container_width=True)
    else:
        st.info("Performance data unavailable.")

with col2:
    st.subheader("Risk Level Distribution")
    risk_counts = df_risk['Risk_Level'].value_counts().reset_index()
    risk_counts.columns = ['Risk_Level', 'Count']
    fig_pie = px.pie(
        risk_counts, 
        values='Count', 
        names='Risk_Level',
        color='Risk_Level',
        color_discrete_map={
            'CRITICAL': '#ff4b4b',
            'HIGH': '#ffa500',
            'MEDIUM': '#f9d71c',
            'LOW': '#00cc96'
        },
        title="Stock Risk Segments"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.write("---")

# --- REORDER RECOMMENDATIONS ---
st.subheader("Suggested Reorder Quantities")
urgent_df = df_risk[df_risk['Risk_Level'].isin(['CRITICAL', 'HIGH'])]
if not urgent_df.empty:
    reorder_display = urgent_df[['StockCode', 'Current_Inventory', 'Weekly_Forecast', 'Suggested_Reorder_Qty']].copy()
    reorder_display = reorder_display.rename(columns={
        'Current_Inventory': 'Current Stock',
        'Weekly_Forecast': 'Weekly Forecast',
        'Suggested_Reorder_Qty': 'Reorder Qty'
    })
    st.table(
        reorder_display.sort_values(by='Weekly Forecast', ascending=False)
    )
else:
    st.success("✅ No urgent reorders needed based on current forecasts.")
