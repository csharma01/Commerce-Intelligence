import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Admin items to exclude
ADMIN_CODES = ['M', 'POST', 'DOT', 'ADJUST', 'ADJUST2', 'BANK CHARGES', 
               'PADS', 'gift_0001_80343', 'B', 'CRUK']

def calculate_elasticity(df, min_observations=30):
    """
    Calculate price elasticity of demand for each SKU using log-log OLS regression.
    
    Args:
        df (pd.DataFrame): Cleaned transaction data.
        min_observations (int): Minimum weekly records required per SKU.
        
    Returns:
        pd.DataFrame: Results including elasticity coefficients and significance.
    """
    logger.info("Starting elasticity calculation...")
    
    # Filter to UK only
    initial_rows = len(df)
    df_filtered = df[df['Country'] == 'United Kingdom'].copy()
    retained_rows = len(df_filtered)
    removed_rows = initial_rows - retained_rows
    logger.info(f"Filtered to UK only: {retained_rows} rows retained, {removed_rows} rows removed")
    
    # Exclude partial months and admin items
    df_filtered = df_filtered[~df_filtered['is_partial_month'] & ~df_filtered['StockCode'].isin(ADMIN_CODES)].copy()
    
    # Weekly aggregation
    # Ensure Quantity and Price are positive for log transformation
    df_filtered = df_filtered[(df_filtered['Quantity'] > 0) & (df_filtered['Price'] > 0)]
    
    weekly_df = df_filtered.groupby(['StockCode', 'Description', 'Category', 'customer_type', 'Year', 'Week']).agg({
        'Quantity': 'sum',
        'Revenue': 'sum'
    }).reset_index()
    weekly_df['Price'] = weekly_df['Revenue'] / weekly_df['Quantity']
    
    # ISSUE 1 - Price floor filter
    initial_count = len(weekly_df)
    weekly_df = weekly_df[weekly_df['Price'] >= 0.50]
    removed_count = initial_count - len(weekly_df)
    logger.info(f"Price floor filter (< £0.50) removed {removed_count} weekly records.")
    
    # Filter for SKUs/Segments with enough observations
    segment_counts = weekly_df.groupby(['StockCode', 'customer_type']).size()
    segments_to_analyze = segment_counts[segment_counts >= min_observations].index
    
    results = []
    
    for (sku, segment) in segments_to_analyze:
        sku_data = weekly_df[
            (weekly_df['StockCode'] == sku) & 
            (weekly_df['customer_type'] == segment)
        ].copy()
        
        # Log transformations
        sku_data['log_q'] = np.log(sku_data['Quantity'])
        sku_data['log_p'] = np.log(sku_data['Price'])
        
        # Skip if price variation is zero (can't regress)
        if sku_data['log_p'].nunique() <= 1:
            continue
            
        try:
            X = sm.add_constant(sku_data['log_p'])
            model = sm.OLS(sku_data['log_q'], X).fit()
            
            # Extract metrics
            elasticity = model.params['log_p']
            p_value = model.pvalues['log_p']
            r_squared = model.rsquared
            
            results.append({
                'StockCode': sku,
                'customer_type': segment,
                'Description': sku_data['Description'].iloc[0],
                'Category': sku_data['Category'].iloc[0],
                'elasticity_coefficient': elasticity,
                'r_squared': r_squared,
                'p_value': p_value,
                'observation_count': len(sku_data),
                'mean_price': sku_data['Price'].mean(),
                'mean_weekly_quantity': sku_data['Quantity'].mean(),
                'total_revenue': sku_data['Revenue'].sum(),
                'is_significant': p_value < 0.05
            })
        except Exception as e:
            logger.warning(f"Failed to calculate elasticity for SKU {sku} ({segment}): {e}")
            
    return pd.DataFrame(results)

def classify_elasticity(elasticity_df):
    """
    Classify SKUs by elasticity type and commercial priority.
    
    Args:
        elasticity_df (pd.DataFrame): Elasticity calculation results.
        
    Returns:
        pd.DataFrame: DataFrame with added classification columns.
    """
    logger.info("Classifying elasticity types and priorities...")
    
    # Elasticity Type
    conditions_type = [
        (elasticity_df['elasticity_coefficient'] >= -1.0) & (elasticity_df['elasticity_coefficient'] <= 0),
        (elasticity_df['elasticity_coefficient'] < -1.0)
    ]
    choices_type = ['INELASTIC', 'ELASTIC']
    elasticity_df['elasticity_type'] = np.select(conditions_type, choices_type, default='UNUSUAL')

    # Commercial Priority
    conditions_priority = [
        (elasticity_df['elasticity_type'] == 'INELASTIC') & elasticity_df['is_significant'],
        (elasticity_df['elasticity_type'] == 'ELASTIC') & elasticity_df['is_significant']
    ]
    choices_priority = ['HIGH', 'MEDIUM']
    elasticity_df['commercial_priority'] = np.select(conditions_priority, choices_priority, default='LOW')

    # Opportunity calculation
    annual_revenue = elasticity_df['mean_weekly_quantity'] * elasticity_df['mean_price'] * 52
    opportunity = annual_revenue * (0.10 + 0.11 * elasticity_df['elasticity_coefficient'])
    elasticity_df['estimated_annual_opportunity'] = np.where(
        elasticity_df['commercial_priority'] == 'HIGH',
        np.maximum(opportunity, 0.0),
        0.0
    )
    
    return elasticity_df

def save_results(elasticity_df, output_path='outputs/elasticity_results.csv'):
    """
    Save full results and segment-specific results to CSV.
    
    Args:
        elasticity_df (pd.DataFrame): Classified elasticity results.
        output_path (str): Path for the full results CSV.
    """
    logger.info(f"Saving results to {output_path}...")
    
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    elasticity_df.to_csv(output_path, index=False)
    
    # Save segment-specific results
    wholesale = elasticity_df[elasticity_df['customer_type'] == 'WHOLESALE']
    retail = elasticity_df[elasticity_df['customer_type'] == 'RETAIL']
    
    wholesale.to_csv('outputs/elasticity_wholesale.csv', index=False)
    retail.to_csv('outputs/elasticity_retail.csv', index=False)
    
    # Save high priority SKUs
    high_priority = elasticity_df[elasticity_df['commercial_priority'] == 'HIGH'].sort_values(
        'estimated_annual_opportunity', ascending=False
    )
    high_priority_path = 'outputs/high_priority_skus.csv'
    high_priority.to_csv(high_priority_path, index=False)
    
    # Log summaries per segment
    for segment in ['WHOLESALE', 'RETAIL']:
        seg_df = elasticity_df[elasticity_df['customer_type'] == segment]
        seg_high = seg_df[seg_df['commercial_priority'] == 'HIGH']
        
        logger.info(f"Summary ({segment}): Total SKUs analyzed: {len(seg_df)}")
        logger.info(f"Summary ({segment}): HIGH priority SKUs: {len(seg_high)}")
        logger.info(f"Summary ({segment}): Total Annual Opportunity: £{seg_high['estimated_annual_opportunity'].sum():,.2f}")

def run_category_analysis(elasticity_df):
    """
    Perform category-level analysis of elasticity.
    
    Args:
        elasticity_df (pd.DataFrame): Classified elasticity results.
        
    Returns:
        pd.DataFrame: Category summary results.
    """
    logger.info("Running category analysis...")
    
    category_stats = elasticity_df.groupby('Category').agg({
        'elasticity_coefficient': 'mean',
        'total_revenue': 'sum',
        'StockCode': 'count'
    }).rename(columns={
        'elasticity_coefficient': 'avg_elasticity',
        'StockCode': 'sku_count'
    })
    
    # Vectorized percentage inelastic
    inelastic_mask = elasticity_df['elasticity_type'] == 'INELASTIC'
    category_stats['pct_inelastic'] = inelastic_mask.groupby(
        elasticity_df['Category']).mean() * 100

    # Vectorized high priority count
    high_priority_mask = elasticity_df['commercial_priority'] == 'HIGH'
    category_stats['high_priority_count'] = high_priority_mask.groupby(
        elasticity_df['Category']).sum().astype(int)
    
    category_stats = category_stats.reset_index()
    
    category_path = 'outputs/category_elasticity.csv'
    category_stats.to_csv(category_path, index=False)
    
    return category_stats

def main():
    # Load data
    data_path = 'data/processed/retail_clean.csv'
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    # Run pipeline
    elasticity_df = calculate_elasticity(df)
    elasticity_df = classify_elasticity(elasticity_df)
    save_results(elasticity_df)
    run_category_analysis(elasticity_df)
    
    # Print tables
    
    # 1. TOP 10 WHOLESALE SKUs by revenue
    wholesale_top = elasticity_df[elasticity_df['customer_type'] == 'WHOLESALE'].sort_values(
        'total_revenue', ascending=False).head(10)
    
    print("\n" + "="*95)
    print(f"{'TOP 10 WHOLESALE SKUs BY REVENUE WITH ELASTICITY':^95}")
    print("="*95)
    print(f"{'Description':<35} | {'Category':<15} | {'Elasticity':>10} | {'Revenue (£)':>12}")
    print("-"*95)
    for _, row in wholesale_top.iterrows():
        print(f"{row['Description'][:35]:<35} | {row['Category']:<15} | {row['elasticity_coefficient']:>10.2f} | {row['total_revenue']:>12,.2f}")
    print("="*95 + "\n")

    # 2. TOP 10 RETAIL SKUs by revenue
    retail_top = elasticity_df[elasticity_df['customer_type'] == 'RETAIL'].sort_values(
        'total_revenue', ascending=False).head(10)
    
    print("\n" + "="*95)
    print(f"{'TOP 10 RETAIL SKUs BY REVENUE WITH ELASTICITY':^95}")
    print("="*95)
    print(f"{'Description':<35} | {'Category':<15} | {'Elasticity':>10} | {'Revenue (£)':>12}")
    print("-"*95)
    for _, row in retail_top.iterrows():
        print(f"{row['Description'][:35]:<35} | {row['Category']:<15} | {row['elasticity_coefficient']:>10.2f} | {row['total_revenue']:>12,.2f}")
    print("="*95 + "\n")

    # 3. TOP 15 HIGH PRIORITY across both segments
    high_priority = elasticity_df[elasticity_df['commercial_priority'] == 'HIGH'].sort_values(
        'estimated_annual_opportunity', ascending=False
    ).head(15)
    
    print("\n" + "="*95)
    print(f"{'TOP 15 HIGH PRIORITY SKUs (BOTH SEGMENTS)':^95}")
    print("="*95)
    print(f"{'Description':<35} | {'Segment':<10} | {'Elasticity':>10} | {'Price':>8} | {'Opp (£)':>10}")
    print("-"*95)
    for _, row in high_priority.iterrows():
        print(f"{row['Description'][:35]:<35} | {row['customer_type']:<10} | {row['elasticity_coefficient']:>10.2f} | {row['mean_price']:>8.2f} | {row['estimated_annual_opportunity']:>10.2f}")
    print("="*95 + "\n")

if __name__ == "__main__":
    main()
