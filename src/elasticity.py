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
    
    weekly_df = df_filtered.groupby(['StockCode', 'Description', 'Category', 'Year', 'Week']).agg({
        'Quantity': 'sum',
        'Price': 'mean',
        'Revenue': 'sum'
    }).reset_index()
    
    # ISSUE 1 - Price floor filter
    initial_count = len(weekly_df)
    weekly_df = weekly_df[weekly_df['Price'] >= 0.50]
    removed_count = initial_count - len(weekly_df)
    logger.info(f"Price floor filter (< £0.50) removed {removed_count} weekly records.")
    
    # Filter for SKUs with enough observations
    sku_counts = weekly_df.groupby('StockCode').size()
    skus_to_analyze = sku_counts[sku_counts >= min_observations].index
    
    results = []
    
    for sku in skus_to_analyze:
        sku_data = weekly_df[weekly_df['StockCode'] == sku].copy()
        
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
            logger.warning(f"Failed to calculate elasticity for SKU {sku}: {e}")
            
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
    
    def get_type(row):
        coeff = row['elasticity_coefficient']
        if -1.0 <= coeff <= 0:
            return 'INELASTIC'
        elif coeff < -1.0:
            return 'ELASTIC'
        else:
            return 'UNUSUAL'
            
    def get_priority(row):
        if row['elasticity_type'] == 'INELASTIC' and row['is_significant']:
            return 'HIGH'
        elif row['elasticity_type'] == 'ELASTIC' and row['is_significant']:
            return 'MEDIUM'
        else:
            return 'LOW'
            
    def get_opportunity(row):
        if row['commercial_priority'] == 'HIGH':
            # Formula: mean_weekly_quantity * mean_price * 52 * 0.10 * (1 + elasticity_coefficient)
            return row['mean_weekly_quantity'] * row['mean_price'] * 52 * 0.10 * (1 + row['elasticity_coefficient'])
        return 0.0

    elasticity_df['elasticity_type'] = elasticity_df.apply(get_type, axis=1)
    elasticity_df['commercial_priority'] = elasticity_df.apply(get_priority, axis=1)
    elasticity_df['estimated_annual_opportunity'] = elasticity_df.apply(get_opportunity, axis=1)
    
    return elasticity_df

def save_results(elasticity_df, output_path='outputs/elasticity_results.csv'):
    """
    Save full results and high priority SKUs to CSV.
    
    Args:
        elasticity_df (pd.DataFrame): Classified elasticity results.
        output_path (str): Path for the full results CSV.
    """
    logger.info(f"Saving results to {output_path}...")
    
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    elasticity_df.to_csv(output_path, index=False)
    
    # Save high priority SKUs
    high_priority = elasticity_df[elasticity_df['commercial_priority'] == 'HIGH'].sort_values(
        'estimated_annual_opportunity', ascending=False
    )
    high_priority_path = 'outputs/high_priority_skus.csv'
    high_priority.to_csv(high_priority_path, index=False)
    
    # Log summary
    total_skus = len(elasticity_df)
    sig_count = elasticity_df['is_significant'].sum()
    high_count = len(high_priority)
    total_opp = high_priority['estimated_annual_opportunity'].sum()
    
    logger.info(f"Summary: Total SKUs analyzed: {total_skus}")
    logger.info(f"Summary: Significant elasticities: {sig_count}")
    logger.info(f"Summary: HIGH priority SKUs: {high_count}")
    logger.info(f"Summary: Total Annual Opportunity: £{total_opp:,.2f}")

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
    
    # Calculate percentage inelastic
    def pct_inelastic(x):
        return (x == 'INELASTIC').mean() * 100
        
    category_stats['pct_inelastic'] = elasticity_df.groupby('Category')['elasticity_type'].apply(pct_inelastic)
    
    # Calculate high priority count
    category_stats['high_priority_count'] = elasticity_df[elasticity_df['commercial_priority'] == 'HIGH'].groupby('Category').size()
    category_stats['high_priority_count'] = category_stats['high_priority_count'].fillna(0).astype(int)
    
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
    
    # Print top 10 revenue SKUs
    top_revenue = elasticity_df.sort_values('total_revenue', ascending=False).head(10)
    
    print("\n" + "="*80)
    print(f"{'TOP 10 REVENUE SKUs WITH ELASTICITY':^80}")
    print("="*80)
    print(f"{'Description':<35} | {'Category':<15} | {'Elasticity':>10} | {'Revenue (£)':>12}")
    print("-"*80)
    
    for _, row in top_revenue.iterrows():
        print(f"{row['Description'][:35]:<35} | {row['Category']:<15} | {row['elasticity_coefficient']:>10.2f} | {row['total_revenue']:>12,.2f}")
    print("="*80 + "\n")

    # Print top 15 HIGH priority SKUs
    high_priority = elasticity_df[elasticity_df['commercial_priority'] == 'HIGH'].sort_values(
        'estimated_annual_opportunity', ascending=False
    ).head(15)
    
    print("\n" + "="*80)
    print(f"{'TOP 15 HIGH PRIORITY SKUs':^80}")
    print("="*80)
    print(f"{'Description':<35} | {'Category':<15} | {'Elasticity':>10} | {'Price':>8} | {'Opp (£)':>10}")
    print("-"*80)
    
    for _, row in high_priority.iterrows():
        print(f"{row['Description'][:35]:<35} | {row['Category']:<15} | {row['elasticity_coefficient']:>10.2f} | {row['mean_price']:>8.2f} | {row['estimated_annual_opportunity']:>10.2f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
