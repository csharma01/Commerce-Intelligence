"""
Data cleaning pipeline for Commerce Intelligence.
Handles loading, filtering, and feature engineering for the Online Retail dataset.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(filepath: str) -> pd.DataFrame:
    """Load and concatenate all sheets from an Excel file."""
    logger.info(f"Loading data from {filepath}")
    xls = pd.ExcelFile(filepath)
    df_list = []
    for sheet_name in xls.sheet_names:
        df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
        df_list.append(df_sheet)
    df = pd.concat(df_list, ignore_index=True)
    logger.info(f"Loaded {len(df)} rows across {len(xls.sheet_names)} sheets.")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps to the DataFrame."""
    initial_rows = len(df)
    
    # 2. Remove cancelled invoices (InvoiceNo starting with "C")
    # Ensure Invoice column is string for filtering
    invoice_col = 'Invoice' if 'Invoice' in df.columns else 'InvoiceNo'
    df[invoice_col] = df[invoice_col].astype(str)
    df = df[~df[invoice_col].str.startswith('C')]
    logger.info(f"Removed cancelled invoices. Rows remaining: {len(df)}")
    
    # 3. Remove rows where Quantity <= 0
    df = df[df['Quantity'] > 0]
    logger.info(f"Removed rows with Quantity <= 0. Rows remaining: {len(df)}")
    
    # 4. Remove rows where Price <= 0
    df = df[df['Price'] > 0]
    logger.info(f"Removed rows with Price <= 0. Rows remaining: {len(df)}")
    
    # 5. Drop rows missing CustomerID and log how many were dropped
    customer_col = 'Customer ID' if 'Customer ID' in df.columns else 'CustomerID'
    missing_customer = df[customer_col].isna().sum()
    df = df.dropna(subset=[customer_col])
    logger.info(f"Dropped {missing_customer} rows missing {customer_col}. Rows remaining: {len(df)}")
    
    # 6. Remove duplicate rows
    duplicates = df.duplicated().sum()
    df = df.drop_duplicates()
    logger.info(f"Removed {duplicates} duplicate rows. Rows remaining: {len(df)}")
    
    # --- ADDITIONAL CLEANING STEPS (A-C) ---
    
    # STEP A: Remove administrative StockCodes
    admin_codes = ['M', 'POST', 'DOT', 'ADJUST', 'ADJUST2', 'BANK CHARGES', 'PADS', 'gift_0001_80343', 'B', 'CRUK']
    initial_count = len(df)
    df = df[~df['StockCode'].astype(str).isin(admin_codes)]
    logger.info(f"STEP A: Removed {initial_count - len(df)} rows with administrative StockCodes.")

    # STEP B: Remove rows with blank or null StockCode
    initial_count = len(df)
    df = df[df['StockCode'].notna()]
    df = df[df['StockCode'].astype(str).str.strip() != ""]
    logger.info(f"STEP B: Removed {initial_count - len(df)} rows with blank/null StockCodes.")

    # Standardise Description early to support Step C and E
    if 'Description' in df.columns:
        df['Description'] = df['Description'].astype(str).str.strip().str.upper()
        logger.info(f"Standardized Description column.")

    # STEP C: Remove rows where Description contains adjustment keywords
    adj_keywords = ['ADJUSTMENT', 'MANUAL', 'DOTCOM POSTAGE', 'AMAZON FEE', 'BANK CHARGES', 'SAMPLES']
    initial_count = len(df)
    pattern = '|'.join(adj_keywords)
    df = df[~df['Description'].str.contains(pattern, case=False, na=False)]
    logger.info(f"STEP C: Removed {initial_count - len(df)} rows with adjustment keywords in Description.")

    # --- FEATURE ENGINEERING (D-F) ---

    # 8. Create a Revenue column (Required for Step F)
    df['Revenue'] = df['Quantity'] * df['Price']
    logger.info(f"Created Revenue column.")

    # 9. Parse InvoiceDate (Required for Step D)
    if 'InvoiceDate' in df.columns:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['Year'] = df['InvoiceDate'].dt.year
        df['Month'] = df['InvoiceDate'].dt.month
        df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
        df['Week'] = df['InvoiceDate'].dt.isocalendar().week
        logger.info(f"Parsed InvoiceDate and extracted time features.")

    # STEP D: Flag the partial December 2011 month
    df['is_partial_month'] = (df['Year'] == 2011) & (df['Month'] == 12)
    logger.info(f"STEP D: Flagged {df['is_partial_month'].sum()} rows as partial month (Dec 2011).")

    # STEP E: Add proper product categories
    def assign_category(desc):
        if any(k in desc for k in ['CANDLE', 'LIGHT', 'T-LIGHT', 'LANTERN']): return 'Candles & Lighting'
        if 'BAG' in desc: return 'Bags'
        if any(k in desc for k in ['MUG', 'CUP']): return 'Mugs & Drinkware'
        if any(k in desc for k in ['FRAME', 'PICTURE']): return 'Frames & Pictures'
        if any(k in desc for k in ['BUNTING', 'BANNER', 'GARLAND']): return 'Bunting & Decoration'
        if any(k in desc for k in ['CHRISTMAS', 'XMAS']): return 'Christmas'
        if any(k in desc for k in ['DOORMAT', 'DOOR MAT']): return 'Doormats'
        if any(k in desc for k in ['CAKE', 'BAKING']): return 'Kitchen & Baking'
        if any(k in desc for k in ['BOX', 'STORAGE', 'JAR']): return 'Storage'
        if any(k in desc for k in ['SET', 'KIT']): return 'Sets & Kits'
        if any(k in desc for k in ['BIRD', 'BUTTERFLY', 'ANIMAL']): return 'Nature & Animals'
        return 'Home & Giftware'

    df['Category'] = df['Description'].apply(assign_category)
    logger.info(f"STEP E: Assigned product categories.")

    # STEP F: Add customer_type column
    customer_col = 'Customer ID' if 'Customer ID' in df.columns else 'CustomerID'
    invoice_col = 'Invoice' if 'Invoice' in df.columns else 'InvoiceNo'
    
    # Calculate average revenue per order (Invoice) per Customer
    order_revenue = df.groupby([customer_col, invoice_col])['Revenue'].sum().reset_index()
    avg_order_rev = order_revenue.groupby(customer_col)['Revenue'].mean().reset_index()
    avg_order_rev.columns = [customer_col, 'AvgOrderRevenue']
    
    df = df.merge(avg_order_rev, on=customer_col, how='left')
    df['customer_type'] = np.where(df['AvgOrderRevenue'] > 300, 'WHOLESALE', 'RETAIL')
    logger.info(f"STEP F: Added customer_type based on order revenue threshold.")
        
    final_rows = len(df)
    logger.info(f"Cleaning complete. Total rows removed: {initial_rows - final_rows}. Final rows: {final_rows}")
    
    # Print Final Summary
    print("\n" + "="*30)
    print("FINAL CLEANING SUMMARY")
    print("="*30)
    print(f"Total Rows: {len(df):,}")
    print("\nRows per Category:")
    print(df['Category'].value_counts())
    print("\nCustomer Type Distribution (Unique Customers):")
    unique_customers = df.drop_duplicates(subset=[customer_col])
    print(unique_customers['customer_type'].value_counts())
    print("="*30 + "\n")
    
    return df

def main():
    """Main execution function for the cleaning pipeline."""
    input_path = "data/raw/online_retail_II.xlsx"
    output_path = "data/processed/retail_clean.csv"
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        df = load_data(input_path)
        clean_df = clean_data(df)
        clean_df.to_csv(output_path, index=False)
        logger.info(f"Cleaned data successfully saved to {output_path}")
    except Exception as e:
        logger.error(f"Error during data cleaning: {e}")
        raise

if __name__ == "__main__":
    main()
