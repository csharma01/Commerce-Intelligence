import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_forecast_features(df):
    """
    Prepares features for demand forecasting.
    """
    logging.info("Preparing forecast features...")
    
    # Exclude partial months and filter to UK
    df_clean = df[(df['is_partial_month'] == False) & (df['Country'] == 'United Kingdom')].copy()
    
    # Convert InvoiceDate to datetime if it's not
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    
    # Aggregate to weekly level per StockCode
    # Week starts on Monday
    df_clean['Week_Start'] = df_clean['InvoiceDate'] - pd.to_timedelta(df_clean['InvoiceDate'].dt.dayofweek, unit='d')
    df_clean['Week_Start'] = df_clean['Week_Start'].dt.normalize()
    
    weekly_df = df_clean.groupby(['StockCode', 'Week_Start']).agg({
        'Quantity': 'sum',
        'Price': 'mean' # VWAP would be better, but mean is specified for simplicity here or as a proxy
    }).reset_index()
    
    weekly_df.rename(columns={'Quantity': 'weekly_quantity', 'Price': 'avg_price'}, inplace=True)
    
    # Filter to top 50 SKUs by total revenue (from original df)
    top_50_skus = df_clean.groupby('StockCode')['Revenue'].sum().nlargest(50).index.tolist()
    weekly_df = weekly_df[weekly_df['StockCode'].isin(top_50_skus)].copy()
    
    # Create complete week grid for all SKUs
    all_weeks = pd.date_range(
        start=weekly_df['Week_Start'].min(),
        end=weekly_df['Week_Start'].max(),
        freq='W-MON'
    )
    full_grid = pd.MultiIndex.from_product(
        [top_50_skus, all_weeks],
        names=['StockCode', 'Week_Start']
    ).to_frame(index=False)

    weekly_df = full_grid.merge(weekly_df, on=['StockCode','Week_Start'], how='left')
    weekly_df['weekly_quantity'] = weekly_df['weekly_quantity'].fillna(0)
    weekly_df['avg_price'] = weekly_df.groupby('StockCode')['avg_price'].ffill()
    weekly_df['avg_price'] = weekly_df['avg_price'].fillna(0)

    # Sort for feature creation
    weekly_df = weekly_df.sort_values(['StockCode', 'Week_Start'])
    
    # Create features per StockCode
    featured_skus = []
    for sku in top_50_skus:
        sku_df = weekly_df[weekly_df['StockCode'] == sku].copy()
        
        # Ensure all weeks are present (gap filling if necessary, but here we assume continuous enough)
        # Lag features
        for lag in [1, 2, 4, 8]:
            sku_df[f'lag_{lag}'] = sku_df['weekly_quantity'].shift(lag)
            
        # Rolling features
        sku_df['rolling_4_mean'] = sku_df['weekly_quantity'].shift(1).rolling(window=4).mean()
        sku_df['rolling_4_std'] = sku_df['weekly_quantity'].shift(1).rolling(window=4).std()
        sku_df['rolling_8_mean'] = sku_df['weekly_quantity'].shift(1).rolling(window=8).mean()
        
        # Calendar features
        sku_df['week_of_year'] = sku_df['Week_Start'].dt.isocalendar().week.astype(int)
        sku_df['month'] = sku_df['Week_Start'].dt.month
        sku_df['is_q4'] = sku_df['month'].isin([10, 11, 12]).astype(int)
        
        # Price change
        sku_df['price_change'] = sku_df['avg_price'].shift(1).pct_change()
        
        featured_skus.append(sku_df)
        
    final_df = pd.concat(featured_skus)
    
    # Drop rows where any lag feature is NaN
    cols_to_check = ['lag_1', 'lag_2', 'lag_4', 'lag_8', 'rolling_4_mean', 'rolling_8_mean']
    final_df = final_df.dropna(subset=cols_to_check)
    
    # Fill NaN for price_change (first row per SKU)
    final_df['price_change'] = final_df['price_change'].fillna(0)
    
    logging.info(f"Prepared features for {len(top_50_skus)} SKUs. Total rows: {len(final_df)}")
    return final_df

def train_forecast_models(features_df):
    """
    Trains one LightGBM model per StockCode using walk-forward validation.
    """
    logging.info("Training forecast models...")
    models_dict = {}
    performance_records = []
    
    features = ['lag_1', 'lag_2', 'lag_4', 'lag_8', 'rolling_4_mean', 
                'rolling_4_std', 'rolling_8_mean', 'week_of_year', 
                'month', 'is_q4', 'price_change']
    target = 'weekly_quantity'
    
    skus = features_df['StockCode'].unique()
    
    os.makedirs('outputs/models', exist_ok=True)
    
    for sku in skus:
        sku_df = features_df[features_df['StockCode'] == sku].sort_values('Week_Start')
        
        if len(sku_df) < 10:
            logging.warning(f"Not enough data for SKU {sku}, skipping.")
            continue
            
        # Date-based split
        split_date = pd.Timestamp('2011-09-01')
        train_df = sku_df[sku_df['Week_Start'] < split_date]
        test_df = sku_df[sku_df['Week_Start'] >= split_date]
        
        if len(test_df) == 0:
            continue
            
        # Tweedie objective handles intermittent retail demand:
        # expects mass of zeros (ghost weeks) and long tail (bulk orders)
        # tweedie_variance_power=1.5 is the retail sweet spot between
        # Poisson (1.0) and Gamma (2.0) distributions
        params = {
            'objective': 'tweedie',
            'tweedie_variance_power': 1.5,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'min_child_samples': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(train_df[features], train_df[target])
        
        # Predictions
        preds = model.predict(test_df[features])
        
        # Metrics
        mae = mean_absolute_error(test_df[target], preds)
        rmse = np.sqrt(mean_squared_error(test_df[target], preds))
        
        # Naive baseline (lag_1)
        naive_preds = test_df['lag_1']
        naive_mae = mean_absolute_error(test_df[target], naive_preds)
        
        improvement = (naive_mae - mae) / naive_mae if naive_mae > 0 else 0
        
        performance_records.append({
            'StockCode': sku,
            'MAE': mae,
            'RMSE': rmse,
            'Naive_MAE': naive_mae,
            'Improvement_Over_Naive': improvement
        })
        
        # Save model
        models_dict[sku] = model
        joblib.dump(model, f'outputs/models/lgb_{sku}.pkl')
        
    performance_df = pd.DataFrame(performance_records)
    performance_df.to_csv('outputs/model_performance.csv', index=False)
    logging.info("Model performance saved to outputs/model_performance.csv")
    logging.info(f"Trained {len(models_dict)} models.")
    return models_dict, performance_df

def generate_stock_risk_report(models_dict, features_df):
    """
    Generates stock risk report based on 4-week forecasts.
    """
    logging.info("Generating stock risk report...")
    risk_records = []
    
    features_list = ['lag_1', 'lag_2', 'lag_4', 'lag_8', 'rolling_4_mean', 
                'rolling_4_std', 'rolling_8_mean', 'week_of_year', 
                'month', 'is_q4', 'price_change']
    
    for sku, model in models_dict.items():
        sku_df = features_df[features_df['StockCode'] == sku].sort_values('Week_Start')
        last_row = sku_df.iloc[-1].copy()
        
        
        forecast_val = model.predict(pd.DataFrame([last_row[features_list]]))[0]
        # Assume steady demand for 4 weeks for the sake of the exercise
        weekly_forecast = max(0, forecast_val)
        total_4w_forecast = weekly_forecast * 4
        
        # Simulate inventory
        avg_weekly = sku_df['weekly_quantity'].mean()
        inventory = 4 * avg_weekly
        
        # Risk levels
        if weekly_forecast > 0:
            days_until_stockout = inventory / (weekly_forecast / 7)
        else:
            days_until_stockout = 365 # No stockout if no demand
            
        if days_until_stockout < 7:
            risk = 'CRITICAL'
        elif days_until_stockout < 14:
            risk = 'HIGH'
        elif days_until_stockout < 30:
            risk = 'MEDIUM'
        else:
            risk = 'LOW'
            
        suggested_reorder = max(0, (6 * weekly_forecast) - inventory)
        
        risk_records.append({
            'StockCode': sku,
            'Weekly_Forecast': weekly_forecast,
            'Current_Inventory': inventory,
            'Days_Until_Stockout': days_until_stockout,
            'Risk_Level': risk,
            'Suggested_Reorder_Qty': suggested_reorder
        })
        
    risk_df = pd.DataFrame(risk_records).sort_values('Days_Until_Stockout')
    risk_df.to_csv('outputs/stock_risk_report.csv', index=False)
    return risk_df

def main():
    try:
        df = pd.read_csv('data/processed/retail_clean.csv')
    except FileNotFoundError:
        logging.error("Cleaned data not found. Run cleaning.py first.")
        return
        
    features_df = prepare_forecast_features(df)
    models_dict, perf_df = train_forecast_models(features_df)
    risk_df = generate_stock_risk_report(models_dict, features_df)
    
    print("\nModel Performance Summary:")
    print(perf_df[['MAE', 'RMSE', 'Improvement_Over_Naive']].describe().loc[['mean', '50%']])
    
    print("\nTop 10 Critical SKUs:")
    print(risk_df[risk_df['Risk_Level'].isin(['CRITICAL', 'HIGH'])].head(10))

if __name__ == "__main__":
    main()
