import pytest
import pandas as pd
import numpy as np
import os
from src.forecasting import prepare_forecast_features, train_forecast_models, generate_stock_risk_report

@pytest.fixture
def mock_data():
    """
    Creates mock data for testing without reading from the real data/ directory.
    Uses realistic-looking StockCodes to test against 'SKU1'/'SKU2' leftovers.
    """
    # Create 2 years of weekly data for 2 SKUs
    dates = pd.date_range(start='2010-01-01', end='2011-12-31', freq='W-MON')
    data = []
    for sku in ['10001', '10002']:
        for date in dates:
            data.append({
                'StockCode': sku,
                'InvoiceDate': date,
                'Quantity': np.random.randint(10, 100),
                'Price': np.random.uniform(1.0, 10.0),
                'Revenue': np.random.uniform(100, 1000),
                'Country': 'United Kingdom',
                'is_partial_month': False
            })
    return pd.DataFrame(data)

def test_prepare_forecast_features(mock_data):
    # No file I/O here, but we ensure it works with the mock data
    features_df = prepare_forecast_features(mock_data)
    
    assert not features_df.empty
    assert 'lag_1' in features_df.columns
    assert 'rolling_4_mean' in features_df.columns
    assert 'week_of_year' in features_df.columns
    assert features_df['StockCode'].nunique() <= 50
    # After lags and rolling windows, we should have fewer rows than raw data
    assert len(features_df) < len(mock_data)

def test_train_forecast_models(mock_data, tmp_path, monkeypatch):
    """
    Tests model training and ensures no writing to the production outputs/ directory.
    """
    # Redirect all file operations to tmp_path
    monkeypatch.chdir(tmp_path)
    
    features_df = prepare_forecast_features(mock_data)
    
    # Ensure we have enough data for training in the mock
    models_dict, perf_df = train_forecast_models(features_df)
    
    assert isinstance(models_dict, dict)
    assert not perf_df.empty
    assert 'MAE' in perf_df.columns
    assert 'Improvement_Over_Naive' in perf_df.columns
    
    # Verify file creation in the temporary directory
    for sku in models_dict.keys():
        assert os.path.exists(f'outputs/models/lgb_{sku}.pkl')
    assert os.path.exists('outputs/model_performance.csv')

def test_generate_stock_risk_report(mock_data, tmp_path, monkeypatch):
    """
    Tests risk report generation and ensures no writing to the production outputs/ directory.
    Also checks for mock data leftovers.
    """
    # Redirect all file operations to tmp_path
    monkeypatch.chdir(tmp_path)
    
    features_df = prepare_forecast_features(mock_data)
    models_dict, _ = train_forecast_models(features_df)
    
    risk_df = generate_stock_risk_report(models_dict, features_df)
    
    assert not risk_df.empty
    assert 'Risk_Level' in risk_df.columns
    assert 'Days_Until_Stockout' in risk_df.columns
    
    # BUG CHECK: Ensure StockCode values are never 'SKU1' or 'SKU2'
    assert 'SKU1' not in risk_df['StockCode'].values
    assert 'SKU2' not in risk_df['StockCode'].values
    
    # Verify file creation in the temporary directory
    assert os.path.exists('outputs/stock_risk_report.csv')
