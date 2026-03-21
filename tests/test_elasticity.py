import pytest
import pandas as pd
import numpy as np
import os
from src.elasticity import calculate_elasticity, classify_elasticity, ADMIN_CODES

@pytest.fixture
def mock_df():
    """Create a mock dataframe for testing elasticity calculation."""
    # 40 weeks of data for one SKU to pass min_observations=30
    dates = pd.date_range(start='2010-01-01', periods=40, freq='W')
    
    data = []
    # Normal SKU
    for i, date in enumerate(dates):
        price = 10.0 - (i * 0.1) # Declining price
        quantity = 100 + (i * 5) # Increasing quantity (elastic)
        data.append({
            'InvoiceDate': date,
            'StockCode': '12345',
            'Description': 'TEST PRODUCT',
            'Category': 'Test Category',
            'Quantity': quantity,
            'Price': price,
            'Revenue': quantity * price,
            'Year': date.year,
            'Month': date.month,
            'Week': date.isocalendar()[1],
            'is_partial_month': False,
            'Country': 'United Kingdom',
            'customer_type': 'WHOLESALE'
        })
        
    # Admin SKU
    for i, date in enumerate(dates[:5]):
        data.append({
            'InvoiceDate': date,
            'StockCode': 'POST',
            'Description': 'POSTAGE',
            'Category': 'Admin',
            'Quantity': 1,
            'Price': 5.0,
            'Revenue': 5.0,
            'Year': date.year,
            'Month': date.month,
            'Week': date.isocalendar()[1],
            'is_partial_month': False,
            'Country': 'United Kingdom',
            'customer_type': 'WHOLESALE'
        })
        
    return pd.DataFrame(data)

def test_elasticity_columns(mock_df):
    """Test if output has all required columns."""
    results = calculate_elasticity(mock_df, min_observations=10)
    results = classify_elasticity(results)
    
    expected_cols = [
        'StockCode', 'customer_type', 'Description', 'Category', 
        'elasticity_coefficient', 'r_squared', 'p_value', 'observation_count', 
        'mean_price', 'mean_weekly_quantity', 'total_revenue', 'is_significant',
        'elasticity_type', 'commercial_priority', 'estimated_annual_opportunity'
    ]
    
    for col in expected_cols:
        assert col in results.columns

def test_elasticity_type_not_null(mock_df):
    """Test if elasticity_type is never null."""
    results = calculate_elasticity(mock_df, min_observations=10)
    results = classify_elasticity(results)
    
    assert results['elasticity_type'].isnull().sum() == 0
    assert all(results['elasticity_type'].isin(['INELASTIC', 'ELASTIC', 'UNUSUAL']))

def test_opportunity_zero_for_non_high(mock_df):
    """Test if estimated_annual_opportunity is 0 for non-HIGH rows."""
    results = calculate_elasticity(mock_df, min_observations=10)
    results = classify_elasticity(results)
    
    non_high = results[results['commercial_priority'] != 'HIGH']
    assert (non_high['estimated_annual_opportunity'] == 0).all()

def test_no_admin_codes(mock_df):
    """Test if admin codes are excluded from results."""
    results = calculate_elasticity(mock_df, min_observations=1)
    
    assert not results['StockCode'].isin(ADMIN_CODES).any()

def test_elasticity_calculation_logic():
    """Test basic elasticity calculation logic with a known elastic relationship."""
    # Q = 100 * P^-2  => log(Q) = log(100) - 2*log(P) => elasticity = -2
    prices = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1] * 4, dtype=float) # 40 observations
    quantities = 1000 * (prices ** -2)
    
    data = []
    for i in range(40):
        data.append({
            'StockCode': 'ELASTIC_SKU',
            'Description': 'Elastic Item',
            'Category': 'Test',
            'Quantity': quantities[i],
            'Price': prices[i],
            'Revenue': quantities[i] * prices[i],
            'Year': 2010,
            'Week': (i % 52) + 1,
            'is_partial_month': False,
            'Country': 'United Kingdom',
            'customer_type': 'WHOLESALE'
        })
    df = pd.DataFrame(data)
    
    results = calculate_elasticity(df, min_observations=30)
    results = classify_elasticity(results)
    
    coeff = results.iloc[0]['elasticity_coefficient']
    # Should be close to -2
    assert -2.1 < coeff < -1.9
    assert results.iloc[0]['elasticity_type'] == 'ELASTIC'
    assert results.iloc[0]['is_significant'] == True
