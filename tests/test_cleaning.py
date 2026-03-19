import pandas as pd
import pytest
from src.cleaning import clean_data

@pytest.fixture
def sample_data():
    """Create a sample dataframe with known issues to test the cleaning pipeline."""
    return pd.DataFrame({
        'Invoice': ['536365', 'C536379', '536366', '536367', '536368', '536369', '536370', '536370'],
        'StockCode': ['85123A', 'D', '22633', '84879', '22752', '21730', '22727', '22727'],
        'Description': [' WHITE HANGING HEART T-LIGHT HOLDER', 'Discount', 'HAND WARMER UNION JACK', 'ASSORTED COLOUR BIRD ORNAMENT', 'SET 7 BABUSHKA NESTING BOXES', 'GLASS STAR FROSTED T-LIGHT HOLDER', 'ALARM CLOCK BAKELIKE RED ', 'ALARM CLOCK BAKELIKE RED '],
        'Quantity': [6, -1, 6, 0, 2, 3, 4, 4],
        'InvoiceDate': ['2010-12-01 08:26:00', '2010-12-01 09:41:00', '2010-12-01 08:28:00', '2010-12-01 08:34:00', '2010-12-01 08:34:00', '2010-12-01 08:35:00', '2010-12-01 08:45:00', '2010-12-01 08:45:00'],
        'Price': [2.55, 27.50, 1.85, 1.69, 7.65, -1.00, 3.75, 3.75],
        'Customer ID': [17850, 14527, 17850, None, 17850, 17850, 12583, 12583],
        'Country': ['United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom', 'France', 'France']
    })

def test_clean_data(sample_data):
    """Test that all cleaning steps work as expected."""
    df_clean = clean_data(sample_data)
    
    # 2. Cancelled invoices removed
    assert not any(df_clean['Invoice'].str.startswith('C'))
    
    # 3 & 4. Positive quantity and price
    assert all(df_clean['Quantity'] > 0)
    assert all(df_clean['Price'] > 0)
    
    # 5. Missing customer IDs removed
    assert not df_clean['Customer ID'].isna().any()
    
    # 6. Duplicates removed (last two rows were identical except for maybe index)
    assert len(df_clean) == 4 # Rows kept: 0, 2, 4, 6 (since 7 is duplicate)
    
    # 7. Description standardized
    assert 'WHITE HANGING HEART T-LIGHT HOLDER' in df_clean['Description'].values
    assert 'ALARM CLOCK BAKELIKE RED' in df_clean['Description'].values
    
    # 8. Revenue column created
    assert 'Revenue' in df_clean.columns
    assert df_clean.loc[df_clean['Invoice'] == '536365', 'Revenue'].iloc[0] == pytest.approx(15.30)
    
    # 9. Date features extracted
    assert 'Year' in df_clean.columns
    assert 'Month' in df_clean.columns
    assert 'DayOfWeek' in df_clean.columns
    assert 'Week' in df_clean.columns
    
    # Check date types
    assert df_clean['Year'].iloc[0] == 2010
