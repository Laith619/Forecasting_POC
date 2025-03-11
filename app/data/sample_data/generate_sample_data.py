import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data(
    start_date: str = '2023-01-01',
    end_date: str = '2024-02-29',
    base_units: float = 100,
    base_price: float = 50,
    base_views: float = 1000,
    base_ad_spend: float = 200
):
    """Generate sample e-commerce data with realistic patterns."""
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create base trend (slight upward)
    trend = np.linspace(1, 1.2, len(dates))
    
    # Create seasonal patterns
    day_of_week = pd.Series(dates.dayofweek)
    month_of_year = pd.Series(dates.month)
    
    # Weekly seasonality (weekend effect)
    weekly_pattern = 1 + 0.2 * (day_of_week >= 5)
    
    # Monthly seasonality (holiday effects)
    monthly_pattern = 1 + 0.3 * (month_of_year == 12)  # December boost
    monthly_pattern += 0.2 * (month_of_year == 11)     # November boost
    monthly_pattern += 0.1 * (month_of_year == 1)      # January sales
    
    # Combine patterns
    combined_pattern = (trend * weekly_pattern * monthly_pattern).values
    
    # Add random noise
    noise = np.random.normal(1, 0.1, len(dates))
    final_pattern = combined_pattern * noise
    
    # Generate metrics
    units_sold = (base_units * final_pattern).round()
    
    # Price varies slightly with demand
    price_pattern = 1 + 0.1 * (final_pattern - 1)
    price = base_price * price_pattern
    
    # GMV calculation
    gmv = units_sold * price
    
    # Page views (conversion rate varies with season)
    conversion_rate = 0.1 * (1 + 0.2 * (final_pattern - 1))
    page_views = (units_sold / conversion_rate).round()
    
    # Ad spend and sales
    ad_spend = base_ad_spend * (1 + 0.3 * np.random.random(len(dates)))
    ad_roas = 2.5 + 0.5 * np.random.random(len(dates))  # ROAS between 2.5 and 3
    ad_sales = ad_spend * ad_roas
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'units_sold': units_sold,
        'gmv': gmv,
        'page_views': page_views,
        'ad_spend': ad_spend,
        'ad_sales': ad_sales
    })
    
    return df

if __name__ == "__main__":
    # Generate sample data
    df = generate_sample_data()
    
    # Save to CSV
    output_file = 'sample_ecommerce_data.csv'
    df.to_csv(output_file, index=False)
    print(f"Sample data saved to {output_file}")
    
    # Display first few rows and basic stats
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nBasic statistics:")
    print(df.describe()) 