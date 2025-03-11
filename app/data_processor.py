import pandas as pd
import numpy as np
from typing import List, Dict, Union, Any
from datetime import datetime
import logging
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.required_columns = [
            'Week Start', 'Units Sold', 'GMV', 'Page Views',
            'Ad Spend', 'Ad Sales'
        ]
        self.output_columns = [
            'date', 'units_sold', 'gmv', 'page_views',
            'ad_spend', 'ad_sales'
        ]
        self.metrics = {}
        logger.info("DataProcessor initialized with required columns: %s", self.required_columns)

    def process_data(self, data: Union[pd.DataFrame, Any]) -> pd.DataFrame:
        """Process the input data through validation, cleaning, and preparation."""
        try:
            logger.info("Starting data processing")
            
            # Check if data is a Streamlit UploadedFile object and convert to DataFrame
            if hasattr(data, 'read'):
                try:
                    # Read the uploaded file into a pandas DataFrame
                    data = pd.read_csv(data)
                    logger.info("Successfully read uploaded file into DataFrame with shape: %s", data.shape)
                except Exception as e:
                    logger.error("Error reading uploaded file: %s", str(e))
                    raise ValueError(f"Could not read uploaded file: {str(e)}")
            
            # Transform data first - make sure we're working with a DataFrame now
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame after processing uploads")
                
            transformed_data = self._transform_weekly_data(data)
            
            # Handle missing data
            transformed_data = self.handle_missing_data(transformed_data)
            
            # Validate the data
            if not self.validate_data(transformed_data):
                raise ValueError("Data validation failed")
            
            # Clean and prepare the data
            cleaned_data = self.clean_data(transformed_data)
            
            # Engineer features
            cleaned_data = self.engineer_features(cleaned_data)
            
            # Calculate metrics
            self.calculate_metrics(cleaned_data)
            
            return cleaned_data
            
        except Exception as e:
            logger.error("Error in process_data: %s", str(e))
            raise

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate the input data structure and content."""
        try:
            logger.info("Starting data validation")
            
            # Check numeric columns
            numeric_columns = ['units_sold', 'gmv', 'page_views', 'ad_spend', 'ad_sales']
            
            for col in numeric_columns:
                logger.info("Validating column %s", col)
                
                # Convert to numeric, coercing errors to NaN
                data[col] = pd.to_numeric(data[col], errors='coerce')
                
                # Check for all NaN
                if data[col].isna().all():
                    logger.error("Column %s contains all NaN values", col)
                    return False
                
                # Check for negative values
                if (data[col] < 0).any():
                    logger.warning("Column %s contains negative values, will be replaced with 0", col)
                    data[col] = data[col].clip(lower=0)
            
            logger.info("Data validation completed successfully")
            return True
            
        except Exception as e:
            logger.error("Error in validate_data: %s", str(e))
            return False

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare the data for forecasting."""
        try:
            logger.info("Starting data cleaning")
            
            df = data.copy()
            
            # Handle missing values for numeric columns
            numeric_columns = ['units_sold', 'gmv', 'page_views', 'ad_spend', 'ad_sales']
            for col in numeric_columns:
                logger.info("Cleaning column: %s", col)
                
                # Replace 0s with NaN for better interpolation
                df[col] = df[col].replace(0, np.nan)
                
                # Interpolate missing values
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
                
                # If any NaN remains, use forward fill then backward fill
                df[col] = df[col].ffill().bfill()
                
                # Replace any remaining NaN with 0
                df[col] = df[col].fillna(0)
                
                # Ensure non-negative values
                df[col] = df[col].clip(lower=0)
                
                logger.info("Column %s cleaned. Range: [%.2f, %.2f]", 
                          col, df[col].min(), df[col].max())
            
            # Remove duplicates
            df = df.drop_duplicates()
            logger.info("Duplicates removed")
            
            # Sort by date
            df = df.sort_values('date')
            logger.info("Data sorted by date")
            
            # Calculate derived features
            df['conversion_rate'] = np.where(df['page_views'] > 0, 
                                           df['units_sold'] / df['page_views'], 
                                           0)
            df['roas'] = np.where(df['ad_spend'] > 0, 
                                 df['ad_sales'] / df['ad_spend'], 
                                 0)
            logger.info("Derived features calculated")
            
            # Set date as index
            df.set_index('date', inplace=True)
            logger.info("Date set as index")
            
            logger.info("Data cleaning completed. Shape: %s", df.shape)
            return df
            
        except Exception as e:
            logger.error("Error in clean_data: %s", str(e))
            raise

    def _transform_weekly_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the raw data into the required format."""
        try:
            # Safety check to ensure we have a DataFrame
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Input to _transform_weekly_data must be a pandas DataFrame")
                
            logger.info("Starting data transformation")
            logger.info("Input columns: %s", data.columns.tolist())
            
            # Check for required columns
            missing_cols = [col for col in self.required_columns if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            logger.info("All required columns found")
            
            # Create a copy of the data
            df = data.copy()
            
            # Rename columns to lowercase with underscores
            column_mapping = {
                'Week Start': 'date',
                'Units Sold': 'units_sold',
                'GMV': 'gmv',
                'Page Views': 'page_views',
                'Ad Spend': 'ad_spend',
                'Ad Sales': 'ad_sales'
            }
            df = df.rename(columns=column_mapping)
            logger.info("Columns renamed. New columns: %s", df.columns.tolist())
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            logger.info("Date column converted to datetime")
            
            # Sort by date in ascending order
            df = df.sort_values('date')
            logger.info("Data sorted by date in ascending order")
            
            return df
            
        except Exception as e:
            logger.error("Error in _transform_weekly_data: %s", str(e))
            raise

    def calculate_metrics(self, data: pd.DataFrame) -> None:
        """Calculate performance metrics from processed data."""
        try:
            metrics = {
                'total_gmv': data['gmv'].sum(),
                'total_units': data['units_sold'].sum(),
                'avg_daily_sales': data['units_sold'].mean(),
                'avg_roas': (data['ad_sales'] / data['ad_spend']).mean(),
                'avg_conversion_rate': (data['units_sold'] / data['page_views']).mean(),
                'total_ad_spend': data['ad_spend'].sum(),
                'total_ad_sales': data['ad_sales'].sum()
            }
            self.metrics = metrics
            logger.info("Metrics calculated: %s", metrics)
        except Exception as e:
            logger.error("Error calculating metrics: %s", str(e))
            raise

    def get_metrics(self) -> Dict[str, float]:
        """Get the calculated performance metrics."""
        if not self.metrics:
            logger.warning("No metrics available. Process data first.")
            return {
                'total_gmv': 0.0,
                'total_units': 0,
                'avg_daily_sales': 0.0,
                'avg_roas': 0.0,
                'avg_conversion_rate': 0.0,
                'total_ad_spend': 0.0,
                'total_ad_sales': 0.0
            }
        return self.metrics

    def prepare_for_forecast(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Prepare data for forecasting models."""
        try:
            logger.info("Preparing data for forecasting. Target column: %s", target_col)
            
            # Map the display name to the internal column name
            column_mapping = {
                'units_sold': 'units_sold',
                'gmv': 'gmv',
                'page_views': 'page_views',
                'ad_spend': 'ad_spend',
                'ad_sales': 'ad_sales'
            }
            
            # Get the correct column name
            actual_col = column_mapping.get(target_col, target_col)
            logger.info("Using column: %s", actual_col)
            
            if actual_col not in df.columns:
                raise ValueError(f"Column {actual_col} not found in data")
            
            forecast_df = df[[actual_col]].copy()
            logger.info("Forecast data prepared. Shape: %s", forecast_df.shape)
            return forecast_df
            
        except Exception as e:
            logger.error("Error in prepare_for_forecast: %s", str(e))
            raise

    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset using multiple strategies."""
        try:
            logger.info("Starting missing data handling")
            
            # Create a copy of the DataFrame
            df = df.copy()
            
            # Ensure date column is datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Sort by date
            df = df.sort_values('date')
            
            # Check for gaps in weekly dates
            date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='W-MON')
            missing_dates = set(date_range) - set(df['date'])
            
            if missing_dates:
                logger.warning(f"Found {len(missing_dates)} missing weeks in the data")
                st.warning(f"Found {len(missing_dates)} missing weeks in the data. These will be filled using interpolation.")
                
                # Create placeholder rows for missing dates
                missing_rows = []
                for missing_date in missing_dates:
                    placeholder = pd.Series(index=df.columns)
                    placeholder['date'] = missing_date
                    missing_rows.append(placeholder)
                
                if missing_rows:
                    df = pd.concat([df, pd.DataFrame(missing_rows)], ignore_index=True)
                    df = df.sort_values('date')
            
            # Handle missing values for each column type
            numeric_columns = ['units_sold', 'gmv', 'page_views', 'ad_spend', 'ad_sales']
            
            for col in numeric_columns:
                if col in df.columns:
                    missing_count = df[col].isna().sum()
                    if missing_count > 0:
                        logger.info(f"Handling {missing_count} missing values in {col}")
                        
                        # Strategy 1: Linear interpolation for small gaps
                        df[col] = df[col].interpolate(method='linear', limit_direction='both', limit=3)
                        
                        # Strategy 2: Forward fill + Backward fill for remaining gaps
                        df[col] = df[col].fillna(method='ffill', limit=2)
                        df[col] = df[col].fillna(method='bfill', limit=2)
                        
                        # Strategy 3: Use rolling median for any remaining missing values
                        remaining_missing = df[col].isna().sum()
                        if remaining_missing > 0:
                            rolling_median = df[col].rolling(window=7, min_periods=1, center=True).median()
                            df[col] = df[col].fillna(rolling_median)
                        
                        # Final check: fill any remaining NaN with 0
                        df[col] = df[col].fillna(0)
                        
                        # Ensure non-negative values
                        df[col] = df[col].clip(lower=0)
            
            logger.info("Missing data handling completed")
            return df
            
        except Exception as e:
            logger.error(f"Error handling missing data: {str(e)}")
            raise

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for improved forecasting."""
        try:
            logger.info("Starting feature engineering")
            
            # Create a copy of the DataFrame
            df = df.copy()
            
            # Get the date index as a Series for feature creation
            date_series = df.index
            
            # 1. Time-based features
            df['year'] = date_series.year
            df['month'] = date_series.month
            df['week_of_year'] = date_series.isocalendar().week
            df['day_of_week'] = date_series.dayofweek
            df['is_month_start'] = date_series.is_month_start.astype(int)
            df['is_month_end'] = date_series.is_month_end.astype(int)
            
            # 2. Lag features for key metrics
            lag_periods = [1, 2, 3, 4]  # Previous 1-4 weeks
            for col in ['units_sold', 'page_views', 'ad_spend']:
                if col in df.columns:
                    for lag in lag_periods:
                        lag_col = f'{col}_lag_{lag}'
                        df[lag_col] = df[col].shift(lag)
            
            # 3. Rolling statistics
            windows = [7, 14, 28]  # 1 week, 2 weeks, 4 weeks
            for col in ['units_sold', 'page_views', 'ad_spend']:
                if col in df.columns:
                    for window in windows:
                        # Rolling mean
                        df[f'{col}_rolling_mean_{window}d'] = df[col].rolling(window=window, min_periods=1).mean()
                        # Rolling std
                        df[f'{col}_rolling_std_{window}d'] = df[col].rolling(window=window, min_periods=1).std()
            
            # 4. Interaction features
            if all(col in df.columns for col in ['page_views', 'units_sold', 'ad_spend']):
                # Conversion rate
                df['conversion_rate'] = (df['units_sold'] / df['page_views'].replace(0, np.nan)).fillna(0)
                # Ad efficiency
                df['ad_efficiency'] = (df['units_sold'] / df['ad_spend'].replace(0, np.nan)).fillna(0)
                # Revenue per visitor
                df['revenue_per_visitor'] = (df['gmv'] / df['page_views'].replace(0, np.nan)).fillna(0)
            
            # 5. Growth rates
            for col in ['units_sold', 'page_views', 'ad_spend']:
                if col in df.columns:
                    df[f'{col}_pct_change'] = df[col].pct_change().fillna(0)
            
            # 6. Seasonality features
            if len(df) >= 52:  # Only if we have at least a year of data
                for col in ['units_sold', 'page_views']:
                    if col in df.columns:
                        # Weekly seasonality
                        weekly_avg = df.groupby('day_of_week')[col].transform('mean')
                        weekly_std = df.groupby('day_of_week')[col].transform('std')
                        df[f'{col}_day_of_week_norm'] = (df[col] - weekly_avg) / weekly_std.replace(0, 1)
                        
                        # Monthly seasonality
                        monthly_avg = df.groupby('month')[col].transform('mean')
                        monthly_std = df.groupby('month')[col].transform('std')
                        df[f'{col}_month_norm'] = (df[col] - monthly_avg) / monthly_std.replace(0, 1)
            
            # Drop rows with NaN values created by lag features
            df = df.fillna(0)
            
            logger.info("Feature engineering completed. New features added: %s", 
                       [col for col in df.columns if col not in self.output_columns])
            
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise 