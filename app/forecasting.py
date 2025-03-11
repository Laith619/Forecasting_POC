import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Dict, Tuple, Optional, List, Any
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForecastConfig:
    """Configuration class for forecasting parameters."""
    def __init__(
        self,
        seasonality_mode: str = 'multiplicative',
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = True,
        daily_seasonality: bool = False,
        seasonality_prior_scale: float = 10.0,
        changepoint_prior_scale: float = 0.05,
        changepoint_range: float = 0.8,
        interval_width: float = 0.95,
        growth: str = 'linear'
    ):
        self.seasonality_mode = seasonality_mode
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_prior_scale = seasonality_prior_scale
        self.changepoint_prior_scale = changepoint_prior_scale
        self.changepoint_range = changepoint_range
        self.interval_width = interval_width
        self.growth = growth

class ForecastingEngine:
    def __init__(self, config: ForecastConfig = None):
        self.models = {}
        self.metrics = {}
        self.forecasts = {}
        self.config = config or ForecastConfig()
        
        # Define metric relationships and order with conversion rates
        self.metric_relationships = {
            'ad_spend': [],  # Independent variable (user input)
            'page_views': ['ad_spend'],  # Traffic = f(ad_spend)
            'units_sold': ['page_views', 'ad_spend'],  # Conversion = f(traffic, ad_spend)
            'gmv': ['units_sold'],  # Revenue = f(units)
            'ad_sales': ['ad_spend', 'units_sold']  # ROAS = f(ad_spend, units)
        }
        
        # Define key performance indicators
        self.kpis = {
            'conversion_rate': lambda data: data['units_sold'] / data['page_views'],
            'aov': lambda data: data['gmv'] / data['units_sold'],
            'roas': lambda data: data['ad_sales'] / data['ad_spend']
        }
        
        self.available_metrics = list(self.metric_relationships.keys())
        logger.info("ForecastingEngine initialized with metric relationships and KPIs")

    def format_weekly_forecast(self, forecast: pd.DataFrame, target_col: str) -> List[Dict[str, Any]]:
        """Format forecast into weekly periods with interpretations.
        
        Args:
            forecast: DataFrame containing the forecast data
            target_col: The display name for the metric being forecast
            
        Returns:
            List of dictionaries with formatted weekly forecast data
        """
        try:
            logger.info("Starting to format weekly forecast for %s", target_col)
            logger.info("Forecast data shape: %s", forecast.shape)
            
            # Get the last historical date from the forecast itself
            # (assumes forecast contains both historical and future data)
            last_historical_date = None
            
            # Check if 'y' column exists (historical values)
            if 'y' in forecast.columns:
                # Find where y is not null (historical data)
                historical_mask = forecast['y'].notna()
                if historical_mask.any():
                    last_historical_date = forecast.index[historical_mask].max()
                    logger.info("Found last historical date using y column: %s", last_historical_date)
            
            # If we couldn't find it using y values, try a different approach
            if last_historical_date is None:
                # Try to find the date where NaN values in 'y' start
                if 'y' in forecast.columns:
                    # Get the first date where y is NaN
                    sorted_forecast = forecast.sort_index()
                    y_null_mask = sorted_forecast['y'].isna()
                    if y_null_mask.any() and not y_null_mask.all():
                        first_null_idx = y_null_mask.idxmax() if not y_null_mask.iloc[0] else None
                        if first_null_idx is not None:
                            # Get the date right before the first null
                            first_null_date = sorted_forecast.index.get_loc(first_null_idx)
                            if first_null_date > 0:
                                last_historical_date = sorted_forecast.index[first_null_date - 1]
                                logger.info("Found last historical date using y null pattern: %s", last_historical_date)
            
            # If still not found, use the ds column
            if last_historical_date is None and 'ds' in forecast.columns:
                # Get the current date minus 1 day (assuming forecasts start from tomorrow)
                today = pd.Timestamp.now().normalize()
                last_historical_date = today - pd.Timedelta(days=1)
                logger.info("Using current date minus 1 day as last historical date: %s", last_historical_date)
                
                # Try to find an actual date close to this in the data
                closest_date = forecast.index[forecast.index <= last_historical_date]
                if not closest_date.empty:
                    last_historical_date = closest_date.max()
                    logger.info("Adjusted to closest date in data: %s", last_historical_date)
            
            # Last resort - just use a portion of the data
            if last_historical_date is None:
                # Assume the last 20% of dates are future dates
                dates = forecast.index.sort_values()
                cutoff_idx = int(len(dates) * 0.8)
                last_historical_date = dates[cutoff_idx]
                logger.info("Using 80/20 split to determine last historical date: %s", last_historical_date)
            
            logger.info("Last historical date determined: %s", last_historical_date)
            
            # Get future predictions using the determined historical cutoff
            future_forecast = forecast[forecast.index > last_historical_date].copy()
            logger.info("Future forecast shape: %s", future_forecast.shape)
            
            if future_forecast.empty:
                logger.warning("No future forecast data found. Forecast shape: %s, index range: %s to %s", 
                              forecast.shape, 
                              forecast.index.min() if not forecast.empty else "N/A", 
                              forecast.index.max() if not forecast.empty else "N/A")
                return []
            
            # Ensure the future forecast has the required columns
            required_columns = ['yhat', 'yhat_lower', 'yhat_upper']
            for col in required_columns:
                if col not in future_forecast.columns:
                    logger.error("Required column '%s' missing from forecast data", col)
                    return []
                    
            # Log a preview of the future forecast
            logger.info("Future forecast preview (first row):\n%s", 
                        future_forecast[['yhat', 'yhat_lower', 'yhat_upper']].head(1))
            
            formatted_weeks = []
            previous_forecast = None
            
            # Determine the start date of the first week
            # (align to start of week - Monday)
            first_date = future_forecast.index.min()
            start_of_week = first_date - pd.Timedelta(days=first_date.weekday())
            
            # Create week start dates
            week_starts = pd.date_range(start=start_of_week, 
                                        end=future_forecast.index.max(), 
                                        freq='W-MON')
            
            # If there are no full weeks, use days instead
            if len(week_starts) == 0:
                logger.info("No full weeks in forecast period, using daily data instead")
                week_starts = pd.date_range(start=first_date, 
                                          end=future_forecast.index.max(), 
                                          freq='D')
                
            logger.info("Processing %d weeks from %s to %s", 
                        len(week_starts), 
                        week_starts[0] if len(week_starts) > 0 else "N/A", 
                        week_starts[-1] if len(week_starts) > 0 else "N/A")
            
            # Process each week
            for week_start in week_starts:
                try:
                    week_end = week_start + pd.Timedelta(days=6)
                    
                    # Get data for the current week
                    week_mask = (future_forecast.index >= week_start) & (future_forecast.index <= week_end)
                    week_data = future_forecast[week_mask]
                    
                    if week_data.empty:
                        logger.info("No data for week %s to %s", week_start, week_end)
                        continue
                        
                    # Take the average of the week's forecasts
                    yhat = week_data['yhat'].mean()
                    yhat_lower = week_data['yhat_lower'].mean()
                    yhat_upper = week_data['yhat_upper'].mean()
                    
                    # Calculate week-over-week change
                    wow_change = 0
                    if previous_forecast is not None and previous_forecast != 0:
                        wow_change = ((yhat - previous_forecast) / previous_forecast) * 100
                    
                    previous_forecast = yhat
                    
                    # Ensure values are positive
                    yhat = max(0, yhat)
                    yhat_lower = max(0, yhat_lower)
                    yhat_upper = max(0, yhat_upper)
                    
                    # Format the forecast data
                    weekly_data = {
                        'week_start': week_start.strftime('%Y-%m-%d'),
                        'week_end': week_end.strftime('%Y-%m-%d'),
                        'forecasted_units': int(round(yhat, 0)),
                        'lower_bound': int(round(yhat_lower, 0)),
                        'upper_bound': int(round(yhat_upper, 0)),
                        'wow_change': round(wow_change, 1),
                        'interpretation': self._generate_interpretation(
                            pd.Series({
                                'yhat': yhat, 
                                'yhat_lower': yhat_lower, 
                                'yhat_upper': yhat_upper
                            }), 
                            wow_change
                        )
                    }
                    
                    # Add seasonality components if available
                    weekly_data['weekly_seasonality'] = 0
                    weekly_data['monthly_seasonality'] = 0
                    
                    if 'weekly' in week_data.columns:
                        weekly_data['weekly_seasonality'] = round(week_data['weekly'].mean(), 2)
                    if 'yearly' in week_data.columns:
                        weekly_data['monthly_seasonality'] = round(week_data['yearly'].mean(), 2)
                    
                    formatted_weeks.append(weekly_data)
                    logger.info("Processed week: %s to %s, forecast: %d units", 
                                weekly_data['week_start'], 
                                weekly_data['week_end'], 
                                weekly_data['forecasted_units'])
                except Exception as e:
                    logger.error("Error processing week starting %s: %s", week_start, str(e))
                    continue
            
            logger.info("Completed weekly forecast formatting. Generated %d weeks of data.", len(formatted_weeks))
            return formatted_weeks
            
        except Exception as e:
            logger.error("Error in format_weekly_forecast: %s", str(e))
            return []

    def _generate_interpretation(self, row: pd.Series, wow_change: float) -> str:
        """Generate interpretation for weekly forecast."""
        try:
            interpretation = []
            
            # Trend interpretation
            if wow_change > 5:
                interpretation.append("Expected significant increase")
            elif wow_change < -5:
                interpretation.append("Expected significant decrease")
            else:
                interpretation.append("Relatively stable forecast")
                
            # Confidence interval interpretation
            yhat = max(0.1, row['yhat'])  # Avoid division by zero
            uncertainty = (row['yhat_upper'] - row['yhat_lower']) / yhat * 100
            if uncertainty > 30:
                interpretation.append("High uncertainty in forecast")
            elif uncertainty > 15:
                interpretation.append("Moderate uncertainty")
            else:
                interpretation.append("Good confidence in forecast")
                
            return ". ".join(interpretation) + "."
            
        except Exception as e:
            logger.error("Error in _generate_interpretation: %s", str(e))
            return "Unable to generate interpretation"

    def prepare_training_data(self, data: pd.DataFrame, target_col: str, regressor_cols: List[str] = None) -> pd.DataFrame:
        """Prepare data for Prophet with regressors."""
        try:
            # Create a copy of the data
            df = data.copy()
            
            # Reset index to get date as a column if it's in the index
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
            
            # Ensure we have the target column
            if target_col not in df.columns:
                raise ValueError(f"Target column {target_col} not found in data")
            
            # Rename columns for Prophet
            df = df.rename(columns={df.index.name if df.index.name else 'date': 'ds',
                                  target_col: 'y'})
            
            # Add regressor columns if provided
            if regressor_cols:
                for reg in regressor_cols:
                    if reg not in df.columns:
                        raise ValueError(f"Regressor {reg} not found in data")
                    # Ensure regressor values are numeric
                    df[reg] = pd.to_numeric(df[reg], errors='coerce')
            
            # Sort by date
            df = df.sort_values('ds')
            
            # Remove any duplicate dates
            df = df.drop_duplicates(subset=['ds'])
            
            logger.info("Data prepared for Prophet. Shape: %s", df.shape)
            return df
            
        except Exception as e:
            logger.error("Error in prepare_training_data: %s", str(e))
            raise

    def train_prophet_model(self, data: pd.DataFrame, target_col: str, regressor_cols: List[str] = None) -> None:
        """Train Prophet model with regressors for correlated metrics."""
        try:
            logger.info("Starting Prophet model training for %s with regressors: %s", 
                       target_col, regressor_cols if regressor_cols else "None")
            
            # Store the original data for later use in generating forecasts
            self.original_data = data.copy()
            
            # Prepare data for Prophet
            df = self.prepare_training_data(data, target_col, regressor_cols)
            
            # For units_sold, add additional preprocessing to improve forecast quality
            if target_col == 'units_sold':
                # Get historical statistics
                historical_mean = df['y'].mean()
                historical_std = df['y'].std()
                historical_min = df['y'].min()
                
                # Detect and handle outliers more conservatively
                df = self._handle_outliers(df, target_col)
                
                # Add minimum growth constraint
                df['min_sales'] = historical_min * 0.8  # 80% of historical minimum
                
                # Configure Prophet model with better parameters for units_sold
                model = Prophet(
                    growth='linear',
                    seasonality_mode='multiplicative',
                    weekly_seasonality=True,
                    yearly_seasonality=True,
                    daily_seasonality=False,
                    seasonality_prior_scale=25.0,  # Increased for stronger seasonality
                    changepoint_prior_scale=0.01,  # Reduced to prevent overfitting
                    changepoint_range=0.95,  # Increased to capture more recent trends
                    interval_width=0.95
                )
                
                # Add floor as regressor to prevent negative forecasts
                model.add_regressor('min_sales', mode='additive')
                
                # Add page_views as a regressor if available
                if 'page_views' in data.columns:
                    # Calculate correlation between page_views and units_sold
                    correlation = data['page_views'].corr(data['units_sold'])
                    logger.info(f"Correlation between page_views and units_sold: {correlation:.2f}")
                    
                    if abs(correlation) > 0.3:  # Only use if correlation is significant
                        # Normalize page_views between 0.1 and 1
                        page_views_data = data['page_views'].copy()
                        page_views_min = page_views_data.min()
                        page_views_max = page_views_data.max()
                        page_views_normalized = 0.1 + 0.9 * (page_views_data - page_views_min) / (page_views_max - page_views_min)
                        
                        # Add to DataFrame
                        dates = pd.to_datetime(df['ds'])
                        for date, value in zip(dates, page_views_normalized):
                            df.loc[df['ds'] == date, 'page_views'] = value
                        
                        # Add as regressor with appropriate mode based on correlation
                        if correlation > 0:
                            model.add_regressor('page_views', mode='additive', standardize=False)
                            logger.info("Added page_views as positive additive regressor")
                        else:
                            model.add_regressor('page_views', mode='multiplicative', standardize=False)
                            logger.info("Added page_views as inverse multiplicative regressor")
                    else:
                        logger.info("Skipping page_views regressor due to low correlation")
                
                logger.info("Using optimized configuration for units_sold with floor constraint")
            else:
                # Configure Prophet model with user settings for other metrics
                model = Prophet(
                    seasonality_mode=self.config.seasonality_mode,
                    weekly_seasonality=self.config.weekly_seasonality,
                    yearly_seasonality=self.config.yearly_seasonality,
                    daily_seasonality=self.config.daily_seasonality,
                    seasonality_prior_scale=self.config.seasonality_prior_scale,
                    changepoint_prior_scale=self.config.changepoint_prior_scale,
                    changepoint_range=self.config.changepoint_range,
                    interval_width=self.config.interval_width,
                    growth=self.config.growth
                )
            
            # Add regressors if provided
            if regressor_cols:
                for reg in regressor_cols:
                    # Add regressor with appropriate mode based on relationship
                    if reg in ['ad_spend', 'page_views']:
                        model.add_regressor(reg, mode='multiplicative')
                    else:
                        model.add_regressor(reg, mode='additive')
                logger.info("Added regressors with appropriate modes: %s", regressor_cols)
            
            # Calculate and log pre-training KPIs
            self._log_kpis(data, target_col)
            
            # Fit the model
            model.fit(df)
            logger.info("Model fitting completed")
            
            # Store the model
            self.models[target_col] = model
            
            # Generate in-sample predictions and calculate metrics
            initial_forecast = self._generate_initial_forecast(model, df, target_col)
            
            # Log post-training metrics and KPIs
            self._log_forecast_metrics(df['y'].values, initial_forecast['yhat'].values, target_col)
            
            return initial_forecast
            
        except Exception as e:
            logger.error("Error in train_prophet_model: %s", str(e))
            raise

    def _handle_outliers(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Handle outliers in the target column to improve forecast quality."""
        try:
            # Get the y column values
            y_values = df['y'].values
            
            # Calculate IQR for outlier detection
            q1 = np.percentile(y_values, 25)
            q3 = np.percentile(y_values, 75)
            iqr = q3 - q1
            
            # Define outlier bounds
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Identify outliers
            outliers = (y_values < lower_bound) | (y_values > upper_bound)
            outlier_count = np.sum(outliers)
            
            if outlier_count > 0:
                logger.info("Detected %d outliers in %s", outlier_count, target_col)
                
                # Replace outliers with the median of nearby points (3 on each side)
                df_copy = df.copy()
                for i in range(len(df_copy)):
                    if outliers[i]:
                        # Get window of points around the outlier (excluding the outlier itself)
                        start_idx = max(0, i - 3)
                        end_idx = min(len(df_copy), i + 4)
                        window = np.concatenate([
                            y_values[start_idx:i],
                            y_values[i+1:end_idx]
                        ])
                        
                        if len(window) > 0:
                            # Replace with median of window
                            df_copy.loc[df_copy.index[i], 'y'] = np.median(window)
                            logger.info("Replaced outlier at index %d (value: %.2f) with median: %.2f", 
                                      i, y_values[i], df_copy.loc[df_copy.index[i], 'y'])
                
                return df_copy
            
            return df
            
        except Exception as e:
            logger.error("Error handling outliers: %s", str(e))
            return df

    def _log_kpis(self, data: pd.DataFrame, target_col: str) -> None:
        """Log relevant KPIs before training."""
        try:
            logger.info("Pre-training KPIs for %s:", target_col)
            if target_col == 'units_sold' and 'page_views' in data.columns:
                conv_rate = (data['units_sold'] / data['page_views']).mean()
                logger.info("Average conversion rate: %.4f", conv_rate)
            elif target_col == 'ad_sales' and 'ad_spend' in data.columns:
                roas = (data['ad_sales'] / data['ad_spend']).mean()
                logger.info("Average ROAS: %.2f", roas)
        except Exception as e:
            logger.error("Error calculating KPIs: %s", str(e))

    def _log_forecast_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, target_col: str) -> None:
        """Log comprehensive forecast metrics."""
        try:
            metrics = {
                'mape': mean_absolute_percentage_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mean_error': np.mean(y_true - y_pred),
                'mean_absolute_error': np.mean(np.abs(y_true - y_pred))
            }
            
            self.metrics[target_col] = metrics
            
            logger.info("Model performance metrics for %s:", target_col)
            logger.info("MAPE: %.2f%%", metrics['mape'] * 100)
            logger.info("RMSE: %.2f", metrics['rmse'])
            logger.info("Mean Error: %.2f", metrics['mean_error'])
            logger.info("Mean Absolute Error: %.2f", metrics['mean_absolute_error'])
            
        except Exception as e:
            logger.error("Error calculating metrics: %s", str(e))

    def generate_forecast(self, target_col: str, future_periods: int) -> pd.DataFrame:
        """Generate forecast for target column using trained model."""
        try:
            if target_col not in self.models:
                raise ValueError(f"No trained model found for {target_col}")
            
            model = self.models[target_col]
            historical_forecast = self.forecasts[target_col]
            
            # Get the last historical date
            if isinstance(historical_forecast.index, pd.DatetimeIndex):
                last_historical_date = historical_forecast.index.max()
            else:
                last_historical_date = pd.to_datetime(historical_forecast['ds']).max()
            
            logger.info("Last historical date: %s", last_historical_date)
            
            # Create future dates dataframe
            future_dates = pd.date_range(
                start=last_historical_date + pd.Timedelta(days=1),
                periods=future_periods * 7,  # Convert weeks to days
                freq='D'
            )
            
            future = pd.DataFrame({'ds': future_dates})
            logger.info("Created future dates from %s to %s", 
                       future_dates[0], future_dates[-1])
            
            # For units_sold, add floor constraint to future dates
            if target_col == 'units_sold':
                historical_min = historical_forecast['y'].min()
                future['min_sales'] = historical_min * 0.8
            
            # Add regressor values for future dates if needed
            regressors = [r for r in model.extra_regressors.keys() if r != 'trend' and r != 'additive_terms' and r != 'multiplicative_terms']
            if regressors:
                logger.info("Adding regressor values to future dataframe: %s", regressors)
                
                # Handle page_views regressor for units_sold
                if target_col == 'units_sold' and 'page_views' in regressors:
                    # Get the last 4 weeks of historical page_views
                    if hasattr(self, 'original_data') and 'page_views' in self.original_data.columns:
                        recent_page_views = self.original_data['page_views'].tail(28).values
                        page_views_mean = np.mean(recent_page_views)
                        page_views_std = np.std(recent_page_views)
                        
                        # Generate future page_views with slight random variation
                        future_page_views = np.random.normal(
                            page_views_mean, 
                            page_views_std * 0.1,  # Reduced variation
                            size=len(future)
                        )
                        
                        # Normalize between 0.1 and 1
                        page_views_min = min(recent_page_views)
                        page_views_max = max(recent_page_views)
                        page_views_normalized = 0.1 + 0.9 * (future_page_views - page_views_min) / (page_views_max - page_views_min)
                        
                        future['page_views'] = np.clip(page_views_normalized, 0.1, 1.0)
                        logger.info("Added normalized future page_views with mean: %.2f", np.mean(future['page_views']))
                    else:
                        # Fallback to constant value
                        future['page_views'] = 0.5
                        logger.warning("Using default page_views value of 0.5")
            
            # Generate prediction
            future_forecast = model.predict(future)
            
            # Log the forecast values
            logger.info("Future forecast preview (first 3 days): %s", 
                      future_forecast[['ds', 'yhat']].head(3).to_dict('records'))
            logger.info("Future forecast stats - min: %.2f, max: %.2f, mean: %.2f", 
                      future_forecast['yhat'].min(), 
                      future_forecast['yhat'].max(),
                      future_forecast['yhat'].mean())
            
            # Set datetime index
            future_forecast.set_index(pd.DatetimeIndex(future_forecast['ds']), inplace=True)
            
            # Create a copy of historical forecast for combining
            combined_forecast = historical_forecast.copy()
            
            # Remove any future dates from historical forecast
            if isinstance(combined_forecast.index, pd.DatetimeIndex):
                combined_forecast = combined_forecast[combined_forecast.index <= last_historical_date]
            else:
                combined_forecast = combined_forecast[pd.to_datetime(combined_forecast['ds']) <= last_historical_date]
            
            # Combine historical and future forecasts
            forecast = pd.concat([combined_forecast, future_forecast])
            
            # Post-process the forecast
            if target_col == 'units_sold':
                # Calculate reasonable growth limits based on historical data
                historical_mean = combined_forecast['y'].mean()
                historical_std = combined_forecast['y'].std()
                historical_min = combined_forecast['y'].min()
                max_reasonable_value = historical_mean + 2 * historical_std  # More conservative limit
                
                # Get recent trend
                recent_values = combined_forecast['y'].tail(14)  # Last 2 weeks
                recent_trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                
                # Apply reasonable limits to future values
                future_mask = forecast.index > last_historical_date
                future_values = forecast.loc[future_mask, 'yhat'].values
                
                # Apply trend-aware bounds
                for i in range(len(future_values)):
                    # Allow gradual trend continuation
                    trend_adjustment = recent_trend * i * 0.5  # Dampen trend effect
                    min_bound = max(historical_min * 0.8, historical_mean - 2 * historical_std + trend_adjustment)
                    max_bound = min(max_reasonable_value, historical_mean + 2 * historical_std + trend_adjustment)
                    
                    future_values[i] = np.clip(future_values[i], min_bound, max_bound)
                
                # Update forecast with bounded values
                forecast.loc[future_mask, 'yhat'] = future_values
                
                # Adjust confidence intervals proportionally
                forecast.loc[future_mask, 'yhat_lower'] = forecast.loc[future_mask, 'yhat'] * 0.8
                forecast.loc[future_mask, 'yhat_upper'] = forecast.loc[future_mask, 'yhat'] * 1.2
                
                logger.info("Applied trend-aware bounds to forecast: min = %.2f, max = %.2f", 
                          min_bound, max_bound)
            else:
                # Ensure non-negative values for all metrics
                forecast['yhat'] = forecast['yhat'].clip(lower=0)
                forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
                forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
            
            # Store the forecast
            self.forecasts[target_col] = forecast
            
            # Get future date range for logging
            future_data = forecast[forecast.index > last_historical_date]
            future_min = future_data['yhat'].min()
            future_max = future_data['yhat'].max()
            
            logger.info("Complete forecast range: [%.2f, %.2f]", forecast['yhat'].min(), forecast['yhat'].max())
            logger.info("Future forecast range: [%.2f, %.2f]", future_min, future_max)
            logger.info("Future forecast rows: %d", len(future_data))
            
            return forecast
            
        except Exception as e:
            logger.error("Error in generate_forecast: %s", str(e))
            raise

    def get_model_metrics(self, target_col: str) -> Dict[str, float]:
        """Get model performance metrics."""
        try:
            if target_col not in self.metrics:
                raise ValueError(f"No metrics found for {target_col}")
            return self.metrics[target_col]
        except Exception as e:
            logger.error("Error in get_model_metrics: %s", str(e))
            raise

    def get_forecast_components(self, target_col: str) -> Dict[str, pd.Series]:
        """Get forecast components for visualization."""
        logger.info("Getting forecast components for %s", target_col)
        
        if target_col not in self.forecasts:
            logger.error("No forecast found for %s", target_col)
            raise ValueError(f"No forecast found for {target_col}")
            
        forecast = self.forecasts[target_col]
        components = {
            'trend': forecast['trend'],
            'weekly': forecast['weekly'] if 'weekly' in forecast.columns else None,
            'monthly': forecast['monthly'] if 'monthly' in forecast.columns else None,
            'yhat': forecast['yhat'],
            'yhat_lower': forecast['yhat_lower'],
            'yhat_upper': forecast['yhat_upper']
        }
        logger.info("Components retrieved: %s", list(components.keys()))
        return components

    def train_all_metrics(self, data: pd.DataFrame) -> None:
        """Train Prophet models in order of dependencies."""
        try:
            logger.info("Starting to train models for all metrics considering relationships")
            
            # Ensure data has a datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                data = data.set_index('date')
            
            # Train models in order of dependencies
            for metric, predictors in self.metric_relationships.items():
                if metric in data.columns:
                    logger.info("Training model for %s with predictors: %s", metric, predictors)
                    
                    # Prepare training data with regressors
                    train_data = data.copy()
                    
                    # Add regressor columns if any
                    if predictors:
                        for pred in predictors:
                            if pred in self.forecasts:
                                # Use forecasted values as regressors
                                train_data[pred] = self.forecasts[pred]['yhat'].values
                            elif pred in data.columns:
                                # Use actual values for available historical data
                                train_data[pred] = data[pred]
                            else:
                                raise ValueError(f"Predictor {pred} not found in data or forecasts")
                    
                    self.train_prophet_model(train_data, metric, predictors)
                else:
                    logger.warning("Metric %s not found in data", metric)
            
            logger.info("All models trained successfully")
            
        except Exception as e:
            logger.error("Error in train_all_metrics: %s", str(e))
            raise

    def generate_all_forecasts(self, periods: int) -> Dict[str, pd.DataFrame]:
        """Generate forecasts for all trained metrics."""
        try:
            logger.info("Generating forecasts for all metrics. Periods: %d", periods)
            all_forecasts = {}
            
            for metric in self.available_metrics:
                if metric in self.models:
                    logger.info("Generating forecast for %s", metric)
                    forecast = self.generate_forecast(metric, periods)
                    all_forecasts[metric] = forecast
                else:
                    logger.warning("No trained model found for %s", metric)
            
            return all_forecasts
            
        except Exception as e:
            logger.error("Error in generate_all_forecasts: %s", str(e))
            raise

    def format_all_forecasts(self, forecasts: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Format forecasts for all metrics into weekly periods."""
        try:
            logger.info("Formatting forecasts for all metrics")
            
            # Get the common date range for all forecasts
            all_dates = set()
            for metric, forecast in forecasts.items():
                # Get the last historical date from stored forecasts
                last_historical_date = self.forecasts[metric].index.max()
                
                # Convert forecast dates to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(forecast['ds']):
                    forecast['ds'] = pd.to_datetime(forecast['ds'])
                
                # Get future dates from the forecast
                future_dates = forecast[forecast['ds'] > last_historical_date]['ds']
                all_dates.update(future_dates)
            
            future_dates = sorted(list(all_dates))
            logger.info("Found %d future dates to forecast", len(future_dates))
            
            formatted_weeks = []
            for date in future_dates:
                week_data = {
                    'week_start': pd.to_datetime(date).strftime('%Y-%m-%d'),
                    'week_end': (pd.to_datetime(date) + pd.Timedelta(days=6)).strftime('%Y-%m-%d')
                }
                
                # Add forecasts for each metric
                for metric, forecast in forecasts.items():
                    date_mask = forecast['ds'] == pd.to_datetime(date)
                    if date_mask.any():
                        metric_row = forecast[date_mask].iloc[0]
                        
                        # Format metric values
                        week_data[f'{metric}_forecast'] = int(round(max(0, metric_row['yhat']), 0))
                        week_data[f'{metric}_lower'] = int(round(max(0, metric_row['yhat_lower']), 0))
                        week_data[f'{metric}_upper'] = int(round(max(0, metric_row['yhat_upper']), 0))
                        week_data[f'{metric}_trend'] = round(metric_row['trend'], 2)
                        
                        # Calculate week-over-week change
                        if len(formatted_weeks) > 0 and f'{metric}_forecast' in formatted_weeks[-1]:
                            prev_value = formatted_weeks[-1][f'{metric}_forecast']
                            if prev_value > 0:
                                wow_change = ((week_data[f'{metric}_forecast'] - prev_value) / prev_value) * 100
                                week_data[f'{metric}_wow_change'] = round(wow_change, 2)
                        
                        # Add seasonality if available
                        if 'weekly' in metric_row:
                            week_data[f'{metric}_weekly_seasonality'] = round(metric_row['weekly'], 2)
                        if 'yearly' in metric_row:
                            week_data[f'{metric}_yearly_seasonality'] = round(metric_row['yearly'], 2)
                
                formatted_weeks.append(week_data)
                logger.info("Processed week: %s to %s", week_data['week_start'], week_data['week_end'])
            
            return formatted_weeks
            
        except Exception as e:
            logger.error("Error in format_all_forecasts: %s", str(e))
            raise

    def _generate_initial_forecast(self, model: Prophet, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Generate and validate initial forecast."""
        try:
            # Generate initial predictions
            initial_forecast = model.predict(df)
            
            # Set datetime index
            initial_forecast.set_index(pd.to_datetime(initial_forecast['ds']), inplace=True)
            
            # Add historical values
            df_hist = df[['y']].copy()
            df_hist.index = pd.to_datetime(df['ds'])
            initial_forecast = pd.concat([initial_forecast, df_hist], axis=1)
            
            # Ensure non-negative values
            initial_forecast['yhat'] = initial_forecast['yhat'].clip(lower=0)
            initial_forecast['yhat_lower'] = initial_forecast['yhat_lower'].clip(lower=0)
            initial_forecast['yhat_upper'] = initial_forecast['yhat_upper'].clip(lower=0)
            
            # Calculate and log forecast quality metrics
            self._validate_forecast_quality(initial_forecast, target_col)
            
            # Store the forecast
            self.forecasts[target_col] = initial_forecast
            
            return initial_forecast
            
        except Exception as e:
            logger.error("Error generating initial forecast: %s", str(e))
            raise

    def _validate_forecast_quality(self, forecast: pd.DataFrame, target_col: str) -> None:
        """Validate forecast quality and log warnings for potential issues."""
        try:
            # Check for unrealistic growth
            max_historical = forecast['y'].max()
            max_forecast = forecast['yhat'].max()
            growth_rate = (max_forecast - max_historical) / max_historical * 100
            
            if abs(growth_rate) > 50:
                logger.warning(
                    "%s forecast shows %.1f%% change from historical max. "
                    "Consider adjusting changepoint_prior_scale.",
                    target_col, growth_rate
                )
            
            # Check confidence interval width
            avg_interval_width = (
                (forecast['yhat_upper'] - forecast['yhat_lower']) / forecast['yhat']
            ).mean() * 100
            
            if avg_interval_width > 50:
                logger.warning(
                    "%s forecast shows wide confidence intervals (%.1f%% of forecast). "
                    "Consider increasing seasonality_prior_scale.",
                    target_col, avg_interval_width
                )
            
            # Log forecast summary
            logger.info("Forecast validation for %s:", target_col)
            logger.info("Historical range: [%.2f, %.2f]", 
                       forecast['y'].min(), forecast['y'].max())
            logger.info("Forecast range: [%.2f, %.2f]", 
                       forecast['yhat'].min(), forecast['yhat'].max())
            logger.info("Average confidence interval: ±%.1f%%", avg_interval_width/2)
            
        except Exception as e:
            logger.error("Error validating forecast: %s", str(e)) 