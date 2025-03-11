import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from typing import Dict, List, Optional, Tuple, Any
import logging
import streamlit as st
from statsmodels.tsa.seasonal import STL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ARIMAForecaster:
    """ARIMA/SARIMAX forecasting model implementation."""
    
    def __init__(self):
        self.model = None
        self.model_order = None
        self.seasonal_order = None
        self.training_data = None
        self.fitted_model = None
        self.forecast_results = None
        self.exog_columns = None
        self.original_target = None
        self.seasonal_period = 12  # Default to monthly seasonality
    
    def train(self, data: pd.DataFrame, target_col: str, exog_cols: Optional[List[str]] = None, seasonal_period: Optional[int] = None) -> None:
        """Train ARIMA or SARIMAX model on the provided data."""
        try:
            logger.info(f"Starting ARIMA model training for {target_col}")
            
            # Store training data
            self.training_data = data.copy()
            self.exog_columns = exog_cols
            
            # Set seasonal period
            if seasonal_period is not None:
                self.seasonal_period = seasonal_period
            logger.info(f"Using seasonal period: {self.seasonal_period}")
            
            # Prepare target variable
            y = data[target_col].values
            
            # Store original target for component extraction
            self.original_target = data[target_col]
            
            # Prepare exogenous variables if provided
            X = None
            if exog_cols and all(col in data.columns for col in exog_cols):
                X = data[exog_cols].values
                logger.info(f"Using exogenous variables: {exog_cols}")
            
            # Use auto_arima to find the best parameters
            with st.spinner("Finding optimal ARIMA parameters..."):
                # Determine if seasonality should be enabled based on seasonal_period
                use_seasonality = self.seasonal_period > 0
                
                autoarima_model = pm.auto_arima(
                    y,
                    X=X,
                    seasonal=use_seasonality,  # Enable seasonality if period > 0
                    m=self.seasonal_period if use_seasonality else 1,  # Use configured seasonal period
                    start_p=0, start_q=0,
                    max_p=3, max_q=3, max_d=2,
                    start_P=0, start_Q=0,
                    max_P=1, max_Q=1, max_D=1,
                    information_criterion='aic',
                    trace=True,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True,
                    seasonal_test='ch'  # Use CH test to determine if seasonality should be included
                )
            
            # Get the optimal parameters
            self.model_order = autoarima_model.order
            self.seasonal_order = autoarima_model.seasonal_order
            
            logger.info(f"Optimal ARIMA order: {self.model_order}")
            logger.info(f"Optimal seasonal order: {self.seasonal_order}")
            
            # Create and fit SARIMAX model with the optimal parameters
            model = SARIMAX(
                y,
                exog=X,
                order=self.model_order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            # Fit model
            self.model = model.fit(disp=False)
            
            # Generate in-sample predictions for historical period
            if X is not None:
                in_sample_predictions = self.model.predict(exog=X)
            else:
                in_sample_predictions = self.model.predict()
            
            # Create historical forecast dataframe
            self.forecast_results = pd.DataFrame({
                'ds': data.index,
                'y': y,
                'yhat': in_sample_predictions,
            })
            
            # Add prediction intervals
            error_std = np.std(self.model.resid)
            self.forecast_results['yhat_lower'] = self.forecast_results['yhat'] - 1.96 * error_std
            self.forecast_results['yhat_upper'] = self.forecast_results['yhat'] + 1.96 * error_std
            
            # Ensure non-negative values
            for col in ['yhat', 'yhat_lower', 'yhat_upper']:
                self.forecast_results[col] = self.forecast_results[col].clip(lower=0)
            
            # Set datetime index
            self.forecast_results.set_index(pd.DatetimeIndex(self.forecast_results['ds']), inplace=True)
            
            logger.info("ARIMA model training completed")
            
        except Exception as e:
            logger.error(f"Error training ARIMA model: {str(e)}")
            raise
    
    def predict(self, future_periods: int) -> pd.DataFrame:
        """Generate forecasts for future periods."""
        try:
            if not self.model:
                logger.error("Model not trained. Please call train method first.")
                raise ValueError("Model not trained. Please call train method first.")
            
            logger.info(f"Generating ARIMA forecast for {future_periods} periods")
            
            # Generate future dates
            last_date = self.training_data.index[-1]
            date_range = pd.date_range(start=last_date + pd.Timedelta(days=7), 
                                      periods=future_periods, 
                                      freq='W-MON')
            
            # Handle future exogenous variables if model was trained with them
            if self.exog_columns and len(self.exog_columns) > 0:
                logger.info(f"Model uses exogenous variables: {self.exog_columns}")
                
                # Generate future exogenous values
                future_exog = self._generate_future_exogenous(future_periods, date_range)
                logger.info(f"Generated future exogenous data with shape: {future_exog.shape}")
                
                # Forecast with exogenous variables
                forecast_values = self.model.forecast(steps=future_periods, exog=future_exog)
            else:
                logger.info("No exogenous variables used in forecast")
                # Forecast without exogenous variables
                forecast_values = self.model.forecast(steps=future_periods)
            
            # Create confidence intervals
            preds = forecast_values
            error_std = np.std(self.model.resid)
            
            # Create future dataframe
            future_df = pd.DataFrame({
                'ds': date_range,
                'yhat': preds,
                'yhat_lower': preds - 1.96 * error_std,
                'yhat_upper': preds + 1.96 * error_std
            })
            
            # Ensure non-negative values
            for col in ['yhat', 'yhat_lower', 'yhat_upper']:
                future_df[col] = future_df[col].clip(lower=0)
            
            # Set datetime index
            future_df.set_index(pd.DatetimeIndex(future_df['ds']), inplace=True)
            
            # Extract components
            self._extract_forecast_components(future_df)
            
            # Combine with historical results
            complete_forecast = pd.concat([self.forecast_results, future_df])
            
            logger.info("Forecast generated successfully")
            return complete_forecast
            
        except Exception as e:
            logger.error(f"Error generating ARIMA forecast: {str(e)}")
            raise
    
    def _extract_forecast_components(self, future_df: pd.DataFrame) -> None:
        """Extract trend and seasonal components from the forecast."""
        try:
            # If model has seasonal component
            if hasattr(self.model, 'seasonal_order') and self.model.seasonal_order[0] > 0:
                logger.info("Extracting seasonal components")
                
                # Get the last year of historical data to estimate seasonal pattern
                historical_data = self.original_target.iloc[-52:] if len(self.original_target) >= 52 else self.original_target
                
                # Decompose historical data to extract seasonal pattern
                # Use STL decomposition which works well with various seasonal patterns
                if len(historical_data) >= 2 * self.seasonal_period:
                    decomposition = STL(historical_data, seasonal=self.seasonal_period).fit()
                    
                    # Extract seasonal component
                    seasonal_pattern = decomposition.seasonal
                    
                    # Repeat the pattern for future periods
                    seasonal_values = []
                    for i in range(len(future_df)):
                        idx = i % len(seasonal_pattern)
                        seasonal_values.append(seasonal_pattern.iloc[idx])
                    
                    future_df['seasonal'] = seasonal_values
                else:
                    # If not enough data for STL, use a simple approach
                    future_df['seasonal'] = 0
                    logger.warning("Not enough historical data for proper seasonal decomposition")
            else:
                future_df['seasonal'] = 0
            
            # Add trend component - use the forecast values minus the seasonal component
            future_df['trend'] = future_df['yhat'] - future_df['seasonal']
            
            # Add residual component (for historical data only)
            future_df['residual'] = 0  # Future residuals are unknown
            
            logger.info("Forecast components extracted successfully")
        except Exception as e:
            logger.error(f"Error extracting forecast components: {str(e)}")
            future_df['trend'] = future_df['yhat']
            future_df['seasonal'] = 0
            future_df['residual'] = 0
            logger.warning("Using fallback components due to extraction error")
    
    def _generate_future_exogenous(self, future_periods: int, future_dates: pd.DatetimeIndex) -> np.ndarray:
        """Generate future values for exogenous variables."""
        future_exog = np.zeros((future_periods, len(self.exog_columns)))
        
        for i, col in enumerate(self.exog_columns):
            # Get historical data for this exogenous variable
            historical_values = self.training_data[col].values
            
            # Calculate mean and standard deviation
            mean_value = np.mean(historical_values[-12:])  # Use last 12 weeks
            std_value = np.std(historical_values[-12:])
            
            # For page_views, use a smoother projection based on recent trend
            if col == 'page_views':
                # Calculate recent trend (slope of linear regression)
                recent_values = historical_values[-12:]
                x = np.arange(len(recent_values))
                slope, intercept = np.polyfit(x, recent_values, 1)
                
                # Generate future values with slight random variation but following trend
                for j in range(future_periods):
                    next_value = intercept + slope * (len(recent_values) + j)
                    # Add some noise, but less than the full standard deviation
                    noise = np.random.normal(0, std_value * 0.2)
                    future_exog[j, i] = max(0, next_value + noise)
                
                logger.info(f"Generated trend-based future values for {col} with mean: {np.mean(future_exog[:, i]):.2f}")
            else:
                # For other variables, use mean with some random variation
                future_exog[:, i] = np.random.normal(mean_value, std_value * 0.2, size=future_periods)
                future_exog[:, i] = np.clip(future_exog[:, i], 0, None)  # Ensure non-negative
                
                logger.info(f"Generated random future values for {col} with mean: {np.mean(future_exog[:, i]):.2f}")
        
        return future_exog
    
    def get_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics for the ARIMA model."""
        try:
            if self.model is None:
                logger.error("Model not trained. Call train() first.")
                raise ValueError("Model not trained. Call train() first.")
            
            # Get residuals
            residuals = self.model.resid
            
            # Get actual values - ensure we have them
            if not hasattr(self, 'original_target') or self.original_target is None:
                logger.warning("Original target not available, using limited metrics")
                # Calculate metrics based only on residuals
                mse = np.mean(residuals ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(residuals))
                
                # Get AIC and BIC
                aic = self.model.aic if hasattr(self.model, 'aic') else None
                bic = self.model.bic if hasattr(self.model, 'bic') else None
                
                metrics = {
                    'rmse': rmse,
                    'mae': mae,
                    'aic': aic,
                    'bic': bic,
                    'mape': None
                }
            else:
                # Full metrics calculation with actual values
                y_true = self.original_target.values
                y_pred = y_true - residuals
                
                # Calculate metrics
                mse = np.mean(residuals ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(residuals))
                
                # Calculate MAPE (avoid division by zero)
                nonzero_mask = y_true != 0
                mape = np.mean(np.abs(residuals[nonzero_mask] / y_true[nonzero_mask])) if any(nonzero_mask) else None
                
                # Get AIC and BIC
                aic = self.model.aic if hasattr(self.model, 'aic') else None
                bic = self.model.bic if hasattr(self.model, 'bic') else None
                
                metrics = {
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape,
                    'aic': aic,
                    'bic': bic,
                    'mean_absolute_error': mae,
                    'mean_error': np.mean(residuals)
                }
            
            logger.info(f"ARIMA metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating ARIMA metrics: {str(e)}")
            return {
                'rmse': None,
                'mae': None,
                'mape': None,
                'aic': None,
                'bic': None,
                'mean_absolute_error': None,
                'mean_error': None
            }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get a summary of the ARIMA model and its parameters."""
        try:
            if self.model is None:
                logger.error("Model not trained. Please call train method first.")
                raise ValueError("Model not trained. Please call train method first.")
            
            # Get metrics
            metrics = self.get_metrics()
            
            # Create summary
            summary = {
                'order': self.model_order,
                'seasonal_order': self.seasonal_order,
                'aic': metrics.get('aic', None),
                'bic': metrics.get('bic', None),
                'mae': metrics.get('mae', None),
                'rmse': metrics.get('rmse', None),
                'mape': metrics.get('mape', None),
                'seasonal_period': self.seasonal_period
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting model summary: {str(e)}")
            return {
                'order': None,
                'seasonal_order': None,
                'aic': None,
                'bic': None,
                'error_message': str(e)
            } 