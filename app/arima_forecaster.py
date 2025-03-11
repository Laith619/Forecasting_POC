import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from typing import Dict, List, Optional, Tuple, Any
import logging
import streamlit as st

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
    
    def train(self, data: pd.DataFrame, target_col: str, exog_cols: Optional[List[str]] = None) -> None:
        """Train ARIMA or SARIMAX model on the provided data."""
        try:
            logger.info(f"Starting ARIMA model training for {target_col}")
            
            # Store training data
            self.training_data = data.copy()
            self.exog_columns = exog_cols
            
            # Prepare target variable
            y = data[target_col].values
            
            # Prepare exogenous variables if provided
            X = None
            if exog_cols and all(col in data.columns for col in exog_cols):
                X = data[exog_cols].values
                logger.info(f"Using exogenous variables: {exog_cols}")
            
            # Use auto_arima to find the best parameters
            with st.spinner("Finding optimal ARIMA parameters..."):
                autoarima_model = pm.auto_arima(
                    y,
                    X=X,
                    seasonal=True,  # Enable seasonality
                    m=12,  # Monthly seasonality instead of yearly
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
            
            # Train the model with optimal parameters
            with st.spinner("Training ARIMA model with optimal parameters..."):
                self.model = SARIMAX(
                    y,
                    exog=X,
                    order=self.model_order,
                    seasonal_order=self.seasonal_order
                )
                
                self.fitted_model = self.model.fit(disp=False)
            
            logger.info("ARIMA model training completed")
            
            # Store in-sample predictions
            self.forecast_results = pd.DataFrame({
                'ds': data.index,
                'y': y,
                'yhat': self.fitted_model.fittedvalues
            })
            
            # Add prediction intervals
            forecast_errors = self.fitted_model.resid
            error_std = forecast_errors.std()
            self.forecast_results['yhat_lower'] = self.forecast_results['yhat'] - 1.96 * error_std
            self.forecast_results['yhat_upper'] = self.forecast_results['yhat'] + 1.96 * error_std
            
            # Ensure non-negative values
            for col in ['yhat', 'yhat_lower', 'yhat_upper']:
                self.forecast_results[col] = self.forecast_results[col].clip(lower=0)
            
            # Set datetime index
            self.forecast_results.set_index(pd.DatetimeIndex(self.forecast_results['ds']), inplace=True)
            
        except Exception as e:
            logger.error(f"Error training ARIMA model: {str(e)}")
            raise
    
    def predict(self, future_periods: int, exog_future: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Generate forecasts for future periods."""
        try:
            if self.fitted_model is None:
                raise ValueError("Model not trained. Call train() first.")
            
            logger.info(f"Generating forecast for {future_periods} periods")
            
            # Generate forecast
            if exog_future is not None:
                forecast = self.fitted_model.forecast(steps=future_periods, exog=exog_future)
            else:
                forecast = self.fitted_model.forecast(steps=future_periods)
            
            # Create future dates
            last_date = self.training_data.index.max()
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=7),  # Weekly data
                periods=future_periods,
                freq='W-MON'
            )
            
            # Create forecast dataframe
            future_df = pd.DataFrame({
                'ds': future_dates,
                'yhat': forecast
            })
            
            # Add prediction intervals
            forecast_errors = self.fitted_model.resid
            error_std = forecast_errors.std()
            future_df['yhat_lower'] = future_df['yhat'] - 1.96 * error_std
            future_df['yhat_upper'] = future_df['yhat'] + 1.96 * error_std
            
            # Ensure non-negative values
            for col in ['yhat', 'yhat_lower', 'yhat_upper']:
                future_df[col] = future_df[col].clip(lower=0)
            
            # Set datetime index
            future_df.set_index(pd.DatetimeIndex(future_df['ds']), inplace=True)
            
            # Add trend component (simple linear trend)
            future_df['trend'] = np.linspace(
                self.forecast_results['yhat'].iloc[-1],
                future_df['yhat'].iloc[-1],
                len(future_df)
            )
            
            # Combine with historical results
            complete_forecast = pd.concat([self.forecast_results, future_df])
            
            logger.info("Forecast generated successfully")
            return complete_forecast
            
        except Exception as e:
            logger.error(f"Error generating ARIMA forecast: {str(e)}")
            raise
    
    def get_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics for the ARIMA model."""
        try:
            if self.fitted_model is None:
                raise ValueError("Model not trained. Call train() first.")
            
            # Get residuals
            residuals = self.fitted_model.resid
            y_true = self.training_data['y'].values
            y_pred = y_true - residuals
            
            # Calculate metrics
            mse = np.mean(residuals ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(residuals))
            
            # Calculate MAPE (avoid division by zero)
            nonzero_mask = y_true != 0
            mape = np.mean(np.abs(residuals[nonzero_mask] / y_true[nonzero_mask])) * 100
            
            # Get AIC and BIC
            aic = self.fitted_model.aic
            bic = self.fitted_model.bic
            
            metrics = {
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'aic': aic,
                'bic': bic
            }
            
            logger.info("Model metrics calculated: %s", metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating ARIMA metrics: {str(e)}")
            return {
                'rmse': np.nan,
                'mae': np.nan,
                'mape': np.nan,
                'aic': np.nan,
                'bic': np.nan
            }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get a summary of the model configuration and performance."""
        return {
            'model_order': self.model_order,
            'seasonal_order': self.seasonal_order,
            'exog_variables': self.exog_columns,
            'metrics': self.get_metrics() if self.fitted_model is not None else None,
            'training_data_points': len(self.training_data) if self.training_data is not None else 0
        } 