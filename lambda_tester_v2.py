"""
Time Series Forecasting Script
This script performs time series forecasting on weekly sales data using Prophet.
It loads data from an Excel file, processes it, trains a forecasting model,
and evaluates the predictions.
"""

# Import required libraries
import os
import json
import pytz
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from prophet import Prophet
import matplotlib.pyplot as plt

# Set up logging configuration
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def load_and_prepare_data():
    """
    Load weekly data from Excel file and prepare it for analysis
    Returns:
        pandas.DataFrame: Processed weekly data
    """
    # Load the weekly data from Excel file
    df_weekly = pd.read_excel("Sample_weekly_data.xlsx")
    
    print("Dataset shape:", df_weekly.shape)
    print("\nFirst few rows of the data:")
    print(df_weekly.head())
    
    return df_weekly

def prepare_prophet_data(df):
    """
    Prepare data in the format required by Prophet
    Args:
        df (pandas.DataFrame): Input dataframe with weekly data
    Returns:
        pandas.DataFrame: Formatted data for Prophet
    """
    # Prophet requires columns named 'ds' (date) and 'y' (target variable)
    prophet_df = df[['Week Start', 'Units Sold']].copy()
    prophet_df.columns = ['ds', 'y']
    return prophet_df

def train_prophet_model(data):
    """
    Train Prophet forecasting model
    Args:
        data (pandas.DataFrame): Training data in Prophet format
    Returns:
        tuple: (Prophet model, DataFrame with predictions)
    """
    # Initialize and train Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    
    model.fit(data)
    
    # Create future dates for forecasting
    future_dates = model.make_future_dataframe(periods=12, freq='W')
    
    # Make predictions
    forecast = model.predict(future_dates)
    
    return model, forecast

def evaluate_model(actual, predicted):
    """
    Calculate model performance metrics
    Args:
        actual (array-like): Actual values
        predicted (array-like): Predicted values
    Returns:
        dict: Dictionary containing performance metrics
    """
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    r2 = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
    
    return {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 2)
    }

def plot_forecast(model, forecast, actual_data):
    """
    Plot the forecasting results
    Args:
        model (Prophet): Trained Prophet model
        forecast (pandas.DataFrame): Forecast results
        actual_data (pandas.DataFrame): Original data
    """
    plt.figure(figsize=(10, 6))
    
    # Convert dates directly using pandas datetime objects
    forecast_dates = pd.to_datetime(forecast['ds'])
    actual_dates = pd.to_datetime(actual_data['ds'])
    
    # Plot actual values
    plt.plot(actual_dates, actual_data['y'], 'b.', label='Actual')
    
    # Plot predicted values and confidence intervals
    plt.plot(forecast_dates, forecast['yhat'], 'r-', label='Predicted')
    plt.fill_between(forecast_dates, 
                    forecast['yhat_lower'], 
                    forecast['yhat_upper'], 
                    color='r', 
                    alpha=0.1, 
                    label='Confidence Interval')
    
    plt.title('Time Series Forecast')
    plt.xlabel('Date')
    plt.ylabel('Units Sold')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('forecast_plot.png')
    plt.close()

def main():
    """
    Main function to orchestrate the forecasting process
    """
    # Load and prepare data
    df_weekly = load_and_prepare_data()
    
    # Prepare data for Prophet
    prophet_data = prepare_prophet_data(df_weekly)
    
    # Train model and make predictions
    model, forecast = train_prophet_model(prophet_data)
    
    # Evaluate model performance
    metrics = evaluate_model(
        prophet_data['y'].values,
        forecast['yhat'][:len(prophet_data)]
    )
    
    # Print performance metrics
    print("\nModel Performance Metrics:")
    print(f"Mean Absolute Error: {metrics['MAE']}")
    print(f"Root Mean Square Error: {metrics['RMSE']}")
    print(f"R-squared Score: {metrics['R2']}")
    
    # Create visualization
    plot_forecast(model, forecast, prophet_data)
    
    print("\nForecast plot has been saved as 'forecast_plot.png'")

if __name__ == "__main__":
    main()
