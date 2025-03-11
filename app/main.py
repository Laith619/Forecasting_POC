import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_processor import DataProcessor
from forecasting import ForecastingEngine, ForecastConfig
from inventory import InventoryManager
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from arima_forecaster import ARIMAForecaster

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_forecast_plot(historical_data: pd.DataFrame, forecast_data: pd.DataFrame, target_col: str) -> go.Figure:
    """Create an interactive plot showing historical data and forecast."""
    fig = go.Figure()

    # Plot historical data
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data[target_col],
        name='Historical Data',
        mode='lines',
        line=dict(color='blue')
    ))

    # Plot forecast
    fig.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=forecast_data['yhat'],
        name='Forecast',
        mode='lines',
        line=dict(color='red')
    ))

    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=forecast_data['yhat_upper'],
        fill=None,
        mode='lines',
        line=dict(color='rgba(255,0,0,0.2)'),
        name='Upper Bound'
    ))
    fig.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=forecast_data['yhat_lower'],
        fill='tonexty',
        mode='lines',
        line=dict(color='rgba(255,0,0,0.2)'),
        name='Lower Bound'
    ))

    fig.update_layout(
        title=f'{target_col} Forecast',
        xaxis_title='Date',
        yaxis_title=target_col,
        hovermode='x unified',
        showlegend=True
    )

    return fig

def create_metrics_dashboard(metrics: Dict[str, float], model_metrics: Dict[str, float]) -> None:
    """Create a dashboard showing key performance metrics."""
    col1, col2, col3 = st.columns(3)
    
    # Display business metrics
    with col1:
        st.metric("Total GMV", f"${metrics.get('total_gmv', 0):,.2f}")
        st.metric("Average Daily Sales", f"{metrics.get('avg_daily_sales', 0):.1f} units")
    
    with col2:
        st.metric("Total Units Sold", f"{metrics.get('total_units', 0):,.0f}")
        st.metric("Average ROAS", f"{metrics.get('avg_roas', 0):.2f}x")
    
    # Only display forecast metrics if they exist in model_metrics
    with col3:
        if model_metrics and 'mape' in model_metrics and model_metrics['mape'] is not None:
            st.metric("Forecast MAPE", f"{model_metrics['mape']*100:.1f}%")
        else:
            st.metric("Forecast MAPE", "N/A")
            
        if model_metrics and 'rmse' in model_metrics and model_metrics['rmse'] is not None:
            st.metric("Forecast RMSE", f"{model_metrics['rmse']:.2f}")
        else:
            st.metric("Forecast RMSE", "N/A")

def create_inventory_dashboard(inventory_metrics: dict):
    """Create an inventory management dashboard."""
    cols = st.columns(3)
    
    # Format inventory metrics
    formatted_metrics = {
        'Current Stock': f"{inventory_metrics['current_stock']:,.0f}",
        'Safety Stock': f"{inventory_metrics['safety_stock']:,.0f}",
        'Reorder Point': f"{inventory_metrics['reorder_point']:,.0f}",
        'Days to Stockout': f"{inventory_metrics['days_to_stockout']:.0f}",
        'Risk Level': inventory_metrics['risk_level'],
        'Suggested Order': f"{inventory_metrics['suggested_order']:,.0f}"
    }
    
    # Display metrics in columns
    for i, (metric, value) in enumerate(formatted_metrics.items()):
        with cols[i % 3]:
            st.metric(metric, value)

def display_weekly_forecast(weekly_forecasts: List[Dict[str, Any]]) -> None:
    """Display weekly forecast in an expandable format."""
    st.subheader("Weekly Forecast Breakdown")
    
    for week in weekly_forecasts:
        with st.expander(f"Week of {week['week_start']} to {week['week_end']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Forecasted Units", f"{week['forecasted_units']:.0f}")
                st.metric("Week-over-Week Change", f"{week['wow_change']}%")
                
            with col2:
                st.metric("Lower Bound", f"{week['lower_bound']:.0f}")
                st.metric("Upper Bound", f"{week['upper_bound']:.0f}")
                
            st.write("**Forecast Interpretation:**")
            st.write(week['interpretation'])
            
            if week['weekly_seasonality'] != 0:
                st.write(f"Weekly Seasonal Impact: {week['weekly_seasonality']:.2%}")
            if week['monthly_seasonality'] != 0:
                st.write(f"Monthly Seasonal Impact: {week['monthly_seasonality']:.2%}")

def create_metric_plot(forecast_data: dict, metric_name: str) -> go.Figure:
    """Create a plot for a single metric's forecast."""
    fig = go.Figure()
    
    # Add historical data
    historical = forecast_data[metric_name]
    historical_dates = historical[historical['y'].notna()]['ds']
    fig.add_trace(go.Scatter(
        x=historical_dates,
        y=historical[historical['y'].notna()]['y'],
        name='Historical',
        mode='lines+markers'
    ))
    
    # Add forecast
    future_dates = historical[historical['ds'] > historical_dates.max()]['ds']
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=historical[historical['ds'] > historical_dates.max()]['yhat'],
        name='Forecast',
        mode='lines',
        line=dict(dash='dash')
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=historical[historical['ds'] > historical_dates.max()]['yhat_upper'],
        fill=None,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=historical[historical['ds'] > historical_dates.max()]['yhat_lower'],
        fill='tonexty',
        mode='lines',
        line=dict(width=0),
        name='95% Confidence'
    ))
    
    fig.update_layout(
        title=f'{metric_name.replace("_", " ").title()} Forecast',
        xaxis_title='Date',
        yaxis_title=metric_name.replace('_', ' ').title(),
        hovermode='x unified'
    )
    return fig

def ensure_metrics(forecast_df, target_col):
    """Ensure metrics are calculated and available for display.
    
    Args:
        forecast_df (pd.DataFrame): DataFrame with historical and forecasted values
        target_col (str): Target column name
        
    Returns:
        dict: Dictionary of metrics
    """
    try:
        # Create default metrics dictionary
        metrics = {
            'mape': 0,
            'rmse': 0,
            'mean_error': 0,
            'mean_absolute_error': 0
        }
        
        # Extract historical data points
        if 'y' in forecast_df.columns:
            historical_mask = forecast_df['y'].notna()
            if historical_mask.any():
                y_true = forecast_df.loc[historical_mask, 'y'].values
                y_pred = forecast_df.loc[historical_mask, 'yhat'].values
                
                # Only calculate if we have enough data
                if len(y_true) > 0 and len(y_pred) > 0:
                    # MAPE calculation
                    try:
                        # Filter out zeros to avoid division by zero
                        nonzero_mask = y_true != 0
                        if nonzero_mask.any():
                            filtered_y_true = y_true[nonzero_mask]
                            filtered_y_pred = y_pred[nonzero_mask]
                            metrics['mape'] = mean_absolute_percentage_error(filtered_y_true, filtered_y_pred)
                        else:
                            metrics['mape'] = 0
                    except Exception as e:
                        logging.warning(f"Error calculating MAPE: {str(e)}")
                        metrics['mape'] = 0
                    
                    # RMSE calculation
                    try:
                        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
                    except Exception as e:
                        logging.warning(f"Error calculating RMSE: {str(e)}")
                        metrics['rmse'] = 0
                    
                    # Mean error
                    try:
                        metrics['mean_error'] = np.mean(y_true - y_pred)
                    except Exception as e:
                        logging.warning(f"Error calculating Mean Error: {str(e)}")
                        metrics['mean_error'] = 0
                    
                    # MAE
                    try:
                        metrics['mean_absolute_error'] = np.mean(np.abs(y_true - y_pred))
                    except Exception as e:
                        logging.warning(f"Error calculating MAE: {str(e)}")
                        metrics['mean_absolute_error'] = 0
        
        return metrics
    
    except Exception as e:
        logging.error(f"Error ensuring metrics: {str(e)}")
        # Return empty metrics as fallback
        return {
            'mape': 0,
            'rmse': 0,
            'mean_error': 0,
            'mean_absolute_error': 0
        }

def create_model_metrics_dashboard(metrics_dict: Dict[str, Dict[str, float]]) -> None:
    """Create a dashboard showing metrics for all models."""
    if not metrics_dict:
        st.warning("No model metrics available")
        return
    
    # Create columns for each model
    cols = st.columns(len(metrics_dict))
    
    for i, (model_name, metrics) in enumerate(metrics_dict.items()):
        with cols[i]:
            st.subheader(f"{model_name} Metrics")
            
            # Display each metric
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    # Format the metric name
                    display_name = metric_name.upper() if metric_name.isupper() else metric_name.replace('_', ' ').title()
                    
                    # Format the value
                    if 'percentage' in metric_name.lower() or metric_name.lower() == 'mape':
                        formatted_value = f"{value:.2f}%"
                    elif metric_name.lower() in ['aic', 'bic']:
                        formatted_value = f"{value:.1f}"
                    else:
                        formatted_value = f"{value:.2f}"
                    
                    # Display metric
                    st.metric(
                        label=display_name,
                        value=formatted_value
                    )

def main():
    try:
        st.title("E-commerce Forecasting Dashboard")
        st.write("Configure your forecast settings and upload data to generate predictions")
        
        # Initialize processors
        data_processor = DataProcessor()
        
        # Sidebar for configuration
        st.sidebar.header("Forecast Configuration")
        
        # Model selection
        st.sidebar.subheader("Model Selection")
        use_prophet = st.sidebar.checkbox("Prophet", value=True, 
                                      help="Facebook's Prophet model for time series forecasting")
        use_arima = st.sidebar.checkbox("ARIMA", value=False,
                                     help="Auto-regressive Integrated Moving Average model")
        
        # Prophet configuration
        if use_prophet:
            with st.sidebar.expander("Prophet Configuration", expanded=False):
                seasonality_mode = st.selectbox(
                    "Seasonality Mode",
                    options=["multiplicative", "additive"],
                    index=0,
                    help="How seasonal patterns combine with the trend"
                )
                
                growth = st.selectbox(
                    "Growth Model",
                    options=["linear", "logistic"],
                    index=0,
                    help="Shape of the growth trend"
                )
                
                changepoint_prior_scale = st.slider(
                    "Changepoint Prior Scale",
                    min_value=0.001, max_value=0.5, 
                    value=0.05, step=0.001,
                    help="Controls trend flexibility"
                )
                
                seasonality_prior_scale = st.slider(
                    "Seasonality Prior Scale",
                    min_value=0.01, max_value=10.0, 
                    value=10.0, step=0.01,
                    help="Controls strength of seasonality"
                )
        
        # ARIMA configuration
        if use_arima:
            with st.sidebar.expander("ARIMA Configuration", expanded=False):
                auto_arima = st.checkbox(
                    "Auto-select parameters",
                    value=True,
                    help="Automatically find optimal ARIMA parameters"
                )
                
                if not auto_arima:
                    p = st.slider("AR order (p)", 0, 5, 1)
                    d = st.slider("Differencing (d)", 0, 2, 1)
                    q = st.slider("MA order (q)", 0, 5, 1)
                    seasonal = st.checkbox("Include seasonality", value=True)
        
        # Forecast horizon
        forecast_periods = st.sidebar.slider(
            "Forecast Horizon (weeks)",
            min_value=1,
            max_value=52,
            value=8,
            help="Number of weeks to forecast"
        )
        
        # File uploader
        uploaded_file = st.file_uploader("Upload CSV data", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # Process data
                with st.spinner("Processing uploaded data..."):
                    data = data_processor.process_data(uploaded_file)
                
                if data is not None:
                    st.success("Data processed successfully!")
                    
                    # Add item selection if Item Id column exists
                    selected_item = None
                    if 'Item Id' in data.columns:
                        st.sidebar.subheader("Product Selection")
                        unique_items = data['Item Id'].unique()
                        selected_item = st.sidebar.selectbox(
                            "Choose a product to forecast:",
                            options=unique_items,
                            format_func=lambda x: f"Item {x}",
                            help="Select a specific product to generate forecasts for"
                        )
                        
                        # Filter data for selected item
                        if selected_item is not None:
                            data = data[data['Item Id'] == selected_item].copy()
                            st.write(f"Showing data for Product ID: **{selected_item}**")
                    
                    # Calculate and display metrics
                    metrics = data_processor.get_metrics()
                    
                    st.header("Performance Metrics")
                    create_metrics_dashboard(metrics, {})
                    
                    # Display raw data
                    with st.expander("Raw Data"):
                        st.dataframe(data)
                    
                    # Generate forecasts
                    if st.button("Generate Forecast"):
                        with st.spinner("Training models and generating forecasts..."):
                            try:
                                all_forecasts = {}
                                all_metrics = {}
                                
                                # Configure Prophet
                                if use_prophet:
                                    config = ForecastConfig(
                                        seasonality_mode=seasonality_mode,
                                        growth=growth,
                                        changepoint_prior_scale=changepoint_prior_scale,
                                        seasonality_prior_scale=seasonality_prior_scale
                                    )
                                    
                                    # Initialize Prophet forecasting engine
                                    prophet_engine = ForecastingEngine(config)
                                    
                                    # Train Prophet model
                                    st.info("Training Prophet model...")
                                    prophet_forecast = prophet_engine.train_prophet_model(
                                        data, 
                                        'units_sold',
                                        regressor_cols=['page_views'] if 'page_views' in data.columns else None
                                    )
                                    
                                    # Generate Prophet forecast
                                    prophet_future = prophet_engine.generate_forecast('units_sold', forecast_periods)
                                    all_forecasts['Prophet'] = prophet_future
                                    all_metrics['Prophet'] = prophet_engine.get_model_metrics('units_sold')
                                
                                # Configure ARIMA
                                if use_arima:
                                    # Initialize ARIMA forecaster
                                    arima_forecaster = ARIMAForecaster()
                                    
                                    # Train ARIMA model
                                    st.info("Training ARIMA model...")
                                    arima_forecaster.train(
                                        data,
                                        'units_sold',
                                        exog_cols=['page_views'] if 'page_views' in data.columns else None
                                    )
                                    
                                    # Generate ARIMA forecast
                                    arima_future = arima_forecaster.predict(forecast_periods)
                                    all_forecasts['ARIMA'] = arima_future
                                    all_metrics['ARIMA'] = arima_forecaster.get_metrics()
                                
                                # Display results
                                st.header("Forecast Results")
                                
                                # Create tabs for different models
                                if all_forecasts:
                                    tabs = st.tabs(list(all_forecasts.keys()))
                                    
                                    for tab, (model_name, forecast) in zip(tabs, all_forecasts.items()):
                                        with tab:
                                            # Create and display plot
                                            fig = create_forecast_plot(
                                                historical_data=data,
                                                forecast_data=forecast,
                                                target_col='units_sold'
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                            # Display weekly breakdown
                                            st.subheader("Weekly Forecast Breakdown")
                                            if model_name == 'Prophet':
                                                weekly_forecast = prophet_engine.format_weekly_forecast(
                                                    forecast,
                                                    'Units Sold'
                                                )
                                            else:  # ARIMA
                                                weekly_forecast = prophet_engine.format_weekly_forecast(
                                                    forecast,
                                                    'Units Sold'
                                                )
                                            
                                            if weekly_forecast:
                                                df_weekly = pd.DataFrame(weekly_forecast)
                                                st.dataframe(df_weekly)
                                            else:
                                                st.warning("No weekly forecast data available")
                                    
                                    # Display metrics comparison
                                    st.header("Model Performance Comparison")
                                    create_model_metrics_dashboard(all_metrics)
                                else:
                                    st.warning("No forecasts generated. Please select at least one model.")
                                
                            except Exception as e:
                                st.error(f"Error generating forecasts: {str(e)}")
                                logger.error(f"Error in forecast generation: {str(e)}")
                
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                logger.error(f"Error in data processing: {str(e)}")
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main() 