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
    
    # Define metric descriptions and formatting
    metric_info = {
        'mape': {
            'display': 'MAPE',
            'description': 'Mean Absolute Percentage Error - lower is better',
            'format': lambda x: f"{x*100:.2f}%",
            'color': lambda x: 'green' if x < 0.1 else ('orange' if x < 0.2 else 'red')
        },
        'rmse': {
            'display': 'RMSE',
            'description': 'Root Mean Squared Error - lower is better',
            'format': lambda x: f"{x:.2f}",
            'color': lambda x: 'inherit'
        },
        'mean_error': {
            'display': 'Mean Error',
            'description': 'Average error (bias) - closer to zero is better',
            'format': lambda x: f"{x:.2f}",
            'color': lambda x: 'green' if abs(x) < 1 else ('orange' if abs(x) < 3 else 'red')
        },
        'mean_absolute_error': {
            'display': 'MAE',
            'description': 'Mean Absolute Error - lower is better',
            'format': lambda x: f"{x:.2f}",
            'color': lambda x: 'inherit'
        },
        'aic': {
            'display': 'AIC',
            'description': 'Akaike Information Criterion - lower is better',
            'format': lambda x: f"{x:.2f}",
            'color': lambda x: 'inherit'
        },
        'bic': {
            'display': 'BIC',
            'description': 'Bayesian Information Criterion - lower is better',
            'format': lambda x: f"{x:.2f}",
            'color': lambda x: 'inherit'
        }
    }
    
    for i, (model_name, metrics) in enumerate(metrics_dict.items()):
        with cols[i]:
            st.subheader(f"{model_name} Metrics")
            
            # Create a styled metrics display
            metrics_html = "<div class='metrics-container'>"
            
            # Display each metric
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    # Get metric info or use defaults
                    info = metric_info.get(metric_name.lower(), {
                        'display': metric_name.upper() if metric_name.isupper() else metric_name.replace('_', ' ').title(),
                        'description': '',
                        'format': lambda x: f"{x:.2f}",
                        'color': lambda x: 'inherit'
                    })
                    
                    # Format the value
                    formatted_value = info['format'](value)
                    color = info['color'](value)
                    
                    # Add to HTML
                    metrics_html += f"""
                    <div class='metric-box'>
                        <div class='metric-name' title='{info['description']}'>{info['display']}</div>
                        <div class='metric-value' style='color: {color}'>{formatted_value}</div>
                    </div>
                    """
                    
                    # Also display as regular metric for accessibility
                    st.metric(info['display'], formatted_value)
            
            metrics_html += "</div>"
            
            # Add CSS for metrics
            st.markdown("""
            <style>
            .metrics-container {
                display: flex;
                flex-direction: column;
                gap: 10px;
                margin-bottom: 20px;
            }
            .metric-box {
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 10px;
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            }
            .metric-name {
                font-size: 0.9rem;
                color: #6c757d;
                font-weight: 500;
            }
            .metric-value {
                font-size: 1.2rem;
                font-weight: 600;
                margin-top: 5px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Display model-specific information
            if model_name == 'Prophet':
                st.info("""
                **About Prophet Metrics:**
                - MAPE below 10% indicates excellent forecast accuracy
                - Prophet excels at capturing seasonal patterns
                - Lower AIC/BIC values indicate better model fit
                """)
            elif model_name == 'ARIMA':
                st.info("""
                **About ARIMA Metrics:**
                - RMSE is in the same units as your target variable
                - AIC/BIC help compare different ARIMA configurations
                - Mean Error close to zero indicates unbiased forecasts
                """)

def create_component_plot(forecast_data, model_type):
    """Create a component plot showing trend, seasonality, etc."""
    # Identify important components to display
    important_components = []
    
    # Always include trend
    if 'trend' in forecast_data.columns:
        important_components.append('trend')
    
    # Add seasonal components
    for col in forecast_data.columns:
        if ('seasonal' in col or 'seasonality' in col) and col not in important_components:
            important_components.append(col)
            
    # Add key regressors if available (limit to top 2)
    regressor_cols = [col for col in forecast_data.columns 
                    if 'effect' in col and col not in important_components]
    if regressor_cols:
        important_components.extend(regressor_cols[:2])
    
    # Limit the total number of components to prevent plotting errors
    # 8 is a reasonable maximum to ensure proper spacing
    important_components = important_components[:8]
    
    # Create subplots with limited components
    fig = make_subplots(
        rows=len(important_components),
        cols=1,
        subplot_titles=[col.replace('_', ' ').title() for col in important_components],
        vertical_spacing=max(0.05, 1.0 / (len(important_components) * 2))  # Ensure spacing is adequate
    )
    
    row = 1
    for col in important_components:
        fig.add_trace(
            go.Scatter(
                x=forecast_data['ds'],
                y=forecast_data[col],
                mode='lines',
                name=col.replace('_', ' ').title(),
                line=dict(width=2)
            ),
            row=row, col=1
        )
        row += 1
    
    fig.update_layout(
        height=200 * len(important_components),  # Adjust height based on number of components
        width=900,
        title_text=f"{model_type} Model Components",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(title_text="Date")
    
    return fig

def main():
    try:
        # Configure page
        st.set_page_config(
            page_title="E-commerce Forecasting App", 
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Apply custom CSS
        try:
            with open('app/styles.css') as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        except FileNotFoundError:
            # If styles.css doesn't exist, use default styling
            st.markdown("""
            <style>
            h1 {color: #2c3e50;}
            .stButton > button {background-color: #2980b9; color: white;}
            </style>
            """, unsafe_allow_html=True)
        
        # Initialize processors
        data_processor = DataProcessor()
        
        # App title
        st.title("E-commerce Sales Forecasting")
        st.markdown("""
        This application helps e-commerce businesses forecast future sales by analyzing historical data.
        Upload your data, configure forecast settings, and get predictions with visual insights.
        """)
        
        # Sidebar
        with st.sidebar:
            st.header("Configuration")
            
            # File uploader
            st.subheader("1. Upload Data")
            uploaded_file = st.file_uploader(
                "Upload your e-commerce data (CSV or Excel)", 
                type=["csv", "xlsx", "xls"],
                help="Upload a file containing your time series data. The file should include a date column and at least one numeric column for forecasting."
            )
            
            if uploaded_file:
                # Data info
                file_details = {"Filename": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": f"{uploaded_file.size / 1024:.2f} KB"}
                st.write(file_details)
                
                # Data preview toggle
                show_preview = st.checkbox("Show data preview", value=True, help="Display a preview of the uploaded data")
                
                # Parameters section
                st.subheader("2. Forecast Settings")
                
                # Model selection
                st.write("Select models to use:")
                use_prophet = st.checkbox("Prophet", value=True, help="Prophet is good for data with strong seasonality patterns and can incorporate holidays")
                use_arima = st.checkbox("ARIMA", value=True, help="ARIMA is effective for stationary time series and can capture complex temporal dependencies")
                
                # Advanced model configuration with expandable section
                with st.expander("Advanced Model Configuration", expanded=False):
                    # Prophet configuration
                    st.subheader("Prophet Settings")
                    seasonality_mode = st.selectbox(
                        "Seasonality Mode",
                        options=["multiplicative", "additive"],
                        index=0,
                        help="How seasonal patterns combine with the trend. Multiplicative is better for data with increasing seasonal variations."
                    )
                    
                    growth = st.selectbox(
                        "Growth Model",
                        options=["logistic", "linear"],
                        index=0,
                        help="Shape of the growth trend. Logistic growth has a maximum cap."
                    )
                    
                    changepoint_prior_scale = st.slider(
                        "Changepoint Prior Scale",
                        min_value=0.001, max_value=0.5, 
                        value=0.05, step=0.001,
                        help="Controls trend flexibility. Higher values allow more flexibility."
                    )
                    
                    seasonality_prior_scale = st.slider(
                        "Seasonality Prior Scale",
                        min_value=0.01, max_value=10.0, 
                        value=10.0, step=0.01,
                        help="Controls strength of seasonality. Higher values allow stronger seasonal patterns."
                    )
                    
                    # ARIMA configuration
                    st.subheader("ARIMA Settings")
                    seasonal_period = st.slider(
                        "Seasonal Period",
                        min_value=0, max_value=52, 
                        value=12, step=1,
                        help="Length of seasonal cycle (0 for non-seasonal). For weekly data, use 52 for annual seasonality or 12 for quarterly."
                    )
                
                # Forecast horizon
                forecast_periods = st.slider(
                    "Forecast Horizon (weeks)", 
                    min_value=1, 
                    max_value=52, 
                    value=8,
                    help="Number of weeks to forecast into the future. Longer horizons typically have higher uncertainty."
                )
                
                # Target variable
                target_col = st.selectbox(
                    "Target Variable",
                    options=["units_sold"],
                    help="The variable you want to forecast"
                )
                
                # Exogenous variable
                use_exog = st.checkbox("Use Page Views as Additional Feature", value=True, help="Include page views as an external variable that may influence sales")
                
                # Execute forecast button
                forecast_button = st.button("Generate Forecast", use_container_width=True, help="Click to generate forecasts using the selected models")
                
                # Add information about models
                with st.expander("📚 About the Models"):
                    st.markdown("""
                    ### Prophet
                    Facebook's Prophet is designed for forecasting time series with strong seasonal effects and several seasons of historical data. It works best with daily data that has at least a few months of history.
                    
                    **Key Features:**
                    - Handles missing data and outliers
                    - Automatically detects changes in trends
                    - Models multiple seasonalities (daily, weekly, yearly)
                    - Can incorporate holiday effects
                    
                    ### ARIMA
                    Auto Regressive Integrated Moving Average is a traditional statistical method that works well on stationary time series data.
                    
                    **Key Features:**
                    - Captures temporal dependencies in time series
                    - Works well with data that shows constant statistical properties over time
                    - Can incorporate exogenous variables (ARIMAX)
                    - Can handle seasonal patterns (SARIMAX)
                    """)
            
                # Add help information
                with st.expander("❓ Need Help?"):
                    st.markdown("""
                    ### How to Use This App
                    
                    1. **Upload Your Data**: Use the file uploader to import your time series data (CSV or Excel).
                    2. **Configure Settings**: Select models and forecast horizon.
                    3. **Generate Forecast**: Click the button to run the models.
                    4. **Explore Results**: View forecasts, component breakdowns, and model performance metrics.
                    
                    ### Data Format Requirements
                    
                    Your data should include:
                    - A date column (daily or weekly)
                    - A numeric target column (e.g., units_sold)
                    - Optional: Additional features (e.g., page_views)
                    
                    ### Understanding the Results
                    
                    - **Forecast Plot**: Shows historical data and future predictions with confidence intervals
                    - **Components**: Breaks down the forecast into trend and seasonal factors
                    - **Model Performance**: Metrics like MAPE, RMSE to evaluate accuracy
                    """)
            
        # Main content area
        if uploaded_file is not None:
            try:
                # Process data
                with st.spinner("Processing uploaded data..."):
                    try:
                        data = data_processor.process_data(uploaded_file)
                        st.success("Data processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing data: {str(e)}")
                        logger.error(f"Data processing error: {str(e)}")
                        st.stop()
                
                if data is not None:
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
                    try:
                        metrics = data_processor.get_metrics()
                        
                        st.header("Performance Metrics")
                        create_metrics_dashboard(metrics, {})
                    except Exception as e:
                        st.warning(f"Could not calculate metrics: {str(e)}")
                        logger.warning(f"Metrics calculation error: {str(e)}")
                    
                    # Display raw data
                    with st.expander("Raw Data"):
                        st.dataframe(data)
                    
                    # Generate forecasts
                    if forecast_button:
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
                                
                                # Train ARIMA model if selected
                                if use_arima:
                                    # Initialize ARIMA forecaster
                                    arima_forecaster = ARIMAForecaster()
                                    
                                    # Train with spinner and error handling
                                    with st.spinner("Training ARIMA model..."):
                                        try:
                                            st.info("Training ARIMA model...")
                                            # Use the seasonal_period from configuration if defined
                                            exog_cols = ['page_views'] if use_exog and 'page_views' in data.columns else None
                                            
                                            # Configure ARIMA parameters
                                            arima_params = {
                                                "seasonal_period": seasonal_period
                                            }
                                            
                                            # Log exogenous variables used
                                            if exog_cols:
                                                st.write(f"Using {', '.join(exog_cols)} as exogenous variables")
                                                logger.info(f"Using exogenous variables: {exog_cols}")
                                            
                                            # Train the model
                                            arima_forecaster.train(
                                                data, 
                                                target_col, 
                                                exog_cols, 
                                                seasonal_period=seasonal_period
                                            )
                                            
                                            # Display model summary on success
                                            st.success("ARIMA model trained successfully!")
                                            model_summary = arima_forecaster.get_model_summary()
                                            st.write(f"Model Order: {model_summary.get('order', 'Not available')}")
                                            st.write(f"Seasonal Order: {model_summary.get('seasonal_order', 'Not available')}")
                                            
                                            # Generate forecast
                                            arima_future = arima_forecaster.predict(forecast_periods)
                                            all_forecasts['ARIMA'] = arima_future
                                            
                                            # Get metrics
                                            all_metrics['ARIMA'] = arima_forecaster.get_metrics()
                                            
                                        except Exception as e:
                                            st.error(f"Error training ARIMA model: {str(e)}")
                                            logger.error(f"ARIMA training error: {str(e)}")
                                            st.warning("Falling back to Prophet model only")
                                
                                # Display results
                                st.header("Forecast Results")
                                
                                # Create tabs for different models
                                if all_forecasts:
                                    tabs = st.tabs(list(all_forecasts.keys()))
                                    
                                    for tab, (model_name, forecast) in zip(tabs, all_forecasts.items()):
                                        with tab:
                                            # Create subtabs for forecast and components
                                            forecast_tab, components_tab, data_tab = st.tabs(["Forecast", "Components", "Data"])
                                            
                                            with forecast_tab:
                                                # Create and display plot
                                                fig = create_forecast_plot(
                                                    historical_data=data,
                                                    forecast_data=forecast,
                                                    target_col='units_sold'
                                                )
                                                st.plotly_chart(fig, use_container_width=True)
                                            
                                            with components_tab:
                                                st.subheader("Model Components")
                                                
                                                # Display component explanation based on model
                                                if model_name == 'Prophet':
                                                    st.info("""
                                                    **Understanding the Components:**
                                                    - **Trend**: The overall upward or downward movement of the time series over time
                                                    - **Weekly Seasonality**: Repeated patterns that occur on a weekly basis
                                                    - **Yearly Seasonality**: Annual patterns in the data
                                                    - **Holidays**: Effects of holidays or special events (if configured)
                                                    
                                                    These components help you understand what's driving your forecast. If one component shows stronger effects, it might indicate where to focus your business strategies.
                                                    """)
                                                elif model_name == 'ARIMA':
                                                    st.info("""
                                                    **Understanding the Components:**
                                                    - **Trend**: The overall direction of the time series
                                                    - **Seasonal**: Repeating patterns at regular intervals
                                                    - **Residual**: The unexplained variation in the data
                                                    
                                                    ARIMA models break down your data into these components to capture different aspects of the time series pattern. Understanding each component can help identify what factors are most influential in your sales patterns.
                                                    """)
                                                
                                                # Check if forecast contains component columns
                                                component_columns = [col for col in forecast.columns 
                                                                   if col not in ['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']]
                                                
                                                if component_columns:
                                                    comp_fig = create_component_plot(forecast, model_name)
                                                    st.plotly_chart(comp_fig, use_container_width=True)
                                                else:
                                                    st.warning(f"No component breakdown available for {model_name} model")
                                            
                                            with data_tab:
                                                st.subheader("Forecast Data")
                                                st.markdown("""
                                                This table shows the weekly forecast values with confidence intervals. You can download this data for further analysis or reporting.
                                                
                                                - **Week Start/End**: The time period for the forecast
                                                - **Forecasted Units**: The predicted value
                                                - **Lower/Upper Bound**: 95% confidence interval (the range where the actual value is likely to fall)
                                                - **WoW Change**: Week-over-week percentage change
                                                """)
                                                # Format weekly forecast data for display
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
                                    st.markdown("""
                                    These metrics help you evaluate and compare the accuracy of different forecasting models:
                                    
                                    - **MAPE**: Mean Absolute Percentage Error - lower is better, shows average % error
                                    - **RMSE**: Root Mean Squared Error - lower is better, penalizes large errors
                                    - **AIC/BIC**: Information criteria used for model selection - lower is better
                                    """)
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