import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import io
import base64
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Time Series Forecasting Tool",
    page_icon="📈",
    layout="wide"
)

# Custom CSS to enhance the UI
st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    .title {
        font-size: 2.5rem;
        color: #4B8BBE;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.5rem;
        color: #306998;
        margin-bottom: 0.5rem;
    }
    .text-info {
        font-size: 1rem;
        color: #333333;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown("<div class='title'>Time Series Forecasting Application</div>", unsafe_allow_html=True)
st.markdown("""
This application helps you forecast future values based on your time series data.
Upload your Excel file, and the app will generate predictions using Prophet, a forecasting model developed by Facebook.
""")

# Functions from lambda_tester_v2.py
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

def create_forecast_plot(model, forecast, actual_data):
    """
    Create the forecasting plot
    Args:
        model (Prophet): Trained Prophet model
        forecast (pandas.DataFrame): Forecast results
        actual_data (pandas.DataFrame): Original data
    Returns:
        matplotlib.figure.Figure: Plot figure
    """
    fig = plt.figure(figsize=(12, 8))
    
    # Convert dates using pandas datetime objects
    forecast_dates = pd.to_datetime(forecast['ds'])
    actual_dates = pd.to_datetime(actual_data['ds'])
    
    # Plot actual values
    plt.plot(actual_dates, actual_data['y'], 'bo', label='Actual')
    
    # Plot predicted values and confidence intervals
    plt.plot(forecast_dates, forecast['yhat'], 'r-', label='Predicted')
    plt.fill_between(forecast_dates, 
                    forecast['yhat_lower'], 
                    forecast['yhat_upper'], 
                    color='r', 
                    alpha=0.1, 
                    label='Confidence Interval (95%)')
    
    plt.title('Time Series Forecast', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Units Sold', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def get_download_link(fig, filename="forecast_plot.png"):
    """
    Generate a download link for the plot
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode the image to base64
    img_str = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">Download Plot</a>'
    return href

def get_components_plot(model):
    """Create a plot showing the components of the forecast"""
    fig = plt.figure(figsize=(12, 10))
    model.plot_components(model.predict(model.history))
    plt.tight_layout()
    return fig

def explain_metrics(metrics):
    """
    Generate explanations for the metrics
    """
    explanations = {
        'MAE': f"""
        **Mean Absolute Error (MAE): {metrics['MAE']}**  
        This represents the average absolute difference between predicted and actual values.
        Lower values indicate better performance. In practical terms, on average, the predictions
        are off by about {metrics['MAE']} units from the actual values.
        """,
        
        'RMSE': f"""
        **Root Mean Square Error (RMSE): {metrics['RMSE']}**  
        This is similar to MAE but gives higher weight to larger errors. It's the square root of the
        average of squared differences between prediction and actual values.
        A lower RMSE indicates better fit. The value of {metrics['RMSE']} gives more emphasis to 
        larger prediction errors than MAE.
        """,
        
        'R2': f"""
        **R-squared (R²): {metrics['R2']}**  
        This indicates how well the model explains the variance in the data.
        An R² value of 1 means perfect prediction, 0 means the model just predicts the mean of the data.
        Negative values (like {metrics['R2']}) suggest that the model is performing worse than just using
        the mean as a prediction. This might indicate that there's high volatility in the data or that additional
        factors might be needed to improve prediction.
        """
    }
    return explanations

# Main application logic
def main():
    # File upload section
    st.markdown("<div class='subtitle'>Upload Your Data</div>", unsafe_allow_html=True)
    st.write("Please upload an Excel file with time series data. The file should have columns 'Week Start' and 'Units Sold'.")
    
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
    
    sample_data_checkbox = st.checkbox("Use sample data instead")
    
    if sample_data_checkbox:
        # Use the existing Sample_weekly_data.xlsx
        if os.path.exists("Sample_weekly_data.xlsx"):
            df_weekly = pd.read_excel("Sample_weekly_data.xlsx")
            st.success("Sample data loaded successfully!")
        else:
            st.error("Sample data file not found. Please upload your own file.")
            return
    elif uploaded_file is not None:
        try:
            df_weekly = pd.read_excel(uploaded_file)
            st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error: {e}")
            return
    else:
        # Show waiting message
        st.info("Please upload a file or use the sample data to proceed.")
        return
    
    # Display the data
    st.markdown("<div class='subtitle'>Data Preview</div>", unsafe_allow_html=True)
    st.write(f"Dataset shape: {df_weekly.shape}")
    st.dataframe(df_weekly.head())
    
    # Check if required columns exist
    if 'Week Start' not in df_weekly.columns or 'Units Sold' not in df_weekly.columns:
        st.error("Error: The uploaded file must contain 'Week Start' and 'Units Sold' columns.")
        return
    
    # Process the data
    with st.spinner("Processing data and generating forecast..."):
        # Prepare data for Prophet
        prophet_data = prepare_prophet_data(df_weekly)
        
        # Train model and make predictions
        model, forecast = train_prophet_model(prophet_data)
        
        # Evaluate model
        metrics = evaluate_model(
            prophet_data['y'].values,
            forecast['yhat'][:len(prophet_data)]
        )
    
    # Results section
    st.markdown("<div class='subtitle'>Forecast Results</div>", unsafe_allow_html=True)
    
    # Show metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Mean Absolute Error (MAE)", f"{metrics['MAE']}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Root Mean Square Error (RMSE)", f"{metrics['RMSE']}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("R-squared Score", f"{metrics['R2']}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Forecast Plot", "Model Components", "Forecast Data"])
    
    with tab1:
        st.markdown("<div class='subtitle'>Forecast Visualization</div>", unsafe_allow_html=True)
        fig = create_forecast_plot(model, forecast, prophet_data)
        st.pyplot(fig)
        st.markdown(get_download_link(fig), unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<div class='subtitle'>Model Components</div>", unsafe_allow_html=True)
        st.write("This visualization shows the different components of the forecast:")
        comp_fig = get_components_plot(model)
        st.pyplot(comp_fig)
        st.markdown(get_download_link(comp_fig, "forecast_components.png"), unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<div class='subtitle'>Forecast Data</div>", unsafe_allow_html=True)
        forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_display.columns = ['Date', 'Predicted Value', 'Lower Bound (95%)', 'Upper Bound (95%)']
        forecast_display['Date'] = pd.to_datetime(forecast_display['Date']).dt.strftime('%Y-%m-%d')
        
        # Highlight the future predictions
        historical_data_len = len(prophet_data)
        future_forecast = forecast_display.iloc[historical_data_len:].copy()
        
        st.write("### Future Predictions")
        st.dataframe(future_forecast)
        
        # Download link for forecast data
        csv = future_forecast.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="forecast_data.csv">Download forecast data as CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    # Explanations section
    st.markdown("<div class='subtitle'>Understanding the Results</div>", unsafe_allow_html=True)
    
    explanations = explain_metrics(metrics)
    
    with st.expander("What do these metrics mean?", expanded=True):
        st.markdown(explanations['MAE'])
        st.markdown(explanations['RMSE'])
        st.markdown(explanations['R2'])
    
    with st.expander("How to interpret the forecast?"):
        st.markdown("""
        ### Understanding the Forecast
        
        The red line shows the predicted values for your time series, while the blue dots represent the actual historical data.
        The light red shaded area represents the 95% confidence interval - there is a 95% probability that the actual future values
        will fall within this range.
        
        ### Reading the Components Plot
        
        The components plot breaks down the forecast into its constituent parts:
        
        - **Trend**: The overall upward or downward movement in the data over time
        - **Yearly**: Seasonal patterns that repeat on a yearly basis
        - **Weekly**: Patterns that occur on a weekly basis
        
        Examining these components can help you understand what factors are driving the changes in your data.
        
        ### Taking Action Based on the Forecast
        
        To make the most of this forecast:
        
        1. **Focus on trends**: Look for clear upward or downward movements
        2. **Consider seasonality**: Plan for seasonal peaks and valleys
        3. **Evaluate confidence intervals**: Wider intervals indicate more uncertainty
        4. **Monitor performance**: Compare actual results with predictions over time to improve forecasting
        """)

# Run the app
if __name__ == "__main__":
    main() 