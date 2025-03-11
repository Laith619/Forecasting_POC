# E-commerce Sales Forecasting Application

A comprehensive web-based forecasting application that enables e-commerce businesses to upload their sales data, select specific products, and generate accurate forecasts using multiple forecasting models. The application provides interactive visualizations, insightful metrics, and actionable business intelligence.

## Features

- **Data Processing**
  - Easy CSV/Excel file upload with comprehensive data validation
  - Automated handling of missing data and outliers
  - Advanced feature engineering (lag features, rolling statistics, seasonality)
  - Product selection to filter and forecast specific items

- **Multiple Forecasting Models**
  - Prophet model with optimized configuration for e-commerce data
  - ARIMA/SARIMAX model with exogenous variables support
  - Future support for LSTM and XGBoost models
  - Configurable model parameters through the UI

- **Interactive Visualizations**
  - Forecast plots with confidence intervals
  - Weekly breakdown of forecasted values
  - Model component visualization (trend, seasonality)
  - Performance metrics dashboard

- **User Experience**
  - Clean, intuitive Streamlit interface
  - Comprehensive error handling
  - Explanatory tooltips and guidance
  - Model comparison capabilities

## Implementation Status

Current progress: **80% complete**

- ✅ Data ingestion and preprocessing (100%)
- ⚠️ Forecasting models implementation (57%)
- ✅ User interface and UX design (100%)
- ⚠️ Metrics and visualization (86%)
- ⚠️ Testing and deployment (71%)

For detailed implementation status, see [Implementation Plan](implementation_plan.md).

## Requirements

The application requires the following packages:
```
streamlit>=1.20.0
prophet>=1.1.1
pandas>=1.5.0
numpy>=1.22.0
scikit-learn>=1.0.0
plotly>=5.10.0
statsmodels>=0.13.5
pmdarima>=2.0.3
scipy>=1.9.0
```

For future features (LSTM and XGBoost models):
```
tensorflow>=2.10.0
xgboost>=1.7.0
```

## Installation

### Option 1: Using Docker (Recommended)

The easiest way to run the application is using Docker:

```bash
# Using Docker directly
./run_docker.sh

# OR using Docker Compose
./run_compose.sh
```

This will build and start the application container, making it available at http://localhost:8501.

### Option 2: Local Installation

It's recommended to install the dependencies using conda:

```bash
conda create -n forecasting_env python=3.11
conda activate forecasting_env
conda install -c conda-forge numpy pandas prophet matplotlib statsmodels
conda install -c conda-forge streamlit plotly scikit-learn pmdarima scipy
```

Alternatively, you can install using pip:

```bash
pip install -r requirements.txt
```

## Usage

### Using Docker

1. Start the application:
   ```bash
   ./run_docker.sh
   ```
   or
   ```bash
   ./run_compose.sh
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

### Local Installation

1. Activate your environment:
   ```bash
   conda activate forecasting_env
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run app/main.py
   ```

3. Upload your data:
   - Your CSV file should contain the required columns: 'Week Start', 'Units Sold', 'GMV', 'Page Views', 'Ad Spend', 'Ad Sales'
   - Optional columns: 'Item Id' for product selection

4. Configure your forecast:
   - Select a specific product if your data contains multiple products
   - Choose which forecasting model(s) to use
   - Set model parameters and forecast horizon

5. Explore the results:
   - View the forecast plot with confidence intervals
   - Analyze the weekly breakdown of forecasted values
   - Compare model performance metrics
   - Understand model components and seasonality

## Project Structure

- **app/main.py**: Main Streamlit application entry point
- **app/data_processor.py**: Data validation, cleaning, and feature engineering
- **app/forecasting.py**: Prophet model implementation and forecasting engine
- **app/arima_forecaster.py**: ARIMA/SARIMAX model implementation
- **Dockerfile**: Configuration for containerizing the application
- **docker-compose.yml**: Docker Compose configuration for easy deployment
- **run_docker.sh**: Helper script for Docker operations
- **run_compose.sh**: Helper script for Docker Compose operations
- **implementation_plan.md**: Detailed implementation plan and progress tracking
- **ecommerce_forecast_enhancement_plan.md**: Enhancement roadmap and feature details

## Understanding Metrics

- **MAPE**: Mean Absolute Percentage Error - average percentage difference between forecasted and actual values
- **RMSE**: Root Mean Square Error - square root of the average of squared differences between forecasted and actual values
- **MAE**: Mean Absolute Error - average absolute difference between forecasted and actual values
- **AIC/BIC**: Information criteria for model selection (lower is better)

## Future Enhancements

- LSTM deep learning model for complex pattern recognition
- XGBoost model for high accuracy forecasting
- Enhanced model comparison dashboard
- Downloadable forecast reports
- Automated insights and recommendations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available for personal and commercial use. 