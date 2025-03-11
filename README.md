# Time Series Forecasting Tool

A web-based forecasting application that allows users to upload their time series data and visualize predictions using the Prophet forecasting model.

## Features

- **Easy File Upload**: Import Excel files containing time series data
- **Interactive Visualizations**: View forecast plots, model components, and data tables
- **Explanatory Information**: Understand what metrics mean and how to interpret results
- **Downloadable Results**: Save your forecast plots and prediction data for later use

## Requirements

The application requires the following packages:
- Python 3.9+
- numpy
- pandas
- prophet
- matplotlib
- pytz
- openpyxl
- streamlit

## Installation

It's recommended to install the dependencies using conda:

```bash
conda create -n forecast_env python=3.9
conda activate forecast_env
conda install -c conda-forge numpy pandas prophet matplotlib pytz openpyxl streamlit
```

Alternatively, you can install using pip:

```bash
pip install -r requirements.txt
```

## Usage

1. Activate your environment:
   ```bash
   conda activate forecast_env
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run forecast_app.py
   ```

3. Upload your Excel file or use the sample data:
   - Your file should contain columns 'Week Start' and 'Units Sold'
   - The application will process the data and generate forecasts

4. Explore the results:
   - View the forecast plot
   - Analyze model components
   - Download prediction data
   - Understand performance metrics

## File Descriptions

- **forecast_app.py**: The main Streamlit application
- **lambda_tester_v2.py**: The original forecasting script (for command-line use)
- **Sample_weekly_data.xlsx**: Sample data file for testing

## Interpreting the Results

- **MAE (Mean Absolute Error)**: Average magnitude of errors in the predictions
- **RMSE (Root Mean Square Error)**: Square root of the average of squared errors
- **R² (R-squared)**: Proportion of variance in the data explained by the model

## License

This project is open source and available for personal and commercial use. 