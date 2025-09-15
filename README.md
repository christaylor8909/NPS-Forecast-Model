# NPS Forecast Model

A web application for forecasting Net Promoter Score (NPS) trends using polynomial regression with optional seasonality adjustment.

## Features

- **CSV Data Upload**: Upload dealership-specific monthly NPS data
- **Interactive Forecasting**: 3-month NPS trend forecasting with confidence intervals
- **Brand Analysis**: Individual brand trend analysis (Chrysler, Dodge, Jeep, Fiat, etc.)
- **Seasonality Adjustment**: Optional seasonal pattern adjustment
- **Model Statistics**: Comprehensive model performance metrics (R², MAE, RMSE, MAPE)
- **Real-time Updates**: Dynamic chart updates based on user interactions

## Data Format

The application expects CSV files with the following structure:
- First column: Dealership names (e.g., "IAP Australia Chrysler NPS NPS")
- Subsequent columns: Monthly NPS data with dates as headers (e.g., "2024/10", "2024/11")

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python simple_forecast_app.py
   ```

## Usage

1. Open the application in your browser (default: http://127.0.0.1:8050)
2. Upload a CSV file with your NPS data
3. View the forecast chart and statistics
4. Toggle seasonality adjustment and brand selection options
5. Analyze model performance metrics

## Deployment

The application is configured for deployment on Render. Simply connect your GitHub repository to Render and deploy using the provided `render.yaml` configuration.

## Technologies Used

- **Dash**: Web application framework
- **Plotly**: Interactive data visualization
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning models
- **NumPy**: Numerical computing
- **Bootstrap**: UI styling

## Model Details

- **Algorithm**: Polynomial regression (degree 2)
- **Forecast Period**: 3 months ahead
- **Seasonality**: Optional month-based seasonal adjustment
- **Confidence Intervals**: Statistical confidence bounds for predictions