# NPS Forecast Model - Final Version

A clean, professional web-based application for forecasting Net Promoter Score (NPS) trends with interactive visualizations.

## Features

- **Clean Professional Interface**: No emojis, business-appropriate design
- **File Upload**: Drag and drop CSV file upload functionality
- **Automatic Data Processing**: Handles your CSV format automatically
- **Interactive Charts**: Beautiful forecast visualizations with confidence intervals
- **Real-time Updates**: Instant forecast updates when data is uploaded

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

```bash
python final_app.py
```

Then open your browser and navigate to: `http://127.0.0.1:8050/`

### Uploading Your Data

1. **Prepare your CSV file** with the following format:
   - Each row represents a dealership
   - Columns are date headers (e.g., 2024/10, 2024/11, etc.)
   - Values are NPS scores for each month

2. **Upload the file**:
   - Click "Drag and Drop or Select Files"
   - Select your CSV file
   - The app will automatically process all dealerships and combine them into one forecast

3. **View the forecast**:
   - The chart will update automatically showing the trend
   - Key metrics will show current and forecasted NPS
   - Data summary provides additional insights

## Data Format

Your CSV should look like this:

```
2024/10,2024/11,2024/12,2025/01,2025/02,2025/03,2025/04,2025/05,2025/06,2025/07,2025/08,2025/09
0.0,56.5,0.0,0.0,0.0,0.0,31.4,0.0,27.3,62.9,32.4,0.0
0.0,0.0,0.0,0.0,0.0,0.0,-33.3,0.0,-100.0,-100.0,-100.0,0.0
...
```

- **Rows**: Each row represents a different dealership
- **Columns**: Date headers in YYYY/MM format
- **Values**: NPS scores (can be positive, negative, or zero)
- **Zero values**: Will be automatically filtered out
- **Combined Forecast**: All dealerships are combined into one overall forecast

## Features Overview

### Dashboard Components

1. **File Upload Section**:
   - Drag and drop interface
   - CSV file support
   - Automatic data processing

2. **Key Metrics**:
   - Current NPS score
   - 3-month forecast
   - Average NPS

3. **Interactive Chart**:
   - Historical NPS trend
   - 3-month forecast with confidence intervals
   - Hover tooltips for detailed information

4. **Data Summary**:
   - Number of data points used
   - Date range of historical data
   - NPS range and trend direction
   - Confidence interval information

### Forecasting Model

- **Algorithm**: Polynomial regression (degree 2)
- **Forecast Period**: 3 months ahead
- **Confidence Level**: 95% confidence intervals
- **Model Performance**: MAE and RMSE metrics
- **Data Combination**: All dealerships combined into one time series

## File Structure

```
NPS-Forecast-Model/
├── final_app.py           # Main final application
├── data_loader.py         # Data loading and preprocessing
├── forecast_model.py      # Forecasting algorithms
├── requirements.txt       # Python dependencies
└── README_FINAL.md       # This file
```

## Troubleshooting

### Upload Issues
- Make sure your CSV file has the correct format
- Check that date columns are in YYYY/MM format
- Ensure NPS values are numerical

### Data Processing
- Zero values are automatically filtered out
- Missing values are handled gracefully
- All dealerships are combined into one forecast

### Port Issues
If port 8050 is already in use, modify the port in `final_app.py`:
```python
app.run(debug=True, host='127.0.0.1', port=8051)  # Change to different port
```

## Technical Details

- **Framework**: Dash (Python web framework)
- **Visualization**: Plotly
- **Styling**: Bootstrap
- **Forecasting**: Scikit-learn (Polynomial Regression)
- **Data Processing**: Pandas, NumPy

## Support

If you encounter any issues:
1. Check the console output for error messages
2. Verify your CSV format matches the requirements
3. Ensure all dependencies are installed correctly

The application is designed to be simple, clean, and professional - perfect for business presentations and analysis.
