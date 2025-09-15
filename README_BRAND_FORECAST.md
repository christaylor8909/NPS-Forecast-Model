# NPS Forecast Model - Brand/Dealership Version

A comprehensive web-based application for forecasting Net Promoter Score (NPS) trends for individual brands and dealerships with interactive visualizations.

## Features

- **Individual Brand Forecasts**: Separate forecast lines for each brand/dealership
- **Overall NPS Metrics**: Combined statistics in the key metrics area
- **Interactive Multi-Line Chart**: Visualize all brands simultaneously
- **Confidence Intervals**: 95% confidence bounds for each forecast
- **Target Line**: Reference line at NPS 65
- **Professional Interface**: Clean, business-appropriate design

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

```bash
python brand_forecast_app.py
```

Then open your browser and navigate to: `http://127.0.0.1:8050/`

### Uploading Your Data

1. **Prepare your CSV file** with the following format:
   - Each row represents a different brand/dealership
   - Columns are date headers (e.g., 2024/10, 2024/11, etc.)
   - Values are NPS scores for each month

2. **Upload the file**:
   - Click "Drag and Drop or Select Files"
   - Select your CSV file
   - The app will automatically process each brand/dealership separately

3. **View the results**:
   - Each brand/dealership gets its own forecast line
   - Overall NPS metrics are shown in the key metrics section
   - Interactive chart shows all brands with their individual trends

## Data Format

Your CSV should look like this:

```
2024/10,2024/11,2024/12,2025/01,2025/02,2025/03,2025/04,2025/05,2025/06,2025/07,2025/08,2025/09
0.0,56.5,0.0,0.0,0.0,0.0,31.4,0.0,27.3,62.9,32.4,0.0
0.0,0.0,0.0,0.0,0.0,0.0,-33.3,0.0,-100.0,-100.0,-100.0,0.0
...
```

- **Rows**: Each row represents a different brand/dealership
- **Columns**: Date headers in YYYY/MM format
- **Values**: NPS scores (can be positive, negative, or zero)
- **Zero values**: Will be automatically filtered out
- **Individual Forecasts**: Each brand gets its own forecast model

## Features Overview

### Dashboard Components

1. **File Upload Section**:
   - Drag and drop interface
   - CSV file support
   - Automatic data processing

2. **Overall NPS Metrics**:
   - Current NPS (combined across all brands)
   - 3-month forecast (combined)
   - Average NPS (combined)

3. **Interactive Multi-Brand Chart**:
   - Historical data for each brand/dealership
   - Individual 3-month forecasts
   - Confidence intervals for each brand
   - Target line at NPS 65
   - Color-coded legend

4. **Data Summary**:
   - Number of data points used
   - Date range of historical data
   - NPS range and trend direction
   - Confidence interval information

### Forecasting Model

- **Algorithm**: Polynomial regression (degree 2) for each brand
- **Forecast Period**: 3 months ahead for each brand
- **Confidence Level**: 95% confidence intervals
- **Model Performance**: MAE and RMSE metrics per brand
- **Individual Models**: Each brand/dealership gets its own forecast model

## Chart Features

### Visual Elements
- **Historical Lines**: Solid lines showing past NPS data
- **Forecast Lines**: Dashed lines showing future predictions
- **Confidence Bands**: Semi-transparent areas showing uncertainty
- **Target Line**: Red dotted line at NPS 65
- **Color Coding**: Each brand has a unique color
- **Interactive Legend**: Click to show/hide specific brands

### Interactivity
- **Hover Tooltips**: Detailed information on hover
- **Zoom and Pan**: Navigate through the data
- **Legend Control**: Toggle visibility of individual brands
- **Responsive Design**: Works on all screen sizes

## File Structure

```
NPS-Forecast-Model/
├── brand_forecast_app.py    # Main brand forecast application
├── data_loader.py           # Data loading and preprocessing
├── forecast_model.py        # Forecasting algorithms
├── requirements.txt         # Python dependencies
└── README_BRAND_FORECAST.md # This file
```

## Troubleshooting

### Upload Issues
- Make sure your CSV file has the correct format
- Check that date columns are in YYYY/MM format
- Ensure NPS values are numerical

### Data Processing
- Zero values are automatically filtered out
- Missing values are handled gracefully
- Each brand needs at least 3 data points for forecasting

### Chart Display
- If a brand doesn't appear, it may not have enough data points
- Check the legend to see which brands are loaded
- Use the legend to toggle brand visibility

### Port Issues
If port 8050 is already in use, modify the port in `brand_forecast_app.py`:
```python
app.run(debug=True, host='127.0.0.1', port=8051)  # Change to different port
```

## Technical Details

- **Framework**: Dash (Python web framework)
- **Visualization**: Plotly with multi-line charts
- **Styling**: Bootstrap
- **Forecasting**: Scikit-learn (Polynomial Regression)
- **Data Processing**: Pandas, NumPy
- **Chart Features**: Interactive legends, confidence intervals, target lines

## Support

If you encounter any issues:
1. Check the console output for error messages
2. Verify your CSV format matches the requirements
3. Ensure all dependencies are installed correctly
4. Check that each brand has sufficient data points (minimum 3)

The application provides comprehensive brand-level forecasting while maintaining overall NPS insights - perfect for detailed business analysis and presentation.
