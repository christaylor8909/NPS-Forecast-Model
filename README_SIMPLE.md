# NPS Forecast Model - Simple Version

A clean, simple web-based application for forecasting Net Promoter Score (NPS) trends with an overall trend line and optional individual brand visualization.

## Features

- **Simple Overall Trend**: One main NPS trend line (average across all brands)
- **Optional Brand Lines**: Toggle to show individual brand trends
- **Clean Interface**: Simple, uncluttered design
- **Brand Detection**: Automatically detects brands from dealership names
- **Interactive Chart**: Clean visualization with confidence intervals

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

```bash
python simple_forecast_app.py
```

Then open your browser and navigate to: `http://127.0.0.1:8050/`

### Uploading Your Data

1. **Prepare your CSV file** with the following format:
   - Each row represents a different dealership
   - Columns are date headers (e.g., 2024/10, 2024/11, etc.)
   - Values are NPS scores for each month

2. **Upload the file**:
   - Click "Drag and Drop or Select Files"
   - Select your CSV file
   - The app will automatically process the data

3. **View the results**:
   - Main chart shows overall NPS trend (average across all brands)
   - Check "Show Individual Brands" to see brand-specific lines
   - Overall NPS metrics are shown in the key metrics section

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
- **Brand Detection**: Automatically detects brands (Jeep, Chrysler, Dodge, etc.) from dealership names

## Features Overview

### Dashboard Components

1. **File Upload Section**:
   - Drag and drop interface
   - CSV file support
   - Automatic data processing

2. **Chart Options**:
   - Checkbox to show/hide individual brand lines
   - Simple toggle for additional detail

3. **Overall NPS Metrics**:
   - Current NPS (average across all brands)
   - 3-month forecast (overall)
   - Average NPS (overall)

4. **Simple Forecast Chart**:
   - Main overall NPS trend line (thick blue line)
   - Forecast line (thick orange dashed line)
   - Confidence interval (light orange area)
   - Optional individual brand lines (thinner, colored lines)
   - Target line at NPS 65

5. **Data Summary**:
   - Number of data points used
   - Date range of historical data
   - NPS range and trend direction
   - Confidence interval information

### Brand Detection

The app automatically detects brands from dealership names by looking for:
- **Jeep** - in dealership names
- **Chrysler** - in dealership names
- **Dodge** - in dealership names
- **Fiat** - in dealership names
- **Ram** - in dealership names
- **Alfa Romeo** - in dealership names

Examples:
- "IAP Australia Chrysler NPS NPS" → **Chrysler**
- "IAP Australia Jeep 22868 NPS NPS" → **Jeep**
- "IAP Australia Dodge NPS NPS" → **Dodge**

### Forecasting Model

- **Algorithm**: Polynomial regression (degree 2)
- **Forecast Period**: 3 months ahead
- **Confidence Level**: 95% confidence intervals
- **Overall Trend**: Average of all brands/dealerships
- **Individual Brands**: Separate lines for each detected brand

## File Structure

```
NPS-Forecast-Model/
├── simple_forecast_app.py    # Main simple application
├── data_loader.py            # Data loading and preprocessing
├── forecast_model.py         # Forecasting algorithms
├── requirements.txt          # Python dependencies
└── README_SIMPLE.md         # This file
```

## Troubleshooting

### Upload Issues
- Make sure your CSV file has the correct format
- Check that date columns are in YYYY/MM format
- Ensure NPS values are numerical

### Data Processing
- Zero values are automatically filtered out
- Missing values are handled gracefully
- Brand detection works automatically

### Chart Display
- Main chart always shows overall NPS trend
- Individual brand lines are optional (use checkbox)
- Target line is always visible at NPS 65

### Port Issues
If port 8050 is already in use, modify the port in `simple_forecast_app.py`:
```python
app.run(debug=True, host='127.0.0.1', port=8051)  # Change to different port
```

## Technical Details

- **Framework**: Dash (Python web framework)
- **Visualization**: Plotly with simple line charts
- **Styling**: Bootstrap
- **Forecasting**: Scikit-learn (Polynomial Regression)
- **Data Processing**: Pandas, NumPy
- **Brand Detection**: String matching in dealership names

## Support

If you encounter any issues:
1. Check the console output for error messages
2. Verify your CSV format matches the requirements
3. Ensure all dependencies are installed correctly
4. Check that brand names are properly formatted in your data

The application provides a clean, simple interface focused on the overall NPS trend with the option to drill down into individual brand performance when needed.
