# NPS Forecast Model - Enhanced Version

A comprehensive web-based application for forecasting Net Promoter Score (NPS) trends with advanced statistical analysis, seasonality adjustment, and brand-specific drill-down capabilities.

## 🚀 New Features

### 📊 Enhanced Statistics & Model Performance
- **R² (Coefficient of Determination)**: Measures how well the model explains variance
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
- **RMSE (Root Mean Square Error)**: Standard deviation of prediction errors
- **MAPE (Mean Absolute Percentage Error)**: Percentage-based error measurement
- **Model Degree**: Polynomial degree used for forecasting
- **Seasonality Adjustment**: Toggle to account for seasonal patterns

### 🎛️ Advanced Chart Options
- **Seasonality Adjustment**: Automatically accounts for monthly/quarterly patterns
- **Individual Brand Toggles**: Select specific brands to display (Jeep, Fiat, Chrysler, etc.)
- **Dynamic Brand Detection**: Automatically identifies brands from dealership names
- **Interactive Brand Selection**: Check/uncheck individual brands for focused analysis

### 📈 Improved Layout
- **Reorganized Interface**: Chart options moved below data summary for better workflow
- **Comprehensive Data Summary**: Two-column layout with data overview and model performance
- **Professional Statistics Display**: Clear presentation of all model metrics

## 🛠️ Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Running the Application

```bash
python simple_forecast_app.py
```

Then open your browser and navigate to: `http://127.0.0.1:8050/`

### 📁 Data Format

Your CSV file should have:
- **First column**: Dealership names (brands detected automatically)
- **Subsequent columns**: Date headers in YYYY/MM format
- **Data cells**: NPS scores

Example:
```csv
Dealership,2023/01,2023/02,2023/03,2023/04
IAP Australia Jeep NPS,45.2,47.8,46.1,48.5
IAP Australia Fiat NPS,42.1,43.9,44.2,45.8
IAP Australia Chrysler NPS,48.3,49.1,47.9,50.2
```

### 🎛️ Using the Enhanced Features

#### 1. **Upload Your Data**
- Drag and drop your CSV file or click to select
- The app will automatically process and detect brands

#### 2. **Review Model Statistics**
- **Data Overview**: Points, date range, NPS range, trend direction
- **Model Performance**: R², MAE, RMSE, MAPE, model degree

#### 3. **Configure Chart Options**
- **Show Individual Brands**: Toggle to display brand-specific lines
- **Adjust for Seasonality**: Account for monthly/quarterly patterns
- **Brand Selection**: Choose specific brands to display

#### 4. **Analyze Results**
- **Overall Trend**: Thick blue line showing average NPS across all brands
- **Forecast**: Orange dashed line with confidence intervals
- **Individual Brands**: Colored lines for each selected brand
- **Target Line**: Red dotted line at NPS 65

## 📊 Statistical Metrics Explained

### **R² (Coefficient of Determination)**
- **Range**: 0 to 1 (higher is better)
- **Meaning**: Proportion of variance in NPS explained by the model
- **Good**: >0.7, **Excellent**: >0.9

### **MAE (Mean Absolute Error)**
- **Range**: 0 to ∞ (lower is better)
- **Meaning**: Average absolute difference between predicted and actual NPS
- **Interpretation**: If MAE = 3.5, predictions are off by ~3.5 NPS points on average

### **RMSE (Root Mean Square Error)**
- **Range**: 0 to ∞ (lower is better)
- **Meaning**: Standard deviation of prediction errors
- **Interpretation**: Penalizes larger errors more heavily than MAE

### **MAPE (Mean Absolute Percentage Error)**
- **Range**: 0% to ∞ (lower is better)
- **Meaning**: Average percentage error of predictions
- **Interpretation**: If MAPE = 8%, predictions are off by ~8% on average

## 🔧 Technical Details

### **Seasonality Adjustment**
- Adds month and quarter features to the polynomial model
- Accounts for seasonal patterns in NPS data
- Improves forecast accuracy for data with seasonal trends

### **Brand Detection**
- Automatically identifies brands from dealership names
- Supports: Jeep, Chrysler, Dodge, Fiat, Ram, Alfa Romeo
- Handles various naming conventions and formats

### **Model Architecture**
- **Base Model**: Polynomial regression (degree 2)
- **Seasonality**: Optional monthly feature encoding
- **Forecasting**: 3-month ahead predictions with confidence intervals
- **Confidence Level**: 95% prediction intervals

## 🎨 Interface Features

### **Layout Organization**
1. **Header**: Title and description
2. **File Upload**: Drag-and-drop CSV upload
3. **Key Metrics**: Current NPS, 3-month forecast, average NPS
4. **Chart**: Interactive NPS forecast visualization
5. **Data Summary**: Comprehensive statistics and model performance
6. **Chart Options**: Seasonality and brand selection controls

### **Interactive Elements**
- **Hover Information**: Detailed data points on chart
- **Brand Toggles**: Real-time brand line visibility
- **Seasonality Toggle**: Instant model recalculation
- **Responsive Design**: Works on desktop and tablet

## 🔍 Troubleshooting

### **Common Issues**

1. **"No valid data found"**
   - Check CSV format (first column = dealership names)
   - Ensure NPS values are numeric
   - Remove empty rows

2. **"Error processing file"**
   - Verify CSV encoding (UTF-8 recommended)
   - Check for special characters in dealership names
   - Ensure date format is YYYY/MM

3. **Brands not detected**
   - Verify brand names are in dealership names
   - Check spelling (Jeep, Fiat, Chrysler, etc.)
   - Ensure brand names are at the end of dealership names

### **Performance Tips**
- **Large datasets**: Consider sampling for faster processing
- **Many brands**: Use brand selection to focus on specific brands
- **Seasonal data**: Enable seasonality adjustment for better accuracy

## 📈 Best Practices

### **Data Quality**
- Ensure consistent date formats
- Remove outliers that might skew forecasts
- Include at least 6 months of data for reliable forecasts

### **Model Interpretation**
- **R² > 0.8**: Model explains most variance (good fit)
- **MAE < 5**: Predictions within 5 NPS points (acceptable)
- **MAPE < 10%**: Predictions within 10% error (good)

### **Forecast Usage**
- Use confidence intervals to assess uncertainty
- Consider seasonal patterns for business planning
- Monitor actual vs. predicted for model validation

## 🚀 Future Enhancements

- **Multiple Model Types**: ARIMA, Exponential Smoothing
- **Advanced Seasonality**: Custom seasonal patterns
- **Export Functionality**: Download forecasts and statistics
- **Alert System**: Notifications for forecast thresholds
- **Historical Comparison**: Compare different time periods

---

**Ready to forecast your NPS trends with advanced statistical analysis!** 🎯
