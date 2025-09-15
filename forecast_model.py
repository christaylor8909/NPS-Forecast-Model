"""
Forecast model for NPS data
This module contains various forecasting methods for NPS trends
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class NPSForecastModel:
    """
    NPS forecasting model using multiple approaches
    """
    
    def __init__(self, data):
        """
        Initialize the forecast model
        
        Args:
            data (pd.DataFrame): NPS data with Date and NPS columns
        """
        self.data = data.copy()
        self.model = None
        self.poly_features = None
        self.forecast_df = None
        
    def prepare_data(self):
        """Prepare data for modeling"""
        # Convert dates to numeric values (days since first date)
        self.data['Date_numeric'] = (self.data['Date'] - self.data['Date'].min()).dt.days
        
        # Remove any missing values
        self.data = self.data.dropna()
        
        return self.data
    
    def fit_polynomial_model(self, degree=2, adjust_seasonality=False):
        """
        Fit a polynomial regression model
        
        Args:
            degree (int): Degree of polynomial features
            adjust_seasonality (bool): Whether to adjust for seasonality
        """
        self.prepare_data()
        
        X = self.data['Date_numeric'].values.reshape(-1, 1)
        y = self.data['NPS'].values
        
        # Add seasonality features if requested
        if adjust_seasonality:
            # Add month and quarter features
            self.data['Month'] = self.data['Date'].dt.month
            self.data['Quarter'] = self.data['Date'].dt.quarter
            
            # Create seasonality features
            month_features = np.zeros((len(self.data), 12))
            for i, month in enumerate(self.data['Month']):
                month_features[i, month-1] = 1
            
            # Combine time and seasonality features
            X = np.hstack([X, month_features])
        
        # Create polynomial features
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = self.poly_features.fit_transform(X)
        
        # Fit the model
        self.model = LinearRegression()
        self.model.fit(X_poly, y)
        
        # Calculate model performance
        y_pred = self.model.predict(X_poly)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Calculate R-squared
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Calculate additional statistics
        mape = np.mean(np.abs((y - y_pred) / y)) * 100  # Mean Absolute Percentage Error
        mse = mean_squared_error(y, y_pred)
        
        # Store model statistics
        self.model_stats = {
            'mae': mae,
            'rmse': rmse,
            'r_squared': r_squared,
            'mape': mape,
            'mse': mse,
            'degree': degree,
            'seasonality_adjusted': adjust_seasonality
        }
        
        print(f"Model fitted with degree {degree} polynomial")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r_squared:.3f}")
        print(f"MAPE: {mape:.2f}%")
        
        return self.model
    
    def create_forecast(self, months_ahead=3, confidence_level=0.95):
        """
        Create forecast for the next N months
        
        Args:
            months_ahead (int): Number of months to forecast
            confidence_level (float): Confidence level for prediction intervals
            
        Returns:
            pd.DataFrame: Forecast data with confidence intervals
        """
        if self.model is None:
            self.fit_polynomial_model()
        
        # Generate future dates
        last_date = self.data['Date'].max()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1), 
            periods=months_ahead * 30, 
            freq='D'
        )
        
        # Convert future dates to numeric
        future_dates_numeric = (future_dates - self.data['Date'].min()).days
        X_future = future_dates_numeric.values.reshape(-1, 1)
        
        # Add seasonality features if model was trained with seasonality
        if hasattr(self, 'model_stats') and self.model_stats.get('seasonality_adjusted', False):
            # Add month features for future dates
            future_months = pd.to_datetime(future_dates).month
            month_features = np.zeros((len(future_dates), 12))
            for i, month in enumerate(future_months):
                month_features[i, month-1] = 1
            
            # Combine time and seasonality features
            X_future = np.hstack([X_future, month_features])
        
        X_future_poly = self.poly_features.transform(X_future)
        
        # Make predictions
        future_predictions = self.model.predict(X_future_poly)
        
        # Calculate confidence intervals
        # Get residuals from training data
        X_train = self.data['Date_numeric'].values.reshape(-1, 1)
        
        # Add seasonality features for training data if needed
        if hasattr(self, 'model_stats') and self.model_stats.get('seasonality_adjusted', False):
            train_months = self.data['Month']
            train_month_features = np.zeros((len(self.data), 12))
            for i, month in enumerate(train_months):
                train_month_features[i, month-1] = 1
            X_train = np.hstack([X_train, train_month_features])
        
        X_train_poly = self.poly_features.transform(X_train)
        y_train_pred = self.model.predict(X_train_poly)
        residuals = self.data['NPS'].values - y_train_pred
        std_error = np.std(residuals)
        
        # Calculate z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        # Create forecast dataframe
        self.forecast_df = pd.DataFrame({
            'Date': future_dates,
            'NPS': future_predictions,
            'Type': 'Forecast',
            'Upper_Bound': future_predictions + z_score * std_error,
            'Lower_Bound': future_predictions - z_score * std_error,
            'Confidence_Level': confidence_level
        })
        
        return self.forecast_df
    
    def get_monthly_forecast(self, months_ahead=3):
        """
        Get monthly aggregated forecast
        
        Args:
            months_ahead (int): Number of months to forecast
            
        Returns:
            pd.DataFrame: Monthly forecast data
        """
        if self.forecast_df is None:
            self.create_forecast(months_ahead)
        
        # Aggregate daily forecasts to monthly
        monthly_forecast = self.forecast_df.copy()
        monthly_forecast['YearMonth'] = monthly_forecast['Date'].dt.to_period('M')
        
        monthly_agg = monthly_forecast.groupby('YearMonth').agg({
            'NPS': 'mean',
            'Upper_Bound': 'mean',
            'Lower_Bound': 'mean'
        }).reset_index()
        
        # Convert back to datetime
        monthly_agg['Date'] = monthly_agg['YearMonth'].dt.to_timestamp()
        monthly_agg = monthly_agg.drop('YearMonth', axis=1)
        
        return monthly_agg
    
    def get_forecast_summary(self):
        """
        Get summary statistics of the forecast
        
        Returns:
            dict: Summary statistics
        """
        if self.forecast_df is None:
            return None
        
        # Get monthly forecast
        monthly_forecast = self.get_monthly_forecast()
        
        summary = {
            'forecast_periods': len(monthly_forecast),
            'current_nps': self.data['NPS'].iloc[-1],
            'forecast_nps_1m': monthly_forecast['NPS'].iloc[0] if len(monthly_forecast) > 0 else None,
            'forecast_nps_3m': monthly_forecast['NPS'].iloc[-1] if len(monthly_forecast) > 0 else None,
            'trend_direction': 'Increasing' if monthly_forecast['NPS'].iloc[-1] > self.data['NPS'].iloc[-1] else 'Decreasing',
            'confidence_interval': f"±{((monthly_forecast['Upper_Bound'] - monthly_forecast['Lower_Bound']) / 2).mean():.1f}",
            'data_points_used': len(self.data),
            'date_range': f"{self.data['Date'].min().strftime('%Y-%m-%d')} to {self.data['Date'].max().strftime('%Y-%m-%d')}"
        }
        
        # Add model statistics if available
        if hasattr(self, 'model_stats'):
            summary.update({
                'r_squared': self.model_stats['r_squared'],
                'mae': self.model_stats['mae'],
                'rmse': self.model_stats['rmse'],
                'mape': self.model_stats['mape'],
                'mse': self.model_stats['mse'],
                'model_degree': self.model_stats['degree'],
                'seasonality_adjusted': self.model_stats['seasonality_adjusted']
            })
        
        return summary
    
    def get_model_statistics(self):
        """
        Get comprehensive model statistics
        
        Returns:
            dict: Model performance statistics
        """
        if not hasattr(self, 'model_stats'):
            return None
        
        return self.model_stats
    
    def plot_forecast(self, fig=None):
        """
        Create a plot of the forecast
        
        Args:
            fig: Plotly figure object (optional)
            
        Returns:
            plotly.graph_objects.Figure: Updated figure
        """
        import plotly.graph_objects as go
        
        if fig is None:
            fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=self.data['Date'],
            y=self.data['NPS'],
            mode='lines+markers',
            name='Historical NPS',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6)
        ))
        
        if self.forecast_df is not None:
            # Add forecast data
            fig.add_trace(go.Scatter(
                x=self.forecast_df['Date'],
                y=self.forecast_df['NPS'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#ff7f0e', width=3, dash='dash'),
                marker=dict(size=6)
            ))
            
            # Add confidence interval
            fig.add_trace(go.Scatter(
                x=self.forecast_df['Date'],
                y=self.forecast_df['Upper_Bound'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=self.forecast_df['Date'],
                y=self.forecast_df['Lower_Bound'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255, 127, 14, 0.2)',
                name='Confidence Interval',
                hoverinfo='skip'
            ))
            
            # Add vertical line to separate historical and forecast
            last_historical_date = self.data['Date'].max()
            # Use add_shape instead of add_vline to avoid timestamp issues
            fig.add_shape(
                type="line",
                x0=last_historical_date, x1=last_historical_date,
                y0=0, y1=1,
                yref="paper",
                line=dict(color="gray", width=2, dash="dot"),
            )
            fig.add_annotation(
                x=last_historical_date,
                y=0.95,
                yref="paper",
                text="Forecast Start",
                showarrow=False,
                font=dict(color="gray")
            )
        
        # Update layout
        fig.update_layout(
            title="NPS Trend and 3-Month Forecast",
            xaxis_title="Date",
            yaxis_title="NPS Score",
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig

def create_forecast_model(data, months_ahead=3):
    """
    Convenience function to create a forecast model
    
    Args:
        data (pd.DataFrame): NPS data
        months_ahead (int): Number of months to forecast
        
    Returns:
        NPSForecastModel: Fitted forecast model
    """
    model = NPSForecastModel(data)
    model.fit_polynomial_model()
    model.create_forecast(months_ahead)
    return model

if __name__ == "__main__":
    # Test the forecast model
    from data_loader import create_sample_data
    
    # Create sample data
    data = create_sample_data()
    
    # Create and fit model
    model = NPSForecastModel(data)
    model.fit_polynomial_model()
    
    # Create forecast
    forecast = model.create_forecast(months_ahead=3)
    
    # Get summary
    summary = model.get_forecast_summary()
    print("Forecast Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Get monthly forecast
    monthly_forecast = model.get_monthly_forecast()
    print(f"\nMonthly Forecast:")
    print(monthly_forecast)
