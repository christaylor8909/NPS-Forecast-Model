import dash
from dash import dcc, html, Input, Output, callback, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
import base64
import io
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_loader import load_nps_data
from forecast_model import NPSForecastModel

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "NPS Forecast Model"

# Global variables to store data
current_data = None
forecast_model = None
forecast_df = None
forecast_summary = None

def process_dealership_data(df):
    """Process the dealership CSV data into the format needed for forecasting"""
    # Clean up the data - remove unnamed columns
    df_clean = df.copy()
    df_clean = df_clean.loc[:, ~df_clean.columns.str.contains('^Unnamed')]
    
    # Check if we have dealership names in the first column or if it's just date columns
    first_col = df_clean.iloc[:, 0]
    
    if first_col.dtype == 'object' and not any(char in str(first_col.iloc[0]) for char in ['/', '-', '2024', '2025']):
        # First column contains dealership names
        df_clean['Dealership'] = df_clean.iloc[:, 0]
        date_columns = [col for col in df_clean.columns if col != 'Dealership']
    else:
        # No dealership names, create generic ones based on row index
        df_clean['Dealership'] = [f'Dealership {i+1}' for i in range(len(df_clean))]
        date_columns = [col for col in df_clean.columns if col != 'Dealership']
    
    # Melt the data to long format
    df_long = pd.melt(df_clean, 
                      id_vars=['Dealership'], 
                      value_vars=date_columns,
                      var_name='Date', 
                      value_name='NPS')
    
    # Convert date column to datetime
    try:
        # Try different date formats
        df_long['Date'] = pd.to_datetime(df_long['Date'], format='%Y/%m')
    except:
        try:
            df_long['Date'] = pd.to_datetime(df_long['Date'])
        except:
            try:
                # Handle the case where dates might be in string format like '2024/10'
                df_long['Date'] = pd.to_datetime(df_long['Date'].astype(str), format='%Y/%m')
            except:
                print("Warning: Could not parse dates, using sample dates")
                # Create dates based on the number of unique date columns
                unique_dates = df_long['Date'].unique()
                date_range = pd.date_range(start='2024-10-01', periods=len(unique_dates), freq='M')
                date_mapping = dict(zip(unique_dates, date_range))
                df_long['Date'] = df_long['Date'].map(date_mapping)
    
    # Remove rows with missing or zero NPS values
    df_long = df_long[(df_long['NPS'].notna()) & (df_long['NPS'] != 0)]
    
    # Sort by dealership and date
    df_long = df_long.sort_values(['Dealership', 'Date']).reset_index(drop=True)
    
    # For forecasting, we need to combine all dealerships into one time series
    # Group by date and take the mean NPS across all dealerships
    df_combined = df_long.groupby('Date')['NPS'].mean().reset_index()
    df_combined = df_combined.sort_values('Date').reset_index(drop=True)
    
    return df_combined

def create_sample_data():
    """Create sample NPS data for demonstration"""
    dates = pd.date_range(start='2022-01-01', end='2024-09-15', freq='M')
    np.random.seed(42)
    # Simulate NPS trend with some seasonality
    base_nps = 45
    trend = np.linspace(0, 15, len(dates))  # Upward trend
    seasonal = 8 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)  # Annual seasonality
    noise = np.random.normal(0, 4, len(dates))  # Random noise
    
    nps_values = base_nps + trend + seasonal + noise
    
    # Ensure NPS values are within reasonable bounds (0-100)
    nps_values = np.clip(nps_values, 0, 100)
    
    df = pd.DataFrame({
        'Date': dates,
        'NPS': nps_values.round(1)
    })
    
    return df

# Initialize with sample data
current_data = create_sample_data()
forecast_model = NPSForecastModel(current_data)
forecast_model.fit_polynomial_model()
forecast_df = forecast_model.create_forecast(months_ahead=3)
forecast_summary = forecast_model.get_forecast_summary()

# App layout
app.layout = dbc.Container([
    # Header with better spacing
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("NPS Forecast Model", className="text-center mb-2", style={'fontSize': '2.5rem', 'fontWeight': 'bold'}),
                html.P("Forecasting NPS trends for the next 3 months", className="text-center text-muted mb-4", style={'fontSize': '1.1rem'})
            ], style={'paddingTop': '2rem', 'paddingBottom': '1rem'})
        ])
    ]),
    
    # File Upload Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Upload Dealership Data", className="card-title mb-3"),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px',
                            'backgroundColor': '#f8f9fa'
                        },
                        multiple=False
                    ),
                    html.Div(id='upload-status', className="mt-2")
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Key Metrics with better spacing
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Key Metrics", className="card-title mb-3"),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H2(id='current-nps', children=f"{forecast_summary['current_nps']:.1f}", className="text-primary mb-1", style={'fontSize': '2.5rem', 'fontWeight': 'bold'}),
                                html.P("Current NPS", className="text-muted mb-0", style={'fontSize': '1rem'})
                            ], className="text-center")
                        ], width=4),
                        dbc.Col([
                            html.Div([
                                html.H2(id='forecast-nps', children=f"{forecast_summary['forecast_nps_3m']:.1f}", className="text-success mb-1", style={'fontSize': '2.5rem', 'fontWeight': 'bold'}),
                                html.P("3-Month Forecast", className="text-muted mb-0", style={'fontSize': '1rem'})
                            ], className="text-center")
                        ], width=4),
                        dbc.Col([
                            html.Div([
                                html.H2(id='average-nps', children=f"{current_data['NPS'].mean():.1f}", className="text-info mb-1", style={'fontSize': '2.5rem', 'fontWeight': 'bold'}),
                                html.P("Average NPS", className="text-muted mb-0", style={'fontSize': '1rem'})
                            ], className="text-center")
                        ], width=4)
                    ])
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Chart with better spacing
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("NPS Forecast Chart", className="card-title mb-3"),
                    dcc.Graph(id='nps-forecast-chart', style={'height': '600px'})
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Data Summary with better spacing
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Data Summary", className="card-title mb-3"),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.P(f"Data Points: {forecast_summary['data_points_used']}", className="mb-2", style={'fontSize': '1rem'}),
                            html.P(f"Date Range: {forecast_summary['date_range']}", className="mb-2", style={'fontSize': '1rem'}),
                            html.P(f"NPS Range: {current_data['NPS'].min():.1f} to {current_data['NPS'].max():.1f}", className="mb-2", style={'fontSize': '1rem'}),
                            html.P(f"Trend: {forecast_summary['trend_direction']}", className="mb-2", style={'fontSize': '1rem'}),
                            html.P(f"Confidence Interval: {forecast_summary['confidence_interval']}", className="mb-0", style={'fontSize': '1rem'})
                        ])
                    ])
                ])
            ])
        ], width=12)
    ], className="mb-5")  # Extra bottom margin
], fluid=True, style={'padding': '0 15px'})

# Callback for file upload
@app.callback(
    [Output('upload-status', 'children'),
     Output('nps-forecast-chart', 'figure'),
     Output('current-nps', 'children'),
     Output('forecast-nps', 'children'),
     Output('average-nps', 'children')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output(contents, filename):
    global current_data, forecast_model, forecast_df, forecast_summary
    
    if contents is None:
        # Return current data
        fig = forecast_model.plot_forecast()
        return "", fig, f"{forecast_summary['current_nps']:.1f}", f"{forecast_summary['forecast_nps_3m']:.1f}", f"{current_data['NPS'].mean():.1f}"
    
    try:
        # Parse the uploaded file
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        if 'csv' in filename.lower():
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return dbc.Alert("Please upload a CSV file.", color="danger"), go.Figure(), "N/A", "N/A", "N/A"
        
        # Process the dealership data
        df_processed = process_dealership_data(df)
        
        if len(df_processed) == 0:
            return dbc.Alert("No valid data found in the uploaded file.", color="warning"), go.Figure(), "N/A", "N/A", "N/A"
        
        # Store the processed data globally
        current_data = df_processed
        
        # Create forecast model with the new data
        forecast_model = NPSForecastModel(current_data)
        forecast_model.fit_polynomial_model()
        forecast_df = forecast_model.create_forecast(months_ahead=3)
        forecast_summary = forecast_model.get_forecast_summary()
        
        # Create the chart
        fig = forecast_model.plot_forecast()
        
        # Update metrics
        current_nps = current_data['NPS'].iloc[-1] if len(current_data) > 0 else 0
        forecast_nps = forecast_summary['forecast_nps_3m'] if forecast_summary else 0
        average_nps = current_data['NPS'].mean() if len(current_data) > 0 else 0
        
        return dbc.Alert(f"Successfully loaded {len(df_processed)} data points from your CSV file!", color="success"), fig, f"{current_nps:.1f}", f"{forecast_nps:.1f}", f"{average_nps:.1f}"
        
    except Exception as e:
        return dbc.Alert(f"Error processing file: {str(e)}", color="danger"), go.Figure(), "N/A", "N/A", "N/A"

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8050)
