import dash
from dash import dcc, html, Input, Output, callback, State, dash_table
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
available_dealerships = []

def process_dealership_data(df):
    """Process the dealership CSV data into the format needed for forecasting"""
    # Clean up the data
    df_clean = df.copy()
    
    # Use the first column as dealership names
    df_clean['Dealership'] = df_clean.iloc[:, 0]
    
    # Get date columns (all columns except the first one)
    date_columns = [col for col in df_clean.columns if col != 'Dealership']
    
    # Melt the data to long format
    df_long = pd.melt(df_clean, 
                      id_vars=['Dealership'], 
                      value_vars=date_columns,
                      var_name='Date', 
                      value_name='NPS')
    
    # Convert date column to datetime
    df_long['Date'] = pd.to_datetime(df_long['Date'], format='%Y/%m')
    
    # Remove rows with missing or zero NPS values
    df_long = df_long[(df_long['NPS'].notna()) & (df_long['NPS'] != 0)]
    
    # Sort by dealership and date
    df_long = df_long.sort_values(['Dealership', 'Date']).reset_index(drop=True)
    
    return df_long

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
                    html.H4("📁 Upload Dealership Data", className="card-title mb-3"),
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
    
    # Dealership Selection
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🏢 Select Dealership", className="card-title mb-3"),
                    dcc.Dropdown(
                        id='dealership-dropdown',
                        options=[{'label': 'Sample Data', 'value': 'sample'}],
                        value='sample',
                        placeholder="Select a dealership to forecast...",
                        style={'marginBottom': '10px'}
                    ),
                    html.Div(id='dealership-info')
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Key Metrics with better spacing
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("📊 Key Metrics", className="card-title mb-3"),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H2(f"{forecast_summary['current_nps']:.1f}", className="text-primary mb-1", style={'fontSize': '2.5rem', 'fontWeight': 'bold'}),
                                html.P("Current NPS", className="text-muted mb-0", style={'fontSize': '1rem'})
                            ], className="text-center")
                        ], width=4),
                        dbc.Col([
                            html.Div([
                                html.H2(f"{forecast_summary['forecast_nps_3m']:.1f}", className="text-success mb-1", style={'fontSize': '2.5rem', 'fontWeight': 'bold'}),
                                html.P("3-Month Forecast", className="text-muted mb-0", style={'fontSize': '1rem'})
                            ], className="text-center")
                        ], width=4),
                        dbc.Col([
                            html.Div([
                                html.H2(f"{current_data['NPS'].mean():.1f}", className="text-info mb-1", style={'fontSize': '2.5rem', 'fontWeight': 'bold'}),
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
                    html.H4("📈 NPS Forecast Chart", className="card-title mb-3"),
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
                    html.H4("📋 Data Summary", className="card-title mb-3"),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.P(f"📊 Data Points: {forecast_summary['data_points_used']}", className="mb-2", style={'fontSize': '1rem'}),
                            html.P(f"📅 Date Range: {forecast_summary['date_range']}", className="mb-2", style={'fontSize': '1rem'}),
                            html.P(f"📈 NPS Range: {current_data['NPS'].min():.1f} to {current_data['NPS'].max():.1f}", className="mb-2", style={'fontSize': '1rem'}),
                            html.P(f"📊 Trend: {forecast_summary['trend_direction']}", className="mb-2", style={'fontSize': '1rem'}),
                            html.P(f"🎯 Confidence Interval: {forecast_summary['confidence_interval']}", className="mb-0", style={'fontSize': '1rem'})
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
     Output('dealership-dropdown', 'options'),
     Output('dealership-dropdown', 'value')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output(contents, filename):
    global current_data, forecast_model, forecast_df, forecast_summary, available_dealerships
    
    if contents is None:
        return "", [{'label': 'Sample Data', 'value': 'sample'}], 'sample'
    
    try:
        # Parse the uploaded file
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        if 'csv' in filename.lower():
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return dbc.Alert("Please upload a CSV file.", color="danger"), [{'label': 'Sample Data', 'value': 'sample'}], 'sample'
        
        # Process the dealership data
        df_processed = process_dealership_data(df)
        
        if len(df_processed) == 0:
            return dbc.Alert("No valid data found in the uploaded file.", color="warning"), [{'label': 'Sample Data', 'value': 'sample'}], 'sample'
        
        # Get unique dealerships
        available_dealerships = df_processed['Dealership'].unique().tolist()
        dropdown_options = [{'label': 'Sample Data', 'value': 'sample'}] + [{'label': dealer, 'value': dealer} for dealer in available_dealerships]
        
        # Store the processed data globally
        current_data = df_processed
        
        return dbc.Alert(f"✅ Successfully loaded {len(df_processed)} data points for {len(available_dealerships)} dealerships!", color="success"), dropdown_options, 'sample'
        
    except Exception as e:
        return dbc.Alert(f"Error processing file: {str(e)}", color="danger"), [{'label': 'Sample Data', 'value': 'sample'}], 'sample'

# Callback for dealership selection
@app.callback(
    [Output('dealership-info', 'children'),
     Output('nps-forecast-chart', 'figure')],
    [Input('dealership-dropdown', 'value')]
)
def update_dealership(selected_dealership):
    global current_data, forecast_model, forecast_df, forecast_summary
    
    if selected_dealership == 'sample' or current_data is None:
        # Use sample data
        sample_data = create_sample_data()
        forecast_model = NPSForecastModel(sample_data)
        forecast_model.fit_polynomial_model()
        forecast_df = forecast_model.create_forecast(months_ahead=3)
        forecast_summary = forecast_model.get_forecast_summary()
        
        info = dbc.Alert("Using sample data for demonstration", color="info")
        fig = forecast_model.plot_forecast()
        
    else:
        # Filter data for selected dealership
        dealer_data = current_data[current_data['Dealership'] == selected_dealership].copy()
        
        if len(dealer_data) < 3:
            info = dbc.Alert(f"Insufficient data for {selected_dealership}. Need at least 3 data points.", color="warning")
            fig = go.Figure()
            fig.add_annotation(text="Insufficient data for forecasting", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        else:
            # Create forecast for this dealership
            forecast_model = NPSForecastModel(dealer_data)
            forecast_model.fit_polynomial_model()
            forecast_df = forecast_model.create_forecast(months_ahead=3)
            forecast_summary = forecast_model.get_forecast_summary()
            
            info = dbc.Alert(f"📊 {selected_dealership}: {len(dealer_data)} data points from {dealer_data['Date'].min().strftime('%Y-%m')} to {dealer_data['Date'].max().strftime('%Y-%m')}", color="success")
            fig = forecast_model.plot_forecast()
    
    return info, fig

# Callback for updating metrics when dealership changes
@app.callback(
    [Output('current-nps', 'children'),
     Output('forecast-nps', 'children'),
     Output('average-nps', 'children')],
    [Input('dealership-dropdown', 'value')]
)
def update_metrics(selected_dealership):
    global forecast_summary, current_data
    
    if selected_dealership == 'sample' or current_data is None:
        sample_data = create_sample_data()
        return f"{sample_data['NPS'].iloc[-1]:.1f}", f"{sample_data['NPS'].mean() + 5:.1f}", f"{sample_data['NPS'].mean():.1f}"
    
    dealer_data = current_data[current_data['Dealership'] == selected_dealership]
    if len(dealer_data) < 3:
        return "N/A", "N/A", "N/A"
    
    return f"{dealer_data['NPS'].iloc[-1]:.1f}", f"{forecast_summary['forecast_nps_3m']:.1f}", f"{dealer_data['NPS'].mean():.1f}"

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8050)
