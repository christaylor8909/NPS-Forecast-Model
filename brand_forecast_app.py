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
forecast_models = {}
forecast_dfs = {}
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

def create_brand_forecast_chart(dealership_data, forecast_models, forecast_dfs):
    """Create a chart showing individual brand forecasts"""
    fig = go.Figure()
    
    # Define colors for different brands
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Add historical data and forecasts for each dealership
    for i, (dealership, model) in enumerate(forecast_models.items()):
        if dealership in forecast_dfs:
            color = colors[i % len(colors)]
            
            # Get data for this dealership
            dealer_data = dealership_data[dealership_data['Dealership'] == dealership]
            
            if len(dealer_data) > 0:
                # Add historical data
                fig.add_trace(go.Scatter(
                    x=dealer_data['Date'],
                    y=dealer_data['NPS'],
                    mode='lines+markers',
                    name=f'{dealership} (Historical)',
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    showlegend=True
                ))
                
                # Add forecast data
                forecast_df = forecast_dfs[dealership]
                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'],
                    y=forecast_df['NPS'],
                    mode='lines+markers',
                    name=f'{dealership} (Forecast)',
                    line=dict(color=color, width=2, dash='dash'),
                    marker=dict(size=6),
                    showlegend=True
                ))
                
                # Add confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'],
                    y=forecast_df['Upper_Bound'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'],
                    y=forecast_df['Lower_Bound'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
                    name=f'{dealership} (Confidence)',
                    hoverinfo='skip',
                    showlegend=False
                ))
    
    # Add target line
    fig.add_hline(
        y=65,
        line_dash="dot",
        line_color="red",
        annotation_text="Target (65)",
        annotation_position="top right"
    )
    
    # Update layout
    fig.update_layout(
        title="NPS Forecast by Brand/Dealership",
        xaxis_title="Date",
        yaxis_title="NPS Score",
        hovermode='x unified',
        template='plotly_white',
        height=600,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig

# Initialize with sample data
sample_data = create_sample_data()
forecast_model = NPSForecastModel(sample_data)
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
                    html.H4("Overall NPS Metrics", className="card-title mb-3"),
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
                                html.H2(id='average-nps', children=f"{sample_data['NPS'].mean():.1f}", className="text-info mb-1", style={'fontSize': '2.5rem', 'fontWeight': 'bold'}),
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
                    html.H4("NPS Forecast by Brand/Dealership", className="card-title mb-3"),
                    dcc.Graph(id='nps-forecast-chart', style={'height': '700px'})
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
                            html.P(f"NPS Range: {sample_data['NPS'].min():.1f} to {sample_data['NPS'].max():.1f}", className="mb-2", style={'fontSize': '1rem'}),
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
    global current_data, forecast_models, forecast_dfs, forecast_summary
    
    if contents is None:
        # Return current data with sample chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sample_data['Date'],
            y=sample_data['NPS'],
            mode='lines+markers',
            name='Sample Data',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6)
        ))
        fig.update_layout(
            title="NPS Forecast Model",
            xaxis_title="Date",
            yaxis_title="NPS Score",
            template='plotly_white',
            height=700
        )
        return "", fig, f"{forecast_summary['current_nps']:.1f}", f"{forecast_summary['forecast_nps_3m']:.1f}", f"{sample_data['NPS'].mean():.1f}"
    
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
        
        # Create individual forecast models for each dealership
        forecast_models = {}
        forecast_dfs = {}
        
        for dealership in df_processed['Dealership'].unique():
            dealer_data = df_processed[df_processed['Dealership'] == dealership].copy()
            
            if len(dealer_data) >= 3:  # Need at least 3 data points for forecasting
                # Create forecast model for this dealership
                model = NPSForecastModel(dealer_data)
                model.fit_polynomial_model()
                forecast_df = model.create_forecast(months_ahead=3)
                
                forecast_models[dealership] = model
                forecast_dfs[dealership] = forecast_df
        
        # Create overall forecast for metrics
        df_combined = df_processed.groupby('Date')['NPS'].mean().reset_index()
        df_combined = df_combined.sort_values('Date').reset_index(drop=True)
        
        overall_model = NPSForecastModel(df_combined)
        overall_model.fit_polynomial_model()
        overall_forecast = overall_model.create_forecast(months_ahead=3)
        forecast_summary = overall_model.get_forecast_summary()
        
        # Create the brand forecast chart
        fig = create_brand_forecast_chart(df_processed, forecast_models, forecast_dfs)
        
        # Update metrics
        current_nps = df_combined['NPS'].iloc[-1] if len(df_combined) > 0 else 0
        forecast_nps = forecast_summary['forecast_nps_3m'] if forecast_summary else 0
        average_nps = df_combined['NPS'].mean() if len(df_combined) > 0 else 0
        
        return dbc.Alert(f"Successfully loaded {len(df_processed)} data points for {len(forecast_models)} brands/dealerships!", color="success"), fig, f"{current_nps:.1f}", f"{forecast_nps:.1f}", f"{average_nps:.1f}"
        
    except Exception as e:
        return dbc.Alert(f"Error processing file: {str(e)}", color="danger"), go.Figure(), "N/A", "N/A", "N/A"

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8050)
