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
overall_forecast_model = None
overall_forecast_df = None
overall_forecast_summary = None
brand_data = {}
available_brands = []

def process_dealership_data(df):
    """Process the dealership CSV data into the format needed for forecasting"""
    # Clean up the data - remove unnamed columns
    df_clean = df.copy()
    df_clean = df_clean.loc[:, ~df_clean.columns.str.contains('^Unnamed')]
    
    # Use the first column as dealership names, ensure it's string
    first_col_name = df_clean.columns[0]
    df_clean['Dealership'] = df_clean[first_col_name].astype(str)
    # Remove any rows where dealership name is NaN or empty
    df_clean = df_clean[df_clean['Dealership'].notna() & (df_clean['Dealership'] != 'nan') & (df_clean['Dealership'] != '')]
    date_columns = [col for col in df_clean.columns if col != 'Dealership']
    
    # Melt the data to long format
    df_long = pd.melt(df_clean, 
                      id_vars=['Dealership'], 
                      value_vars=date_columns,
                      var_name='Date', 
                      value_name='NPS')
    
    # Convert date column to datetime
    try:
        df_long['Date'] = pd.to_datetime(df_long['Date'], format='%Y/%m')
    except:
        try:
            df_long['Date'] = pd.to_datetime(df_long['Date'])
        except:
            try:
                df_long['Date'] = pd.to_datetime(df_long['Date'].astype(str), format='%Y/%m')
            except:
                print("Warning: Could not parse dates, using sample dates")
                unique_dates = df_long['Date'].unique()
                date_range = pd.date_range(start='2024-10-01', periods=len(unique_dates), freq='M')
                date_mapping = dict(zip(unique_dates, date_range))
                df_long['Date'] = df_long['Date'].map(date_mapping)
    
    # Remove rows with missing or zero NPS values
    df_long = df_long[(df_long['NPS'].notna()) & (df_long['NPS'] != 0)]
    
    # Sort by dealership and date
    df_long = df_long.sort_values(['Dealership', 'Date']).reset_index(drop=True)
    
    return df_long

def extract_brand_name(dealership_name):
    """Extract brand name from dealership name"""
    # Convert to string if it's not already
    if not isinstance(dealership_name, str):
        dealership_name = str(dealership_name)
    
    # Look for brand names at the end of the dealership name
    brands = ['Jeep', 'Chrysler', 'Dodge', 'Fiat', 'Ram', 'Alfa Romeo']
    
    for brand in brands:
        if brand.lower() in dealership_name.lower():
            return brand
    
    # If no brand found, return the original name
    return dealership_name

def create_sample_data():
    """Create sample NPS data for demonstration"""
    dates = pd.date_range(start='2022-01-01', end='2024-09-15', freq='M')
    np.random.seed(42)
    base_nps = 45
    trend = np.linspace(0, 15, len(dates))
    seasonal = 8 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
    noise = np.random.normal(0, 4, len(dates))
    
    nps_values = base_nps + trend + seasonal + noise
    nps_values = np.clip(nps_values, 0, 100)
    
    df = pd.DataFrame({
        'Date': dates,
        'NPS': nps_values.round(1)
    })
    
    return df

def create_simple_forecast_chart(overall_data, overall_forecast_df, brand_data, show_brands):
    """Create a simple forecast chart"""
    fig = go.Figure()
    
    # Add overall NPS trend (always shown)
    fig.add_trace(go.Scatter(
        x=overall_data['Date'],
        y=overall_data['NPS'],
        mode='lines+markers',
        name='Overall NPS (Historical)',
        line=dict(color='#1f77b4', width=4),
        marker=dict(size=8)
    ))
    
    # Add overall forecast
    fig.add_trace(go.Scatter(
        x=overall_forecast_df['Date'],
        y=overall_forecast_df['NPS'],
        mode='lines+markers',
        name='Overall NPS (Forecast)',
        line=dict(color='#ff7f0e', width=4, dash='dash'),
        marker=dict(size=8)
    ))
    
    # Add confidence interval for overall forecast
    fig.add_trace(go.Scatter(
        x=overall_forecast_df['Date'],
        y=overall_forecast_df['Upper_Bound'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=overall_forecast_df['Date'],
        y=overall_forecast_df['Lower_Bound'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255, 127, 14, 0.2)',
        name='Confidence Interval',
        hoverinfo='skip'
    ))
    
    # Add individual brand lines if requested
    if show_brands and brand_data:
        colors = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        color_idx = 0
        
        for brand, data in brand_data.items():
            if len(data) > 0:
                color = colors[color_idx % len(colors)]
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=data['NPS'],
                    mode='lines+markers',
                    name=f'{brand}',
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    opacity=0.7
                ))
                color_idx += 1
    
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
        title="NPS Forecast Model",
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
overall_forecast_model = NPSForecastModel(sample_data)
overall_forecast_model.fit_polynomial_model()
overall_forecast_df = overall_forecast_model.create_forecast(months_ahead=3)
overall_forecast_summary = overall_forecast_model.get_forecast_summary()

# App layout
app.layout = dbc.Container([
    # Header
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
    
    
    # Key Metrics
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Overall NPS Metrics", className="card-title mb-3"),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H2(id='current-nps', children=f"{float(overall_forecast_summary['current_nps']):.1f}", className="text-primary mb-1", style={'fontSize': '2.5rem', 'fontWeight': 'bold'}),
                                html.P("Current NPS", className="text-muted mb-0", style={'fontSize': '1rem'})
                            ], className="text-center")
                        ], width=4),
                        dbc.Col([
                            html.Div([
                                html.H2(id='forecast-nps', children=f"{float(overall_forecast_summary['forecast_nps_3m']):.1f}", className="text-primary mb-1", style={'fontSize': '2.5rem', 'fontWeight': 'bold'}),
                                html.P("3-Month Forecast", className="text-muted mb-0", style={'fontSize': '1rem'})
                            ], className="text-center")
                        ], width=4),
                        dbc.Col([
                            html.Div([
                                html.H2(id='average-nps', children=f"{sample_data['NPS'].mean():.1f}", className="text-primary mb-1", style={'fontSize': '2.5rem', 'fontWeight': 'bold'}),
                                html.P("Average NPS", className="text-muted mb-0", style={'fontSize': '1rem'})
                            ], className="text-center")
                        ], width=4)
                    ])
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Chart
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
    
    # Data Summary
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Data Summary & Model Statistics", className="card-title mb-3"),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.H5("Data Overview", className="mb-3"),
                            html.P(id='data-points', children=f"Data Points: {overall_forecast_summary['data_points_used']}", className="mb-2", style={'fontSize': '1rem'}),
                            html.P(id='date-range', children=f"Date Range: {overall_forecast_summary['date_range']}", className="mb-2", style={'fontSize': '1rem'}),
                            html.P(id='nps-range', children=f"NPS Range: {sample_data['NPS'].min():.1f} to {sample_data['NPS'].max():.1f}", className="mb-2", style={'fontSize': '1rem'}),
                            html.P(id='trend', children=f"Trend: {overall_forecast_summary['trend_direction']}", className="mb-2", style={'fontSize': '1rem'}),
                            html.P(id='confidence-interval', children=f"Confidence Interval: {overall_forecast_summary['confidence_interval']}", className="mb-0", style={'fontSize': '1rem'})
                        ], width=6),
                        dbc.Col([
                            html.H5("Model Performance", className="mb-3"),
                            html.P(id='r-squared', children=f"R² (Coefficient of Determination): {overall_forecast_summary.get('r_squared', 'N/A'):.3f}" if isinstance(overall_forecast_summary.get('r_squared'), (int, float)) else "R² (Coefficient of Determination): N/A", className="mb-2", style={'fontSize': '1rem'}),
                            html.P(id='mae', children=f"MAE (Mean Absolute Error): {overall_forecast_summary.get('mae', 'N/A'):.2f}" if isinstance(overall_forecast_summary.get('mae'), (int, float)) else "MAE (Mean Absolute Error): N/A", className="mb-2", style={'fontSize': '1rem'}),
                            html.P(id='rmse', children=f"RMSE (Root Mean Square Error): {overall_forecast_summary.get('rmse', 'N/A'):.2f}" if isinstance(overall_forecast_summary.get('rmse'), (int, float)) else "RMSE (Root Mean Square Error): N/A", className="mb-2", style={'fontSize': '1rem'}),
                            html.P(id='mape', children=f"MAPE (Mean Absolute Percentage Error): {overall_forecast_summary.get('mape', 'N/A'):.2f}%" if isinstance(overall_forecast_summary.get('mape'), (int, float)) else "MAPE (Mean Absolute Percentage Error): N/A", className="mb-2", style={'fontSize': '1rem'}),
                            html.P(id='model-degree', children=f"Model Degree: {overall_forecast_summary.get('model_degree', 'N/A')}", className="mb-0", style={'fontSize': '1rem'})
                        ], width=6)
                    ])
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Chart Options
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Chart Options", className="card-title mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.H6("General Options", className="mb-2"),
                            dbc.Checklist(
                                id='chart-options',
                                options=[
                                    {'label': 'Show Individual Brands', 'value': 'show_brands'},
                                    {'label': 'Adjust for Seasonality', 'value': 'adjust_seasonality'}
                                ],
                                value=[],
                                inline=False
                            )
                        ], width=6),
                        dbc.Col([
                            html.H6("Brand Selection", className="mb-2"),
                            html.Div(id='brand-options', children=[
                                html.P("Upload data to see available brands", className="text-muted")
                            ])
                        ], width=6)
                    ])
                ])
            ])
        ], width=12)
    ], className="mb-5")
], fluid=True, style={'padding': '0 15px'})

# Callback for file upload and chart updates
@app.callback(
    [Output('upload-status', 'children'),
     Output('nps-forecast-chart', 'figure'),
     Output('current-nps', 'children'),
     Output('forecast-nps', 'children'),
     Output('average-nps', 'children'),
     Output('brand-options', 'children'),
     Output('data-points', 'children'),
     Output('date-range', 'children'),
     Output('nps-range', 'children'),
     Output('trend', 'children'),
     Output('confidence-interval', 'children'),
     Output('r-squared', 'children'),
     Output('mae', 'children'),
     Output('rmse', 'children'),
     Output('mape', 'children'),
     Output('model-degree', 'children')],
    [Input('upload-data', 'contents'),
     Input('chart-options', 'value')],
    [State('upload-data', 'filename')]
)
def update_output(contents, chart_options, filename):
    global current_data, overall_forecast_model, overall_forecast_df, overall_forecast_summary, brand_data, available_brands
    
    show_brands = 'show_brands' in chart_options
    adjust_seasonality = 'adjust_seasonality' in chart_options
    
    if contents is None:
        # Return current data with sample chart
        fig = create_simple_forecast_chart(sample_data, overall_forecast_df, {}, show_brands)
        brand_options = html.P("Upload data to see available brands", className="text-muted")
        
        # Sample data summary
        data_points = f"Data Points: {len(sample_data)}"
        date_range = f"Date Range: {sample_data['Date'].min().strftime('%Y-%m-%d')} to {sample_data['Date'].max().strftime('%Y-%m-%d')}"
        nps_range = f"NPS Range: {sample_data['NPS'].min():.1f} to {sample_data['NPS'].max():.1f}"
        trend = f"Trend: {'Increasing' if sample_data['NPS'].iloc[-1] > sample_data['NPS'].iloc[0] else 'Decreasing'}"
        confidence_interval = f"Confidence Interval: {overall_forecast_summary.get('confidence_interval', 'N/A')}"
        r_squared = f"R² (Coefficient of Determination): {float(overall_forecast_summary.get('model_stats', {}).get('r_squared', 0)):.3f}"
        mae = f"MAE (Mean Absolute Error): {float(overall_forecast_summary.get('model_stats', {}).get('mae', 0)):.2f}"
        rmse = f"RMSE (Root Mean Square Error): {float(overall_forecast_summary.get('model_stats', {}).get('rmse', 0)):.2f}"
        mape = f"MAPE (Mean Absolute Percentage Error): {float(overall_forecast_summary.get('model_stats', {}).get('mape', 0)):.2f}%"
        model_degree = f"Model Degree: {overall_forecast_summary.get('model_stats', {}).get('degree', 'N/A')}"
        
        return "", fig, f"{float(overall_forecast_summary.get('current_nps', 0)):.1f}", f"{float(overall_forecast_summary.get('forecast_nps_3m', 0)):.1f}", f"{sample_data['NPS'].mean():.1f}", brand_options, data_points, date_range, nps_range, trend, confidence_interval, r_squared, mae, rmse, mape, model_degree
    
    try:
        # Parse the uploaded file
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        if 'csv' in filename.lower():
            # Read CSV with first column as string to preserve dealership names
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), dtype={0: str})
        else:
            return dbc.Alert("Please upload a CSV file.", color="danger"), go.Figure(), "N/A", "N/A", "N/A", html.P("Error loading data", className="text-muted"), "Data Points: N/A", "Date Range: N/A", "NPS Range: N/A", "Trend: N/A", "Confidence Interval: N/A", "R² (Coefficient of Determination): N/A", "MAE (Mean Absolute Error): N/A", "RMSE (Root Mean Square Error): N/A", "MAPE (Mean Absolute Percentage Error): N/A", "Model Degree: N/A"
        
        # Process the dealership data
        df_processed = process_dealership_data(df)
        
        if len(df_processed) == 0:
            return dbc.Alert("No valid data found in the uploaded file.", color="warning"), go.Figure(), "N/A", "N/A", "N/A", html.P("Error loading data", className="text-muted"), "Data Points: N/A", "Date Range: N/A", "NPS Range: N/A", "Trend: N/A", "Confidence Interval: N/A", "R² (Coefficient of Determination): N/A", "MAE (Mean Absolute Error): N/A", "RMSE (Root Mean Square Error): N/A", "MAPE (Mean Absolute Percentage Error): N/A", "Model Degree: N/A"
        
        # Store the processed data globally
        current_data = df_processed
        
        # Create overall forecast (average across all dealerships)
        df_combined = df_processed.groupby('Date')['NPS'].mean().reset_index()
        df_combined = df_combined.sort_values('Date').reset_index(drop=True)
        
        overall_forecast_model = NPSForecastModel(df_combined)
        overall_forecast_model.fit_polynomial_model(adjust_seasonality=adjust_seasonality)
        overall_forecast_df = overall_forecast_model.create_forecast(months_ahead=3)
        overall_forecast_summary = overall_forecast_model.get_forecast_summary()
        
        # Group data by brand for individual brand lines
        brand_data = {}
        for dealership in df_processed['Dealership'].unique():
            brand = extract_brand_name(dealership)
            dealer_data = df_processed[df_processed['Dealership'] == dealership]
            
            if brand not in brand_data:
                brand_data[brand] = []
            brand_data[brand].append(dealer_data)
        
        # Combine data for each brand (average across dealerships of same brand)
        combined_brand_data = {}
        available_brands = []
        for brand, data_list in brand_data.items():
            if data_list:
                # Combine all data for this brand
                combined_data = pd.concat(data_list, ignore_index=True)
                # Group by date and take average
                brand_avg = combined_data.groupby('Date')['NPS'].mean().reset_index()
                brand_avg = brand_avg.sort_values('Date').reset_index(drop=True)
                combined_brand_data[brand] = brand_avg
                available_brands.append(brand)
        
        # Create brand selection options
        brand_options = dbc.Checklist(
            id='brand-selection',
            options=[{'label': brand, 'value': brand} for brand in available_brands],
            value=available_brands if show_brands else [],
            inline=False
        )
        
        # Create the chart
        fig = create_simple_forecast_chart(df_combined, overall_forecast_df, combined_brand_data, show_brands)
        
        # Update metrics
        current_nps = df_combined['NPS'].iloc[-1] if len(df_combined) > 0 else 0
        forecast_nps = overall_forecast_summary['forecast_nps_3m'] if overall_forecast_summary else 0
        average_nps = df_combined['NPS'].mean() if len(df_combined) > 0 else 0
        
        # Calculate data summary for uploaded data
        data_points = f"Data Points: {len(df_processed)}"
        date_range = f"Date Range: {df_processed['Date'].min().strftime('%Y-%m-%d')} to {df_processed['Date'].max().strftime('%Y-%m-%d')}"
        nps_range = f"NPS Range: {df_processed['NPS'].min():.1f} to {df_processed['NPS'].max():.1f}"
        trend = f"Trend: {'Increasing' if df_combined['NPS'].iloc[-1] > df_combined['NPS'].iloc[0] else 'Decreasing'}"
        confidence_interval = f"Confidence Interval: {overall_forecast_summary.get('confidence_interval', 'N/A')}"
        r_squared = f"R² (Coefficient of Determination): {float(overall_forecast_summary.get('model_stats', {}).get('r_squared', 0)):.3f}"
        mae = f"MAE (Mean Absolute Error): {float(overall_forecast_summary.get('model_stats', {}).get('mae', 0)):.2f}"
        rmse = f"RMSE (Root Mean Square Error): {float(overall_forecast_summary.get('model_stats', {}).get('rmse', 0)):.2f}"
        mape = f"MAPE (Mean Absolute Percentage Error): {float(overall_forecast_summary.get('model_stats', {}).get('mape', 0)):.2f}%"
        model_degree = f"Model Degree: {overall_forecast_summary.get('model_stats', {}).get('degree', 'N/A')}"
        
        return dbc.Alert(f"Successfully loaded {len(df_processed)} data points for {len(combined_brand_data)} brands!", color="success"), fig, f"{current_nps:.1f}", f"{forecast_nps:.1f}", f"{average_nps:.1f}", brand_options, data_points, date_range, nps_range, trend, confidence_interval, r_squared, mae, rmse, mape, model_degree
        
    except Exception as e:
        return dbc.Alert(f"Error processing file: {str(e)}", color="danger"), go.Figure(), "N/A", "N/A", "N/A", html.P("Error loading data", className="text-muted"), "Data Points: N/A", "Date Range: N/A", "NPS Range: N/A", "Trend: N/A", "Confidence Interval: N/A", "R² (Coefficient of Determination): N/A", "MAE (Mean Absolute Error): N/A", "RMSE (Root Mean Square Error): N/A", "MAPE (Mean Absolute Percentage Error): N/A", "Model Degree: N/A"

# Callback for brand selection and seasonality updates
@app.callback(
    [Output('nps-forecast-chart', 'figure', allow_duplicate=True),
     Output('current-nps', 'children', allow_duplicate=True),
     Output('forecast-nps', 'children', allow_duplicate=True),
     Output('average-nps', 'children', allow_duplicate=True)],
    [Input('brand-selection', 'value'),
     Input('chart-options', 'value')],
    prevent_initial_call=True
)
def update_chart_options(selected_brands, chart_options):
    global current_data, overall_forecast_model, overall_forecast_df, brand_data
    
    if current_data is None:
        return go.Figure(), "N/A", "N/A", "N/A"
    
    # Check if seasonality adjustment is enabled
    adjust_seasonality = 'adjust_seasonality' in chart_options if chart_options else False
    
    # Create overall data
    df_combined = current_data.groupby('Date')['NPS'].mean().reset_index()
    df_combined = df_combined.sort_values('Date').reset_index(drop=True)
    
    # Recreate forecast model with seasonality adjustment if needed
    if adjust_seasonality != (hasattr(overall_forecast_model, 'model_stats') and overall_forecast_model.model_stats.get('seasonality_adjusted', False)):
        overall_forecast_model = NPSForecastModel(df_combined)
        overall_forecast_model.fit_polynomial_model(adjust_seasonality=adjust_seasonality)
        overall_forecast_df = overall_forecast_model.create_forecast(months_ahead=3)
    
    # Filter brand data based on selection
    filtered_brand_data = {}
    if selected_brands:
        for brand in selected_brands:
            if brand in brand_data:
                # Combine all data for this brand
                combined_data = pd.concat(brand_data[brand], ignore_index=True)
                # Group by date and take average
                brand_avg = combined_data.groupby('Date')['NPS'].mean().reset_index()
                brand_avg = brand_avg.sort_values('Date').reset_index(drop=True)
                filtered_brand_data[brand] = brand_avg
    
    # Create the chart
    fig = create_simple_forecast_chart(df_combined, overall_forecast_df, filtered_brand_data, len(selected_brands) > 0)
    
    # Update metrics
    current_nps = df_combined['NPS'].iloc[-1] if len(df_combined) > 0 else 0
    forecast_nps = overall_forecast_model.get_forecast_summary()['forecast_nps_3m'] if overall_forecast_model else 0
    average_nps = df_combined['NPS'].mean() if len(df_combined) > 0 else 0
    
    return fig, f"{current_nps:.1f}", f"{forecast_nps:.1f}", f"{average_nps:.1f}"

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=False, host='0.0.0.0', port=port)
