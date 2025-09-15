import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_loader import load_nps_data
from forecast_model import NPSForecastModel

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "NPS Forecast Model"

# Load data using our data loader
data_path = r"C:\Users\t0355lp\OneDrive - Stellantis\Forecasting\NPS Trend_export_15_09_2025.xlsx"
df = load_nps_data(data_path)

# Create forecast model
forecast_model = NPSForecastModel(df)
forecast_model.fit_polynomial_model()
forecast_df = forecast_model.create_forecast(months_ahead=3)

# Get forecast summary
forecast_summary = forecast_model.get_forecast_summary()

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("NPS Forecast Model", className="text-center mb-4"),
            html.P("Forecasting NPS trends for the next 3 months", className="text-center text-muted mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Key Metrics", className="card-title"),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.H5(f"{forecast_summary['current_nps']:.1f}", className="text-primary"),
                            html.P("Current NPS", className="text-muted mb-0")
                        ]),
                        dbc.Col([
                            html.H5(f"{forecast_summary['forecast_nps_3m']:.1f}", className="text-success"),
                            html.P("3-Month Forecast", className="text-muted mb-0")
                        ]),
                        dbc.Col([
                            html.H5(f"{df['NPS'].mean():.1f}", className="text-info"),
                            html.P("Average NPS", className="text-muted mb-0")
                        ])
                    ])
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("NPS Forecast Chart", className="card-title"),
                    dcc.Graph(id='nps-forecast-chart')
                ])
            ])
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Data Summary", className="card-title"),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.P(f"Data Points: {forecast_summary['data_points_used']}", className="mb-1"),
                            html.P(f"Date Range: {forecast_summary['date_range']}", className="mb-1"),
                            html.P(f"NPS Range: {df['NPS'].min():.1f} to {df['NPS'].max():.1f}", className="mb-1"),
                            html.P(f"Trend: {forecast_summary['trend_direction']}", className="mb-1"),
                            html.P(f"Confidence Interval: {forecast_summary['confidence_interval']}", className="mb-0")
                        ])
                    ])
                ])
            ])
        ], width=12)
    ], className="mt-4")
], fluid=True)

# Callback for the chart
@app.callback(
    Output('nps-forecast-chart', 'figure'),
    Input('nps-forecast-chart', 'id')
)
def update_chart(_):
    # Use the forecast model's plot method
    fig = forecast_model.plot_forecast()
    return fig

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8050)
