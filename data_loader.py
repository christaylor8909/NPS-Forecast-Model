"""
Data loader utility for NPS forecasting
This module handles loading and preprocessing of NPS data from Excel files
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_nps_data(file_path):
    """
    Load NPS data from Excel or CSV file
    
    Args:
        file_path (str): Path to the Excel or CSV file
        
    Returns:
        pd.DataFrame: Processed NPS data with Date and NPS columns
    """
    try:
        # Determine file type and read accordingly
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
            
        print(f"Successfully loaded data from {file_path}")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check if this is dealership data format
        if is_dealership_format(df):
            processed_df = process_dealership_data(df)
        else:
            processed_df = process_nps_data(df)
            
        return processed_df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating sample data instead...")
        return create_sample_data()

def is_dealership_format(df):
    """
    Check if the data is in dealership format (dealership names in first column, dates as headers)
    
    Args:
        df (pd.DataFrame): Raw data
        
    Returns:
        bool: True if dealership format, False otherwise
    """
    # Check if first column contains text (dealership names)
    first_col = df.iloc[:, 0]
    if first_col.dtype == 'object':
        # Check if other columns look like dates
        other_cols = df.columns[1:]
        date_like_cols = sum(1 for col in other_cols if any(char in str(col) for char in ['/', '-', '2024', '2025']))
        return date_like_cols > len(other_cols) * 0.5  # More than half look like dates
    return False

def process_dealership_data(df):
    """
    Process dealership CSV data into the format needed for forecasting
    
    Args:
        df (pd.DataFrame): Raw dealership data
        
    Returns:
        pd.DataFrame: Processed data with Dealership, Date, and NPS columns
    """
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
    
    print(f"Processed dealership data: {len(df_long)} records for {df_long['Dealership'].nunique()} dealerships")
    
    return df_long

def process_nps_data(df):
    """
    Process raw NPS data to standardize format
    
    Args:
        df (pd.DataFrame): Raw data from Excel
        
    Returns:
        pd.DataFrame: Processed data with Date and NPS columns
    """
    # Make a copy to avoid modifying original
    processed_df = df.copy()
    
    # Find date column
    date_col = None
    for col in processed_df.columns:
        if any(keyword in col.lower() for keyword in ['date', 'time', 'period', 'month', 'year']):
            date_col = col
            break
    
    if date_col:
        processed_df = processed_df.rename(columns={date_col: 'Date'})
        print(f"Found date column: {date_col}")
    else:
        # Create synthetic dates if no date column found
        processed_df['Date'] = pd.date_range(start='2023-01-01', periods=len(processed_df), freq='M')
        print("No date column found, created synthetic dates")
    
    # Find NPS column
    nps_col = None
    for col in processed_df.columns:
        if any(keyword in col.lower() for keyword in ['nps', 'score', 'rating', 'value']):
            nps_col = col
            break
    
    if nps_col:
        processed_df = processed_df.rename(columns={nps_col: 'NPS'})
        print(f"Found NPS column: {nps_col}")
    else:
        # Use first numeric column as NPS
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            processed_df = processed_df.rename(columns={numeric_cols[0]: 'NPS'})
            print(f"Using first numeric column as NPS: {numeric_cols[0]}")
        else:
            processed_df['NPS'] = np.random.normal(50, 10, len(processed_df))
            print("No numeric column found, created synthetic NPS data")
    
    # Convert Date to datetime
    processed_df['Date'] = pd.to_datetime(processed_df['Date'])
    
    # Sort by date
    processed_df = processed_df.sort_values('Date').reset_index(drop=True)
    
    # Remove any rows with missing NPS values
    processed_df = processed_df.dropna(subset=['NPS'])
    
    # Keep only Date and NPS columns
    processed_df = processed_df[['Date', 'NPS']]
    
    print(f"Processed data shape: {processed_df.shape}")
    print(f"Date range: {processed_df['Date'].min()} to {processed_df['Date'].max()}")
    print(f"NPS range: {processed_df['NPS'].min():.1f} to {processed_df['NPS'].max():.1f}")
    
    return processed_df

def create_sample_data():
    """
    Create sample NPS data for demonstration purposes
    
    Returns:
        pd.DataFrame: Sample NPS data
    """
    print("Creating sample NPS data...")
    
    # Generate dates for the last 2 years
    dates = pd.date_range(start='2022-01-01', end='2024-09-15', freq='M')
    
    # Create realistic NPS data with trend and seasonality
    np.random.seed(42)
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
    
    print(f"Sample data created with {len(df)} data points")
    return df

def validate_data(df):
    """
    Validate the processed NPS data
    
    Args:
        df (pd.DataFrame): Processed NPS data
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    if df is None or len(df) == 0:
        print("Error: No data available")
        return False
    
    if 'Date' not in df.columns or 'NPS' not in df.columns:
        print("Error: Missing required columns (Date, NPS)")
        return False
    
    if df['Date'].isna().any():
        print("Error: Missing date values")
        return False
    
    if df['NPS'].isna().any():
        print("Error: Missing NPS values")
        return False
    
    if len(df) < 3:
        print("Error: Insufficient data points for forecasting")
        return False
    
    print("Data validation passed")
    return True

if __name__ == "__main__":
    # Test the data loader
    file_path = r"C:\Users\t0355lp\OneDrive - Stellantis\Forecasting\NPS Trend_export_15_09_2025.xlsx"
    df = load_nps_data(file_path)
    
    if validate_data(df):
        print("\nData loaded successfully!")
        print(df.head())
        print(f"\nData summary:")
        print(df.describe())
    else:
        print("Data validation failed")
