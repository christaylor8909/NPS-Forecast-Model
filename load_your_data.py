"""
Script to help you load your actual NPS data
Run this script to test loading your data before using the main app
"""

import pandas as pd
import os
from data_loader import load_nps_data, validate_data

def main():
    print("NPS Data Loader - Test Script")
    print("=" * 40)
    
    # Your data file path
    data_path = r"C:\Users\t0355lp\OneDrive - Stellantis\Forecasting\NPS Trend_export_15_09_2025.xlsx"
    
    print(f"Attempting to load data from:")
    print(f"  {data_path}")
    print()
    
    # Check if file exists
    if not os.path.exists(data_path):
        print("❌ ERROR: File not found!")
        print("Please check the file path and make sure the file exists.")
        return
    
    # Check if file is accessible
    try:
        # Try to read the file
        df_raw = pd.read_excel(data_path)
        print("✅ File is accessible")
        print(f"   Raw data shape: {df_raw.shape}")
        print(f"   Columns: {list(df_raw.columns)}")
        print()
        
        # Show first few rows
        print("First 5 rows of raw data:")
        print(df_raw.head())
        print()
        
    except Exception as e:
        print(f"❌ ERROR: Cannot read file - {e}")
        print("\nPossible solutions:")
        print("1. Make sure the file is not open in Excel")
        print("2. Check file permissions")
        print("3. Try copying the file to a different location")
        return
    
    # Try to load and process the data
    print("Processing data...")
    try:
        df_processed = load_nps_data(data_path)
        
        if validate_data(df_processed):
            print("✅ Data loaded and processed successfully!")
            print()
            print("Processed data summary:")
            print(f"   Shape: {df_processed.shape}")
            print(f"   Date range: {df_processed['Date'].min()} to {df_processed['Date'].max()}")
            print(f"   NPS range: {df_processed['NPS'].min():.1f} to {df_processed['NPS'].max():.1f}")
            print(f"   Average NPS: {df_processed['NPS'].mean():.1f}")
            print()
            
            print("First 5 rows of processed data:")
            print(df_processed.head())
            print()
            
            print("✅ Your data is ready to use with the forecast app!")
            print("   Run: python app.py")
            
        else:
            print("❌ Data validation failed")
            
    except Exception as e:
        print(f"❌ ERROR: Data processing failed - {e}")
        print("\nThe app will use sample data instead.")

if __name__ == "__main__":
    main()
