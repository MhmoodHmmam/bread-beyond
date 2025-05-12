import pandas as pd
import numpy as np
from pathlib import Path

def load_data(file_path):
    """
    Load data from Excel file and perform initial preprocessing
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file
        
    Returns:
    --------
    pandas.DataFrame
        Processed DataFrame
    """
    # Load data
    df = pd.read_excel(file_path)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract date components
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    # Calculate additional metrics
    df['ROI'] = (df['Daily Revenue'] - df['Ad Spend']) / df['Ad Spend']
    df['Profit'] = df['Daily Revenue'] - df['Ad Spend']
    df['Conversion Rate'] = df['Conversions'] / df['Ad Spend']
    
    # Handle any potential missing values
    df = df.fillna({
        'Ad Spend': df['Ad Spend'].median(),
        'Conversions': df['Conversions'].median(),
        'Daily Revenue': df['Daily Revenue'].median()
    })
    
    return df

def save_processed_data(df, output_path):
    """
    Save processed data to CSV
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Processed data
    output_path : str
        Path to save the CSV file
    """
    # Create directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    return True

def main():
    """Main function to process data"""
    # Define file paths
    input_file = "data/01JTBTJ3CJ4JKZ758BZQ9YT51P.xlsx"
    output_file = "data/processed/bakery_data_processed.csv"
    
    # Load and process data
    df = load_data(input_file)
    
    # Save processed data
    save_processed_data(df, output_file)
    
    print(f"Data processed and saved to {output_file}")
    
    return df

if __name__ == "__main__":
    main()