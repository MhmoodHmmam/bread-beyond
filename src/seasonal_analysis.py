import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_seasonal_products(df):
    """
    Analyze product performance by season
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Processed data
        
    Returns:
    --------
    tuple
        Seasonal performance and top products by season
    """
    # Seasonal product performance
    seasonal_perf = df.groupby(['Season', 'Service Category']).agg({
        'Daily Revenue': ['sum', 'mean'],
        'Conversions': ['sum', 'mean'],
        'Ad Spend': ['sum', 'mean'],
        'ROI': 'mean'
    }).round(2)
    
    # Find top product for each season
    top_by_season = df.groupby(['Season', 'Service Category'])['Daily Revenue'].sum().reset_index()
    top_products = top_by_season.sort_values('Daily Revenue', ascending=False).groupby('Season').first()
    
    return seasonal_perf, top_products

def analyze_seasonality_over_time(df):
    """
    Analyze seasonality patterns over time
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Processed data
        
    Returns:
    --------
    pandas.DataFrame
        Time series of seasonal patterns
    """
    # Group by month and season
    monthly_seasonal = df.groupby(['Month', 'Season', 'Service Category']).agg({
        'Daily Revenue': 'sum',
        'Conversions': 'sum'
    }).reset_index()
    
    # Pivot to get seasonal patterns
    pivot_df = monthly_seasonal.pivot_table(
        index=['Month', 'Season'],
        columns='Service Category',
        values='Daily Revenue',
        aggfunc='sum'
    ).reset_index()
    
    return monthly_seasonal, pivot_df

def plot_seasonal_performance(seasonal_perf, top_products):
    """
    Create visualizations for seasonal product performance
    
    Parameters:
    -----------
    seasonal_perf : pandas.DataFrame
        Seasonal performance data
    top_products : pandas.DataFrame
        Top products by season
        
    Returns:
    --------
    matplotlib.figure.Figure
        Plot figure
    """
    # Create visualization
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Total revenue by season and product
    seasonal_data = seasonal_perf['Daily Revenue']['sum'].unstack()
    seasonal_data.plot(kind='bar', ax=ax[0])
    ax[0].set_title('Total Revenue by Season and Product')
    ax[0].set_xlabel('Season')
    ax[0].set_ylabel('Total Revenue ($)')
    
    # Plot 2: Top product by season
    top_products['Daily Revenue'].plot(kind='bar', ax=ax[1], color='teal')
    ax[1].set_title('Top Product by Season')
    ax[1].set_xlabel('Season')
    ax[1].set_ylabel('Total Revenue ($)')
    
    plt.tight_layout()
    
    return fig

def get_seasonal_recommendations(seasonal_perf, top_products):
    """
    Generate recommendations based on seasonal analysis
    
    Parameters:
    -----------
    seasonal_perf : pandas.DataFrame
        Seasonal performance data
    top_products : pandas.DataFrame
        Top products by season
        
    Returns:
    --------
    dict
        Recommendations for each season
    """
    recommendations = {}
    
    for season in top_products.index:
        top_product = top_products.loc[season, 'Service Category']
        revenue = top_products.loc[season, 'Daily Revenue']
        
        # Generate recommendation
        recommendations[season] = {
            'top_product': top_product,
            'revenue': revenue,
            'recommendation': f"Focus on {top_product} promotions during {season} season"
        }
    
    return recommendations

def main(df):
    """
    Main function for seasonal analysis
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Processed data
        
    Returns:
    --------
    dict
        Analysis results
    """
    # Perform seasonal analysis
    seasonal_perf, top_products = analyze_seasonal_products(df)
    
    # Analyze seasonality over time
    monthly_seasonal, pivot_df = analyze_seasonality_over_time(df)
    
    # Get recommendations
    recommendations = get_seasonal_recommendations(seasonal_perf, top_products)
    
    # Create plot
    fig = plot_seasonal_performance(seasonal_perf, top_products)
    
    # Return results
    results = {
        'seasonal_performance': seasonal_perf,
        'top_products': top_products,
        'monthly_seasonal': monthly_seasonal,
        'recommendations': recommendations,
        'figure': fig
    }
    
    return results