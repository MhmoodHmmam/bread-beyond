import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_marketing_channels(df):
    """
    Analyze marketing channel performance
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Processed data
        
    Returns:
    --------
    tuple
        Marketing channel performance and best channels by season
    """
    # Overall channel performance
    channel_perf = df.groupby('Marketing Channel').agg({
        'Daily Revenue': ['sum', 'mean'],
        'Conversions': ['sum', 'mean'],
        'Ad Spend': ['sum', 'mean'],
        'ROI': 'mean'
    }).round(2)
    
    # Channel performance by season
    channel_by_season = df.groupby(['Season', 'Marketing Channel']).agg({
        'Daily Revenue': 'sum',
        'ROI': 'mean',
        'Conversions': 'sum',
        'Ad Spend': 'sum'
    }).reset_index()
    
    # Find best channel for each season
    best_channels = channel_by_season.sort_values('ROI', ascending=False).groupby('Season').first()
    
    return channel_perf, best_channels, channel_by_season

def calculate_channel_efficiency(df):
    """
    Calculate marketing channel efficiency metrics
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Processed data
        
    Returns:
    --------
    pandas.DataFrame
        Channel efficiency metrics
    """
    # Calculate efficiency metrics
    efficiency = df.groupby('Marketing Channel').agg({
        'Daily Revenue': 'sum',
        'Ad Spend': 'sum',
        'Conversions': 'sum'
    })
    
    # Calculate derived metrics
    efficiency['Revenue per Conversion'] = efficiency['Daily Revenue'] / efficiency['Conversions']
    efficiency['Cost per Conversion'] = efficiency['Ad Spend'] / efficiency['Conversions']
    efficiency['ROI'] = (efficiency['Daily Revenue'] - efficiency['Ad Spend']) / efficiency['Ad Spend']
    efficiency['ROAS'] = efficiency['Daily Revenue'] / efficiency['Ad Spend']
    
    return efficiency.round(2)

def analyze_channel_product_performance(df):
    """
    Analyze marketing channel performance by product
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Processed data
        
    Returns:
    --------
    pandas.DataFrame
        Channel performance by product
    """
    # Channel performance by product
    channel_product = df.groupby(['Marketing Channel', 'Service Category']).agg({
        'Daily Revenue': 'sum',
        'ROI': 'mean',
        'Conversions': 'sum'
    }).reset_index()
    
    # Find best product for each channel
    best_products = channel_product.sort_values('Daily Revenue', ascending=False).groupby('Marketing Channel').first()
    
    return channel_product, best_products

def plot_marketing_performance(channel_perf, channel_by_season):
    """
    Create visualizations for marketing channel performance
    
    Parameters:
    -----------
    channel_perf : pandas.DataFrame
        Marketing channel performance
    channel_by_season : pandas.DataFrame
        Channel performance by season
        
    Returns:
    --------
    matplotlib.figure.Figure
        Plot figure
    """
    # Create visualization
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Overall channel performance
    channel_revenue = channel_perf['Daily Revenue']['sum']
    channel_revenue.plot(kind='bar', ax=ax[0], color='teal')
    ax[0].set_title('Total Revenue by Marketing Channel')
    ax[0].set_xlabel('Marketing Channel')
    ax[0].set_ylabel('Total Revenue ($)')
    
    # Plot 2: ROI by channel
    channel_roi = channel_perf['ROI']['mean']
    channel_roi.plot(kind='bar', ax=ax[1], color='coral')
    ax[1].set_title('Average ROI by Marketing Channel')
    ax[1].set_xlabel('Marketing Channel')
    ax[1].set_ylabel('ROI')
    
    plt.tight_layout()
    
    return fig

def create_promotional_calendar(top_products, best_channels):
    """
    Create a promotional calendar based on analysis
    
    Parameters:
    -----------
    top_products : pandas.DataFrame
        Top products by season
    best_channels : pandas.DataFrame
        Best channels by season
        
    Returns:
    --------
    pandas.DataFrame
        Promotional calendar
    """
    # Combine top products and best channels
    promo_calendar = pd.merge(
        top_products.reset_index(), 
        best_channels[['Marketing Channel', 'ROI']].reset_index(),
        on='Season'
    )
    
    # Map seasons to months
    season_months = {
        'Winter': [12, 1, 2],
        'Spring': [3, 4, 5],
        'Summer': [6, 7, 8],
        'Fall': [9, 10, 11]
    }
    
    # Create detailed promotional calendar
    detailed_calendar = []
    current_year = pd.Timestamp.now().year
    
    for _, row in promo_calendar.iterrows():
        season = row['Season']
        product = row['Service Category']
        channel = row['Marketing Channel']
        
        for month in season_months[season]:
            detailed_calendar.append({
                'Month': month,
                'Year': current_year,
                'Season': season,
                'Focus Product': product,
                'Marketing Channel': channel,
                'Estimated ROI': row['ROI']
            })
    
    return pd.DataFrame(detailed_calendar)

def get_marketing_recommendations(channel_perf, best_channels, channel_product):
    """
    Generate recommendations based on marketing analysis
    
    Parameters:
    -----------
    channel_perf : pandas.DataFrame
        Channel performance
    best_channels : pandas.DataFrame
        Best channels by season
    channel_product : pandas.DataFrame
        Channel performance by product
        
    Returns:
    --------
    dict
        Marketing recommendations
    """
    recommendations = {}
    
    # Overall channel recommendations
    channel_roi = channel_perf['ROI']['mean']
    best_channel = channel_roi.idxmax()
    worst_channel = channel_roi.idxmin()
    
    recommendations['overall'] = f"Focus marketing budget on {best_channel} which has the highest ROI"
    recommendations['optimization'] = f"Optimize or reduce spend on {worst_channel} which has the lowest ROI"
    
    # Seasonal channel recommendations
    for season in best_channels.index:
        channel = best_channels.loc[season, 'Marketing Channel']
        recommendations[f"season_{season}"] = f"Use {channel} for {season} promotions"
    
    return recommendations

def main(df):
    """
    Main function for marketing analysis
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Processed data
        
    Returns:
    --------
    dict
        Analysis results
    """
    # Analyze marketing channels
    channel_perf, best_channels, channel_by_season = analyze_marketing_channels(df)
    
    # Calculate channel efficiency
    efficiency = calculate_channel_efficiency(df)
    
    # Analyze channel-product performance
    channel_product, best_products = analyze_channel_product_performance(df)
    
    # Get recommendations
    recommendations = get_marketing_recommendations(channel_perf, best_channels, channel_product)
    
    # Create promotional calendar
    from seasonal_analysis import analyze_seasonal_products
    seasonal_perf, top_products = analyze_seasonal_products(df)
    promo_calendar = create_promotional_calendar(top_products, best_channels)
    
    # Create plot
    fig = plot_marketing_performance(channel_perf, channel_by_season)
    
    # Return results
    results = {
        'channel_performance': channel_perf,
        'best_channels': best_channels,
        'channel_by_season': channel_by_season,
        'efficiency_metrics': efficiency,
        'channel_product': channel_product,
        'best_products_by_channel': best_products,
        'promotional_calendar': promo_calendar,
        'recommendations': recommendations,
        'figure': fig
    }
    
    return results