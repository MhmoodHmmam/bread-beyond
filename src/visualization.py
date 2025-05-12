import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_revenue_by_category_plot(df):
    """
    Create a bar chart of revenue by product category
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Processed data
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plot
    """
    # Calculate revenue by category
    revenue_by_category = df.groupby('Service Category')['Daily Revenue'].sum().reset_index()
    
    # Create bar chart
    fig = px.bar(
        revenue_by_category, 
        x='Service Category', 
        y='Daily Revenue',
        title='Revenue by Product Category',
        labels={'Daily Revenue': 'Total Revenue ($)', 'Service Category': 'Product Category'},
        color='Service Category',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    return fig

def create_revenue_by_season_plot(df):
    """
    Create a bar chart of revenue by season
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Processed data
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plot
    """
    # Calculate revenue by season
    revenue_by_season = df.groupby('Season')['Daily Revenue'].sum().reset_index()
    
    # Create bar chart
    fig = px.bar(
        revenue_by_season, 
        x='Season', 
        y='Daily Revenue',
        title='Revenue by Season',
        labels={'Daily Revenue': 'Total Revenue ($)', 'Season': 'Season'},
        color='Season',
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    
    # Customize order of seasons
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    fig.update_xaxes(categoryorder='array', categoryarray=season_order)
    
    return fig

def create_seasonal_product_heatmap(df):
    """
    Create a heatmap of seasonal product performance
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Processed data
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plot
    """
    # Calculate revenue by season and product
    seasonal_product = df.groupby(['Season', 'Service Category'])['Daily Revenue'].sum().reset_index()
    
    # Create pivot table
    pivot_df = seasonal_product.pivot(index='Season', columns='Service Category', values='Daily Revenue')
    
    # Reorder seasons
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    pivot_df = pivot_df.reindex(season_order)
    
    # Create heatmap
    fig = px.imshow(
        pivot_df,
        labels=dict(x='Product Category', y='Season', color='Revenue ($)'),
        x=pivot_df.columns,
        y=pivot_df.index,
        color_continuous_scale='YlGnBu',
        title='Product Performance by Season (Revenue)'
    )
    
    # Add annotations
    for i, season in enumerate(pivot_df.index):
        for j, product in enumerate(pivot_df.columns):
            fig.add_annotation(
                x=j, 
                y=i,
                text=f"${pivot_df.iloc[i, j]:,.0f}",
                showarrow=False,
                font=dict(color='black')
            )
    
    return fig

def create_marketing_channel_plots(df):
    """
    Create marketing channel performance plots
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Processed data
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plot
    """
    # Calculate metrics by channel
    channel_metrics = df.groupby('Marketing Channel').agg({
        'Daily Revenue': 'sum',
        'Ad Spend': 'sum',
        'Conversions': 'sum'
    }).reset_index()
    
    # Calculate ROI
    channel_metrics['ROI'] = (channel_metrics['Daily Revenue'] - channel_metrics['Ad Spend']) / channel_metrics['Ad Spend']
    
    # Create subplot with 2 metrics
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Revenue by Channel', 'ROI by Channel'))
    
    # Add revenue bar chart
    fig.add_trace(
        go.Bar(
            x=channel_metrics['Marketing Channel'],
            y=channel_metrics['Daily Revenue'],
            name='Revenue',
            marker_color='teal'
        ),
        row=1, col=1
    )
    
    # Add ROI bar chart
    fig.add_trace(
        go.Bar(
            x=channel_metrics['Marketing Channel'],
            y=channel_metrics['ROI'],
            name='ROI',
            marker_color='coral'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text='Marketing Channel Performance',
        height=500,
        showlegend=False
    )
    
    fig.update_yaxes(title_text='Total Revenue ($)', row=1, col=1)
    fig.update_yaxes(title_text='ROI', row=1, col=2)
    
    return fig

def create_customer_segment_plots(df):
    """
    Create customer segment visualizations
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Processed data with customer segments
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plot
    """
    # Check if Value Segment column exists
    if 'Value Segment' not in df.columns:
        # Create value segments
        bins = [0, 10, 20, 50, np.inf]
        labels = ['Low', 'Medium', 'High', 'Premium']
        df['Value Segment'] = pd.cut(df['Purchase Value'], bins=bins, labels=labels)
    
    # Customer metrics by segment
    segment_metrics = df.groupby('Value Segment').agg({
        'Daily Revenue': 'sum',
        'Purchase Value': 'mean',
        'Conversions': 'sum'
    }).reset_index()
    
    # Create subplot with 2 metrics
    fig = make_subplots(rows=1, cols=2, 
                        specs=[[{"type": "bar"}, {"type": "pie"}]],
                        subplot_titles=('Revenue by Segment', 'Segment Distribution'))
    
    # Add revenue bar chart
    fig.add_trace(
        go.Bar(
            x=segment_metrics['Value Segment'],
            y=segment_metrics['Daily Revenue'],
            name='Revenue',
            marker_color='purple'
        ),
        row=1, col=1
    )
    
    # Add pie chart for segment distribution
    segment_counts = df['Value Segment'].value_counts().reset_index()
    segment_counts.columns = ['Value Segment', 'Count']
    
    fig.add_trace(
        go.Pie(
            labels=segment_counts['Value Segment'],
            values=segment_counts['Count'],
            name='Segment Distribution'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text='Customer Segment Analysis',
        height=500
    )
    
    fig.update_yaxes(title_text='Total Revenue ($)', row=1, col=1)
    
    return fig

def create_promotional_calendar_heatmap(promo_calendar):
    """
    Create a heatmap visualization of the promotional calendar
    
    Parameters:
    -----------
    promo_calendar : pandas.DataFrame
        Promotional calendar data
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plot
    """
    # Create pivot table
    pivot_table = pd.pivot_table(
        promo_calendar, 
        values='Estimated ROI', 
        index='Month', 
        columns='Focus Product',
        aggfunc='mean'
    )
    
    # Create heatmap
    fig = px.imshow(
        pivot_table,
        labels=dict(x='Product', y='Month', color='Estimated ROI'),
        x=pivot_table.columns,
        y=pivot_table.index,
        color_continuous_scale='YlGnBu',
        title='Promotional Calendar - Estimated ROI by Month and Product'
    )
    
    # Add annotations
    for i, month in enumerate(pivot_table.index):
        for j, product in enumerate(pivot_table.columns):
            if not pd.isna(pivot_table.iloc[i, j]):
                fig.add_annotation(
                    x=j, 
                    y=i,
                    text=f"{pivot_table.iloc[i, j]:.2f}",
                    showarrow=False,
                    font=dict(color='black')
                )
    
    # Add channel information
    channel_info = promo_calendar.groupby(['Month', 'Focus Product'])['Marketing Channel'].first().reset_index()
    channel_pivot = channel_info.pivot(index='Month', columns='Focus Product', values='Marketing Channel')
    
    for i, month in enumerate(channel_pivot.index):
        for j, product in enumerate(channel_pivot.columns):
            if not pd.isna(channel_pivot.iloc[i, j]):
                fig.add_annotation(
                    x=j, 
                    y=i,
                    text=f"{channel_pivot.iloc[i, j]}",
                    showarrow=False,
                    font=dict(color='white', size=8),
                    yshift=-15
                )
    
    return fig