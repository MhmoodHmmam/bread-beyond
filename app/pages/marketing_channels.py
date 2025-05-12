import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import modules
from src.data_processing import load_data
from src.marketing_analysis import analyze_marketing_channels, calculate_channel_efficiency, analyze_channel_product_performance, get_marketing_recommendations
from src.visualization import create_marketing_channel_plots

# Set page config
st.set_page_config(
    page_title="Marketing Channels - Data Analytics",
    page_icon="🧁",
    layout="wide"
)

# Load data
@st.cache_data
def load_cached_data():
    try:
        return load_data("data/01JTBTJ3CJ4JKZ758BZQ9YT51P.xlsx")
    except FileNotFoundError:
        st.error("Data file not found. Please place the data file in the data directory.")
        return None

df = load_cached_data()

if df is None:
    st.stop()

# Page content
st.title("Marketing Channel Analysis")

st.write("""
This page analyzes the performance of different marketing channels to identify the most 
effective channels for each season and product category.
""")

# Analyze marketing channels
channel_perf, best_channels, channel_by_season = analyze_marketing_channels(df)
efficiency_metrics = calculate_channel_efficiency(df)
channel_product, best_products = analyze_channel_product_performance(df)

# Display marketing channel performance
st.subheader("Marketing Channel Performance")

channel_plots = create_marketing_channel_plots(df)
st.plotly_chart(channel_plots, use_container_width=True)

# Efficiency metrics
st.subheader("Marketing Channel Efficiency Metrics")

# Format the efficiency metrics dataframe
efficiency_formatted = efficiency_metrics.copy()
efficiency_formatted = efficiency_formatted.reset_index()

# Display as table
st.dataframe(efficiency_formatted)

# ROI comparison
st.subheader("ROI by Marketing Channel")

roi_data = efficiency_metrics['ROI'].sort_values(ascending=False)

fig = px.bar(
    roi_data,
    title='Return on Investment (ROI) by Marketing Channel',
    labels={'value': 'ROI', 'index': 'Marketing Channel'},
    color=roi_data.values,
    color_continuous_scale='RdYlGn'
)

st.plotly_chart(fig, use_container_width=True)

# Cost per conversion comparison
st.subheader("Cost per Conversion by Channel")

cpc_data = efficiency_metrics['Cost per Conversion'].sort_values()

fig = px.bar(
    cpc_data,
    title='Cost per Conversion by Marketing Channel',
    labels={'value': 'Cost per Conversion ($)', 'index': 'Marketing Channel'},
    color=cpc_data.values,
    color_continuous_scale='RdYlGn_r'
)

st.plotly_chart(fig, use_container_width=True)

# Channel performance by season
st.subheader("Channel Performance by Season")

# Create line chart
channel_season_data = channel_by_season.pivot(index='Season', columns='Marketing Channel', values='ROI')

# Reorder seasons
season_order = ['Winter', 'Spring', 'Summer', 'Fall']
channel_season_data = channel_season_data.reindex(season_order)

fig = px.line(
    channel_season_data,
    title='ROI by Season and Marketing Channel',
    labels={'value': 'ROI', 'index': 'Season'},
    markers=True
)

st.plotly_chart(fig, use_container_width=True)

# Display best channels by season
col1, col2, col3, col4 = st.columns(4)

seasons = ["Spring", "Summer", "Fall", "Winter"]
cols = [col1, col2, col3, col4]

for i, season in enumerate(seasons):
    if season in best_channels.index:
        channel = best_channels.loc[season, 'Marketing Channel']
        roi = best_channels.loc[season, 'ROI']
        cols[i].metric(f"Best in {season}", channel, f"ROI: {roi:.2f}")

# Channel-product performance
st.subheader("Marketing Channel Performance by Product")

# Create heatmap of channel-product performance
pivot_df = channel_product.pivot(
    index='Marketing Channel',
    columns='Service Category',
    values='Daily Revenue'
)

fig = px.imshow(
    pivot_df,
    labels=dict(x='Product Category', y='Marketing Channel', color='Revenue ($)'),
    x=pivot_df.columns,
    y=pivot_df.index,
    color_continuous_scale='Viridis',
    title='Marketing Channel Performance by Product (Revenue)'
)

# Add annotations
for i, channel in enumerate(pivot_df.index):
    for j, product in enumerate(pivot_df.columns):
        fig.add_annotation(
            x=j, 
            y=i,
            text=f"${pivot_df.iloc[i, j]:,.0f}",
            showarrow=False,
            font=dict(color='white')
        )

st.plotly_chart(fig, use_container_width=True)

# Interactive channel-product analysis
st.subheader("Interactive Channel-Product Analysis")

col1, col2 = st.columns(2)

with col1:
    selected_channel = st.selectbox("Select Marketing Channel:", df['Marketing Channel'].unique())

with col2:
    selected_product = st.selectbox("Select Product:", df['Service Category'].unique())

if selected_channel and selected_product:
    # Filter data
    filtered_data = df[
        (df['Marketing Channel'] == selected_channel) & 
        (df['Service Category'] == selected_product)
    ]
    
    # Calculate metrics
    metrics = {
        'Total Revenue': filtered_data['Daily Revenue'].sum(),
        'Total Ad Spend': filtered_data['Ad Spend'].sum(),
        'Total Conversions': filtered_data['Conversions'].sum(),
        'Average ROI': filtered_data['ROI'].mean(),
        'Revenue per Conversion': filtered_data['Daily Revenue'].sum() / filtered_data['Conversions'].sum()
    }
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Total Revenue", f"${metrics['Total Revenue']:,.2f}")
    col2.metric("Total Ad Spend", f"${metrics['Total Ad Spend']:,.2f}")
    col3.metric("Total Conversions", f"{metrics['Total Conversions']:,.0f}")
    
    col1, col2 = st.columns(2)
    
    col1.metric("Average ROI", f"{metrics['Average ROI']:.2f}")
    col2.metric("Revenue per Conversion", f"${metrics['Revenue per Conversion']:,.2f}")
    
    # Performance by season
    seasonal_perf = filtered_data.groupby('Season').agg({
        'Daily Revenue': 'sum',
        'Conversions': 'sum',
        'ROI': 'mean'
    }).reset_index()
    
    fig = px.bar(
        seasonal_perf,
        x='Season',
        y='Daily Revenue',
        color='Season',
        title=f'Seasonal Performance: {selected_channel} - {selected_product}',
        labels={'Daily Revenue': 'Total Revenue ($)', 'Season': 'Season'}
    )
    
    # Reorder seasons
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    fig.update_xaxes(categoryorder='array', categoryarray=season_order)
    
    st.plotly_chart(fig, use_container_width=True)

# Marketing recommendations
st.subheader("Marketing Channel Recommendations")

recommendations = get_marketing_recommendations(channel_perf, best_channels, channel_product)

for key, rec in recommendations.items():
    st.info(f"**{key.replace('_', ' ').title()}**: {rec}")

# Budget allocation tool
st.subheader("Marketing Budget Allocation Tool")

st.write("""
Use this tool to simulate different budget allocations across marketing channels.
""")

# Get total ad spend
total_budget = df['Ad Spend'].sum()

st.write(f"Total Current Ad Spend: ${total_budget:,.2f}")

# Sliders for budget allocation
st.write("Adjust budget allocation percentages:")

col1, col2 = st.columns(2)

channel_allocations = {}
channels = df['Marketing Channel'].unique()

for i, channel in enumerate(channels):
    current_spend = df[df['Marketing Channel'] == channel]['Ad Spend'].sum()
    current_pct = (current_spend / total_budget) * 100
    
    if i % 2 == 0:
        column = col1
    else:
        column = col2
        
    channel_allocations[channel] = column.slider(
        f"{channel} (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(current_pct),
        step=1.0
    )

# Check if allocations sum to 100%
total_allocation = sum(channel_allocations.values())

if abs(total_allocation - 100.0) > 0.01:
    st.warning(f"Total allocation: {total_allocation:.1f}% (should be 100%)")
else:
    st.success("Budget allocation is valid (100%)")
    
    # Calculate projected revenue based on ROI
    projected_revenue = 0
    channel_projections = {}
    
    for channel, allocation in channel_allocations.items():
        channel_budget = (allocation / 100) * total_budget
        channel_roi = efficiency_metrics.loc[channel, 'ROI']
        channel_revenue = channel_budget * (1 + channel_roi)
        
        channel_projections[channel] = {
            'Budget': channel_budget,
            'ROI': channel_roi,
            'Projected Revenue': channel_revenue
        }
        
        projected_revenue += channel_revenue
    
    # Display projections
    st.subheader("Projected Results")
    
    col1, col2 = st.columns(2)
    
    # Current metrics
    current_revenue = df['Daily Revenue'].sum()
    current_roi = (current_revenue - total_budget) / total_budget
    
    col1.metric(
        "Current Revenue", 
        f"${current_revenue:,.2f}", 
        f"ROI: {current_roi:.2f}"
    )
    
    # Projected metrics
    projected_roi = (projected_revenue - total_budget) / total_budget
    
    col2.metric(
        "Projected Revenue", 
        f"${projected_revenue:,.2f}", 
        f"{((projected_revenue - current_revenue) / current_revenue) * 100:.1f}%"
    )
    
    # Display channel projections
    st.write("Channel Projections:")
    
    projection_df = pd.DataFrame.from_dict(channel_projections, orient='index')
    st.dataframe(projection_df)
    
    # Visualize projections
    fig = px.bar(
        projection_df,
        y=projection_df.index,
        x='Projected Revenue',
        orientation='h',
        title='Projected Revenue by Channel',
        labels={'Projected Revenue': 'Projected Revenue ($)', 'index': 'Marketing Channel'},
        color='ROI',
        color_continuous_scale='RdYlGn'
    )
    
    st.plotly_chart(fig, use_container_width=True)