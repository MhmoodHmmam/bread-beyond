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
from src.seasonal_analysis import analyze_seasonal_products, analyze_seasonality_over_time, get_seasonal_recommendations
from src.visualization import create_seasonal_product_heatmap

# Set page config
st.set_page_config(
    page_title="Seasonal Insights - Data Analytics",
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
st.title("Seasonal Product Analysis")

st.write("""
This page provides detailed insights into how product performance varies by season,
helping to identify opportunities for seasonal promotions and product focus.
""")

# Perform seasonal analysis
seasonal_perf, top_products = analyze_seasonal_products(df)
monthly_seasonal, pivot_df = analyze_seasonality_over_time(df)

# Display seasonal metrics
st.subheader("Seasonal Performance Metrics")

# Convert multi-index to regular columns
seasonal_perf_flat = seasonal_perf.copy()
seasonal_perf_flat.columns = ['_'.join(col).strip() for col in seasonal_perf_flat.columns.values]
seasonal_perf_flat = seasonal_perf_flat.reset_index()

# Allow filtering by product or season
filter_option = st.radio("Filter by:", ["None", "Product", "Season"], horizontal=True)

if filter_option == "Product":
    selected_product = st.selectbox("Select Product:", df['Service Category'].unique())
    filtered_df = seasonal_perf_flat[seasonal_perf_flat['Service Category'] == selected_product]
elif filter_option == "Season":
    selected_season = st.selectbox("Select Season:", df['Season'].unique())
    filtered_df = seasonal_perf_flat[seasonal_perf_flat['Season'] == selected_season]
else:
    filtered_df = seasonal_perf_flat

# Display metrics
st.dataframe(filtered_df)

# Seasonal heatmap
st.subheader("Product Performance by Season")
seasonal_heatmap = create_seasonal_product_heatmap(df)
st.plotly_chart(seasonal_heatmap, use_container_width=True)

# Time series analysis
st.subheader("Seasonal Trends Over Time")

# Create time series plot
monthly_data = df.groupby(['Month', 'Season', 'Service Category'])['Daily Revenue'].sum().reset_index()

# Allow product selection for time series
selected_products = st.multiselect(
    "Select Products for Time Series:",
    df['Service Category'].unique(),
    default=df['Service Category'].unique()[0]
)

if selected_products:
    filtered_monthly = monthly_data[monthly_data['Service Category'].isin(selected_products)]
    
    fig = px.line(
        filtered_monthly, 
        x='Month', 
        y='Daily Revenue',
        color='Service Category',
        facet_col='Season',
        facet_col_wrap=2,
        title='Monthly Revenue by Season and Product',
        labels={'Daily Revenue': 'Total Revenue ($)', 'Month': 'Month'},
        markers=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Interactive seasonal comparison
st.subheader("Interactive Seasonal Comparison")

col1, col2 = st.columns(2)

with col1:
    season1 = st.selectbox("Select First Season:", df['Season'].unique(), index=0)
    
with col2:
    season2 = st.selectbox("Select Second Season:", df['Season'].unique(), index=2)

if season1 != season2:
    # Get data for selected seasons
    season_data = df[df['Season'].isin([season1, season2])]
    
    # Group by season and product
    comparison_data = season_data.groupby(['Season', 'Service Category']).agg({
        'Daily Revenue': 'sum',
        'Conversions': 'sum',
        'ROI': 'mean'
    }).reset_index()
    
    # Create comparison chart
    fig = px.bar(
        comparison_data,
        x='Service Category',
        y='Daily Revenue',
        color='Season',
        barmode='group',
        title=f'Revenue Comparison: {season1} vs {season2}',
        labels={'Daily Revenue': 'Total Revenue ($)', 'Service Category': 'Product Category'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show percentage differences
    st.subheader("Percentage Differences")
    
    # Pivot data for calculation
    pivot_compare = comparison_data.pivot(index='Service Category', columns='Season', values='Daily Revenue')
    
    # Calculate percentage difference
    pivot_compare['Pct_Diff'] = ((pivot_compare[season2] - pivot_compare[season1]) / pivot_compare[season1]) * 100
    
    # Display as dataframe
    st.dataframe(pivot_compare)
    
    # Product recommendations based on comparison
    st.subheader("Product Recommendations")
    
    for product in pivot_compare.index:
        pct_diff = pivot_compare.loc[product, 'Pct_Diff']
        
        if pct_diff > 10:
            st.success(f"**{product}**: Performs {pct_diff:.1f}% better in {season2} than {season1}. Focus promotions in {season2}.")
        elif pct_diff < -10:
            st.warning(f"**{product}**: Performs {abs(pct_diff):.1f}% better in {season1} than {season2}. Focus promotions in {season1}.")
        else:
            st.info(f"**{product}**: Performs similarly in both seasons (difference: {pct_diff:.1f}%).")

# Seasonal recommendations
st.subheader("Seasonal Action Plan")

recommendations = get_seasonal_recommendations(seasonal_perf, top_products)

for season, rec in recommendations.items():
    with st.expander(f"{season} Strategy"):
        st.write(f"**Top Product:** {rec['top_product']}")
        st.write(f"**Revenue:** ${rec['revenue']:,.2f}")
        st.write(f"**Recommendation:** {rec['recommendation']}")
        
        # Get product data for this season
        season_product_data = df[df['Season'] == season].groupby('Service Category').agg({
            'Daily Revenue': 'sum',
            'Conversions': 'sum',
            'ROI': 'mean'
        }).reset_index()
        
        # Create bar chart
        fig = px.bar(
            season_product_data,
            x='Service Category',
            y='Daily Revenue',
            color='Service Category',
            title=f'Product Performance in {season}',
            labels={'Daily Revenue': 'Total Revenue ($)', 'Service Category': 'Product'},
        )
        st.plotly_chart(fig, use_container_width=True)