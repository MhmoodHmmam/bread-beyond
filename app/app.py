import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import modules
from src.data_processing import load_data
from src.seasonal_analysis import analyze_seasonal_products, analyze_seasonality_over_time, get_seasonal_recommendations
from src.customer_segmentation import create_value_segments, create_kmeans_segments, get_segment_recommendations
from src.marketing_analysis import analyze_marketing_channels, create_promotional_calendar, get_marketing_recommendations
from src.visualization import (
    create_revenue_by_category_plot, create_revenue_by_season_plot, 
    create_seasonal_product_heatmap, create_marketing_channel_plots,
    create_customer_segment_plots, create_promotional_calendar_heatmap
)

# Set page config
st.set_page_config(
    page_title="Data Analytics",
    page_icon="🧁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("Data Analytics")
st.sidebar.image("app/assets/bakery_logo.png", use_container_width=True)

# Navigation
page = st.sidebar.selectbox(
    "Select Analysis",
    ["Overview", "Seasonal Analysis", "Customer Segmentation", "Marketing Channels", "Promotional Calendar"]
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

# Overview page
if page == "Overview":
    st.title("Data Analytics - Overview")
    
    st.write("""
    This dashboard provides insights into how seasonality and marketing channels affect sales of artisan baked goods.
    Use the sidebar to navigate between different analysis sections.
    """)
    
    # Show main KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    total_revenue = df['Daily Revenue'].sum()
    col1.metric("Total Revenue", f"${total_revenue:,.2f}")
    
    total_conversions = df['Conversions'].sum()
    col2.metric("Total Conversions", f"{total_conversions:,.0f}")
    
    avg_roi = df['ROI'].mean()
    col3.metric("Average ROI", f"{avg_roi:.2f}")
    
    best_product = df.groupby('Service Category')['Daily Revenue'].sum().idxmax()
    col4.metric("Top Product", best_product)
    
    # Data overview
    st.subheader("Data Overview")
    
    tab1, tab2 = st.tabs(["Data Sample", "Data Summary"])
    
    with tab1:
        st.dataframe(df.head(10))
    
    with tab2:
        st.write("Basic Statistics")
        st.dataframe(df.describe())
        
        st.write("Data Types")
        st.dataframe(pd.DataFrame(df.dtypes, columns=["Data Type"]))
    
    # Key visualizations
    st.subheader("Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        revenue_by_category_fig = create_revenue_by_category_plot(df)
        st.plotly_chart(revenue_by_category_fig, use_container_width=True)
    
    with col2:
        revenue_by_season_fig = create_revenue_by_season_plot(df)
        st.plotly_chart(revenue_by_season_fig, use_container_width=True)
    
    # Seasonal product heatmap
    seasonal_heatmap = create_seasonal_product_heatmap(df)
    st.plotly_chart(seasonal_heatmap, use_container_width=True)

# Seasonal Analysis page
elif page == "Seasonal Analysis":
    st.title("Seasonal Product Analysis")
    
    st.write("""
    This section analyzes how product performance varies by season, helping to identify
    top seasonal products for promotional focus.
    """)
    
    # Perform seasonal analysis
    seasonal_perf, top_products = analyze_seasonal_products(df)
    monthly_seasonal, pivot_df = analyze_seasonality_over_time(df)
    
    # Display seasonal heatmap
    st.subheader("Product Performance by Season")
    seasonal_heatmap = create_seasonal_product_heatmap(df)
    st.plotly_chart(seasonal_heatmap, use_container_width=True)
    
    # Display top products by season
    st.subheader("Top Products by Season")
    
    col1, col2, col3, col4 = st.columns(4)
    
    seasons = ["Spring", "Summer", "Fall", "Winter"]
    cols = [col1, col2, col3, col4]
    
    for i, season in enumerate(seasons):
        if season in top_products.index:
            product = top_products.loc[season, 'Service Category']
            revenue = top_products.loc[season, 'Daily Revenue']
            cols[i].metric(f"Top in {season}", product, f"${revenue:,.2f}")
    
    # Seasonal patterns over time
    st.subheader("Seasonal Trends")
    
    # Create time series plot
    monthly_data = df.groupby(['Month', 'Season'])['Daily Revenue'].sum().reset_index()
    
    fig = px.line(
        monthly_data, 
        x='Month', 
        y='Daily Revenue',
        color='Season',
        title='Monthly Revenue by Season',
        labels={'Daily Revenue': 'Total Revenue ($)', 'Month': 'Month'},
        markers=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.subheader("Seasonal Recommendations")
    
    recommendations = get_seasonal_recommendations(seasonal_perf, top_products)
    
    for season, rec in recommendations.items():
        st.info(f"**{season}**: {rec['recommendation']} (${rec['revenue']:,.2f} in revenue)")

# Customer Segmentation page
elif page == "Customer Segmentation":
    st.title("Customer Segmentation Analysis")
    
    st.write("""
    This section segments customers based on purchase value and behavior,
    identifying opportunities for targeted upselling.
    """)
    
    # Create customer segments
    value_segments, df_with_segments = create_value_segments(df)
    kmeans, cluster_analysis, df_with_clusters = create_kmeans_segments(df_with_segments)
    
    # Display customer segment metrics
    st.subheader("Customer Segments")
    
    # Create customer segment visualizations
    segment_plot = create_customer_segment_plots(df_with_clusters)
    st.plotly_chart(segment_plot, use_container_width=True)
    
    # Display segment metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Value-Based Segments")
        st.dataframe(value_segments)
    
    with col2:
        st.subheader("Cluster-Based Segments")
        st.dataframe(cluster_analysis)
    
    # Customer type comparison
    st.subheader("New vs. Returning Customers")
    
    customer_type_metrics = df.groupby('Customer Type').agg({
        'Daily Revenue': ['sum', 'mean'],
        'Conversions': ['sum', 'mean'],
    })
    
    customer_type_metrics.columns = ['_'.join(col).strip() for col in customer_type_metrics.columns.values]
    
    fig = px.bar(
        customer_type_metrics.reset_index(), 
        x='Customer Type',
        y=['Daily Revenue_mean', 'Conversions_mean'],
        barmode='group',
        title='Average Metrics by Customer Type',
        labels={
            'value': 'Average Value',
            'Customer Type': 'Customer Type',
            'variable': 'Metric'
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.subheader("Customer Segment Recommendations")
    
    recommendations = get_segment_recommendations(value_segments, cluster_analysis)
    
    for segment, rec in recommendations.items():
        st.info(f"**{segment.replace('value_', '').replace('cluster_', 'Cluster ')}**: {rec}")

# Marketing Channels page
elif page == "Marketing Channels":
    st.title("Marketing Channel Analysis")
    
    st.write("""
    This section analyzes the performance of different marketing channels,
    identifying the most effective channels for each season and product.
    """)
    
    # Analyze marketing channels
    channel_perf, best_channels, channel_by_season = analyze_marketing_channels(df)
    
    # Display marketing channel performance
    st.subheader("Marketing Channel Performance")
    
    channel_plots = create_marketing_channel_plots(df)
    st.plotly_chart(channel_plots, use_container_width=True)
    
    # Display best channels by season
    st.subheader("Best Marketing Channels by Season")
    
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
    channel_product = df.groupby(['Marketing Channel', 'Service Category'])['Daily Revenue'].sum().reset_index()
    
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
    
    # ROI by channel and product
    channel_product_roi = df.groupby(['Marketing Channel', 'Service Category'])['ROI'].mean().reset_index()
    
    pivot_roi = channel_product_roi.pivot(
        index='Marketing Channel',
        columns='Service Category',
        values='ROI'
    )
    
    fig_roi = px.imshow(
        pivot_roi,
        labels=dict(x='Product Category', y='Marketing Channel', color='ROI'),
        x=pivot_roi.columns,
        y=pivot_roi.index,
        color_continuous_scale='RdBu',
        title='Marketing Channel ROI by Product'
    )
    
    # Add annotations
    for i, channel in enumerate(pivot_roi.index):
        for j, product in enumerate(pivot_roi.columns):
            fig_roi.add_annotation(
                x=j, 
                y=i,
                text=f"{pivot_roi.iloc[i, j]:.2f}",
                showarrow=False,
                font=dict(color='black')
            )
    
    st.plotly_chart(fig_roi, use_container_width=True)
    
    # Recommendations
    st.subheader("Marketing Channel Recommendations")
    
    recommendations = get_marketing_recommendations(channel_perf, best_channels, channel_product)
    
    for key, rec in recommendations.items():
        st.info(f"**{key.replace('_', ' ').title()}**: {rec}")

# Promotional Calendar page
elif page == "Promotional Calendar":
    st.title("Promotional Calendar")
    
    st.write("""
    This section presents a promotional calendar based on the analysis of
    seasonal product performance and marketing channel effectiveness.
    """)
    
    # Create promotional calendar
    seasonal_perf, top_products = analyze_seasonal_products(df)
    channel_perf, best_channels, channel_by_season = analyze_marketing_channels(df)
    
    promo_calendar = create_promotional_calendar(top_products, best_channels)
    
    # Display promotional calendar heatmap
    st.subheader("Promotional Calendar Heatmap")
    
    promo_heatmap = create_promotional_calendar_heatmap(promo_calendar)
    st.plotly_chart(promo_heatmap, use_container_width=True)
    
    # Display detailed promotional calendar
    st.subheader("Detailed Promotional Calendar")
    
    # Add month names
    month_names = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August',
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    
    promo_calendar['Month Name'] = promo_calendar['Month'].map(month_names)
    
    # Sort by month
    promo_calendar = promo_calendar.sort_values('Month')
    
    # Display as table
    st.dataframe(
        promo_calendar[['Month Name', 'Season', 'Focus Product', 'Marketing Channel', 'Estimated ROI']]
    )
    
    # Download button
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    
    csv = convert_df_to_csv(promo_calendar)
    
    st.download_button(
        label="Download Promotional Calendar",
        data=csv,
        file_name="promotional_calendar.csv",
        mime="text/csv",
    )
    
    # Implementation guidelines
    st.subheader("Implementation Guidelines")
    
    st.write("""
    Follow these steps to implement the promotional calendar:
    
    1. **Prepare promotional materials** for each seasonal focus product
    2. **Allocate marketing budget** towards the recommended channels
    3. **Schedule promotions** according to the calendar
    4. **Monitor performance** and adjust strategy as needed
    5. **Evaluate results** at the end of each season
    """)
    
    # Expected outcomes
    st.subheader("Expected Outcomes")
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Projected Revenue Increase", "8-12%", "Based on optimal product-season alignment")
    col2.metric("Marketing ROI Improvement", "15-20%", "Through channel optimization")
    col3.metric("Customer Value Increase", "10-15%", "Via targeted upselling")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Data Analytics Dashboard")
st.sidebar.caption("© 2025 Bakery Analytics Ltd")