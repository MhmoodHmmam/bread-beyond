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
from src.customer_segmentation import create_value_segments, create_kmeans_segments, get_segment_recommendations
from src.visualization import create_customer_segment_plots

# Set page config
st.set_page_config(
    page_title="Customer Segments - Data Analytics",
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
st.title("Customer Segmentation Analysis")

st.write("""
This page provides detailed customer segmentation analysis to identify high-value customer groups
and develop targeted upselling strategies.
""")

# Create customer segments
value_segments, df_with_segments = create_value_segments(df)

# K-means clustering parameters
st.subheader("Customer Clustering")

n_clusters = st.slider("Number of Clusters", min_value=2, max_value=5, value=3)
kmeans, cluster_analysis, df_with_clusters = create_kmeans_segments(df_with_segments, n_clusters)

# Display customer segment visualizations
segment_plot = create_customer_segment_plots(df_with_clusters)
st.plotly_chart(segment_plot, use_container_width=True)

# Display segment metrics
col1, col2 = st.columns(2)

with col1:
    st.subheader("Value-Based Segments")
    st.dataframe(value_segments)
    
    # Value segment distribution
    value_counts = df_with_clusters['Value Segment'].value_counts().reset_index()
    value_counts.columns = ['Segment', 'Count']
    
    fig = px.pie(
        value_counts, 
        values='Count', 
        names='Segment',
        title='Value Segment Distribution',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Cluster-Based Segments")
    st.dataframe(cluster_analysis)
    
    # Cluster distribution
    cluster_counts = df_with_clusters['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    
    fig = px.pie(
        cluster_counts, 
        values='Count', 
        names='Cluster',
        title='Cluster Distribution',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Cluster visualization
st.subheader("Customer Cluster Visualization")

# Select features for visualization
x_feature = st.selectbox(
    "Select X-axis Feature:",
    ['Purchase Value', 'Conversions', 'Daily Revenue', 'ROI'],
    index=0
)

y_feature = st.selectbox(
    "Select Y-axis Feature:",
    ['Purchase Value', 'Conversions', 'Daily Revenue', 'ROI'],
    index=1
)

# Create scatter plot
fig = px.scatter(
    df_with_clusters,
    x=x_feature,
    y=y_feature,
    color='Cluster',
    title=f'Customer Clusters: {x_feature} vs {y_feature}',
    labels={x_feature: x_feature, y_feature: y_feature},
    hover_data=['Customer Type', 'Service Category']
)

st.plotly_chart(fig, use_container_width=True)

# Product preferences by segment
st.subheader("Product Preferences by Segment")

segment_type = st.radio("Segment Type:", ["Value Segment", "Cluster"], horizontal=True)

if segment_type == "Value Segment":
    # Product preferences by value segment
    product_by_segment = df_with_clusters.groupby(['Value Segment', 'Service Category'])['Daily Revenue'].sum().reset_index()
    
    fig = px.bar(
        product_by_segment,
        x='Service Category',
        y='Daily Revenue',
        color='Value Segment',
        barmode='group',
        title='Product Preferences by Value Segment',
        labels={'Daily Revenue': 'Total Revenue ($)', 'Service Category': 'Product Category'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top product by segment
    top_product_by_segment = product_by_segment.sort_values('Daily Revenue', ascending=False).groupby('Value Segment').first()
    
    st.subheader("Top Product by Value Segment")
    st.dataframe(top_product_by_segment[['Service Category', 'Daily Revenue']])
else:
    # Product preferences by cluster
    product_by_cluster = df_with_clusters.groupby(['Cluster', 'Service Category'])['Daily Revenue'].sum().reset_index()
    
    fig = px.bar(
        product_by_cluster,
        x='Service Category',
        y='Daily Revenue',
        color='Cluster',
        barmode='group',
        title='Product Preferences by Cluster',
        labels={'Daily Revenue': 'Total Revenue ($)', 'Service Category': 'Product Category'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top product by cluster
    top_product_by_cluster = product_by_cluster.sort_values('Daily Revenue', ascending=False).groupby('Cluster').first()
    
    st.subheader("Top Product by Cluster")
    st.dataframe(top_product_by_cluster[['Service Category', 'Daily Revenue']])

# Customer type analysis
st.subheader("New vs. Returning Customers")

customer_type_metrics = df.groupby('Customer Type').agg({
    'Daily Revenue': ['sum', 'mean'],
    'Conversions': ['sum', 'mean'],
    'Purchase Value': 'mean'
})

customer_type_metrics.columns = ['_'.join(col).strip() for col in customer_type_metrics.columns.values]
customer_type_metrics = customer_type_metrics.reset_index()

# Create comparison chart
fig = px.bar(
    customer_type_metrics,
    x='Customer Type',
    y=['Daily Revenue_mean', 'Purchase Value_mean'],
    barmode='group',
    title='Average Metrics by Customer Type',
    labels={
        'value': 'Average Value',
        'Customer Type': 'Customer Type',
        'variable': 'Metric'
    }
)

st.plotly_chart(fig, use_container_width=True)

# Customer segment recommendations
st.subheader("Customer Segment Recommendations")

recommendations = get_segment_recommendations(value_segments, cluster_analysis)

tab1, tab2 = st.tabs(["Value Segment Recommendations", "Cluster Recommendations"])

with tab1:
    for key, rec in recommendations.items():
        if key.startswith('value_'):
            segment = key.replace('value_', '')
            st.info(f"**{segment} Segment**: {rec}")
            
            # Get segment data
            segment_data = df_with_clusters[df_with_clusters['Value Segment'] == segment]
            
            # Product preferences for this segment
            product_prefs = segment_data.groupby('Service Category')['Daily Revenue'].sum().sort_values(ascending=False)
            
            # Display as bar chart
            fig = px.bar(
                product_prefs,
                title=f'Product Preferences for {segment} Segment',
                labels={'value': 'Total Revenue ($)', 'index': 'Product Category'}
            )
            
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    for key, rec in recommendations.items():
        if key.startswith('cluster_'):
            cluster = key.replace('cluster_', '')
            st.info(f"**Cluster {cluster}**: {rec}")
            
            # Get cluster data
            cluster_data = df_with_clusters[df_with_clusters['Cluster'] == int(cluster)]
            
            # Product preferences for this cluster
            product_prefs = cluster_data.groupby('Service Category')['Daily Revenue'].sum().sort_values(ascending=False)
            
            # Display as bar chart
            fig = px.bar(
                product_prefs,
                title=f'Product Preferences for Cluster {cluster}',
                labels={'value': 'Total Revenue ($)', 'index': 'Product Category'}
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Upsell strategy
st.subheader("Upsell Strategy Implementation")

st.write("""
Follow these steps to implement the upsell strategy:

1. **Identify customer segments** in your customer database
2. **Match product recommendations** to each segment
3. **Develop targeted messaging** for each segment
4. **Implement promotional campaigns** based on segment preferences
5. **Monitor and optimize** based on response rates
""")

# Expected outcomes
col1, col2, col3 = st.columns(3)

col1.metric("Avg. Order Value Increase", "15-20%", "Through targeted upselling")
col2.metric("Customer Retention", "+10%", "Via personalized recommendations")
col3.metric("Revenue Growth", "12-18%", "From high-value segments")

# Generate downloadable segment analysis report
st.subheader("Download Segment Analysis Report")

@st.cache_data
def generate_segment_report():
    # Combine segment analysis data
    report_data = {
        'value_segments': value_segments.reset_index(),
        'cluster_analysis': cluster_analysis.reset_index(),
        'top_product_by_segment': product_by_segment.sort_values('Daily Revenue', ascending=False).groupby('Value Segment').first(),
        'customer_type_metrics': customer_type_metrics
    }
    
    return report_data

report_data = generate_segment_report()

with pd.ExcelWriter('customer_segment_analysis.xlsx') as writer:
    for key, df in report_data.items():
        df.to_excel(writer, sheet_name=key, index=False)

with open('customer_segment_analysis.xlsx', 'rb') as f:
    st.download_button(
        label="Download Segment Analysis Report",
        data=f,
        file_name="customer_segment_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )