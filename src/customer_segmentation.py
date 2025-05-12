import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

def preprocess_for_segmentation(df):
    """
    Preprocess data for customer segmentation
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Processed data
        
    Returns:
    --------
    pandas.DataFrame
        Data prepared for segmentation
    """
    # Calculate metrics for segmentation
    # For this example, we'll use customer types and purchase values
    
    # Create a DataFrame with purchase value
    df['Purchase Value'] = df['Daily Revenue'] / df['Conversions']
    df['Purchase Value'] = df['Purchase Value'].fillna(0)
    
    # Group by customer type
    customer_data = df.groupby('Customer Type').agg({
        'Daily Revenue': ['sum', 'mean'],
        'Purchase Value': ['mean', 'max', 'min'],
        'Conversions': ['sum', 'mean']
    })
    
    # Flatten multi-index columns
    customer_data.columns = ['_'.join(col).strip() for col in customer_data.columns.values]
    
    return customer_data

def create_value_segments(df):
    """Segment customers based on purchase value"""
    
    # First ensure Purchase Value column exists
    if 'Purchase Value' not in df.columns:
        df['Purchase Value'] = df['Daily Revenue'] / df['Conversions']
        df['Purchase Value'] = df['Purchase Value'].fillna(0)
    
    # Create segments based on purchase value
    bins = [0, 10, 20, 50, np.inf]
    labels = ['Low', 'Medium', 'High', 'Premium']
    df['Value Segment'] = pd.cut(df['Purchase Value'], bins=bins, labels=labels)
    
    # Analyze segments
    value_segments = df.groupby('Value Segment').agg({
        'Daily Revenue': 'sum',
        'Purchase Value': 'mean',
        'Conversions': 'sum'
    }).round(2)
    
    return value_segments, df

def create_kmeans_segments(df, n_clusters=3):
    """
    Create customer segments using K-means clustering
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Processed data
    n_clusters : int
        Number of clusters for K-means
        
    Returns:
    --------
    tuple
        K-means model and segmented data
    """
    # Select features for clustering
    features = ['Purchase Value', 'Conversions']
    X = df[features].values
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Analyze clusters
    cluster_analysis = df.groupby('Cluster').agg({
        'Purchase Value': 'mean',
        'Conversions': 'mean',
        'Daily Revenue': 'sum'
    }).round(2)
    
    # Save the model
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    with open("models/customer_segments.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    
    return kmeans, cluster_analysis, df

def plot_customer_segments(df):
    """
    Visualize customer segments
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Segmented data
        
    Returns:
    --------
    matplotlib.figure.Figure
        Plot figure
    """
    # Create visualization
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Value segments
    value_counts = df['Value Segment'].value_counts()
    value_counts.plot(kind='pie', ax=ax[0], autopct='%1.1f%%')
    ax[0].set_title('Customer Value Segments')
    ax[0].set_ylabel('')
    
    # Plot 2: Clusters
    if 'Cluster' in df.columns:
        scatter = ax[1].scatter(
            df['Purchase Value'], 
            df['Conversions'],
            c=df['Cluster'], 
            cmap='viridis',
            alpha=0.6
        )
        ax[1].set_title('Customer Clusters')
        ax[1].set_xlabel('Purchase Value')
        ax[1].set_ylabel('Conversions')
        ax[1].legend(*scatter.legend_elements(), title="Clusters")
    
    plt.tight_layout()
    
    return fig

def get_segment_recommendations(value_segments, cluster_analysis=None):
    """
    Generate recommendations for customer segments
    
    Parameters:
    -----------
    value_segments : pandas.DataFrame
        Value-based segments
    cluster_analysis : pandas.DataFrame, optional
        Cluster-based segments
        
    Returns:
    --------
    dict
        Recommendations for each segment
    """
    recommendations = {}
    
    # Value segment recommendations
    for segment in value_segments.index:
        if segment == 'Low':
            rec = "Offer entry-level products with discounts to encourage repeat purchases"
        elif segment == 'Medium':
            rec = "Focus on upselling to premium products with targeted promotions"
        elif segment == 'High':
            rec = "Provide special offers for high-margin items and loyalty rewards"
        else:  # Premium
            rec = "Develop exclusive offerings and personalized experiences"
            
        recommendations[f"value_{segment}"] = rec
    
    # Cluster recommendations
    if cluster_analysis is not None:
        for cluster in cluster_analysis.index:
            avg_value = cluster_analysis.loc[cluster, 'Purchase Value']
            avg_conv = cluster_analysis.loc[cluster, 'Conversions']
            
            if avg_value > 20 and avg_conv > 5:
                rec = "High-value loyal customers: Focus on retention and exclusive offerings"
            elif avg_value > 20:
                rec = "High-spenders with low frequency: Encourage more frequent purchases"
            elif avg_conv > 5:
                rec = "Frequent buyers with lower spend: Focus on upselling higher-value items"
            else:
                rec = "Low-value customers: Incentivize with promotions to increase frequency and spend"
                
            recommendations[f"cluster_{cluster}"] = rec
    
    return recommendations

def main(df):
    """
    Main function for customer segmentation
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Processed data
        
    Returns:
    --------
    dict
        Analysis results
    """
    # Preprocess data for segmentation
    customer_data = preprocess_for_segmentation(df)
    
    # Create value segments
    value_segments, df_with_segments = create_value_segments(df)
    
    # Create K-means segments
    kmeans, cluster_analysis, df_with_clusters = create_kmeans_segments(df_with_segments)
    
    # Get recommendations
    recommendations = get_segment_recommendations(value_segments, cluster_analysis)
    
    # Create plot
    fig = plot_customer_segments(df_with_clusters)
    
    # Return results
    results = {
        'customer_data': customer_data,
        'value_segments': value_segments,
        'cluster_analysis': cluster_analysis,
        'recommendations': recommendations,
        'figure': fig,
        'segmented_data': df_with_clusters
    }
    
    return results