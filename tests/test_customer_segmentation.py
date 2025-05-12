import unittest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.customer_segmentation import create_value_segments, create_kmeans_segments, get_segment_recommendations

class TestCustomerSegmentation(unittest.TestCase):
    """Tests for customer segmentation functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a sample DataFrame
        self.sample_data = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=20),
            'Service Category': ['Cupcake', 'Cake', 'Cookie', 'Bread', 'Cupcake'] * 4,
            'Marketing Channel': ['In-store Promo', 'Local Magazine', 'Instagram', 'Google Maps'] * 5,
            'Season': ['Winter', 'Spring', 'Summer', 'Fall'] * 5,
            'Customer Type': ['Returning', 'New'] * 10,
            'Ad Spend': [50, 75, 100, 150] * 5,
            'Conversions': [5, 7, 10, 15] * 5,
            'Daily Revenue': [100, 150, 200, 250] * 5
        })
        
        # Add calculated columns
        self.sample_data['ROI'] = (self.sample_data['Daily Revenue'] - self.sample_data['Ad Spend']) / self.sample_data['Ad Spend']
        self.sample_data['Profit'] = self.sample_data['Daily Revenue'] - self.sample_data['Ad Spend']
        self.sample_data['Purchase Value'] = self.sample_data['Daily Revenue'] / self.sample_data['Conversions']
    
    def test_create_value_segments(self):
        """Test creating value segments"""
        # Create value segments
        value_segments, df_with_segments = create_value_segments(self.sample_data)
        
        # Check results
        self.assertIsInstance(value_segments, pd.DataFrame)
        self.assertIsInstance(df_with_segments, pd.DataFrame)
        
        # Check if value segments were created
        self.assertIn('Value Segment', df_with_segments.columns)
        
        # Check if all segments are present
        segments = df_with_segments['Value Segment'].unique()
        self.assertGreaterEqual(len(segments), 1)
        
        # Check if value_segments has one entry per segment
        self.assertEqual(len(value_segments), len(segments))
    
    def test_create_kmeans_segments(self):
        """Test creating K-means segments"""
        # Add value segments first
        _, df_with_segments = create_value_segments(self.sample_data)
        
        # Create K-means segments
        kmeans, cluster_analysis, df_with_clusters = create_kmeans_segments(df_with_segments, n_clusters=2)
        
        # Check results
        self.assertIsNotNone(kmeans)
        self.assertIsInstance(cluster_analysis, pd.DataFrame)
        self.assertIsInstance(df_with_clusters, pd.DataFrame)
        
        # Check if clusters were created
        self.assertIn('Cluster', df_with_clusters.columns)
        
        # Check if all clusters are present
        clusters = df_with_clusters['Cluster'].unique()
        self.assertEqual(len(clusters), 2)  # We specified 2 clusters
        
        # Check if cluster_analysis has one entry per cluster
        self.assertEqual(len(cluster_analysis), 2)
    
    def test_get_segment_recommendations(self):
        """Test getting segment recommendations"""
        # Create value segments
        value_segments, _ = create_value_segments(self.sample_data)
        
        # Create mock cluster analysis
        cluster_analysis = pd.DataFrame({
            'Purchase Value': [10, 30],
            'Conversions': [3, 8],
            'Daily Revenue': [100, 300]
        }, index=[0, 1])
        
        # Get recommendations
        recommendations = get_segment_recommendations(value_segments, cluster_analysis)
        
        # Check results
        self.assertIsInstance(recommendations, dict)
        
        # Check if there are recommendations for each value segment
        for segment in value_segments.index:
            self.assertIn(f'value_{segment}', recommendations)
        
        # Check if there are recommendations for each cluster
        for cluster in cluster_analysis.index:
            self.assertIn(f'cluster_{cluster}', recommendations)

if __name__ == '__main__':
    unittest.main()