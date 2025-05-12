import unittest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.marketing_analysis import analyze_marketing_channels, calculate_channel_efficiency, analyze_channel_product_performance

class TestMarketingAnalysis(unittest.TestCase):
    """Tests for marketing analysis functions"""
    
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
    
    def test_analyze_marketing_channels(self):
        """Test analyzing marketing channels"""
        # Analyze marketing channels
        channel_perf, best_channels, channel_by_season = analyze_marketing_channels(self.sample_data)
        
        # Check results
        self.assertIsInstance(channel_perf, pd.DataFrame)
        self.assertIsInstance(best_channels, pd.DataFrame)
        self.assertIsInstance(channel_by_season, pd.DataFrame)
        
        # Check if all channels are included
        self.assertEqual(len(channel_perf), len(self.sample_data['Marketing Channel'].unique()))
        
        # Check if all seasons have a best channel
        self.assertEqual(len(best_channels), len(self.sample_data['Season'].unique()))
        
        # Check if channel_by_season has correct structure
        self.assertIn('Season', channel_by_season.columns)
        self.assertIn('Marketing Channel', channel_by_season.columns)
        self.assertIn('Daily Revenue', channel_by_season.columns)
    
    def test_calculate_channel_efficiency(self):
        """Test calculating channel efficiency"""
        # Calculate channel efficiency
        efficiency = calculate_channel_efficiency(self.sample_data)
        
        # Check results
        self.assertIsInstance(efficiency, pd.DataFrame)
        
        # Check if all channels are included
        self.assertEqual(len(efficiency), len(self.sample_data['Marketing Channel'].unique()))
        
        # Check if efficiency metrics are calculated
        self.assertIn('Revenue per Conversion', efficiency.columns)
        self.assertIn('Cost per Conversion', efficiency.columns)
        self.assertIn('ROI', efficiency.columns)
    
    def test_analyze_channel_product_performance(self):
        """Test analyzing channel-product performance"""
        # Analyze channel-product performance
        channel_product, best_products = analyze_channel_product_performance(self.sample_data)
        
        # Check results
        self.assertIsInstance(channel_product, pd.DataFrame)
        self.assertIsInstance(best_products, pd.DataFrame)
        
        # Check if channel_product has correct structure
        self.assertIn('Marketing Channel', channel_product.columns)
        self.assertIn('Service Category', channel_product.columns)
        self.assertIn('Daily Revenue', channel_product.columns)
        
        # Check if best_products has one entry per channel
        self.assertEqual(len(best_products), len(self.sample_data['Marketing Channel'].unique()))

if __name__ == '__main__':
    unittest.main()