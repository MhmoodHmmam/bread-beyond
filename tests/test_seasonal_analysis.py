import unittest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing import load_data
from src.seasonal_analysis import analyze_seasonal_products, analyze_seasonality_over_time, get_seasonal_recommendations

class TestSeasonalAnalysis(unittest.TestCase):
    """Tests for seasonal analysis functions"""
    
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
        self.sample_data['Year'] = self.sample_data['Date'].dt.year
        self.sample_data['Month'] = self.sample_data['Date'].dt.month
    
    def test_analyze_seasonal_products(self):
        """Test analyzing seasonal products"""
        # Analyze seasonal products
        seasonal_perf, top_products = analyze_seasonal_products(self.sample_data)
        
        # Check results
        self.assertIsInstance(seasonal_perf, pd.DataFrame)
        self.assertIsInstance(top_products, pd.DataFrame)
        
        # Check if all seasons are included
        self.assertEqual(len(seasonal_perf.index.unique(level='Season')), 4)
        
        # Check if top_products has one entry per season
        self.assertEqual(len(top_products), 4)
        
        # Check if the columns are correct
        self.assertIn('Daily Revenue', top_products.columns)
        self.assertIn('Service Category', top_products.columns)
    
    def test_analyze_seasonality_over_time(self):
        """Test analyzing seasonality over time"""
        # Analyze seasonality over time
        monthly_seasonal, pivot_df = analyze_seasonality_over_time(self.sample_data)
        
        # Check results
        self.assertIsInstance(monthly_seasonal, pd.DataFrame)
        self.assertIsInstance(pivot_df, pd.DataFrame)
        
        # Check if all months are included
        self.assertGreaterEqual(len(monthly_seasonal['Month'].unique()), 1)
        
        # Check if pivot table has correct shape
        self.assertEqual(pivot_df.shape[1], len(self.sample_data['Service Category'].unique()) + 2)  # +2 for Month and Season
    
    def test_get_seasonal_recommendations(self):
        """Test getting seasonal recommendations"""
        # Analyze seasonal products
        seasonal_perf, top_products = analyze_seasonal_products(self.sample_data)
        
        # Get recommendations
        recommendations = get_seasonal_recommendations(seasonal_perf, top_products)
        
        # Check results
        self.assertIsInstance(recommendations, dict)
        
        # Check if all seasons have recommendations
        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            self.assertIn(season, recommendations)
            
            # Check recommendation structure
            self.assertIn('top_product', recommendations[season])
            self.assertIn('revenue', recommendations[season])
            self.assertIn('recommendation', recommendations[season])

if __name__ == '__main__':
    unittest.main()