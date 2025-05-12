# File: tests/test_data_processing.py

import unittest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing import load_data, save_processed_data

class TestDataProcessing(unittest.TestCase):
    """Tests for data processing functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a sample Excel file
        self.sample_data = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=10),
            'Service Category': ['Cupcake', 'Cake', 'Cookie', 'Bread', 'Cupcake', 
                                'Cake', 'Cookie', 'Bread', 'Cupcake', 'Cake'],
            'Marketing Channel': ['In-store Promo', 'Local Magazine', 'Instagram', 'Google Maps',
                                 'In-store Promo', 'Local Magazine', 'Instagram', 'Google Maps',
                                 'In-store Promo', 'Local Magazine'],
            'Season': ['Winter', 'Spring', 'Summer', 'Fall', 'Winter', 
                      'Spring', 'Summer', 'Fall', 'Winter', 'Spring'],
            'Customer Type': ['Returning', 'New', 'Returning', 'New', 'Returning',
                            'New', 'Returning', 'New', 'Returning', 'New'],
            'Ad Spend': [50, 75, 100, 150, 50, 75, 100, 150, 50, 75],
            'Conversions': [5, 7, 10, 15, 5, 7, 10, 15, 5, 7],
            'Daily Revenue': [100, 150, 200, 250, 100, 150, 200, 250, 100, 150]
        })
        
        # Save sample data
        self.test_file = 'tests/test_data.xlsx'
        self.test_output = 'tests/output/test_processed.csv'
        self.sample_data.to_excel(self.test_file, index=False)
    
    def tearDown(self):
        """Tear down test fixtures"""
        # Remove test files
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        
        if os.path.exists(self.test_output):
            os.remove(self.test_output)
    
    def test_load_data(self):
        """Test loading data from Excel file"""
        # Load the test data
        df = load_data(self.test_file)
        
        # Check basic properties
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 10)
        
        # Check if additional columns were created
        self.assertIn('Year', df.columns)
        self.assertIn('Month', df.columns)
        self.assertIn('ROI', df.columns)
        self.assertIn('Profit', df.columns)
        
        # Check if calculations are correct
        self.assertTrue((df['ROI'] == (df['Daily Revenue'] - df['Ad Spend']) / df['Ad Spend']).all())
        self.assertTrue((df['Profit'] == df['Daily Revenue'] - df['Ad Spend']).all())
    
    def test_save_processed_data(self):
        """Test saving processed data to CSV"""
        # Load data
        df = load_data(self.test_file)
        
        # Save processed data
        result = save_processed_data(df, self.test_output)
        
        # Check if file was created
        self.assertTrue(os.path.exists(self.test_output))
        self.assertTrue(result)
        
        # Load saved data and check if it matches
        saved_df = pd.read_csv(self.test_output)
        self.assertEqual(len(saved_df), len(df))
        self.assertEqual(saved_df.shape[1], df.shape[1])

if __name__ == '__main__':
    unittest.main()