"""
Tests for data preprocessing and dataset functionality.
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessing import UKDALEPreprocessor
from src.data.dataset import EnergyDisaggregationDataset, DataLoaderFactory
from src.utils.config import get_default_config


class TestUKDALEPreprocessor(unittest.TestCase):
    """Test cases for UKDALEPreprocessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_default_config()
        self.preprocessor = UKDALEPreprocessor(self.config)
    
    def test_create_synthetic_data(self):
        """Test synthetic data generation."""
        data = self.preprocessor._create_synthetic_data()
        
        # Check that all required keys are present
        expected_keys = ['aggregate', 'hvac', 'kitchen', 'electronics']
        for key in expected_keys:
            self.assertIn(key, data)
            self.assertIn('timestamp', data[key].columns)
            self.assertIn('power', data[key].columns)
        
        # Check data length consistency
        lengths = [len(data[key]) for key in expected_keys]
        self.assertTrue(all(length == lengths[0] for length in lengths))
        
        # Check that aggregate is approximately sum of appliances
        aggregate_power = data['aggregate']['power'].values
        sum_appliances = (data['hvac']['power'].values + 
                         data['kitchen']['power'].values + 
                         data['electronics']['power'].values)
        
        # Allow for some difference due to noise
        diff = np.abs(aggregate_power - sum_appliances)
        self.assertTrue(np.mean(diff) < 50)  # Average difference less than 50W
    
    def test_clean_data(self):
        """Test data cleaning functionality."""
        # Create test data with negative values and outliers
        raw_data = self.preprocessor._create_synthetic_data()
        
        # Introduce negative values and outliers
        raw_data['hvac']['power'].iloc[0] = -100
        raw_data['kitchen']['power'].iloc[1] = 10000  # Outlier
        
        cleaned_data = self.preprocessor.clean_data(raw_data)
        
        # Check that negative values are removed
        for key in cleaned_data:
            self.assertTrue(np.all(cleaned_data[key]['power'] >= 0))
    
    def test_create_sequences(self):
        """Test sequence creation for CNN input."""
        data = self.preprocessor._create_synthetic_data()
        normalized_data = self.preprocessor.normalize_data(data)
        
        X, y = self.preprocessor.create_sequences(normalized_data)
        
        # Check shapes
        expected_samples = len(normalized_data['aggregate']) - self.config['data']['sequence_length'] + 1
        self.assertEqual(X.shape[0], expected_samples)
        self.assertEqual(X.shape[1], self.config['data']['sequence_length'])
        
        # Check target shapes
        for appliance in self.config['data']['appliances']:
            self.assertEqual(len(y[appliance]), expected_samples)
    
    def test_prepare_data_pipeline(self):
        """Test complete data preparation pipeline."""
        X, y = self.preprocessor.prepare_data("dummy_path")
        
        # Check output shapes
        self.assertEqual(len(X.shape), 2)
        self.assertEqual(X.shape[1], self.config['data']['sequence_length'])
        
        # Check that we have targets for all appliances
        for appliance in self.config['data']['appliances']:
            self.assertIn(appliance, y)
            self.assertEqual(len(y[appliance]), X.shape[0])


class TestEnergyDataset(unittest.TestCase):
    """Test cases for EnergyDisaggregationDataset."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_default_config()
        self.preprocessor = UKDALEPreprocessor(self.config)
        self.X, self.y = self.preprocessor.prepare_data("dummy_path")
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        dataset = EnergyDisaggregationDataset(
            self.X, self.y, self.config['data']['appliances']
        )
        
        self.assertEqual(len(dataset), len(self.X))
        
        # Test getting an item
        input_seq, targets = dataset[0]
        
        # Check input shape (should have channel dimension)
        self.assertEqual(len(input_seq.shape), 2)
        self.assertEqual(input_seq.shape[0], 1)  # Channel dimension
        self.assertEqual(input_seq.shape[1], self.config['data']['sequence_length'])
        
        # Check targets
        self.assertEqual(len(targets), len(self.config['data']['appliances']))
        for appliance in self.config['data']['appliances']:
            self.assertIn(appliance, targets)
    
    def test_data_loader_factory(self):
        """Test data loader creation."""
        train_loader, val_loader, test_loader = DataLoaderFactory.create_data_loaders(
            self.X, self.y, self.config['data']['appliances'],
            batch_size=32, validation_split=0.2, test_split=0.1
        )
        
        # Check that loaders are created
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)
        
        # Test getting a batch
        input_batch, target_batch = next(iter(train_loader))
        
        # Check batch shapes
        self.assertEqual(len(input_batch.shape), 3)  # (batch, channel, sequence)
        self.assertTrue(input_batch.shape[0] <= 32)  # Batch size
        self.assertEqual(input_batch.shape[1], 1)     # Channel
        self.assertEqual(input_batch.shape[2], self.config['data']['sequence_length'])
        
        # Check target batch
        for appliance in self.config['data']['appliances']:
            self.assertIn(appliance, target_batch)
            self.assertEqual(len(target_batch[appliance]), input_batch.shape[0])


if __name__ == '__main__':
    unittest.main()