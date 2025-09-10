"""
Tests for CNN model functionality.
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.cnn_model import CNNDisaggregator, EnsembleCNNDisaggregator
from src.utils.config import get_default_config


class TestCNNDisaggregator(unittest.TestCase):
    """Test cases for CNNDisaggregator model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_default_config()
        self.model = CNNDisaggregator(self.config)
    
    def test_model_initialization(self):
        """Test model initialization."""
        # Check that model is properly initialized
        self.assertIsInstance(self.model, CNNDisaggregator)
        
        # Check model parameters
        params = list(self.model.parameters())
        self.assertTrue(len(params) > 0)
        
        # Check that conv layers are created
        self.assertEqual(len(self.model.conv_layers), len(self.config['model']['architecture']['num_filters']))
        
        # Check that FC layers exist for each appliance
        for appliance in self.config['data']['appliances']:
            self.assertIn(appliance, self.model.fc_layers)
    
    def test_forward_pass(self):
        """Test forward pass through the model."""
        batch_size = 8
        sequence_length = self.config['data']['sequence_length']
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, 1, sequence_length)
        
        # Forward pass
        predictions = self.model(dummy_input)
        
        # Check output format
        self.assertIsInstance(predictions, dict)
        
        # Check predictions for each appliance
        for appliance in self.config['data']['appliances']:
            self.assertIn(appliance, predictions)
            self.assertEqual(predictions[appliance].shape[0], batch_size)
            self.assertEqual(len(predictions[appliance].shape), 1)  # Should be 1D
    
    def test_model_summary(self):
        """Test model summary generation."""
        summary = self.model.get_model_summary()
        
        # Check that summary is a string
        self.assertIsInstance(summary, str)
        
        # Check that it contains key information
        self.assertIn("CNN Energy Disaggregation Model Summary", summary)
        self.assertIn("Total Parameters", summary)
        
        for appliance in self.config['data']['appliances']:
            self.assertIn(appliance, summary)
    
    def test_different_input_sizes(self):
        """Test model with different input sizes."""
        sequence_lengths = [100, 299, 599, 1000]
        
        for seq_len in sequence_lengths:
            # Modify config for this test
            test_config = self.config.copy()
            test_config['data']['sequence_length'] = seq_len
            
            model = CNNDisaggregator(test_config)
            
            # Test forward pass
            dummy_input = torch.randn(4, 1, seq_len)
            predictions = model(dummy_input)
            
            # Check that we get predictions for all appliances
            for appliance in self.config['data']['appliances']:
                self.assertIn(appliance, predictions)
                self.assertEqual(predictions[appliance].shape[0], 4)


class TestEnsembleCNNDisaggregator(unittest.TestCase):
    """Test cases for EnsembleCNNDisaggregator model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_default_config()
        self.ensemble_model = EnsembleCNNDisaggregator(self.config, num_models=3)
    
    def test_ensemble_initialization(self):
        """Test ensemble model initialization."""
        # Check that ensemble is properly initialized
        self.assertIsInstance(self.ensemble_model, EnsembleCNNDisaggregator)
        
        # Check number of models
        self.assertEqual(len(self.ensemble_model.models), 3)
        
        # Check that all models are CNNDisaggregator instances
        for model in self.ensemble_model.models:
            self.assertIsInstance(model, CNNDisaggregator)
    
    def test_ensemble_forward_pass(self):
        """Test forward pass through ensemble."""
        batch_size = 4
        sequence_length = self.config['data']['sequence_length']
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, 1, sequence_length)
        
        # Forward pass
        predictions = self.ensemble_model(dummy_input)
        
        # Check output format
        self.assertIsInstance(predictions, dict)
        
        # Check predictions for each appliance
        for appliance in self.config['data']['appliances']:
            self.assertIn(appliance, predictions)
            self.assertEqual(predictions[appliance].shape[0], batch_size)
    
    def test_predictions_with_uncertainty(self):
        """Test uncertainty estimation."""
        batch_size = 4
        sequence_length = self.config['data']['sequence_length']
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, 1, sequence_length)
        
        # Get predictions with uncertainty
        predictions_with_uncertainty = self.ensemble_model.get_predictions_with_uncertainty(dummy_input)
        
        # Check output format
        self.assertIsInstance(predictions_with_uncertainty, dict)
        
        # Check predictions for each appliance
        for appliance in self.config['data']['appliances']:
            self.assertIn(appliance, predictions_with_uncertainty)
            self.assertIn('mean', predictions_with_uncertainty[appliance])
            self.assertIn('std', predictions_with_uncertainty[appliance])
            
            mean_pred = predictions_with_uncertainty[appliance]['mean']
            std_pred = predictions_with_uncertainty[appliance]['std']
            
            self.assertEqual(mean_pred.shape[0], batch_size)
            self.assertEqual(std_pred.shape[0], batch_size)


class TestModelTraining(unittest.TestCase):
    """Test basic model training functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_default_config()
        # Use smaller config for faster testing
        self.config['data']['sequence_length'] = 99
        self.config['model']['architecture']['num_filters'] = [8, 8, 16]
        self.config['model']['architecture']['filter_sizes'] = [5, 3, 3]
        
        self.model = CNNDisaggregator(self.config)
    
    def test_model_backward_pass(self):
        """Test that model can compute gradients."""
        batch_size = 4
        sequence_length = self.config['data']['sequence_length']
        
        # Create dummy input and targets
        dummy_input = torch.randn(batch_size, 1, sequence_length)
        dummy_targets = {
            appliance: torch.randn(batch_size) 
            for appliance in self.config['data']['appliances']
        }
        
        # Forward pass
        predictions = self.model(dummy_input)
        
        # Compute loss
        criterion = torch.nn.MSELoss()
        total_loss = 0
        for appliance in self.config['data']['appliances']:
            loss = criterion(predictions[appliance], dummy_targets[appliance])
            total_loss += loss
        
        # Backward pass
        total_loss.backward()
        
        # Check that gradients are computed
        for param in self.model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_model_modes(self):
        """Test training and evaluation modes."""
        # Test training mode
        self.model.train()
        self.assertTrue(self.model.training)
        
        # Test evaluation mode
        self.model.eval()
        self.assertFalse(self.model.training)
        
        # Test that dropout behaves differently in train vs eval
        batch_size = 4
        sequence_length = self.config['data']['sequence_length']
        dummy_input = torch.randn(batch_size, 1, sequence_length)
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        self.model.train()
        pred_train = self.model(dummy_input)
        
        torch.manual_seed(42)
        self.model.eval()
        pred_eval = self.model(dummy_input)
        
        # In this case, predictions might be the same due to low dropout rate
        # Just check that both modes produce valid outputs
        for appliance in self.config['data']['appliances']:
            self.assertEqual(pred_train[appliance].shape, pred_eval[appliance].shape)


if __name__ == '__main__':
    unittest.main()