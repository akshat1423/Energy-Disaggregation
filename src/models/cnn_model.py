"""
CNN model for energy disaggregation.
Based on neural network architectures for Non-Intrusive Load Monitoring (NILM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class CNNDisaggregator(nn.Module):
    """
    CNN model for energy disaggregation.
    
    This model takes aggregate power consumption as input and predicts
    individual appliance power consumption.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the CNN model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super(CNNDisaggregator, self).__init__()
        
        self.config = config
        model_config = config['model']['architecture']
        
        self.input_channels = model_config['input_channels']
        self.num_filters = model_config['num_filters']
        self.filter_sizes = model_config['filter_sizes']
        self.dropout_rate = model_config['dropout_rate']
        self.sequence_length = config['data']['sequence_length']
        self.appliances = config['data']['appliances']
        
        # Build the convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = self.input_channels
        
        for i, (num_filters, filter_size) in enumerate(zip(self.num_filters, self.filter_sizes)):
            conv_layer = nn.Conv1d(
                in_channels=in_channels,
                out_channels=num_filters,
                kernel_size=filter_size,
                padding=filter_size // 2  # Same padding
            )
            self.conv_layers.append(conv_layer)
            in_channels = num_filters
        
        # Calculate the size after convolutions
        self.conv_output_size = self._get_conv_output_size()
        
        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Fully connected layers for each appliance
        self.fc_layers = nn.ModuleDict()
        for appliance in self.appliances:
            self.fc_layers[appliance] = nn.Sequential(
                nn.Linear(self.conv_output_size, 128),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(64, 1)
            )
    
    def _get_conv_output_size(self) -> int:
        """Calculate the output size of convolutional layers."""
        # Create a dummy input to calculate output size
        dummy_input = torch.randn(1, self.input_channels, self.sequence_length)
        
        x = dummy_input
        for conv_layer in self.conv_layers:
            x = F.relu(conv_layer(x))
        
        # Global average pooling will reduce to (batch_size, num_filters)
        return x.size(1)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, sequence_length)
            
        Returns:
            Dictionary of predictions for each appliance
        """
        # Pass through convolutional layers
        for conv_layer in self.conv_layers:
            x = F.relu(conv_layer(x))
            x = self.dropout(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(-1)  # Remove the last dimension
        
        # Apply dropout
        x = self.dropout(x)
        
        # Generate predictions for each appliance
        predictions = {}
        for appliance in self.appliances:
            predictions[appliance] = self.fc_layers[appliance](x).squeeze(-1)
        
        return predictions
    
    def get_model_summary(self) -> str:
        """Get a summary of the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        summary = f"""
CNN Energy Disaggregation Model Summary:
========================================
Input Shape: ({self.input_channels}, {self.sequence_length})
Target Appliances: {', '.join(self.appliances)}

Convolutional Layers:
"""
        for i, (num_filters, filter_size) in enumerate(zip(self.num_filters, self.filter_sizes)):
            summary += f"  Conv1D_{i+1}: {num_filters} filters, kernel_size={filter_size}\n"
        
        summary += f"""
Fully Connected Layers (per appliance):
  FC1: {self.conv_output_size} -> 128
  FC2: 128 -> 64
  FC3: 64 -> 1

Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
"""
        return summary


class EnsembleCNNDisaggregator(nn.Module):
    """
    Ensemble model combining multiple CNN disaggregators.
    Can be used for improved performance and uncertainty estimation.
    """
    
    def __init__(self, config: Dict, num_models: int = 3):
        """
        Initialize the ensemble model.
        
        Args:
            config: Configuration dictionary
            num_models: Number of models in the ensemble
        """
        super(EnsembleCNNDisaggregator, self).__init__()
        
        self.num_models = num_models
        self.appliances = config['data']['appliances']
        
        # Create multiple CNN models
        self.models = nn.ModuleList([
            CNNDisaggregator(config) for _ in range(num_models)
        ])
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of averaged predictions for each appliance
        """
        all_predictions = []
        
        # Get predictions from all models
        for model in self.models:
            predictions = model(x)
            all_predictions.append(predictions)
        
        # Average the predictions
        ensemble_predictions = {}
        for appliance in self.appliances:
            appliance_preds = torch.stack([pred[appliance] for pred in all_predictions])
            ensemble_predictions[appliance] = torch.mean(appliance_preds, dim=0)
        
        return ensemble_predictions
    
    def get_predictions_with_uncertainty(self, x: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Get predictions with uncertainty estimates.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary containing mean and std for each appliance
        """
        all_predictions = []
        
        # Get predictions from all models
        for model in self.models:
            predictions = model(x)
            all_predictions.append(predictions)
        
        # Calculate mean and standard deviation
        result = {}
        for appliance in self.appliances:
            appliance_preds = torch.stack([pred[appliance] for pred in all_predictions])
            result[appliance] = {
                'mean': torch.mean(appliance_preds, dim=0),
                'std': torch.std(appliance_preds, dim=0)
            }
        
        return result