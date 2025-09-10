"""
Configuration management utilities.
"""

import yaml
import json
import os
from typing import Dict, Any


class ConfigManager:
    """Configuration management class."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = {}
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        file_ext = os.path.splitext(config_path)[1].lower()
        
        try:
            with open(config_path, 'r') as f:
                if file_ext in ['.yaml', '.yml']:
                    self.config = yaml.safe_load(f)
                elif file_ext == '.json':
                    self.config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {file_ext}")
            
            print(f"Configuration loaded from: {config_path}")
            return self.config
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return {}
    
    def save_config(self, config_path: str = None) -> bool:
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save configuration file
            
        Returns:
            True if successful, False otherwise
        """
        if config_path is None:
            config_path = self.config_path
        
        if config_path is None:
            print("No config path specified")
            return False
        
        file_ext = os.path.splitext(config_path)[1].lower()
        
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as f:
                if file_ext in ['.yaml', '.yml']:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                elif file_ext == '.json':
                    json.dump(self.config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {file_ext}")
            
            print(f"Configuration saved to: {config_path}")
            return True
            
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports nested keys with dots)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key.
        
        Args:
            key: Configuration key (supports nested keys with dots)
            value: Value to set
        """
        keys = key.split('.')
        config_ref = self.config
        
        # Navigate to the parent dict
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        
        # Set the value
        config_ref[keys[-1]] = value
    
    def update(self, other_config: Dict[str, Any]) -> None:
        """
        Update configuration with another dictionary.
        
        Args:
            other_config: Configuration dictionary to merge
        """
        self._deep_update(self.config, other_config)
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """
        Deep update of nested dictionaries.
        
        Args:
            base_dict: Base dictionary to update
            update_dict: Update dictionary
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def validate_config(self) -> bool:
        """
        Validate configuration completeness.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        required_keys = [
            'data.sequence_length',
            'data.appliances',
            'model.type',
            'training.batch_size',
            'training.learning_rate',
            'training.num_epochs'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                print(f"Missing required configuration key: {key}")
                return False
        
        # Validate appliances list
        appliances = self.get('data.appliances', [])
        if not appliances:
            print("No appliances specified in configuration")
            return False
        
        # Validate model architecture
        if self.get('model.type') == 'cnn':
            required_arch_keys = [
                'model.architecture.input_channels',
                'model.architecture.num_filters',
                'model.architecture.filter_sizes'
            ]
            
            for key in required_arch_keys:
                if self.get(key) is None:
                    print(f"Missing required CNN architecture key: {key}")
                    return False
        
        print("Configuration validation passed")
        return True
    
    def print_config(self) -> None:
        """Print formatted configuration."""
        print("Current Configuration:")
        print("=" * 50)
        self._print_dict(self.config, indent=0)
        print("=" * 50)
    
    def _print_dict(self, d: Dict, indent: int = 0) -> None:
        """
        Recursively print dictionary with indentation.
        
        Args:
            d: Dictionary to print
            indent: Current indentation level
        """
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                self._print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for energy disaggregation.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'data': {
            'sequence_length': 599,
            'window_size': 599,
            'sample_rate': 6,
            'appliances': ['hvac', 'kitchen', 'electronics'],
            'normalization': 'zscore',
            'noise_threshold': 10
        },
        'model': {
            'type': 'cnn',
            'architecture': {
                'input_channels': 1,
                'num_filters': [30, 30, 40, 50],
                'filter_sizes': [10, 8, 6, 5],
                'dropout_rate': 0.2,
                'output_size': 1
            }
        },
        'training': {
            'batch_size': 512,
            'learning_rate': 0.001,
            'num_epochs': 100,
            'validation_split': 0.2,
            'early_stopping_patience': 10,
            'optimizer': 'adam',
            'loss_function': 'mse'
        },
        'evaluation': {
            'metrics': ['mae', 'rmse', 'r2_score', 'nrmse']
        },
        'paths': {
            'data_dir': 'data/uk_dale',
            'models_dir': 'models',
            'results_dir': 'results',
            'logs_dir': 'logs'
        }
    }