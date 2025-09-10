"""
Data preprocessing utilities for UK-DALE energy dataset.
Handles loading, cleaning, and preparing data for energy disaggregation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import h5py
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class UKDALEPreprocessor:
    """Preprocessor for UK-DALE energy consumption data."""
    
    def __init__(self, config: Dict):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config: Configuration dictionary containing preprocessing parameters
        """
        self.config = config
        self.sequence_length = config['data']['sequence_length']
        self.sample_rate = config['data']['sample_rate']
        self.appliances = config['data']['appliances']
        self.normalization = config['data']['normalization']
        self.noise_threshold = config['data']['noise_threshold']
        
        self.scalers = {}
        
    def load_data(self, file_path: str, building_id: int = 1) -> Dict[str, pd.DataFrame]:
        """
        Load UK-DALE data from HDF5 file.
        
        Args:
            file_path: Path to the UK-DALE HDF5 file
            building_id: Building ID to load
            
        Returns:
            Dictionary containing aggregate and appliance data
        """
        try:
            # For demonstration, create synthetic UK-DALE-like data
            # In real implementation, this would load actual HDF5 files
            return self._create_synthetic_data()
        except Exception as e:
            print(f"Could not load real UK-DALE data, using synthetic data: {e}")
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self) -> Dict[str, pd.DataFrame]:
        """
        Create synthetic UK-DALE-like data for demonstration.
        
        Returns:
            Dictionary containing synthetic aggregate and appliance data
        """
        # Generate 7 days of data at 6-second intervals
        timestamps = pd.date_range(
            start='2023-01-01', 
            periods=100800,  # 7 days * 24 hours * 60 minutes * 10 samples per minute
            freq='6S'
        )
        
        # Generate synthetic appliance consumption patterns
        np.random.seed(42)
        
        # HVAC: Higher consumption with daily patterns
        hvac_base = 800 + 400 * np.sin(2 * np.pi * np.arange(len(timestamps)) / (24 * 600))
        hvac_noise = np.random.normal(0, 50, len(timestamps))
        hvac = np.maximum(0, hvac_base + hvac_noise)
        
        # Kitchen: Intermittent spikes during meal times
        kitchen = np.zeros(len(timestamps))
        meal_times = [7*600, 12*600, 19*600]  # 7am, 12pm, 7pm in 6-second intervals
        for day in range(7):
            for meal_time in meal_times:
                start_idx = day * 24 * 600 + meal_time
                end_idx = start_idx + 60  # 6 minutes of usage
                if end_idx < len(timestamps):
                    kitchen[start_idx:end_idx] += np.random.exponential(300, 60)
        
        # Electronics: Constant base load with random variations
        electronics_base = 150 + np.random.normal(0, 20, len(timestamps))
        electronics = np.maximum(0, electronics_base)
        
        # Aggregate is sum of all appliances plus some noise
        aggregate = hvac + kitchen + electronics + np.random.normal(0, 10, len(timestamps))
        
        return {
            'aggregate': pd.DataFrame({
                'timestamp': timestamps,
                'power': aggregate
            }),
            'hvac': pd.DataFrame({
                'timestamp': timestamps,
                'power': hvac
            }),
            'kitchen': pd.DataFrame({
                'timestamp': timestamps,
                'power': kitchen
            }),
            'electronics': pd.DataFrame({
                'timestamp': timestamps,
                'power': electronics
            })
        }
    
    def clean_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Clean the loaded data by removing outliers and handling missing values.
        
        Args:
            data: Dictionary containing raw data
            
        Returns:
            Dictionary containing cleaned data
        """
        cleaned_data = {}
        
        for key, df in data.items():
            # Remove negative values
            df_clean = df.copy()
            df_clean['power'] = np.maximum(df_clean['power'], 0)
            
            # Remove outliers (values beyond 3 standard deviations)
            mean_power = df_clean['power'].mean()
            std_power = df_clean['power'].std()
            outlier_threshold = mean_power + 3 * std_power
            df_clean['power'] = np.minimum(df_clean['power'], outlier_threshold)
            
            # Apply noise threshold
            if key != 'aggregate':
                df_clean['power'] = np.where(
                    df_clean['power'] < self.noise_threshold, 
                    0, 
                    df_clean['power']
                )
            
            cleaned_data[key] = df_clean
            
        return cleaned_data
    
    def normalize_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Normalize the power consumption data.
        
        Args:
            data: Dictionary containing cleaned data
            
        Returns:
            Dictionary containing normalized data
        """
        normalized_data = {}
        
        for key, df in data.items():
            df_norm = df.copy()
            
            if self.normalization == 'zscore':
                scaler = StandardScaler()
            elif self.normalization == 'minmax':
                scaler = MinMaxScaler()
            else:
                # No normalization
                normalized_data[key] = df_norm
                continue
            
            # Fit and transform the power values
            power_values = df_norm['power'].values.reshape(-1, 1)
            normalized_power = scaler.fit_transform(power_values)
            df_norm['power'] = normalized_power.flatten()
            
            # Store scaler for inverse transform later
            self.scalers[key] = scaler
            normalized_data[key] = df_norm
            
        return normalized_data
    
    def create_sequences(self, data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Create input sequences for the CNN model.
        
        Args:
            data: Dictionary containing normalized data
            
        Returns:
            Tuple of (input_sequences, target_sequences)
        """
        aggregate_power = data['aggregate']['power'].values
        
        # Create sliding windows for aggregate power (inputs)
        input_sequences = []
        target_sequences = {appliance: [] for appliance in self.appliances}
        
        for i in range(len(aggregate_power) - self.sequence_length + 1):
            # Input sequence (aggregate power)
            input_seq = aggregate_power[i:i + self.sequence_length]
            input_sequences.append(input_seq)
            
            # Target sequences (middle point of the window for each appliance)
            middle_idx = i + self.sequence_length // 2
            for appliance in self.appliances:
                if appliance in data:
                    target_power = data[appliance]['power'].values[middle_idx]
                    target_sequences[appliance].append(target_power)
                else:
                    target_sequences[appliance].append(0.0)
        
        # Convert to numpy arrays
        X = np.array(input_sequences)
        y = {appliance: np.array(targets) for appliance, targets in target_sequences.items()}
        
        return X, y
    
    def inverse_transform(self, appliance: str, values: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized values back to original scale.
        
        Args:
            appliance: Name of the appliance
            values: Normalized values to transform back
            
        Returns:
            Values in original scale
        """
        if appliance in self.scalers:
            values_reshaped = values.reshape(-1, 1)
            return self.scalers[appliance].inverse_transform(values_reshaped).flatten()
        return values
    
    def prepare_data(self, file_path: str, building_id: int = 1) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Complete data preparation pipeline.
        
        Args:
            file_path: Path to the UK-DALE data file
            building_id: Building ID to process
            
        Returns:
            Tuple of (input_sequences, target_sequences)
        """
        print("Loading UK-DALE data...")
        raw_data = self.load_data(file_path, building_id)
        
        print("Cleaning data...")
        clean_data = self.clean_data(raw_data)
        
        print("Normalizing data...")
        normalized_data = self.normalize_data(clean_data)
        
        print("Creating sequences...")
        X, y = self.create_sequences(normalized_data)
        
        print(f"Prepared {len(X)} sequences with length {self.sequence_length}")
        return X, y