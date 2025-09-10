"""
PyTorch Dataset classes for energy disaggregation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Tuple, List


class EnergyDisaggregationDataset(Dataset):
    """PyTorch Dataset for energy disaggregation task."""
    
    def __init__(self, X: np.ndarray, y: Dict[str, np.ndarray], appliances: List[str]):
        """
        Initialize the dataset.
        
        Args:
            X: Input sequences (aggregate power consumption)
            y: Target sequences for each appliance
            appliances: List of appliance names
        """
        self.X = torch.FloatTensor(X)
        self.y = {appliance: torch.FloatTensor(y[appliance]) for appliance in appliances}
        self.appliances = appliances
        
        # Reshape X to add channel dimension (N, 1, sequence_length)
        if len(self.X.shape) == 2:
            self.X = self.X.unsqueeze(1)
    
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (input_sequence, target_dict)
        """
        input_seq = self.X[idx]
        targets = {appliance: self.y[appliance][idx] for appliance in self.appliances}
        
        return input_seq, targets


class DataLoaderFactory:
    """Factory class for creating data loaders."""
    
    @staticmethod
    def create_data_loaders(
        X: np.ndarray, 
        y: Dict[str, np.ndarray], 
        appliances: List[str],
        batch_size: int = 32,
        validation_split: float = 0.2,
        test_split: float = 0.1,
        shuffle: bool = True
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test data loaders.
        
        Args:
            X: Input sequences
            y: Target sequences for each appliance
            appliances: List of appliance names
            batch_size: Batch size for data loaders
            validation_split: Fraction of data for validation
            test_split: Fraction of data for testing
            shuffle: Whether to shuffle the data
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Calculate split indices
        total_samples = len(X)
        test_size = int(total_samples * test_split)
        val_size = int(total_samples * validation_split)
        train_size = total_samples - test_size - val_size
        
        # Create indices
        indices = np.arange(total_samples)
        if shuffle:
            np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Split the data
        X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]
        
        y_train = {app: y[app][train_indices] for app in appliances}
        y_val = {app: y[app][val_indices] for app in appliances}
        y_test = {app: y[app][test_indices] for app in appliances}
        
        # Create datasets
        train_dataset = EnergyDisaggregationDataset(X_train, y_train, appliances)
        val_dataset = EnergyDisaggregationDataset(X_val, y_val, appliances)
        test_dataset = EnergyDisaggregationDataset(X_test, y_test, appliances)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        return train_loader, val_loader, test_loader