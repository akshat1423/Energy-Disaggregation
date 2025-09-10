"""
Training pipeline for energy disaggregation models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime


class EnergyDisaggregationTrainer:
    """Trainer class for energy disaggregation models."""
    
    def __init__(self, model: nn.Module, config: Dict, device: str = 'cpu'):
        """
        Initialize the trainer.
        
        Args:
            model: The neural network model
            config: Configuration dictionary
            device: Device to train on ('cpu' or 'cuda')
        """
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        
        # Training configuration
        training_config = config['training']
        self.learning_rate = training_config['learning_rate']
        self.num_epochs = training_config['num_epochs']
        self.early_stopping_patience = training_config['early_stopping_patience']
        
        # Initialize optimizer
        if training_config['optimizer'].lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif training_config['optimizer'].lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {training_config['optimizer']}")
        
        # Initialize loss function
        if training_config['loss_function'].lower() == 'mse':
            self.criterion = nn.MSELoss()
        elif training_config['loss_function'].lower() == 'mae':
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {training_config['loss_function']}")
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.appliance_losses = {appliance: [] for appliance in config['data']['appliances']}
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(self.device)
            
            # Move targets to device
            targets_device = {}
            for appliance, target in targets.items():
                targets_device[appliance] = target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(inputs)
            
            # Calculate loss for each appliance
            total_loss = 0
            appliance_losses = {}
            
            for appliance in self.config['data']['appliances']:
                appliance_loss = self.criterion(
                    predictions[appliance], 
                    targets_device[appliance]
                )
                appliance_losses[appliance] = appliance_loss.item()
                total_loss += appliance_loss
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            epoch_loss += total_loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Avg Loss': f'{epoch_loss/num_batches:.4f}'
            })
        
        return epoch_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Tuple of (average_validation_loss, appliance_losses)
        """
        self.model.eval()
        epoch_loss = 0.0
        appliance_losses = {appliance: 0.0 for appliance in self.config['data']['appliances']}
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                
                # Move targets to device
                targets_device = {}
                for appliance, target in targets.items():
                    targets_device[appliance] = target.to(self.device)
                
                # Forward pass
                predictions = self.model(inputs)
                
                # Calculate loss for each appliance
                total_loss = 0
                for appliance in self.config['data']['appliances']:
                    appliance_loss = self.criterion(
                        predictions[appliance], 
                        targets_device[appliance]
                    )
                    appliance_losses[appliance] += appliance_loss.item()
                    total_loss += appliance_loss
                
                epoch_loss += total_loss.item()
                num_batches += 1
        
        # Average the losses
        avg_loss = epoch_loss / num_batches
        for appliance in appliance_losses:
            appliance_losses[appliance] /= num_batches
        
        return avg_loss, appliance_losses
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        save_dir: str = "models"
    ) -> Dict:
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            save_dir: Directory to save the best model
            
        Returns:
            Training history dictionary
        """
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model: {type(self.model).__name__}")
        
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_appliance_losses = self.validate_epoch(val_loader)
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            for appliance, loss in val_appliance_losses.items():
                self.appliance_losses[appliance].append(loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            for appliance, loss in val_appliance_losses.items():
                print(f"  {appliance.capitalize()} Val Loss: {loss:.4f}")
            
            # Early stopping and model saving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save the best model
                model_path = os.path.join(save_dir, "best_model.pth")
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'config': self.config,
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'train_loss': train_loss
                }, model_path)
                print(f"  New best model saved with val loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                
            # Check early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'appliance_losses': self.appliance_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'training_time': datetime.now().isoformat()
        }
        
        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Model and history saved in: {save_dir}")
        
        return history
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Validation loss: {checkpoint['val_loss']:.4f}")
        
        return checkpoint
    
    def predict(self, data_loader: DataLoader) -> Dict[str, np.ndarray]:
        """
        Make predictions on a dataset.
        
        Args:
            data_loader: DataLoader for the dataset
            
        Returns:
            Dictionary of predictions for each appliance
        """
        self.model.eval()
        
        all_predictions = {appliance: [] for appliance in self.config['data']['appliances']}
        
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(self.device)
                predictions = self.model(inputs)
                
                for appliance in self.config['data']['appliances']:
                    pred_cpu = predictions[appliance].cpu().numpy()
                    all_predictions[appliance].extend(pred_cpu)
        
        # Convert to numpy arrays
        for appliance in all_predictions:
            all_predictions[appliance] = np.array(all_predictions[appliance])
        
        return all_predictions