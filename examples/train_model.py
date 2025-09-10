#!/usr/bin/env python3
"""
Training script for CNN-based energy disaggregation model.
This script demonstrates how to train a model on UK-DALE dataset.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import argparse
from pathlib import Path

from src.data.preprocessing import UKDALEPreprocessor
from src.data.dataset import DataLoaderFactory
from src.models.cnn_model import CNNDisaggregator
from src.training.trainer import EnergyDisaggregationTrainer
from src.utils.config import ConfigManager
from src.utils.visualization import EnergyVisualization


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Energy Disaggregation Model')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, default='data/uk_dale/house_1.h5',
                       help='Path to UK-DALE data file')
    parser.add_argument('--output', type=str, default='models',
                       help='Output directory for trained model')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config_manager = ConfigManager(args.config)
    if not config_manager.validate_config():
        print("Invalid configuration. Exiting.")
        return
    
    config = config_manager.config
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data preprocessor
    print("Initializing data preprocessor...")
    preprocessor = UKDALEPreprocessor(config)
    
    # Prepare data
    print("Preparing training data...")
    try:
        X, y = preprocessor.prepare_data(args.data, building_id=1)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using synthetic data for demonstration...")
        X, y = preprocessor.prepare_data("dummy_path", building_id=1)
    
    print(f"Data prepared: {X.shape[0]} samples, sequence length: {X.shape[1]}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = DataLoaderFactory.create_data_loaders(
        X, y, 
        appliances=config['data']['appliances'],
        batch_size=config['training']['batch_size'],
        validation_split=config['training']['validation_split']
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Initialize model
    print("Initializing model...")
    model = CNNDisaggregator(config)
    print(model.get_model_summary())
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = EnergyDisaggregationTrainer(model, config, device=device)
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(train_loader, val_loader, save_dir=str(output_dir))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_predictions = trainer.predict(test_loader)
    
    # Get test targets for evaluation
    test_targets = {appliance: [] for appliance in config['data']['appliances']}
    for _, targets in test_loader:
        for appliance in config['data']['appliances']:
            test_targets[appliance].extend(targets[appliance].numpy())
    
    for appliance in test_targets:
        test_targets[appliance] = np.array(test_targets[appliance])
    
    # Inverse transform predictions and targets (if normalized)
    for appliance in config['data']['appliances']:
        if appliance in preprocessor.scalers:
            test_predictions[appliance] = preprocessor.inverse_transform(
                appliance, test_predictions[appliance]
            )
            test_targets[appliance] = preprocessor.inverse_transform(
                appliance, test_targets[appliance]
            )
    
    # Create visualizations
    print("Generating visualizations...")
    visualizer = EnergyVisualization()
    
    # Plot training history
    visualizer.plot_training_history(
        history, 
        save_path=output_dir / "training_history.png"
    )
    
    # Plot predictions vs true values
    visualizer.plot_predictions_comparison(
        test_targets, test_predictions, 
        save_path=output_dir / "test_predictions.png"
    )
    
    # Calculate and display basic metrics
    from src.evaluation.metrics import EvaluationMetrics
    evaluator = EvaluationMetrics(config['data']['appliances'])
    metrics = evaluator.compute_metrics(test_targets, test_predictions)
    evaluator.print_metrics_summary(metrics)
    
    # Save final configuration
    config_manager.save_config(str(output_dir / "final_config.yaml"))
    
    print(f"\nTraining completed successfully!")
    print(f"Model and results saved to: {output_dir}")
    print(f"Best validation loss: {history['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()