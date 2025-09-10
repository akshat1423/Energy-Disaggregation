#!/usr/bin/env python3
"""
Evaluation script for CNN-based energy disaggregation model.
This script demonstrates how to evaluate a trained model and generate comprehensive reports.
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
from src.evaluation.metrics import EvaluationMetrics
from src.utils.config import ConfigManager
from src.utils.visualization import EnergyVisualization


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Energy Disaggregation Model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (will try to load from model dir)')
    parser.add_argument('--data', type=str, default='data/uk_dale/house_1.h5',
                       help='Path to UK-DALE data file for evaluation')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for evaluation results')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for evaluation')
    parser.add_argument('--sample-length', type=int, default=2880,
                       help='Number of samples to use for visualization (0 for all)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model checkpoint
    print(f"Loading model from: {args.model}")
    checkpoint = torch.load(args.model, map_location=device)
    
    # Load configuration
    if args.config:
        config_path = args.config
    else:
        # Try to find config in model directory
        model_dir = Path(args.model).parent
        config_path = model_dir / "final_config.yaml"
        if not config_path.exists():
            config_path = "config/model_config.yaml"
    
    print(f"Loading configuration from: {config_path}")
    config_manager = ConfigManager(str(config_path))
    
    # Use config from checkpoint if available
    if 'config' in checkpoint:
        config_manager.update(checkpoint['config'])
    
    config = config_manager.config
    
    # Initialize model
    print("Initializing model...")
    model = CNNDisaggregator(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Training validation loss: {checkpoint.get('val_loss', 'unknown')}")
    
    # Initialize data preprocessor
    print("Initializing data preprocessor...")
    preprocessor = UKDALEPreprocessor(config)
    
    # Prepare evaluation data
    print("Preparing evaluation data...")
    try:
        X, y = preprocessor.prepare_data(args.data, building_id=1)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using synthetic data for demonstration...")
        X, y = preprocessor.prepare_data("dummy_path", building_id=1)
    
    print(f"Evaluation data: {X.shape[0]} samples")
    
    # Create data loader (using all data for evaluation)
    from torch.utils.data import DataLoader
    from src.data.dataset import EnergyDisaggregationDataset
    
    eval_dataset = EnergyDisaggregationDataset(X, y, config['data']['appliances'])
    eval_loader = DataLoader(eval_dataset, batch_size=512, shuffle=False)
    
    # Make predictions
    print("Making predictions...")
    trainer = EnergyDisaggregationTrainer(model, config, device=device)
    predictions = trainer.predict(eval_loader)
    
    # Get true values
    true_values = y.copy()
    
    # Inverse transform if normalization was applied
    print("Applying inverse transformations...")
    for appliance in config['data']['appliances']:
        if appliance in preprocessor.scalers:
            predictions[appliance] = preprocessor.inverse_transform(
                appliance, predictions[appliance]
            )
            true_values[appliance] = preprocessor.inverse_transform(
                appliance, true_values[appliance]
            )
    
    # Limit data for visualization if requested
    if args.sample_length > 0:
        sample_length = min(args.sample_length, len(next(iter(true_values.values()))))
        true_values_vis = {k: v[:sample_length] for k, v in true_values.items()}
        predictions_vis = {k: v[:sample_length] for k, v in predictions.items()}
    else:
        true_values_vis = true_values
        predictions_vis = predictions
    
    # Initialize evaluation metrics
    print("Computing evaluation metrics...")
    evaluator = EvaluationMetrics(config['data']['appliances'])
    
    # Compute metrics
    metrics = evaluator.compute_metrics(true_values, predictions)
    
    # Print metrics summary
    evaluator.print_metrics_summary(metrics)
    
    # Generate comprehensive evaluation report
    print("Generating evaluation report...")
    report_path = evaluator.generate_evaluation_report(
        true_values_vis, predictions_vis,
        model_name="CNN Energy Disaggregation Model",
        save_dir=str(output_dir)
    )
    
    # Create visualization dashboard
    print("Creating visualization dashboard...")
    visualizer = EnergyVisualization()
    
    # Load training history if available
    history_path = Path(args.model).parent / "training_history.json"
    history = None
    if history_path.exists():
        import json
        with open(history_path, 'r') as f:
            history = json.load(f)
    
    # Generate dashboard
    visualizer.create_dashboard(
        true_values_vis, predictions_vis,
        history=history,
        save_dir=str(output_dir)
    )
    
    # Additional detailed plots
    print("Generating additional visualizations...")
    
    # Error distribution
    visualizer.plot_error_distribution(
        true_values_vis, predictions_vis,
        save_path=output_dir / "detailed_error_distribution.png"
    )
    
    # Scatter plot comparison
    evaluator.plot_scatter_comparison(
        true_values_vis, predictions_vis,
        save_path=output_dir / "detailed_scatter_comparison.png"
    )
    
    # Metrics comparison bar chart
    evaluator.plot_metrics_comparison(
        metrics,
        save_path=output_dir / "detailed_metrics_comparison.png"
    )
    
    # Create aggregate vs individual appliances plot
    if len(true_values_vis[config['data']['appliances'][0]]) > 0:
        # Create synthetic aggregate for visualization
        aggregate_true = sum(true_values_vis.values())
        
        visualizer.plot_aggregate_vs_appliances(
            aggregate_true,
            true_values_vis,
            time_range=(0, min(1440, len(aggregate_true))),  # First 24 hours
            title="True Energy Consumption Breakdown",
            save_path=output_dir / "energy_breakdown_true.png"
        )
        
        aggregate_pred = sum(predictions_vis.values())
        visualizer.plot_aggregate_vs_appliances(
            aggregate_pred,
            predictions_vis,
            time_range=(0, min(1440, len(aggregate_pred))),  # First 24 hours
            title="Predicted Energy Consumption Breakdown",
            save_path=output_dir / "energy_breakdown_predicted.png"
        )
    
    # Save detailed metrics to JSON
    import json
    metrics_path = output_dir / "detailed_metrics.json"
    with open(metrics_path, 'w') as f:
        # Convert numpy types to Python native types for JSON serialization
        json_metrics = {}
        for appliance, app_metrics in metrics.items():
            json_metrics[appliance] = {k: float(v) for k, v in app_metrics.items()}
        
        json.dump({
            'metrics': json_metrics,
            'model_info': {
                'checkpoint_path': args.model,
                'epoch': checkpoint.get('epoch', 'unknown'),
                'training_val_loss': float(checkpoint.get('val_loss', 0)),
                'config': config
            },
            'evaluation_summary': {
                'total_samples': len(true_values[config['data']['appliances'][0]]),
                'appliances': config['data']['appliances'],
                'avg_mae': float(np.mean([metrics[app]['mae'] for app in metrics])),
                'avg_rmse': float(np.mean([metrics[app]['rmse'] for app in metrics])),
                'avg_r2': float(np.mean([metrics[app]['r2_score'] for app in metrics]))
            }
        }, f, indent=2)
    
    print(f"\nEvaluation completed successfully!")
    print(f"Results saved to: {output_dir}")
    print(f"Detailed report: {report_path}")
    print(f"Metrics summary: {metrics_path}")
    
    # Summary statistics
    print("\nSUMMARY STATISTICS:")
    print("-" * 40)
    avg_mae = np.mean([metrics[app]['mae'] for app in metrics])
    avg_rmse = np.mean([metrics[app]['rmse'] for app in metrics])
    avg_r2 = np.mean([metrics[app]['r2_score'] for app in metrics])
    
    print(f"Average MAE across appliances:  {avg_mae:.4f} W")
    print(f"Average RMSE across appliances: {avg_rmse:.4f} W")
    print(f"Average R² across appliances:   {avg_r2:.4f}")
    
    # Best and worst performing appliances
    mae_values = {app: metrics[app]['mae'] for app in metrics}
    best_appliance = min(mae_values.keys(), key=lambda x: mae_values[x])
    worst_appliance = max(mae_values.keys(), key=lambda x: mae_values[x])
    
    print(f"Best performing appliance:      {best_appliance} (MAE: {mae_values[best_appliance]:.4f})")
    print(f"Worst performing appliance:     {worst_appliance} (MAE: {mae_values[worst_appliance]:.4f})")


if __name__ == "__main__":
    main()