"""
Visualization utilities for energy disaggregation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import os


class EnergyVisualization:
    """Class for visualizing energy consumption data and model results."""
    
    def __init__(self, style: str = "seaborn-v0_8"):
        """
        Initialize visualization settings.
        
        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use("default")
        
        sns.set_palette("husl")
        self.colors = sns.color_palette("husl", 8)
    
    def plot_aggregate_vs_appliances(
        self,
        aggregate: np.ndarray,
        appliances: Dict[str, np.ndarray],
        time_range: Optional[tuple] = None,
        title: str = "Energy Consumption Breakdown",
        save_path: Optional[str] = None
    ):
        """
        Plot aggregate consumption alongside individual appliances.
        
        Args:
            aggregate: Aggregate power consumption
            appliances: Dictionary of appliance power consumption
            time_range: Tuple of (start_idx, end_idx) for plotting range
            title: Plot title
            save_path: Path to save the plot
        """
        if time_range:
            start, end = time_range
            aggregate = aggregate[start:end]
            appliances = {k: v[start:end] for k, v in appliances.items()}
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot aggregate consumption
        x = np.arange(len(aggregate))
        ax1.plot(x, aggregate, color='black', linewidth=2, label='Aggregate')
        ax1.set_title('Aggregate Power Consumption')
        ax1.set_ylabel('Power (W)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot individual appliances
        for i, (appliance, consumption) in enumerate(appliances.items()):
            color = self.colors[i % len(self.colors)]
            ax2.plot(x, consumption, color=color, label=appliance.capitalize(), alpha=0.8)
        
        ax2.set_title('Individual Appliance Consumption')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Power (W)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_training_history(
        self,
        history: Dict,
        save_path: Optional[str] = None
    ):
        """
        Plot training history including losses.
        
        Args:
            history: Training history dictionary
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot overall losses
        epochs = range(1, len(history['train_losses']) + 1)
        axes[0].plot(epochs, history['train_losses'], label='Training Loss', marker='o')
        axes[0].plot(epochs, history['val_losses'], label='Validation Loss', marker='s')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot appliance-specific validation losses
        for i, (appliance, losses) in enumerate(history['appliance_losses'].items()):
            color = self.colors[i % len(self.colors)]
            axes[1].plot(epochs, losses, label=appliance.capitalize(), 
                        color=color, marker='o')
        
        axes[1].set_title('Appliance-specific Validation Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        
        plt.show()
    
    def plot_daily_patterns(
        self,
        data: Dict[str, np.ndarray],
        samples_per_day: int = 14400,  # Assuming 6-second intervals: 24*60*10
        title: str = "Daily Energy Consumption Patterns",
        save_path: Optional[str] = None
    ):
        """
        Plot daily consumption patterns for each appliance.
        
        Args:
            data: Dictionary of appliance consumption data
            samples_per_day: Number of samples per day
            title: Plot title
            save_path: Path to save the plot
        """
        num_appliances = len(data)
        fig, axes = plt.subplots(num_appliances, 1, figsize=(15, 4 * num_appliances))
        
        if num_appliances == 1:
            axes = [axes]
        
        for i, (appliance, consumption) in enumerate(data.items()):
            # Reshape data into days
            num_days = len(consumption) // samples_per_day
            if num_days > 0:
                daily_data = consumption[:num_days * samples_per_day].reshape(num_days, samples_per_day)
                
                # Calculate hourly averages
                samples_per_hour = samples_per_day // 24
                hourly_data = daily_data.reshape(num_days, 24, samples_per_hour).mean(axis=2)
                
                # Plot each day
                for day in range(min(7, num_days)):  # Plot up to 7 days
                    alpha = 0.3 if day > 0 else 0.8
                    axes[i].plot(range(24), hourly_data[day], alpha=alpha, 
                               color=self.colors[i % len(self.colors)])
                
                # Plot average pattern
                avg_pattern = hourly_data.mean(axis=0)
                axes[i].plot(range(24), avg_pattern, linewidth=3, 
                           color='black', label='Average')
                
                axes[i].set_title(f'{appliance.capitalize()} Daily Pattern')
                axes[i].set_xlabel('Hour of Day')
                axes[i].set_ylabel('Average Power (W)')
                axes[i].set_xticks(range(0, 24, 4))
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Daily patterns plot saved to: {save_path}")
        
        plt.show()
    
    def plot_error_distribution(
        self,
        y_true: Dict[str, np.ndarray],
        y_pred: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ):
        """
        Plot error distribution for each appliance.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save the plot
        """
        num_appliances = len(y_true)
        fig, axes = plt.subplots(num_appliances, 1, figsize=(12, 4 * num_appliances))
        
        if num_appliances == 1:
            axes = [axes]
        
        for i, appliance in enumerate(y_true.keys()):
            if appliance in y_pred:
                errors = y_true[appliance] - y_pred[appliance]
                
                # Plot histogram
                axes[i].hist(errors, bins=50, alpha=0.7, density=True, 
                           color=self.colors[i % len(self.colors)])
                
                # Add statistics
                mean_error = np.mean(errors)
                std_error = np.std(errors)
                axes[i].axvline(mean_error, color='red', linestyle='--', 
                              label=f'Mean: {mean_error:.2f}')
                axes[i].axvline(mean_error + std_error, color='orange', 
                              linestyle=':', alpha=0.7, label=f'±1σ: {std_error:.2f}')
                axes[i].axvline(mean_error - std_error, color='orange', 
                              linestyle=':', alpha=0.7)
                
                axes[i].set_title(f'{appliance.capitalize()} Prediction Error Distribution')
                axes[i].set_xlabel('Error (True - Predicted)')
                axes[i].set_ylabel('Density')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Error distribution plot saved to: {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(
        self,
        data: Dict[str, np.ndarray],
        title: str = "Appliance Power Consumption Correlation",
        save_path: Optional[str] = None
    ):
        """
        Plot correlation matrix between appliances.
        
        Args:
            data: Dictionary of appliance consumption data
            title: Plot title
            save_path: Path to save the plot
        """
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Compute correlation matrix
        corr_matrix = df.corr()
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.3f')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation matrix saved to: {save_path}")
        
        plt.show()
    
    def create_dashboard(
        self,
        y_true: Dict[str, np.ndarray],
        y_pred: Dict[str, np.ndarray],
        history: Optional[Dict] = None,
        save_dir: str = "results"
    ):
        """
        Create a comprehensive visualization dashboard.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            history: Training history (optional)
            save_dir: Directory to save plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print("Generating visualization dashboard...")
        
        # 1. Predictions comparison
        self.plot_predictions_comparison(
            y_true, y_pred,
            save_path=os.path.join(save_dir, "predictions_comparison.png")
        )
        
        # 2. Error distributions
        self.plot_error_distribution(
            y_true, y_pred,
            save_path=os.path.join(save_dir, "error_distribution.png")
        )
        
        # 3. Daily patterns (if enough data)
        sample_length = min([len(v) for v in y_true.values()])
        if sample_length > 14400:  # More than 1 day of data
            self.plot_daily_patterns(
                y_true,
                save_path=os.path.join(save_dir, "daily_patterns.png")
            )
        
        # 4. Correlation matrix
        self.plot_correlation_matrix(
            y_true,
            save_path=os.path.join(save_dir, "correlation_matrix.png")
        )
        
        # 5. Training history (if available)
        if history:
            self.plot_training_history(
                history,
                save_path=os.path.join(save_dir, "training_history.png")
            )
        
        print(f"Dashboard generated in: {save_dir}")
    
    def plot_predictions_comparison(
        self,
        y_true: Dict[str, np.ndarray],
        y_pred: Dict[str, np.ndarray],
        sample_length: int = 1440,
        save_path: Optional[str] = None
    ):
        """
        Plot predictions vs true values comparison.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            sample_length: Length of sample to plot
            save_path: Path to save the plot
        """
        num_appliances = len(y_true)
        fig, axes = plt.subplots(num_appliances, 1, figsize=(15, 4 * num_appliances))
        
        if num_appliances == 1:
            axes = [axes]
        
        for i, appliance in enumerate(y_true.keys()):
            if appliance in y_pred:
                true_vals = y_true[appliance][:sample_length]
                pred_vals = y_pred[appliance][:sample_length]
                
                x = np.arange(len(true_vals))
                
                axes[i].plot(x, true_vals, label='True', alpha=0.8, 
                           linewidth=2, color='blue')
                axes[i].plot(x, pred_vals, label='Predicted', alpha=0.8, 
                           linewidth=2, color='red')
                
                axes[i].set_title(f'{appliance.capitalize()} Power Consumption Comparison')
                axes[i].set_xlabel('Time Steps')
                axes[i].set_ylabel('Power (W)')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Predictions comparison saved to: {save_path}")
        
        plt.show()