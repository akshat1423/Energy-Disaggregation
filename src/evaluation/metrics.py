"""
Evaluation metrics for energy disaggregation models.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class EvaluationMetrics:
    """Class for computing and managing evaluation metrics."""
    
    def __init__(self, appliances: List[str]):
        """
        Initialize the evaluation metrics.
        
        Args:
            appliances: List of appliance names
        """
        self.appliances = appliances
    
    def compute_metrics(
        self, 
        y_true: Dict[str, np.ndarray], 
        y_pred: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute evaluation metrics for all appliances.
        
        Args:
            y_true: True values for each appliance
            y_pred: Predicted values for each appliance
            
        Returns:
            Dictionary of metrics for each appliance
        """
        metrics = {}
        
        for appliance in self.appliances:
            if appliance not in y_true or appliance not in y_pred:
                continue
                
            true_values = y_true[appliance]
            pred_values = y_pred[appliance]
            
            # Ensure same shape
            min_len = min(len(true_values), len(pred_values))
            true_values = true_values[:min_len]
            pred_values = pred_values[:min_len]
            
            appliance_metrics = {}
            
            # Mean Absolute Error (MAE)
            appliance_metrics['mae'] = mean_absolute_error(true_values, pred_values)
            
            # Root Mean Square Error (RMSE)
            appliance_metrics['rmse'] = np.sqrt(mean_squared_error(true_values, pred_values))
            
            # R-squared Score
            appliance_metrics['r2_score'] = r2_score(true_values, pred_values)
            
            # Normalized RMSE (NRMSE)
            mean_true = np.mean(true_values)
            if mean_true > 0:
                appliance_metrics['nrmse'] = appliance_metrics['rmse'] / mean_true
            else:
                appliance_metrics['nrmse'] = float('inf')
            
            # Mean Absolute Percentage Error (MAPE)
            appliance_metrics['mape'] = self._compute_mape(true_values, pred_values)
            
            # Signal Aggregate Error (SAE)
            appliance_metrics['sae'] = self._compute_sae(true_values, pred_values)
            
            # Energy-based metrics
            total_energy_true = np.sum(true_values)
            total_energy_pred = np.sum(pred_values)
            
            if total_energy_true > 0:
                appliance_metrics['energy_error'] = abs(total_energy_true - total_energy_pred) / total_energy_true
            else:
                appliance_metrics['energy_error'] = float('inf')
            
            metrics[appliance] = appliance_metrics
        
        return metrics
    
    def _compute_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Mean Absolute Percentage Error (MAPE).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAPE value
        """
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return float('inf')
        
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def _compute_sae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Signal Aggregate Error (SAE).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            SAE value
        """
        sum_true = np.sum(y_true)
        sum_pred = np.sum(y_pred)
        
        if sum_true == 0:
            return float('inf')
        
        return abs(sum_true - sum_pred) / sum_true
    
    def print_metrics_summary(self, metrics: Dict[str, Dict[str, float]]):
        """
        Print a formatted summary of metrics.
        
        Args:
            metrics: Computed metrics dictionary
        """
        print("\n" + "="*60)
        print("ENERGY DISAGGREGATION EVALUATION RESULTS")
        print("="*60)
        
        for appliance in self.appliances:
            if appliance in metrics:
                print(f"\n{appliance.upper()}:")
                print("-" * 30)
                app_metrics = metrics[appliance]
                
                print(f"MAE:          {app_metrics['mae']:.4f}")
                print(f"RMSE:         {app_metrics['rmse']:.4f}")
                print(f"NRMSE:        {app_metrics['nrmse']:.4f}")
                print(f"R² Score:     {app_metrics['r2_score']:.4f}")
                print(f"MAPE:         {app_metrics['mape']:.2f}%")
                print(f"SAE:          {app_metrics['sae']:.4f}")
                print(f"Energy Error: {app_metrics['energy_error']:.4f}")
        
        # Overall summary
        print(f"\n{'OVERALL SUMMARY'}")
        print("-" * 30)
        
        avg_mae = np.mean([metrics[app]['mae'] for app in metrics])
        avg_rmse = np.mean([metrics[app]['rmse'] for app in metrics])
        avg_r2 = np.mean([metrics[app]['r2_score'] for app in metrics])
        
        print(f"Average MAE:   {avg_mae:.4f}")
        print(f"Average RMSE:  {avg_rmse:.4f}")
        print(f"Average R²:    {avg_r2:.4f}")
        print("="*60)
    
    def plot_predictions(
        self, 
        y_true: Dict[str, np.ndarray], 
        y_pred: Dict[str, np.ndarray],
        sample_length: int = 1440,  # 24 hours at 1-minute intervals
        save_path: str = None
    ):
        """
        Plot predictions vs true values for all appliances.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            sample_length: Length of sample to plot
            save_path: Path to save the plot
        """
        num_appliances = len(self.appliances)
        fig, axes = plt.subplots(num_appliances, 1, figsize=(15, 4 * num_appliances))
        
        if num_appliances == 1:
            axes = [axes]
        
        for i, appliance in enumerate(self.appliances):
            if appliance not in y_true or appliance not in y_pred:
                continue
            
            true_vals = y_true[appliance][:sample_length]
            pred_vals = y_pred[appliance][:sample_length]
            
            x = np.arange(len(true_vals))
            
            axes[i].plot(x, true_vals, label='True', alpha=0.7, color='blue')
            axes[i].plot(x, pred_vals, label='Predicted', alpha=0.7, color='red')
            axes[i].set_title(f'{appliance.capitalize()} Power Consumption')
            axes[i].set_xlabel('Time Steps')
            axes[i].set_ylabel('Power (W)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_scatter_comparison(
        self, 
        y_true: Dict[str, np.ndarray], 
        y_pred: Dict[str, np.ndarray],
        save_path: str = None
    ):
        """
        Plot scatter plots comparing predictions vs true values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save the plot
        """
        num_appliances = len(self.appliances)
        fig, axes = plt.subplots(1, num_appliances, figsize=(5 * num_appliances, 5))
        
        if num_appliances == 1:
            axes = [axes]
        
        for i, appliance in enumerate(self.appliances):
            if appliance not in y_true or appliance not in y_pred:
                continue
            
            true_vals = y_true[appliance]
            pred_vals = y_pred[appliance]
            
            # Sample points for better visualization
            if len(true_vals) > 5000:
                indices = np.random.choice(len(true_vals), 5000, replace=False)
                true_vals = true_vals[indices]
                pred_vals = pred_vals[indices]
            
            axes[i].scatter(true_vals, pred_vals, alpha=0.5, s=1)
            
            # Perfect prediction line
            max_val = max(np.max(true_vals), np.max(pred_vals))
            axes[i].plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
            
            axes[i].set_xlabel('True Values (W)')
            axes[i].set_ylabel('Predicted Values (W)')
            axes[i].set_title(f'{appliance.capitalize()} Predictions')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Scatter plot saved to: {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(
        self, 
        metrics: Dict[str, Dict[str, float]],
        save_path: str = None
    ):
        """
        Plot a comparison of metrics across appliances.
        
        Args:
            metrics: Computed metrics
            save_path: Path to save the plot
        """
        metric_names = ['mae', 'rmse', 'r2_score', 'mape']
        metric_labels = ['MAE', 'RMSE', 'R² Score', 'MAPE (%)']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (metric_name, label) in enumerate(zip(metric_names, metric_labels)):
            values = []
            labels = []
            
            for appliance in self.appliances:
                if appliance in metrics:
                    values.append(metrics[appliance][metric_name])
                    labels.append(appliance.capitalize())
            
            bars = axes[i].bar(labels, values, alpha=0.7)
            axes[i].set_title(label)
            axes[i].set_ylabel(label)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics comparison saved to: {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(
        self,
        y_true: Dict[str, np.ndarray],
        y_pred: Dict[str, np.ndarray],
        model_name: str = "CNN Energy Disaggregation Model",
        save_dir: str = "results"
    ) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            save_dir: Directory to save results
            
        Returns:
            Path to the generated report
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Compute metrics
        metrics = self.compute_metrics(y_true, y_pred)
        
        # Generate plots
        self.plot_predictions(
            y_true, y_pred, 
            save_path=os.path.join(save_dir, "predictions_comparison.png")
        )
        
        self.plot_scatter_comparison(
            y_true, y_pred,
            save_path=os.path.join(save_dir, "scatter_comparison.png")
        )
        
        self.plot_metrics_comparison(
            metrics,
            save_path=os.path.join(save_dir, "metrics_comparison.png")
        )
        
        # Generate text report
        report_path = os.path.join(save_dir, "evaluation_report.txt")
        with open(report_path, 'w') as f:
            f.write(f"Energy Disaggregation Evaluation Report\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            for appliance in self.appliances:
                if appliance in metrics:
                    f.write(f"{appliance.upper()}:\n")
                    f.write("-" * 30 + "\n")
                    app_metrics = metrics[appliance]
                    
                    f.write(f"MAE:          {app_metrics['mae']:.4f}\n")
                    f.write(f"RMSE:         {app_metrics['rmse']:.4f}\n")
                    f.write(f"NRMSE:        {app_metrics['nrmse']:.4f}\n")
                    f.write(f"R² Score:     {app_metrics['r2_score']:.4f}\n")
                    f.write(f"MAPE:         {app_metrics['mape']:.2f}%\n")
                    f.write(f"SAE:          {app_metrics['sae']:.4f}\n")
                    f.write(f"Energy Error: {app_metrics['energy_error']:.4f}\n\n")
            
            # Overall summary
            avg_mae = np.mean([metrics[app]['mae'] for app in metrics])
            avg_rmse = np.mean([metrics[app]['rmse'] for app in metrics])
            avg_r2 = np.mean([metrics[app]['r2_score'] for app in metrics])
            
            f.write("OVERALL SUMMARY:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average MAE:   {avg_mae:.4f}\n")
            f.write(f"Average RMSE:  {avg_rmse:.4f}\n")
            f.write(f"Average R²:    {avg_r2:.4f}\n")
        
        print(f"Evaluation report generated: {report_path}")
        return report_path