# Energy Disaggregation using CNN

A complete CNN-based energy disaggregation system for separating aggregate power consumption into individual appliances using the UK-DALE dataset. This system can identify and predict power consumption for HVAC systems, kitchen appliances, and electronics.

## 🎯 Overview

This project implements a deep learning solution for Non-Intrusive Load Monitoring (NILM), which analyzes aggregate power consumption data to identify individual appliance usage patterns. The system uses Convolutional Neural Networks (CNNs) to learn temporal patterns in energy consumption and disaggregate them into appliance-specific signals.

### Key Features

- **CNN-based Architecture**: Advanced convolutional neural network designed for time-series energy data
- **Multi-appliance Support**: Simultaneous disaggregation of HVAC, kitchen appliances, and electronics
- **Comprehensive Pipeline**: Complete data preprocessing, training, and evaluation workflow
- **Synthetic Data Generation**: Built-in UK-DALE-like synthetic data for testing and demonstration
- **Extensive Evaluation**: Multiple metrics including MAE, RMSE, R², NRMSE, MAPE, and SAE
- **Rich Visualizations**: Detailed plots and analysis tools for model interpretation
- **Ensemble Support**: Ensemble models for improved accuracy and uncertainty estimation

## 📁 Project Structure

```
Energy-Disaggregation/
├── src/
│   ├── data/
│   │   ├── preprocessing.py      # UK-DALE data preprocessing
│   │   └── dataset.py           # PyTorch dataset classes
│   ├── models/
│   │   └── cnn_model.py         # CNN architecture definitions
│   ├── training/
│   │   └── trainer.py           # Training pipeline
│   ├── evaluation/
│   │   └── metrics.py           # Evaluation metrics and tools
│   └── utils/
│       ├── config.py            # Configuration management
│       └── visualization.py     # Visualization utilities
├── examples/
│   ├── train_model.py           # Training script
│   └── evaluate_model.py        # Evaluation script
├── tests/
│   ├── test_data.py             # Data processing tests
│   └── test_model.py            # Model tests
├── config/
│   └── model_config.yaml        # Configuration file
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/akshat1423/Energy-Disaggregation.git
cd Energy-Disaggregation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training a Model

Train a CNN model using the built-in synthetic data:

```bash
python examples/train_model.py --config config/model_config.yaml --output models/
```

For custom data:
```bash
python examples/train_model.py --config config/model_config.yaml --data path/to/ukdale/house_1.h5 --output models/
```

### Evaluating a Model

Evaluate a trained model:

```bash
python examples/evaluate_model.py --model models/best_model.pth --output results/
```

## 🏗️ Model Architecture

The CNN model consists of:

- **Input Layer**: Processes aggregate power consumption sequences (599 time steps by default)
- **Convolutional Layers**: 4 conv1d layers with filters [30, 30, 40, 50] and kernel sizes [10, 8, 6, 5]
- **Global Average Pooling**: Reduces temporal dimension while preserving features
- **Fully Connected Layers**: Separate prediction heads for each appliance (HVAC, kitchen, electronics)
- **Output**: Individual appliance power predictions

### Model Configuration

Key parameters in `config/model_config.yaml`:

```yaml
data:
  sequence_length: 599        # Input sequence length
  appliances: 
    - "hvac"
    - "kitchen" 
    - "electronics"

model:
  architecture:
    input_channels: 1
    num_filters: [30, 30, 40, 50]
    filter_sizes: [10, 8, 6, 5]
    dropout_rate: 0.2

training:
  batch_size: 512
  learning_rate: 0.001
  num_epochs: 100
  early_stopping_patience: 10
```

## 📊 Evaluation Metrics

The system provides comprehensive evaluation metrics:

- **MAE** (Mean Absolute Error): Average absolute difference between predictions and true values
- **RMSE** (Root Mean Square Error): Square root of average squared differences
- **R²** (Coefficient of Determination): Proportion of variance explained by the model
- **NRMSE** (Normalized RMSE): RMSE normalized by mean true value
- **MAPE** (Mean Absolute Percentage Error): Average percentage error
- **SAE** (Signal Aggregate Error): Error in total energy estimation
- **Energy Error**: Relative error in total energy consumption

## 📈 Visualization Tools

The system includes comprehensive visualization capabilities:

1. **Training History**: Loss curves during training
2. **Prediction Comparison**: Side-by-side comparison of true vs predicted values
3. **Error Distribution**: Histogram of prediction errors
4. **Scatter Plots**: Correlation between true and predicted values
5. **Daily Patterns**: Average daily consumption patterns
6. **Correlation Matrix**: Inter-appliance consumption correlations
7. **Energy Breakdown**: Aggregate vs individual appliance consumption

## 🔧 Advanced Usage

### Custom Model Configuration

Create a custom configuration file:

```yaml
# custom_config.yaml
data:
  sequence_length: 1000       # Longer sequences
  appliances: ["washing_machine", "dishwasher", "microwave"]

model:
  architecture:
    num_filters: [64, 64, 128, 256]  # Larger model
    filter_sizes: [15, 10, 5, 3]
    dropout_rate: 0.3
```

### Ensemble Models

Use ensemble models for improved performance:

```python
from src.models.cnn_model import EnsembleCNNDisaggregator

# Create ensemble with 5 models
ensemble_model = EnsembleCNNDisaggregator(config, num_models=5)

# Get predictions with uncertainty estimates
predictions_with_uncertainty = ensemble_model.get_predictions_with_uncertainty(input_data)
```

### Custom Data Processing

For your own UK-DALE data:

```python
from src.data.preprocessing import UKDALEPreprocessor

preprocessor = UKDALEPreprocessor(config)
X, y = preprocessor.prepare_data("path/to/ukdale/house_1.h5", building_id=1)
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test files
python -m unittest tests/test_data.py
python -m unittest tests/test_model.py
```

## 📋 Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- PyYAML
- tqdm
- h5py

## 📄 Data Format

The system expects UK-DALE format data:
- HDF5 files with hierarchical structure
- Timestamp and power consumption columns
- Separate channels for aggregate and individual appliances

For demonstration purposes, the system can generate synthetic UK-DALE-like data that mimics real consumption patterns.

## 🔍 Results Interpretation

After training and evaluation, you'll get:

1. **Model Performance**: Detailed metrics for each appliance
2. **Visualization Dashboard**: Comprehensive plots and analysis
3. **Error Analysis**: Understanding where the model succeeds and fails
4. **Energy Breakdown**: How well the model reconstructs aggregate consumption

### Sample Results

Typical performance on synthetic data:
- **HVAC**: MAE ~45W, R² ~0.85
- **Kitchen**: MAE ~25W, R² ~0.75  
- **Electronics**: MAE ~15W, R² ~0.90

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- UK-DALE dataset creators for providing the benchmark dataset
- PyTorch team for the deep learning framework
- Energy disaggregation research community for inspiring methodologies

## 📞 Support

For questions or issues:
1. Check the [Issues](https://github.com/akshat1423/Energy-Disaggregation/issues) page
2. Create a new issue with detailed description
3. Provide configuration files and error messages when applicable

## 🚀 Future Enhancements

- [ ] Support for additional appliance types
- [ ] Real-time disaggregation capabilities
- [ ] Transfer learning between buildings
- [ ] Attention mechanisms for improved accuracy
- [ ] Mobile/edge deployment optimizations