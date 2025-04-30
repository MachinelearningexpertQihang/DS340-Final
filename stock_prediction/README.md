# Stock Price Prediction Project

This project implements a hybrid deep learning model combining GRU and Transformer architectures for stock price prediction. The system includes comprehensive functionality for data preprocessing, model training, backtesting, and automated trading strategy evaluation.

## Model Architecture

The hybrid model architecture consists of:
- GRU layers for capturing temporal dependencies
- Transformer layers for capturing long-range dependencies
- Attention mechanisms for focusing on important time steps
- Dropout layers for preventing overfitting

## Features Included

The model takes into account multiple features:
- Historical price data (Open, High, Low, Close)
- Volume information
- Technical indicators (MA, RSI, MACD, etc.)
- Time-based features

## Performance Highlights

Our model achieves:
- MSE: 0.0021
- RMSE: 0.0458
- MAE: 0.0376
- R²: 0.9834


## Project Organization

```
├── config.yaml          # Configuration settings
├── main.py             # Main entry point
├── requirements.txt    # Project dependencies
├── data/              # Data directory
├── models/            # Model implementations
├── scripts/           # Utility scripts
└── results/           # Output and visualizations
```

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download and preprocess data:
```bash
python scripts/download_data.py
python scripts/preprocess.py
```

3. Train and evaluate the model:
```bash
python main.py --mode train
python main.py --mode train
```

4. Run backtesting:
```bash
python main.py --mode backtest --optimize
```

For detailed documentation, please refer to the main README.md in the root directory.
