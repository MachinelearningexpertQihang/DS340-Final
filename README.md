# Stock Prediction and Automated Trading System

A deep learning-based stock prediction system using a GRU-Transformer hybrid model for price prediction and automated trading strategy backtesting.

## Features

- 🤖 Hybrid Deep Learning Model (GRU-Transformer) for price prediction
- 📈 Support for multiple technical indicators
- 🔄 Automated trading strategy backtesting system
- 📊 Detailed performance evaluation and visualization
- ⚙️ Parameter optimization functionality
- 🎯 Stop-loss and take-profit risk management

## System Requirements

- Python 3.8+
- PyTorch
- pandas
- numpy
- scikit-learn
- matplotlib
- yaml

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd stock_prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

Download stock data:
```bash
python scripts/download_data.py
```

### 2. Train Model

Train the prediction model:
```bash
python main.py --mode train
```

### 3. Backtest Trading Strategy

Run backtest (with default parameters):
```bash
python main.py --mode backtest
```

Run backtest (with parameter optimization):
```bash
python main.py --mode backtest --optimize
```

### 4. Predict Future Trends

Generate 30-day predictions:
```bash
python main.py --mode predict
```

## Project Structure

```
stock_prediction/
├── config.yaml          # Configuration file
├── data/               # Data directory
│   ├── raw/           # Raw data
│   └── processed/     # Processed data
├── models/            # Model directory
│   ├── gru.py
│   ├── transformer.py
│   └── gru_transformer.py
├── scripts/           # Script files
│   ├── download_data.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── backtest.py
│   └── utils.py
└── results/           # Results output directory
```

## Configuration

Main parameters can be configured in `config.yaml`:

- Data parameters: sequence length, train-test split ratio, etc.
- Model parameters: model type, hidden dimensions, Transformer parameters, etc.
- Training parameters: batch size, learning rate, epochs, etc.
- Evaluation parameters: whether to perform inverse normalization, etc.

## Trading Strategy

Current implemented trading strategies include:

- Prediction-based trend trading
- Stop-loss and take-profit mechanisms
- Position management
- Transaction cost consideration

## Performance Metrics

Latest backtest results show:

- Total Return: +15.84%
- Annual Return: +3.96%
- Volatility: 17.27%
- Sharpe Ratio: 0.1137
- Maximum Drawdown: -16.35%
- Number of Trades: 390
- Win Rate: 18.97%
- Volatility Prediction Accuracy: 77.33%
- Average Position Size: 50%

