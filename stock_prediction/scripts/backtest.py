import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import numpy._core.multiarray as ma

from models.gru import GRUModel
from models.transformer import TransformerModel
from models.gru_transformer import GRUTransformerModel
from scripts.utils import load_config, load_model, plot_predictions

class BacktestStrategy:
    """
    Backtesting strategy for volatility prediction and trading
    """
    def __init__(self, config_path='config.yaml', model_path=None):
        self.config = load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.config['training']['use_gpu'] else 'cpu')
        print(f"Using device: {self.device}")
        
        
        torch.serialization.add_safe_globals([MinMaxScaler, ma._reconstruct])
        
        # Load test data
        test_data_path = os.path.join(self.config['paths']['processed_data_dir'], 'test_data.pt')
        self.test_data = torch.load(test_data_path, weights_only=False)
        
        if 'scaler' not in self.test_data:
            raise KeyError("The 'scaler' key is missing in the test_data.pt file")
            
        self.X_test, self.y_test = self.test_data['X'], self.test_data['y']
        self.scaler = self.test_data['scaler']
        
        # Move data to device
        self.X_test = self.X_test.to(self.device)
        self.y_test = self.y_test.to(self.device)
        
        self.input_dim = self.X_test.shape[2]
        self.output_dim = self.y_test.shape[1]
        
        raw_data_path = os.path.join(self.config['paths']['raw_data_dir'], self.config['data']['filename'])
        self.raw_data = pd.read_csv(raw_data_path)
        self.raw_data['Date'] = pd.to_datetime(self.raw_data['Datetime'])
        
        self._initialize_model(model_path)
        
        # Strategy parameters
        self.max_position = 1.0
        self.high_vol_threshold = 0.3  # High volatility threshold
        self.low_vol_threshold = 0.1   # Low volatility threshold
        
    def _initialize_model(self, model_path=None):
        """Initialize the model based on configuration"""
        model_type = self.config['model']['type']
        
        if model_type == 'gru':
            self.model = GRUModel(
                input_dim=self.input_dim,
                hidden_dim=self.config['model']['gru']['hidden_dim'],
                num_layers=self.config['model']['gru']['num_layers'],
                output_dim=self.output_dim,
                dropout=self.config['model']['dropout']
            )
        elif model_type == 'transformer':
            self.model = TransformerModel(
                input_dim=self.input_dim,
                d_model=self.config['model']['transformer']['d_model'],
                nhead=self.config['model']['transformer']['nhead'],
                num_layers=self.config['model']['transformer']['num_layers'],
                dim_feedforward=self.config['model']['transformer']['dim_feedforward'],
                output_dim=self.output_dim,
                dropout=self.config['model']['dropout'],
                max_len=self.config['data']['seq_length']
            )
        elif model_type == 'gru_transformer':
            self.model = GRUTransformerModel(
                input_dim=self.input_dim,
                gru_hidden_dim=self.config['model']['gru']['hidden_dim'],
                gru_num_layers=self.config['model']['gru']['num_layers'],
                transformer_d_model=self.config['model']['transformer']['d_model'],
                transformer_nhead=self.config['model']['transformer']['nhead'],
                transformer_num_layers=self.config['model']['transformer']['num_layers'],
                transformer_dim_feedforward=self.config['model']['transformer']['dim_feedforward'],
                output_dim=self.output_dim,
                dropout=self.config['model']['dropout'],
                max_len=self.config['data']['seq_length']
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Load model weights
        if model_path is None:
            model_path = os.path.join(self.config['paths']['model_dir'], 'best_model.pth')
        self.model_path = model_path
            
        self.model = load_model(self.model, model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def _generate_signals(self, prices, predicted_volatility):
        """
        Generate trading signals based on predicted volatility
        
        Strategy logic:
        - Reduce position size when predicted volatility is high
        - Increase position size when predicted volatility is low
        - Consider volatility trend for position direction
        """
        signals = np.zeros(len(predicted_volatility))
        position_sizes = np.ones(len(predicted_volatility))
        
        for i in range(1, len(predicted_volatility)):
            # Calculate volatility trend
            vol_trend = predicted_volatility[i] - predicted_volatility[i-1]
            
            # Adjust position size based on volatility level
            if predicted_volatility[i] > self.high_vol_threshold:
                position_sizes[i] = self.max_position * 0.5  # Reduce position in high volatility
            elif predicted_volatility[i] < self.low_vol_threshold:
                position_sizes[i] = self.max_position  # Full position in low volatility
            else:
                # Linear scaling between thresholds
                vol_range = self.high_vol_threshold - self.low_vol_threshold
                scale = (self.high_vol_threshold - predicted_volatility[i]) / vol_range
                position_sizes[i] = self.max_position * (0.5 + 0.5 * scale)
            
            # Determine position direction based on volatility trend
            if vol_trend > 0:
                signals[i] = 1  # Long when volatility is increasing
            else:
                signals[i] = -1  # Short when volatility is decreasing
                
            # Apply position sizing
            signals[i] *= position_sizes[i]
        
        return signals
    
    def calculate_performance_metrics(self, backtest_df):
        """
        Calculate performance metrics for volatility trading strategy
        """
        returns = backtest_df['Strategy Returns']
        
        # Standard metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - 0.02) / volatility  # Assuming 2% risk-free rate
        
        # Drawdown analysis
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Volatility-specific metrics
        vol_prediction_accuracy = backtest_df['Directional Accuracy'].mean()
        avg_position_size = np.abs(backtest_df['Position']).mean()
        
        # Trading metrics
        trades = backtest_df['Position'].diff() != 0
        num_trades = trades.sum()
        winning_trades = returns[trades] > 0
        win_rate = winning_trades.mean() if num_trades > 0 else 0
        
        return {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Number of Trades': num_trades,
            'Win Rate': win_rate,
            'Volatility Prediction Accuracy': vol_prediction_accuracy,
            'Average Position Size': avg_position_size
        }
    
    def run(self):
        """
        Run the backtesting strategy
        
        Returns:
            tuple: (backtest_df, metrics) containing the backtest results and performance metrics
        """
        # 确保使用正确的数据加载方式
        torch.serialization.add_safe_globals([MinMaxScaler, ma._reconstruct])
        X_test, y_test = self.X_test.cpu().numpy(), self.y_test.cpu().numpy()
        
        # 获取日期范围
        train_size = int(len(self.raw_data) * self.config['data']['train_test_split'])
        seq_length = self.config['data']['seq_length']
        test_dates = self.raw_data['Date'].iloc[train_size + seq_length:train_size + seq_length + len(y_test)].reset_index(drop=True)
        
        # 生成预测
        with torch.no_grad():
            predictions = self.model(self.X_test).cpu().numpy()
        
        # 创建回测DataFrame
        backtest_df = pd.DataFrame({
            'Date': test_dates,
            'Actual Volatility': y_test.flatten(),
            'Predicted Volatility': predictions.flatten(),
            'Position': np.zeros(len(predictions))  # Will be filled by trading strategy
        })
        
        # Generate trading signals
        signals = self._generate_signals(backtest_df['Actual Volatility'].values, 
                                      backtest_df['Predicted Volatility'].values)
        backtest_df['Position'] = signals
        
        # Calculate returns
        backtest_df['Market Returns'] = backtest_df['Actual Volatility'].pct_change()
        backtest_df['Strategy Returns'] = backtest_df['Position'].shift(1) * backtest_df['Market Returns']
        
        # Calculate directional accuracy
        backtest_df['Directional Accuracy'] = (
            (backtest_df['Actual Volatility'].diff() > 0) == 
            (backtest_df['Predicted Volatility'].diff() > 0)
        ).astype(float)
        
        # Drop NaN values from returns calculations
        backtest_df = backtest_df.dropna()
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(backtest_df)
        
        return backtest_df, metrics

def plot_backtest_results(backtest_df):
    """
    Plot backtest results including cumulative returns and volatility predictions with trade signals
    
    Args:
        backtest_df (pd.DataFrame): DataFrame containing backtest results
    """
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Plot cumulative returns with trade signals
    plt.figure(figsize=(15, 8))
    cumulative_returns = (1 + backtest_df['Strategy Returns']).cumprod()
    market_returns = (1 + backtest_df['Market Returns']).cumprod()
    
    # Plot returns
    plt.plot(backtest_df['Date'], cumulative_returns, label='Strategy Returns', color='blue')
    plt.plot(backtest_df['Date'], market_returns, label='Market Returns', color='gray', alpha=0.5)
    
    # Add trade signals
    long_signals = backtest_df[backtest_df['Position'] > 0]
    short_signals = backtest_df[backtest_df['Position'] < 0]
    
    # Plot long and short positions
    plt.scatter(long_signals['Date'], 
               (1 + long_signals['Strategy Returns']).cumprod(),
               color='green', marker='^', label='Long Signal', alpha=0.7)
    plt.scatter(short_signals['Date'],
               (1 + short_signals['Strategy Returns']).cumprod(),
               color='red', marker='v', label='Short Signal', alpha=0.7)
    
    plt.title('Cumulative Returns and Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/backtest_cumulative_returns.png')
    plt.close()
    
    # Plot volatility predictions vs actual with trade signals
    plt.figure(figsize=(15, 8))
    plt.plot(backtest_df['Date'], backtest_df['Actual Volatility'], 
             label='Actual Volatility', color='gray')
    plt.plot(backtest_df['Date'], backtest_df['Predicted Volatility'],
             label='Predicted Volatility', color='blue')
    
    # Add trade signals on volatility plot
    plt.scatter(long_signals['Date'], 
               long_signals['Predicted Volatility'],
               color='green', marker='^', label='Long Signal', alpha=0.7)
    plt.scatter(short_signals['Date'],
               short_signals['Predicted Volatility'],
               color='red', marker='v', label='Short Signal', alpha=0.7)
    
    plt.title('Volatility Prediction and Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/backtest_volatility.png')
    plt.close()
    
    # Plot position sizes over time
    plt.figure(figsize=(15, 8))
    plt.plot(backtest_df['Date'], backtest_df['Position'], color='blue')
    plt.fill_between(backtest_df['Date'], 
                    backtest_df['Position'],
                    0, 
                    where=(backtest_df['Position'] > 0),
                    color='green',
                    alpha=0.3,
                    label='Long Position')
    plt.fill_between(backtest_df['Date'],
                    backtest_df['Position'],
                    0,
                    where=(backtest_df['Position'] < 0),
                    color='red',
                    alpha=0.3,
                    label='Short Position')
    
    plt.title('Strategy Position Size')
    plt.xlabel('Date')
    plt.ylabel('Position Size')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/backtest_positions.png')
    plt.close()
    
    # Plot drawdown
    plt.figure(figsize=(15, 8))
    cumulative_returns = (1 + backtest_df['Strategy Returns']).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    
    plt.plot(backtest_df['Date'], drawdown, color='red')
    plt.fill_between(backtest_df['Date'], drawdown, 0, color='red', alpha=0.3)
    
    plt.title('Strategy Drawdown Analysis')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/backtest_drawdown.png')
    plt.close()
    
    # Plot trading accuracy over time
    plt.figure(figsize=(15, 8))
    rolling_accuracy = backtest_df['Directional Accuracy'].rolling(window=20).mean()
    plt.plot(backtest_df['Date'], rolling_accuracy, color='blue', label='20-Period Moving Average')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Level (50%)')
    
    plt.title('Strategy Prediction Accuracy (20-Period Moving Average)')
    plt.xlabel('Date')
    plt.ylabel('Prediction Accuracy')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('results/backtest_accuracy.png')
    plt.close()

def run_backtest(config_path='config.yaml', model_path=None, optimize=False):
    """
    Run backtesting analysis
    
    Args:
        config_path (str): Path to config file
        model_path (str, optional): Path to model checkpoint
        optimize (bool): Whether to optimize strategy parameters
    """
    print("Initializing backtesting strategy...")
    strategy = BacktestStrategy(config_path, model_path)
    
    print("Running backtest simulation...")
    backtest_df, metrics = strategy.run()
    
    print("\nBacktest Results:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            if 'Rate' in metric or 'Return' in metric:
                print(f"{metric}: {value:.2%}")
            else:
                print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    print("\nGenerating plots...")
    plot_backtest_results(backtest_df)
    
    # Save results
    backtest_df.to_csv('results/backtest_results.csv', index=False)
    pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value']).to_csv('results/backtest_metrics.csv', index=False)
    
    print("\nBacktest completed. Results saved to 'results' directory.")
    return backtest_df, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run backtesting for volatility prediction model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--optimize', action='store_true', help='Optimize strategy parameters')
    args = parser.parse_args()
    
    run_backtest(args.config, args.model_path, args.optimize)