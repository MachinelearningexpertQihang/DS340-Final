import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
from tqdm import tqdm

from models.gru import GRUModel
from models.transformer import TransformerModel
from models.gru_transformer import GRUTransformerModel
from scripts.utils import load_config, load_model, plot_predictions

class BacktestStrategy:
    """
    Backtesting strategy for stock prediction model
    """
    def __init__(self, config_path='config.yaml', model_path=None):
        """
        Initialize backtesting strategy
        
        Args:
            config_path (str): Path to config file
            model_path (str, optional): Path to model checkpoint. If None, use the best model.
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.config['training']['use_gpu'] else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load test data
        test_data_path = os.path.join(self.config['paths']['processed_data_dir'], 'test_data.pt')
        self.test_data = torch.load(test_data_path)
        self.X_test, self.y_test = self.test_data['X'], self.test_data['y']
        self.scaler = self.test_data['scaler']
        
        # Move data to device
        self.X_test = self.X_test.to(self.device)
        self.y_test = self.y_test.to(self.device)
        
        # Get input dimensions
        self.input_dim = self.X_test.shape[2]  # Number of features
        self.output_dim = self.y_test.shape[1]  # Number of output dimensions
        
        # Load raw data for dates
        raw_data_path = os.path.join(self.config['paths']['raw_data_dir'], self.config['data']['filename'])
        self.raw_data = pd.read_csv(raw_data_path)
        self.raw_data['Date'] = pd.to_datetime(self.raw_data['Date'])
        
        # Initialize model
        self._initialize_model(model_path)
        
        # Trading parameters (can be set via set_parameters method)
        self.initial_capital = 10000.0
        self.position_size = 1.0  # Fraction of capital to use per trade
        self.stop_loss = 0.05     # 5% stop loss
        self.take_profit = 0.10   # 10% take profit
        self.commission = 0.001   # 0.1% commission per trade
        
    def _initialize_model(self, model_path=None):
        """
        Initialize the model based on configuration
        
        Args:
            model_path (str, optional): Path to model checkpoint. If None, use the best model.
        """
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
        
        self.model = load_model(self.model, model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def set_parameters(self, initial_capital=10000.0, position_size=1.0, 
                      stop_loss=0.05, take_profit=0.10, commission=0.001):
        """
        Set trading parameters
        
        Args:
            initial_capital (float): Initial capital
            position_size (float): Fraction of capital to use per trade (0.0-1.0)
            stop_loss (float): Stop loss percentage (0.0-1.0)
            take_profit (float): Take profit percentage (0.0-1.0)
            commission (float): Commission percentage per trade (0.0-1.0)
        """
        self.initial_capital = initial_capital
        self.position_size = max(0.0, min(1.0, position_size))  # Ensure between 0 and 1
        self.stop_loss = max(0.0, stop_loss)
        self.take_profit = max(0.0, take_profit)
        self.commission = max(0.0, commission)
    
    def _get_prediction_dates(self):
        """
        Get dates for the test predictions
        
        Returns:
            list: List of dates corresponding to test predictions
        """
        # Get the sequence length
        seq_length = self.config['data']['seq_length']
        
        # Calculate the start index in the raw data
        train_size = int(len(self.raw_data) * self.config['data']['train_test_split'])
        start_idx = train_size + seq_length
        
        # Get the dates
        dates = self.raw_data['Date'].iloc[start_idx:start_idx + len(self.y_test)].reset_index(drop=True)
        
        return dates
    
    def _get_predictions(self):
        """
        Get model predictions on test data
        
        Returns:
            tuple: (actual_prices, predicted_prices, dates)
        """
        with torch.no_grad():
            y_pred = self.model(self.X_test)
        
        # Move tensors to CPU for evaluation
        y_test_cpu = self.y_test.cpu().numpy()
        y_pred_cpu = y_pred.cpu().numpy()
        
        # Inverse transform predictions if needed
        if self.config['evaluation']['inverse_transform']:
            # Create dummy arrays with the same shape as the original data
            y_test_dummy = np.zeros((len(y_test_cpu), self.scaler.n_features_in_))
            y_pred_dummy = np.zeros((len(y_pred_cpu), self.scaler.n_features_in_))
            
            # Put the predicted values in the first column (assuming Close price is the target)
            y_test_dummy[:, 0] = y_test_cpu.flatten()
            y_pred_dummy[:, 0] = y_pred_cpu.flatten()
            
            # Inverse transform
            y_test_inv = self.scaler.inverse_transform(y_test_dummy)[:, 0]
            y_pred_inv = self.scaler.inverse_transform(y_pred_dummy)[:, 0]
        else:
            y_test_inv = y_test_cpu.flatten()
            y_pred_inv = y_pred_cpu.flatten()
        
        # Get dates
        dates = self._get_prediction_dates()
        
        return y_test_inv, y_pred_inv, dates
    
    def _generate_signals(self, actual_prices, predicted_prices):
        """
        Generate trading signals based on predictions
        
        Args:
            actual_prices (numpy.ndarray): Actual prices
            predicted_prices (numpy.ndarray): Predicted prices
            
        Returns:
            pandas.DataFrame: DataFrame with signals
        """
        # Calculate price change percentage for next day
        price_change_pct = np.zeros_like(predicted_prices)
        price_change_pct[:-1] = (predicted_prices[1:] - actual_prices[:-1]) / actual_prices[:-1]
        
        # Generate signals: 1 for buy, -1 for sell, 0 for hold
        signals = np.zeros_like(price_change_pct)
        signals[price_change_pct > 0.01] = 1    # Buy if predicted increase > 1%
        signals[price_change_pct < -0.01] = -1  # Sell if predicted decrease > 1%
        
        return signals
    
    def run_backtest(self):
        """
        Run backtest simulation
        
        Returns:
            pandas.DataFrame: DataFrame with backtest results
        """
        # Get predictions and dates
        actual_prices, predicted_prices, dates = self._get_predictions()
        
        # Generate signals
        signals = self._generate_signals(actual_prices, predicted_prices)
        
        # Create DataFrame for backtest results
        backtest_df = pd.DataFrame({
            'Date': dates,
            'Actual_Price': actual_prices,
            'Predicted_Price': predicted_prices,
            'Signal': signals
        })
        
        # Initialize trading variables
        capital = self.initial_capital
        shares = 0
        entry_price = 0
        position_type = 0  # 0: no position, 1: long, -1: short
        stop_loss_price = 0
        take_profit_price = 0
        
        # Add columns for tracking
        backtest_df['Capital'] = 0.0
        backtest_df['Shares'] = 0
        backtest_df['Position'] = 0
        backtest_df['Trade_PnL'] = 0.0
        backtest_df['Trade'] = 0
        
        # Run simulation
        trade_count = 0
        
        for i in range(len(backtest_df)):
            current_price = backtest_df.loc[i, 'Actual_Price']
            current_signal = backtest_df.loc[i, 'Signal']
            
            # Check if we need to close position due to stop loss or take profit
            if position_type != 0:
                # For long positions
                if position_type == 1:
                    if current_price <= stop_loss_price or current_price >= take_profit_price:
                        # Close long position
                        trade_pnl = shares * (current_price * (1 - self.commission) - entry_price)
                        capital += shares * current_price * (1 - self.commission)
                        backtest_df.loc[i, 'Trade_PnL'] = trade_pnl
                        backtest_df.loc[i, 'Trade'] = -1  # Close position
                        shares = 0
                        position_type = 0
                
                # For short positions (if implemented)
                elif position_type == -1:
                    if current_price >= stop_loss_price or current_price <= take_profit_price:
                        # Close short position (simplified)
                        trade_pnl = shares * (entry_price - current_price * (1 + self.commission))
                        capital += shares * entry_price + trade_pnl
                        backtest_df.loc[i, 'Trade_PnL'] = trade_pnl
                        backtest_df.loc[i, 'Trade'] = 1  # Close position
                        shares = 0
                        position_type = 0
            
            # Check for new signals if we don't have an open position
            if position_type == 0:
                if current_signal == 1:  # Buy signal
                    # Open long position
                    position_size_amount = capital * self.position_size
                    shares = position_size_amount / (current_price * (1 + self.commission))
                    entry_price = current_price * (1 + self.commission)
                    capital -= position_size_amount
                    position_type = 1
                    stop_loss_price = entry_price * (1 - self.stop_loss)
                    take_profit_price = entry_price * (1 + self.take_profit)
                    trade_count += 1
                    backtest_df.loc[i, 'Trade'] = 1  # Open position
                
                elif current_signal == -1 and False:  # Sell signal (short selling disabled by default)
                    # Open short position (simplified)
                    position_size_amount = capital * self.position_size
                    shares = position_size_amount / current_price
                    entry_price = current_price
                    position_type = -1
                    stop_loss_price = entry_price * (1 + self.stop_loss)
                    take_profit_price = entry_price * (1 - self.take_profit)
                    trade_count += 1
                    backtest_df.loc[i, 'Trade'] = -1  # Open position
            
            # Update tracking columns
            backtest_df.loc[i, 'Capital'] = capital
            backtest_df.loc[i, 'Shares'] = shares
            backtest_df.loc[i, 'Position'] = position_type
            
            # Calculate portfolio value
            portfolio_value = capital
            if shares > 0:
                portfolio_value += shares * current_price
            
            backtest_df.loc[i, 'Portfolio_Value'] = portfolio_value
        
        # Calculate returns
        backtest_df['Daily_Return'] = backtest_df['Portfolio_Value'].pct_change()
        
        # Calculate cumulative returns
        backtest_df['Cumulative_Return'] = (1 + backtest_df['Daily_Return']).cumprod() - 1
        
        # Calculate buy and hold returns
        backtest_df['Buy_Hold_Value'] = self.initial_capital * (backtest_df['Actual_Price'] / backtest_df['Actual_Price'].iloc[0])
        backtest_df['Buy_Hold_Return'] = backtest_df['Buy_Hold_Value'].pct_change()
        backtest_df['Buy_Hold_Cumulative'] = (1 + backtest_df['Buy_Hold_Return']).cumprod() - 1
        
        return backtest_df
    
    def calculate_performance_metrics(self, backtest_df):
        """
        Calculate performance metrics from backtest results
        
        Args:
            backtest_df (pandas.DataFrame): DataFrame with backtest results
            
        Returns:
            dict: Dictionary of performance metrics
        """
        # Filter out NaN values
        returns = backtest_df['Daily_Return'].dropna()
        buy_hold_returns = backtest_df['Buy_Hold_Return'].dropna()
        
        # Calculate metrics
        total_return = backtest_df['Portfolio_Value'].iloc[-1] / self.initial_capital - 1
        buy_hold_return = backtest_df['Buy_Hold_Value'].iloc[-1] / self.initial_capital - 1
        
        # Annualized return (assuming 252 trading days per year)
        n_days = len(returns)
        ann_factor = 252 / n_days
        ann_return = (1 + total_return) ** ann_factor - 1
        ann_buy_hold_return = (1 + buy_hold_return) ** ann_factor - 1
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        buy_hold_volatility = buy_hold_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = ann_return / volatility if volatility != 0 else 0
        buy_hold_sharpe = ann_buy_hold_return / buy_hold_volatility if buy_hold_volatility != 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max - 1)
        max_drawdown = drawdown.min()
        
        buy_hold_cumulative = (1 + buy_hold_returns).cumprod()
        buy_hold_running_max = buy_hold_cumulative.cummax()
        buy_hold_drawdown = (buy_hold_cumulative / buy_hold_running_max - 1)
        buy_hold_max_drawdown = buy_hold_drawdown.min()
        
        # Win rate
        trades = backtest_df[backtest_df['Trade_PnL'] != 0]
        if len(trades) > 0:
            win_rate = len(trades[trades['Trade_PnL'] > 0]) / len(trades)
            avg_win = trades[trades['Trade_PnL'] > 0]['Trade_PnL'].mean() if len(trades[trades['Trade_PnL'] > 0]) > 0 else 0
            avg_loss = trades[trades['Trade_PnL'] < 0]['Trade_PnL'].mean() if len(trades[trades['Trade_PnL'] < 0]) > 0 else 0
            profit_factor = abs(trades[trades['Trade_PnL'] > 0]['Trade_PnL'].sum() / trades[trades['Trade_PnL'] < 0]['Trade_PnL'].sum()) if trades[trades['Trade_PnL'] < 0]['Trade_PnL'].sum() != 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Number of trades
        num_trades = len(trades)
        
        metrics = {
            'Total Return': total_return,
            'Buy & Hold Return': buy_hold_return,
            'Annualized Return': ann_return,
            'Buy & Hold Ann. Return': ann_buy_hold_return,
            'Volatility': volatility,
            'Buy & Hold Volatility': buy_hold_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Buy & Hold Sharpe': buy_hold_sharpe,
            'Max Drawdown': max_drawdown,
            'Buy & Hold Max Drawdown': buy_hold_max_drawdown,
            'Win Rate': win_rate,
            'Average Win': avg_win,
            'Average Loss': avg_loss,
            'Profit Factor': profit_factor,
            'Number of Trades': num_trades
        }
        
        return metrics
    
    def plot_results(self, backtest_df, metrics):
        """
        Plot backtest results
        
        Args:
            backtest_df (pandas.DataFrame): DataFrame with backtest results
            metrics (dict): Dictionary of performance metrics
        """
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Plot portfolio value vs buy & hold
        plt.figure(figsize=(14, 7))
        plt.plot(backtest_df['Date'], backtest_df['Portfolio_Value'], label='Strategy')
        plt.plot(backtest_df['Date'], backtest_df['Buy_Hold_Value'], label='Buy & Hold')
        plt.title('Portfolio Value vs Buy & Hold')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/backtest_portfolio_value.png')
        plt.close()
        
        # Plot cumulative returns
        plt.figure(figsize=(14, 7))
        plt.plot(backtest_df['Date'], backtest_df['Cumulative_Return'], label='Strategy')
        plt.plot(backtest_df['Date'], backtest_df['Buy_Hold_Cumulative'], label='Buy & Hold')
        plt.title('Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/backtest_cumulative_returns.png')
        plt.close()
        
        # Plot drawdown
        returns = backtest_df['Daily_Return'].dropna()
        buy_hold_returns = backtest_df['Buy_Hold_Return'].dropna()
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max - 1)
        
        buy_hold_cumulative = (1 + buy_hold_returns).cumprod()
        buy_hold_running_max = buy_hold_cumulative.cummax()
        buy_hold_drawdown = (buy_hold_cumulative / buy_hold_running_max - 1)
        
        plt.figure(figsize=(14, 7))
        plt.plot(backtest_df['Date'].iloc[1:], drawdown.values, label='Strategy')
        plt.plot(backtest_df['Date'].iloc[1:], buy_hold_drawdown.values, label='Buy & Hold')
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/backtest_drawdown.png')
        plt.close()
        
        # Plot trades
        trades = backtest_df[backtest_df['Trade'] != 0]
        if len(trades) > 0:
            plt.figure(figsize=(14, 7))
            plt.plot(backtest_df['Date'], backtest_df['Actual_Price'])
            
            # Plot buy signals
            buys = trades[trades['Trade'] == 1]
            plt.scatter(buys['Date'], buys['Actual_Price'], color='green', marker='^', s=100, label='Buy')
            
            # Plot sell signals
            sells = trades[trades['Trade'] == -1]
            plt.scatter(sells['Date'], sells['Actual_Price'], color='red', marker='v', s=100, label='Sell')
            
            plt.title('Trading Signals')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.savefig('results/backtest_trades.png')
            plt.close()
        
        # Create a summary table
        plt.figure(figsize=(10, 8))
        plt.axis('off')
        
        # Create table data
        table_data = []
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'Return' in key or 'Drawdown' in key or 'Rate' in key:
                    formatted_value = f"{value:.2%}"
                else:
                    formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            table_data.append([key, formatted_value])
        
        # Create table
        table = plt.table(
            cellText=table_data,
            colLabels=['Metric', 'Value'],
            cellLoc='center',
            loc='center',
            bbox=[0.2, 0.0, 0.6, 1.0]
        )
        
        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        plt.title('Backtest Performance Metrics', fontsize=16, pad=20)
        plt.savefig('results/backtest_metrics.png', bbox_inches='tight')
        plt.close()
        
        # Save results to CSV
        backtest_df.to_csv('results/backtest_results.csv', index=False)
        
        # Save metrics to CSV
        pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value']).to_csv('results/backtest_metrics.csv', index=False)
    
    def optimize_parameters(self, position_sizes=None, stop_losses=None, take_profits=None):
        """
        Optimize trading parameters
        
        Args:
            position_sizes (list, optional): List of position sizes to test
            stop_losses (list, optional): List of stop loss percentages to test
            take_profits (list, optional): List of take profit percentages to test
            
        Returns:
            tuple: (best_params, optimization_results)
        """
        if position_sizes is None:
            position_sizes = [0.25, 0.5, 0.75, 1.0]
        
        if stop_losses is None:
            stop_losses = [0.02, 0.05, 0.1]
        
        if take_profits is None:
            take_profits = [0.03, 0.05, 0.1, 0.15]
        
        results = []
        
        # Store original parameters
        orig_position_size = self.position_size
        orig_stop_loss = self.stop_loss
        orig_take_profit = self.take_profit
        
        total_combinations = len(position_sizes) * len(stop_losses) * len(take_profits)
        progress_bar = tqdm(total=total_combinations, desc="Optimizing Parameters")
        
        for ps in position_sizes:
            for sl in stop_losses:
                for tp in take_profits:
                    # Set parameters
                    self.set_parameters(position_size=ps, stop_loss=sl, take_profit=tp)
                    
                    # Run backtest
                    backtest_df = self.run_backtest()
                    metrics = self.calculate_performance_metrics(backtest_df)
                    
                    # Store results
                    results.append({
                        'Position_Size': ps,
                        'Stop_Loss': sl,
                        'Take_Profit': tp,
                        'Total_Return': metrics['Total Return'],
                        'Sharpe_Ratio': metrics['Sharpe Ratio'],
                        'Max_Drawdown': metrics['Max Drawdown'],
                        'Win_Rate': metrics['Win Rate'],
                        'Profit_Factor': metrics['Profit Factor'],
                        'Num_Trades': metrics['Number of Trades']
                    })
                    
                    progress_bar.update(1)
        
        progress_bar.close()
        
        # Restore original parameters
        self.set_parameters(position_size=orig_position_size, stop_loss=orig_stop_loss, take_profit=orig_take_profit)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Find best parameters based on Sharpe ratio
        best_sharpe_idx = results_df['Sharpe_Ratio'].idxmax()
        best_params = results_df.iloc[best_sharpe_idx][['Position_Size', 'Stop_Loss', 'Take_Profit']].to_dict()
        
        # Save optimization results
        results_df.to_csv('results/parameter_optimization.csv', index=False)
        
        return best_params, results_df
    
    def run_with_optimal_parameters(self):
        """
        Run backtest with optimal parameters
        
        Returns:
            pandas.DataFrame: DataFrame with backtest results
        """
        # Run parameter optimization
        best_params, _ = self.optimize_parameters()
        
        # Set optimal parameters
        self.set_parameters(
            position_size=best_params['Position_Size'],
            stop_loss=best_params['Stop_Loss'],
            take_profit=best_params['Take_Profit']
        )
        
        print(f"Running backtest with optimal parameters:")
        print(f"Position Size: {best_params['Position_Size']}")
        print(f"Stop Loss: {best_params['Stop_Loss']}")
        print(f"Take Profit: {best_params['Take_Profit']}")
        
        # Run backtest
        backtest_df = self.run_backtest()
        metrics = self.calculate_performance_metrics(backtest_df)
        
        # Plot results
        self.plot_results(backtest_df, metrics)
        
        return backtest_df, metrics

def run_backtest(config_path='config.yaml', model_path=None, optimize=False):
    """
    Run backtest
    
    Args:
        config_path (str): Path to config file
        model_path (str, optional): Path to model checkpoint. If None, use the best model.
        optimize (bool): Whether to optimize parameters
        
    Returns:
        tuple: (backtest_df, metrics)
    """
    # Initialize strategy
    strategy = BacktestStrategy(config_path, model_path)
    
    if optimize:
        # Run with optimal parameters
        backtest_df, metrics = strategy.run_with_optimal_parameters()
    else:
        # Run backtest
        backtest_df = strategy.run_backtest()
        metrics = strategy.calculate_performance_metrics(backtest_df)
        strategy.plot_results(backtest_df, metrics)
    
    # Print metrics
    print("\nBacktest Performance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'Return' in key or 'Drawdown' in key or 'Rate' in key:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    return backtest_df, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backtest stock prediction model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--optimize', action='store_true', help='Optimize trading parameters')
    
    args = parser.parse_args()
    
    run_backtest(args.config, args.model_path, args.optimize)