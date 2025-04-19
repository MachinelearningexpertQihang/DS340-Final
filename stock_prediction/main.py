import argparse
import os
from scripts.preprocess import preprocess_data
from scripts.train import train_model
from scripts.evaluate import evaluate_model, predict_future
from scripts.download_data import download_stock_data
from scripts.utils import load_config

def main():
    """
    Main entry point for the stock prediction project
    """
    parser = argparse.ArgumentParser(description='Stock Price Prediction with GRU+Transformer')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['download', 'preprocess', 'train', 'evaluate', 'predict', 'all'], 
                        help='Mode to run')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol for download mode')
    parser.add_argument('--period', type=str, default='5y', help='Period to download (e.g., 1y, 5y, max)')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint for evaluation/prediction')
    parser.add_argument('--days_ahead', type=int, default=30, help='Number of days to predict ahead')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create necessary directories
    os.makedirs(config['paths']['raw_data_dir'], exist_ok=True)
    os.makedirs(config['paths']['processed_data_dir'], exist_ok=True)
    os.makedirs(config['paths']['model_dir'], exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Download data if in download mode or if raw data doesn't exist
    raw_data_path = os.path.join(config['paths']['raw_data_dir'], config['data']['filename'])
    if args.mode == 'download' or (args.mode == 'all' and not os.path.exists(raw_data_path)):
        print(f"\n=== Downloading Stock Data for {args.ticker} ===")
        download_stock_data(args.ticker, period=args.period, config_path=args.config)
    
    # Check if raw data exists before proceeding
    if not os.path.exists(raw_data_path):
        print(f"Raw data file not found at {raw_data_path}")
        print("Please download stock data first using --mode download")
        return
    
    # Run the specified mode
    if args.mode == 'preprocess' or args.mode == 'all':
        print("\n=== Preprocessing Data ===")
        preprocess_data(args.config)
    
    if args.mode == 'train' or args.mode == 'all':
        print("\n=== Training Model ===")
        train_model(args.config)
    
    if args.mode == 'evaluate' or args.mode == 'all':
        print("\n=== Evaluating Model ===")
        evaluate_model(args.config, args.model_path)
    
    if args.mode == 'predict' or args.mode == 'all':
        print(f"\n=== Predicting {args.days_ahead} Days Ahead ===")
        predict_future(args.config, args.model_path, args.days_ahead)
    
    print("\n=== Process Completed ===")

if __name__ == "__main__":
    main()