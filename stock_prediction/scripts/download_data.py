import os
import argparse
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scripts.utils import load_config

def download_stock_data(ticker, start_date=None, end_date=None, period=None, interval='1h', config_path='config.yaml'):
    """
    Download stock data from Yahoo Finance
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str, optional): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format
        period (str, optional): Period to download (e.g., '1y', '5y', 'max')
        interval (str, optional): Data interval (e.g., '1m', '5m', '1h', '1d')
        config_path (str): Path to config file
        
    Returns:
        str: Path to the saved data file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create directory if it doesn't exist
    os.makedirs(config['paths']['raw_data_dir'], exist_ok=True)
    
    if period and interval == '1h':
        # 对高频数据限制周期，Yahoo只支持最近730天的数据
        if period.endswith('y') or period.endswith('Y'):
            print("警告: 高频数据仅支持最近730天的数据，自动将周期从 {} 调整为 '730d'".format(period))
            period = '730d'
    
    if period:
        print(f"Downloading data for {ticker} with period {period} and interval {interval}...")
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False)
        except Exception as e:
            print(f"Error downloading data: {str(e)}")
            return None
    else:
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if not start_date and not period:
            # Default to 10 years of data if neither start_date nor period is provided
            start_date = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')
        
        print(f"Downloading data for {ticker} from {start_date} to {end_date} with interval {interval}...")
        
        try:
            # Download data for the specified date range and interval
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        except Exception as e:
            print(f"Error downloading data: {str(e)}")
            return None
        
    if data.empty:
        print(f"No data found for ticker {ticker}")
        return None
    
    # Reset index to make Date a column
    data.reset_index(inplace=True)
    
    # Save data to CSV
    output_path = os.path.join(config['paths']['raw_data_dir'], f"{ticker}_hourly_data.csv")
    data.to_csv(output_path, index=False)
    
    print(f"Successfully downloaded {len(data)} rows of data for {ticker}")
    # 注意，根据不同模式，Datetime 列的名称可能不同
    date_col = 'Datetime' if 'Datetime' in data.columns else data.columns[0]
    print(f"Date range: {data[date_col].min()} to {data[date_col].max()}")
    print(f"Data saved to {output_path}")
    
    return output_path

def main():
    """
    Main function to download stock data from command line
    """
    parser = argparse.ArgumentParser(description='Download stock data from Yahoo Finance')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol')
    parser.add_argument('--start_date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--period', type=str, help='Period (e.g., 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    
    args = parser.parse_args()
    
    download_stock_data(
        args.ticker,
        args.start_date,
        args.end_date,
        args.period,
        args.config
    )

if __name__ == "__main__":
    main()

