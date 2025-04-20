import pandas as pd

def preprocess_data(config_path):
    # Load configuration
    config = load_config(config_path)
    
    # Load raw data
    raw_data_path = os.path.join(config['paths']['raw_data_dir'], config['data']['filename'])
    df = pd.read_csv(raw_data_path)
    
    # Convert date column to datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    elif 'date' in df.columns:
        df['Date'] = pd.to_datetime(df['date'])  # 将小写的 'date' 统一命名为 'Date'
        df.drop(columns=['date'], inplace=True)
    else:
        raise KeyError("No date column found in data.")
    
    # Sort data by date
    df.sort_values('Date', inplace=True)
    
    # Save processed data
    processed_data_path = os.path.join(config['paths']['processed_data_dir'], config['data']['processed_filename'])
    df.to_csv(processed_data_path, index=False)