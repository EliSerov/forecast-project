import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, ticker):
        self.ticker = ticker
        self.start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 years
        self.end_date = datetime.now().strftime('%Y-%m-%d')
    
    def load_data(self):
        # Load stock data from Yahoo Finance
        try:
            logger.info(f"Loading data for {self.ticker} from {self.start_date} to {self.end_date}")
            
            # Download data
            stock = yf.Ticker(self.ticker)
            df = stock.history(start=self.start_date, end=self.end_date)
            
            if df.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")

            df = df[['Close']].copy()
            df = df.dropna()
            
            # Ensure we have enough data
            if len(df) < 100:
                raise ValueError(f"Insufficient data points: {len(df)}")
            
            logger.info(f"Successfully loaded {len(df)} data points for {self.ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data for {self.ticker}: {e}")
            return None
    
    def prepare_features(self, df, lag_periods=30):
        # Create lagged features for ML models
        df_processed = df.copy()
        
        # Create lag features
        for i in range(1, lag_periods + 1):
            df_processed[f'lag_{i}'] = df_processed['Close'].shift(i)
        
        # Additional features
        df_processed['rolling_mean_7'] = df_processed['Close'].rolling(window=7).mean()
        df_processed['rolling_std_7'] = df_processed['Close'].rolling(window=7).std()
        df_processed['price_change'] = df_processed['Close'].pct_change()
        
        # Drop NaN values
        df_processed = df_processed.dropna()
        
        return df_processed
