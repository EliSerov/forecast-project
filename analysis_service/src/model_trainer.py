import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, df, test_size=0.2):
        self.df = df
        self.test_size = test_size
        self.models = {}
        self.results = {}
        
    def prepare_ml_data(self):
        # Prepare data for models
        df_processed = self.df.copy()
        
        # Create lagged features (30 days)
        for i in range(1, 31):
            df_processed[f'lag_{i}'] = df_processed['Close'].shift(i)
        
        # Target variable (next day price)
        df_processed['target'] = df_processed['Close'].shift(-1)
        df_processed = df_processed.dropna()
        
        # Split into features and target
        X = df_processed.drop(['target', 'Close'], axis=1)
        y = df_processed['target']
        
        # Time-based split
        split_idx = int(len(X) * (1 - self.test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self):
        # Random forest
        try:
            X_train, X_test, y_train, y_test = self.prepare_ml_data()
            
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = mean_absolute_percentage_error(y_test, y_pred)
            
            return model, rmse, mape
            
        except Exception as e:
            logger.error(f"Error training Random Forest: {e}")
            return None, float('inf'), float('inf')
    
    def train_ridge_regression(self):
        # Regression
        try:
            X_train, X_test, y_train, y_test = self.prepare_ml_data()
            
            model = Ridge(alpha=1.0, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = mean_absolute_percentage_error(y_test, y_pred)
            
            return model, rmse, mape
            
        except Exception as e:
            logger.error(f"Error training Ridge Regression: {e}")
            return None, float('inf'), float('inf')
    
    def train_arima(self):
        # ARIMA
        try:
            # Use only Close prices for ARIMA
            data = self.df['Close'].values
            
            # Time-based split
            split_idx = int(len(data) * (1 - self.test_size))
            train_data, test_data = data[:split_idx], data[split_idx:]
            
            model = ARIMA(train_data, order=(2, 1, 2))
            model_fit = model.fit()
            
            # Forecast on test set
            forecast = model_fit.forecast(steps=len(test_data))
            
            rmse = np.sqrt(mean_squared_error(test_data, forecast))
            mape = mean_absolute_percentage_error(test_data, forecast)
            
            return model_fit, rmse, mape
            
        except Exception as e:
            logger.error(f"Error training ARIMA: {e}")
            return None, float('inf'), float('inf')
    
    
    def train_simple_lstm(self):
        # LSTM model (simplified)
        try:
            # For simplicity, using a linear model as LSTM placeholder
            from sklearn.linear_model import LinearRegression
            
            X_train, X_test, y_train, y_test = self.prepare_ml_data()
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = mean_absolute_percentage_error(y_test, y_pred)
            
            return model, rmse, mape, 'Linear Regression (LSTM placeholder)'
            
        except Exception as e:
            logger.error(f"Error training LSTM placeholder: {e}")
            return None, float('inf'), float('inf'), 'Failed'
    
    def train_and_select_best(self):
        # Train all models
        logger.info("Training multiple models...")
        
        models_to_train = [
            ('Random Forest', self.train_random_forest),
            ('Ridge Regression', self.train_ridge_regression),
            ('ARIMA', self.train_arima),
            ('LSTM', self.train_simple_lstm)
        ]
        
        best_model = None
        best_rmse = float('inf')
        best_model_name = ''
        
        for model_name, train_func in models_to_train:
            try:
                model, rmse, mape = train_func()
                
                if model is not None and rmse < best_rmse:
                    best_model = model
                    best_rmse = rmse
                    best_model_name = model_name
                
                logger.info(f"{model_name}: RMSE={rmse:.4f}, MAPE={mape:.4f}")
                self.results[model_name] = {'rmse': rmse, 'mape': mape}
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        if best_model is None:
            # Fallback to simplest model
            logger.warning("All models failed, using last value as forecast")
            best_model = None
            best_rmse = float('inf')
            best_model_name = 'Persistence'
        
        logger.info(f"Best model: {best_model_name} with RMSE: {best_rmse:.4f}")
        return best_model, best_rmse, best_model_name
