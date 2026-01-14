import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class Forecaster:
    def __init__(self, model, historical_data):
        self.model = model
        self.historical_data = historical_data
        self.current_price = historical_data['Close'].iloc[-1]
    
    def forecast(self, days=30):
        # Generate forecast
        try:
            if self.model is None:
                # Persistence model (use last value)
                return [self.current_price] * days, self.current_price
            
            model_type = type(self.model).__name__.lower()
            
            if 'arima' in model_type or 'holt' in model_type:
                # Statsmodels models
                forecast = self.model.forecast(steps=days)
                return forecast.tolist(), self.current_price
            
            elif 'forest' in model_type or 'ridge' in model_type or 'regression' in model_type:
                # Scikit-learn models - use recursive forecasting
                return self._recursive_forecast_ml(days), self.current_price
            
            else:
                return [self.current_price] * days, self.current_price
                
        except Exception as e:
            logger.error(f"Error in forecasting: {e}")
            return [self.current_price] * days, self.current_price
    
    def _recursive_forecast_ml(self, days):
        # Recursive forecasting
        try:
            forecast = []
            current_features = self._get_last_features()
            
            for i in range(days):
                # Make prediction for next day
                next_price = self.model.predict([current_features])[0]
                forecast.append(next_price)
                
                # Update features for next prediction
                current_features = self._update_features(current_features, next_price)
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error in recursive forecasting: {e}")
            return [self.current_price] * days
    
    def _get_last_features(self):
        df_processed = self.historical_data.copy()
        
        # Create lagged features same as in training
        for i in range(1, 31):
            df_processed[f'lag_{i}'] = df_processed['Close'].shift(i)
        
        # Get the last row with all features
        last_row = df_processed.iloc[-1:].drop(['Close'], axis=1)
        
        return last_row.values[0]
    
    def _update_features(self, current_features, new_price):
        # Shift features and add new price
        new_features = current_features.copy()
        
        # Shift all lag features
        for i in range(29, 0, -1):
            new_features[i] = new_features[i-1]
        
        # Add new price as lag_1
        new_features[0] = new_price
        
        return new_features
