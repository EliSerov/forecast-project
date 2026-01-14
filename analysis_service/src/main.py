import asyncio
import json
import base64
import logging
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io

import redis.asyncio as redis
from data_loader import DataLoader
from model_trainer import ModelTrainer
from forecaster import Forecaster
from strategist import InvestmentStrategist

# Configuration
REDIS_URL = os.getenv('REDIS_URL')

# Redis connection
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/analysis_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def process_analysis_task(task_data):
    try:
        ticker = task_data['ticker']
        amount = task_data['amount']
        
        logger.info(f"Processing analysis for {ticker}, amount {amount}")
        
        # 1. Load data
        data_loader = DataLoader(ticker)
        df = data_loader.load_data()
        
        if df is None or df.empty:
            raise ValueError(f"Не удалось загрузить данные для тикера {ticker}")
        
        # 2. Train models and select best
        trainer = ModelTrainer(df)
        best_model, best_rmse, model_name = trainer.train_and_select_best()
        
        # 3. Make forecast
        forecaster = Forecaster(best_model, df)
        forecast, current_price = forecaster.forecast(30)
        
        # 4. Generate recommendations
        strategist = InvestmentStrategist(forecast, amount)
        recommendations, profit_info = strategist.generate_recommendations()
        
        # 5. Create plot
        plot_image = create_plot(df, forecast, ticker)
        
        # 6. Prepare results
        result = {
            'status': 'success',
            'user_id': task_data['user_id'],
            'chat_id': task_data['chat_id'],
            'ticker': ticker,
            'amount': amount,
            'wait_message_id': task_data.get('wait_message_id'),
            'current_price': current_price,
            'predicted_price': forecast[-1],
            'price_change_percent': ((forecast[-1] - current_price) / current_price) * 100,
            'recommendations': recommendations,
            'profit_calculation': profit_info.get('transactions', []),
            'best_model': model_name,
            'rmse': best_rmse,
            'calculated_profit': profit_info.get('profit', 0),
            'plot_image': plot_image
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing task: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'user_id': task_data.get('user_id'),
            'chat_id': task_data.get('chat_id'),
            'wait_message_id': task_data.get('wait_message_id')
        }

def create_plot(historical_data, forecast, ticker):
    
    plt.figure(figsize=(12, 6))

    historical_dates = historical_data.index
    plt.plot(historical_dates, historical_data['Close'], label='Исторические данные', linewidth=2)

    forecast_dates = [historical_dates[-1] + timedelta(days=i) for i in range(1, len(forecast)+1)]
    plt.plot(forecast_dates, forecast, label='Прогноз', linewidth=2, linestyle='--')
    
    plt.title(f'Прогноз цен акций {ticker} на 30 дней')
    plt.xlabel('Дата')
    plt.ylabel('Цена ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    plt.close()
    
    return base64.b64encode(buffer.read()).decode('latin1')

async def worker():
    while True:
        try:
            # Get task from queue
            task_json = await redis_client.blpop('analysis_queue', timeout=0)
            if task_json:
                task_data = json.loads(task_json[1])
                
                # Process task
                result = await process_analysis_task(task_data)

                await redis_client.publish('results_channel', json.dumps(result))
                logger.info(f"Task completed for user {task_data.get('user_id')}")
                
        except Exception as e:
            logger.error(f"Worker error: {e}")
            await asyncio.sleep(1)

async def main():
    logger.info("Analysis service started")
    await worker()

if __name__ == '__main__':
    asyncio.run(main())
