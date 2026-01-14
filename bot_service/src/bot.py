import asyncio
import json
import logging
import os
import base64
from datetime import datetime

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart, Command
import redis.asyncio as redis

BOT_TOKEN = os.getenv('BOT_TOKEN')
REDIS_URL = os.getenv('REDIS_URL')

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

redis_client = redis.from_url(REDIS_URL, decode_responses=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bot_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dp.message(CommandStart())
async def send_welcome(message: types.Message):
    welcome_text = """
    Добро пожаловать в бот для анализа акций!

    Чтобы получить прогноз, отправьте сообщение в формате:
    `<тикер> <сумма>`

    Например:
    `AAPL 1000` - анализ акций Apple с условной инвестицией $1000

    Доступные команды:
    /start - показать приветствие
    /help - помощь 
    """
    await message.answer(welcome_text)

@dp.message(Command("help"))
async def send_help(message: types.Message):
    help_text = """
    **Как пользоваться ботом:**

    **Основная команда:**
    `<тикер> <сумма>` - получить прогноз для акции
    
    **Примеры:**
    `AAPL 1000` - анализ Apple с инвестицией $1000
    `GOOGL 5000` - анализ Google с инвестицией $5000

    **Результат:**
    - График прогноза цены на 30 дней
    - Расчет потенциальной прибыли
    - Рекомендации по инвестициям
    - Использованная модель ML

    **Поддерживаемые тикеры:**
    Любые тикеры, доступные через Yahoo Finance (AAPL, GOOGL, MSFT, TSLA, и т.д.)

    """
    await message.answer(help_text, parse_mode='Markdown')

def log_user_request(user_id, ticker, amount, best_model, metric_value, profit):
    
    log_entry = {
        'user_id': user_id,
        'timestamp': datetime.now().isoformat(),
        'ticker': ticker,
        'amount': amount,
        'best_model': best_model,
        'metric_value': metric_value,
        'calculated_profit': profit
    }
    
    with open('logs/user_requests.log', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

@dp.message(F.text)
async def handle_stock_request(message: types.Message):
    try:
        
        parts = message.text.strip().split()
        if len(parts) != 2:
            await message.answer("Неверный формат. Используйте: <тикер> <сумма>\nНапример: AAPL 1000")
            return

        ticker, amount = parts[0].upper(), float(parts[1])

        wait_msg = await message.answer("Загружаем данные и обучаем модели... Это может занять несколько минут.")

        # Create task
        task_data = {
            'user_id': message.from_user.id,
            'chat_id': message.chat.id,
            'ticker': ticker,
            'amount': amount,
            'wait_message_id': wait_msg.message_id
        }
        
        await redis_client.rpush('analysis_queue', json.dumps(task_data))
        logger.info(f"Task queued for user {message.from_user.id}, ticker {ticker}")

    except ValueError:
        await message.answer("Сумма должна быть числом!\nПример: AAPL 1000")
    except Exception as e:
        logger.error(f"Error handling request: {e}")
        await message.answer("Произошла ошибка. Попробуйте позже.")

async def result_listener():
    
    pubsub = redis_client.pubsub()
    await pubsub.subscribe('results_channel')
    
    async for message in pubsub.listen():
        if message['type'] == 'message':
            try:
                result = json.loads(message['data'])
                chat_id = result['chat_id']
                wait_message_id = result.get('wait_message_id')
                
                
                if wait_message_id:
                    try:
                        await bot.delete_message(chat_id, wait_message_id)
                    except:
                        pass  
                
                
                if result['status'] == 'success':
                    
                    if 'plot_image' in result and result['plot_image']:
                        try:
                            
                            image_data = base64.b64decode(result['plot_image'])
                            plot_file = types.BufferedInputFile(image_data, filename='forecast_plot.png')
                            await bot.send_photo(chat_id, plot_file)
                        except Exception as e:
                            logger.error(f"Error sending plot: {e}")
                            
                            await bot.send_message(chat_id, "График временно недоступен")
                    
                    # Send text results
                    response_text = f"""
**Результаты анализа для {result['ticker']}**

**Прогноз на 30 дней:**
- Текущая цена: ${result['current_price']:.2f}
- Прогнозируемая цена через 30 дней: ${result['predicted_price']:.2f}
- Изменение: {result['price_change_percent']:+.2f}%

**Рекомендации:**
{result['recommendations']}

**Расчет прибыли для инвестиции ${result['amount']:.2f}:**
{result['profit_calculation']}

**Использованная модель:** {result['best_model']}
**Метрика качества (RMSE):** {result['rmse']:.4f}
                    """
                    await bot.send_message(chat_id, response_text, parse_mode='Markdown')
                    
                    # Log the request
                    log_user_request(
                        result['user_id'], result['ticker'], result['amount'],
                        result['best_model'], result['rmse'], result.get('calculated_profit', 0)
                    )
                    
                else:
                    error_msg = result.get('error', 'Неизвестная ошибка')
                    await bot.send_message(chat_id, f"Ошибка анализа: {error_msg}")
                    
            except Exception as e:
                logger.error(f"Error processing result: {e}")
                # Try to send generic error message
                try:
                    if 'chat_id' in result:
                        await bot.send_message(result['chat_id'], "Произошла ошибка при обработке результатов")
                except:
                    pass

async def main():

    os.makedirs('logs', exist_ok=True)

    asyncio.create_task(result_listener())
    
    # Start bot
    logger.info("Bot service started")
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
