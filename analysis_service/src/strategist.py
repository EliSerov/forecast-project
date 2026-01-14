import numpy as np
import logging

logger = logging.getLogger(__name__)

class InvestmentStrategist:
    def __init__(self, forecast_prices, investment_amount):
        self.forecast_prices = forecast_prices
        self.investment_amount = investment_amount
        self.days = len(forecast_prices)
    
    def find_local_extrema(self, window=3):

        prices = np.array(self.forecast_prices)
        buy_points = []
        sell_points = []
        
        for i in range(window, self.days - window):
            # Check for local minimum (buy point)
            if (prices[i] <= prices[i-window:i]).all() and \
               (prices[i] <= prices[i+1:i+window+1]).all():
                buy_points.append((i, prices[i]))
            
            # Check for local maximum (sell point)
            if (prices[i] >= prices[i-window:i]).all() and \
               (prices[i] >= prices[i+1:i+window+1]).all():
                sell_points.append((i, prices[i]))
        
        return buy_points, sell_points
    
    def generate_trading_signals(self, buy_points, sell_points):

        signals = []
        current_action = 'hold'
        last_buy_price = None
        last_buy_day = None
        
        all_points = sorted(buy_points + sell_points, key=lambda x: x[0])
        
        for day, price in all_points:
            if (day, price) in buy_points and current_action != 'buy':
                signals.append(('buy', day, price))
                current_action = 'buy'
                last_buy_price = price
                last_buy_day = day
            
            elif (day, price) in sell_points and current_action == 'buy':
                signals.append(('sell', day, price))
                current_action = 'sell'
                last_buy_price = None
                last_buy_day = None
        
        return signals
    
    def calculate_profit(self, trading_signals):

        if not trading_signals:
            return 0, "Нет торговых сигналов для расчета прибыли"
        
        cash = self.investment_amount
        shares = 0
        transactions = []
        
        for action, day, price in trading_signals:
            if action == 'buy' and cash > 0:
                shares_bought = cash / price
                shares += shares_bought
                cash = 0
                transactions.append(f"День {day}: Покупка по ${price:.2f}, куплено {shares_bought:.2f} акций")
            
            elif action == 'sell' and shares > 0:
                cash = shares * price
                shares = 0
                transactions.append(f"День {day}: Продажа по ${price:.2f}, выручено ${cash:.2f}")
        
        # Final valuation
        if shares > 0:
            final_price = self.forecast_prices[-1]
            cash = shares * final_price
            transactions.append(f"День {self.days}: Финалная продажа по ${final_price:.2f}, выручено ${cash:.2f}")
        
        profit = cash - self.investment_amount
        profit_percent = (profit / self.investment_amount) * 100
        
        profit_info = {
            'initial_investment': self.investment_amount,
            'final_value': cash,
            'profit': profit,
            'profit_percent': profit_percent,
            'transactions': transactions,
            'total_profit': profit
        }
        
        return profit, profit_info
    
    def generate_recommendations(self):

        try:
            # Find trading points
            buy_points, sell_points = self.find_local_extrema()
            
            if not buy_points and not sell_points:
                return self._generate_fallback_recommendations()

            signals = self.generate_trading_signals(buy_points, sell_points)
            
            if not signals:
                return self._generate_fallback_recommendations()
            
            # Calculate profit
            profit, profit_info = self.calculate_profit(signals)

            recommendations = self._format_recommendations(signals, buy_points, sell_points)
            profit_text = self._format_profit_info(profit_info)
            
            return recommendations, profit_info
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return self._generate_fallback_recommendations()
    
    def _format_recommendations(self, signals, buy_points, sell_points):

        if not signals:
            return "Не найдено четких торговых сигналов. Рекомендуется удержание позиции."
        
        recommendations = ["**Торговые рекомендации:**"]
        
        if buy_points:
            buy_days = [f"день {day} (${price:.2f})" for day, price in buy_points[:3]]  # Top 3 buys
            recommendations.append(f"📈 **Покупать:** {', '.join(buy_days)}")
        
        if sell_points:
            sell_days = [f"день {day} (${price:.2f})" for day, price in sell_points[:3]]  # Top 3 sells
            recommendations.append(f"📉 **Продавать:** {', '.join(sell_days)}")
        
        if signals:
            recommendations.append("\n**Рекомендуемая стратегия:**")
            for i, (action, day, price) in enumerate(signals[:5], 1):  # Show first 5 signals
                recommendations.append(f"{i}. День {day}: {action.upper()} по ${price:.2f}")
        
        return "\n".join(recommendations)
    
    def _format_profit_info(self, profit_info):

        profit_text = [
            "**Расчет прибыли:**",
            f"Начальная инвестиция: ${profit_info['initial_investment']:.2f}",
            f"Финальная стоимость: ${profit_info['final_value']:.2f}",
            f"Прибыль: ${profit_info['profit']:.2f} ({profit_info['profit_percent']:+.2f}%)"
        ]
        
        if profit_info['transactions']:
            profit_text.append("\n**Транзакции:**")
            profit_text.extend(profit_info['transactions'])
        
        return "\n".join(profit_text)
    
    def _generate_fallback_recommendations(self):

        # Simple buy-and-hold strategy
        initial_price = self.forecast_prices[0]
        final_price = self.forecast_prices[-1]
        
        shares = self.investment_amount / initial_price
        final_value = shares * final_price
        profit = final_value - self.investment_amount
        profit_percent = (profit / self.investment_amount) * 100
        
        profit_info = {
            'initial_investment': self.investment_amount,
            'final_value': final_value,
            'profit': profit,
            'profit_percent': profit_percent,
            'transactions': [
                f"День 0: Покупка по ${initial_price:.2f}",
                f"День {self.days}: Продажа по ${final_price:.2f}"
            ],
            'total_profit': profit
        }
        
        recommendations = """**Рекомендации:**
Не найдено четких точек входа/выхода. Рекомендуется стратегия "купи и держи".

**Альтернативная стратегия:**
- День 0: Покупка по текущей цене
- День 30: Продажа по прогнозируемой цене"""

        profit_text = self._format_profit_info(profit_info)
        
        return recommendations, profit_info
