"""
MT5 Forex Trading Bot

This script runs a trading bot for MetaTrader 5 with AI-enhanced strategies.
It can be executed independently from the web dashboard.
"""

import os
import time
import logging
import json
import sys
from datetime import datetime
import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("MT5TradingBot")

# Import local modules
from strategy import MovingAverageRSIStrategy, MACDStrategy
from enhanced_strategy import create_strategy, AIEnhancedMAStrategy, AIEnhancedMACDStrategy
from ai_strategy_manager import ai_strategy_manager

# Mock MT5 connector functions for demo purposes
# In a real implementation, these would connect to the MetaTrader 5 API
def get_historical_data(symbol, timeframe, n_bars=1000):
    """
    Get historical price data for a symbol
    
    Args:
        symbol (str): Trading symbol (e.g., 'EURUSD')
        timeframe (str): Trading timeframe (e.g., 'H1', 'D1')
        n_bars (int): Number of bars to retrieve
        
    Returns:
        DataFrame: OHLCV data
    """
    logger.info(f"Getting historical data for {symbol} on {timeframe} timeframe")
    
    # Generate synthetic data for demonstration
    # In a real implementation, this would retrieve data from MT5
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='H')
    
    # Generate price data with some trend and noise
    base_price = 1.2000 if symbol.startswith('EUR') else 1.5000
    close = np.random.normal(0, 0.0002, n_bars).cumsum() + base_price
    
    # Create some realistic price movement
    high = close + np.random.normal(0, 0.0003, n_bars).clip(0, 0.005)
    low = close - np.random.normal(0, 0.0003, n_bars).clip(0, 0.005)
    open_price = close.copy()
    open_price[1:] = close[:-1]
    open_price[0] = open_price[1]
    
    volume = np.random.randint(10, 200, n_bars)
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    return data

def place_order(symbol, order_type, volume, price=0, sl=0, tp=0, comment=""):
    """
    Place a new order
    
    Args:
        symbol (str): Trading symbol
        order_type (str): 'BUY' or 'SELL'
        volume (float): Trade volume (lot size)
        price (float): Order price (for pending orders)
        sl (float): Stop loss price
        tp (float): Take profit price
        comment (str): Order comment
        
    Returns:
        dict: Order result info
    """
    logger.info(f"Placing {order_type} order for {symbol}, volume: {volume}, SL: {sl}, TP: {tp}")
    
    # In a real implementation, this would send the order to MT5
    return {
        'ticket': 123456789,
        'symbol': symbol,
        'type': order_type,
        'volume': volume,
        'price': price if price > 0 else (
            get_current_price(symbol) + 0.0001 if order_type == 'BUY' else 
            get_current_price(symbol) - 0.0001
        ),
        'sl': sl,
        'tp': tp,
        'comment': comment,
        'time': datetime.now().isoformat()
    }

def get_open_positions():
    """
    Get all open positions
    
    Returns:
        list: List of open position dictionaries
    """
    logger.info("Getting open positions")
    
    # In a real implementation, this would retrieve positions from MT5
    return []

def get_current_price(symbol):
    """
    Get current bid and ask prices for a symbol
    
    Args:
        symbol (str): Trading symbol
        
    Returns:
        dict: Bid and ask prices
    """
    # In a real implementation, this would get real-time prices from MT5
    base_price = 1.2000 if symbol.startswith('EUR') else 1.5000
    bid = base_price + np.random.normal(0, 0.0001)
    ask = bid + 0.0002
    
    return {'bid': bid, 'ask': ask, 'symbol': symbol}

def get_account_info():
    """
    Get account information
    
    Returns:
        dict: Account information including balance, equity, etc.
    """
    logger.info("Getting account information")
    
    # In a real implementation, this would retrieve account info from MT5
    return {
        'balance': 10000.00,
        'equity': 10050.00,
        'margin': 0.00,
        'free_margin': 10050.00,
        'margin_level': 0.00,
        'leverage': 100,
        'currency': 'USD'
    }

class TradingBot:
    """
    Main trading bot class
    """
    
    def __init__(self, config=None):
        """
        Initialize the trading bot
        
        Args:
            config (dict, optional): Bot configuration
        """
        self.config = config or self._load_default_config()
        self.strategies = {}
        self.running = False
        self.last_check_time = {}
        self.initialize_strategies()
        
        logger.info("Trading bot initialized with configuration:")
        logger.info(json.dumps(self.config, indent=2))
    
    def _load_default_config(self):
        """
        Load default configuration if none is provided
        
        Returns:
            dict: Default configuration
        """
        return {
            'strategies': [
                {
                    'name': 'MA-RSI EURUSD H1',
                    'symbol': 'EURUSD',
                    'timeframe': 'H1',
                    'type': 'MA_RSI',
                    'params': {
                        'slow_ma_period': 50,
                        'fast_ma_period': 20,
                        'rsi_period': 14,
                        'rsi_overbought': 70,
                        'rsi_oversold': 30
                    },
                    'risk': {
                        'risk_per_trade': 1.0,
                        'max_risk_total': 5.0,
                        'use_atr_for_sl': True,
                        'atr_period': 14,
                        'atr_multiplier': 2.0,
                        'fixed_sl_pips': 50,
                        'fixed_tp_pips': 100,
                        'risk_reward_ratio': 2.0
                    },
                    'use_ai': True,
                    'check_interval': 3600  # Check every hour (in seconds)
                },
                {
                    'name': 'MACD GBPUSD H4',
                    'symbol': 'GBPUSD',
                    'timeframe': 'H4',
                    'type': 'MACD',
                    'params': {
                        'macd_fast_ema': 12,
                        'macd_slow_ema': 26,
                        'macd_signal_period': 9
                    },
                    'risk': {
                        'risk_per_trade': 1.0,
                        'max_risk_total': 5.0,
                        'use_atr_for_sl': True,
                        'atr_period': 14,
                        'atr_multiplier': 2.0,
                        'fixed_sl_pips': 50,
                        'fixed_tp_pips': 100,
                        'risk_reward_ratio': 2.0
                    },
                    'use_ai': True,
                    'check_interval': 14400  # Check every 4 hours (in seconds)
                }
            ],
            'check_positions_interval': 300,  # Check open positions every 5 minutes
            'enable_notifications': True,
            'data_fetch_depth': 1000,  # Number of candles to fetch for analysis
            'enable_trading': True,  # Set to False for paper trading
            'log_level': 'INFO'
        }
    
    def initialize_strategies(self):
        """
        Initialize strategy instances
        """
        logger.info("Initializing trading strategies")
        
        for strategy_config in self.config['strategies']:
            name = strategy_config['name']
            symbol = strategy_config['symbol']
            timeframe = strategy_config['timeframe']
            strategy_type = strategy_config['type']
            params = strategy_config['params']
            use_ai = strategy_config.get('use_ai', True)
            
            try:
                # Create strategy using factory function
                strategy_instance = create_strategy(
                    strategy_type=strategy_type,
                    symbol=symbol,
                    timeframe=timeframe,
                    params=params,
                    use_ai=use_ai
                )
                
                self.strategies[name] = {
                    'instance': strategy_instance,
                    'config': strategy_config,
                    'last_signal': 0,
                    'last_check': datetime.now()
                }
                
                logger.info(f"Strategy '{name}' initialized successfully")
                self.last_check_time[name] = 0  # Initialize last check time
                
            except Exception as e:
                logger.error(f"Error initializing strategy '{name}': {str(e)}")
    
    def check_and_trade(self, strategy_name):
        """
        Check for trading signals and execute trades
        
        Args:
            strategy_name (str): Name of the strategy to check
            
        Returns:
            dict: Trade result info or None
        """
        if strategy_name not in self.strategies:
            logger.error(f"Strategy '{strategy_name}' not found")
            return None
        
        strategy_data = self.strategies[strategy_name]
        strategy = strategy_data['instance']
        config = strategy_data['config']
        
        symbol = config['symbol']
        timeframe = config['timeframe']
        
        try:
            # Get historical data
            data = get_historical_data(symbol, timeframe, self.config['data_fetch_depth'])
            
            # Calculate trading signal
            signal = strategy.calculate_signal(data)
            
            # Log signal
            signal_text = "BUY" if signal > 0 else "SELL" if signal < 0 else "HOLD"
            logger.info(f"Strategy '{strategy_name}' signal: {signal_text}")
            
            # Update last signal
            strategy_data['last_signal'] = signal
            strategy_data['last_check'] = datetime.now()
            
            # Execute trade if signal is not HOLD and trading is enabled
            if signal != 0 and self.config['enable_trading']:
                # Check account info
                account_info = get_account_info()
                
                # Calculate position size based on risk
                balance = account_info['balance']
                risk_percent = config['risk']['risk_per_trade']
                risk_amount = balance * (risk_percent / 100)
                
                # Get current price
                price_info = get_current_price(symbol)
                entry_price = price_info['ask'] if signal > 0 else price_info['bid']
                
                # Calculate stop loss and take profit
                sl, tp = strategy.calculate_sl_tp(data, signal, entry_price)
                
                # Calculate position size (simplified)
                # In a real implementation, this would use pip value and leverage
                pip_value = 10  # Simplified assumption: 1 pip = $10 per standard lot
                sl_pips = abs(entry_price - sl) * 10000  # Convert to pips
                position_size = risk_amount / (sl_pips * pip_value)
                position_size = max(0.01, min(position_size, 10.0))  # Limit size
                
                # Place order
                order_type = "BUY" if signal > 0 else "SELL"
                order_result = place_order(
                    symbol=symbol,
                    order_type=order_type,
                    volume=position_size,
                    price=0,  # Market order
                    sl=sl,
                    tp=tp,
                    comment=f"Strategy: {strategy_name}"
                )
                
                logger.info(f"Order placed: {order_result}")
                return order_result
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking strategy '{strategy_name}': {str(e)}")
            return None
    
    def check_positions(self):
        """
        Check and update open positions
        """
        try:
            # Get open positions
            positions = get_open_positions()
            
            for position in positions:
                symbol = position['symbol']
                position_type = position['type']
                
                # Find matching strategy
                matching_strategy = None
                for name, strategy_data in self.strategies.items():
                    if strategy_data['config']['symbol'] == symbol:
                        matching_strategy = strategy_data['instance']
                        break
                
                if matching_strategy:
                    # Get current market data
                    data = get_historical_data(symbol, strategy_data['config']['timeframe'], 100)
                    
                    # Update position (e.g., trailing stop)
                    updated = matching_strategy.update_position(position, data)
                    if updated:
                        logger.info(f"Position {position['ticket']} updated")
        
        except Exception as e:
            logger.error(f"Error checking positions: {str(e)}")
    
    def run(self):
        """
        Run the trading bot
        """
        logger.info("Starting trading bot")
        self.running = True
        
        try:
            while self.running:
                current_time = time.time()
                
                # Check strategies
                for name, strategy_data in self.strategies.items():
                    check_interval = strategy_data['config']['check_interval']
                    if current_time - self.last_check_time.get(name, 0) >= check_interval:
                        self.check_and_trade(name)
                        self.last_check_time[name] = current_time
                
                # Check open positions
                if current_time - self.last_check_time.get('positions', 0) >= self.config['check_positions_interval']:
                    self.check_positions()
                    self.last_check_time['positions'] = current_time
                
                # Sleep to prevent high CPU usage
                time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
        except Exception as e:
            logger.error(f"Error in trading bot: {str(e)}")
        finally:
            self.running = False
            logger.info("Trading bot stopped")
    
    def stop(self):
        """
        Stop the trading bot
        """
        logger.info("Stopping trading bot...")
        self.running = False

# Main execution
if __name__ == "__main__":
    # Train AI models if needed
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        logger.info("Training AI models...")
        
        # Get historical data for training
        eurusd_data = get_historical_data("EURUSD", "H1", 5000)
        gbpusd_data = get_historical_data("GBPUSD", "H4", 5000)
        
        # Create strategy instances for training
        ma_rsi_strategy = AIEnhancedMAStrategy("EURUSD", "H1", use_ai=False)
        macd_strategy = AIEnhancedMACDStrategy("GBPUSD", "H4", use_ai=False)
        
        # Train models
        ma_rsi_results = ma_rsi_strategy.train_ai_models(eurusd_data)
        macd_results = macd_strategy.train_ai_models(gbpusd_data)
        
        logger.info("Training completed")
        logger.info(f"MA-RSI training results: {json.dumps(ma_rsi_results, indent=2)}")
        logger.info(f"MACD training results: {json.dumps(macd_results, indent=2)}")
        
        sys.exit(0)
    
    # Create and run trading bot
    bot = TradingBot()
    bot.run()