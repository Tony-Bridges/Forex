"""
Trading Strategy Module
Contains strategy implementations for the MT5 Trading Bot
"""

import numpy as np
import pandas as pd
import logging
import config

# Set up logging
logger = logging.getLogger(__name__)

# Simple utility functions to calculate indicators (replacing TA-Lib)
def calculate_ma(data, period, price='close'):
    """Calculate moving average"""
    return data[price].rolling(window=period).mean()

def calculate_ema(data, period, price='close'):
    """Calculate exponential moving average"""
    return data[price].ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14, price='close'):
    """Calculate RSI"""
    delta = data[price].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(data, period=14):
    """Calculate Average True Range"""
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    return true_range.rolling(period).mean()

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9, price='close'):
    """Calculate MACD"""
    fast_ema = calculate_ema(data, fast_period, price)
    slow_ema = calculate_ema(data, slow_period, price)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    
    return macd_line, signal_line, macd_histogram

class MovingAverageRSIStrategy:
    """
    A trading strategy combining Moving Averages crossover with RSI confirmation
    """
    
    def __init__(self, symbol, timeframe, slow_ma_period=50, fast_ma_period=20, 
                 rsi_period=14, rsi_overbought=70, rsi_oversold=30):
        """
        Initialize the strategy parameters
        Args:
            symbol (str): Trading symbol
            timeframe (int): Trading timeframe
            slow_ma_period (int): Period for the slow moving average
            fast_ma_period (int): Period for the fast moving average
            rsi_period (int): Period for RSI calculation
            rsi_overbought (int): RSI level considered overbought
            rsi_oversold (int): RSI level considered oversold
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.slow_ma_period = slow_ma_period
        self.fast_ma_period = fast_ma_period
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        
        logger.info(f"Strategy initialized for {symbol} with parameters: "
                    f"Slow MA={slow_ma_period}, Fast MA={fast_ma_period}, "
                    f"RSI period={rsi_period}, RSI overbought={rsi_overbought}, "
                    f"RSI oversold={rsi_oversold}")
    
    def calculate_signal(self, data):
        """
        Calculate trading signal based on strategy rules
        Args:
            data (DataFrame): Market data with OHLC prices
        Returns:
            int: Signal (1 for buy, -1 for sell, 0 for no action)
        """
        if data is None or len(data) < max(self.slow_ma_period, self.rsi_period) + 10:
            logger.warning("Not enough data points for signal calculation")
            return 0
        
        # Calculate indicators
        data = self._calculate_indicators(data)
        
        # Get the latest values
        current = data.iloc[-1]
        previous = data.iloc[-2]
        
        # Check for buy signal
        if (previous['fast_ma'] <= previous['slow_ma'] and current['fast_ma'] > current['slow_ma'] and
            current['rsi'] < self.rsi_overbought and current['rsi'] > self.rsi_oversold):
            return 1
        
        # Check for sell signal
        elif (previous['fast_ma'] >= previous['slow_ma'] and current['fast_ma'] < current['slow_ma'] and
              current['rsi'] < self.rsi_overbought and current['rsi'] > self.rsi_oversold):
            return -1
        
        # No signal
        return 0
    
    def _calculate_indicators(self, data):
        """
        Calculate the technical indicators used in the strategy
        Args:
            data (DataFrame): Market data with OHLC prices
        Returns:
            DataFrame: Data with indicators added
        """
        data = data.copy()
        
        # Calculate moving averages
        data['slow_ma'] = calculate_ma(data, self.slow_ma_period)
        data['fast_ma'] = calculate_ma(data, self.fast_ma_period)
        
        # Calculate RSI
        data['rsi'] = calculate_rsi(data, self.rsi_period)
        
        # Calculate ATR for stop loss if needed
        data['atr'] = calculate_atr(data)
        
        return data
    
    def calculate_sl_tp(self, data, signal, entry_price):
        """
        Calculate stop loss and take profit levels
        Args:
            data (DataFrame): Market data with indicators
            signal (int): Trade direction (1 for buy, -1 for sell)
            entry_price (float): Entry price for the trade
        Returns:
            tuple: (stop_loss_price, take_profit_price)
        """
        # Ensure indicators are calculated
        data = self._calculate_indicators(data)
        
        # Get the latest values
        current = data.iloc[-1]
        
        # Simple default values for Replit environment without MT5
        pip_value = 0.0001
        point_value = 0.00001
        
        # Calculate stop loss based on ATR or fixed pips
        if 'atr' in current and not np.isnan(current['atr']):
            sl_distance = current['atr'] * 2.0  # ATR multiplier
        else:
            sl_distance = 50 * pip_value  # Fixed 50 pips
        
        # Calculate take profit based on risk-reward ratio or fixed pips
        risk_reward_ratio = 2.0
        tp_distance = sl_distance * risk_reward_ratio
        
        # Apply stop loss and take profit based on trade direction
        if signal > 0:  # Buy
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # Sell
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
        
        # Round to 5 decimal places (common for forex)
        stop_loss = round(stop_loss, 5)
        take_profit = round(take_profit, 5)
        
        return stop_loss, take_profit
    
    def update_position(self, position, data):
        """
        Update an existing position (trailing stop, etc.)
        Args:
            position (dict): Position information
            data (DataFrame): Current market data
        Returns:
            bool: True if position was updated, False otherwise
        """
        # Check if trailing stop is enabled
        position_type = position['type']  # 0 for buy, 1 for sell
        current_price = position['current_price']
        open_price = position['open_price']
        sl = position['sl']
        
        # Calculate new stop loss based on trailing distance
        new_sl = None
        
        # For buy positions (long)
        if position_type == 0 and current_price > open_price:
            # Calculate trailing stop (example: if price moved X pips, move SL by X pips)
            price_movement = current_price - open_price
            trailing_distance = price_movement * 0.5  # Example: trail by 50% of movement
            
            potential_sl = open_price + trailing_distance
            
            # Only update if new SL is higher than current SL
            if sl == 0 or potential_sl > sl:
                new_sl = potential_sl
        
        # For sell positions (short)
        elif position_type == 1 and current_price < open_price:
            # Calculate trailing stop
            price_movement = open_price - current_price
            trailing_distance = price_movement * 0.5
            
            potential_sl = open_price - trailing_distance
            
            # Only update if new SL is lower than current SL
            if sl == 0 or potential_sl < sl:
                new_sl = potential_sl
        
        # Update position if needed
        if new_sl is not None:
            # Round to 5 decimal places
            new_sl = round(new_sl, 5)
            
            # In a real implementation, we would call MT5 to modify the position
            logger.info(f"Would update position SL from {sl} to {new_sl}")
            return True
        
        return False

class MACDStrategy:
    """
    A trading strategy using MACD and support/resistance levels
    This is an additional strategy class that can be used as an alternative
    """
    
    def __init__(self, symbol, timeframe, fast_ema=12, slow_ema=26, signal_period=9):
        """
        Initialize the MACD strategy
        Args:
            symbol (str): Trading symbol
            timeframe (int): Trading timeframe
            fast_ema (int): Fast EMA period for MACD
            slow_ema (int): Slow EMA period for MACD
            signal_period (int): Signal line period for MACD
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.signal_period = signal_period
        
        logger.info(f"MACD Strategy initialized for {symbol} with parameters: "
                    f"Fast EMA={fast_ema}, Slow EMA={slow_ema}, Signal Period={signal_period}")
    
    def calculate_signal(self, data):
        """
        Calculate trading signal based on MACD crossovers
        Args:
            data (DataFrame): Market data with OHLC prices
        Returns:
            int: Signal (1 for buy, -1 for sell, 0 for no action)
        """
        if data is None or len(data) < self.slow_ema + self.signal_period + 10:
            logger.warning("Not enough data points for MACD calculation")
            return 0
        
        # Calculate MACD
        data = self._calculate_indicators(data)
        
        # Get the latest values
        current = data.iloc[-1]
        previous = data.iloc[-2]
        
        # Check for buy signal (MACD line crosses above Signal line)
        if previous['macd'] < previous['macd_signal'] and current['macd'] > current['macd_signal']:
            # Additional filter: check if histogram is increasing
            if current['macd_hist'] > previous['macd_hist']:
                return 1
        
        # Check for sell signal (MACD line crosses below Signal line)
        elif previous['macd'] > previous['macd_signal'] and current['macd'] < current['macd_signal']:
            # Additional filter: check if histogram is decreasing
            if current['macd_hist'] < previous['macd_hist']:
                return -1
        
        # No signal
        return 0
    
    def _calculate_indicators(self, data):
        """
        Calculate MACD and other indicators
        Args:
            data (DataFrame): Market data with OHLC prices
        Returns:
            DataFrame: Data with indicators added
        """
        data = data.copy()
        
        # Calculate MACD
        macd, macd_signal, macd_hist = calculate_macd(
            data, 
            fast_period=self.fast_ema, 
            slow_period=self.slow_ema, 
            signal_period=self.signal_period
        )
        
        data['macd'] = macd
        data['macd_signal'] = macd_signal
        data['macd_hist'] = macd_hist
        
        # Calculate ATR for stop loss
        data['atr'] = calculate_atr(data, period=14)
        
        return data
    
    def calculate_sl_tp(self, data, signal, entry_price):
        """
        Calculate stop loss and take profit based on ATR
        Args:
            data (DataFrame): Market data with indicators
            signal (int): Trade direction (1 for buy, -1 for sell)
            entry_price (float): Entry price for the trade
        Returns:
            tuple: (stop_loss_price, take_profit_price)
        """
        # Calculate indicators if not already calculated
        data = self._calculate_indicators(data)
        
        # Get ATR value
        atr = data.iloc[-1]['atr']
        
        # Set multipliers
        sl_multiplier = 2.0  # Stop loss at 2x ATR
        tp_multiplier = 3.0  # Take profit at 3x ATR (1.5:1 risk-reward ratio)
        
        # Calculate stop loss and take profit distances
        sl_distance = atr * sl_multiplier
        tp_distance = atr * tp_multiplier
        
        # Calculate prices based on direction
        if signal > 0:  # Buy
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # Sell
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
        
        # Round to 5 decimal places
        stop_loss = round(stop_loss, 5)
        take_profit = round(take_profit, 5)
        
        return stop_loss, take_profit
    
    def update_position(self, position, data):
        """
        Trailing stop logic for MACD strategy
        Args:
            position (dict): Position information
            data (DataFrame): Current market data
        Returns:
            bool: True if position was updated, False otherwise
        """
        # Similar to the implementation in MovingAverageRSIStrategy
        # But customized for MACD strategy
        return False