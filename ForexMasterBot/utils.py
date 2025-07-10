"""
Utility functions for the MT5 Trading Bot
"""

import time
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def format_price(price, digits=5):
    """
    Format price with appropriate digits
    
    Args:
        price (float): Price to format
        digits (int): Number of digits to use
        
    Returns:
        str: Formatted price
    """
    return f"{price:.{digits}f}"

def calculate_pips_value(symbol, point_value):
    """
    Calculate the value of 1 pip for a symbol
    
    Args:
        symbol (str): Trading symbol
        point_value (float): Symbol point value
        
    Returns:
        float: Pip value
    """
    # Common Forex pairs have 10000 points per pip
    if symbol.endswith('JPY'):
        return point_value * 100  # JPY pairs have 2 decimal places, others have 4
    else:
        return point_value * 10000

def calculate_position_size(balance, risk_percent, sl_pips, per_pip_value):
    """
    Calculate position size based on account balance and risk settings
    
    Args:
        balance (float): Account balance
        risk_percent (float): Risk percentage (0-100)
        sl_pips (float): Stop loss in pips
        per_pip_value (float): Value of 1 pip for the symbol
        
    Returns:
        float: Position size in lots
    """
    # Calculate risk amount
    risk_amount = balance * (risk_percent / 100)
    
    # Calculate position size
    if sl_pips > 0 and per_pip_value > 0:
        position_size = risk_amount / (sl_pips * per_pip_value)
    else:
        position_size = 0.01  # Minimum position size
    
    # Round to 2 decimal places
    return round(position_size, 2)

def timeframe_to_seconds(timeframe):
    """
    Convert timeframe string to seconds
    
    Args:
        timeframe (str): Timeframe string (e.g., 'M5', 'H1')
        
    Returns:
        int: Timeframe in seconds
    """
    if timeframe.startswith('M'):
        return int(timeframe[1:]) * 60
    elif timeframe.startswith('H'):
        return int(timeframe[1:]) * 3600
    elif timeframe.startswith('D'):
        return int(timeframe[1:]) * 86400
    else:
        return 3600  # Default to 1 hour

def human_readable_time(seconds):
    """
    Convert seconds to human readable time string
    
    Args:
        seconds (int): Time in seconds
        
    Returns:
        str: Human readable time
    """
    if seconds < 60:
        return f"{seconds} seconds"
    elif seconds < 3600:
        return f"{seconds // 60} minutes"
    elif seconds < 86400:
        return f"{seconds // 3600} hours"
    else:
        return f"{seconds // 86400} days"

def save_json(data, filepath):
    """
    Save data to JSON file
    
    Args:
        data (dict): Data to save
        filepath (str): Path to save file
        
    Returns:
        bool: Success status
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file: {str(e)}")
        return False

def load_json(filepath):
    """
    Load data from JSON file
    
    Args:
        filepath (str): Path to file
        
    Returns:
        dict: Loaded data or None if error
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file: {str(e)}")
        return None

def calculate_drawdown(equity_curve):
    """
    Calculate maximum drawdown from equity curve
    
    Args:
        equity_curve (list): List of equity values
        
    Returns:
        tuple: (max_drawdown_amount, max_drawdown_percent)
    """
    max_drawdown = 0
    max_drawdown_pct = 0
    peak = equity_curve[0]
    
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        
        drawdown = peak - equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            max_drawdown_pct = (drawdown / peak) * 100
    
    return max_drawdown, max_drawdown_pct

def calculate_sharpe_ratio(returns, risk_free_rate=0):
    """
    Calculate Sharpe ratio
    
    Args:
        returns (list): List of period returns (percentages)
        risk_free_rate (float): Risk-free rate
        
    Returns:
        float: Sharpe ratio
    """
    if not returns:
        return 0
    
    import numpy as np
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    
    if np.std(excess_returns) == 0:
        return 0
    
    return np.mean(excess_returns) / np.std(excess_returns)

def generate_sample_data(symbol='EURUSD', periods=500):
    """
    Generate sample OHLCV data for testing and preview purposes
    
    Args:
        symbol (str): Trading symbol
        periods (int): Number of data points to generate
        
    Returns:
        pd.DataFrame: Sample market data with OHLCV columns
    """
    # Generate realistic price movements
    np.random.seed(42)  # For reproducible results
    
    # Starting price
    start_price = 1.1000 if symbol == 'EURUSD' else 1.2000
    
    # Generate price changes using random walk
    price_changes = np.random.normal(0, 0.001, periods)  # Small random movements
    
    # Generate time series
    dates = pd.date_range(start=datetime.now() - timedelta(days=periods), 
                         periods=periods, freq='H')
    
    # Calculate prices
    prices = [start_price]
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Create OHLCV data
    data = []
    for i in range(periods):
        price = prices[i]
        
        # Generate realistic OHLC from close price
        volatility = np.random.uniform(0.0001, 0.0005)  # Random volatility
        high = price * (1 + volatility)
        low = price * (1 - volatility)
        
        # Adjust open price based on previous close
        if i > 0:
            open_price = data[i-1]['close'] + np.random.normal(0, 0.0001)
        else:
            open_price = price
        
        # Generate volume
        volume = np.random.randint(100, 1000)
        
        data.append({
            'time': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('time', inplace=True)
    
    return df

def calculate_profit_factor(gross_profit, gross_loss):
    """
    Calculate profit factor
    
    Args:
        gross_profit (float): Total gross profit
        gross_loss (float): Total gross loss (positive number)
        
    Returns:
        float: Profit factor
    """
    if gross_loss == 0:
        return float('inf')  # Avoid division by zero
    
    return gross_profit / gross_loss if gross_loss > 0 else 0