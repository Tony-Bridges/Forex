"""
Backtesting Module for MT5 Trading Bot

This module provides functionality to test trading strategies on historical data.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from strategy import MovingAverageRSIStrategy, MACDStrategy
from enhanced_strategy import AIEnhancedMAStrategy, AIEnhancedMACDStrategy, create_strategy
import utils

# Set up logging
logger = logging.getLogger(__name__)

class Backtester:
    """
    Backtester class for testing strategies on historical data
    """
    
    def __init__(self, strategy_instance, symbol, timeframe, start_date, end_date, initial_balance=10000):
        """
        Initialize the backtester
        
        Args:
            strategy_instance: Strategy object (MovingAverageRSIStrategy, MACDStrategy, etc.)
            symbol (str): Trading symbol (e.g., 'EURUSD')
            timeframe (str): Timeframe (e.g., 'H1', 'D1')
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            initial_balance (float): Initial account balance
        """
        self.strategy = strategy_instance
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.equity_curve = [initial_balance]
        self.trades = []
        self.positions = []
        
        logger.info(f"Backtester initialized: {symbol} {timeframe} from {start_date} to {end_date}")
    
    def _generate_historical_data(self):
        """
        Generate historical data for backtesting
        
        In a real implementation, this would fetch data from MT5 or another source.
        For the demo version, we generate synthetic data.
        
        Returns:
            DataFrame: Historical price data
        """
        # Convert string dates to datetime
        try:
            start = datetime.strptime(self.start_date, '%Y-%m-%d')
            end = datetime.strptime(self.end_date, '%Y-%m-%d')
        except ValueError:
            logger.error("Invalid date format, using default 1 year period")
            end = datetime.now()
            start = end - timedelta(days=365)
        
        # Calculate number of periods
        if self.timeframe.startswith('M'):
            # Minutes timeframes
            minutes = int(self.timeframe[1:])
            period_td = timedelta(minutes=minutes)
        elif self.timeframe.startswith('H'):
            # Hours timeframes
            hours = int(self.timeframe[1:])
            period_td = timedelta(hours=hours)
        elif self.timeframe.startswith('D'):
            # Days timeframes
            days = int(self.timeframe[1:])
            period_td = timedelta(days=days)
        else:
            # Default to 1 hour
            period_td = timedelta(hours=1)
        
        # Generate dates
        total_seconds = int((end - start).total_seconds())
        period_seconds = int(period_td.total_seconds())
        periods = total_seconds // period_seconds
        
        dates = [start + period_td * i for i in range(periods)]
        
        # Generate price data
        np.random.seed(42)  # For reproducibility
        
        # Starting price
        base_price = 1.2000 if self.symbol.startswith('EUR') else 1.5000
        
        # Generate close prices with some trend and noise
        close = np.random.normal(0, 0.0002, periods).cumsum() + base_price
        
        # Add some trends and patterns
        t = np.linspace(0, 10, periods)
        close += 0.01 * np.sin(t)  # Add cyclical pattern
        
        # Create some realistic price movement
        high = close + np.random.normal(0, 0.0003, periods).clip(0, 0.005)
        low = close - np.random.normal(0, 0.0003, periods).clip(0, 0.005)
        open_price = close.copy()
        open_price[1:] = close[:-1]
        open_price[0] = open_price[1]
        
        # Create volume with some randomness
        volume = np.random.randint(10, 200, periods)
        
        # Create DataFrame
        data = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)
        
        return data
    
    def run(self):
        """
        Run the backtest
        
        Returns:
            dict: Backtest results
        """
        logger.info("Starting backtest...")
        
        # Get historical data
        data = self._generate_historical_data()
        
        # Ensure we have enough data
        if len(data) < 100:
            logger.error("Not enough data for backtest")
            return {
                'success': False,
                'error': 'Not enough data for backtest'
            }
        
        # Initialize tracking variables
        self.current_balance = self.initial_balance
        self.equity_curve = [self.initial_balance]
        self.trades = []
        self.positions = []
        
        # Process each candle
        for i in range(100, len(data)):
            # Use a window of data for signal calculation
            window = data.iloc[i-100:i+1].copy()
            
            # Process open positions first (check for TP/SL hits)
            self._process_positions(window.iloc[-1])
            
            # Calculate signal
            signal = self.strategy.calculate_signal(window)
            
            # Execute trades based on signal
            if signal != 0 and not self._has_open_position():
                self._execute_trade(signal, window)
            
            # Update equity curve
            self.equity_curve.append(self._calculate_equity(window.iloc[-1]['close']))
        
        # Close any remaining positions at the end of the test
        if self.positions:
            self._close_positions(data.iloc[-1]['close'])
        
        # Calculate performance metrics
        results = self._calculate_performance()
        
        logger.info(f"Backtest completed: {results['total_trades']} trades, {results['win_rate']:.2f}% win rate")
        
        return results
    
    def _process_positions(self, current_data):
        """
        Process open positions (check for TP/SL hits)
        
        Args:
            current_data (Series): Current price data
        """
        if not self.positions:
            return
        
        positions_to_close = []
        
        for position in self.positions:
            # Get current price
            current_price = current_data['close']
            
            # Check for stop loss hit
            if position['type'] == 'BUY' and current_data['low'] <= position['sl']:
                # Stop loss hit for long position
                position['close_price'] = position['sl']
                position['exit_reason'] = 'SL'
                position['close_time'] = current_data.name
                positions_to_close.append(position)
                continue
                
            if position['type'] == 'SELL' and current_data['high'] >= position['sl']:
                # Stop loss hit for short position
                position['close_price'] = position['sl']
                position['exit_reason'] = 'SL'
                position['close_time'] = current_data.name
                positions_to_close.append(position)
                continue
            
            # Check for take profit hit
            if position['type'] == 'BUY' and current_data['high'] >= position['tp']:
                # Take profit hit for long position
                position['close_price'] = position['tp']
                position['exit_reason'] = 'TP'
                position['close_time'] = current_data.name
                positions_to_close.append(position)
                continue
                
            if position['type'] == 'SELL' and current_data['low'] <= position['tp']:
                # Take profit hit for short position
                position['close_price'] = position['tp']
                position['exit_reason'] = 'TP'
                position['close_time'] = current_data.name
                positions_to_close.append(position)
                continue
            
            # Update trailing stop if enabled
            if self.strategy.update_position(position, None):
                logger.debug(f"Updated position {position['ticket']}")
            
            # Update position current price
            position['current_price'] = current_price
        
        # Close positions
        for position in positions_to_close:
            self._close_position(position)
    
    def _execute_trade(self, signal, data):
        """
        Execute a trade based on signal
        
        Args:
            signal (int): Trade signal (1 for buy, -1 for sell)
            data (DataFrame): Current price data
        """
        # Ensure we have enough data
        if len(data) < 2:
            return
        
        # Get current price
        current_price = data.iloc[-1]['close']
        
        # Determine trade direction
        if signal > 0:
            trade_type = 'BUY'
        elif signal < 0:
            trade_type = 'SELL'
        else:
            return
        
        # Calculate SL and TP
        sl, tp = self.strategy.calculate_sl_tp(data, signal, current_price)
        
        # Calculate position size (simplified - 2% risk per trade)
        risk_pct = 2.0
        risk_amount = self.current_balance * (risk_pct / 100)
        
        # Calculate SL distance in pips
        sl_distance = abs(current_price - sl)
        sl_pips = sl_distance * 10000
        
        # Calculate position size based on risk
        pip_value = 10  # Simplified: $10 per pip for 1 lot
        if sl_pips > 0:
            position_size = risk_amount / (sl_pips * pip_value)
        else:
            position_size = 0.01  # Minimum position size
        
        # Limit position size
        position_size = max(0.01, min(position_size, 10.0))
        
        # Create position
        position = {
            'ticket': len(self.trades) + 1,
            'type': trade_type,
            'symbol': self.symbol,
            'volume': position_size,
            'open_price': current_price,
            'current_price': current_price,
            'sl': sl,
            'tp': tp,
            'open_time': data.iloc[-1].name,
            'close_time': None,
            'close_price': None,
            'profit': 0,
            'exit_reason': None
        }
        
        # Add position
        self.positions.append(position)
        logger.debug(f"Opened {trade_type} position at {current_price}, SL: {sl}, TP: {tp}")
    
    def _close_position(self, position):
        """
        Close a position and calculate profit
        
        Args:
            position (dict): Position information
        """
        # Calculate profit
        if position['type'] == 'BUY':
            profit_pips = (position['close_price'] - position['open_price']) * 10000
        else:
            profit_pips = (position['open_price'] - position['close_price']) * 10000
        
        # Calculate profit amount
        pip_value = 10 * position['volume']  # $10 per pip for 1 lot
        profit_amount = profit_pips * pip_value
        
        # Update position
        position['profit'] = profit_amount
        
        # Update account balance
        self.current_balance += profit_amount
        
        # Add to trades history
        self.trades.append(position)
        
        # Remove from open positions
        self.positions.remove(position)
        
        logger.debug(f"Closed position {position['ticket']}, profit: {profit_amount:.2f}")
    
    def _close_positions(self, close_price):
        """
        Close all open positions
        
        Args:
            close_price (float): Closing price to use
        """
        for position in self.positions.copy():
            position['close_price'] = close_price
            position['exit_reason'] = 'END_TEST'
            position['close_time'] = datetime.now()
            self._close_position(position)
    
    def _has_open_position(self):
        """
        Check if there are any open positions
        
        Returns:
            bool: True if there are open positions
        """
        return len(self.positions) > 0
    
    def _calculate_equity(self, current_price):
        """
        Calculate current equity
        
        Args:
            current_price (float): Current price
            
        Returns:
            float: Current equity
        """
        equity = self.current_balance
        
        # Add floating profit/loss from open positions
        for position in self.positions:
            if position['type'] == 'BUY':
                profit_pips = (current_price - position['open_price']) * 10000
            else:
                profit_pips = (position['open_price'] - current_price) * 10000
            
            # Calculate profit amount
            pip_value = 10 * position['volume']  # $10 per pip for 1 lot
            profit_amount = profit_pips * pip_value
            
            equity += profit_amount
        
        return equity
    
    def _calculate_performance(self):
        """
        Calculate performance metrics
        
        Returns:
            dict: Performance metrics
        """
        # Basic metrics
        total_trades = len(self.trades)
        profitable_trades = sum(1 for t in self.trades if t['profit'] > 0)
        losing_trades = sum(1 for t in self.trades if t['profit'] <= 0)
        
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum(t['profit'] for t in self.trades if t['profit'] > 0)
        gross_loss = abs(sum(t['profit'] for t in self.trades if t['profit'] <= 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        
        # Calculate drawdown
        max_drawdown, max_drawdown_pct = utils.calculate_drawdown(self.equity_curve)
        
        # Calculate Sharpe ratio
        returns = [(self.equity_curve[i] / self.equity_curve[i-1]) - 1 for i in range(1, len(self.equity_curve))]
        sharpe_ratio = utils.calculate_sharpe_ratio(returns)
        
        return {
            'success': True,
            'initial_balance': self.initial_balance,
            'final_balance': self.current_balance,
            'profit': self.current_balance - self.initial_balance,
            'profit_percent': ((self.current_balance / self.initial_balance) - 1) * 100,
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'max_drawdown_percent': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'equity_curve': self.equity_curve,
            'trades': self.trades
        }
    
    def plot_equity_curve(self, save_path=None):
        """
        Plot equity curve
        
        Args:
            save_path (str): Path to save plot, or None to display
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve)
        plt.title('Equity Curve')
        plt.xlabel('Trade Number')
        plt.ylabel('Account Balance')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()