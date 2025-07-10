"""
Risk Management Module
Handles position sizing, risk assessment, and overall risk controls
"""

import logging
import math
import mt5_connector
import config

class RiskManager:
    """
    Risk Manager class responsible for evaluating trade risk and position sizing
    """
    
    def __init__(self, risk_per_trade=1.0, max_risk_total=5.0):
        """
        Initialize the Risk Manager
        Args:
            risk_per_trade (float): Percentage of account balance to risk per trade
            max_risk_total (float): Maximum percentage of account balance to risk across all trades
        """
        self.risk_per_trade = risk_per_trade
        self.max_risk_total = max_risk_total
        
        logging.info(f"Risk Manager initialized with {risk_per_trade}% risk per trade, "
                    f"{max_risk_total}% maximum total risk")
    
    def evaluate_trade(self, signal, account_info, open_positions, symbol):
        """
        Evaluate whether a trade should be executed based on risk parameters
        Args:
            signal (int): Trade direction (1 for buy, -1 for sell)
            account_info (dict): Account information from MT5
            open_positions (list): Currently open positions
            symbol (str): Symbol to trade
        Returns:
            tuple: (execute_trade (bool), lot_size (float))
        """
        # First check if we're already in a position for this symbol
        for position in open_positions:
            if position['symbol'] == symbol:
                # Check if we're already in a position with the same direction
                if (signal > 0 and position['type'] == 0) or (signal < 0 and position['type'] == 1):
                    logging.info(f"Already in a {signal > 0 and 'buy' or 'sell'} position for {symbol}")
                    return False, 0.0
        
        # Check account health
        if not self._check_account_health(account_info):
            return False, 0.0
        
        # Check total risk exposure
        if not self._check_risk_exposure(account_info, open_positions):
            return False, 0.0
        
        # Calculate position size based on risk
        lot_size = self._calculate_position_size(signal, account_info, symbol)
        if lot_size <= 0:
            return False, 0.0
        
        return True, lot_size
    
    def _check_account_health(self, account_info):
        """
        Check if the account is healthy enough for trading
        Args:
            account_info (dict): Account information
        Returns:
            bool: True if account is healthy, False otherwise
        """
        # Check margin level
        if account_info['margin'] > 0:
            margin_level = (account_info['equity'] / account_info['margin']) * 100
            if margin_level < 200:  # Minimum margin level of 200%
                logging.warning(f"Margin level too low: {margin_level:.2f}%. Minimum required: 200%")
                return False
        
        # Check if equity is less than balance by a significant amount
        if account_info['balance'] > 0:
            equity_balance_ratio = account_info['equity'] / account_info['balance']
            if equity_balance_ratio < 0.9:  # Equity should be at least 90% of balance
                logging.warning(f"Equity ({account_info['equity']}) is significantly less than "
                               f"balance ({account_info['balance']}). Ratio: {equity_balance_ratio:.2f}")
                return False
        
        # Check if free margin is sufficient
        if account_info['free_margin'] < 100:  # Minimum $100 free margin
            logging.warning(f"Free margin too low: {account_info['free_margin']}")
            return False
        
        return True
    
    def _check_risk_exposure(self, account_info, open_positions):
        """
        Check if the current risk exposure is within limits
        Args:
            account_info (dict): Account information
            open_positions (list): Open positions
        Returns:
            bool: True if risk exposure is acceptable, False otherwise
        """
        # Calculate total risk from open positions
        total_risk = 0.0
        
        for position in open_positions:
            # Skip positions without stop loss
            if position['sl'] == 0:
                continue
            
            # Calculate risk for this position
            if position['type'] == 0:  # Buy
                risk = (position['open_price'] - position['sl']) * position['volume']
            else:  # Sell
                risk = (position['sl'] - position['open_price']) * position['volume']
            
            # Convert to account currency and percentage
            # This is simplified, in a real scenario you'd need to account for contract size and tick value
            risk_percent = (risk / account_info['balance']) * 100
            total_risk += risk_percent
        
        # Check against max risk limit
        if total_risk >= self.max_risk_total:
            logging.warning(f"Total risk exposure ({total_risk:.2f}%) exceeds maximum allowed ({self.max_risk_total}%)")
            return False
        
        return True
    
    def _calculate_position_size(self, signal, account_info, symbol):
        """
        Calculate appropriate position size based on risk parameters
        Args:
            signal (int): Trade direction (1 for buy, -1 for sell)
            account_info (dict): Account information
            symbol (str): Symbol to trade
        Returns:
            float: Calculated lot size
        """
        # Get symbol information
        symbol_info = mt5_connector.get_symbol_info(symbol)
        if symbol_info is None:
            logging.error(f"Failed to get symbol info for {symbol}")
            return 0.0
        
        # Get current market data
        data = mt5_connector.get_current_data(symbol, config.TIMEFRAME, 100)
        if data is None or len(data) < 20:
            logging.error(f"Failed to get market data for {symbol}")
            return 0.0
        
        # Calculate ATR for stop loss distance
        import talib
        atr = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14).iloc[-1]
        
        # Calculate stop loss distance in pips
        stop_loss_pips = atr * config.ATR_MULTIPLIER if config.USE_ATR_FOR_SL else config.FIXED_SL_PIPS
        
        # Convert pips to price for this symbol
        pip_value = 0.0001 if symbol_info['digits'] == 5 or symbol_info['digits'] == 3 else 0.01
        stop_loss_distance = stop_loss_pips * pip_value / symbol_info['point']
        
        # Calculate risk amount
        risk_amount = account_info['balance'] * (self.risk_per_trade / 100)
        
        # Calculate lot size
        lot_size = mt5_connector.calculate_lot_size(symbol, risk_amount, stop_loss_pips)
        
        # Ensure lot size is within symbol limits
        lot_size = max(symbol_info['min_lot'], min(lot_size, symbol_info['max_lot']))
        
        logging.info(f"Calculated lot size for {symbol}: {lot_size}, based on {self.risk_per_trade}% risk "
                    f"and {stop_loss_pips} pips stop loss")
        
        return lot_size
    
    def adjust_risk_based_on_performance(self, performance_metrics):
        """
        Adjust risk parameters based on recent trading performance
        Args:
            performance_metrics (dict): Trading performance metrics
        """
        # This is a simple example of adaptive risk
        # In a real bot, you'd implement more sophisticated logic
        
        if 'win_rate' in performance_metrics and 'consecutive_losses' in performance_metrics:
            win_rate = performance_metrics['win_rate']
            consecutive_losses = performance_metrics['consecutive_losses']
            
            # Reduce risk after consecutive losses
            if consecutive_losses >= 3:
                new_risk = max(0.5, self.risk_per_trade * 0.75)  # Reduce by 25%, minimum 0.5%
                logging.info(f"Reducing risk from {self.risk_per_trade}% to {new_risk}% after "
                            f"{consecutive_losses} consecutive losses")
                self.risk_per_trade = new_risk
            
            # Increase risk after good performance
            elif win_rate > 0.6 and consecutive_losses == 0:
                new_risk = min(2.0, self.risk_per_trade * 1.25)  # Increase by 25%, maximum 2%
                logging.info(f"Increasing risk from {self.risk_per_trade}% to {new_risk}% due to "
                            f"good performance (win rate: {win_rate:.2f})")
                self.risk_per_trade = new_risk
    
    def calculate_correlation(self, open_positions, symbols):
        """
        Calculate correlation between symbols to avoid over-exposure
        Args:
            open_positions (list): Currently open positions
            symbols (list): List of symbols to check correlation
        Returns:
            dict: Correlation matrix
        """
        import numpy as np
        import pandas as pd
        
        # Get historical data for all symbols
        price_data = {}
        
        for symbol in symbols:
            data = mt5_connector.get_current_data(symbol, config.TIMEFRAME, 100)
            if data is not None and len(data) > 0:
                price_data[symbol] = data['close']
        
        # Check if we have enough data
        if len(price_data) < 2:
            logging.warning("Not enough data to calculate correlations")
            return {}
        
        # Calculate pairwise correlations
        df = pd.DataFrame(price_data)
        correlations = df.corr().to_dict()
        
        # Log high correlations
        for symbol1 in correlations:
            for symbol2, corr in correlations[symbol1].items():
                if symbol1 != symbol2 and abs(corr) > 0.8:
                    logging.warning(f"High correlation ({corr:.2f}) between {symbol1} and {symbol2}")
        
        return correlations
