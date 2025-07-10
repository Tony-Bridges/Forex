"""
MetaTrader 5 Connector Module
Provides functions to interact with the MetaTrader 5 terminal
Includes simulation mode for development and testing
"""

import pandas as pd
import numpy as np
import time
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import config

# Try to import MetaTrader5, fall back to simulation if not available
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
    logging.info("MetaTrader5 package imported successfully")
except ImportError:
    MT5_AVAILABLE = False
    logging.warning("MetaTrader5 package not available - using simulation mode")
    # Create a mock MT5 class for simulation
    class MockMT5:
        @staticmethod
        def initialize(timeout=None):
            return True
        @staticmethod
        def shutdown():
            pass
        @staticmethod
        def terminal_info():
            return {"connected": True}
        @staticmethod
        def account_info():
            return type('AccountInfo', (), {
                'login': 123456,
                'balance': 10000.0,
                'equity': 10000.0,
                'margin': 0.0,
                'margin_free': 10000.0,
                'leverage': 100,
                'currency': 'USD'
            })()
        @staticmethod
        def symbol_info(symbol):
            return type('SymbolInfo', (), {
                'name': symbol,
                'point': 0.00001 if 'JPY' not in symbol else 0.001,
                'digits': 5 if 'JPY' not in symbol else 3,
                'spread': 1.5,
                'trade_tick_value': 1.0,
                'visible': True
            })()
        @staticmethod
        def symbol_select(symbol, enable):
            return True
        @staticmethod
        def copy_rates_from_pos(symbol, timeframe, start_pos, count):
            # Generate realistic price data for simulation
            base_price = 1.2000 if 'EUR' in symbol else 1.0000
            if 'JPY' in symbol:
                base_price = 110.0
            
            dates = pd.date_range(end=datetime.now(), periods=count, freq='H')
            data = []
            
            for i, date in enumerate(dates):
                # Add some random variation
                variation = np.random.normal(0, 0.001)
                price = base_price + variation
                
                data.append({
                    'time': int(date.timestamp()),
                    'open': price,
                    'high': price + abs(np.random.normal(0, 0.0005)),
                    'low': price - abs(np.random.normal(0, 0.0005)),
                    'close': price + np.random.normal(0, 0.0002),
                    'tick_volume': np.random.randint(100, 1000),
                    'spread': 1,
                    'real_volume': np.random.randint(100, 1000)
                })
            
            return np.array([(d['time'], d['open'], d['high'], d['low'], d['close'], 
                            d['tick_volume'], d['spread'], d['real_volume']) for d in data],
                          dtype=[('time', '<u4'), ('open', '<f8'), ('high', '<f8'), 
                                ('low', '<f8'), ('close', '<f8'), ('tick_volume', '<u8'), 
                                ('spread', '<i4'), ('real_volume', '<u8')])
        @staticmethod
        def order_send(request):
            # Simulate order execution
            return type('OrderResult', (), {
                'retcode': 10009,  # TRADE_RETCODE_DONE
                'deal': np.random.randint(100000, 999999),
                'order': np.random.randint(100000, 999999),
                'volume': request.get('volume', 0.01),
                'price': request.get('price', 1.2000),
                'bid': request.get('price', 1.2000) - 0.0001,
                'ask': request.get('price', 1.2000) + 0.0001,
                'comment': 'Simulated trade'
            })()
        @staticmethod
        def positions_get(symbol=None):
            # Return empty positions for simulation
            return []
        @staticmethod
        def last_error():
            return (0, "No error")
    
    mt5 = MockMT5()

class MT5TradingBot:
    """
    Enhanced MT5 Trading Bot with AI integration and simulation capabilities
    """
    
    def __init__(self):
        self.is_connected = False
        self.simulation_mode = not MT5_AVAILABLE
        self.simulation_balance = 10000.0
        self.simulation_positions = []
        self.simulation_orders = []
        
        # Store trading performance
        self.performance_data = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'equity_history': [self.simulation_balance],
            'trade_history': []
        }
        
        logging.info(f"MT5 Trading Bot initialized - Simulation mode: {self.simulation_mode}")
    
    def connect(self, login: Optional[int] = None, password: Optional[str] = None, 
                server: Optional[str] = None) -> bool:
        """
        Connect to MetaTrader 5 terminal
        
        Args:
            login: MT5 account login
            password: MT5 account password  
            server: MT5 server name
            
        Returns:
            bool: True if connected successfully
        """
        if self.simulation_mode:
            self.is_connected = True
            logging.info("Connected to MT5 simulation mode")
            return True
        
        try:
            # Initialize MT5
            if not mt5.initialize():
                logging.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Login if credentials provided
            if login and password and server:
                if not mt5.login(login, password, server):
                    logging.error(f"MT5 login failed: {mt5.last_error()}")
                    return False
            
            # Check connection
            if not mt5.terminal_info():
                logging.error("MT5 terminal not available")
                return False
            
            self.is_connected = True
            logging.info("Connected to MT5 successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error connecting to MT5: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MT5"""
        if not self.simulation_mode:
            mt5.shutdown()
        self.is_connected = False
        logging.info("Disconnected from MT5")
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        if not self.is_connected:
            logging.error("Not connected to MT5")
            return None
        
        try:
            account_info = mt5.account_info()
            if account_info is None:
                logging.error(f"Failed to get account info: {mt5.last_error()}")
                return None
            
            return {
                'login': account_info.login,
                'balance': account_info.balance if not self.simulation_mode else self.simulation_balance,
                'equity': account_info.equity if not self.simulation_mode else self.simulation_balance,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'leverage': account_info.leverage,
                'currency': account_info.currency,
                'simulation_mode': self.simulation_mode
            }
            
        except Exception as e:
            logging.error(f"Error getting account info: {e}")
            return None
    
    def get_market_data(self, symbol: str, timeframe: str = 'H1', count: int = 100) -> pd.DataFrame:
        """
        Get historical market data
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (M1, M5, M15, M30, H1, H4, D1)
            count: Number of bars to retrieve
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.is_connected:
            logging.error("Not connected to MT5")
            return None
        
        try:
            # Convert timeframe
            tf_map = {
                'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
                'H1': 16385, 'H4': 16388, 'D1': 16408
            }
            
            mt5_timeframe = tf_map.get(timeframe, 16385)  # Default to H1
            
            # Get rates
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
            
            if rates is None:
                logging.error(f"Failed to get rates for {symbol}: {mt5.last_error()}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            logging.error(f"Error getting market data: {e}")
            return None
    
    def send_order(self, symbol: str, order_type: str, volume: float, 
                   price: Optional[float] = None, sl: Optional[float] = None, 
                   tp: Optional[float] = None, comment: str = "AI Trading Bot") -> Dict:
        """
        Send trading order
        
        Args:
            symbol: Trading symbol
            order_type: 'BUY' or 'SELL'
            volume: Lot size
            price: Entry price (None for market order)
            sl: Stop loss price
            tp: Take profit price
            comment: Order comment
            
        Returns:
            Dict with order result
        """
        if not self.is_connected:
            logging.error("Not connected to MT5")
            return None
        
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logging.error(f"Symbol {symbol} not found")
                return None
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol) if not self.simulation_mode else None
            
            if order_type.upper() == 'BUY':
                action = 0  # MT5 BUY
                if price is None:
                    price = tick.ask if tick else 1.2000
            else:
                action = 1  # MT5 SELL
                if price is None:
                    price = tick.bid if tick else 1.2000
            
            # Create order request
            request = {
                "action": action,
                "symbol": symbol,
                "volume": volume,
                "type": 0,  # Market order
                "price": price,
                "sl": sl,
                "tp": tp,
                "comment": comment,
                "type_time": 0,  # Good till cancel
                "type_filling": 0,  # Fill or kill
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result is None:
                logging.error(f"Order failed: {mt5.last_error()}")
                return None
            
            # Update simulation data
            if self.simulation_mode:
                self._update_simulation_trade(symbol, order_type, volume, price, sl, tp)
            
            # Record trade
            trade_data = {
                'symbol': symbol,
                'type': order_type,
                'volume': volume,
                'price': price,
                'sl': sl,
                'tp': tp,
                'time': datetime.now(),
                'ticket': result.order,
                'comment': comment
            }
            
            self.performance_data['trade_history'].append(trade_data)
            self.performance_data['total_trades'] += 1
            
            logging.info(f"Order executed: {order_type} {volume} {symbol} at {price}")
            
            return {
                'success': True,
                'ticket': result.order,
                'deal': result.deal,
                'volume': result.volume,
                'price': result.price,
                'retcode': result.retcode,
                'comment': result.comment
            }
            
        except Exception as e:
            logging.error(f"Error sending order: {e}")
            return None
    
    def _update_simulation_trade(self, symbol: str, order_type: str, volume: float, 
                                price: float, sl: Optional[float], tp: Optional[float]):
        """Update simulation balance and positions"""
        # Simple simulation - just track if trade would be profitable
        # In real implementation, this would be more sophisticated
        profit_loss = np.random.normal(0, 50)  # Random P&L for simulation
        
        self.simulation_balance += profit_loss
        self.performance_data['total_profit'] += profit_loss
        self.performance_data['equity_history'].append(self.simulation_balance)
        
        if profit_loss > 0:
            self.performance_data['winning_trades'] += 1
        else:
            self.performance_data['losing_trades'] += 1
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get current positions"""
        if not self.is_connected:
            return []
        
        try:
            positions = mt5.positions_get(symbol=symbol)
            position_list = []
            
            for pos in positions:
                position_list.append({
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'BUY' if pos.type == 0 else 'SELL',
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'price_current': pos.price_current,
                    'profit': pos.profit,
                    'swap': pos.swap,
                    'comment': pos.comment
                })
            
            return position_list
            
        except Exception as e:
            logging.error(f"Error getting positions: {e}")
            return []
    
    def get_performance_report(self) -> Dict:
        """Get trading performance report"""
        total_trades = self.performance_data['total_trades']
        win_rate = (self.performance_data['winning_trades'] / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': self.performance_data['winning_trades'],
            'losing_trades': self.performance_data['losing_trades'],
            'win_rate': win_rate,
            'total_profit': self.performance_data['total_profit'],
            'current_balance': self.simulation_balance if self.simulation_mode else self.get_account_info()['balance'],
            'equity_history': self.performance_data['equity_history'],
            'simulation_mode': self.simulation_mode
        }

# Global MT5 bot instance
mt5_bot = MT5TradingBot()

def initialize():
    """
    Initialize connection to the MetaTrader 5 terminal
    Returns:
        bool: True if connection successful, False otherwise
    """
    # Ensure that MT5 is not already initialized
    if mt5.terminal_info() is not None:
        mt5.shutdown()
    
    # Initialize MT5 connection
    if not mt5.initialize(timeout=config.MT5_TIMEOUT):
        logging.error(f"MT5 initialization failed. Error code: {mt5.last_error()}")
        return False
    
    # Check if connection is established
    if not mt5.terminal_info():
        logging.error("Failed to connect to MetaTrader 5 terminal")
        return False
    
    # Check if account is authorized
    if not mt5.account_info():
        logging.error("Failed to get account info. Make sure you're authorized in the terminal")
        mt5.shutdown()
        return False
    
    logging.info("MT5 connection established successfully")
    return True

def shutdown():
    """
    Close the connection to MetaTrader 5
    """
    mt5.shutdown()
    logging.info("MT5 connection closed")

def get_account_info():
    """
    Get account information from MT5
    Returns:
        dict: Account information
    """
    account_info = mt5.account_info()
    if account_info is None:
        logging.error(f"Failed to get account info. Error: {mt5.last_error()}")
        return None
    
    # Convert to dictionary
    account_dict = {
        'login': account_info.login,
        'balance': account_info.balance,
        'equity': account_info.equity,
        'margin': account_info.margin,
        'free_margin': account_info.margin_free,
        'leverage': account_info.leverage,
        'currency': account_info.currency
    }
    
    return account_dict

def get_symbol_info(symbol):
    """
    Get information about a specific symbol
    Args:
        symbol (str): Symbol name
    Returns:
        dict: Symbol information
    """
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logging.error(f"Failed to get symbol info for {symbol}. Error: {mt5.last_error()}")
        return None
    
    # Check if symbol is enabled for trading
    if not symbol_info.visible:
        logging.warning(f"Symbol {symbol} is not visible in Market Watch. Enabling it...")
        if not mt5.symbol_select(symbol, True):
            logging.error(f"Failed to enable symbol {symbol}. Error: {mt5.last_error()}")
            return None
    
    # Convert to dictionary
    info_dict = {
        'symbol': symbol_info.name,
        'point': symbol_info.point,
        'digits': symbol_info.digits,
        'spread': symbol_info.spread,
        'tick_value': symbol_info.trade_tick_value,
        'min_lot': symbol_info.volume_min,
        'max_lot': symbol_info.volume_max,
        'lot_step': symbol_info.volume_step,
        'ask': symbol_info.ask,
        'bid': symbol_info.bid
    }
    
    return info_dict

def get_current_data(symbol, timeframe, num_candles=500):
    """
    Get historical data for a specific symbol and timeframe
    Args:
        symbol (str): Symbol name
        timeframe (int): Timeframe constant (mt5.TIMEFRAME_*)
        num_candles (int): Number of candles to retrieve
    Returns:
        DataFrame: Historical price data
    """
    # Get rates
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
    
    if rates is None or len(rates) == 0:
        logging.error(f"Failed to get data for {symbol}. Error: {mt5.last_error()}")
        return None
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(rates)
    
    # Convert timestamp to datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Set index to time
    df.set_index('time', inplace=True)
    
    return df

def get_current_price(symbol, direction=1):
    """
    Get current price for a symbol
    Args:
        symbol (str): Symbol name
        direction (int): 1 for buy (ask), -1 for sell (bid)
    Returns:
        float: Current price
    """
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logging.error(f"Failed to get symbol info for {symbol}. Error: {mt5.last_error()}")
        return None
    
    if direction > 0:
        return symbol_info.ask  # For buy orders
    else:
        return symbol_info.bid  # For sell orders

def get_open_positions(symbol=None):
    """
    Get all open positions, optionally filtered by symbol
    Args:
        symbol (str, optional): Symbol to filter positions
    Returns:
        list: Open positions
    """
    if symbol:
        positions = mt5.positions_get(symbol=symbol)
    else:
        positions = mt5.positions_get()
    
    if positions is None:
        logging.debug(f"No open positions for {symbol if symbol else 'all symbols'}")
        return []
    
    # Convert to list of dictionaries
    positions_list = []
    for position in positions:
        pos_dict = {
            'ticket': position.ticket,
            'symbol': position.symbol,
            'type': position.type,  # 0 for buy, 1 for sell
            'volume': position.volume,
            'open_price': position.price_open,
            'current_price': position.price_current,
            'sl': position.sl,
            'tp': position.tp,
            'profit': position.profit,
            'open_time': datetime.fromtimestamp(position.time)
        }
        positions_list.append(pos_dict)
    
    return positions_list

def calculate_lot_size(symbol, risk_amount, stop_loss_pips):
    """
    Calculate the lot size based on risk amount and stop loss
    Args:
        symbol (str): Symbol name
        risk_amount (float): Amount to risk in account currency
        stop_loss_pips (float): Stop loss in pips
    Returns:
        float: Calculated lot size
    """
    if stop_loss_pips <= 0:
        logging.error("Stop loss must be greater than 0")
        return 0.01  # Minimum lot size
    
    # Get symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logging.error(f"Failed to get symbol info for {symbol}. Error: {mt5.last_error()}")
        return 0.01
    
    # Calculate lot size
    point = symbol_info.point
    digits = symbol_info.digits
    tick_size = symbol_info.trade_tick_size
    tick_value = symbol_info.trade_tick_value
    contract_size = symbol_info.trade_contract_size
    
    # Convert stop loss from pips to price points
    pip_value = 0.0001 if digits == 5 or digits == 3 else 0.01  # For 5/3 digit brokers or 2 digit brokers
    price_distance = stop_loss_pips * pip_value / point
    
    # Calculate lot size based on risk
    if price_distance > 0 and tick_value > 0:
        lot_size = risk_amount / (price_distance * tick_value / tick_size)
    else:
        logging.error("Invalid price distance or tick value")
        return 0.01
    
    # Adjust to symbol's lot step
    lot_step = symbol_info.volume_step
    lot_size = round(lot_size / lot_step) * lot_step
    
    # Ensure lot size is within allowed range
    min_lot = symbol_info.volume_min
    max_lot = symbol_info.volume_max
    
    lot_size = max(min_lot, min(lot_size, max_lot))
    
    return lot_size

def execute_trade(symbol, trade_type, lot_size, sl=0.0, tp=0.0, comment="MT5 ForexBot"):
    """
    Execute a trade on MT5
    Args:
        symbol (str): Symbol name
        trade_type (int): 1 for buy, -1 for sell
        lot_size (float): Lot size for the trade
        sl (float): Stop loss price
        tp (float): Take profit price
        comment (str): Trade comment
    Returns:
        bool: True if trade was successful, False otherwise
    """
    # Get symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logging.error(f"Failed to get symbol info for {symbol}. Error: {mt5.last_error()}")
        return False
    
    # Check if symbol is enabled for trading
    if not symbol_info.visible:
        logging.warning(f"Symbol {symbol} is not visible in Market Watch. Enabling it...")
        if not mt5.symbol_select(symbol, True):
            logging.error(f"Failed to enable symbol {symbol}. Error: {mt5.last_error()}")
            return False
    
    # Prepare trade request
    price = symbol_info.ask if trade_type > 0 else symbol_info.bid
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_BUY if trade_type > 0 else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": config.SLIPPAGE_POINTS,
        "magic": 123456,  # Magic number for identifying bot trades
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    
    # Execute trade
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Trade failed for {symbol}. Error code: {result.retcode}, " 
                     f"Message: {result.comment}")
        return False
    
    logging.info(f"Trade executed for {symbol}. Order ID: {result.order}")
    return True

def modify_position(ticket, sl=None, tp=None):
    """
    Modify an existing position's stop loss and/or take profit
    Args:
        ticket (int): Position ticket number
        sl (float, optional): New stop loss price
        tp (float, optional): New take profit price
    Returns:
        bool: True if modification was successful, False otherwise
    """
    # Get position info
    position = mt5.positions_get(ticket=ticket)
    if position is None or len(position) == 0:
        logging.error(f"Position with ticket {ticket} not found. Error: {mt5.last_error()}")
        return False
    
    position = position[0]
    
    # Only update if different values provided
    if (sl is not None and abs(sl - position.sl) < 0.0001) and \
       (tp is not None and abs(tp - position.tp) < 0.0001):
        return True  # No changes needed
    
    # Use existing values if not specified
    if sl is None:
        sl = position.sl
    if tp is None:
        tp = position.tp
    
    # Prepare modification request
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": position.symbol,
        "position": ticket,
        "sl": sl,
        "tp": tp
    }
    
    # Execute modification
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Position modification failed for ticket {ticket}. Error code: {result.retcode}, "
                     f"Message: {result.comment}")
        return False
    
    logging.info(f"Position {ticket} modified. New SL: {sl}, New TP: {tp}")
    return True

def close_position(ticket):
    """
    Close an existing position
    Args:
        ticket (int): Position ticket number
    Returns:
        bool: True if closing was successful, False otherwise
    """
    # Get position info
    position = mt5.positions_get(ticket=ticket)
    if position is None or len(position) == 0:
        logging.error(f"Position with ticket {ticket} not found. Error: {mt5.last_error()}")
        return False
    
    position = position[0]
    
    # Prepare close request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,  # Opposite of position type
        "position": ticket,
        "price": mt5.symbol_info_tick(position.symbol).bid if position.type == 0 else mt5.symbol_info_tick(position.symbol).ask,
        "deviation": config.SLIPPAGE_POINTS,
        "magic": 123456,
        "comment": "Close position",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    
    # Execute close
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Position close failed for ticket {ticket}. Error code: {result.retcode}, "
                     f"Message: {result.comment}")
        return False
    
    logging.info(f"Position {ticket} closed successfully")
    return True

def get_historical_deals(from_date, to_date=None, symbol=None):
    """
    Get historical deals from MT5
    Args:
        from_date (datetime): Start date for deals
        to_date (datetime, optional): End date for deals, defaults to current time
        symbol (str, optional): Symbol to filter deals
    Returns:
        DataFrame: Historical deals
    """
    if to_date is None:
        to_date = datetime.now()
    
    # Convert datetime to MT5 format
    from_date = datetime.combine(from_date.date(), datetime.min.time())
    to_date = datetime.combine(to_date.date(), datetime.max.time())
    
    # Prepare filter
    request_filter = {
        "date_from": from_date,
        "date_to": to_date
    }
    
    if symbol:
        request_filter["symbol"] = symbol
    
    # Get deals
    deals = mt5.history_deals_get(from_date, to_date, symbol=symbol if symbol else None)
    
    if deals is None or len(deals) == 0:
        logging.debug("No historical deals found")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    return df
