"""
Logging Module
Configures and manages logging for the MT5 Trading Bot
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime
import config

def setup_logger(log_level=None, log_file=None):
    """
    Set up the logger for the trading bot
    Args:
        log_level (str, optional): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (str, optional): Path to log file
    """
    # Use provided values or defaults from config
    log_level = log_level or config.LOG_LEVEL
    log_file = log_file or config.LOG_FILE
    
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
        
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler
    try:
        # Create directory for log file if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Use rotating file handler to limit log file size
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        root_logger.warning(f"Could not create log file: {str(e)}")
    
    # Log the initialization
    logging.info(f"Logger initialized with level {log_level}")

def get_trade_logger():
    """
    Create a specialized logger for trade execution
    Returns:
        Logger: Logger for trade execution
    """
    # Create a dedicated logger for trades
    trade_logger = logging.getLogger('trades')
    
    # Only set up handlers if they don't exist
    if not trade_logger.handlers:
        trade_logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Create trades log file
        try:
            # Use current date in filename
            today = datetime.now().strftime('%Y%m%d')
            trade_log_file = f"trades_{today}.log"
            
            # Create directory if needed
            if not os.path.exists('logs'):
                os.makedirs('logs')
                
            # Create file handler
            file_handler = logging.FileHandler(os.path.join('logs', trade_log_file))
            file_handler.setFormatter(formatter)
            trade_logger.addHandler(file_handler)
        except Exception as e:
            logging.warning(f"Could not create trade log file: {str(e)}")
    
    return trade_logger

def log_trade(trade_type, symbol, lot_size, entry_price, sl, tp, comment=None):
    """
    Log a trade execution
    Args:
        trade_type (str): Type of trade (BUY, SELL)
        symbol (str): Symbol traded
        lot_size (float): Lot size
        entry_price (float): Entry price
        sl (float): Stop loss price
        tp (float): Take profit price
        comment (str, optional): Additional comment
    """
    trade_logger = get_trade_logger()
    
    trade_info = f"EXECUTED {trade_type} {symbol} {lot_size} lots @ {entry_price} (SL: {sl}, TP: {tp})"
    if comment:
        trade_info += f" - {comment}"
        
    trade_logger.info(trade_info)
    
    # Also log to main logger at info level
    logging.info(f"Trade: {trade_info}")

def log_trade_result(ticket, symbol, trade_type, profit, duration, exit_reason):
    """
    Log the result of a closed trade
    Args:
        ticket (int): Trade ticket number
        symbol (str): Symbol traded
        trade_type (str): Type of trade (BUY, SELL)
        profit (float): Profit/loss amount
        duration (str): Trade duration
        exit_reason (str): Reason for trade exit (TP, SL, manual, etc.)
    """
    trade_logger = get_trade_logger()
    
    result_type = "PROFIT" if profit > 0 else "LOSS"
    result_info = f"CLOSED #{ticket} {symbol} {trade_type} with {result_type} {abs(profit):.2f} " \
                  f"after {duration} - {exit_reason}"
    
    trade_logger.info(result_info)
    
    # Also log to main logger
    log_level = logging.INFO if profit > 0 else logging.WARNING
    logging.log(log_level, f"Trade Result: {result_info}")

def log_error(function_name, error_message, exception=None):
    """
    Log an error with function context
    Args:
        function_name (str): Name of the function where error occurred
        error_message (str): Error description
        exception (Exception, optional): Exception object
    """
    log_message = f"Error in {function_name}: {error_message}"
    
    if exception:
        log_message += f" - Exception: {str(exception)}"
        
    logging.error(log_message)
    
    # For critical errors, consider sending notifications
    if "critical" in error_message.lower() or "fatal" in error_message.lower():
        try:
            from utils import send_email_notification
            send_email_notification(
                subject=f"Critical Error in MT5 Trading Bot",
                message=log_message
            )
        except Exception:
            # Don't let a notification failure cause more problems
            pass
