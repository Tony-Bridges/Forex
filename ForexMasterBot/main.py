"""
MT5 Trading Bot - Main Application

A web-based forex trading bot for MetaTrader 5 with technical analysis, 
risk management, and backtesting capabilities.
"""

from app import app
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting MT5 Trading Bot application")
    
    # Create database tables via app (already done in app.py when imported)
    
    # The application is run by Replit's workflow system
    logger.info("Application initialized and ready")