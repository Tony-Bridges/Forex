"""
Configuration file for MT5 Trading Bot
"""

import os

# Trading Settings
USE_ATR_FOR_SL = True
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0
FIXED_SL_PIPS = 50
FIXED_TP_PIPS = 100
RISK_REWARD_RATIO = 2.0

# Risk Management
RISK_PER_TRADE = 1.0  # Percentage of account balance to risk per trade
MAX_RISK_TOTAL = 5.0  # Maximum total risk as percentage of account balance
MAX_POSITIONS = 10    # Maximum number of open positions

# Timeframes (in seconds)
TIMEFRAMES = {
    'M1': 60,
    'M5': 300,
    'M15': 900,
    'M30': 1800,
    'H1': 3600,
    'H4': 14400,
    'D1': 86400,
    'W1': 604800
}

# Symbols
ALLOWED_SYMBOLS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 
    'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY'
]

# API Keys (loaded from environment)
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
XAI_API_KEY = os.environ.get('XAI_API_KEY')
PERPLEXITY_API_KEY = os.environ.get('PERPLEXITY_API_KEY')

# AI Settings
USE_AI_ENHANCEMENT = True
AI_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence level to use AI signals
AI_MODEL_REFRESH_INTERVAL = 24 * 3600  # How often to retrain AI models (in seconds)

# System Settings
DATA_DIR = 'data'
LOG_DIR = 'logs'
AI_MODELS_DIR = 'ai_models'