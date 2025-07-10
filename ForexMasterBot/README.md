# MetaTrader 5 Forex Trading Bot

## Overview
This is a Python-based forex trading bot that integrates with MetaTrader 5. The bot implements technical analysis strategies, risk management, and backtesting capabilities to automate forex trading.

## Features
- **MT5 Integration**: Direct connection to MetaTrader 5 platform via its Python API
- **Trading Strategies**: Implementation of Moving Average crossover with RSI confirmation
- **Risk Management**: Position sizing based on account balance and risk percentage (1-2% per trade)
- **Backtesting**: Ability to test strategies on historical data with performance metrics
- **Comprehensive Logging**: Detailed logs of trading activities and errors
- **Configurable Parameters**: Easy customization through a single configuration file

## Requirements
- Python 3.8 or higher
- MetaTrader 5 terminal installed
- Required Python packages:
  - MetaTrader5
  - pandas
  - numpy
  - matplotlib
  - talib

## Installation
1. Ensure you have MetaTrader 5 installed and properly configured with a trading account
2. Install the required Python packages:
   ```
   pip install pandas numpy matplotlib MetaTrader5
   ```
3. Install TA-Lib:
   - Windows: Follow instructions at https://github.com/mrjbq7/ta-lib
   - Linux: `apt-get install ta-lib`
   - macOS: `brew install ta-lib`
4. Install the Python TA-Lib wrapper:
   ```
   pip install ta-lib
   ```
5. Clone or download this repository

## Configuration
Edit the `config.py` file to customize your trading parameters:
- Trading symbol (default: "EURUSD")
- Timeframe for analysis
- Strategy parameters (Moving Average periods, RSI levels)
- Risk management settings (risk percentage, maximum risk)
- Stop loss and take profit preferences

## Usage

### Running the Trading Bot
Execute the main script to start the bot:
