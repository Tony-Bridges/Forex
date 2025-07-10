# MT5 Forex Trading Bot - User Manual

## Table of Contents

1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Getting Started](#getting-started)
5. [Web Dashboard](#web-dashboard)
6. [Trading Strategies](#trading-strategies)
7. [AI Enhancement](#ai-enhancement)
8. [Backtesting](#backtesting)
9. [Risk Management](#risk-management)
10. [Live Trading](#live-trading)
11. [Performance Monitoring](#performance-monitoring)
12. [Troubleshooting](#troubleshooting)
13. [FAQ](#faq)

## Introduction

The MT5 Forex Trading Bot is an advanced algorithmic trading system that combines traditional technical analysis with cutting-edge artificial intelligence to deliver optimized trading decisions in the forex market.

### Key Features

- **Multiple Trading Strategies**: Includes Moving Average + RSI and MACD strategies
- **AI Enhancement**: Uses multiple learning models to improve trading performance over time
- **Advanced Backtesting**: Test strategies on historical data with detailed performance metrics
- **Risk Management**: Sophisticated position sizing and stop-loss calculation
- **Web Dashboard**: User-friendly interface for configuration and monitoring
- **MetaTrader 5 Integration**: Seamless connection to MT5 trading platform

## System Requirements

### Minimum Requirements

- **Operating System**: Windows 10/11 (for MT5 integration), any OS for web dashboard
- **Memory**: 4GB RAM
- **Storage**: 1GB free disk space
- **Internet**: Stable broadband connection
- **MetaTrader 5**: Installed and configured with a trading account
- **Web Browser**: Chrome, Firefox, or Edge (latest versions)

### Recommended

- **Memory**: 8GB RAM or more
- **CPU**: Multi-core processor
- **Internet**: High-speed, low-latency connection

## Installation

### Web Dashboard Installation

1. Clone the repository or download the source code
2. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```
3. Set up the PostgreSQL database:
   ```
   createdb mt5_trading_bot
   ```
4. Initialize the database:
   ```
   python init_db.py
   ```
5. Start the web server:
   ```
   python main.py
   ```
6. Access the dashboard at http://localhost:5000

### MetaTrader 5 Integration

1. Install MetaTrader 5 on a Windows machine
2. Copy the MT5 configuration files to the appropriate directory:
   ```
   copy mt5_scripts\* [MT5 installation path]\MQL5\Experts\
   ```
3. In MT5, navigate to Tools > Options > Expert Advisors
4. Enable "Allow WebRequest for listed URLs" and add your server URL
5. Restart MT5 and attach the EA to a chart

## Getting Started

### Initial Setup

1. Register a new account on the web dashboard
2. Create a new trading account connection:
   - Go to "Trading Accounts" and click "New Account"
   - Enter your MT5 account details
   - Test the connection
3. Create your first strategy:
   - Go to "Strategies" and click "New Strategy"
   - Choose a strategy type and configure parameters
   - Save the strategy
4. Run a backtest:
   - Go to "Backtesting" and select your strategy
   - Choose a symbol, timeframe, and date range
   - Click "Run Backtest"
5. Review results and fine-tune your strategy

## Web Dashboard

The web dashboard provides a user-friendly interface to manage all aspects of your trading system.

### Dashboard Sections

- **Home**: Overview of system status and performance metrics
- **Strategies**: Create, edit, and manage trading strategies
- **Trading Accounts**: Manage MT5 account connections
- **Backtesting**: Test strategies on historical data
- **Live Trading**: Control and monitor live trading sessions
- **Performance**: Detailed analysis of trading performance
- **Settings**: System configuration and user preferences

### User Management

- **Registration**: Create a new user account
- **Login**: Secure authentication system
- **Profile**: Manage your user profile and preferences
- **Security**: Change password and security settings

## Trading Strategies

The system offers multiple trading strategies that can be customized to suit your trading style.

### Moving Average + RSI Strategy

This strategy combines moving average crossovers with RSI confirmation.

#### Parameters

- **Slow MA Period**: Period for the slow moving average (default: 50)
- **Fast MA Period**: Period for the fast moving average (default: 20)
- **RSI Period**: Period for RSI calculation (default: 14)
- **RSI Overbought**: Level considered overbought (default: 70)
- **RSI Oversold**: Level considered oversold (default: 30)

#### Signal Logic

- **Buy Signal**: Fast MA crosses above Slow MA and RSI is not overbought
- **Sell Signal**: Fast MA crosses below Slow MA and RSI is not oversold

### MACD Strategy

This strategy uses MACD crossovers and histogram confirmation.

#### Parameters

- **Fast EMA**: Period for the fast EMA component (default: 12)
- **Slow EMA**: Period for the slow EMA component (default: 26)
- **Signal Period**: Period for the signal line (default: 9)

#### Signal Logic

- **Buy Signal**: MACD line crosses above Signal line and histogram is increasing
- **Sell Signal**: MACD line crosses below Signal line and histogram is decreasing

## AI Enhancement

The system incorporates multiple AI modules to enhance trading decisions.

### AI Components

- **AITradingAdvisor**: Uses machine learning models to predict trade signals
- **AIMarketAnalyst**: Uses advanced LLMs to analyze market conditions
- **ModelManager**: Manages the training and deployment of multiple models

### Learning Models

The system uses several types of machine learning models:

- **Random Forest Classifier**: For pattern recognition and classification
- **Gradient Boosting Classifier**: For improved prediction accuracy
- **Logistic Regression**: For probability-based trading signals
- **LSTM Neural Network**: For time series forecasting
- **MLP Network**: For complex non-linear pattern detection

### Training Process

The AI models are trained automatically using historical data:

1. **Data Preparation**: Technical indicators are calculated and features extracted
2. **Feature Engineering**: Additional derived features are created
3. **Training**: Multiple models are trained with different algorithms
4. **Validation**: Models are evaluated using cross-validation
5. **Deployment**: The best-performing models are deployed for prediction

### AI-Enhanced Decision Making

The system combines signals from:

1. **Traditional Technical Analysis**: Using standard trading rules
2. **Machine Learning Models**: Using pattern recognition
3. **Market Analysis from LLMs**: Using natural language understanding

These signals are weighted based on confidence levels to produce the final trading decision.

## Backtesting

The backtesting module allows you to test strategies on historical data.

### Running a Backtest

1. Select a strategy from your strategy list
2. Choose a symbol (e.g., EURUSD, GBPUSD)
3. Select a timeframe (e.g., M15, H1, D1)
4. Specify the date range for testing
5. Set the initial balance for the test
6. Click "Run Backtest"

### Backtest Results

The system provides detailed performance metrics:

- **Profit/Loss**: Total and percentage gain/loss
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Drawdown**: Maximum equity drawdown in amount and percentage
- **Sharpe Ratio**: Risk-adjusted return metric
- **Equity Curve**: Visual representation of account balance over time
- **Trade List**: Detailed list of all trades with entry/exit points

### Optimization

You can optimize strategy parameters using the backtesting module:

1. Select a strategy to optimize
2. Choose parameters to optimize and their ranges
3. Select optimization criteria (e.g., max profit, best Sharpe ratio)
4. Run the optimization process
5. Review results and select the best parameter set

## Risk Management

The system incorporates sophisticated risk management capabilities.

### Position Sizing

- **Percentage Risk**: Risk a fixed percentage of account balance per trade
- **Fixed Lot Size**: Use a specific lot size for all trades
- **ATR-Based Sizing**: Adjust position size based on market volatility

### Stop Loss Methods

- **Fixed Pips**: Set stop loss at a fixed distance from entry
- **ATR-Based**: Dynamic stop loss based on market volatility
- **Support/Resistance**: Place stop loss at key market levels
- **AI-Enhanced**: Use AI to identify optimal stop loss levels

### Take Profit Methods

- **Fixed Pips**: Set take profit at a fixed distance from entry
- **Risk-Reward Ratio**: Set take profit based on risk-reward ratio
- **Support/Resistance**: Place take profit at key market levels
- **AI-Enhanced**: Use AI to identify optimal take profit levels

### Advanced Features

- **Trailing Stop**: Automatically move stop loss as price moves in your favor
- **Break-Even Stop**: Move stop loss to entry price after reaching a threshold
- **Partial Close**: Close parts of a position at different profit levels
- **Maximum Risk Limit**: Cap total portfolio risk across all open trades

## Live Trading

The system can be used for live trading through the MT5 integration.

### Setting Up Live Trading

1. Configure your MT5 connection in "Trading Accounts"
2. Select a strategy and configure its parameters
3. Enable the strategy for live trading
4. Set risk management parameters
5. Start the trading session

### Monitoring Live Trading

The dashboard provides real-time monitoring of live trading:

- **Open Positions**: View all current open positions
- **Trade History**: Review completed trades
- **Account Metrics**: Monitor balance, equity, margin, and more
- **Market Data**: View current market prices and indicators
- **Strategy Performance**: Track performance of active strategies

### Trade Management

You can manage open trades directly from the dashboard:

- **Close Position**: Close any open position manually
- **Modify Stop Loss/Take Profit**: Adjust exit levels for open positions
- **Add/Remove Trailing Stop**: Enable or disable trailing stops

## Performance Monitoring

The system provides comprehensive performance monitoring tools.

### Performance Metrics

- **Profit/Loss**: Total and percentage gain/loss
- **Win Rate**: Percentage of profitable trades
- **Average Win/Loss**: Average profit on winning vs. losing trades
- **Expectancy**: Expected return per trade
- **Drawdown**: Maximum equity drawdown
- **Recovery Factor**: Ratio of profit to maximum drawdown
- **Sharpe Ratio**: Risk-adjusted return metric
- **Sortino Ratio**: Downside risk-adjusted return metric

### Performance Reports

The system generates detailed performance reports:

- **Daily/Weekly/Monthly Reports**: Performance breakdown by time period
- **Symbol Reports**: Performance by currency pair
- **Strategy Reports**: Performance by trading strategy
- **AI Enhancement Reports**: Impact of AI on trading decisions

### Performance Visualization

Multiple charts and visualizations are available:

- **Equity Curve**: Account balance over time
- **Drawdown Chart**: Drawdown percentage over time
- **Trade Distribution**: Analysis of trade entry/exit times
- **Win/Loss Distribution**: Analysis of winning and losing trades
- **Heat Map**: Performance by day of week and time of day

## Troubleshooting

### Common Issues

#### Connection Problems

- **MT5 Connection Failed**: Check MT5 is running and WebRequest URLs are properly configured
- **Database Connection Error**: Verify PostgreSQL is running and credentials are correct
- **Web Server Not Starting**: Check port availability and permissions

#### Trading Issues

- **No Signals Generated**: Verify strategy parameters and market data
- **Orders Not Executing**: Check MT5 journal for errors and account status
- **Incorrect Position Sizing**: Verify risk settings and account balance

#### AI-Related Issues

- **AI Models Not Loading**: Check model files and directory permissions
- **High Memory Usage**: Reduce model complexity or number of active models
- **Slow Prediction Time**: Optimize model size or hardware resources

### Logs and Diagnostics

The system maintains detailed logs for troubleshooting:

- **Application Logs**: General system operation logs
- **Trade Logs**: Detailed logs of trade execution
- **Error Logs**: System errors and exceptions
- **MT5 Communication Logs**: MT5 API communication

Access logs in the `/logs` directory or through the dashboard's "Diagnostics" section.

## FAQ

### General Questions

**Q: Can I use this bot without MetaTrader 5?**
A: The web dashboard for strategy configuration and backtesting works without MT5, but live trading requires MT5 integration.

**Q: Does the bot work with other trading platforms?**
A: Currently, only MetaTrader 5 is supported for live trading.

**Q: What currencies can I trade?**
A: Any forex pair available on your MT5 broker can be traded.

### AI-Related Questions

**Q: How does the AI enhancement improve trading?**
A: The AI models identify patterns that traditional indicators might miss and adapt to changing market conditions over time.

**Q: How often are the AI models updated?**
A: The models are automatically retrained periodically based on new market data and trade results.

**Q: Can I disable the AI components?**
A: Yes, each strategy can be configured to use or disable AI enhancement.

### Performance Questions

**Q: What kind of performance can I expect?**
A: Performance varies based on market conditions, strategy configurations, and risk settings. Always run backtests to evaluate expected performance.

**Q: How much historical data is needed for good AI performance?**
A: At least 6 months of historical data is recommended for initial model training.