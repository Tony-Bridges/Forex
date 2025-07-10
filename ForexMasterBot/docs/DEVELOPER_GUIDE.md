# MT5 Forex Trading Bot - Developer Guide

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Core Modules](#core-modules)
4. [AI System](#ai-system)
5. [Web Dashboard](#web-dashboard)
6. [Database Schema](#database-schema)
7. [MT5 Integration](#mt5-integration)
8. [Development Setup](#development-setup)
9. [Adding New Features](#adding-new-features)
10. [Testing](#testing)
11. [Deployment](#deployment)
12. [Performance Optimization](#performance-optimization)
13. [Best Practices](#best-practices)
14. [Contribution Guidelines](#contribution-guidelines)

## Architecture Overview

The MT5 Forex Trading Bot is designed with a modular architecture that separates concerns and promotes maintainability. The system consists of several key components:

### High-Level Components

1. **Core Trading Engine**: Implements trading strategies, risk management, and MT5 communication
2. **AI System**: Multiple ML models and natural language processing for enhanced decision making
3. **Web Dashboard**: Flask-based web interface for configuration and monitoring
4. **Database**: PostgreSQL database for storing user data, strategies, and trading history
5. **MT5 Connector**: Interface between the trading engine and MetaTrader 5 platform

### Data Flow

1. **Market Data Flow**: 
   - MT5 platform → MT5 Connector → Trading Engine → Strategy Evaluation → AI Enhancement → Signal Generation

2. **Order Flow**:
   - Signal Generation → Risk Management → Order Creation → MT5 Connector → MT5 Platform

3. **User Interaction Flow**:
   - Web Dashboard → Database → Trading Engine Configuration → Strategy and Risk Settings

4. **AI Learning Flow**:
   - Historical Data → Feature Engineering → Model Training → Model Validation → Model Deployment

## Project Structure

```
├── ai_models.py           # AI model implementations
├── ai_strategy_manager.py # AI strategy coordination
├── app.py                 # Flask application setup
├── backtester.py          # Backtesting functionality
├── config.py              # System configuration
├── docs/                  # Documentation
│   ├── USER_MANUAL.md     # End-user documentation
│   └── DEVELOPER_GUIDE.md # Developer documentation
├── enhanced_strategy.py   # AI-enhanced trading strategies
├── forms.py               # Web form definitions
├── logger.py              # Logging utilities
├── main.py                # Application entry point
├── models.py              # Database models
├── mt5_connector.py       # MetaTrader 5 integration
├── risk_manager.py        # Risk management functionality
├── static/                # Static web assets
│   ├── css/               # Stylesheets
│   ├── js/                # JavaScript files
│   └── img/               # Images
├── strategy.py            # Traditional trading strategies
├── templates/             # HTML templates
│   ├── base.html          # Base template
│   ├── index.html         # Dashboard template
│   └── ...                # Other page templates
├── utils.py               # Utility functions
└── views.py               # Web route handlers
```

## Core Modules

### Trading Strategy Module (`strategy.py` and `enhanced_strategy.py`)

The strategy module implements various technical analysis-based trading strategies. Each strategy follows a common interface but implements its own signal generation logic.

**Key Classes**:
- `MovingAverageRSIStrategy`: Combines MA crossovers with RSI confirmation
- `MACDStrategy`: Uses MACD crossovers and histogram analysis
- `AIEnhancedStrategyBase`: Base class for AI-enhanced strategies 
- `AIEnhancedMAStrategy`: AI-enhanced MA+RSI strategy
- `AIEnhancedMACDStrategy`: AI-enhanced MACD strategy

**Strategy Interface**:
```python
def calculate_signal(self, data):
    """
    Calculate trading signal based on strategy rules
    Args:
        data (DataFrame): Market data with OHLC prices
    Returns:
        int: Signal (1 for buy, -1 for sell, 0 for no action)
    """
    
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

def update_position(self, position, data):
    """
    Update an existing position (trailing stop, etc.)
    Args:
        position (dict): Position information
        data (DataFrame): Current market data
    Returns:
        bool: True if position was updated, False otherwise
    """
```

### Risk Management Module (`risk_manager.py`)

The risk management module handles position sizing, stop loss calculation, and portfolio risk control.

**Key Functions**:
- `calculate_position_size()`: Determines appropriate lot size based on risk parameters
- `calculate_max_risk()`: Ensures total portfolio risk does not exceed limits
- `validate_sl_tp()`: Verifies stop loss and take profit levels

### Backtesting Module (`backtester.py`)

The backtesting module allows testing strategies on historical data and generates performance metrics.

**Key Components**:
- `Backtester` class: Runs simulations on historical data
- Performance calculation functions
- Visualization utilities
- Optimization algorithms

### MT5 Connector Module (`mt5_connector.py`)

The MT5 connector handles communication with the MetaTrader 5 platform.

**Key Functions**:
- `connect()`: Establishes connection to MT5 terminal
- `get_historical_data()`: Retrieves historical price data
- `place_order()`: Executes trades
- `modify_position()`: Updates existing positions
- `get_account_info()`: Retrieves account balance and status

## AI System

The AI system enhances trading decisions using multiple machine learning models and large language models.

### AI Models Module (`ai_models.py`)

**Key Components**:
- `ModelManager`: Manages the creation, training, and serving of multiple AI models
- `AITradingAdvisor`: Uses ML models to predict trading signals
- `AIMarketAnalyst`: Uses LLMs to analyze market conditions

### AI Strategy Manager (`ai_strategy_manager.py`)

Coordinates the AI components and integrates them with the trading strategies.

**Key Functions**:
- `get_ai_enhanced_signal()`: Combines traditional and AI signals
- `train_models()`: Trains ML models for specific strategies
- `generate_performance_report()`: Analyzes AI contribution to performance

### Machine Learning Models

The system uses various ML model types:
- **Classification Models**: For directional prediction (buy, sell, hold)
- **Regression Models**: For price movement prediction
- **Sequence Models**: For time series analysis and pattern recognition

### LLM Integration

The system integrates OpenAI and Anthropic APIs for market analysis:
- **GPT-4o**: For technical pattern recognition and market sentiment analysis
- **Claude-3**: For alternative perspective and risk assessment

## Web Dashboard

The web dashboard provides a user interface for configuring and monitoring the trading system.

### Flask Application (`app.py`)

Sets up the Flask application with required extensions:
- `Flask-SQLAlchemy`: For database ORM
- `Flask-Login`: For user authentication
- `Flask-WTF`: For form handling
- `Flask-Bootstrap`: For UI styling

### Views (`views.py`)

Implements route handlers for different dashboard pages:
- Dashboard overview
- Strategy management
- Account management
- Backtesting
- Live trading control
- Performance monitoring

### Forms (`forms.py`)

Defines WTForms classes for user input validation:
- `LoginForm`: User authentication
- `RegistrationForm`: New user registration
- `StrategyForm`: Strategy configuration
- `TradingAccountForm`: MT5 account setup
- `BacktestForm`: Backtest parameter configuration

### Templates (`templates/`)

HTML templates using Jinja2 templating engine:
- Base template with common layout
- Page-specific templates
- Reusable components

## Database Schema

### User Model

```python
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    strategies = db.relationship('Strategy', backref='user', lazy='dynamic')
    trading_accounts = db.relationship('TradingAccount', backref='user', lazy='dynamic')
    backtest_results = db.relationship('BacktestResult', backref='user', lazy='dynamic')
```

### Strategy Model

```python
class Strategy(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    strategy_type = db.Column(db.String(50), nullable=False)  # 'MA_RSI', 'MACD', etc.
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Strategy Parameters
    slow_ma_period = db.Column(db.Integer, default=50)
    fast_ma_period = db.Column(db.Integer, default=20)
    rsi_period = db.Column(db.Integer, default=14)
    rsi_overbought = db.Column(db.Integer, default=70)
    rsi_oversold = db.Column(db.Integer, default=30)
    
    macd_fast_ema = db.Column(db.Integer, default=12)
    macd_slow_ema = db.Column(db.Integer, default=26)
    macd_signal_period = db.Column(db.Integer, default=9)
    
    # Risk Parameters
    risk_per_trade = db.Column(db.Float, default=1.0)
    max_risk_total = db.Column(db.Float, default=5.0)
    use_atr_for_sl = db.Column(db.Boolean, default=True)
    atr_period = db.Column(db.Integer, default=14)
    atr_multiplier = db.Column(db.Float, default=2.0)
    fixed_sl_pips = db.Column(db.Integer, default=50)
    fixed_tp_pips = db.Column(db.Integer, default=100)
    risk_reward_ratio = db.Column(db.Float, default=2.0)
    
    backtest_results = db.relationship('BacktestResult', backref='strategy', lazy='dynamic')
```

### TradingAccount Model

```python
class TradingAccount(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    account_number = db.Column(db.String(50), nullable=False)
    broker = db.Column(db.String(100))
    balance = db.Column(db.Float, default=0.0)
    currency = db.Column(db.String(3), default='USD')
    is_demo = db.Column(db.Boolean, default=True)
    mt5_config = db.Column(db.Text)  # JSON configuration for MT5 connection
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    trades = db.relationship('Trade', backref='account', lazy='dynamic')
```

### BacktestResult Model

```python
class BacktestResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), nullable=False)
    timeframe = db.Column(db.String(20), nullable=False)
    start_date = db.Column(db.DateTime, nullable=False)
    end_date = db.Column(db.DateTime, nullable=False)
    initial_balance = db.Column(db.Float, nullable=False)
    final_balance = db.Column(db.Float, nullable=False)
    total_trades = db.Column(db.Integer, nullable=False)
    profitable_trades = db.Column(db.Integer, nullable=False)
    losing_trades = db.Column(db.Integer, nullable=False)
    win_rate = db.Column(db.Float, nullable=False)
    profit_factor = db.Column(db.Float, nullable=False)
    max_drawdown = db.Column(db.Float, nullable=False)
    max_drawdown_percent = db.Column(db.Float, nullable=False)
    sharpe_ratio = db.Column(db.Float)
    equity_curve = db.Column(db.Text)  # JSON string of equity curve data
    trades_data = db.Column(db.Text)  # JSON string of trades data
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    strategy_id = db.Column(db.Integer, db.ForeignKey('strategy.id'), nullable=False)
```

### Trade Model

```python
class Trade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ticket = db.Column(db.String(50), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    trade_type = db.Column(db.String(10), nullable=False)  # 'BUY', 'SELL'
    volume = db.Column(db.Float, nullable=False)
    open_price = db.Column(db.Float, nullable=False)
    close_price = db.Column(db.Float)
    stop_loss = db.Column(db.Float)
    take_profit = db.Column(db.Float)
    open_time = db.Column(db.DateTime, nullable=False)
    close_time = db.Column(db.DateTime)
    profit = db.Column(db.Float, default=0.0)
    pips = db.Column(db.Float, default=0.0)
    status = db.Column(db.String(20), default='OPEN')  # 'OPEN', 'CLOSED', 'PENDING'
    exit_reason = db.Column(db.String(50))  # 'TP', 'SL', 'manual', etc.
    comment = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    account_id = db.Column(db.Integer, db.ForeignKey('trading_account.id'), nullable=False)
```

## MT5 Integration

The MT5 integration allows communication between the trading bot and the MetaTrader 5 platform. This is implemented through the `mt5_connector.py` module and MQL5 scripts on the MT5 side.

### Python Connector (`mt5_connector.py`)

The connector provides functions to:
- Establish and manage connection to MT5
- Retrieve account information and market data
- Place, modify, and close orders
- Monitor open positions

### MQL5 Expert Advisor

The EA on the MT5 side is responsible for:
- Receiving commands from the Python connector
- Executing trades on the market
- Sending trade confirmations back to the Python application

### Communication Protocol

The system uses one of two communication methods:
1. **Direct API**: Using the MetaTrader5 Python package (Windows-only)
2. **HTTP API**: Using a REST API approach with WebRequests in MQL5

## Development Setup

### Prerequisites

- Python 3.8 or higher
- PostgreSQL 12 or higher
- MetaTrader 5 platform (for live trading)
- Git

### Installation Steps

1. Clone the repository:
   ```
   git clone https://github.com/username/mt5-forex-trading-bot.git
   cd mt5-forex-trading-bot
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```
   export FLASK_APP=main.py
   export FLASK_ENV=development
   export DATABASE_URL=postgresql://user:password@localhost/mt5_trading_bot
   export SECRET_KEY=your_secret_key
   ```
   On Windows, use `set` instead of `export`.

5. Create the database:
   ```
   createdb mt5_trading_bot
   ```

6. Initialize the database:
   ```
   flask db upgrade
   ```

7. Run the application:
   ```
   flask run
   ```

## Adding New Features

### Adding a New Trading Strategy

1. Create a new strategy class in `strategy.py`:
   ```python
   class NewStrategy:
       def __init__(self, symbol, timeframe, param1, param2):
           self.symbol = symbol
           self.timeframe = timeframe
           self.param1 = param1
           self.param2 = param2
           
       def calculate_signal(self, data):
           # Implement signal logic
           return signal
           
       def calculate_sl_tp(self, data, signal, entry_price):
           # Implement SL/TP calculation
           return sl, tp
           
       def update_position(self, position, data):
           # Implement position update logic
           return updated
   ```

2. Create an AI-enhanced version in `enhanced_strategy.py`:
   ```python
   class AIEnhancedNewStrategy(AIEnhancedStrategyBase):
       def __init__(self, symbol, timeframe, param1, param2, use_ai=True):
           traditional_strategy = NewStrategy(symbol, timeframe, param1, param2)
           super().__init__(traditional_strategy, use_ai)
   ```

3. Add strategy parameters to the database model in `models.py`

4. Update the form in `forms.py` to include the new strategy type and parameters

5. Add the strategy to the factory function in `enhanced_strategy.py`:
   ```python
   def create_strategy(strategy_type, symbol, timeframe, params=None, use_ai=True):
       # Existing code...
       
       elif strategy_type == 'NEW_STRATEGY':
           return AIEnhancedNewStrategy(
               symbol=symbol,
               timeframe=timeframe,
               param1=params.get('param1', default_value),
               param2=params.get('param2', default_value),
               use_ai=use_ai
           )
   ```

### Adding a New AI Model

1. Add the model to `ai_models.py`:
   ```python
   def train_model(self, model_name, strategy_type, training_data):
       # Existing code...
       
       elif model_name == 'new_model_type':
           # Initialize and train your new model
           model = YourNewModel(params)
           model.fit(X_train, y_train)
           
           # Save the model
           # ...
           
           # Return results
           return {
               'success': True,
               'model_name': model_name,
               'accuracy': accuracy,
               # ... other metrics
           }
   ```

2. Update the `AVAILABLE_MODELS` dictionary in `ModelManager` class:
   ```python
   AVAILABLE_MODELS = {
       # Existing models...
       'new_model_type': 'Your New Model Description'
   }
   ```

3. Add prediction logic for the new model:
   ```python
   def predict(self, model_name, strategy_type, data):
       # Existing code...
       
       elif model_name == 'new_model_type':
           # Prediction logic for your new model
           predictions = model.predict(X)
           return predictions
   ```

### Adding a New Dashboard Feature

1. Create a new route in `views.py`:
   ```python
   @app.route('/new-feature')
   @login_required
   def new_feature():
       # Your route logic
       return render_template('new_feature.html', data=data)
   ```

2. Create the corresponding template in `templates/new_feature.html`

3. Add a link to the new feature in the navigation menu in `templates/base.html`

## Testing

### Unit Testing

Unit tests are located in the `tests` directory. Run them using:
```
python -m unittest discover tests
```

Key test modules:
- `test_strategy.py`: Tests for trading strategies
- `test_risk_manager.py`: Tests for risk management functions
- `test_backtester.py`: Tests for backtesting functionality
- `test_ai_models.py`: Tests for AI models

### Integration Testing

Integration tests verify the interaction between components:
- Database operations
- MT5 communication
- Web interface functionality

Run integration tests:
```
python -m unittest discover tests/integration
```

### Backtesting

Use the backtesting module to verify strategy performance:
```python
from backtester import Backtester
from strategy import MovingAverageRSIStrategy
from risk_manager import RiskManager

# Create strategy and risk manager
strategy = MovingAverageRSIStrategy('EURUSD', 'H1', 50, 20, 14, 70, 30)
risk_manager = RiskManager(risk_per_trade=1.0)

# Create and run backtester
backtester = Backtester(
    strategy_instance=strategy,
    risk_manager_instance=risk_manager,
    symbol='EURUSD',
    timeframe='H1',
    start_date='2022-01-01',
    end_date='2022-12-31',
    initial_balance=10000
)

results = backtester.run()
```

## Deployment

### Production Server Setup

1. Use Gunicorn as the WSGI server:
   ```
   gunicorn -w 4 -b 0.0.0.0:5000 main:app
   ```

2. Use Nginx as a reverse proxy:
   ```nginx
   server {
       listen 80;
       server_name your_domain.com;
       
       location / {
           proxy_pass http://127.0.0.1:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

3. Configure PostgreSQL for production:
   - Enable SSL
   - Tune for performance
   - Set up regular backups

### Docker Deployment

A Dockerfile is provided for containerized deployment:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=main.py
ENV DATABASE_URL=postgresql://user:password@db/mt5_trading_bot
ENV SECRET_KEY=production_secret_key

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "main:app"]
```

Use Docker Compose for orchestration with PostgreSQL:

```yaml
version: '3'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    depends_on:
      - db
    environment:
      - DATABASE_URL=postgresql://user:password@db/mt5_trading_bot
      - SECRET_KEY=production_secret_key
    volumes:
      - ./ai_models:/app/ai_models
      - ./logs:/app/logs
      
  db:
    image: postgres:13
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=mt5_trading_bot

volumes:
  postgres_data:
```

## Performance Optimization

### Database Optimization

- Use proper indexing for frequently queried fields
- Implement caching for read-heavy operations
- Use connection pooling
- Optimize query patterns

### AI Model Optimization

- Use model quantization for reduced size
- Implement lazy loading of models
- Consider using optimized inference frameworks like ONNX Runtime
- Use model pruning to reduce complexity

### Web Dashboard Optimization

- Implement client-side caching
- Use AJAX for dynamic content
- Minimize CSS and JavaScript assets
- Use pagination for large datasets

## Best Practices

### Code Style

Follow PEP 8 guidelines for Python code style:
- Use 4 spaces for indentation
- Limit line length to 79 characters
- Use descriptive variable names
- Add docstrings to classes and functions

### Error Handling

Implement robust error handling:
- Use try-except blocks for operations that may fail
- Log all exceptions with context
- Provide meaningful error messages to users
- Implement graceful degradation when components fail

### Security

Follow security best practices:
- Store sensitive credentials in environment variables
- Use HTTPS for all communication
- Implement proper authentication and authorization
- Validate all user input
- Use parameterized queries to prevent SQL injection
- Regularly update dependencies

### Logging

Implement comprehensive logging:
- Use different log levels appropriately
- Include context in log messages
- Configure log rotation
- Store logs in a centralized location for production

## Contribution Guidelines

### Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Write tests for your changes
5. Run the test suite to ensure all tests pass
6. Commit your changes: `git commit -m "Add feature: your feature description"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a pull request

### Code Review

All pull requests will be reviewed according to:
- Adherence to code style guidelines
- Test coverage for new features
- Documentation quality
- Performance implications
- Security considerations

### Development Workflow

1. Issues are tracked in the issue tracker
2. Feature requests and bug reports should include detailed descriptions
3. Major changes should be discussed with the team before implementation
4. Follow semantic versioning for releases