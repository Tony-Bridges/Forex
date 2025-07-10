# MT5 Forex Trading Bot

## Overview

The MT5 Forex Trading Bot is a comprehensive Python-based forex trading application that integrates with MetaTrader 5 for automated trading. The system combines traditional technical analysis with AI-powered enhancements, providing a web-based dashboard for strategy management, backtesting, and live trading monitoring.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Flask with Bootstrap for responsive web interface
- **Templates**: Jinja2 templating engine with dark theme styling
- **Components**: Dashboard, strategy management, backtesting interface, user authentication
- **Styling**: Bootstrap 5 with custom CSS, Font Awesome icons, Chart.js for data visualization

### Backend Architecture
- **Core Framework**: Python Flask application
- **Database ORM**: SQLAlchemy with declarative base
- **Authentication**: Flask-Login for session management
- **Form Handling**: WTForms for form validation and rendering
- **Trading Engine**: Modular strategy system with AI enhancement capabilities

### AI System
- **Machine Learning**: Scikit-learn based models (RandomForest, GradientBoosting, LogisticRegression)
- **Neural Networks**: TensorFlow/Keras support (disabled in Replit environment)
- **Strategy Enhancement**: AI-powered signal generation and market analysis
- **Performance Tracking**: Model evaluation and continuous learning capabilities

## Key Components

### Trading System
- **Strategy Module**: Implements MA+RSI and MACD strategies with customizable parameters
- **Built-in Strategies**: 5 professional trading strategies ready for deployment
  - Trend-Following Crossover + MACD Confirmation
  - Volume-Backed Breakout Strategy
  - RSI + Bollinger Band Mean Reversion
  - Liquidity Zones + VWAP Scalping
  - Multi-Signal AI Strategy
- **Risk Manager**: Position sizing, drawdown protection, and risk assessment
- **MT5 Connector**: Interface for MetaTrader 5 API communication
- **Backtester**: Historical strategy testing with performance metrics

### AI Enhancement
- **AI Models**: Multiple ML models for trading signal generation
- **Strategy Manager**: Coordinates AI models with traditional strategies
- **Market Analysis**: Technical indicator analysis with AI insights
- **Learning System**: Continuous model improvement based on performance

### Web Dashboard
- **User Management**: Registration, login, and session handling
- **Strategy Management**: CRUD operations for trading strategies
- **Backtesting Interface**: Historical testing with visual results
- **Performance Monitoring**: Real-time trading metrics and analytics

## Data Flow

1. **Market Data**: MT5 → Connector → Strategy Engine → AI Enhancement → Signal Generation
2. **Order Flow**: Signal → Risk Management → Order Creation → MT5 Execution
3. **User Interface**: Web Dashboard → Database → Strategy Configuration → Trading Engine
4. **AI Learning**: Historical Data → Feature Engineering → Model Training → Deployment

## External Dependencies

### Required Services
- **MetaTrader 5**: Trading platform integration
- **Database**: PostgreSQL for data persistence
- **AI Services**: Optional external AI API integrations (OpenAI, Anthropic, etc.)

### Python Libraries
- **Core**: Flask, SQLAlchemy, pandas, numpy
- **Trading**: MetaTrader5, TA-Lib (technical analysis)
- **AI**: scikit-learn, TensorFlow (optional)
- **Web**: Flask-Login, Flask-Bootstrap, WTForms
- **Visualization**: matplotlib, Chart.js

### Configuration
- Environment variables for API keys and database connection
- Configurable trading parameters (risk levels, timeframes, symbols)
- AI model settings and confidence thresholds

## Deployment Strategy

### Development Environment
- **Platform**: Replit with Python runtime
- **Database**: PostgreSQL instance
- **Web Server**: Gunicorn with Flask application
- **Static Files**: Bootstrap CDN and local assets

### Production Considerations
- **Scalability**: Modular design allows for microservices deployment
- **Security**: Environment-based configuration, secure authentication
- **Monitoring**: Comprehensive logging and error tracking
- **Backup**: Database backups and strategy configuration persistence

### Key Files
- `app.py`: Flask application initialization and configuration
- `main.py`: Application entry point
- `views.py`: Web routes and request handlers
- `models.py`: Database schema definitions
- `strategy.py`: Core trading strategy implementations
- `builtin_strategies.py`: Professional built-in trading strategies
- `enhanced_strategy.py`: AI-enhanced strategy wrappers
- `ai_models.py`: Machine learning model implementations
- `mt5_connector.py`: MetaTrader 5 API interface
- `backtester.py`: Historical testing framework
- `risk_manager.py`: Position sizing and risk controls

The system is designed for easy deployment on cloud platforms with minimal configuration changes, supporting both development and production environments.