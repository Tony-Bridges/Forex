from app import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

class User(UserMixin, db.Model):
    """
    User model for authentication and account management
    """
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    strategies = db.relationship('Strategy', backref='user', lazy='dynamic')
    trading_accounts = db.relationship('TradingAccount', backref='user', lazy='dynamic')
    backtest_results = db.relationship('BacktestResult', backref='user', lazy='dynamic')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Strategy(db.Model):
    """
    Trading strategy configuration
    """
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
    
    def __repr__(self):
        return f'<Strategy {self.name} ({self.strategy_type})>'

class TradingAccount(db.Model):
    """
    MT5 trading account configuration
    """
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
    
    def __repr__(self):
        return f'<TradingAccount {self.name} ({self.account_number})>'

class BacktestResult(db.Model):
    """
    Results from strategy backtesting
    """
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
    
    def __repr__(self):
        return f'<BacktestResult {self.id} ({self.symbol} {self.timeframe})>'

class Trade(db.Model):
    """
    Record of executed trades
    """
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
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    account_id = db.Column(db.Integer, db.ForeignKey('trading_account.id'), nullable=False)
    
    def __repr__(self):
        return f'<Trade {self.ticket} {self.symbol} {self.trade_type}>'