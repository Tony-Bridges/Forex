from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, TextAreaField
from wtforms import SelectField, FloatField, IntegerField, DateField, DecimalField
from wtforms.validators import DataRequired, Email, EqualTo, Length, ValidationError, Optional, NumberRange
from datetime import datetime, timedelta

class LoginForm(FlaskForm):
    """
    Login form for user authentication
    """
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class RegistrationForm(FlaskForm):
    """
    Registration form for new users
    """
    username = StringField('Username', validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

class StrategyForm(FlaskForm):
    """
    Form for creating and editing trading strategies
    """
    name = StringField('Strategy Name', validators=[DataRequired(), Length(min=3, max=100)])
    description = TextAreaField('Description', validators=[Length(max=500)])
    strategy_type = SelectField('Strategy Type', choices=[
        ('MA_RSI', 'Moving Average + RSI'),
        ('MACD', 'MACD'),
        ('TREND_CROSSOVER', 'Trend-Following Crossover + MACD'),
        ('BREAKOUT', 'Volume-Backed Breakout'),
        ('MEAN_REVERSION', 'RSI + Bollinger Band Mean Reversion'),
        ('SCALPING', 'Liquidity Zones + VWAP Scalping'),
        ('MULTI_SIGNAL', 'Multi-Signal AI Strategy')
    ])
    
    # MA_RSI strategy parameters
    slow_ma_period = IntegerField('Slow MA Period', default=50, validators=[NumberRange(min=5, max=200)])
    fast_ma_period = IntegerField('Fast MA Period', default=20, validators=[NumberRange(min=3, max=100)])
    rsi_period = IntegerField('RSI Period', default=14, validators=[NumberRange(min=2, max=50)])
    rsi_overbought = IntegerField('RSI Overbought Level', default=70, validators=[NumberRange(min=50, max=90)])
    rsi_oversold = IntegerField('RSI Oversold Level', default=30, validators=[NumberRange(min=10, max=50)])
    
    # MACD strategy parameters
    macd_fast_ema = IntegerField('MACD Fast EMA', default=12, validators=[NumberRange(min=3, max=50)])
    macd_slow_ema = IntegerField('MACD Slow EMA', default=26, validators=[NumberRange(min=10, max=100)])
    macd_signal_period = IntegerField('MACD Signal Period', default=9, validators=[NumberRange(min=2, max=50)])
    
    # Risk management parameters
    risk_per_trade = FloatField('Risk Per Trade (%)', default=1.0, validators=[NumberRange(min=0.1, max=5.0)])
    max_risk_total = FloatField('Maximum Total Risk (%)', default=5.0, validators=[NumberRange(min=1.0, max=20.0)])
    use_atr_for_sl = BooleanField('Use ATR for Stop Loss', default=True)
    atr_period = IntegerField('ATR Period', default=14, validators=[NumberRange(min=5, max=50)])
    atr_multiplier = FloatField('ATR Multiplier', default=2.0, validators=[NumberRange(min=0.5, max=5.0)])
    fixed_sl_pips = IntegerField('Fixed Stop Loss (pips)', default=50, validators=[NumberRange(min=10, max=200)])
    fixed_tp_pips = IntegerField('Fixed Take Profit (pips)', default=100, validators=[NumberRange(min=10, max=400)])
    risk_reward_ratio = FloatField('Risk-Reward Ratio', default=2.0, validators=[NumberRange(min=0.5, max=5.0)])
    
    submit = SubmitField('Save Strategy')

class BacktestForm(FlaskForm):
    """
    Form for running strategy backtests
    """
    strategy_id = SelectField('Strategy', validators=[DataRequired()], coerce=int)
    symbol = SelectField('Symbol', choices=[
        ('EURUSD', 'EUR/USD'),
        ('GBPUSD', 'GBP/USD'),
        ('USDJPY', 'USD/JPY'),
        ('AUDUSD', 'AUD/USD'),
        ('USDCAD', 'USD/CAD'),
        ('NZDUSD', 'NZD/USD'),
        ('EURGBP', 'EUR/GBP'),
        ('EURJPY', 'EUR/JPY')
    ])
    timeframe = SelectField('Timeframe', choices=[
        ('M5', '5 Minutes'),
        ('M15', '15 Minutes'),
        ('M30', '30 Minutes'),
        ('H1', '1 Hour'),
        ('H4', '4 Hours'),
        ('D1', 'Daily'),
        ('W1', 'Weekly')
    ])
    start_date = DateField('Start Date', default=lambda: datetime.now() - timedelta(days=365))
    end_date = DateField('End Date', default=datetime.now)
    initial_balance = FloatField('Initial Balance', default=10000.0, validators=[NumberRange(min=1000.0)])
    use_ai = BooleanField('Use AI Enhancement', default=True)
    
    submit = SubmitField('Run Backtest')

class AccountForm(FlaskForm):
    """
    Form for managing trading accounts
    """
    name = StringField('Account Name', validators=[DataRequired(), Length(min=3, max=100)])
    account_number = StringField('Account Number', validators=[DataRequired(), Length(min=5, max=50)])
    broker = StringField('Broker', validators=[DataRequired(), Length(min=2, max=100)])
    balance = FloatField('Current Balance', default=0.0)
    currency = SelectField('Currency', choices=[
        ('USD', 'US Dollar'),
        ('EUR', 'Euro'),
        ('GBP', 'British Pound'),
        ('JPY', 'Japanese Yen'),
        ('AUD', 'Australian Dollar'),
        ('CAD', 'Canadian Dollar'),
        ('CHF', 'Swiss Franc')
    ])
    is_demo = BooleanField('Demo Account', default=True)
    
    # MT5 connection settings
    mt5_login = StringField('MT5 Login', validators=[Optional(), Length(min=5, max=50)])
    mt5_password = PasswordField('MT5 Password', validators=[Optional()])
    mt5_server = StringField('MT5 Server', validators=[Optional(), Length(max=100)])
    
    submit = SubmitField('Save Account')

class MT5ConnectionForm(FlaskForm):
    """
    Form for MT5 connection settings
    """
    login = StringField('MT5 Login', validators=[DataRequired(), Length(min=5, max=50)])
    password = PasswordField('MT5 Password', validators=[DataRequired()])
    server = StringField('MT5 Server', validators=[DataRequired(), Length(max=100)])
    
    # Trading settings
    symbols = StringField('Trading Symbols (comma-separated)', 
                         default='EURUSD,GBPUSD,USDJPY,AUDUSD',
                         validators=[DataRequired()])
    timeframes = StringField('Timeframes (comma-separated)', 
                           default='H1,H4',
                           validators=[DataRequired()])
    check_interval = IntegerField('Check Interval (seconds)', 
                                default=300, 
                                validators=[NumberRange(min=60, max=3600)])
    
    submit = SubmitField('Connect & Start Trading')