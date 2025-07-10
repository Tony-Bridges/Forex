"""
Views for the MT5 Trading Bot application
"""

import os
import logging
from datetime import datetime

from flask import render_template, flash, redirect, url_for, request, jsonify, abort
from flask_login import login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash

from app import db, login_manager
from models import User, Strategy, TradingAccount, BacktestResult, Trade
from strategy import MovingAverageRSIStrategy, MACDStrategy
from ai_trading_engine import trading_engine
from mt5_connector import mt5_bot
from builtin_strategies import builtin_strategy_manager
import utils

# Set up logging
logger = logging.getLogger(__name__)

# Import forms
from forms import LoginForm, RegistrationForm, StrategyForm, BacktestForm, AccountForm, MT5ConnectionForm

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

def init_views(app):
    """
    Initialize views for the application
    
    Args:
        app: Flask application instance
    """
    
    # Home page
    @app.route('/')
    def index():
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        return render_template('landing.html', title='Welcome to MT5 Trading Bot')
    
    # Dashboard
    @app.route('/dashboard')
    @login_required
    def dashboard():
        strategies = Strategy.query.filter_by(user_id=current_user.id).all()
        accounts = TradingAccount.query.filter_by(user_id=current_user.id).all()
        
        # Get some overview statistics
        strategy_count = len(strategies)
        account_count = len(accounts)
        backtest_count = BacktestResult.query.filter_by(user_id=current_user.id).count()
        
        # Temporarily set trade_count to 0 due to schema issue
        trade_count = 0
        
        # Get account balance
        total_balance = sum(account.balance for account in accounts) if accounts else 0
        
        # Set recent_trades to empty list due to schema issue
        recent_trades = []
        
        return render_template(
            'index.html',
            title='Dashboard',
            strategies=strategies,
            accounts=accounts,
            strategy_count=strategy_count,
            account_count=account_count,
            backtest_count=backtest_count,
            trade_count=trade_count,
            total_balance=total_balance,
            recent_trades=recent_trades
        )
    
    # Login
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        
        form = LoginForm()
        if form.validate_on_submit():
            user = User.query.filter_by(username=form.username.data).first()
            if user is None or not user.check_password(form.password.data):
                flash('Invalid username or password', 'danger')
                return redirect(url_for('login'))
            login_user(user)
            next_page = request.args.get('next')
            if not next_page or not next_page.startswith('/'):
                next_page = url_for('dashboard')
            return redirect(next_page)
        
        return render_template('login.html', title='Sign In', form=form)
    
    # Logout
    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        return redirect(url_for('index'))
    
    # Register
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        
        form = RegistrationForm()
        if form.validate_on_submit():
            user = User(username=form.username.data, email=form.email.data)
            user.set_password(form.password.data)
            db.session.add(user)
            db.session.commit()
            flash('Congratulations, you are now registered!', 'success')
            return redirect(url_for('login'))
        
        return render_template('register.html', title='Register', form=form)
    
    # Strategy list
    @app.route('/strategies')
    @login_required
    def strategy_list():
        strategies = Strategy.query.filter_by(user_id=current_user.id).all()
        return render_template('strategy_list.html', title='My Strategies', strategies=strategies)
    
    # Strategy detail
    @app.route('/strategy/<int:id>')
    @login_required
    def strategy_detail(id):
        strategy = Strategy.query.get_or_404(id)
        if strategy.user_id != current_user.id:
            abort(403)  # Forbidden
        
        # Get backtest results for this strategy
        backtest_results = BacktestResult.query.filter_by(strategy_id=strategy.id).order_by(BacktestResult.created_at.desc()).all()
        
        return render_template(
            'strategy_detail.html',
            title=strategy.name,
            strategy=strategy,
            backtest_results=backtest_results
        )
    
    # New strategy
    @app.route('/strategy/new', methods=['GET', 'POST'])
    @login_required
    def new_strategy():
        form = StrategyForm()
        if form.validate_on_submit():
            strategy = Strategy(
                name=form.name.data,
                description=form.description.data,
                strategy_type=form.strategy_type.data,
                user_id=current_user.id,
                slow_ma_period=form.slow_ma_period.data,
                fast_ma_period=form.fast_ma_period.data,
                rsi_period=form.rsi_period.data,
                rsi_overbought=form.rsi_overbought.data,
                rsi_oversold=form.rsi_oversold.data,
                macd_fast_ema=form.macd_fast_ema.data,
                macd_slow_ema=form.macd_slow_ema.data,
                macd_signal_period=form.macd_signal_period.data,
                risk_per_trade=form.risk_per_trade.data,
                max_risk_total=form.max_risk_total.data,
                use_atr_for_sl=form.use_atr_for_sl.data,
                atr_period=form.atr_period.data,
                atr_multiplier=form.atr_multiplier.data,
                fixed_sl_pips=form.fixed_sl_pips.data,
                fixed_tp_pips=form.fixed_tp_pips.data,
                risk_reward_ratio=form.risk_reward_ratio.data
            )
            db.session.add(strategy)
            db.session.commit()
            flash('Strategy created successfully!', 'success')
            return redirect(url_for('strategy_list'))
        
        return render_template('strategy_form.html', title='New Strategy', form=form)
    
    # Edit strategy
    @app.route('/strategy/edit/<int:id>', methods=['GET', 'POST'])
    @login_required
    def edit_strategy(id):
        strategy = Strategy.query.get_or_404(id)
        if strategy.user_id != current_user.id:
            abort(403)  # Forbidden
        
        form = StrategyForm()
        if form.validate_on_submit():
            strategy.name = form.name.data
            strategy.description = form.description.data
            strategy.strategy_type = form.strategy_type.data
            strategy.slow_ma_period = form.slow_ma_period.data
            strategy.fast_ma_period = form.fast_ma_period.data
            strategy.rsi_period = form.rsi_period.data
            strategy.rsi_overbought = form.rsi_overbought.data
            strategy.rsi_oversold = form.rsi_oversold.data
            strategy.macd_fast_ema = form.macd_fast_ema.data
            strategy.macd_slow_ema = form.macd_slow_ema.data
            strategy.macd_signal_period = form.macd_signal_period.data
            strategy.risk_per_trade = form.risk_per_trade.data
            strategy.max_risk_total = form.max_risk_total.data
            strategy.use_atr_for_sl = form.use_atr_for_sl.data
            strategy.atr_period = form.atr_period.data
            strategy.atr_multiplier = form.atr_multiplier.data
            strategy.fixed_sl_pips = form.fixed_sl_pips.data
            strategy.fixed_tp_pips = form.fixed_tp_pips.data
            strategy.risk_reward_ratio = form.risk_reward_ratio.data
            
            db.session.commit()
            flash('Strategy updated successfully!', 'success')
            return redirect(url_for('strategy_detail', id=strategy.id))
        
        # Pre-fill form with existing data
        form.name.data = strategy.name
        form.description.data = strategy.description
        form.strategy_type.data = strategy.strategy_type
        form.slow_ma_period.data = strategy.slow_ma_period
        form.fast_ma_period.data = strategy.fast_ma_period
        form.rsi_period.data = strategy.rsi_period
        form.rsi_overbought.data = strategy.rsi_overbought
        form.rsi_oversold.data = strategy.rsi_oversold
        form.macd_fast_ema.data = strategy.macd_fast_ema
        form.macd_slow_ema.data = strategy.macd_slow_ema
        form.macd_signal_period.data = strategy.macd_signal_period
        form.risk_per_trade.data = strategy.risk_per_trade
        form.max_risk_total.data = strategy.max_risk_total
        form.use_atr_for_sl.data = strategy.use_atr_for_sl
        form.atr_period.data = strategy.atr_period
        form.atr_multiplier.data = strategy.atr_multiplier
        form.fixed_sl_pips.data = strategy.fixed_sl_pips
        form.fixed_tp_pips.data = strategy.fixed_tp_pips
        form.risk_reward_ratio.data = strategy.risk_reward_ratio
        
        return render_template('strategy_form.html', title='Edit Strategy', form=form, strategy=strategy)
    
    # Run backtest
    @app.route('/backtest/run/<int:strategy_id>', methods=['GET', 'POST'])
    @login_required
    def run_backtest(strategy_id):
        strategy = Strategy.query.get_or_404(strategy_id)
        if strategy.user_id != current_user.id:
            abort(403)  # Forbidden
        
        # Simple form for backtest parameters
        if request.method == 'POST':
            symbol = request.form.get('symbol', 'EURUSD')
            timeframe = request.form.get('timeframe', 'H1')
            start_date = request.form.get('start_date')
            end_date = request.form.get('end_date')
            initial_balance = float(request.form.get('initial_balance', 10000))
            
            # Create strategy instance based on type
            if strategy.strategy_type == 'MA_RSI':
                strategy_instance = MovingAverageRSIStrategy(
                    symbol=symbol,
                    timeframe=timeframe,
                    slow_ma_period=strategy.slow_ma_period,
                    fast_ma_period=strategy.fast_ma_period,
                    rsi_period=strategy.rsi_period,
                    rsi_overbought=strategy.rsi_overbought,
                    rsi_oversold=strategy.rsi_oversold
                )
            elif strategy.strategy_type == 'MACD':
                strategy_instance = MACDStrategy(
                    symbol=symbol,
                    timeframe=timeframe,
                    fast_ema=strategy.macd_fast_ema,
                    slow_ema=strategy.macd_slow_ema,
                    signal_period=strategy.macd_signal_period
                )
            else:
                flash('Unknown strategy type', 'danger')
                return redirect(url_for('strategy_detail', id=strategy_id))
            
            try:
                # Import here to avoid circular imports
                from backtester import Backtester
                
                # Run backtest
                backtester = Backtester(
                    strategy_instance=strategy_instance,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    initial_balance=initial_balance
                )
                
                results = backtester.run()
                
                if not results['success']:
                    flash(f"Backtest failed: {results.get('error', 'Unknown error')}", 'danger')
                    return redirect(url_for('strategy_detail', id=strategy_id))
                
                # Save results to database
                backtest_result = BacktestResult(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=datetime.strptime(start_date, '%Y-%m-%d'),
                    end_date=datetime.strptime(end_date, '%Y-%m-%d'),
                    initial_balance=initial_balance,
                    final_balance=results['final_balance'],
                    total_trades=results['total_trades'],
                    profitable_trades=results['profitable_trades'],
                    losing_trades=results['losing_trades'],
                    win_rate=results['win_rate'],
                    profit_factor=results['profit_factor'],
                    max_drawdown=results['max_drawdown'],
                    max_drawdown_percent=results['max_drawdown_percent'],
                    sharpe_ratio=results['sharpe_ratio'],
                    equity_curve=str(results['equity_curve']),  # Convert to string for storage
                    trades_data=str(results['trades']),  # Convert to string for storage
                    user_id=current_user.id,
                    strategy_id=strategy_id
                )
                
                db.session.add(backtest_result)
                db.session.commit()
                
                flash('Backtest completed successfully!', 'success')
                return redirect(url_for('backtest_detail', id=backtest_result.id))
                
            except Exception as e:
                logger.error(f"Backtest error: {str(e)}")
                flash(f"Error running backtest: {str(e)}", 'danger')
                return redirect(url_for('strategy_detail', id=strategy_id))
        
        return render_template(
            'backtest_form.html',
            title='Run Backtest',
            strategy=strategy
        )
    
    # Backtest detail
    @app.route('/backtest/<int:id>')
    @login_required
    def backtest_detail(id):
        backtest = BacktestResult.query.get_or_404(id)
        if backtest.user_id != current_user.id:
            abort(403)  # Forbidden
        
        return render_template(
            'backtest_detail.html',
            title='Backtest Results',
            backtest=backtest
        )
    
    # API route for getting strategy data (for AJAX)
    @app.route('/api/strategy/<int:id>')
    @login_required
    def api_strategy(id):
        strategy = Strategy.query.get_or_404(id)
        if strategy.user_id != current_user.id:
            abort(403)  # Forbidden
        
        return jsonify({
            'id': strategy.id,
            'name': strategy.name,
            'description': strategy.description,
            'strategy_type': strategy.strategy_type,
            'created_at': strategy.created_at.isoformat(),
            'updated_at': strategy.updated_at.isoformat(),
            'params': {
                'slow_ma_period': strategy.slow_ma_period,
                'fast_ma_period': strategy.fast_ma_period,
                'rsi_period': strategy.rsi_period,
                'rsi_overbought': strategy.rsi_overbought,
                'rsi_oversold': strategy.rsi_oversold,
                'macd_fast_ema': strategy.macd_fast_ema,
                'macd_slow_ema': strategy.macd_slow_ema,
                'macd_signal_period': strategy.macd_signal_period,
                'risk_per_trade': strategy.risk_per_trade,
                'max_risk_total': strategy.max_risk_total,
                'use_atr_for_sl': strategy.use_atr_for_sl,
                'atr_period': strategy.atr_period,
                'atr_multiplier': strategy.atr_multiplier,
                'fixed_sl_pips': strategy.fixed_sl_pips,
                'fixed_tp_pips': strategy.fixed_tp_pips,
                'risk_reward_ratio': strategy.risk_reward_ratio
            }
        })
    
    # API route for getting backtest data (for AJAX charts)
    @app.route('/api/backtest/<int:id>')
    @login_required
    def api_backtest(id):
        backtest = BacktestResult.query.get_or_404(id)
        if backtest.user_id != current_user.id:
            abort(403)  # Forbidden
        
        # Note: In a production app, we would parse the JSON/list data correctly
        return jsonify({
            'id': backtest.id,
            'symbol': backtest.symbol,
            'timeframe': backtest.timeframe,
            'start_date': backtest.start_date.isoformat(),
            'end_date': backtest.end_date.isoformat(),
            'initial_balance': backtest.initial_balance,
            'final_balance': backtest.final_balance,
            'total_trades': backtest.total_trades,
            'profitable_trades': backtest.profitable_trades,
            'losing_trades': backtest.losing_trades,
            'win_rate': backtest.win_rate,
            'profit_factor': backtest.profit_factor,
            'max_drawdown': backtest.max_drawdown,
            'max_drawdown_percent': backtest.max_drawdown_percent,
            'sharpe_ratio': backtest.sharpe_ratio,
            'created_at': backtest.created_at.isoformat(),
            'equity_curve': backtest.equity_curve,  # Would need proper parsing
            'trades': backtest.trades_data  # Would need proper parsing
        })
    
    # AI Trading Routes
    @app.route('/ai_trading')
    @login_required
    def ai_trading():
        """AI Trading dashboard"""
        trading_status = trading_engine.get_status()
        account_info = mt5_bot.get_account_info() if mt5_bot.is_connected else None
        performance = mt5_bot.get_performance_report()
        
        return render_template('ai_trading.html', 
                             title='AI Trading',
                             trading_status=trading_status,
                             account_info=account_info,
                             performance=performance)
    
    @app.route('/ai_trading/connect', methods=['GET', 'POST'])
    @login_required
    def ai_trading_connect():
        """Connect to MT5 and start AI trading"""
        form = MT5ConnectionForm()
        
        if form.validate_on_submit():
            try:
                # Parse symbols and timeframes
                symbols = [s.strip().upper() for s in form.symbols.data.split(',')]
                timeframes = [t.strip().upper() for t in form.timeframes.data.split(',')]
                
                # Update trading engine settings
                trading_engine.update_symbols(symbols)
                trading_engine.update_timeframes(timeframes)
                trading_engine.update_check_interval(form.check_interval.data)
                
                # Account configuration
                account_config = {
                    'login': int(form.login.data) if form.login.data.isdigit() else None,
                    'password': form.password.data,
                    'server': form.server.data
                }
                
                # Start trading
                if trading_engine.start_trading(account_config):
                    flash('AI Trading started successfully!', 'success')
                    return redirect(url_for('ai_trading'))
                else:
                    flash('Failed to start AI trading. Check your MT5 connection.', 'danger')
                    
            except Exception as e:
                logger.error(f"Error starting AI trading: {e}")
                flash('Error starting AI trading. Please check your settings.', 'danger')
        
        return render_template('ai_trading_connect.html', 
                             title='Connect to MT5',
                             form=form)
    
    @app.route('/ai_trading/stop')
    @login_required
    def ai_trading_stop():
        """Stop AI trading"""
        trading_engine.stop_trading()
        flash('AI Trading stopped.', 'info')
        return redirect(url_for('ai_trading'))
    
    @app.route('/ai_trading/status')
    @login_required
    def ai_trading_status():
        """Get AI trading status (AJAX)"""
        return jsonify(trading_engine.get_status())
    
    @app.route('/ai_trading/performance')
    @login_required
    def ai_trading_performance():
        """Get AI trading performance (AJAX)"""
        return jsonify(mt5_bot.get_performance_report())
    
    @app.route('/ai_trading/strategies')
    @login_required
    def ai_trading_strategies():
        """Manage AI trading strategies"""
        user_strategies = Strategy.query.filter_by(user_id=current_user.id).all()
        active_strategies = trading_engine.active_strategies
        
        return render_template('ai_trading_strategies.html',
                             title='AI Trading Strategies',
                             user_strategies=user_strategies,
                             active_strategies=active_strategies)
    
    @app.route('/ai_trading/strategies/add/<int:strategy_id>')
    @login_required
    def ai_trading_add_strategy(strategy_id):
        """Add strategy to AI trading"""
        strategy = Strategy.query.get_or_404(strategy_id)
        if strategy.user_id != current_user.id:
            abort(403)
        
        if trading_engine.add_strategy(strategy_id):
            flash(f'Strategy "{strategy.name}" added to AI trading.', 'success')
        else:
            flash('Failed to add strategy to AI trading.', 'danger')
        
        return redirect(url_for('ai_trading_strategies'))
    
    @app.route('/ai_trading/strategies/remove/<int:strategy_id>')
    @login_required
    def ai_trading_remove_strategy(strategy_id):
        """Remove strategy from AI trading"""
        strategy = Strategy.query.get_or_404(strategy_id)
        if strategy.user_id != current_user.id:
            abort(403)
        
        if trading_engine.remove_strategy(strategy_id):
            flash(f'Strategy "{strategy.name}" removed from AI trading.', 'success')
        else:
            flash('Failed to remove strategy from AI trading.', 'danger')
        
        return redirect(url_for('ai_trading_strategies'))
    
    @app.route('/ai_trading/trades')
    @login_required
    def ai_trading_trades():
        """View AI trading trades"""
        trades = Trade.query.filter_by(user_id=current_user.id).order_by(Trade.open_time.desc()).limit(100).all()
        positions = mt5_bot.get_positions() if mt5_bot.is_connected else []
        
        return render_template('ai_trading_trades.html',
                             title='AI Trading Trades',
                             trades=trades,
                             positions=positions)
    
    # Built-in Strategies Showcase
    @app.route('/builtin_strategies')
    @login_required
    def builtin_strategies():
        """Show available built-in strategies"""
        strategies = builtin_strategy_manager.get_strategy_list()
        return render_template('builtin_strategies.html',
                             title='Built-in Trading Strategies',
                             strategies=strategies)
    
    @app.route('/builtin_strategies/preview/<strategy_type>')
    @login_required
    def preview_builtin_strategy(strategy_type):
        """Preview a built-in strategy"""
        strategy = builtin_strategy_manager.get_strategy(strategy_type)
        if not strategy:
            flash('Strategy not found', 'error')
            return redirect(url_for('builtin_strategies'))
        
        # Generate sample data for preview
        sample_data = utils.generate_sample_data()
        signals = strategy.generate_signals(sample_data)
        
        return render_template('strategy_preview.html',
                             title=f'Preview: {strategy.name}',
                             strategy=strategy,
                             signals=signals[-50:])  # Show last 50 signals
    
    @app.route('/builtin_strategies/create/<strategy_type>', methods=['GET', 'POST'])
    @login_required
    def create_builtin_strategy(strategy_type):
        """Create a new strategy from built-in template"""
        builtin_strategy = builtin_strategy_manager.get_strategy(strategy_type)
        if not builtin_strategy:
            flash('Strategy template not found', 'error')
            return redirect(url_for('builtin_strategies'))
        
        if request.method == 'POST':
            # Create new strategy from built-in template
            strategy = Strategy(
                name=request.form.get('name', builtin_strategy.name),
                description=request.form.get('description', f'Built-in {builtin_strategy.name} strategy'),
                strategy_type=strategy_type,
                user_id=current_user.id,
                # Set reasonable defaults for built-in strategies
                slow_ma_period=50,
                fast_ma_period=20,
                rsi_period=14,
                rsi_overbought=70,
                rsi_oversold=30,
                macd_fast_ema=12,
                macd_slow_ema=26,
                macd_signal_period=9,
                risk_per_trade=1.0,
                max_risk_total=5.0,
                use_atr_for_sl=True,
                atr_period=14,
                atr_multiplier=2.0,
                fixed_sl_pips=50,
                fixed_tp_pips=100,
                risk_reward_ratio=2.0
            )
            db.session.add(strategy)
            db.session.commit()
            flash(f'Strategy "{strategy.name}" created successfully!', 'success')
            return redirect(url_for('strategy_detail', id=strategy.id))
        
        return render_template('create_builtin_strategy.html',
                             title=f'Create {builtin_strategy.name}',
                             strategy=builtin_strategy)