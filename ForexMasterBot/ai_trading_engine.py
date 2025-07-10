"""
AI Trading Engine for MT5 Forex Trading Bot

This module integrates AI models with the MT5 trading system to provide
intelligent trading decisions and automated execution.
"""

import logging
import threading
import time
import schedule
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from mt5_connector import mt5_bot
from ai_models import ModelManager, AITradingAdvisor, AIMarketAnalyst  
from ai_strategy_manager import AIStrategyManager
from strategy import MovingAverageRSIStrategy, MACDStrategy
from risk_manager import RiskManager
from models import Strategy, TradingAccount, Trade, db

logger = logging.getLogger(__name__)

class AITradingEngine:
    """
    AI-powered trading engine that combines multiple AI models with MetaTrader 5
    """
    
    def __init__(self):
        self.mt5_bot = mt5_bot
        self.model_manager = ModelManager()
        self.ai_advisor = AITradingAdvisor()
        self.ai_analyst = AIMarketAnalyst()
        self.ai_strategy_manager = AIStrategyManager()
        self.risk_manager = RiskManager()
        
        self.is_running = False
        self.trading_thread = None
        self.active_strategies = {}
        self.performance_tracking = {}
        
        # Trading settings
        self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        self.timeframes = ['H1', 'H4']
        self.check_interval = 300  # 5 minutes
        
        logger.info("AI Trading Engine initialized")
    
    def start_trading(self, account_config: Dict) -> bool:
        """
        Start automated trading
        
        Args:
            account_config: MT5 account configuration
            
        Returns:
            bool: True if trading started successfully
        """
        if self.is_running:
            logger.warning("Trading engine is already running")
            return False
        
        try:
            # Connect to MT5
            if not self.mt5_bot.connect(
                login=account_config.get('login'),
                password=account_config.get('password'),
                server=account_config.get('server')
            ):
                logger.error("Failed to connect to MT5")
                return False
            
            # Load active strategies
            self._load_active_strategies()
            
            # Start trading thread
            self.is_running = True
            self.trading_thread = threading.Thread(target=self._trading_loop)
            self.trading_thread.daemon = True
            self.trading_thread.start()
            
            logger.info("AI Trading Engine started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting trading engine: {e}")
            return False
    
    def stop_trading(self):
        """Stop automated trading"""
        if not self.is_running:
            logger.warning("Trading engine is not running")
            return
        
        self.is_running = False
        if self.trading_thread:
            self.trading_thread.join(timeout=10)
        
        self.mt5_bot.disconnect()
        logger.info("AI Trading Engine stopped")
    
    def _load_active_strategies(self):
        """Load active trading strategies from database"""
        try:
            # Get all strategies from database
            strategies = Strategy.query.all()
            
            for strategy in strategies:
                strategy_config = {
                    'id': strategy.id,
                    'name': strategy.name,
                    'type': strategy.strategy_type,
                    'user_id': strategy.user_id,
                    'parameters': {
                        'slow_ma_period': strategy.slow_ma_period,
                        'fast_ma_period': strategy.fast_ma_period,
                        'rsi_period': strategy.rsi_period,
                        'rsi_overbought': strategy.rsi_overbought,
                        'rsi_oversold': strategy.rsi_oversold,
                        'risk_per_trade': strategy.risk_per_trade,
                        'max_risk_total': strategy.max_risk_total
                    }
                }
                
                # Create strategy instance
                if strategy.strategy_type == 'MA_RSI':
                    strategy_instance = MovingAverageRSIStrategy(
                        symbol='EURUSD',  # Default symbol, will be overridden per trade
                        timeframe='H1',   # Default timeframe, will be overridden per trade
                        slow_ma_period=strategy.slow_ma_period,
                        fast_ma_period=strategy.fast_ma_period,
                        rsi_period=strategy.rsi_period,
                        rsi_overbought=strategy.rsi_overbought,
                        rsi_oversold=strategy.rsi_oversold
                    )
                elif strategy.strategy_type == 'MACD':
                    strategy_instance = MACDStrategy(
                        symbol='EURUSD',  # Default symbol, will be overridden per trade
                        timeframe='H1',   # Default timeframe, will be overridden per trade
                        fast_ema=strategy.macd_fast_ema,
                        slow_ema=strategy.macd_slow_ema,
                        signal_period=strategy.macd_signal_period
                    )
                elif strategy.strategy_type in ['TREND_CROSSOVER', 'BREAKOUT', 'MEAN_REVERSION', 'SCALPING', 'MULTI_SIGNAL']:
                    # Built-in strategies
                    from builtin_strategies import builtin_strategy_manager
                    strategy_instance = builtin_strategy_manager.get_strategy(strategy.strategy_type)
                    if not strategy_instance:
                        logger.warning(f"Built-in strategy not found: {strategy.strategy_type}")
                        continue
                else:
                    logger.warning(f"Unknown strategy type: {strategy.strategy_type}")
                    continue
                
                strategy_config['instance'] = strategy_instance
                self.active_strategies[strategy.id] = strategy_config
                
            logger.info(f"Loaded {len(self.active_strategies)} active strategies")
            
        except Exception as e:
            logger.error(f"Error loading strategies: {e}")
    
    def _trading_loop(self):
        """Main trading loop"""
        logger.info("Starting trading loop")
        
        while self.is_running:
            try:
                # Check each symbol and timeframe
                for symbol in self.symbols:
                    for timeframe in self.timeframes:
                        self._analyze_and_trade(symbol, timeframe)
                
                # Wait before next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _analyze_and_trade(self, symbol: str, timeframe: str):
        """
        Analyze market and execute trades for a specific symbol/timeframe
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
        """
        try:
            # Get market data
            market_data = self.mt5_bot.get_market_data(symbol, timeframe, 200)
            if market_data is None or len(market_data) < 50:
                logger.warning(f"Insufficient data for {symbol} {timeframe}")
                return
            
            # Analyze with each active strategy
            for strategy_id, strategy_config in self.active_strategies.items():
                try:
                    # Get traditional signal
                    strategy_instance = strategy_config['instance']
                    traditional_signal = strategy_instance.generate_signal(market_data)
                    
                    # Get AI-enhanced signal
                    indicators = strategy_instance._calculate_indicators(market_data)
                    ai_signal = self.ai_strategy_manager.get_ai_enhanced_signal(
                        strategy_type=strategy_config['type'],
                        symbol=symbol,
                        timeframe=timeframe,
                        market_data=market_data,
                        indicators=indicators,
                        traditional_signal=traditional_signal
                    )
                    
                    # Execute trades based on AI signal
                    self._execute_ai_signal(symbol, strategy_config, ai_signal)
                    
                except Exception as e:
                    logger.error(f"Error analyzing strategy {strategy_id}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error analyzing {symbol} {timeframe}: {e}")
    
    def _execute_ai_signal(self, symbol: str, strategy_config: Dict, ai_signal: Dict):
        """
        Execute trading based on AI signal
        
        Args:
            symbol: Trading symbol
            strategy_config: Strategy configuration
            ai_signal: AI signal data
        """
        try:
            signal = ai_signal.get('signal', 'HOLD')
            confidence = ai_signal.get('confidence', 0.0)
            
            # Only trade if confidence is high enough
            if confidence < 0.6:
                logger.debug(f"Signal confidence too low: {confidence}")
                return
            
            # Check if we already have a position for this symbol/strategy
            existing_positions = self.mt5_bot.get_positions(symbol)
            strategy_positions = [pos for pos in existing_positions 
                                if f"Strategy_{strategy_config['id']}" in pos.get('comment', '')]
            
            if signal == 'BUY' and not strategy_positions:
                self._place_buy_order(symbol, strategy_config, ai_signal)
            elif signal == 'SELL' and not strategy_positions:
                self._place_sell_order(symbol, strategy_config, ai_signal)
            
        except Exception as e:
            logger.error(f"Error executing AI signal: {e}")
    
    def _place_buy_order(self, symbol: str, strategy_config: Dict, ai_signal: Dict):
        """Place a buy order"""
        try:
            # Calculate position size
            account_info = self.mt5_bot.get_account_info()
            if not account_info:
                logger.error("Failed to get account info")
                return
            
            volume = self.risk_manager.calculate_position_size(
                balance=account_info['balance'],
                risk_percent=strategy_config['parameters']['risk_per_trade'],
                symbol=symbol,
                stop_loss_pips=50  # Default SL
            )
            
            # Get current price for SL/TP calculation
            market_data = self.mt5_bot.get_market_data(symbol, 'M1', 1)
            if market_data is None:
                return
            
            current_price = market_data['close'].iloc[-1]
            
            # Calculate SL/TP
            sl_distance = 50 * 0.0001  # 50 pips (simplified)
            tp_distance = 100 * 0.0001  # 100 pips (simplified)
            
            stop_loss = current_price - sl_distance
            take_profit = current_price + tp_distance
            
            # Place order
            result = self.mt5_bot.send_order(
                symbol=symbol,
                order_type='BUY',
                volume=volume,
                sl=stop_loss,
                tp=take_profit,
                comment=f"AI_Strategy_{strategy_config['id']}"
            )
            
            if result and result.get('success'):
                logger.info(f"Buy order placed: {symbol} {volume} lots")
                self._record_trade(symbol, 'BUY', volume, current_price, 
                                strategy_config, ai_signal, result)
            else:
                logger.error(f"Failed to place buy order for {symbol}")
                
        except Exception as e:
            logger.error(f"Error placing buy order: {e}")
    
    def _place_sell_order(self, symbol: str, strategy_config: Dict, ai_signal: Dict):
        """Place a sell order"""
        try:
            # Calculate position size
            account_info = self.mt5_bot.get_account_info()
            if not account_info:
                logger.error("Failed to get account info")
                return
            
            volume = self.risk_manager.calculate_position_size(
                balance=account_info['balance'],
                risk_percent=strategy_config['parameters']['risk_per_trade'],
                symbol=symbol,
                stop_loss_pips=50  # Default SL
            )
            
            # Get current price for SL/TP calculation
            market_data = self.mt5_bot.get_market_data(symbol, 'M1', 1)
            if market_data is None:
                return
            
            current_price = market_data['close'].iloc[-1]
            
            # Calculate SL/TP
            sl_distance = 50 * 0.0001  # 50 pips (simplified)
            tp_distance = 100 * 0.0001  # 100 pips (simplified)
            
            stop_loss = current_price + sl_distance
            take_profit = current_price - tp_distance
            
            # Place order
            result = self.mt5_bot.send_order(
                symbol=symbol,
                order_type='SELL',
                volume=volume,
                sl=stop_loss,
                tp=take_profit,
                comment=f"AI_Strategy_{strategy_config['id']}"
            )
            
            if result and result.get('success'):
                logger.info(f"Sell order placed: {symbol} {volume} lots")
                self._record_trade(symbol, 'SELL', volume, current_price, 
                                strategy_config, ai_signal, result)
            else:
                logger.error(f"Failed to place sell order for {symbol}")
                
        except Exception as e:
            logger.error(f"Error placing sell order: {e}")
    
    def _record_trade(self, symbol: str, order_type: str, volume: float, 
                     price: float, strategy_config: Dict, ai_signal: Dict, result: Dict):
        """Record trade in database"""
        try:
            # Find a trading account for this user
            account = TradingAccount.query.filter_by(user_id=strategy_config['user_id']).first()
            if not account:
                logger.warning(f"No trading account found for user {strategy_config['user_id']}")
                return
            
            trade = Trade(
                ticket=str(result.get('ticket', '')),
                symbol=symbol,
                trade_type=order_type,
                volume=volume,
                open_price=price,
                open_time=datetime.now(),
                status='OPEN',
                comment=f"AI Strategy {strategy_config['name']} - Confidence: {ai_signal.get('confidence', 0):.2f}",
                user_id=strategy_config['user_id'],
                account_id=account.id
            )
            
            db.session.add(trade)
            db.session.commit()
            
            logger.info(f"Trade recorded: {symbol} {order_type} {volume}")
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    def get_status(self) -> Dict:
        """Get current trading engine status"""
        return {
            'is_running': self.is_running,
            'connected_to_mt5': self.mt5_bot.is_connected,
            'simulation_mode': self.mt5_bot.simulation_mode,
            'active_strategies': len(self.active_strategies),
            'symbols': self.symbols,
            'timeframes': self.timeframes,
            'check_interval': self.check_interval,
            'performance': self.mt5_bot.get_performance_report()
        }
    
    def add_strategy(self, strategy_id: int):
        """Add a strategy to active trading"""
        try:
            strategy = Strategy.query.get(strategy_id)
            if not strategy:
                logger.error(f"Strategy {strategy_id} not found")
                return False
            
            # Load strategy configuration
            strategy_config = {
                'id': strategy.id,
                'name': strategy.name,
                'type': strategy.strategy_type,
                'user_id': strategy.user_id,
                'parameters': {
                    'slow_ma_period': strategy.slow_ma_period,
                    'fast_ma_period': strategy.fast_ma_period,
                    'rsi_period': strategy.rsi_period,
                    'rsi_overbought': strategy.rsi_overbought,
                    'rsi_oversold': strategy.rsi_oversold,
                    'risk_per_trade': strategy.risk_per_trade,
                    'max_risk_total': strategy.max_risk_total
                }
            }
            
            # Create strategy instance
            if strategy.strategy_type == 'MA_RSI':
                strategy_instance = MovingAverageRSIStrategy(
                    slow_ma_period=strategy.slow_ma_period,
                    fast_ma_period=strategy.fast_ma_period,
                    rsi_period=strategy.rsi_period,
                    rsi_overbought=strategy.rsi_overbought,
                    rsi_oversold=strategy.rsi_oversold
                )
            elif strategy.strategy_type == 'MACD':
                strategy_instance = MACDStrategy(
                    fast_ema=strategy.macd_fast_ema,
                    slow_ema=strategy.macd_slow_ema,
                    signal_period=strategy.macd_signal_period
                )
            else:
                logger.error(f"Unknown strategy type: {strategy.strategy_type}")
                return False
            
            strategy_config['instance'] = strategy_instance
            self.active_strategies[strategy.id] = strategy_config
            
            logger.info(f"Strategy {strategy_id} added to active trading")
            return True
            
        except Exception as e:
            logger.error(f"Error adding strategy {strategy_id}: {e}")
            return False
    
    def remove_strategy(self, strategy_id: int):
        """Remove a strategy from active trading"""
        if strategy_id in self.active_strategies:
            del self.active_strategies[strategy_id]
            logger.info(f"Strategy {strategy_id} removed from active trading")
            return True
        return False
    
    def update_symbols(self, symbols: List[str]):
        """Update trading symbols"""
        self.symbols = symbols
        logger.info(f"Trading symbols updated: {symbols}")
    
    def update_timeframes(self, timeframes: List[str]):
        """Update trading timeframes"""
        self.timeframes = timeframes
        logger.info(f"Trading timeframes updated: {timeframes}")
    
    def update_check_interval(self, interval: int):
        """Update check interval in seconds"""
        self.check_interval = interval
        logger.info(f"Check interval updated: {interval} seconds")

# Global trading engine instance
trading_engine = AITradingEngine()