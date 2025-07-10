"""
Enhanced Trading Strategy Module
Extends the base strategy module with AI capabilities

This module provides AI-enhanced versions of the trading strategies.
It uses multiple AI models and market analysis to improve trading decisions.
"""

import logging
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Import original strategy classes
from strategy import MovingAverageRSIStrategy, MACDStrategy

# Import AI modules
from ai_strategy_manager import ai_strategy_manager
from builtin_strategies import builtin_strategy_manager

# Configure logging
logger = logging.getLogger(__name__)

class AIEnhancedStrategyBase:
    """Base class for AI-enhanced strategies"""
    
    def __init__(self, traditional_strategy, use_ai=True):
        """
        Initialize the AI enhanced strategy
        
        Args:
            traditional_strategy: The base strategy to enhance
            use_ai (bool): Whether to use AI enhancement
        """
        self.traditional_strategy = traditional_strategy
        self.use_ai = use_ai
        self.symbol = traditional_strategy.symbol
        self.timeframe = traditional_strategy.timeframe
        self.strategy_type = self._determine_strategy_type()
        
        # Initialize learning metrics
        self.learning_metrics = {
            'signals_generated': 0,
            'ai_signals_used': 0,
            'signal_agreement_rate': 0,
            'last_signals': []
        }
        
        logger.info(f"AI Enhanced Strategy initialized for {self.symbol} (type: {self.strategy_type})")
    
    def _determine_strategy_type(self):
        """Determine the type of the base strategy"""
        if isinstance(self.traditional_strategy, MovingAverageRSIStrategy):
            return 'MA_RSI'
        elif isinstance(self.traditional_strategy, MACDStrategy):
            return 'MACD'
        else:
            return 'UNKNOWN'
    
    def _calculate_indicators(self, data):
        """
        Calculate indicators for the strategy and prepare data for AI models
        
        Args:
            data (DataFrame): Market data with OHLC prices
            
        Returns:
            tuple: (DataFrame with indicators, dict of indicator values)
        """
        # Calculate indicators using traditional strategy
        enhanced_data = self.traditional_strategy._calculate_indicators(data.copy())
        
        # Extract key indicators for AI analysis
        indicators = {}
        
        # Common indicators
        if 'atr' in enhanced_data.columns:
            indicators['atr'] = enhanced_data.iloc[-1]['atr']
            
        # Different indicators based on strategy type
        if self.strategy_type == 'MA_RSI':
            indicators['slow_ma'] = enhanced_data.iloc[-1]['slow_ma']
            indicators['fast_ma'] = enhanced_data.iloc[-1]['fast_ma']
            indicators['rsi'] = enhanced_data.iloc[-1]['rsi']
            
            # Calculate additional features for AI models
            enhanced_data['slow_ma_diff'] = enhanced_data['slow_ma'].diff()
            enhanced_data['fast_ma_diff'] = enhanced_data['fast_ma'].diff()
            enhanced_data['rsi_diff'] = enhanced_data['rsi'].diff()
            
            indicators['slow_ma_diff'] = enhanced_data.iloc[-1]['slow_ma_diff']
            indicators['fast_ma_diff'] = enhanced_data.iloc[-1]['fast_ma_diff']
            indicators['rsi_diff'] = enhanced_data.iloc[-1]['rsi_diff']
            
        elif self.strategy_type == 'MACD':
            indicators['macd'] = enhanced_data.iloc[-1]['macd']
            indicators['macd_signal'] = enhanced_data.iloc[-1]['macd_signal']
            indicators['macd_hist'] = enhanced_data.iloc[-1]['macd_hist']
            
            # Calculate additional features for AI models
            enhanced_data['macd_diff'] = enhanced_data['macd'].diff()
            enhanced_data['macd_signal_diff'] = enhanced_data['macd_signal'].diff()
            enhanced_data['macd_hist_diff'] = enhanced_data['macd_hist'].diff()
            
            indicators['macd_diff'] = enhanced_data.iloc[-1]['macd_diff']
            indicators['macd_signal_diff'] = enhanced_data.iloc[-1]['macd_signal_diff']
            indicators['macd_hist_diff'] = enhanced_data.iloc[-1]['macd_hist_diff']
        
        # Add more complex features (these help AI models learn patterns)
        enhanced_data['daily_range'] = enhanced_data['high'] - enhanced_data['low']
        enhanced_data['body_size'] = abs(enhanced_data['close'] - enhanced_data['open'])
        enhanced_data['upper_shadow'] = enhanced_data.apply(
            lambda x: x['high'] - max(x['open'], x['close']), axis=1)
        enhanced_data['lower_shadow'] = enhanced_data.apply(
            lambda x: min(x['open'], x['close']) - x['low'], axis=1)
        
        # Volume indicators
        if 'volume' in enhanced_data.columns:
            enhanced_data['volume_ma'] = enhanced_data['volume'].rolling(window=20).mean()
            enhanced_data['volume_ratio'] = enhanced_data['volume'] / enhanced_data['volume_ma']
            
            indicators['volume'] = enhanced_data.iloc[-1]['volume']
            indicators['volume_ratio'] = enhanced_data.iloc[-1].get('volume_ratio', 1.0)
            
        # Market timing features
        enhanced_data['hour'] = pd.to_datetime(enhanced_data.index).hour.astype(float)
        enhanced_data['day_of_week'] = pd.to_datetime(enhanced_data.index).dayofweek.astype(float)
        
        indicators['daily_range'] = enhanced_data.iloc[-1]['daily_range']
        
        return enhanced_data, indicators
    
    def _map_signal_to_str(self, signal):
        """Convert numerical signal to string format"""
        if signal > 0:
            return 'BUY'
        elif signal < 0:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _map_str_to_signal(self, signal_str):
        """Convert string signal to numerical format"""
        if signal_str == 'BUY':
            return 1
        elif signal_str == 'SELL':
            return -1
        else:
            return 0
    
    def _prepare_for_ai_training(self, data):
        """
        Prepare data for AI model training
        
        Args:
            data (DataFrame): Historical market data
            
        Returns:
            DataFrame: Data prepared for training
        """
        # Calculate indicators and add features
        enhanced_data, _ = self._calculate_indicators(data)
        
        # Generate target variable (trading signal) using traditional strategy
        # We'll shift it to align current indicators with future signals
        signals = []
        
        for i in range(len(enhanced_data) - 10):  # Skip last 10 rows for prediction window
            window = enhanced_data.iloc[i:i+20]  # Use a window of data for signal calculation
            if len(window) < 20:
                signals.append(0)  # Not enough data
                continue
                
            signal = self.traditional_strategy.calculate_signal(window)
            signals.append(signal)
        
        # Pad with zeros for the skipped rows
        signals.extend([0] * 10)
        
        # Add target variable
        enhanced_data['signal'] = signals
        
        # Shift target to align current indicators with future signals (predictive)
        enhanced_data['signal'] = enhanced_data['signal'].shift(-5)  # Predict 5 candles ahead
        
        # Drop rows with NaN values
        enhanced_data = enhanced_data.dropna()
        
        return enhanced_data
    
    def train_ai_models(self, historical_data):
        """
        Train AI models for this strategy
        
        Args:
            historical_data (DataFrame): Historical market data
            
        Returns:
            dict: Training results
        """
        logger.info(f"Training AI models for {self.strategy_type} strategy on {self.symbol}")
        
        # Prepare data for training
        training_data = self._prepare_for_ai_training(historical_data)
        
        # Log training data metrics
        logger.info(f"Prepared {len(training_data)} samples for training with {training_data.columns.tolist()} features")
        
        # Train models
        results = ai_strategy_manager.train_models(self.strategy_type, training_data)
        
        return results
    
    def calculate_signal(self, data):
        """
        Calculate trading signal with AI enhancement
        
        Args:
            data (DataFrame): Market data with OHLC prices
            
        Returns:
            int: Signal (1 for buy, -1 for sell, 0 for no action)
        """
        # Calculate traditional signal
        traditional_signal = self.traditional_strategy.calculate_signal(data)
        
        # If AI is disabled, return traditional signal only
        if not self.use_ai:
            return traditional_signal
        
        # Enhanced data for AI
        enhanced_data, indicators = self._calculate_indicators(data)
        
        # Convert numerical signal to string format
        traditional_signal_str = self._map_signal_to_str(traditional_signal)
        
        try:
            # Get AI-enhanced signal
            ai_result = ai_strategy_manager.get_ai_enhanced_signal(
                strategy_type=self.strategy_type,
                symbol=self.symbol,
                timeframe=str(self.timeframe),
                market_data=enhanced_data,
                indicators=indicators,
                traditional_signal=traditional_signal_str
            )
            
            # Update learning metrics
            self._update_metrics(traditional_signal_str, ai_result)
            
            # Get final signal
            final_signal_str = ai_result.get('signal', traditional_signal_str)
            final_signal = self._map_str_to_signal(final_signal_str)
            
            # Log signal information
            logger.info(f"Signal for {self.symbol}: Traditional={traditional_signal_str}, "
                       f"AI={ai_result.get('ai_signal')}, Final={final_signal_str}")
            
            return final_signal
            
        except Exception as e:
            logger.error(f"Error getting AI signal: {str(e)}. Using traditional signal.")
            return traditional_signal
    
    def _update_metrics(self, traditional_signal, ai_result):
        """Update learning metrics"""
        self.learning_metrics['signals_generated'] += 1
        
        # Check if AI signal was actually used
        ai_signal = ai_result.get('ai_signal')
        final_signal = ai_result.get('signal')
        
        if ai_signal and final_signal != traditional_signal:
            self.learning_metrics['ai_signals_used'] += 1
        
        # Update agreement rate
        if ai_signal and traditional_signal:
            agreement = ai_signal == traditional_signal
            
            # Calculate running average of agreement rate
            current_count = self.learning_metrics['signals_generated']
            current_rate = self.learning_metrics['signal_agreement_rate']
            new_rate = current_rate + (1 if agreement else 0 - current_rate) / current_count
            self.learning_metrics['signal_agreement_rate'] = new_rate
        
        # Store recent signals
        self.learning_metrics['last_signals'].append({
            'timestamp': datetime.now().isoformat(),
            'traditional': traditional_signal,
            'ai': ai_signal,
            'final': final_signal,
            'confidence': ai_result.get('ai_confidence', 0),
            'sentiment': ai_result.get('market_sentiment'),
            'symbol': self.symbol,
            'timeframe': str(self.timeframe)
        })
        
        # Keep only the last 50 signals
        self.learning_metrics['last_signals'] = self.learning_metrics['last_signals'][-50:]
        
        # Save metrics to file periodically
        if self.learning_metrics['signals_generated'] % 10 == 0:
            self._save_metrics()
    
    def _save_metrics(self):
        """Save learning metrics to file"""
        metrics_dir = 'ai_metrics'
        os.makedirs(metrics_dir, exist_ok=True)
        
        filename = os.path.join(metrics_dir, f"{self.strategy_type}_{self.symbol}_{self.timeframe}.json")
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.learning_metrics, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
    
    def calculate_sl_tp(self, data, signal, entry_price):
        """
        Calculate stop loss and take profit levels with AI enhancement
        
        Args:
            data (DataFrame): Market data with indicators
            signal (int): Trade direction (1 for buy, -1 for sell)
            entry_price (float): Entry price for the trade
            
        Returns:
            tuple: (stop_loss_price, take_profit_price)
        """
        # Use traditional calculation first
        sl, tp = self.traditional_strategy.calculate_sl_tp(data, signal, entry_price)
        
        # If AI is disabled, return traditional levels
        if not self.use_ai:
            return sl, tp
        
        # Enhanced data for AI
        enhanced_data, indicators = self._calculate_indicators(data)
        
        try:
            # For now, we'll just attempt to get market analysis and use key levels
            # This could be expanded with more sophisticated AI models specifically for SL/TP
            market_analysis = ai_strategy_manager.market_analyst.get_full_market_analysis(
                self.symbol, str(self.timeframe), enhanced_data, indicators
            )
            
            if market_analysis.get('success', False):
                key_levels = market_analysis.get('key_levels', {})
                
                # Adjust SL/TP based on key support/resistance levels
                if signal > 0:  # Buy signal
                    # Use closest support level below entry as SL
                    supports = key_levels.get('support', [])
                    if supports:
                        # Filter supports below entry price
                        valid_supports = [s for s in supports if s < entry_price]
                        if valid_supports:
                            # Find closest support
                            ai_sl = max(valid_supports)
                            # Use AI SL if it's not too far from traditional SL
                            if entry_price - ai_sl < (entry_price - sl) * 1.5:
                                sl = ai_sl
                    
                    # Use closest resistance level above entry as TP
                    resistances = key_levels.get('resistance', [])
                    if resistances:
                        # Filter resistances above entry price
                        valid_resistances = [r for r in resistances if r > entry_price]
                        if valid_resistances:
                            # Find closest resistance
                            ai_tp = min(valid_resistances)
                            # Use AI TP if it's reasonable
                            if ai_tp - entry_price > (tp - entry_price) * 0.8:
                                tp = ai_tp
                
                else:  # Sell signal
                    # Use closest resistance level above entry as SL
                    resistances = key_levels.get('resistance', [])
                    if resistances:
                        # Filter resistances above entry price
                        valid_resistances = [r for r in resistances if r > entry_price]
                        if valid_resistances:
                            # Find closest resistance
                            ai_sl = min(valid_resistances)
                            # Use AI SL if it's not too far from traditional SL
                            if ai_sl - entry_price < (sl - entry_price) * 1.5:
                                sl = ai_sl
                    
                    # Use closest support level below entry as TP
                    supports = key_levels.get('support', [])
                    if supports:
                        # Filter supports below entry price
                        valid_supports = [s for s in supports if s < entry_price]
                        if valid_supports:
                            # Find closest support
                            ai_tp = max(valid_supports)
                            # Use AI TP if it's reasonable
                            if entry_price - ai_tp > (entry_price - tp) * 0.8:
                                tp = ai_tp
        
        except Exception as e:
            logger.error(f"Error enhancing SL/TP with AI: {str(e)}. Using traditional levels.")
        
        return sl, tp
    
    def update_position(self, position, data):
        """
        Update an existing position (trailing stop, etc.) with AI enhancement
        
        Args:
            position (dict): Position information
            data (DataFrame): Current market data
            
        Returns:
            bool: True if position was updated, False otherwise
        """
        # Use traditional update logic first
        updated = self.traditional_strategy.update_position(position, data)
        
        # If already updated or AI is disabled, return result
        if updated or not self.use_ai:
            return updated
        
        # TODO: Implement AI-enhanced position management
        # This could include adjusting stop loss based on predicted price movements,
        # partial position closing at key levels, etc.
        
        return False

class AIEnhancedMAStrategy(AIEnhancedStrategyBase):
    """AI-enhanced Moving Average RSI Strategy"""
    
    def __init__(self, symbol, timeframe, slow_ma_period=50, fast_ma_period=20, 
                 rsi_period=14, rsi_overbought=70, rsi_oversold=30, use_ai=True):
        """
        Initialize the AI-enhanced MA-RSI strategy
        
        Args:
            symbol (str): Trading symbol
            timeframe (int): Trading timeframe
            slow_ma_period (int): Period for the slow moving average
            fast_ma_period (int): Period for the fast moving average
            rsi_period (int): Period for RSI calculation
            rsi_overbought (int): RSI level considered overbought
            rsi_oversold (int): RSI level considered oversold
            use_ai (bool): Whether to use AI enhancement
        """
        # Initialize traditional strategy
        traditional_strategy = MovingAverageRSIStrategy(
            symbol, timeframe, slow_ma_period, fast_ma_period, 
            rsi_period, rsi_overbought, rsi_oversold
        )
        
        # Initialize AI-enhanced base
        super().__init__(traditional_strategy, use_ai)

class AIEnhancedMACDStrategy(AIEnhancedStrategyBase):
    """AI-enhanced MACD Strategy"""
    
    def __init__(self, symbol, timeframe, fast_ema=12, slow_ema=26, signal_period=9, use_ai=True):
        """
        Initialize the AI-enhanced MACD strategy
        
        Args:
            symbol (str): Trading symbol
            timeframe (int): Trading timeframe
            fast_ema (int): Fast EMA period for MACD
            slow_ema (int): Slow EMA period for MACD
            signal_period (int): Signal line period for MACD
            use_ai (bool): Whether to use AI enhancement
        """
        # Initialize traditional strategy
        traditional_strategy = MACDStrategy(
            symbol, timeframe, fast_ema, slow_ema, signal_period
        )
        
        # Initialize AI-enhanced base
        super().__init__(traditional_strategy, use_ai)


# Factory function to create appropriate strategy based on type
def create_strategy(strategy_type, symbol, timeframe, params=None, use_ai=True):
    """
    Factory function to create a strategy instance
    
    Args:
        strategy_type (str): Type of strategy ('MA_RSI' or 'MACD')
        symbol (str): Trading symbol
        timeframe (int): Trading timeframe
        params (dict, optional): Strategy parameters
        use_ai (bool): Whether to use AI enhancement
        
    Returns:
        AIEnhancedStrategyBase: Strategy instance
    """
    params = params or {}
    
    if strategy_type == 'MA_RSI':
        return AIEnhancedMAStrategy(
            symbol=symbol,
            timeframe=timeframe,
            slow_ma_period=params.get('slow_ma_period', 50),
            fast_ma_period=params.get('fast_ma_period', 20),
            rsi_period=params.get('rsi_period', 14),
            rsi_overbought=params.get('rsi_overbought', 70),
            rsi_oversold=params.get('rsi_oversold', 30),
            use_ai=use_ai
        )
    elif strategy_type == 'MACD':
        return AIEnhancedMACDStrategy(
            symbol=symbol,
            timeframe=timeframe,
            fast_ema=params.get('macd_fast_ema', 12),
            slow_ema=params.get('macd_slow_ema', 26),
            signal_period=params.get('macd_signal_period', 9),
            use_ai=use_ai
        )
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")