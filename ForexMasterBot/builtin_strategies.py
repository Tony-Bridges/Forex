"""
Built-in Trading Strategies for MT5 Trading Bot

This module contains 5 professional trading strategies that are pre-configured
and ready to use with the AI trading engine.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TrendFollowingCrossoverStrategy:
    """
    Trend-Following Crossover + MACD Confirmation Strategy
    
    Type: Swing / Position
    Core Tools: 50 EMA / 200 EMA cross, MACD cross, ATR for filtering noise
    """
    
    def __init__(self, fast_ema=50, slow_ema=200, macd_fast=12, macd_slow=26, 
                 macd_signal=9, atr_period=14, atr_multiplier=1.5):
        self.name = "Trend-Following Crossover + MACD"
        self.type = "TREND_CROSSOVER"
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators"""
        df = data.copy()
        
        # EMAs
        df['ema_fast'] = df['close'].ewm(span=self.fast_ema).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_ema).mean()
        
        # MACD
        exp1 = df['close'].ewm(span=self.macd_fast).mean()
        exp2 = df['close'].ewm(span=self.macd_slow).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # ATR for noise filtering
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift())
        df['low_close'] = abs(df['low'] - df['close'].shift())
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['true_range'].rolling(window=self.atr_period).mean()
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals"""
        df = self.calculate_indicators(data)
        
        # Initialize signals
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # EMA crossover conditions
        df['ema_cross_up'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift() <= df['ema_slow'].shift())
        df['ema_cross_down'] = (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift() >= df['ema_slow'].shift())
        
        # MACD confirmation
        df['macd_cross_up'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift() <= df['macd_signal'].shift())
        df['macd_cross_down'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift() >= df['macd_signal'].shift())
        
        # ATR filter - only trade if volatility is above threshold
        df['atr_filter'] = df['atr'] > (df['atr'].rolling(20).mean() * 0.8)
        
        # Buy signals: Golden cross + MACD confirmation + ATR filter
        buy_condition = df['ema_cross_up'] & df['macd_cross_up'] & df['atr_filter']
        df.loc[buy_condition, 'signal'] = 1
        df.loc[buy_condition, 'signal_strength'] = 0.8
        
        # Sell signals: Death cross + MACD confirmation + ATR filter
        sell_condition = df['ema_cross_down'] & df['macd_cross_down'] & df['atr_filter']
        df.loc[sell_condition, 'signal'] = -1
        df.loc[sell_condition, 'signal_strength'] = 0.8
        
        return df


class BreakoutStrategy:
    """
    Breakout Strategy (Volume-Backed Support/Resistance Breaks)
    
    Type: Day / Swing
    Core Tools: Volume spike + support/resistance zones, Inside Bar setups
    """
    
    def __init__(self, volume_threshold=1.5, lookback_period=20, atr_period=14):
        self.name = "Volume-Backed Breakout"
        self.type = "BREAKOUT"
        self.volume_threshold = volume_threshold
        self.lookback_period = lookback_period
        self.atr_period = atr_period
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators"""
        df = data.copy()
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(window=self.lookback_period).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.volume_threshold)
        
        # Support/Resistance levels
        df['resistance'] = df['high'].rolling(window=self.lookback_period).max()
        df['support'] = df['low'].rolling(window=self.lookback_period).min()
        
        # ATR for dynamic stops
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift())
        df['low_close'] = abs(df['low'] - df['close'].shift())
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['true_range'].rolling(window=self.atr_period).mean()
        
        # Inside bars
        df['inside_bar'] = (df['high'] < df['high'].shift()) & (df['low'] > df['low'].shift())
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals"""
        df = self.calculate_indicators(data)
        
        # Initialize signals
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # Breakout conditions
        df['resistance_break'] = (df['close'] > df['resistance'].shift()) & df['volume_spike']
        df['support_break'] = (df['close'] < df['support'].shift()) & df['volume_spike']
        
        # Inside bar breakout setup
        df['inside_bar_break_up'] = df['inside_bar'].shift() & (df['close'] > df['high'].shift())
        df['inside_bar_break_down'] = df['inside_bar'].shift() & (df['close'] < df['low'].shift())
        
        # Buy signals: Resistance breakout or inside bar breakout up
        buy_condition = df['resistance_break'] | (df['inside_bar_break_up'] & df['volume_spike'])
        df.loc[buy_condition, 'signal'] = 1
        df.loc[buy_condition, 'signal_strength'] = 0.75
        
        # Sell signals: Support breakout or inside bar breakout down
        sell_condition = df['support_break'] | (df['inside_bar_break_down'] & df['volume_spike'])
        df.loc[sell_condition, 'signal'] = -1
        df.loc[sell_condition, 'signal_strength'] = 0.75
        
        return df


class MeanReversionStrategy:
    """
    RSI + Bollinger Band Mean Reversion Strategy
    
    Type: Intraday / Swing
    Core Tools: RSI(2 or 14), Bollinger Bands, VWAP for snapback confirmation
    """
    
    def __init__(self, rsi_period=14, bb_period=20, bb_std=2, rsi_oversold=30, rsi_overbought=70):
        self.name = "RSI + Bollinger Band Mean Reversion"
        self.type = "MEAN_REVERSION"
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators"""
        df = data.copy()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=self.bb_period).mean()
        df['bb_std'] = df['close'].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * self.bb_std)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * self.bb_std)
        
        # VWAP
        df['vwap'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals"""
        df = self.calculate_indicators(data)
        
        # Initialize signals
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # Mean reversion conditions
        df['oversold_condition'] = (df['rsi'] < self.rsi_oversold) & (df['close'] < df['bb_lower'])
        df['overbought_condition'] = (df['rsi'] > self.rsi_overbought) & (df['close'] > df['bb_upper'])
        
        # VWAP confirmation
        df['vwap_support'] = df['close'] > df['vwap']
        df['vwap_resistance'] = df['close'] < df['vwap']
        
        # Buy signals: Oversold + below BB lower + above VWAP
        buy_condition = df['oversold_condition'] & df['vwap_support']
        df.loc[buy_condition, 'signal'] = 1
        df.loc[buy_condition, 'signal_strength'] = 0.7
        
        # Sell signals: Overbought + above BB upper + below VWAP
        sell_condition = df['overbought_condition'] & df['vwap_resistance']
        df.loc[sell_condition, 'signal'] = -1
        df.loc[sell_condition, 'signal_strength'] = 0.7
        
        return df


class ScalpingStrategy:
    """
    Scalping via Liquidity Zones + VWAP Strategy
    
    Type: Ultra Short-Term
    Core Tools: VWAP, EMA(9/21), bid-ask spread, L2 if available
    """
    
    def __init__(self, ema_fast=9, ema_slow=21, vwap_period=20):
        self.name = "Liquidity Zones + VWAP Scalping"
        self.type = "SCALPING"
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.vwap_period = vwap_period
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators"""
        df = data.copy()
        
        # Fast EMAs
        df['ema_fast'] = df['close'].ewm(span=self.ema_fast).mean()
        df['ema_slow'] = df['close'].ewm(span=self.ema_slow).mean()
        
        # VWAP
        df['vwap'] = (df['close'] * df['volume']).rolling(window=self.vwap_period).sum() / df['volume'].rolling(window=self.vwap_period).sum()
        
        # Liquidity zones (price levels with high volume)
        df['volume_ma'] = df['volume'].rolling(window=10).mean()
        df['high_volume'] = df['volume'] > df['volume_ma'] * 1.2
        
        # Price momentum
        df['price_momentum'] = df['close'].pct_change(periods=3)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals"""
        df = self.calculate_indicators(data)
        
        # Initialize signals
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # Scalping conditions
        df['ema_bullish'] = df['ema_fast'] > df['ema_slow']
        df['ema_bearish'] = df['ema_fast'] < df['ema_slow']
        df['near_vwap'] = abs(df['close'] - df['vwap']) < (df['vwap'] * 0.001)  # Within 0.1% of VWAP
        
        # Buy signals: Bullish EMA, near VWAP, high volume
        buy_condition = df['ema_bullish'] & df['near_vwap'] & df['high_volume'] & (df['price_momentum'] > 0)
        df.loc[buy_condition, 'signal'] = 1
        df.loc[buy_condition, 'signal_strength'] = 0.6
        
        # Sell signals: Bearish EMA, near VWAP, high volume
        sell_condition = df['ema_bearish'] & df['near_vwap'] & df['high_volume'] & (df['price_momentum'] < 0)
        df.loc[sell_condition, 'signal'] = -1
        df.loc[sell_condition, 'signal_strength'] = 0.6
        
        return df


class MultiSignalStrategy:
    """
    Algorithmic Multi-Signal Strategy (AI-Driven Rules)
    
    Type: Flexible / Automated
    Core Tools: Python logic based on combo signals (MACD + RSI + EMA + volume)
    """
    
    def __init__(self, ema_period=21, rsi_period=14, macd_fast=12, macd_slow=26, 
                 macd_signal=9, volume_threshold=1.3):
        self.name = "Multi-Signal AI Strategy"
        self.type = "MULTI_SIGNAL"
        self.ema_period = ema_period
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.volume_threshold = volume_threshold
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators"""
        df = data.copy()
        
        # EMA
        df['ema'] = df['close'].ewm(span=self.ema_period).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=self.macd_fast).mean()
        exp2 = df['close'].ewm(span=self.macd_slow).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Volume
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.volume_threshold)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using AI-driven logic"""
        df = self.calculate_indicators(data)
        
        # Initialize signals
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # Multi-signal scoring system
        df['bull_score'] = 0
        df['bear_score'] = 0
        
        # EMA trend
        df.loc[df['close'] > df['ema'], 'bull_score'] += 1
        df.loc[df['close'] < df['ema'], 'bear_score'] += 1
        
        # RSI momentum
        df.loc[df['rsi'] > 50, 'bull_score'] += 1
        df.loc[df['rsi'] < 50, 'bear_score'] += 1
        
        # MACD signal
        df.loc[df['macd'] > df['macd_signal'], 'bull_score'] += 1
        df.loc[df['macd'] < df['macd_signal'], 'bear_score'] += 1
        
        # Volume confirmation
        df.loc[df['volume_spike'], 'bull_score'] += 1
        df.loc[df['volume_spike'], 'bear_score'] += 1
        
        # Generate signals based on score
        df.loc[df['bull_score'] >= 3, 'signal'] = 1
        df.loc[df['bear_score'] >= 3, 'signal'] = -1
        
        # Signal strength based on score
        df.loc[df['bull_score'] >= 3, 'signal_strength'] = df['bull_score'] / 4
        df.loc[df['bear_score'] >= 3, 'signal_strength'] = df['bear_score'] / 4
        
        return df


class BuiltinStrategyManager:
    """Manager for all built-in strategies"""
    
    def __init__(self):
        self.strategies = {
            'TREND_CROSSOVER': TrendFollowingCrossoverStrategy(),
            'BREAKOUT': BreakoutStrategy(),
            'MEAN_REVERSION': MeanReversionStrategy(),
            'SCALPING': ScalpingStrategy(),
            'MULTI_SIGNAL': MultiSignalStrategy()
        }
    
    def get_strategy(self, strategy_type: str):
        """Get a strategy by type"""
        return self.strategies.get(strategy_type)
    
    def get_all_strategies(self) -> Dict:
        """Get all available strategies"""
        return self.strategies
    
    def get_strategy_list(self) -> List[Dict]:
        """Get list of strategies for UI"""
        return [
            {
                'type': 'TREND_CROSSOVER',
                'name': 'Trend-Following Crossover + MACD',
                'description': 'EMA crossover with MACD confirmation and ATR filtering'
            },
            {
                'type': 'BREAKOUT',
                'name': 'Volume-Backed Breakout',
                'description': 'Support/resistance breaks with volume confirmation'
            },
            {
                'type': 'MEAN_REVERSION',
                'name': 'RSI + Bollinger Band Mean Reversion',
                'description': 'Mean reversion using RSI and Bollinger Bands with VWAP'
            },
            {
                'type': 'SCALPING',
                'name': 'Liquidity Zones + VWAP Scalping',
                'description': 'Ultra short-term scalping with VWAP and EMAs'
            },
            {
                'type': 'MULTI_SIGNAL',
                'name': 'Multi-Signal AI Strategy',
                'description': 'AI-driven combination of multiple technical indicators'
            }
        ]

# Global instance
builtin_strategy_manager = BuiltinStrategyManager()