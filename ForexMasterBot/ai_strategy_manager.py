"""
AI Strategy Manager Module

This module coordinates the use of multiple AI models to enhance forex trading strategies.
It provides a unified interface for generating trading signals, model training, and performance tracking.
"""

import os
import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import traceback

from ai_models import AITradingAdvisor, AIMarketAnalyst

# Setup logging
logger = logging.getLogger(__name__)

class AIStrategyManager:
    """
    Manages the integration of AI models with trading strategies
    """
    
    def __init__(self):
        """Initialize the AI strategy manager"""
        self.trading_advisor = AITradingAdvisor()
        self.market_analyst = AIMarketAnalyst()
        self.strategy_performance = {}
        
        # Load performance history if available
        self._load_performance_history()
    
    def _load_performance_history(self):
        """Load performance history from file"""
        performance_file = 'ai_strategy_performance.json'
        if os.path.exists(performance_file):
            try:
                with open(performance_file, 'r') as f:
                    self.strategy_performance = json.load(f)
            except Exception as e:
                logger.error(f"Error loading performance history: {str(e)}")
    
    def _save_performance_history(self):
        """Save performance history to file"""
        performance_file = 'ai_strategy_performance.json'
        try:
            with open(performance_file, 'w') as f:
                json.dump(self.strategy_performance, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving performance history: {str(e)}")
    
    def get_ai_enhanced_signal(self, strategy_type, symbol, timeframe, market_data, indicators, traditional_signal=None):
        """
        Generate an AI-enhanced trading signal
        
        Args:
            strategy_type (str): Type of strategy (e.g., 'MA_RSI', 'MACD')
            symbol (str): Trading symbol (e.g., 'EURUSD')
            timeframe (str): Timeframe (e.g., 'H1', 'D1')
            market_data (DataFrame): Recent market data
            indicators (dict): Technical indicators calculated for the market data
            traditional_signal (str, optional): Signal from traditional strategy logic
            
        Returns:
            dict: Enhanced trading signal with AI confidence and insights
        """
        try:
            # Get AI trading signal
            ai_signal_result = self.trading_advisor.check_for_signal(strategy_type, market_data)
            
            # Get market analysis
            market_analysis = self.market_analyst.get_full_market_analysis(
                symbol, timeframe, market_data, indicators
            )
            
            # Combine signals
            final_signal = self._combine_signals(
                traditional_signal=traditional_signal,
                ai_signal=ai_signal_result.get('signal'),
                ai_confidence=ai_signal_result.get('ai_confidence', 0),
                analyst_signal=market_analysis.get('trading_recommendation') if market_analysis.get('success', False) else None,
                analyst_confidence=market_analysis.get('confidence', 0) if market_analysis.get('success', False) else 0
            )
            
            # Prepare result
            result = {
                'signal': final_signal,
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'ai_signal': ai_signal_result.get('signal'),
                'ai_confidence': ai_signal_result.get('ai_confidence', 0),
                'models_used': ai_signal_result.get('models_used', []),
                'traditional_signal': traditional_signal,
                'market_sentiment': market_analysis.get('market_sentiment') if market_analysis.get('success', False) else None,
                'analyst_recommendation': market_analysis.get('trading_recommendation') if market_analysis.get('success', False) else None,
                'key_levels': market_analysis.get('key_levels', {}) if market_analysis.get('success', False) else {},
                'strategy_type': strategy_type
            }
            
            # Update and save performance data
            self._update_strategy_performance(strategy_type, 'signal_generated', result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating AI-enhanced signal: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return traditional signal as fallback
            return {
                'signal': traditional_signal,
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'ai_signal': None,
                'ai_confidence': 0,
                'traditional_signal': traditional_signal,
                'strategy_type': strategy_type,
                'error': str(e)
            }
    
    def _combine_signals(self, traditional_signal, ai_signal, ai_confidence, analyst_signal, analyst_confidence):
        """
        Combine signals from different sources into a final trading decision
        
        Args:
            traditional_signal (str): Signal from traditional strategy
            ai_signal (str): Signal from AI models
            ai_confidence (float): Confidence of AI signal
            analyst_signal (str): Signal from market analyst
            analyst_confidence (float): Confidence of analyst signal
            
        Returns:
            str: Final signal (BUY, SELL, HOLD)
        """
        # Default to traditional signal if it exists
        if traditional_signal and not ai_signal and not analyst_signal:
            return traditional_signal
        
        # If only AI signal exists
        if ai_signal and not traditional_signal and not analyst_signal:
            # Only use AI signal if confidence is high enough
            return ai_signal if ai_confidence >= 0.6 else 'HOLD'
        
        # If only analyst signal exists
        if analyst_signal and not traditional_signal and not ai_signal:
            # Only use analyst signal if confidence is high enough
            return analyst_signal.upper() if analyst_confidence >= 0.7 else 'HOLD'
        
        # If we have multiple signals, use a weighted approach
        signals = {}
        if traditional_signal:
            signals[traditional_signal] = 0.4  # Base weight for traditional signal
            
        if ai_signal:
            signals[ai_signal] = 0.3 * ai_confidence  # Weight by confidence
            
        if analyst_signal:
            signals[analyst_signal.upper()] = 0.3 * analyst_confidence  # Weight by confidence
        
        # Get signal with highest weight
        if signals:
            final_signal = max(signals.items(), key=lambda x: x[1])[0]
            
            # Special case: if all signals disagree, favor HOLD
            if len(signals) >= 2 and list(signals.keys()).count('BUY') <= 1 and list(signals.keys()).count('SELL') <= 1:
                return 'HOLD'
                
            return final_signal
        
        # Fallback
        return 'HOLD'
    
    def train_models(self, strategy_type, historical_data):
        """
        Train all AI models for a specific strategy
        
        Args:
            strategy_type (str): Type of strategy
            historical_data (DataFrame): Historical market data with indicators and signals
            
        Returns:
            dict: Training results
        """
        try:
            # Prepare training data
            logger.info(f"Training AI models for {strategy_type} strategy with {len(historical_data)} data points")
            
            # Train all models
            training_results = self.trading_advisor.train_all_models(strategy_type, historical_data)
            
            # Log results
            success_count = sum(1 for result in training_results.values() if result.get('success', False))
            logger.info(f"Training completed: {success_count}/{len(training_results)} models trained successfully")
            
            # Update performance tracking
            self._update_strategy_performance(strategy_type, 'model_training', {
                'timestamp': datetime.now().isoformat(),
                'data_size': len(historical_data),
                'success_count': success_count,
                'total_models': len(training_results),
                'model_results': {name: {'success': result.get('success', False), 
                                        'accuracy': result.get('accuracy', 0) if result.get('success', False) else 0} 
                                 for name, result in training_results.items()}
            })
            
            return {
                'success': success_count > 0,
                'strategy_type': strategy_type,
                'models_trained': success_count,
                'total_models': len(training_results),
                'results': training_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}
    
    def _update_strategy_performance(self, strategy_type, action_type, data):
        """
        Update performance tracking for a strategy
        
        Args:
            strategy_type (str): Type of strategy
            action_type (str): Type of action (e.g., 'signal_generated', 'model_training')
            data (dict): Action data
        """
        if strategy_type not in self.strategy_performance:
            self.strategy_performance[strategy_type] = {
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'signal_count': 0,
                'training_count': 0,
                'actions': []
            }
        
        # Update counters
        if action_type == 'signal_generated':
            self.strategy_performance[strategy_type]['signal_count'] += 1
        elif action_type == 'model_training':
            self.strategy_performance[strategy_type]['training_count'] += 1
        
        # Update timestamp
        self.strategy_performance[strategy_type]['last_updated'] = datetime.now().isoformat()
        
        # Add action to history (limited to last 100 actions)
        self.strategy_performance[strategy_type]['actions'].append({
            'timestamp': datetime.now().isoformat(),
            'action_type': action_type,
            'data': data
        })
        
        # Limit history size
        self.strategy_performance[strategy_type]['actions'] = self.strategy_performance[strategy_type]['actions'][-100:]
        
        # Save to file
        self._save_performance_history()
    
    def get_performance_report(self, strategy_type=None):
        """
        Get performance report for strategies
        
        Args:
            strategy_type (str, optional): Type of strategy, or None for all
            
        Returns:
            dict: Performance report
        """
        if strategy_type:
            # Return report for specific strategy
            if strategy_type in self.strategy_performance:
                return {strategy_type: self.strategy_performance[strategy_type]}
            return {}
        
        # Return summary report for all strategies
        summary = {
            'strategy_count': len(self.strategy_performance),
            'total_signals': sum(s['signal_count'] for s in self.strategy_performance.values()),
            'total_trainings': sum(s['training_count'] for s in self.strategy_performance.values()),
            'strategies': {s: {
                'signal_count': self.strategy_performance[s]['signal_count'],
                'training_count': self.strategy_performance[s]['training_count'],
                'last_updated': self.strategy_performance[s]['last_updated']
            } for s in self.strategy_performance}
        }
        
        return summary
    
    def get_available_models(self, strategy_type=None):
        """
        Get list of available trained models
        
        Args:
            strategy_type (str, optional): Type of strategy, or None for all
            
        Returns:
            list: Available models
        """
        return self.trading_advisor.get_available_models(strategy_type)


# Create global instance
ai_strategy_manager = AIStrategyManager()