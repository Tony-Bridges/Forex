"""
AI Models Module for MT5 Forex Trading Bot

This module provides multiple AI models and learning algorithms to enhance trading strategy performance.
Models include classic ML algorithms, neural networks, and integration with external AI services.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import logging

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Disable TensorFlow by default on Replit due to compatibility issues
TF_AVAILABLE = False
logging.warning("TensorFlow disabled on Replit environment. Using scikit-learn models only.")

# Mock TensorFlow classes (to avoid import errors)
class MockTensorFlow:
    class Sequential:
        def __init__(self, *args):
            pass
    class Dense:
        def __init__(self, *args, **kwargs):
            pass
    class LSTM:
        def __init__(self, *args, **kwargs):
            pass
    class Dropout:
        def __init__(self, *args, **kwargs):
            pass
    class BatchNormalization:
        def __init__(self, *args, **kwargs):
            pass
    class EarlyStopping:
        def __init__(self, *args, **kwargs):
            pass
    class ModelCheckpoint:
        def __init__(self, *args, **kwargs):
            pass

# Create mock objects to use when TensorFlow is not available
Sequential = MockTensorFlow.Sequential
Dense = MockTensorFlow.Dense
LSTM = MockTensorFlow.LSTM
Dropout = MockTensorFlow.Dropout
BatchNormalization = MockTensorFlow.BatchNormalization
EarlyStopping = MockTensorFlow.EarlyStopping
ModelCheckpoint = MockTensorFlow.ModelCheckpoint

# Set up logging
logger = logging.getLogger(__name__)

# Constants
MODEL_SAVE_DIR = 'ai_models'
FEATURES_CONFIG_PATH = os.path.join(MODEL_SAVE_DIR, 'features_config.json')

# Ensure model directory exists
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

class ModelManager:
    """
    Manages the creation, training, and serving of multiple AI models
    """
    
    AVAILABLE_MODELS = {
        'random_forest': 'Random Forest Classifier',
        'gradient_boosting': 'Gradient Boosting Classifier', 
        'logistic_regression': 'Logistic Regression',
        'lstm_network': 'LSTM Neural Network',
        'mlp_network': 'Multi-Layer Perceptron'
    }
    
    def __init__(self):
        """Initialize the model manager"""
        self.models = {}
        self.scalers = {}
        self.feature_configs = self._load_feature_configs()
        
    def _load_feature_configs(self):
        """Load feature configurations from file"""
        if os.path.exists(FEATURES_CONFIG_PATH):
            with open(FEATURES_CONFIG_PATH, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_feature_configs(self):
        """Save feature configurations to file"""
        with open(FEATURES_CONFIG_PATH, 'w') as f:
            json.dump(self.feature_configs, f, indent=4)
    
    def _create_model_path(self, model_name, strategy_type):
        """Create a file path for saving/loading a model"""
        return os.path.join(MODEL_SAVE_DIR, f"{model_name}_{strategy_type}.joblib")
    
    def _create_scaler_path(self, model_name, strategy_type):
        """Create a file path for saving/loading a scaler"""
        return os.path.join(MODEL_SAVE_DIR, f"{model_name}_{strategy_type}_scaler.joblib")
    
    def _create_nn_model_path(self, model_name, strategy_type):
        """Create a file path for saving/loading a neural network model"""
        return os.path.join(MODEL_SAVE_DIR, f"{model_name}_{strategy_type}")
    
    def _prepare_features(self, data, strategy_type, for_training=False):
        """
        Prepare features for model training/prediction based on strategy type
        
        Args:
            data (DataFrame): Price/indicator data
            strategy_type (str): Type of strategy (e.g., 'MA_RSI', 'MACD')
            for_training (bool): Whether this is for training (True) or prediction (False)
            
        Returns:
            tuple: (features, targets) if for_training=True, otherwise just features
        """
        if strategy_type not in self.feature_configs and for_training:
            # Create new feature config if it doesn't exist
            if strategy_type == 'MA_RSI':
                self.feature_configs[strategy_type] = {
                    'features': ['slow_ma', 'fast_ma', 'rsi', 'volume', 'atr', 
                               'slow_ma_diff', 'fast_ma_diff', 'rsi_diff'],
                    'target': 'signal'
                }
            elif strategy_type == 'MACD':
                self.feature_configs[strategy_type] = {
                    'features': ['macd', 'signal_line', 'macd_histogram', 'volume', 'atr',
                               'macd_diff', 'signal_line_diff', 'histogram_diff'],
                    'target': 'signal'
                }
            else:
                # Default features for any other strategy type
                self.feature_configs[strategy_type] = {
                    'features': ['close', 'open', 'high', 'low', 'volume'],
                    'target': 'signal'
                }
            
            # Save the new config
            self._save_feature_configs()
        
        # Get config
        config = self.feature_configs.get(strategy_type, {})
        feature_columns = config.get('features', [])
        target_column = config.get('target', 'signal')
        
        # Make sure all columns exist in data
        for col in feature_columns:
            if col not in data.columns:
                # Try to calculate missing features
                if col.endswith('_diff') and col[:-5] in data.columns:
                    # Calculate difference
                    base_col = col[:-5]
                    data[col] = data[base_col].diff().fillna(0)
                else:
                    logger.warning(f"Feature column {col} not found in data and could not be calculated.")
        
        # Select only features that exist in the data
        actual_features = [col for col in feature_columns if col in data.columns]
        
        if not actual_features:
            raise ValueError("No valid feature columns found in data")
        
        # Extract features and targets
        X = data[actual_features].copy()
        
        # Apply feature scaling
        scaler_key = f"{strategy_type}"
        if for_training:
            # Create a new scaler during training
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[scaler_key] = scaler
        else:
            # Use existing scaler for prediction
            if scaler_key not in self.scalers:
                # Try to load scaler
                self._load_model(None, strategy_type)
            
            if scaler_key in self.scalers:
                X_scaled = self.scalers[scaler_key].transform(X)
            else:
                # Fall back to unscaled features if no scaler is available
                logger.warning(f"No scaler found for {strategy_type}. Using unscaled features.")
                X_scaled = X.values
        
        if for_training:
            if target_column in data.columns:
                y = data[target_column]
                return X_scaled, y
            else:
                raise ValueError(f"Target column '{target_column}' not found in data")
        else:
            return X_scaled
    
    def train_model(self, model_name, strategy_type, training_data):
        """
        Train a model using historical data
        
        Args:
            model_name (str): Type of model to train (e.g., 'random_forest', 'lstm_network')
            strategy_type (str): Type of strategy (e.g., 'MA_RSI', 'MACD')
            training_data (DataFrame): Historical data with features and target signal
            
        Returns:
            dict: Training results including accuracy, metrics
        """
        logger.info(f"Training {model_name} model for {strategy_type} strategy")
        
        # Prepare features and target
        try:
            X, y = self._prepare_features(training_data, strategy_type, for_training=True)
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return {'success': False, 'error': str(e)}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        try:
            # Train model based on type
            if model_name == 'random_forest':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
            elif model_name == 'gradient_boosting':
                model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
            elif model_name == 'logistic_regression':
                model = LogisticRegression(random_state=42, max_iter=1000)
                model.fit(X_train, y_train)
                
            elif model_name == 'lstm_network':
                if not TF_AVAILABLE:
                    return {'success': False, 'error': 'TensorFlow not available'}
                
                # Reshape data for LSTM [samples, time steps, features]
                X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
                X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
                
                # Create model
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(1, X_train.shape[1])),
                    Dropout(0.2),
                    LSTM(50),
                    Dropout(0.2),
                    Dense(1, activation='sigmoid')
                ])
                
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                
                # Set up callbacks
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=10),
                    ModelCheckpoint(filepath=self._create_nn_model_path(model_name, strategy_type),
                                   save_best_only=True)
                ]
                
                # Train model
                history = model.fit(
                    X_train_reshaped, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Evaluate with reshaped test data
                y_pred = (model.predict(X_test_reshaped) > 0.5).astype(int).flatten()
                
            elif model_name == 'mlp_network':
                if not TF_AVAILABLE:
                    return {'success': False, 'error': 'TensorFlow not available'}
                
                # Create model
                model = Sequential([
                    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(32, activation='relu'),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(16, activation='relu'),
                    Dense(1, activation='sigmoid')
                ])
                
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                
                # Set up callbacks
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=10),
                    ModelCheckpoint(filepath=self._create_nn_model_path(model_name, strategy_type),
                                   save_best_only=True)
                ]
                
                # Train model
                history = model.fit(
                    X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Evaluate
                y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
                
            else:
                return {'success': False, 'error': f'Unknown model type: {model_name}'}
            
            # Save the model and scaler
            self.models[f"{model_name}_{strategy_type}"] = model
            
            # For sklearn models
            if model_name in ['random_forest', 'gradient_boosting', 'logistic_regression']:
                y_pred = model.predict(X_test)
                # Save sklearn model
                joblib.dump(model, self._create_model_path(model_name, strategy_type))
                
            # For neural network models, already saved via ModelCheckpoint
            
            # Save scaler
            scaler_key = f"{strategy_type}"
            if scaler_key in self.scalers:
                joblib.dump(self.scalers[scaler_key], self._create_scaler_path(model_name, strategy_type))
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            f1 = f1_score(y_test, y_pred, average='binary')
            conf_matrix = confusion_matrix(y_test, y_pred).tolist()
            
            # Return results
            return {
                'success': True,
                'model_name': model_name,
                'strategy_type': strategy_type,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': conf_matrix,
                'trained_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}
    
    def _load_model(self, model_name, strategy_type):
        """
        Load a model from disk
        
        Args:
            model_name (str): Type of model to load
            strategy_type (str): Type of strategy
            
        Returns:
            bool: True if model was loaded successfully, False otherwise
        """
        model_key = f"{model_name}_{strategy_type}" if model_name else None
        scaler_key = f"{strategy_type}"
        
        # Try loading the scaler first (needed for both prediction and training)
        try:
            # Find any scaler for this strategy type
            if model_name:
                scaler_path = self._create_scaler_path(model_name, strategy_type)
                if os.path.exists(scaler_path):
                    self.scalers[scaler_key] = joblib.load(scaler_path)
                    logger.info(f"Loaded scaler for {strategy_type} from {scaler_path}")
            else:
                # Try all model types to find a scaler
                for m_name in ['random_forest', 'gradient_boosting', 'logistic_regression', 'lstm_network', 'mlp_network']:
                    scaler_path = self._create_scaler_path(m_name, strategy_type)
                    if os.path.exists(scaler_path):
                        self.scalers[scaler_key] = joblib.load(scaler_path)
                        logger.info(f"Loaded scaler for {strategy_type} from {scaler_path}")
                        break
        except Exception as e:
            logger.warning(f"Error loading scaler: {str(e)}")
        
        # Return early if we're just loading the scaler
        if not model_name:
            return scaler_key in self.scalers
        
        # Now try to load the model
        try:
            if model_name in ['random_forest', 'gradient_boosting', 'logistic_regression']:
                # Load sklearn model
                model_path = self._create_model_path(model_name, strategy_type)
                if os.path.exists(model_path):
                    self.models[model_key] = joblib.load(model_path)
                    logger.info(f"Loaded model {model_key} from {model_path}")
                    return True
            
            elif model_name in ['lstm_network', 'mlp_network'] and TF_AVAILABLE:
                # Load TensorFlow model
                model_path = self._create_nn_model_path(model_name, strategy_type)
                if os.path.exists(model_path):
                    self.models[model_key] = load_model(model_path)
                    logger.info(f"Loaded neural network model {model_key} from {model_path}")
                    return True
        
        except Exception as e:
            logger.error(f"Error loading model {model_key}: {str(e)}")
        
        return False
    
    def predict(self, model_name, strategy_type, data):
        """
        Make predictions using a trained model
        
        Args:
            model_name (str): Type of model to use
            strategy_type (str): Type of strategy
            data (DataFrame): Market data with the required features
            
        Returns:
            np.array: Predictions (1 for buy, 0 for hold, -1 for sell)
        """
        model_key = f"{model_name}_{strategy_type}"
        
        # Load model if not already loaded
        if model_key not in self.models:
            if not self._load_model(model_name, strategy_type):
                logger.error(f"Model {model_key} not found or could not be loaded")
                return None
        
        # Prepare features
        try:
            X = self._prepare_features(data, strategy_type, for_training=False)
        except Exception as e:
            logger.error(f"Error preparing features for prediction: {str(e)}")
            return None
        
        # Make predictions based on model type
        try:
            model = self.models[model_key]
            
            if model_name in ['random_forest', 'gradient_boosting', 'logistic_regression']:
                # sklearn models
                predictions = model.predict(X)
                return predictions
            
            elif model_name == 'lstm_network' and TF_AVAILABLE:
                # Reshape for LSTM
                X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
                # Get raw probabilities and convert to binary
                pred_proba = model.predict(X_reshaped)
                predictions = (pred_proba > 0.5).astype(int).flatten()
                return predictions
            
            elif model_name == 'mlp_network' and TF_AVAILABLE:
                # Get raw probabilities and convert to binary
                pred_proba = model.predict(X)
                predictions = (pred_proba > 0.5).astype(int).flatten()
                return predictions
            
            else:
                logger.error(f"Unknown or unsupported model type: {model_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return None

    def list_available_models(self, strategy_type=None):
        """
        List all available pre-trained models
        
        Args:
            strategy_type (str, optional): Filter by strategy type
            
        Returns:
            list: List of available model info
        """
        available_models = []
        
        # Check for model files in the models directory
        for filename in os.listdir(MODEL_SAVE_DIR):
            if filename.endswith('.joblib') and not filename.endswith('_scaler.joblib'):
                # Parse model name and strategy type from filename
                parts = filename.replace('.joblib', '').split('_')
                if len(parts) >= 2:
                    model_name = parts[0]
                    model_strategy = '_'.join(parts[1:])
                    
                    if strategy_type is None or model_strategy == strategy_type:
                        model_path = os.path.join(MODEL_SAVE_DIR, filename)
                        model_stats = os.stat(model_path)
                        
                        available_models.append({
                            'model_name': model_name,
                            'strategy_type': model_strategy,
                            'friendly_name': self.AVAILABLE_MODELS.get(model_name, model_name),
                            'created_at': datetime.fromtimestamp(model_stats.st_ctime).isoformat(),
                            'file_size': model_stats.st_size
                        })
            
            # Check for TensorFlow models (directories)
            elif os.path.isdir(os.path.join(MODEL_SAVE_DIR, filename)):
                # Check if it's a TensorFlow model directory
                if os.path.exists(os.path.join(MODEL_SAVE_DIR, filename, 'saved_model.pb')):
                    # Parse model name and strategy type from directory name
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        model_name = parts[0]
                        model_strategy = '_'.join(parts[1:])
                        
                        if strategy_type is None or model_strategy == strategy_type:
                            model_path = os.path.join(MODEL_SAVE_DIR, filename)
                            model_stats = os.stat(model_path)
                            
                            available_models.append({
                                'model_name': model_name,
                                'strategy_type': model_strategy,
                                'friendly_name': self.AVAILABLE_MODELS.get(model_name, model_name),
                                'created_at': datetime.fromtimestamp(model_stats.st_ctime).isoformat(),
                                'file_size': sum(os.path.getsize(os.path.join(model_path, f)) 
                                            for f in os.listdir(model_path) 
                                            if os.path.isfile(os.path.join(model_path, f)))
                            })
        
        return available_models

    def ensemble_predict(self, strategy_type, data):
        """
        Make predictions using an ensemble of all available models for a strategy
        
        Args:
            strategy_type (str): Type of strategy
            data (DataFrame): Market data with required features
            
        Returns:
            dict: Predictions and confidence metrics
        """
        # Get all available models for this strategy
        available_models = self.list_available_models(strategy_type)
        
        if not available_models:
            logger.warning(f"No trained models available for {strategy_type}")
            return None
        
        # Store predictions from each model
        all_predictions = {}
        
        # Make predictions with each model
        for model_info in available_models:
            model_name = model_info['model_name']
            model_predictions = self.predict(model_name, strategy_type, data)
            
            if model_predictions is not None:
                all_predictions[model_name] = model_predictions
        
        if not all_predictions:
            logger.warning("No successful predictions from any model")
            return None
        
        # Create ensemble prediction by majority voting
        num_models = len(all_predictions)
        ensemble_predictions = np.zeros(len(data))
        
        for model_name, predictions in all_predictions.items():
            # Add predictions, ensuring the same length
            pred_len = min(len(ensemble_predictions), len(predictions))
            ensemble_predictions[:pred_len] += predictions[:pred_len]
        
        # Convert to majority vote (-1, 0, 1)
        ensemble_predictions = np.where(ensemble_predictions > num_models/2, 1, 
                                      np.where(ensemble_predictions < -num_models/2, -1, 0))
        
        # Calculate confidence based on agreement between models
        confidence = np.zeros(len(ensemble_predictions))
        for i in range(len(ensemble_predictions)):
            votes = sum(1 for preds in all_predictions.values() 
                      if i < len(preds) and preds[i] == ensemble_predictions[i])
            confidence[i] = votes / num_models if num_models > 0 else 0
        
        return {
            'predictions': ensemble_predictions,
            'confidence': confidence,
            'model_count': num_models,
            'models_used': list(all_predictions.keys())
        }

class AITradingAdvisor:
    """
    AI-powered trading advisor that uses multiple models to generate trading signals
    """
    
    def __init__(self):
        """Initialize the trading advisor"""
        self.model_manager = ModelManager()
        
    def check_for_signal(self, strategy_type, market_data):
        """
        Check for trading signals using AI models
        
        Args:
            strategy_type (str): Type of strategy
            market_data (DataFrame): Recent market data with indicators
            
        Returns:
            dict: Trading signal and confidence metrics
        """
        # Get predictions from ensemble of models
        ensemble_result = self.model_manager.ensemble_predict(strategy_type, market_data)
        
        if ensemble_result is None:
            logger.warning("No AI signal available, fallback to traditional strategy")
            return {'signal': None, 'ai_confidence': 0, 'models_used': []}
        
        # Get the latest prediction (most recent candle)
        latest_prediction = ensemble_result['predictions'][-1]
        latest_confidence = ensemble_result['confidence'][-1]
        
        # Map numerical prediction to signal
        if latest_prediction == 1:
            signal = 'BUY'
        elif latest_prediction == -1:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return {
            'signal': signal,
            'ai_confidence': latest_confidence,
            'models_used': ensemble_result['models_used'],
            'model_count': ensemble_result['model_count']
        }
    
    def train_all_models(self, strategy_type, training_data):
        """
        Train all available model types on the provided data
        
        Args:
            strategy_type (str): Type of strategy
            training_data (DataFrame): Historical data with features and target signal
            
        Returns:
            dict: Training results for all models
        """
        results = {}
        
        # Train each model type
        for model_name in ModelManager.AVAILABLE_MODELS.keys():
            if model_name in ['lstm_network', 'mlp_network'] and not TF_AVAILABLE:
                results[model_name] = {'success': False, 'error': 'TensorFlow not available'}
                continue
                
            model_result = self.model_manager.train_model(model_name, strategy_type, training_data)
            results[model_name] = model_result
            
        return results
    
    def get_available_models(self, strategy_type=None):
        """
        Get list of all available models
        
        Args:
            strategy_type (str, optional): Filter by strategy type
            
        Returns:
            list: Available models information
        """
        return self.model_manager.list_available_models(strategy_type)
    
    def generate_performance_report(self, strategy_type):
        """
        Generate a performance report for all models of a strategy
        
        Args:
            strategy_type (str): Strategy type
            
        Returns:
            dict: Performance metrics for each model
        """
        # Get list of available models
        models = self.model_manager.list_available_models(strategy_type)
        
        report = {
            'strategy_type': strategy_type,
            'models': models,
            'model_count': len(models),
            'generated_at': datetime.now().isoformat()
        }
        
        return report


try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not available. GPT-based analysis will be disabled.")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic package not available. Claude-based analysis will be disabled.")


class AIMarketAnalyst:
    """
    Uses advanced LLMs (GPT-4, Claude) to analyze market conditions and provide insights
    """
    
    def __init__(self):
        """Initialize the market analyst"""
        self.openai_client = None
        self.anthropic_client = None
        
        # Initialize API clients if available
        if OPENAI_AVAILABLE and os.environ.get('OPENAI_API_KEY'):
            self.openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        
        if ANTHROPIC_AVAILABLE and os.environ.get('ANTHROPIC_API_KEY'):
            self.anthropic_client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
    
    def analyze_market_with_gpt(self, symbol, timeframe, market_data, technical_indicators):
        """
        Analyze market conditions using GPT-4
        
        Args:
            symbol (str): Trading symbol (e.g., 'EURUSD')
            timeframe (str): Timeframe (e.g., 'H1', 'D1')
            market_data (DataFrame): Recent market data
            technical_indicators (dict): Dictionary of technical indicators
            
        Returns:
            dict: Analysis results
        """
        if not self.openai_client:
            return {
                'success': False, 
                'error': 'OpenAI client not available',
                'message': 'To enable AI market analysis with GPT-4, please provide your OpenAI API key.'
            }
        
        try:
            # Prepare data for analysis
            market_summary = self._prepare_market_summary(symbol, timeframe, market_data, technical_indicators)
            
            # Create prompt for GPT
            prompt = f"""
            Analyze the following forex market data for {symbol} on {timeframe} timeframe.
            
            Current Market Data:
            {market_summary}
            
            Please provide:
            1. A concise market sentiment analysis (bullish, bearish, or neutral)
            2. Key support and resistance levels
            3. Potential entry and exit points
            4. Risk management recommendations
            5. Expected market direction in the short and medium term
            
            Format your analysis in JSON with the following structure:
            {{
              "market_sentiment": "bullish|bearish|neutral",
              "strength": 0.1-1.0,
              "key_levels": {{
                "support": [level1, level2, ...],
                "resistance": [level1, level2, ...]
              }},
              "trading_recommendation": "buy|sell|hold",
              "confidence": 0.1-1.0,
              "entry_points": [price1, price2, ...],
              "exit_points": [price1, price2, ...],
              "stop_loss": price,
              "take_profit": price,
              "risk_reward_ratio": number,
              "short_term_forecast": "text",
              "medium_term_forecast": "text",
              "reasoning": "text explanation"
            }}
            """
            
            # Call GPT for analysis
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": "You are an expert forex analyst and trader."},
                         {"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            # Parse the response
            analysis = json.loads(response.choices[0].message.content)
            
            # Add metadata
            analysis['success'] = True
            analysis['model'] = 'gpt-4o'
            analysis['symbol'] = symbol
            analysis['timeframe'] = timeframe
            analysis['timestamp'] = datetime.now().isoformat()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market with GPT: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def analyze_market_with_claude(self, symbol, timeframe, market_data, technical_indicators):
        """
        Analyze market conditions using Claude
        
        Args:
            symbol (str): Trading symbol (e.g., 'EURUSD')
            timeframe (str): Timeframe (e.g., 'H1', 'D1')
            market_data (DataFrame): Recent market data
            technical_indicators (dict): Dictionary of technical indicators
            
        Returns:
            dict: Analysis results
        """
        if not self.anthropic_client:
            return {
                'success': False, 
                'error': 'Anthropic client not available',
                'message': 'To enable AI market analysis with Claude, please provide your Anthropic API key.'
            }
        
        try:
            # Prepare data for analysis
            market_summary = self._prepare_market_summary(symbol, timeframe, market_data, technical_indicators)
            
            # Create prompt for Claude
            prompt = f"""
            Analyze the following forex market data for {symbol} on {timeframe} timeframe.
            
            Current Market Data:
            {market_summary}
            
            Please provide:
            1. A concise market sentiment analysis (bullish, bearish, or neutral)
            2. Key support and resistance levels
            3. Potential entry and exit points
            4. Risk management recommendations
            5. Expected market direction in the short and medium term
            
            Format your analysis in JSON with the following structure:
            {{
              "market_sentiment": "bullish|bearish|neutral",
              "strength": 0.1-1.0,
              "key_levels": {{
                "support": [level1, level2, ...],
                "resistance": [level1, level2, ...]
              }},
              "trading_recommendation": "buy|sell|hold",
              "confidence": 0.1-1.0,
              "entry_points": [price1, price2, ...],
              "exit_points": [price1, price2, ...],
              "stop_loss": price,
              "take_profit": price,
              "risk_reward_ratio": number,
              "short_term_forecast": "text",
              "medium_term_forecast": "text",
              "reasoning": "text explanation"
            }}
            
            Respond only with the JSON object and no additional text.
            """
            
            # Call Claude for analysis
            #the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
            # do not change this unless explicitly requested by the user
            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=0.2,
                system="You are an expert forex analyst and trader. You provide detailed market analysis and trading recommendations in JSON format.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse the response to extract JSON
            content = response.content[0].text
            try:
                # Try to parse as JSON
                analysis = json.loads(content)
            except json.JSONDecodeError:
                # If it's not valid JSON, try to extract JSON from text
                import re
                json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group(1))
                else:
                    # Final attempt - just strip any non-JSON text
                    content = content.strip()
                    if content.startswith('{') and content.endswith('}'):
                        analysis = json.loads(content)
                    else:
                        raise ValueError("Could not extract valid JSON from Claude response")
            
            # Add metadata
            analysis['success'] = True
            analysis['model'] = 'claude-3-5-sonnet-20241022'
            analysis['symbol'] = symbol
            analysis['timeframe'] = timeframe
            analysis['timestamp'] = datetime.now().isoformat()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market with Claude: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _prepare_market_summary(self, symbol, timeframe, market_data, technical_indicators):
        """
        Prepare a summary of market data for AI analysis
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            market_data (DataFrame): Recent market data
            technical_indicators (dict): Technical indicators
            
        Returns:
            str: Formatted market summary
        """
        # Get the latest 5 candles
        recent_data = market_data.tail(5).copy()
        
        # Format candle data
        candle_summary = ""
        for idx, row in recent_data.iterrows():
            candle_summary += f"Time: {idx}, Open: {row['open']:.5f}, High: {row['high']:.5f}, Low: {row['low']:.5f}, Close: {row['close']:.5f}, Volume: {row['volume']}\n"
        
        # Format technical indicators
        indicators_summary = "Technical Indicators:\n"
        for indicator, value in technical_indicators.items():
            if isinstance(value, (int, float)):
                indicators_summary += f"{indicator}: {value:.5f}\n"
            elif isinstance(value, dict):
                indicators_summary += f"{indicator}:\n"
                for k, v in value.items():
                    indicators_summary += f"  {k}: {v:.5f}\n"
        
        # Combine all data
        summary = f"""
        Symbol: {symbol}
        Timeframe: {timeframe}
        Last 5 candles:
        {candle_summary}
        
        {indicators_summary}
        
        Current Price: {recent_data.iloc[-1]['close']:.5f}
        Daily Range: {technical_indicators.get('daily_range', 'Not available')}
        """
        
        return summary
    
    def combine_analyses(self, analyses):
        """
        Combine multiple analyses from different models
        
        Args:
            analyses (list): List of analysis results from different models
            
        Returns:
            dict: Combined analysis
        """
        if not analyses:
            return {'success': False, 'error': 'No analyses to combine'}
        
        successful_analyses = [a for a in analyses if a.get('success', False)]
        if not successful_analyses:
            return {'success': False, 'error': 'No successful analyses to combine'}
        
        # Extract key signals and metadata
        sentiments = []
        recommendations = []
        confidences = []
        models_used = []
        symbol = successful_analyses[0].get('symbol', 'Unknown')
        timeframe = successful_analyses[0].get('timeframe', 'Unknown')
        
        for analysis in successful_analyses:
            sentiments.append(analysis.get('market_sentiment', 'neutral'))
            recommendations.append(analysis.get('trading_recommendation', 'hold'))
            confidences.append(analysis.get('confidence', 0.5))
            models_used.append(analysis.get('model', 'Unknown'))
        
        # Count sentiments
        sentiment_counts = {'bullish': 0, 'bearish': 0, 'neutral': 0}
        for sentiment in sentiments:
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
        
        # Determine consensus sentiment
        max_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])
        consensus_sentiment = max_sentiment[0]
        
        # Count recommendations
        rec_counts = {'buy': 0, 'sell': 0, 'hold': 0}
        for rec in recommendations:
            if rec in rec_counts:
                rec_counts[rec] += 1
        
        # Determine consensus recommendation
        max_rec = max(rec_counts.items(), key=lambda x: x[1])
        consensus_recommendation = max_rec[0]
        
        # Average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Combine support and resistance levels
        all_support = []
        all_resistance = []
        
        for analysis in successful_analyses:
            key_levels = analysis.get('key_levels', {})
            if 'support' in key_levels:
                all_support.extend(key_levels['support'])
            if 'resistance' in key_levels:
                all_resistance.extend(key_levels['resistance'])
        
        # Remove duplicates and sort
        unique_support = sorted(set(all_support))
        unique_resistance = sorted(set(all_resistance))
        
        # Create combined analysis
        combined = {
            'success': True,
            'symbol': symbol,
            'timeframe': timeframe,
            'market_sentiment': consensus_sentiment,
            'trading_recommendation': consensus_recommendation,
            'confidence': avg_confidence,
            'key_levels': {
                'support': unique_support,
                'resistance': unique_resistance
            },
            'sentiment_distribution': sentiment_counts,
            'recommendation_distribution': rec_counts,
            'models_used': models_used,
            'analysis_count': len(successful_analyses),
            'timestamp': datetime.now().isoformat()
        }
        
        return combined
    
    def get_full_market_analysis(self, symbol, timeframe, market_data, technical_indicators):
        """
        Get comprehensive market analysis from multiple AI models
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            market_data (DataFrame): Recent market data
            technical_indicators (dict): Technical indicators
            
        Returns:
            dict: Combined analysis from multiple models
        """
        analyses = []
        
        # Get analysis from GPT if available
        if self.openai_client:
            gpt_analysis = self.analyze_market_with_gpt(symbol, timeframe, market_data, technical_indicators)
            if gpt_analysis.get('success', False):
                analyses.append(gpt_analysis)
        
        # Get analysis from Claude if available
        if self.anthropic_client:
            claude_analysis = self.analyze_market_with_claude(symbol, timeframe, market_data, technical_indicators)
            if claude_analysis.get('success', False):
                analyses.append(claude_analysis)
        
        # Combine analyses
        combined = self.combine_analyses(analyses)
        
        return combined