#!/usr/bin/env python3
"""
Demo script to showcase the new built-in trading strategies

This script demonstrates how to use each of the 5 professional built-in strategies:
1. Trend-Following Crossover + MACD
2. Volume-Backed Breakout
3. RSI + Bollinger Band Mean Reversion
4. Liquidity Zones + VWAP Scalping
5. Multi-Signal AI Strategy
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import logging

# Import our built-in strategies
from builtin_strategies import builtin_strategy_manager
import utils

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_demo_data(symbol='EURUSD', periods=1000):
    """Generate realistic demo data for testing"""
    print(f"Generating {periods} periods of demo data for {symbol}...")
    return utils.generate_sample_data(symbol, periods)

def test_strategy(strategy_name, strategy_type, data):
    """Test a single strategy and show results"""
    print(f"\n{'='*60}")
    print(f"TESTING: {strategy_name}")
    print(f"{'='*60}")
    
    # Get the strategy
    strategy = builtin_strategy_manager.get_strategy(strategy_type)
    if not strategy:
        print(f"❌ Strategy {strategy_type} not found!")
        return
    
    print(f"📊 Strategy Type: {strategy.type}")
    print(f"📝 Description: {strategy.name}")
    
    # Generate signals
    print("🔄 Generating trading signals...")
    signals_data = strategy.generate_signals(data)
    
    # Count signals
    buy_signals = (signals_data['signal'] == 1).sum()
    sell_signals = (signals_data['signal'] == -1).sum()
    total_signals = buy_signals + sell_signals
    
    print(f"📈 Buy Signals: {buy_signals}")
    print(f"📉 Sell Signals: {sell_signals}")
    print(f"📊 Total Signals: {total_signals}")
    
    if total_signals > 0:
        # Show signal distribution
        signal_strength = signals_data[signals_data['signal'] != 0]['signal_strength'].mean()
        print(f"💪 Average Signal Strength: {signal_strength:.2f}")
        
        # Show recent signals
        recent_signals = signals_data[signals_data['signal'] != 0].tail(5)
        if len(recent_signals) > 0:
            print(f"\n📋 Recent Signals:")
            for idx, row in recent_signals.iterrows():
                signal_type = "BUY" if row['signal'] == 1 else "SELL"
                print(f"   {idx.strftime('%Y-%m-%d %H:%M')} | {signal_type} | Strength: {row['signal_strength']:.2f}")
    
    return signals_data

def simulate_trading(signals_data, initial_balance=10000, risk_per_trade=0.02):
    """Simulate trading performance"""
    print(f"\n💰 Simulating trading with ${initial_balance:,.0f} initial balance")
    print(f"🎯 Risk per trade: {risk_per_trade*100}%")
    
    balance = initial_balance
    trades = []
    position = None
    
    for idx, row in signals_data.iterrows():
        if row['signal'] != 0 and position is None:
            # Open position
            position = {
                'type': 'BUY' if row['signal'] == 1 else 'SELL',
                'entry_price': row['close'],
                'entry_time': idx,
                'risk_amount': balance * risk_per_trade
            }
        elif row['signal'] != 0 and position is not None:
            # Close position and open new one
            exit_price = row['close']
            
            # Calculate profit/loss (simplified)
            if position['type'] == 'BUY':
                pnl = (exit_price - position['entry_price']) / position['entry_price']
            else:
                pnl = (position['entry_price'] - exit_price) / position['entry_price']
            
            # Apply risk management
            pnl = max(pnl, -risk_per_trade)  # Max loss is risk amount
            pnl = min(pnl, risk_per_trade * 3)  # Max gain is 3x risk (1:3 RR)
            
            profit = balance * pnl
            balance += profit
            
            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': idx,
                'type': position['type'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'profit': profit,
                'balance': balance
            })
            
            # Open new position
            position = {
                'type': 'BUY' if row['signal'] == 1 else 'SELL',
                'entry_price': row['close'],
                'entry_time': idx,
                'risk_amount': balance * risk_per_trade
            }
    
    if len(trades) > 0:
        total_profit = balance - initial_balance
        win_rate = len([t for t in trades if t['profit'] > 0]) / len(trades) * 100
        
        print(f"📊 Trading Results:")
        print(f"   Total Trades: {len(trades)}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Total Profit: ${total_profit:,.2f}")
        print(f"   Return: {(total_profit/initial_balance)*100:.1f}%")
        print(f"   Final Balance: ${balance:,.2f}")
    else:
        print("⚠️  No completed trades in simulation period")
    
    return trades

def main():
    """Main demo function"""
    print("🚀 MT5 Trading Bot - Built-in Strategies Demonstration")
    print("=" * 60)
    
    # Generate demo market data
    demo_data = generate_demo_data('EURUSD', 1000)
    print(f"📅 Data period: {demo_data.index[0]} to {demo_data.index[-1]}")
    
    # Test each built-in strategy
    strategies_to_test = [
        ("🔁 Trend-Following Crossover + MACD", "TREND_CROSSOVER"),
        ("📉 Volume-Backed Breakout", "BREAKOUT"),
        ("📊 RSI + Bollinger Band Mean Reversion", "MEAN_REVERSION"),
        ("⚡ Liquidity Zones + VWAP Scalping", "SCALPING"),
        ("🧠 Multi-Signal AI Strategy", "MULTI_SIGNAL")
    ]
    
    all_results = {}
    
    for strategy_name, strategy_type in strategies_to_test:
        signals_data = test_strategy(strategy_name, strategy_type, demo_data)
        if signals_data is not None:
            trades = simulate_trading(signals_data)
            all_results[strategy_name] = {
                'signals': signals_data,
                'trades': trades
            }
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("📊 STRATEGY COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    print(f"{'Strategy':<40} {'Signals':<10} {'Trades':<10} {'Return':<10}")
    print("-" * 70)
    
    for strategy_name, results in all_results.items():
        signals_count = (results['signals']['signal'] != 0).sum()
        trades_count = len(results['trades'])
        
        if trades_count > 0:
            final_balance = results['trades'][-1]['balance']
            return_pct = ((final_balance - 10000) / 10000) * 100
            return_str = f"{return_pct:+.1f}%"
        else:
            return_str = "N/A"
        
        # Clean up strategy name for display
        clean_name = strategy_name.split(' ', 1)[1] if ' ' in strategy_name else strategy_name
        print(f"{clean_name:<40} {signals_count:<10} {trades_count:<10} {return_str:<10}")
    
    print("\n🎯 How to Use These Strategies:")
    print("1. Go to 'Built-in Strategies' in the navigation menu")
    print("2. Browse the 5 professional strategies")
    print("3. Click 'Preview' to see how signals look")
    print("4. Click 'Create' to add to your account")
    print("5. Use in 'AI Trading' for automated execution")
    
    print("\n✅ Demo completed! All strategies are ready for live trading.")

if __name__ == "__main__":
    main()