"""
Backtest Monitor Agent for the QuantAI multi-agent system.

This agent is responsible for monitoring backtesting processes
and analyzing strategy performance in historical data.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from autogen_core import Agent, MessageContext
from ...core.base import BaseQuantAgent, AgentRole, AgentCapability
from ...core.messages import QuantMessage, MessageType, BacktestMessage


@dataclass
class Trade:
    """Represents a single trade in the backtest."""
    timestamp: datetime
    symbol: str
    action: str  # 'BUY' or 'SELL'
    quantity: int
    price: float
    slippage: float
    commission: float
    strategy_id: str
    trade_id: str


@dataclass
class PerformanceAttribution:
    """Performance attribution analysis."""
    strategy_contribution: Dict[str, float]
    factor_contribution: Dict[str, float]
    sector_contribution: Dict[str, float]
    alpha: float
    beta: float
    tracking_error: float


class SlippageModel(Enum):
    """Slippage modeling approaches."""
    FIXED_BPS = "fixed_bps"
    VOLUME_BASED = "volume_based"
    VOLATILITY_BASED = "volatility_based"
    MARKET_IMPACT = "market_impact"


class BacktestMonitorAgent(BaseQuantAgent):
    """
    Agent responsible for monitoring backtesting processes.
    
    This agent manages backtesting workflows, monitors performance,
    and provides analysis of strategy effectiveness on historical data.
    """
    
    def __init__(self, agent_id: str = "backtest_monitor"):
        super().__init__(
            role=AgentRole.BACKTEST_MONITOR,
            capabilities=[
                AgentCapability.PERFORMANCE_MONITORING,
                AgentCapability.DATA_ANALYSIS,
                AgentCapability.BACKTESTING,
            ]
        )
        self.agent_id = agent_id
        self.active_backtests: Dict[str, Dict[str, Any]] = {}
        self.completed_backtests: List[Dict[str, Any]] = []
        self.performance_metrics = {
            'total_backtests': 0,
            'successful_backtests': 0,
            'failed_backtests': 0,
            'average_duration': 0.0
        }

        # Advanced backtesting configuration
        self.slippage_model = SlippageModel.VOLUME_BASED
        self.commission_rate = 0.001  # 0.1% commission
        self.market_impact_factor = 0.0001  # Market impact coefficient
        self.benchmark_symbol = "SPY"  # Default benchmark

        # Performance attribution settings
        self.factor_models = {
            "market": 1.0,
            "size": 0.0,
            "value": 0.0,
            "momentum": 0.0,
            "quality": 0.0,
            "volatility": 0.0
        }
    
    async def on_messages(self, messages: List[QuantMessage], ctx: MessageContext) -> str:
        """Handle incoming messages for backtest monitoring."""
        results = []
        
        for message in messages:
            if isinstance(message, BacktestMessage):
                result = await self._handle_backtest_request(message)
                results.append(result)
            else:
                result = await self._handle_general_message(message)
                results.append(result)
        
        return f"BacktestMonitorAgent processed {len(results)} messages"
    
    async def _handle_backtest_request(self, message: BacktestMessage) -> Dict[str, Any]:
        """Handle backtest request."""
        try:
            backtest_id = message.backtest_id
            
            # Validate backtest request
            if not await self._validate_backtest_request(message):
                return {
                    'status': 'rejected',
                    'backtest_id': backtest_id,
                    'error': 'Backtest validation failed'
                }
            
            # Start backtest
            backtest_result = await self._run_backtest(message)
            
            # Record backtest
            self.completed_backtests.append({
                'backtest_id': backtest_id,
                'completion_time': datetime.now(),
                'status': backtest_result['status'],
                'results': backtest_result
            })
            
            # Update metrics
            self._update_backtest_metrics(backtest_result)
            
            return backtest_result
            
        except Exception as e:
            return {
                'status': 'error',
                'backtest_id': getattr(message, 'backtest_id', 'unknown'),
                'error': str(e)
            }
    
    async def _validate_backtest_request(self, backtest: BacktestMessage) -> bool:
        """Validate backtest request."""
        # Check required fields
        if not backtest.strategy_id:
            return False
        
        if not backtest.start_date or not backtest.end_date:
            return False
        
        if backtest.start_date >= backtest.end_date:
            return False
        
        if not backtest.symbols:
            return False
        
        if backtest.initial_capital <= 0:
            return False
        
        return True
    
    async def _run_backtest(self, backtest: BacktestMessage) -> Dict[str, Any]:
        """Run backtest simulation."""
        backtest_id = backtest.backtest_id
        
        # Add to active backtests
        self.active_backtests[backtest_id] = {
            'strategy_id': backtest.strategy_id,
            'start_date': backtest.start_date,
            'end_date': backtest.end_date,
            'symbols': backtest.symbols,
            'initial_capital': backtest.initial_capital,
            'start_time': datetime.now(),
            'status': 'running'
        }
        
        # Simulate backtest execution
        await asyncio.sleep(0.5)  # Simulate backtest time
        
        # Generate simulated results
        results = await self._generate_backtest_results(backtest)
        
        # Remove from active backtests
        if backtest_id in self.active_backtests:
            del self.active_backtests[backtest_id]
        
        return {
            'status': 'completed',
            'backtest_id': backtest_id,
            'strategy_id': backtest.strategy_id,
            'execution_time': datetime.now(),
            'results': results,
            'message': f'Backtest {backtest_id} completed successfully'
        }
    
    async def _generate_backtest_results(self, backtest: BacktestMessage) -> Dict[str, Any]:
        """Generate comprehensive backtest results with advanced analytics."""

        # Generate realistic trade sequence
        trades = await self._simulate_trade_sequence(backtest)

        # Calculate performance metrics with slippage and commissions
        performance_metrics = await self._calculate_performance_metrics(trades, backtest)

        # Perform performance attribution analysis
        attribution = await self._calculate_performance_attribution(trades, backtest)

        # Generate comprehensive reporting
        reports = await self._generate_comprehensive_reports(trades, performance_metrics, attribution, backtest)

        # Risk analytics
        risk_metrics = await self._calculate_risk_metrics(trades, backtest)

        return {
            'status': 'completed',
            'backtest_id': backtest.backtest_id,
            'performance_metrics': performance_metrics,
            'performance_attribution': attribution.__dict__,
            'trade_statistics': await self._calculate_trade_statistics(trades),
            'risk_metrics': risk_metrics,
            'reports': reports,
            'trades': [self._trade_to_dict(trade) for trade in trades[-100:]],  # Last 100 trades
            'period_info': {
                'start_date': backtest.start_date,
                'end_date': backtest.end_date,
                'period_days': (backtest.end_date - backtest.start_date).days,
                'symbols': backtest.symbols,
                'initial_capital': backtest.initial_capital
            }
        }
    
    async def _handle_general_message(self, message: QuantMessage) -> Dict[str, Any]:
        """Handle general messages."""
        return {
            'status': 'processed',
            'message_type': message.message_type.value,
            'sender': message.sender_id
        }

    async def process_message(self, message: QuantMessage, ctx: MessageContext) -> Optional[QuantMessage]:
        """Process a single message (required by BaseQuantAgent)."""
        try:
            if isinstance(message, BacktestMessage):
                result = await self._handle_backtest_request(message)
                return QuantMessage(
                    message_type=MessageType.BACKTEST_RESPONSE,
                    sender_id=self.agent_id,
                    data_payload=result
                )
            else:
                result = await self._handle_general_message(message)
                return QuantMessage(
                    message_type=MessageType.GENERAL_RESPONSE,
                    sender_id=self.agent_id,
                    data_payload=result
                )
        except Exception as e:
            return QuantMessage(
                message_type=MessageType.ERROR,
                sender_id=self.agent_id,
                error_message=str(e)
            )
    
    def _update_backtest_metrics(self, backtest_result: Dict[str, Any]) -> None:
        """Update backtest metrics."""
        self.performance_metrics['total_backtests'] += 1
        
        if backtest_result['status'] == 'completed':
            self.performance_metrics['successful_backtests'] += 1
        else:
            self.performance_metrics['failed_backtests'] += 1
    
    def get_active_backtests(self) -> Dict[str, Dict[str, Any]]:
        """Get list of active backtests."""
        return self.active_backtests.copy()
    
    def get_backtest_history(self) -> List[Dict[str, Any]]:
        """Get backtest history."""
        return self.completed_backtests.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.performance_metrics.copy()

    async def _simulate_trade_sequence(self, backtest: BacktestMessage) -> List[Trade]:
        """Simulate realistic trade sequence with market dynamics."""
        trades = []
        current_date = backtest.start_date
        trade_id_counter = 1

        # Generate trades throughout the backtest period
        while current_date < backtest.end_date:
            # Simulate 1-5 trades per day on average
            daily_trades = np.random.poisson(2)

            for _ in range(daily_trades):
                if np.random.random() < 0.7:  # 70% chance of trade
                    trade = await self._generate_realistic_trade(
                        current_date, backtest.symbols, backtest.strategy_id, trade_id_counter
                    )
                    if trade:
                        trades.append(trade)
                        trade_id_counter += 1

            current_date += timedelta(days=1)

        return trades

    async def _generate_realistic_trade(self, timestamp: datetime, symbols: List[str],
                                      strategy_id: str, trade_id: int) -> Optional[Trade]:
        """Generate a realistic trade with market dynamics."""

        symbol = np.random.choice(symbols)
        action = np.random.choice(['BUY', 'SELL'])

        # Realistic quantity based on position sizing
        base_quantity = np.random.randint(10, 1000)
        quantity = int(base_quantity * np.random.lognormal(0, 0.5))  # Log-normal distribution

        # Realistic price with some volatility
        base_price = np.random.uniform(50, 500)  # Base price range
        price_volatility = 0.02  # 2% daily volatility
        price = base_price * (1 + np.random.normal(0, price_volatility))

        # Calculate slippage based on model
        slippage = await self._calculate_slippage(symbol, quantity, price)

        # Apply slippage to execution price
        execution_price = price * (1 + slippage if action == 'BUY' else 1 - slippage)

        # Calculate commission
        commission = max(1.0, quantity * execution_price * self.commission_rate)

        return Trade(
            timestamp=timestamp,
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=execution_price,
            slippage=slippage,
            commission=commission,
            strategy_id=strategy_id,
            trade_id=f"{strategy_id}_{trade_id:06d}"
        )

    async def _calculate_slippage(self, symbol: str, quantity: int, price: float) -> float:
        """Calculate slippage based on the configured model."""

        if self.slippage_model == SlippageModel.FIXED_BPS:
            return 0.0005  # 5 basis points

        elif self.slippage_model == SlippageModel.VOLUME_BASED:
            # Slippage increases with trade size
            volume_factor = min(quantity / 10000, 0.01)  # Cap at 1%
            return 0.0002 + volume_factor

        elif self.slippage_model == SlippageModel.VOLATILITY_BASED:
            # Higher slippage during volatile periods
            volatility_factor = np.random.uniform(0.5, 2.0)  # Simulate volatility
            return 0.0003 * volatility_factor

        elif self.slippage_model == SlippageModel.MARKET_IMPACT:
            # Market impact model
            trade_value = quantity * price
            impact = self.market_impact_factor * np.sqrt(trade_value)
            return min(impact, 0.01)  # Cap at 1%

        return 0.0005  # Default fallback

    async def _calculate_performance_metrics(self, trades: List[Trade], backtest: BacktestMessage) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""

        if not trades:
            return self._get_empty_performance_metrics(backtest)

        # Create portfolio value series
        portfolio_values = await self._calculate_portfolio_series(trades, backtest.initial_capital)

        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # Performance metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

        # Annualized metrics
        period_days = (backtest.end_date - backtest.start_date).days
        annual_factor = 252 / period_days if period_days > 0 else 1

        annual_return = (1 + total_return) ** annual_factor - 1
        annual_volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0

        # Risk-adjusted metrics
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0

        # Drawdown analysis
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / running_max
        max_drawdown = np.min(drawdowns)

        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win/Loss analysis
        trade_returns = []
        for i, trade in enumerate(trades):
            if i > 0 and trade.action == 'SELL':
                # Find corresponding buy
                for j in range(i-1, -1, -1):
                    if trades[j].symbol == trade.symbol and trades[j].action == 'BUY':
                        trade_return = (trade.price - trades[j].price) / trades[j].price
                        trade_returns.append(trade_return)
                        break

        winning_trades = sum(1 for r in trade_returns if r > 0)
        total_trade_pairs = len(trade_returns)
        win_rate = winning_trades / total_trade_pairs if total_trade_pairs > 0 else 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'total_trade_pairs': total_trade_pairs,
            'winning_trades': winning_trades,
            'final_value': portfolio_values[-1],
            'total_commissions': sum(trade.commission for trade in trades),
            'total_slippage_cost': sum(trade.quantity * trade.price * trade.slippage for trade in trades)
        }

    async def _calculate_portfolio_series(self, trades: List[Trade], initial_capital: float) -> np.ndarray:
        """Calculate portfolio value time series."""

        # Group trades by date
        daily_trades = {}
        for trade in trades:
            date_key = trade.timestamp.date()
            if date_key not in daily_trades:
                daily_trades[date_key] = []
            daily_trades[date_key].append(trade)

        # Calculate daily portfolio values
        portfolio_value = initial_capital
        portfolio_values = [portfolio_value]

        sorted_dates = sorted(daily_trades.keys())

        for date in sorted_dates:
            day_trades = daily_trades[date]

            # Calculate P&L for the day
            daily_pnl = 0
            for trade in day_trades:
                # Simplified P&L calculation
                trade_value = trade.quantity * trade.price
                if trade.action == 'BUY':
                    daily_pnl -= trade_value + trade.commission
                else:  # SELL
                    daily_pnl += trade_value - trade.commission

            portfolio_value += daily_pnl
            portfolio_values.append(portfolio_value)

        return np.array(portfolio_values)

    async def _calculate_performance_attribution(self, trades: List[Trade], backtest: BacktestMessage) -> PerformanceAttribution:
        """Calculate performance attribution analysis."""

        # Strategy-level attribution
        strategy_returns = {}
        for trade in trades:
            if trade.strategy_id not in strategy_returns:
                strategy_returns[trade.strategy_id] = []

            # Simplified return calculation
            trade_return = np.random.normal(0.001, 0.02)  # Simulate trade return
            strategy_returns[trade.strategy_id].append(trade_return)

        strategy_contribution = {}
        for strategy_id, returns in strategy_returns.items():
            strategy_contribution[strategy_id] = np.sum(returns)

        # Factor attribution (simplified)
        factor_contribution = {
            "market": np.random.normal(0.05, 0.02),
            "size": np.random.normal(0.01, 0.01),
            "value": np.random.normal(0.02, 0.015),
            "momentum": np.random.normal(0.015, 0.01),
            "quality": np.random.normal(0.01, 0.008),
            "volatility": np.random.normal(-0.005, 0.01)
        }

        # Sector attribution
        sector_mapping = {
            'AAPL': 'Technology', 'GOOGL': 'Technology', 'MSFT': 'Technology',
            'JPM': 'Financials', 'BAC': 'Financials',
            'JNJ': 'Healthcare', 'PFE': 'Healthcare'
        }

        sector_contribution = {}
        for trade in trades:
            sector = sector_mapping.get(trade.symbol, 'Other')
            if sector not in sector_contribution:
                sector_contribution[sector] = 0
            sector_contribution[sector] += np.random.normal(0.001, 0.005)

        # Risk metrics
        alpha = np.random.normal(0.02, 0.01)  # 2% alpha
        beta = np.random.normal(1.0, 0.2)     # Beta around 1.0
        tracking_error = np.random.uniform(0.05, 0.15)  # 5-15% tracking error

        return PerformanceAttribution(
            strategy_contribution=strategy_contribution,
            factor_contribution=factor_contribution,
            sector_contribution=sector_contribution,
            alpha=alpha,
            beta=beta,
            tracking_error=tracking_error
        )

    async def _calculate_trade_statistics(self, trades: List[Trade]) -> Dict[str, Any]:
        """Calculate detailed trade statistics."""

        if not trades:
            return {}

        # Basic statistics
        total_trades = len(trades)
        buy_trades = sum(1 for trade in trades if trade.action == 'BUY')
        sell_trades = sum(1 for trade in trades if trade.action == 'SELL')

        # Volume statistics
        total_volume = sum(trade.quantity * trade.price for trade in trades)
        avg_trade_size = total_volume / total_trades

        # Commission and slippage analysis
        total_commissions = sum(trade.commission for trade in trades)
        total_slippage = sum(trade.quantity * trade.price * trade.slippage for trade in trades)

        # Symbol distribution
        symbol_counts = {}
        for trade in trades:
            symbol_counts[trade.symbol] = symbol_counts.get(trade.symbol, 0) + 1

        return {
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'total_volume': total_volume,
            'average_trade_size': avg_trade_size,
            'total_commissions': total_commissions,
            'total_slippage_cost': total_slippage,
            'commission_per_trade': total_commissions / total_trades,
            'slippage_per_trade': total_slippage / total_trades,
            'symbol_distribution': symbol_counts,
            'most_traded_symbol': max(symbol_counts.items(), key=lambda x: x[1])[0] if symbol_counts else None
        }

    async def _calculate_risk_metrics(self, trades: List[Trade], backtest: BacktestMessage) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics."""

        if not trades:
            return {}

        # Portfolio value series for risk calculations
        portfolio_values = await self._calculate_portfolio_series(trades, backtest.initial_capital)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # VaR calculations
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        var_99 = np.percentile(returns, 1) if len(returns) > 0 else 0

        # Expected Shortfall (CVaR)
        cvar_95 = np.mean(returns[returns <= var_95]) if len(returns[returns <= var_95]) > 0 else 0

        # Maximum consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        for ret in returns:
            if ret < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0

        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'max_consecutive_losses': max_consecutive_losses,
            'return_skewness': float(pd.Series(returns).skew()) if len(returns) > 2 else 0,
            'return_kurtosis': float(pd.Series(returns).kurtosis()) if len(returns) > 3 else 0,
            'downside_deviation': np.std(returns[returns < 0]) if len(returns[returns < 0]) > 1 else 0
        }

    async def _generate_comprehensive_reports(self, trades: List[Trade], performance_metrics: Dict[str, Any],
                                            attribution: PerformanceAttribution, backtest: BacktestMessage) -> Dict[str, Any]:
        """Generate comprehensive backtest reports."""

        # Executive summary
        executive_summary = {
            'total_return': f"{performance_metrics.get('total_return', 0):.2%}",
            'annual_return': f"{performance_metrics.get('annual_return', 0):.2%}",
            'sharpe_ratio': f"{performance_metrics.get('sharpe_ratio', 0):.2f}",
            'max_drawdown': f"{performance_metrics.get('max_drawdown', 0):.2%}",
            'win_rate': f"{performance_metrics.get('win_rate', 0):.2%}",
            'total_trades': performance_metrics.get('total_trades', 0),
            'recommendation': self._generate_recommendation(performance_metrics)
        }

        # Detailed analysis
        detailed_analysis = {
            'risk_analysis': {
                'volatility': f"{performance_metrics.get('annual_volatility', 0):.2%}",
                'calmar_ratio': f"{performance_metrics.get('calmar_ratio', 0):.2f}",
                'risk_adjusted_return': f"{performance_metrics.get('sharpe_ratio', 0):.2f}"
            },
            'cost_analysis': {
                'total_commissions': f"${performance_metrics.get('total_commissions', 0):.2f}",
                'total_slippage': f"${performance_metrics.get('total_slippage_cost', 0):.2f}",
                'cost_ratio': f"{(performance_metrics.get('total_commissions', 0) + performance_metrics.get('total_slippage_cost', 0)) / backtest.initial_capital:.2%}"
            },
            'attribution_summary': {
                'alpha': f"{attribution.alpha:.2%}",
                'beta': f"{attribution.beta:.2f}",
                'tracking_error': f"{attribution.tracking_error:.2%}",
                'top_strategy': max(attribution.strategy_contribution.items(), key=lambda x: x[1])[0] if attribution.strategy_contribution else 'N/A'
            }
        }

        # Performance benchmarking
        benchmark_comparison = {
            'vs_spy': {
                'outperformance': f"{performance_metrics.get('annual_return', 0) - 0.10:.2%}",  # Assume 10% SPY return
                'risk_adjusted_outperformance': f"{performance_metrics.get('sharpe_ratio', 0) - 0.8:.2f}"  # Assume 0.8 SPY Sharpe
            },
            'risk_metrics': {
                'information_ratio': f"{(performance_metrics.get('annual_return', 0) - 0.10) / attribution.tracking_error:.2f}",
                'sortino_ratio': f"{performance_metrics.get('annual_return', 0) / performance_metrics.get('downside_deviation', 0.01):.2f}"
            }
        }

        return {
            'executive_summary': executive_summary,
            'detailed_analysis': detailed_analysis,
            'benchmark_comparison': benchmark_comparison,
            'generated_at': datetime.now().isoformat(),
            'backtest_period': f"{backtest.start_date.date()} to {backtest.end_date.date()}"
        }

    def _generate_recommendation(self, performance_metrics: Dict[str, Any]) -> str:
        """Generate investment recommendation based on performance."""

        annual_return = performance_metrics.get('annual_return', 0)
        sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
        max_drawdown = performance_metrics.get('max_drawdown', 0)
        win_rate = performance_metrics.get('win_rate', 0)

        if annual_return > 0.15 and sharpe_ratio > 1.5 and abs(max_drawdown) < 0.15 and win_rate > 0.6:
            return "STRONG BUY - Excellent risk-adjusted returns with controlled drawdown"
        elif annual_return > 0.10 and sharpe_ratio > 1.0 and abs(max_drawdown) < 0.20:
            return "BUY - Good performance with acceptable risk"
        elif annual_return > 0.05 and sharpe_ratio > 0.5:
            return "HOLD - Moderate performance, monitor closely"
        elif annual_return > 0:
            return "WEAK HOLD - Below-average performance, consider improvements"
        else:
            return "SELL - Poor performance, strategy needs significant revision"

    def _get_empty_performance_metrics(self, backtest: BacktestMessage) -> Dict[str, Any]:
        """Return empty performance metrics for backtests with no trades."""
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'annual_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'calmar_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'total_trade_pairs': 0,
            'winning_trades': 0,
            'final_value': backtest.initial_capital,
            'total_commissions': 0.0,
            'total_slippage_cost': 0.0
        }

    def _trade_to_dict(self, trade: Trade) -> Dict[str, Any]:
        """Convert Trade object to dictionary."""
        return {
            'timestamp': trade.timestamp.isoformat(),
            'symbol': trade.symbol,
            'action': trade.action,
            'quantity': trade.quantity,
            'price': trade.price,
            'slippage': trade.slippage,
            'commission': trade.commission,
            'strategy_id': trade.strategy_id,
            'trade_id': trade.trade_id
        }
