"""
Performance Attribution System for QuantAI.

This module provides comprehensive performance attribution analysis including:
- Strategy-level performance decomposition
- Factor-based attribution analysis
- Benchmark comparison and relative performance tracking
- Risk-adjusted performance metrics
- Sector and asset class attribution
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path
from loguru import logger


class AttributionType(str, Enum):
    """Types of performance attribution analysis."""
    STRATEGY = "strategy"
    FACTOR = "factor"
    SECTOR = "sector"
    ASSET_CLASS = "asset_class"
    SECURITY = "security"


class PerformanceMetric(str, Enum):
    """Performance metrics for attribution."""
    TOTAL_RETURN = "total_return"
    EXCESS_RETURN = "excess_return"
    ALPHA = "alpha"
    BETA = "beta"
    SHARPE_RATIO = "sharpe_ratio"
    INFORMATION_RATIO = "information_ratio"
    TRACKING_ERROR = "tracking_error"
    MAX_DRAWDOWN = "max_drawdown"


@dataclass
class AttributionResult:
    """Result of performance attribution analysis."""
    attribution_type: AttributionType
    period_start: datetime
    period_end: datetime
    total_return: float
    benchmark_return: float
    excess_return: float
    attribution_breakdown: Dict[str, float]
    risk_metrics: Dict[str, float]
    timestamp: datetime


@dataclass
class FactorExposure:
    """Factor exposure data for attribution."""
    factor_name: str
    exposure: float
    factor_return: float
    contribution: float
    t_stat: float
    p_value: float


@dataclass
class StrategyAttribution:
    """Strategy-level attribution results."""
    strategy_id: str
    strategy_name: str
    total_return: float
    benchmark_return: float
    excess_return: float
    alpha: float
    beta: float
    sharpe_ratio: float
    information_ratio: float
    tracking_error: float
    max_drawdown: float
    win_rate: float
    trades_count: int
    avg_trade_return: float
    best_trade: float
    worst_trade: float
    factor_exposures: List[FactorExposure]


class PerformanceAttributionEngine:
    """
    Advanced Performance Attribution Engine.
    
    Provides comprehensive performance attribution analysis across multiple dimensions
    including strategies, factors, sectors, and individual securities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the performance attribution engine."""
        self.config = config or {}
        
        # Attribution parameters
        self.lookback_periods = self.config.get('lookback_periods', [30, 90, 252, 504])  # Days
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)  # 2% annual
        self.benchmark_symbol = self.config.get('benchmark_symbol', 'SPY')
        
        # Factor model parameters
        self.factor_models = self.config.get('factor_models', {
            'fama_french_3': ['market', 'size', 'value'],
            'fama_french_5': ['market', 'size', 'value', 'profitability', 'investment'],
            'carhart_4': ['market', 'size', 'value', 'momentum'],
            'custom': ['market', 'volatility', 'quality', 'growth', 'momentum']
        })
        
        # Data storage
        self.returns_data: Dict[str, pd.Series] = {}
        self.factor_data: Dict[str, pd.DataFrame] = {}
        self.benchmark_data: Optional[pd.Series] = None
        self.attribution_history: List[AttributionResult] = []
        
        logger.info("Performance Attribution Engine initialized")
    
    async def load_returns_data(
        self, 
        returns_data: Dict[str, pd.Series],
        benchmark_returns: Optional[pd.Series] = None
    ):
        """Load returns data for attribution analysis."""
        self.returns_data = returns_data.copy()
        
        if benchmark_returns is not None:
            self.benchmark_data = benchmark_returns.copy()
        else:
            # Generate synthetic benchmark if not provided
            self.benchmark_data = self._generate_synthetic_benchmark()
        
        logger.info(f"Loaded returns data for {len(self.returns_data)} strategies")
    
    async def load_factor_data(self, factor_data: Dict[str, pd.DataFrame]):
        """Load factor data for factor-based attribution."""
        self.factor_data = factor_data.copy()
        logger.info(f"Loaded factor data: {list(factor_data.keys())}")
    
    async def calculate_strategy_attribution(
        self, 
        strategy_id: str,
        period_days: int = 252,
        factor_model: str = 'fama_french_3'
    ) -> StrategyAttribution:
        """Calculate comprehensive strategy attribution."""
        
        if strategy_id not in self.returns_data:
            raise ValueError(f"Strategy {strategy_id} not found in returns data")
        
        strategy_returns = self.returns_data[strategy_id]
        end_date = strategy_returns.index[-1]
        start_date = end_date - timedelta(days=period_days)
        
        # Filter data for the period
        period_returns = strategy_returns[strategy_returns.index >= start_date]
        period_benchmark = self.benchmark_data[self.benchmark_data.index >= start_date]
        
        # Align dates
        common_dates = period_returns.index.intersection(period_benchmark.index)
        period_returns = period_returns[common_dates]
        period_benchmark = period_benchmark[common_dates]
        
        # Calculate basic performance metrics
        total_return = (1 + period_returns).prod() - 1
        benchmark_return = (1 + period_benchmark).prod() - 1
        excess_return = total_return - benchmark_return
        
        # Calculate risk metrics
        alpha, beta = self._calculate_alpha_beta(period_returns, period_benchmark)
        sharpe_ratio = self._calculate_sharpe_ratio(period_returns)
        information_ratio = self._calculate_information_ratio(period_returns, period_benchmark)
        tracking_error = self._calculate_tracking_error(period_returns, period_benchmark)
        max_drawdown = self._calculate_max_drawdown(period_returns)
        
        # Calculate trade-level metrics
        win_rate = len(period_returns[period_returns > 0]) / len(period_returns)
        trades_count = len(period_returns)
        avg_trade_return = period_returns.mean()
        best_trade = period_returns.max()
        worst_trade = period_returns.min()
        
        # Calculate factor exposures
        factor_exposures = await self._calculate_factor_exposures(
            period_returns, factor_model
        )
        
        return StrategyAttribution(
            strategy_id=strategy_id,
            strategy_name=f"Strategy_{strategy_id}",
            total_return=total_return,
            benchmark_return=benchmark_return,
            excess_return=excess_return,
            alpha=alpha,
            beta=beta,
            sharpe_ratio=sharpe_ratio,
            information_ratio=information_ratio,
            tracking_error=tracking_error,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            trades_count=trades_count,
            avg_trade_return=avg_trade_return,
            best_trade=best_trade,
            worst_trade=worst_trade,
            factor_exposures=factor_exposures
        )
    
    async def calculate_factor_attribution(
        self,
        strategy_id: str,
        factor_model: str = 'fama_french_3',
        period_days: int = 252
    ) -> Dict[str, Any]:
        """Calculate factor-based performance attribution."""
        
        if strategy_id not in self.returns_data:
            raise ValueError(f"Strategy {strategy_id} not found in returns data")
        
        if factor_model not in self.factor_models:
            raise ValueError(f"Factor model {factor_model} not supported")
        
        strategy_returns = self.returns_data[strategy_id]
        end_date = strategy_returns.index[-1]
        start_date = end_date - timedelta(days=period_days)
        
        # Get factor returns for the period
        factor_returns = await self._get_factor_returns(factor_model, start_date, end_date)
        
        # Align dates
        common_dates = strategy_returns.index.intersection(factor_returns.index)
        strategy_returns = strategy_returns[common_dates]
        factor_returns = factor_returns.loc[common_dates]
        
        # Run factor regression
        factor_loadings, alpha, r_squared, residuals = self._run_factor_regression(
            strategy_returns, factor_returns
        )
        
        # Calculate factor contributions
        factor_contributions = {}
        total_factor_return = 0
        
        for factor in factor_returns.columns:
            factor_return = factor_returns[factor].mean() * len(common_dates)  # Annualized
            loading = factor_loadings.get(factor, 0)
            contribution = loading * factor_return
            factor_contributions[factor] = contribution
            total_factor_return += contribution
        
        # Calculate attribution breakdown
        total_return = (1 + strategy_returns).prod() - 1
        alpha_contribution = alpha * len(common_dates)  # Annualized alpha
        
        return {
            'strategy_id': strategy_id,
            'period_days': period_days,
            'total_return': total_return,
            'alpha': alpha_contribution,
            'factor_contributions': factor_contributions,
            'total_factor_return': total_factor_return,
            'residual_return': total_return - total_factor_return - alpha_contribution,
            'r_squared': r_squared,
            'factor_loadings': factor_loadings,
            'tracking_error': np.std(residuals) * np.sqrt(252)
        }
    
    async def calculate_sector_attribution(
        self,
        portfolio_positions: Dict[str, Dict[str, Any]],
        benchmark_weights: Dict[str, float],
        period_days: int = 30
    ) -> Dict[str, Any]:
        """Calculate sector-based attribution analysis."""
        
        # Group positions by sector
        sector_positions = {}
        sector_returns = {}
        
        for symbol, position in portfolio_positions.items():
            sector = position.get('sector', 'Unknown')
            weight = position.get('weight', 0)
            returns = position.get('returns', 0)
            
            if sector not in sector_positions:
                sector_positions[sector] = {'weight': 0, 'returns': []}
            
            sector_positions[sector]['weight'] += weight
            sector_positions[sector]['returns'].append(returns)
        
        # Calculate sector attribution
        attribution_results = {}
        total_allocation_effect = 0
        total_selection_effect = 0
        
        for sector, data in sector_positions.items():
            portfolio_weight = data['weight']
            benchmark_weight = benchmark_weights.get(sector, 0)
            sector_return = np.mean(data['returns']) if data['returns'] else 0
            benchmark_sector_return = benchmark_weight * 0.08  # Assume 8% benchmark return
            
            # Allocation effect: (portfolio_weight - benchmark_weight) * benchmark_sector_return
            allocation_effect = (portfolio_weight - benchmark_weight) * benchmark_sector_return
            
            # Selection effect: benchmark_weight * (sector_return - benchmark_sector_return)
            selection_effect = benchmark_weight * (sector_return - benchmark_sector_return)
            
            attribution_results[sector] = {
                'portfolio_weight': portfolio_weight,
                'benchmark_weight': benchmark_weight,
                'sector_return': sector_return,
                'allocation_effect': allocation_effect,
                'selection_effect': selection_effect,
                'total_effect': allocation_effect + selection_effect
            }
            
            total_allocation_effect += allocation_effect
            total_selection_effect += selection_effect
        
        return {
            'period_days': period_days,
            'sector_attribution': attribution_results,
            'total_allocation_effect': total_allocation_effect,
            'total_selection_effect': total_selection_effect,
            'total_attribution': total_allocation_effect + total_selection_effect
        }
    
    async def generate_attribution_report(
        self,
        strategy_ids: List[str],
        period_days: int = 252,
        include_factor_analysis: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive attribution report."""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'period_days': period_days,
            'strategies': {},
            'summary': {}
        }
        
        strategy_results = []
        
        for strategy_id in strategy_ids:
            try:
                # Calculate strategy attribution
                strategy_attr = await self.calculate_strategy_attribution(
                    strategy_id, period_days
                )
                
                strategy_data = {
                    'basic_metrics': asdict(strategy_attr),
                    'factor_attribution': None
                }
                
                # Add factor attribution if requested
                if include_factor_analysis:
                    factor_attr = await self.calculate_factor_attribution(
                        strategy_id, period_days=period_days
                    )
                    strategy_data['factor_attribution'] = factor_attr
                
                report['strategies'][strategy_id] = strategy_data
                strategy_results.append(strategy_attr)
                
            except Exception as e:
                logger.error(f"Error calculating attribution for {strategy_id}: {e}")
                report['strategies'][strategy_id] = {'error': str(e)}
        
        # Calculate summary statistics
        if strategy_results:
            total_returns = [s.total_return for s in strategy_results]
            excess_returns = [s.excess_return for s in strategy_results]
            sharpe_ratios = [s.sharpe_ratio for s in strategy_results]
            
            report['summary'] = {
                'total_strategies': len(strategy_results),
                'avg_total_return': np.mean(total_returns),
                'avg_excess_return': np.mean(excess_returns),
                'avg_sharpe_ratio': np.mean(sharpe_ratios),
                'best_strategy': max(strategy_results, key=lambda x: x.total_return).strategy_id,
                'worst_strategy': min(strategy_results, key=lambda x: x.total_return).strategy_id,
                'return_dispersion': np.std(total_returns)
            }
        
        return report
    
    def _calculate_alpha_beta(
        self, 
        strategy_returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> Tuple[float, float]:
        """Calculate alpha and beta using linear regression."""
        
        # Convert to excess returns
        rf_daily = self.risk_free_rate / 252
        strategy_excess = strategy_returns - rf_daily
        benchmark_excess = benchmark_returns - rf_daily
        
        # Linear regression
        covariance = np.cov(strategy_excess, benchmark_excess)[0, 1]
        benchmark_variance = np.var(benchmark_excess)
        
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
        alpha = np.mean(strategy_excess) - beta * np.mean(benchmark_excess)
        
        return alpha * 252, beta  # Annualized alpha
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        rf_daily = self.risk_free_rate / 252
        excess_returns = returns - rf_daily
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) != 0 else 0
    
    def _calculate_information_ratio(
        self, 
        strategy_returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> float:
        """Calculate Information ratio."""
        excess_returns = strategy_returns - benchmark_returns
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) != 0 else 0
    
    def _calculate_tracking_error(
        self, 
        strategy_returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> float:
        """Calculate tracking error."""
        excess_returns = strategy_returns - benchmark_returns
        return np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    async def _calculate_factor_exposures(
        self,
        returns: pd.Series,
        factor_model: str
    ) -> List[FactorExposure]:
        """Calculate factor exposures for a strategy."""
        
        factor_exposures = []
        
        # For now, generate synthetic factor exposures
        # In production, this would use actual factor data
        factors = self.factor_models.get(factor_model, [])
        
        for factor in factors:
            # Synthetic factor exposure calculation
            exposure = np.random.normal(0, 0.5)  # Random exposure
            factor_return = np.random.normal(0.08/252, 0.02/np.sqrt(252))  # Daily factor return
            contribution = exposure * factor_return * len(returns)
            
            factor_exposures.append(FactorExposure(
                factor_name=factor,
                exposure=exposure,
                factor_return=factor_return * 252,  # Annualized
                contribution=contribution,
                t_stat=abs(exposure) / 0.1,  # Synthetic t-stat
                p_value=0.05 if abs(exposure) > 0.2 else 0.15  # Synthetic p-value
            ))
        
        return factor_exposures
    
    async def _get_factor_returns(
        self,
        factor_model: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get factor returns for the specified period."""
        
        # Generate synthetic factor returns for testing
        # In production, this would load actual factor data
        factors = self.factor_models.get(factor_model, [])
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        factor_data = {}
        for factor in factors:
            # Generate synthetic factor returns
            returns = np.random.normal(0.08/252, 0.02/np.sqrt(252), len(date_range))
            factor_data[factor] = returns
        
        return pd.DataFrame(factor_data, index=date_range)
    
    def _run_factor_regression(
        self,
        strategy_returns: pd.Series,
        factor_returns: pd.DataFrame
    ) -> Tuple[Dict[str, float], float, float, pd.Series]:
        """Run factor regression analysis."""
        
        # Prepare data
        y = strategy_returns.values
        X = factor_returns.values
        
        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        # OLS regression
        try:
            coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            alpha = coefficients[0]
            factor_loadings = {
                factor: coefficients[i+1] 
                for i, factor in enumerate(factor_returns.columns)
            }
            
            # Calculate R-squared
            y_pred = X_with_intercept @ coefficients
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            residuals = pd.Series(y - y_pred, index=strategy_returns.index)
            
        except np.linalg.LinAlgError:
            # Handle singular matrix
            alpha = 0
            factor_loadings = {factor: 0 for factor in factor_returns.columns}
            r_squared = 0
            residuals = strategy_returns.copy()
        
        return factor_loadings, alpha, r_squared, residuals
    
    def _generate_synthetic_benchmark(self) -> pd.Series:
        """Generate synthetic benchmark returns for testing."""
        
        # Create a simple benchmark with realistic characteristics
        dates = pd.date_range(start='2020-01-01', end=datetime.now(), freq='D')
        returns = np.random.normal(0.08/252, 0.16/np.sqrt(252), len(dates))
        
        return pd.Series(returns, index=dates)
    
    async def save_attribution_results(
        self,
        results: Dict[str, Any],
        filename: Optional[str] = None
    ):
        """Save attribution results to file."""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"attribution_results_{timestamp}.json"

        filepath = Path("data/attribution") / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Attribution results saved to {filepath}")

    async def load_attribution_results(self, filename: str) -> Dict[str, Any]:
        """Load attribution results from file."""

        filepath = Path("data/attribution") / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Attribution file {filepath} not found")

        with open(filepath, 'r') as f:
            results = json.load(f)

        logger.info(f"Attribution results loaded from {filepath}")
        return results

    async def get_portfolio_performance_summary(
        self,
        account_id: Optional[str] = None,
        period_days: int = 30
    ) -> Dict[str, Any]:
        """Get portfolio performance summary for dashboard display."""

        summary = {
            'account_id': account_id,
            'period_days': period_days,
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': {},
            'top_performers': [],
            'underperformers': [],
            'risk_metrics': {},
            'attribution_summary': {}
        }

        if not self.returns_data:
            logger.warning("No returns data available for performance summary")
            return summary

        # Calculate performance for all strategies
        strategy_performances = []

        for strategy_id in self.returns_data.keys():
            try:
                strategy_attr = await self.calculate_strategy_attribution(
                    strategy_id, period_days
                )

                strategy_performances.append({
                    'strategy_id': strategy_id,
                    'total_return': strategy_attr.total_return,
                    'excess_return': strategy_attr.excess_return,
                    'sharpe_ratio': strategy_attr.sharpe_ratio,
                    'max_drawdown': strategy_attr.max_drawdown,
                    'win_rate': strategy_attr.win_rate
                })

            except Exception as e:
                logger.error(f"Error calculating performance for {strategy_id}: {e}")

        if strategy_performances:
            # Calculate aggregate metrics
            total_returns = [s['total_return'] for s in strategy_performances]
            excess_returns = [s['excess_return'] for s in strategy_performances]
            sharpe_ratios = [s['sharpe_ratio'] for s in strategy_performances]

            summary['performance_metrics'] = {
                'total_strategies': len(strategy_performances),
                'avg_total_return': np.mean(total_returns),
                'median_total_return': np.median(total_returns),
                'avg_excess_return': np.mean(excess_returns),
                'avg_sharpe_ratio': np.mean(sharpe_ratios),
                'return_volatility': np.std(total_returns),
                'positive_strategies': len([s for s in strategy_performances if s['total_return'] > 0])
            }

            # Identify top and bottom performers
            sorted_by_return = sorted(strategy_performances, key=lambda x: x['total_return'], reverse=True)
            summary['top_performers'] = sorted_by_return[:3]
            summary['underperformers'] = sorted_by_return[-3:]

            # Risk metrics summary
            drawdowns = [s['max_drawdown'] for s in strategy_performances]
            win_rates = [s['win_rate'] for s in strategy_performances]

            summary['risk_metrics'] = {
                'avg_max_drawdown': np.mean(drawdowns),
                'worst_drawdown': min(drawdowns),
                'avg_win_rate': np.mean(win_rates),
                'best_win_rate': max(win_rates),
                'strategies_with_positive_sharpe': len([s for s in strategy_performances if s['sharpe_ratio'] > 0])
            }

        return summary

    async def calculate_rolling_attribution(
        self,
        strategy_id: str,
        window_days: int = 30,
        step_days: int = 7
    ) -> List[Dict[str, Any]]:
        """Calculate rolling attribution analysis over time."""

        if strategy_id not in self.returns_data:
            raise ValueError(f"Strategy {strategy_id} not found in returns data")

        strategy_returns = self.returns_data[strategy_id]
        rolling_results = []

        # Calculate rolling windows
        end_date = strategy_returns.index[-1]
        current_date = strategy_returns.index[0] + timedelta(days=window_days)

        while current_date <= end_date:
            start_date = current_date - timedelta(days=window_days)

            # Get returns for this window
            window_returns = strategy_returns[
                (strategy_returns.index >= start_date) &
                (strategy_returns.index <= current_date)
            ]

            if len(window_returns) >= 20:  # Minimum observations
                try:
                    # Calculate basic metrics for this window
                    total_return = (1 + window_returns).prod() - 1
                    volatility = window_returns.std() * np.sqrt(252)
                    sharpe_ratio = self._calculate_sharpe_ratio(window_returns)
                    max_drawdown = self._calculate_max_drawdown(window_returns)

                    rolling_results.append({
                        'date': current_date.isoformat(),
                        'period_start': start_date.isoformat(),
                        'period_end': current_date.isoformat(),
                        'total_return': total_return,
                        'volatility': volatility,
                        'sharpe_ratio': sharpe_ratio,
                        'max_drawdown': max_drawdown,
                        'observations': len(window_returns)
                    })

                except Exception as e:
                    logger.warning(f"Error in rolling calculation for {current_date}: {e}")

            current_date += timedelta(days=step_days)

        return rolling_results
