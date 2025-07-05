"""
Advanced Risk Analytics Module for QuantAI.

Implements sophisticated portfolio-level risk metrics, stress testing,
and scenario analysis capabilities for multi-asset trading strategies.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from scipy.optimize import minimize
import json

logger = logging.getLogger(__name__)


class RiskScenario(str, Enum):
    """Risk scenario types for stress testing."""
    MARKET_CRASH = "market_crash"
    VOLATILITY_SPIKE = "volatility_spike"
    INTEREST_RATE_SHOCK = "interest_rate_shock"
    CURRENCY_CRISIS = "currency_crisis"
    SECTOR_ROTATION = "sector_rotation"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    BLACK_SWAN = "black_swan"


class StressTestType(str, Enum):
    """Types of stress tests."""
    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"
    PARAMETRIC = "parametric"
    SCENARIO_BASED = "scenario_based"


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics container."""
    timestamp: datetime
    portfolio_value: float
    
    # VaR metrics
    var_95: float
    var_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    
    # Volatility metrics
    realized_volatility: float
    implied_volatility: float
    volatility_of_volatility: float
    
    # Correlation metrics
    avg_correlation: float
    max_correlation: float
    correlation_stability: float
    
    # Concentration metrics
    herfindahl_index: float
    max_position_weight: float
    effective_positions: int
    
    # Factor exposures
    market_beta: float
    size_factor: float
    value_factor: float
    momentum_factor: float
    quality_factor: float
    
    # Tail risk metrics
    skewness: float
    kurtosis: float
    tail_ratio: float
    
    # Liquidity metrics
    liquidity_score: float
    days_to_liquidate: float


@dataclass
class StressTestResult:
    """Stress test result container."""
    scenario: RiskScenario
    test_type: StressTestType
    timestamp: datetime
    
    # Portfolio impact
    portfolio_pnl: float
    portfolio_pnl_pct: float
    max_drawdown: float
    
    # Position-level impacts
    position_impacts: Dict[str, float]
    worst_positions: List[Tuple[str, float]]
    best_positions: List[Tuple[str, float]]
    
    # Risk metric changes
    var_change: float
    correlation_change: float
    volatility_change: float
    
    # Recovery metrics
    recovery_time_days: Optional[int]
    probability_of_loss: float


class AdvancedRiskAnalytics:
    """
    Advanced Risk Analytics Engine.
    
    Provides sophisticated portfolio-level risk metrics, stress testing,
    and scenario analysis for multi-asset trading strategies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the advanced risk analytics engine."""
        self.config = config or {}
        
        # Risk calculation parameters
        self.confidence_levels = [0.95, 0.99]
        self.lookback_days = self.config.get("lookback_days", 252)
        self.monte_carlo_simulations = self.config.get("monte_carlo_simulations", 10000)
        
        # Historical data storage
        self.price_history: Dict[str, pd.DataFrame] = {}
        self.portfolio_history: List[Dict[str, Any]] = []
        self.risk_metrics_history: List[RiskMetrics] = []
        
        # Current portfolio state
        self.current_positions: Dict[str, Dict[str, Any]] = {}
        self.current_portfolio_value: float = 0.0
        
        # Stress test scenarios
        self.stress_scenarios = self._initialize_stress_scenarios()
        
        logger.info("Advanced Risk Analytics Engine initialized")
    
    def _initialize_stress_scenarios(self) -> Dict[RiskScenario, Dict[str, Any]]:
        """Initialize predefined stress test scenarios."""
        return {
            RiskScenario.MARKET_CRASH: {
                "equity_shock": -0.30,  # 30% equity decline
                "volatility_multiplier": 3.0,
                "correlation_increase": 0.3,
                "duration_days": 30,
                "description": "2008-style market crash scenario"
            },
            RiskScenario.VOLATILITY_SPIKE: {
                "volatility_multiplier": 2.5,
                "correlation_increase": 0.2,
                "duration_days": 14,
                "description": "VIX spike to 40+ scenario"
            },
            RiskScenario.INTEREST_RATE_SHOCK: {
                "rate_shock": 0.02,  # 200 bps increase
                "bond_duration_impact": -0.15,
                "currency_impact": 0.05,
                "duration_days": 60,
                "description": "Sudden 200bps rate increase"
            },
            RiskScenario.CURRENCY_CRISIS: {
                "usd_strength": 0.15,  # 15% USD appreciation
                "em_currency_weakness": -0.25,
                "volatility_multiplier": 2.0,
                "duration_days": 45,
                "description": "Emerging market currency crisis"
            },
            RiskScenario.SECTOR_ROTATION: {
                "tech_decline": -0.20,
                "value_outperformance": 0.15,
                "correlation_decrease": -0.2,
                "duration_days": 90,
                "description": "Major sector rotation event"
            },
            RiskScenario.LIQUIDITY_CRISIS: {
                "bid_ask_widening": 3.0,
                "volume_decline": -0.50,
                "correlation_increase": 0.4,
                "duration_days": 21,
                "description": "Market liquidity crisis"
            },
            RiskScenario.CORRELATION_BREAKDOWN: {
                "correlation_multiplier": 0.3,
                "volatility_multiplier": 1.5,
                "duration_days": 14,
                "description": "Diversification failure scenario"
            },
            RiskScenario.BLACK_SWAN: {
                "extreme_shock": -0.50,  # 50% decline
                "volatility_multiplier": 5.0,
                "correlation_spike": 0.9,
                "duration_days": 7,
                "description": "Extreme tail event"
            }
        }
    
    async def calculate_portfolio_risk_metrics(
        self, 
        positions: Dict[str, Dict[str, Any]],
        portfolio_value: float
    ) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics."""
        
        self.current_positions = positions
        self.current_portfolio_value = portfolio_value
        
        # Calculate position weights
        position_weights = self._calculate_position_weights(positions, portfolio_value)
        
        # Get historical returns for all positions
        returns_matrix = await self._get_returns_matrix(list(positions.keys()))
        
        if returns_matrix.empty:
            logger.warning("No historical data available for risk calculation")
            return self._create_default_risk_metrics()
        
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(returns_matrix, position_weights)
        
        # VaR calculations
        var_95, var_99 = self._calculate_var(portfolio_returns, portfolio_value)
        es_95, es_99 = self._calculate_expected_shortfall(portfolio_returns, portfolio_value)
        
        # Volatility metrics
        realized_vol = self._calculate_realized_volatility(portfolio_returns)
        implied_vol = await self._estimate_implied_volatility(positions)
        vol_of_vol = self._calculate_volatility_of_volatility(portfolio_returns)
        
        # Correlation metrics
        correlation_metrics = self._calculate_correlation_metrics(returns_matrix)
        
        # Concentration metrics
        concentration_metrics = self._calculate_concentration_metrics(position_weights)
        
        # Factor exposures
        factor_exposures = await self._calculate_factor_exposures(positions, returns_matrix)
        
        # Tail risk metrics
        tail_metrics = self._calculate_tail_risk_metrics(portfolio_returns)
        
        # Liquidity metrics
        liquidity_metrics = await self._calculate_liquidity_metrics(positions)
        
        risk_metrics = RiskMetrics(
            timestamp=datetime.now(),
            portfolio_value=portfolio_value,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall_95=es_95,
            expected_shortfall_99=es_99,
            realized_volatility=realized_vol,
            implied_volatility=implied_vol,
            volatility_of_volatility=vol_of_vol,
            avg_correlation=correlation_metrics["avg_correlation"],
            max_correlation=correlation_metrics["max_correlation"],
            correlation_stability=correlation_metrics["correlation_stability"],
            herfindahl_index=concentration_metrics["herfindahl_index"],
            max_position_weight=concentration_metrics["max_position_weight"],
            effective_positions=concentration_metrics["effective_positions"],
            market_beta=factor_exposures["market_beta"],
            size_factor=factor_exposures["size_factor"],
            value_factor=factor_exposures["value_factor"],
            momentum_factor=factor_exposures["momentum_factor"],
            quality_factor=factor_exposures["quality_factor"],
            skewness=tail_metrics["skewness"],
            kurtosis=tail_metrics["kurtosis"],
            tail_ratio=tail_metrics["tail_ratio"],
            liquidity_score=liquidity_metrics["liquidity_score"],
            days_to_liquidate=liquidity_metrics["days_to_liquidate"]
        )
        
        # Store in history
        self.risk_metrics_history.append(risk_metrics)
        
        # Keep only recent history
        if len(self.risk_metrics_history) > 1000:
            self.risk_metrics_history = self.risk_metrics_history[-1000:]
        
        logger.info(f"Calculated risk metrics: VaR95={var_95:.2%}, VaR99={var_99:.2%}")
        return risk_metrics
    
    def _calculate_position_weights(
        self, 
        positions: Dict[str, Dict[str, Any]], 
        portfolio_value: float
    ) -> Dict[str, float]:
        """Calculate position weights in the portfolio."""
        weights = {}
        
        for symbol, position in positions.items():
            position_value = position.get("market_value", 0)
            if portfolio_value > 0:
                weights[symbol] = position_value / portfolio_value
            else:
                weights[symbol] = 0.0
        
        return weights
    
    async def _get_returns_matrix(self, symbols: List[str]) -> pd.DataFrame:
        """Get historical returns matrix for symbols."""
        # In a real implementation, this would fetch actual price data
        # For now, simulate with random data
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=self.lookback_days),
            end=datetime.now(),
            freq='D'
        )
        
        returns_data = {}
        for symbol in symbols:
            # Simulate realistic returns with some correlation
            np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
            returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
            returns_data[symbol] = returns
        
        return pd.DataFrame(returns_data, index=dates)
    
    def _calculate_portfolio_returns(
        self, 
        returns_matrix: pd.DataFrame, 
        weights: Dict[str, float]
    ) -> pd.Series:
        """Calculate portfolio returns from individual asset returns."""
        portfolio_returns = pd.Series(0.0, index=returns_matrix.index)
        
        for symbol, weight in weights.items():
            if symbol in returns_matrix.columns:
                portfolio_returns += returns_matrix[symbol] * weight
        
        return portfolio_returns
    
    def _calculate_var(
        self, 
        returns: pd.Series, 
        portfolio_value: float
    ) -> Tuple[float, float]:
        """Calculate Value at Risk at 95% and 99% confidence levels."""
        if len(returns) < 30:
            return 0.0, 0.0
        
        var_95 = abs(np.percentile(returns, 5)) * portfolio_value
        var_99 = abs(np.percentile(returns, 1)) * portfolio_value
        
        return var_95, var_99

    def _calculate_expected_shortfall(
        self,
        returns: pd.Series,
        portfolio_value: float
    ) -> Tuple[float, float]:
        """Calculate Expected Shortfall (Conditional VaR)."""
        if len(returns) < 30:
            return 0.0, 0.0

        var_95_threshold = np.percentile(returns, 5)
        var_99_threshold = np.percentile(returns, 1)

        # Expected Shortfall is the average of losses beyond VaR
        es_95 = abs(returns[returns <= var_95_threshold].mean()) * portfolio_value
        es_99 = abs(returns[returns <= var_99_threshold].mean()) * portfolio_value

        return es_95, es_99

    def _calculate_realized_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized realized volatility."""
        if len(returns) < 30:
            return 0.0

        return returns.std() * np.sqrt(252)  # Annualized

    async def _estimate_implied_volatility(
        self,
        positions: Dict[str, Dict[str, Any]]
    ) -> float:
        """Estimate portfolio implied volatility from options positions."""
        # Simplified implementation - in practice would use actual options data
        total_weight = 0.0
        weighted_iv = 0.0

        for symbol, position in positions.items():
            if position.get("asset_type") == "options":
                weight = position.get("weight", 0.0)
                # Simulate implied volatility
                iv = 0.20 + np.random.normal(0, 0.05)  # Base 20% IV with noise
                weighted_iv += iv * weight
                total_weight += weight

        if total_weight > 0:
            return weighted_iv / total_weight
        else:
            return 0.20  # Default implied volatility

    def _calculate_volatility_of_volatility(self, returns: pd.Series) -> float:
        """Calculate volatility of volatility (vol of vol)."""
        if len(returns) < 60:
            return 0.0

        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=20).std()

        # Calculate volatility of the rolling volatility
        vol_of_vol = rolling_vol.std()

        return vol_of_vol * np.sqrt(252)  # Annualized

    def _calculate_correlation_metrics(
        self,
        returns_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate correlation-based risk metrics."""
        if returns_matrix.shape[1] < 2:
            return {
                "avg_correlation": 0.0,
                "max_correlation": 0.0,
                "correlation_stability": 1.0
            }

        # Calculate correlation matrix
        corr_matrix = returns_matrix.corr()

        # Extract upper triangle (excluding diagonal)
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        correlations = upper_triangle.stack().values

        avg_correlation = np.mean(correlations)
        max_correlation = np.max(correlations)

        # Calculate correlation stability (inverse of correlation volatility)
        if len(returns_matrix) > 60:
            rolling_corr = returns_matrix.rolling(window=30).corr()
            corr_volatility = rolling_corr.std().mean().mean()
            correlation_stability = 1.0 / (1.0 + corr_volatility)
        else:
            correlation_stability = 1.0

        return {
            "avg_correlation": avg_correlation,
            "max_correlation": max_correlation,
            "correlation_stability": correlation_stability
        }

    def _calculate_concentration_metrics(
        self,
        position_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate portfolio concentration metrics."""
        weights = list(position_weights.values())

        if not weights:
            return {
                "herfindahl_index": 0.0,
                "max_position_weight": 0.0,
                "effective_positions": 0
            }

        # Herfindahl-Hirschman Index
        hhi = sum(w**2 for w in weights)

        # Maximum position weight
        max_weight = max(weights) if weights else 0.0

        # Effective number of positions
        effective_positions = 1.0 / hhi if hhi > 0 else 0

        return {
            "herfindahl_index": hhi,
            "max_position_weight": max_weight,
            "effective_positions": int(effective_positions)
        }

    async def _calculate_factor_exposures(
        self,
        positions: Dict[str, Dict[str, Any]],
        returns_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate factor exposures (market, size, value, momentum, quality)."""

        # Simulate factor exposures based on position characteristics
        # In practice, would use actual factor models (Fama-French, etc.)

        total_value = sum(pos.get("market_value", 0) for pos in positions.values())

        if total_value == 0:
            return {
                "market_beta": 1.0,
                "size_factor": 0.0,
                "value_factor": 0.0,
                "momentum_factor": 0.0,
                "quality_factor": 0.0
            }

        # Weight-averaged factor exposures
        market_beta = 0.0
        size_factor = 0.0
        value_factor = 0.0
        momentum_factor = 0.0
        quality_factor = 0.0

        for symbol, position in positions.items():
            weight = position.get("market_value", 0) / total_value

            # Simulate factor loadings based on asset characteristics
            if position.get("asset_type") == "equity":
                beta = 1.0 + np.random.normal(0, 0.3)  # Market beta around 1.0
                size = np.random.normal(0, 0.5)        # Size factor
                value = np.random.normal(0, 0.3)       # Value factor
                momentum = np.random.normal(0, 0.4)    # Momentum factor
                quality = np.random.normal(0, 0.2)     # Quality factor
            elif position.get("asset_type") == "futures":
                beta = 1.5 + np.random.normal(0, 0.5)  # Higher beta for futures
                size = 0.0
                value = 0.0
                momentum = np.random.normal(0, 0.6)
                quality = 0.0
            elif position.get("asset_type") == "options":
                beta = 2.0 + np.random.normal(0, 1.0)  # High beta for options
                size = 0.0
                value = 0.0
                momentum = 0.0
                quality = 0.0
            else:  # forex, bonds, etc.
                beta = 0.5 + np.random.normal(0, 0.2)
                size = 0.0
                value = 0.0
                momentum = np.random.normal(0, 0.3)
                quality = 0.0

            market_beta += beta * weight
            size_factor += size * weight
            value_factor += value * weight
            momentum_factor += momentum * weight
            quality_factor += quality * weight

        return {
            "market_beta": market_beta,
            "size_factor": size_factor,
            "value_factor": value_factor,
            "momentum_factor": momentum_factor,
            "quality_factor": quality_factor
        }

    def _calculate_tail_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate tail risk metrics (skewness, kurtosis, tail ratio)."""
        if len(returns) < 30:
            return {
                "skewness": 0.0,
                "kurtosis": 3.0,
                "tail_ratio": 1.0
            }

        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns) + 3  # Add 3 for excess kurtosis

        # Tail ratio: ratio of average loss in worst 5% to average gain in best 5%
        worst_5pct = returns[returns <= np.percentile(returns, 5)]
        best_5pct = returns[returns >= np.percentile(returns, 95)]

        if len(worst_5pct) > 0 and len(best_5pct) > 0:
            tail_ratio = abs(worst_5pct.mean()) / best_5pct.mean()
        else:
            tail_ratio = 1.0

        return {
            "skewness": skewness,
            "kurtosis": kurtosis,
            "tail_ratio": tail_ratio
        }

    async def _calculate_liquidity_metrics(
        self,
        positions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate portfolio liquidity metrics."""

        total_value = sum(pos.get("market_value", 0) for pos in positions.values())

        if total_value == 0:
            return {
                "liquidity_score": 1.0,
                "days_to_liquidate": 0.0
            }

        weighted_liquidity = 0.0
        weighted_liquidation_time = 0.0

        for symbol, position in positions.items():
            weight = position.get("market_value", 0) / total_value

            # Assign liquidity scores based on asset type
            asset_type = position.get("asset_type", "equity")

            if asset_type == "equity":
                # Large cap stocks are most liquid
                liquidity_score = 0.9
                days_to_liquidate = 1.0
            elif asset_type == "etf":
                liquidity_score = 0.95
                days_to_liquidate = 0.5
            elif asset_type == "futures":
                liquidity_score = 0.85
                days_to_liquidate = 0.5
            elif asset_type == "options":
                liquidity_score = 0.7  # Options can be less liquid
                days_to_liquidate = 2.0
            elif asset_type == "forex":
                liquidity_score = 0.98  # FX is very liquid
                days_to_liquidate = 0.1
            else:
                liquidity_score = 0.6
                days_to_liquidate = 5.0

            weighted_liquidity += liquidity_score * weight
            weighted_liquidation_time += days_to_liquidate * weight

        return {
            "liquidity_score": weighted_liquidity,
            "days_to_liquidate": weighted_liquidation_time
        }

    def _create_default_risk_metrics(self) -> RiskMetrics:
        """Create default risk metrics when no data is available."""
        return RiskMetrics(
            timestamp=datetime.now(),
            portfolio_value=self.current_portfolio_value,
            var_95=0.0,
            var_99=0.0,
            expected_shortfall_95=0.0,
            expected_shortfall_99=0.0,
            realized_volatility=0.0,
            implied_volatility=0.20,
            volatility_of_volatility=0.0,
            avg_correlation=0.0,
            max_correlation=0.0,
            correlation_stability=1.0,
            herfindahl_index=0.0,
            max_position_weight=0.0,
            effective_positions=0,
            market_beta=1.0,
            size_factor=0.0,
            value_factor=0.0,
            momentum_factor=0.0,
            quality_factor=0.0,
            skewness=0.0,
            kurtosis=3.0,
            tail_ratio=1.0,
            liquidity_score=1.0,
            days_to_liquidate=0.0
        )

    async def run_stress_test(
        self,
        scenario: RiskScenario,
        test_type: StressTestType = StressTestType.PARAMETRIC,
        custom_parameters: Optional[Dict[str, Any]] = None
    ) -> StressTestResult:
        """Run stress test for specified scenario."""

        if not self.current_positions:
            logger.warning("No positions available for stress testing")
            return self._create_default_stress_result(scenario, test_type)

        scenario_params = self.stress_scenarios.get(scenario, {})
        if custom_parameters:
            scenario_params.update(custom_parameters)

        logger.info(f"Running {test_type.value} stress test for {scenario.value}")

        if test_type == StressTestType.PARAMETRIC:
            return await self._run_parametric_stress_test(scenario, scenario_params)
        elif test_type == StressTestType.MONTE_CARLO:
            return await self._run_monte_carlo_stress_test(scenario, scenario_params)
        elif test_type == StressTestType.HISTORICAL:
            return await self._run_historical_stress_test(scenario, scenario_params)
        else:
            return await self._run_scenario_based_stress_test(scenario, scenario_params)

    async def _run_parametric_stress_test(
        self,
        scenario: RiskScenario,
        params: Dict[str, Any]
    ) -> StressTestResult:
        """Run parametric stress test using predefined shocks."""

        position_impacts = {}
        total_portfolio_impact = 0.0

        for symbol, position in self.current_positions.items():
            asset_type = position.get("asset_type", "equity")
            market_value = position.get("market_value", 0)

            # Apply scenario-specific shocks
            shock = self._calculate_position_shock(asset_type, params)
            position_impact = market_value * shock

            position_impacts[symbol] = position_impact
            total_portfolio_impact += position_impact

        # Calculate portfolio-level metrics
        portfolio_pnl_pct = total_portfolio_impact / self.current_portfolio_value if self.current_portfolio_value > 0 else 0.0

        # Estimate risk metric changes
        var_change = abs(portfolio_pnl_pct) * 1.5  # VaR typically increases in stress
        correlation_change = params.get("correlation_increase", 0.0)
        volatility_change = params.get("volatility_multiplier", 1.0) - 1.0

        # Find worst and best performing positions
        sorted_impacts = sorted(position_impacts.items(), key=lambda x: x[1])
        worst_positions = sorted_impacts[:5]  # Top 5 worst
        best_positions = sorted_impacts[-5:]  # Top 5 best

        return StressTestResult(
            scenario=scenario,
            test_type=StressTestType.PARAMETRIC,
            timestamp=datetime.now(),
            portfolio_pnl=total_portfolio_impact,
            portfolio_pnl_pct=portfolio_pnl_pct,
            max_drawdown=abs(portfolio_pnl_pct),
            position_impacts=position_impacts,
            worst_positions=worst_positions,
            best_positions=best_positions,
            var_change=var_change,
            correlation_change=correlation_change,
            volatility_change=volatility_change,
            recovery_time_days=params.get("duration_days"),
            probability_of_loss=0.95 if portfolio_pnl_pct < 0 else 0.05
        )

    def _calculate_position_shock(self, asset_type: str, params: Dict[str, Any]) -> float:
        """Calculate position-specific shock based on asset type and scenario."""

        if asset_type == "equity":
            return params.get("equity_shock", -0.10)
        elif asset_type == "futures":
            # Futures typically have higher volatility
            base_shock = params.get("equity_shock", -0.10)
            return base_shock * 1.5
        elif asset_type == "options":
            # Options have non-linear responses
            base_shock = params.get("equity_shock", -0.10)
            volatility_impact = params.get("volatility_multiplier", 1.0) - 1.0
            return base_shock * 2.0 + volatility_impact * 0.5
        elif asset_type == "forex":
            return params.get("currency_impact", 0.05)
        else:
            return params.get("equity_shock", -0.10) * 0.5

    async def _run_monte_carlo_stress_test(
        self,
        scenario: RiskScenario,
        params: Dict[str, Any]
    ) -> StressTestResult:
        """Run Monte Carlo stress test with random simulations."""

        simulation_results = []

        for _ in range(self.monte_carlo_simulations):
            # Generate random shocks based on scenario parameters
            portfolio_shock = self._generate_random_shock(params)
            simulation_results.append(portfolio_shock)

        # Calculate statistics from simulations
        simulation_array = np.array(simulation_results)

        portfolio_pnl_pct = np.mean(simulation_array)
        portfolio_pnl = portfolio_pnl_pct * self.current_portfolio_value
        max_drawdown = abs(np.percentile(simulation_array, 5))  # 5th percentile

        # Estimate position impacts (simplified)
        position_impacts = {}
        for symbol, position in self.current_positions.items():
            market_value = position.get("market_value", 0)
            position_impacts[symbol] = market_value * portfolio_pnl_pct

        return StressTestResult(
            scenario=scenario,
            test_type=StressTestType.MONTE_CARLO,
            timestamp=datetime.now(),
            portfolio_pnl=portfolio_pnl,
            portfolio_pnl_pct=portfolio_pnl_pct,
            max_drawdown=max_drawdown,
            position_impacts=position_impacts,
            worst_positions=list(position_impacts.items())[:5],
            best_positions=list(position_impacts.items())[-5:],
            var_change=max_drawdown * 1.2,
            correlation_change=params.get("correlation_increase", 0.0),
            volatility_change=params.get("volatility_multiplier", 1.0) - 1.0,
            recovery_time_days=params.get("duration_days"),
            probability_of_loss=len([x for x in simulation_results if x < 0]) / len(simulation_results)
        )

    def _generate_random_shock(self, params: Dict[str, Any]) -> float:
        """Generate random shock for Monte Carlo simulation."""

        base_shock = params.get("equity_shock", -0.10)
        volatility = params.get("volatility_multiplier", 1.0) * 0.02

        # Generate random shock with specified mean and volatility
        return np.random.normal(base_shock, volatility)

    async def _run_historical_stress_test(
        self,
        scenario: RiskScenario,
        params: Dict[str, Any]
    ) -> StressTestResult:
        """Run historical stress test using past market events."""

        # In practice, would use actual historical data from crisis periods
        # For simulation, generate realistic historical scenario

        historical_shocks = self._get_historical_scenario_data(scenario)

        portfolio_impacts = []
        for shock in historical_shocks:
            impact = self._apply_historical_shock(shock)
            portfolio_impacts.append(impact)

        if portfolio_impacts:
            avg_impact = np.mean(portfolio_impacts)
            worst_impact = min(portfolio_impacts)
        else:
            avg_impact = -0.05
            worst_impact = -0.15

        portfolio_pnl = avg_impact * self.current_portfolio_value
        portfolio_pnl_pct = avg_impact
        max_drawdown = abs(worst_impact)

        # Calculate position impacts
        position_impacts = {}
        for symbol, position in self.current_positions.items():
            market_value = position.get("market_value", 0)
            position_impacts[symbol] = market_value * avg_impact

        return StressTestResult(
            scenario=scenario,
            test_type=StressTestType.HISTORICAL,
            timestamp=datetime.now(),
            portfolio_pnl=portfolio_pnl,
            portfolio_pnl_pct=portfolio_pnl_pct,
            max_drawdown=max_drawdown,
            position_impacts=position_impacts,
            worst_positions=list(position_impacts.items())[:5],
            best_positions=list(position_impacts.items())[-5:],
            var_change=max_drawdown * 1.3,
            correlation_change=0.3,  # Historical crises typically increase correlations
            volatility_change=1.5,   # Volatility typically doubles in crises
            recovery_time_days=params.get("duration_days", 90),
            probability_of_loss=0.85
        )

    def _get_historical_scenario_data(self, scenario: RiskScenario) -> List[float]:
        """Get historical data for scenario (simulated for now)."""

        if scenario == RiskScenario.MARKET_CRASH:
            # Simulate 2008 crisis daily returns
            return [-0.05, -0.08, -0.12, -0.03, -0.07, -0.15, -0.09, -0.04, -0.06, -0.11]
        elif scenario == RiskScenario.VOLATILITY_SPIKE:
            # Simulate VIX spike events
            return [-0.03, -0.06, -0.04, -0.08, -0.02, -0.05, -0.07]
        else:
            # Default historical shocks
            return [-0.02, -0.04, -0.03, -0.05, -0.01, -0.03]

    def _apply_historical_shock(self, shock: float) -> float:
        """Apply historical shock to current portfolio."""
        # Simplified - in practice would consider position-specific impacts
        return shock

    async def _run_scenario_based_stress_test(
        self,
        scenario: RiskScenario,
        params: Dict[str, Any]
    ) -> StressTestResult:
        """Run scenario-based stress test with custom logic."""

        # Combine parametric and Monte Carlo approaches
        parametric_result = await self._run_parametric_stress_test(scenario, params)

        # Add scenario-specific adjustments
        if scenario == RiskScenario.BLACK_SWAN:
            # Black swan events have extreme tail impacts
            parametric_result.portfolio_pnl *= 2.0
            parametric_result.portfolio_pnl_pct *= 2.0
            parametric_result.max_drawdown *= 2.0
            parametric_result.probability_of_loss = 0.99

        parametric_result.test_type = StressTestType.SCENARIO_BASED
        return parametric_result

    def _create_default_stress_result(
        self,
        scenario: RiskScenario,
        test_type: StressTestType
    ) -> StressTestResult:
        """Create default stress test result when no positions available."""

        return StressTestResult(
            scenario=scenario,
            test_type=test_type,
            timestamp=datetime.now(),
            portfolio_pnl=0.0,
            portfolio_pnl_pct=0.0,
            max_drawdown=0.0,
            position_impacts={},
            worst_positions=[],
            best_positions=[],
            var_change=0.0,
            correlation_change=0.0,
            volatility_change=0.0,
            recovery_time_days=None,
            probability_of_loss=0.0
        )

    async def run_comprehensive_stress_tests(self) -> Dict[RiskScenario, StressTestResult]:
        """Run all stress test scenarios and return comprehensive results."""

        results = {}

        for scenario in RiskScenario:
            try:
                result = await self.run_stress_test(scenario, StressTestType.PARAMETRIC)
                results[scenario] = result
                logger.info(f"Completed stress test for {scenario.value}: {result.portfolio_pnl_pct:.2%} impact")
            except Exception as e:
                logger.error(f"Error running stress test for {scenario.value}: {e}")
                results[scenario] = self._create_default_stress_result(scenario, StressTestType.PARAMETRIC)

        return results

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""

        if not self.risk_metrics_history:
            return {"error": "No risk metrics available"}

        latest_metrics = self.risk_metrics_history[-1]

        # Calculate risk trends if we have historical data
        risk_trends = {}
        if len(self.risk_metrics_history) > 1:
            prev_metrics = self.risk_metrics_history[-2]

            risk_trends = {
                "var_95_trend": latest_metrics.var_95 - prev_metrics.var_95,
                "volatility_trend": latest_metrics.realized_volatility - prev_metrics.realized_volatility,
                "correlation_trend": latest_metrics.avg_correlation - prev_metrics.avg_correlation,
                "concentration_trend": latest_metrics.herfindahl_index - prev_metrics.herfindahl_index
            }

        # Risk level assessment
        risk_level = self._assess_overall_risk_level(latest_metrics)

        return {
            "timestamp": latest_metrics.timestamp.isoformat(),
            "portfolio_value": latest_metrics.portfolio_value,
            "risk_level": risk_level,
            "key_metrics": {
                "var_95": latest_metrics.var_95,
                "var_99": latest_metrics.var_99,
                "expected_shortfall_95": latest_metrics.expected_shortfall_95,
                "realized_volatility": latest_metrics.realized_volatility,
                "max_correlation": latest_metrics.max_correlation,
                "concentration_risk": latest_metrics.herfindahl_index,
                "liquidity_score": latest_metrics.liquidity_score
            },
            "factor_exposures": {
                "market_beta": latest_metrics.market_beta,
                "size_factor": latest_metrics.size_factor,
                "value_factor": latest_metrics.value_factor,
                "momentum_factor": latest_metrics.momentum_factor,
                "quality_factor": latest_metrics.quality_factor
            },
            "tail_risk": {
                "skewness": latest_metrics.skewness,
                "kurtosis": latest_metrics.kurtosis,
                "tail_ratio": latest_metrics.tail_ratio
            },
            "trends": risk_trends
        }

    def _assess_overall_risk_level(self, metrics: RiskMetrics) -> str:
        """Assess overall portfolio risk level."""

        risk_score = 0

        # VaR assessment
        var_pct = metrics.var_95 / metrics.portfolio_value if metrics.portfolio_value > 0 else 0
        if var_pct > 0.03:
            risk_score += 3
        elif var_pct > 0.02:
            risk_score += 2
        elif var_pct > 0.01:
            risk_score += 1

        # Concentration assessment
        if metrics.herfindahl_index > 0.5:
            risk_score += 2
        elif metrics.herfindahl_index > 0.3:
            risk_score += 1

        # Correlation assessment
        if metrics.max_correlation > 0.8:
            risk_score += 2
        elif metrics.max_correlation > 0.6:
            risk_score += 1

        # Volatility assessment
        if metrics.realized_volatility > 0.3:
            risk_score += 2
        elif metrics.realized_volatility > 0.2:
            risk_score += 1

        # Liquidity assessment
        if metrics.liquidity_score < 0.7:
            risk_score += 2
        elif metrics.liquidity_score < 0.8:
            risk_score += 1

        # Risk level mapping
        if risk_score >= 8:
            return "CRITICAL"
        elif risk_score >= 6:
            return "HIGH"
        elif risk_score >= 4:
            return "MEDIUM"
        elif risk_score >= 2:
            return "LOW"
        else:
            return "MINIMAL"
