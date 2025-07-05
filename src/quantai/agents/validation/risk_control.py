"""
Risk Control Agent (D2) for the QuantAI system.

This agent monitors exposure, leverage, volatility, drawdown and other
risk metrics in real-time to ensure portfolio safety.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from autogen_core import MessageContext
from autogen_core.models import ChatCompletionClient, UserMessage
from loguru import logger

from ...core.base import AgentCapability, AgentRole, ModelCapableAgent
from ...core.messages import MessageType, QuantMessage, RiskMessage, TradeMessage, ControlMessage
from ...core.account_manager import MultiAccountManager
from ...analytics.advanced_risk import AdvancedRiskAnalytics, RiskScenario, StressTestType


class RiskControlAgent(ModelCapableAgent):
    """
    Risk Control Agent (D2) - Monitors and controls portfolio risk.
    
    Capabilities:
    - Real-time risk monitoring and alerting
    - Position size and exposure control
    - Leverage and margin monitoring
    - Volatility and correlation tracking
    - Drawdown monitoring and circuit breakers
    - Risk-adjusted performance measurement
    """
    
    def __init__(
        self,
        model_client: ChatCompletionClient,
        account_manager: Optional[MultiAccountManager] = None,
        max_portfolio_risk: float = 0.02,  # 2% daily VaR
        max_position_size: float = 0.1,    # 10% per position
        max_leverage: float = 2.0,         # 2x leverage
        max_drawdown: float = 0.15,        # 15% max drawdown
        **kwargs
    ):
        super().__init__(
            role=AgentRole.RISK_CONTROL,
            capabilities=[
                AgentCapability.RISK_ASSESSMENT,
                AgentCapability.PERFORMANCE_MONITORING,
            ],
            model_client=model_client,
            system_message=self._get_system_message(),
            **kwargs
        )
        
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.max_drawdown = max_drawdown
        
        # Risk monitoring state
        self._portfolio_positions: Dict[str, Dict[str, Any]] = {}
        self._risk_metrics: Dict[str, float] = {}
        self._risk_alerts: List[Dict[str, Any]] = []
        self._performance_history: List[Dict[str, Any]] = []
        self._last_risk_check: Optional[datetime] = None
        
        # Risk limits and thresholds
        self._risk_limits = self._initialize_risk_limits()
        self._alert_thresholds = self._initialize_alert_thresholds()

        # Multi-account manager integration
        self.account_manager = account_manager or MultiAccountManager()

        # Advanced risk analytics engine
        self._advanced_risk = AdvancedRiskAnalytics(config={
            "lookback_days": 252,
            "monte_carlo_simulations": 5000
        })
    
    def _get_system_message(self) -> str:
        return """You are a Risk Control Agent responsible for comprehensive portfolio risk management.

Your responsibilities:
1. Monitor real-time portfolio risk metrics and exposures
2. Enforce position size limits and leverage constraints
3. Track volatility, correlation, and concentration risks
4. Monitor drawdown and implement circuit breakers
5. Generate risk alerts and recommendations
6. Assess risk-adjusted performance metrics

Risk Management Framework:

1. Position Risk
   - Individual position size limits
   - Sector and geographic concentration
   - Single name exposure limits
   - Correlation-adjusted position sizing

2. Portfolio Risk
   - Value at Risk (VaR) calculation
   - Expected Shortfall (ES) monitoring
   - Portfolio volatility tracking
   - Beta and factor exposures

3. Leverage and Margin
   - Gross and net leverage monitoring
   - Margin utilization tracking
   - Leverage ratio enforcement
   - Margin call prevention

4. Drawdown Control
   - Real-time drawdown monitoring
   - Peak-to-trough tracking
   - Circuit breaker implementation
   - Recovery time analysis

5. Market Risk
   - Delta, gamma, vega exposures
   - Interest rate sensitivity
   - Currency exposure monitoring
   - Commodity price risks

6. Operational Risk
   - Execution risk monitoring
   - Liquidity risk assessment
   - Model risk evaluation
   - Technology risk factors

Risk Metrics:
- Daily VaR (95%, 99% confidence)
- Maximum drawdown
- Sharpe ratio and Sortino ratio
- Beta and tracking error
- Concentration ratios
- Leverage ratios

Alert Levels:
- Green: Normal risk levels
- Yellow: Elevated risk - monitor closely
- Orange: High risk - reduce exposure
- Red: Critical risk - emergency action required

Guidelines:
- Prioritize capital preservation over returns
- Implement graduated risk responses
- Provide clear, actionable risk alerts
- Monitor both absolute and relative risks
- Consider regime changes in risk assessment
- Maintain detailed risk audit trails

Focus on preventing catastrophic losses while allowing for optimal risk-adjusted returns."""
    
    def _initialize_risk_limits(self) -> Dict[str, float]:
        """Initialize risk limits and constraints."""
        return {
            "max_daily_var_95": self.max_portfolio_risk,
            "max_position_size": self.max_position_size,
            "max_sector_exposure": 0.3,  # 30% per sector
            "max_leverage": self.max_leverage,
            "max_drawdown": self.max_drawdown,
            "max_correlation": 0.8,      # Max correlation between positions
            "min_liquidity_days": 5,     # Minimum days to liquidate
            "max_volatility": 0.25,      # 25% annualized volatility
        }
    
    def _initialize_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize alert thresholds for different risk levels."""
        return {
            "yellow": {  # Warning level
                "var_threshold": 0.75,      # 75% of limit
                "drawdown_threshold": 0.7,   # 70% of max drawdown
                "leverage_threshold": 0.8,   # 80% of max leverage
                "position_threshold": 0.8,   # 80% of max position
            },
            "orange": {  # High risk level
                "var_threshold": 0.9,       # 90% of limit
                "drawdown_threshold": 0.85,  # 85% of max drawdown
                "leverage_threshold": 0.95,  # 95% of max leverage
                "position_threshold": 0.95,  # 95% of max position
            },
            "red": {     # Critical level
                "var_threshold": 1.0,       # At limit
                "drawdown_threshold": 1.0,   # At max drawdown
                "leverage_threshold": 1.0,   # At max leverage
                "position_threshold": 1.0,   # At max position
            }
        }
    
    async def process_message(
        self, 
        message: QuantMessage, 
        ctx: MessageContext
    ) -> Optional[QuantMessage]:
        """Process risk monitoring and control messages."""
        
        if isinstance(message, TradeMessage):
            # Monitor trade execution for risk
            await self._monitor_trade_risk(message)
            
        elif isinstance(message, RiskMessage):
            # Process risk assessment requests
            risk_response = await self._assess_portfolio_risk(message)
            return risk_response
        
        # Perform periodic risk checks
        await self._periodic_risk_check()
        
        return None
    
    async def _monitor_trade_risk(self, trade_message: TradeMessage):
        """Monitor individual trade for risk compliance."""
        
        symbol = trade_message.symbol
        action = trade_message.action
        quantity = trade_message.quantity
        
        # Update position tracking
        if symbol not in self._portfolio_positions:
            self._portfolio_positions[symbol] = {
                "quantity": 0,
                "value": 0,
                "last_price": 0,
                "entry_time": datetime.now(),
            }
        
        position = self._portfolio_positions[symbol]
        
        if action == "BUY":
            position["quantity"] += quantity
        elif action == "SELL":
            position["quantity"] -= quantity
        
        # Check position size limits
        await self._check_position_limits(symbol, position)
        
        # Update portfolio risk metrics
        await self._update_risk_metrics()
    
    async def _check_position_limits(self, symbol: str, position: Dict[str, Any]):
        """Check if position violates size limits."""
        
        position_value = abs(position["quantity"] * position.get("last_price", 0))
        portfolio_value = self._calculate_portfolio_value()
        
        if portfolio_value > 0:
            position_weight = position_value / portfolio_value
            
            if position_weight > self._risk_limits["max_position_size"]:
                alert = {
                    "type": "position_limit_breach",
                    "symbol": symbol,
                    "current_weight": position_weight,
                    "limit": self._risk_limits["max_position_size"],
                    "severity": "red",
                    "timestamp": datetime.now(),
                    "action_required": "Reduce position size immediately",
                }
                
                await self._generate_risk_alert(alert)
    
    async def _update_risk_metrics(self):
        """Update portfolio risk metrics."""
        
        # Calculate portfolio value
        portfolio_value = self._calculate_portfolio_value()
        
        # Calculate leverage
        gross_exposure = sum(
            abs(pos["quantity"] * pos.get("last_price", 0))
            for pos in self._portfolio_positions.values()
        )
        leverage = gross_exposure / portfolio_value if portfolio_value > 0 else 0
        
        # Calculate concentration
        max_position_weight = 0
        if portfolio_value > 0:
            position_weights = [
                abs(pos["quantity"] * pos.get("last_price", 0)) / portfolio_value
                for pos in self._portfolio_positions.values()
            ]
            max_position_weight = max(position_weights) if position_weights else 0
        
        # Update metrics
        self._risk_metrics.update({
            "portfolio_value": portfolio_value,
            "gross_exposure": gross_exposure,
            "leverage": leverage,
            "max_position_weight": max_position_weight,
            "position_count": len(self._portfolio_positions),
            "last_updated": datetime.now().timestamp(),
        })
        
        # Check risk limits
        await self._check_risk_limits()
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        return sum(
            pos["quantity"] * pos.get("last_price", 0)
            for pos in self._portfolio_positions.values()
        )
    
    async def _check_risk_limits(self):
        """Check all risk limits and generate alerts."""
        
        # Check leverage limit
        current_leverage = self._risk_metrics.get("leverage", 0)
        if current_leverage > self._risk_limits["max_leverage"]:
            await self._generate_leverage_alert(current_leverage)
        
        # Check position concentration
        max_position = self._risk_metrics.get("max_position_weight", 0)
        if max_position > self._risk_limits["max_position_size"]:
            await self._generate_concentration_alert(max_position)
        
        # Check drawdown (would need historical data)
        # This is simplified - in practice would track high water mark
        current_drawdown = self._calculate_current_drawdown()
        if current_drawdown > self._risk_limits["max_drawdown"]:
            await self._generate_drawdown_alert(current_drawdown)
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        # Simplified calculation - in practice would track historical peaks
        if not self._performance_history:
            return 0.0
        
        current_value = self._risk_metrics.get("portfolio_value", 0)
        peak_value = max(
            entry.get("portfolio_value", 0) 
            for entry in self._performance_history[-252:]  # Last year
        )
        
        if peak_value > 0:
            return (peak_value - current_value) / peak_value
        return 0.0
    
    async def _generate_leverage_alert(self, current_leverage: float):
        """Generate leverage limit alert."""
        
        severity = self._determine_alert_severity(
            current_leverage, self._risk_limits["max_leverage"], "leverage"
        )
        
        alert = {
            "type": "leverage_limit",
            "current_value": current_leverage,
            "limit": self._risk_limits["max_leverage"],
            "severity": severity,
            "timestamp": datetime.now(),
            "action_required": "Reduce gross exposure to lower leverage",
        }
        
        await self._generate_risk_alert(alert)
    
    async def _generate_concentration_alert(self, max_position: float):
        """Generate position concentration alert."""
        
        severity = self._determine_alert_severity(
            max_position, self._risk_limits["max_position_size"], "position"
        )
        
        alert = {
            "type": "position_concentration",
            "current_value": max_position,
            "limit": self._risk_limits["max_position_size"],
            "severity": severity,
            "timestamp": datetime.now(),
            "action_required": "Reduce largest position size",
        }
        
        await self._generate_risk_alert(alert)
    
    async def _generate_drawdown_alert(self, current_drawdown: float):
        """Generate drawdown alert."""
        
        severity = self._determine_alert_severity(
            current_drawdown, self._risk_limits["max_drawdown"], "drawdown"
        )
        
        alert = {
            "type": "drawdown_limit",
            "current_value": current_drawdown,
            "limit": self._risk_limits["max_drawdown"],
            "severity": severity,
            "timestamp": datetime.now(),
            "action_required": "Consider reducing overall exposure",
        }
        
        await self._generate_risk_alert(alert)
    
    def _determine_alert_severity(self, current: float, limit: float, metric_type: str) -> str:
        """Determine alert severity based on thresholds."""
        
        ratio = current / limit
        
        thresholds = self._alert_thresholds
        
        if ratio >= thresholds["red"][f"{metric_type}_threshold"]:
            return "red"
        elif ratio >= thresholds["orange"][f"{metric_type}_threshold"]:
            return "orange"
        elif ratio >= thresholds["yellow"][f"{metric_type}_threshold"]:
            return "yellow"
        else:
            return "green"
    
    async def _generate_risk_alert(self, alert: Dict[str, Any]):
        """Generate and broadcast risk alert."""
        
        self._risk_alerts.append(alert)
        
        # Keep only recent alerts (last 100)
        if len(self._risk_alerts) > 100:
            self._risk_alerts.pop(0)
        
        logger.warning(f"Risk alert: {alert['type']} - {alert['severity']}")
        
        # If critical, send emergency message
        if alert["severity"] == "red":
            await self._send_emergency_alert(alert)
    
    async def _send_emergency_alert(self, alert: Dict[str, Any]):
        """Send emergency alert for critical risk situations."""
        
        emergency_message = ControlMessage(
            message_type=MessageType.KILL_SWITCH,
            sender_id=self.agent_id,
            command="risk_emergency",
            parameters={
                "alert": alert,
                "risk_metrics": self._risk_metrics,
                "action": "immediate_risk_reduction_required",
            }
        )
        
        # This would be sent to the kill switch agent or emergency system
        logger.critical(f"Emergency risk alert: {alert}")
    
    async def _assess_portfolio_risk(self, risk_message: RiskMessage) -> RiskMessage:
        """Assess overall portfolio risk and generate response."""
        
        # Calculate comprehensive risk metrics
        risk_assessment = await self._calculate_comprehensive_risk()
        
        # Use LLM for risk analysis
        llm_analysis = await self._llm_risk_analysis(risk_assessment)
        
        response = RiskMessage(
            message_type=MessageType.RISK_ASSESSMENT,
            sender_id=self.agent_id,
            portfolio_id=risk_message.portfolio_id,
            risk_metrics=risk_assessment,
            exposure_limits=self._risk_limits,
            current_exposure=self._get_current_exposures(),
            risk_alerts=[alert["type"] for alert in self._risk_alerts[-10:]],  # Recent alerts
            var_95=risk_assessment.get("var_95", 0),
            max_drawdown=risk_assessment.get("current_drawdown", 0),
            sharpe_ratio=risk_assessment.get("sharpe_ratio", 0),
            session_id=risk_message.session_id,
            correlation_id=risk_message.correlation_id,
        )
        
        return response
    
    async def _calculate_comprehensive_risk(self) -> Dict[str, float]:
        """Calculate comprehensive risk metrics with advanced analytics."""

        portfolio_value = self._risk_metrics.get("portfolio_value", 0)
        leverage = self._risk_metrics.get("leverage", 0)
        max_position = self._risk_metrics.get("max_position_weight", 0)
        current_drawdown = self._calculate_current_drawdown()

        # Advanced VaR calculation with correlation adjustments
        correlation_adjusted_var = await self._calculate_correlation_adjusted_var()

        # Sector concentration risk
        sector_concentration = await self._calculate_sector_concentration()

        # Dynamic VaR based on market conditions
        dynamic_var = await self._calculate_dynamic_var()

        # Portfolio beta and factor exposures
        portfolio_beta = await self._calculate_portfolio_beta()
        factor_exposures = await self._calculate_factor_exposures()
        
        return {
            "portfolio_value": portfolio_value,
            "leverage": leverage,
            "max_position_weight": max_position,
            "current_drawdown": current_drawdown,
            "var_95": dynamic_var,
            "correlation_adjusted_var": correlation_adjusted_var,
            "sector_concentration": sector_concentration,
            "portfolio_beta": portfolio_beta,
            "factor_exposures": factor_exposures,
            "risk_score": min(1.0, (leverage + max_position + current_drawdown + sector_concentration) / 4)
        }
    
    def _get_current_exposures(self) -> Dict[str, float]:
        """Get current portfolio exposures."""
        
        portfolio_value = self._risk_metrics.get("portfolio_value", 0)
        
        exposures = {}
        for symbol, position in self._portfolio_positions.items():
            if portfolio_value > 0:
                exposure = (position["quantity"] * position.get("last_price", 0)) / portfolio_value
                exposures[symbol] = exposure
        
        return exposures

    async def _calculate_correlation_adjusted_var(self) -> float:
        """Calculate VaR adjusted for position correlations."""
        try:
            positions = list(self._portfolio_positions.values())
            if len(positions) < 2:
                return self._risk_metrics.get("var_95", 0.0)

            # Simplified correlation matrix (in production, use historical data)
            correlation_matrix = np.eye(len(positions))
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    # Assume moderate correlation between positions
                    correlation_matrix[i][j] = correlation_matrix[j][i] = 0.3

            # Position weights and volatilities
            weights = np.array([pos.get("value", 0) for pos in positions])
            total_value = np.sum(weights)
            if total_value > 0:
                weights = weights / total_value

            volatilities = np.array([0.15 for _ in positions])  # 15% assumed volatility

            # Portfolio variance calculation
            portfolio_variance = np.dot(weights, np.dot(correlation_matrix * np.outer(volatilities, volatilities), weights))
            portfolio_volatility = np.sqrt(portfolio_variance)

            # 95% VaR (1.65 standard deviations)
            return total_value * portfolio_volatility * 1.65 / np.sqrt(252)

        except Exception as e:
            logger.error(f"Error calculating correlation-adjusted VaR: {e}")
            return self._risk_metrics.get("var_95", 0.0)

    async def _calculate_sector_concentration(self) -> float:
        """Calculate sector concentration risk using Herfindahl-Hirschman Index."""
        try:
            sector_exposures = {}
            total_value = sum(pos.get("value", 0) for pos in self._portfolio_positions.values())

            if total_value == 0:
                return 0.0

            # Map positions to sectors (simplified)
            sector_mapping = {
                "SPY": "broad_market", "QQQ": "technology", "IWM": "small_cap",
                "XLF": "financials", "XLE": "energy", "XLK": "technology",
                "XLV": "healthcare", "XLI": "industrials", "XLP": "consumer_staples"
            }

            for symbol, position in self._portfolio_positions.items():
                sector = sector_mapping.get(symbol.upper(), "other")
                weight = position.get("value", 0) / total_value
                sector_exposures[sector] = sector_exposures.get(sector, 0) + weight

            # Calculate Herfindahl-Hirschman Index
            hhi = sum(weight**2 for weight in sector_exposures.values())

            # Normalize to 0-1 scale (1 = maximum concentration)
            return min(1.0, hhi)

        except Exception as e:
            logger.error(f"Error calculating sector concentration: {e}")
            return 0.0

    async def _calculate_dynamic_var(self) -> float:
        """Calculate dynamic VaR based on current market conditions."""
        try:
            base_var = self._risk_metrics.get("var_95", 0.0)

            # Market volatility adjustment (simplified)
            # In production, would use VIX or realized volatility
            market_vol_multiplier = 1.0

            # Time-of-day adjustment
            current_hour = datetime.now().hour
            if 9 <= current_hour <= 16:  # Market hours
                time_multiplier = 1.2  # Higher risk during market hours
            else:
                time_multiplier = 0.8  # Lower risk after hours

            # Position concentration adjustment
            concentration_risk = await self._calculate_sector_concentration()
            concentration_multiplier = 1.0 + concentration_risk * 0.5

            return base_var * market_vol_multiplier * time_multiplier * concentration_multiplier

        except Exception as e:
            logger.error(f"Error calculating dynamic VaR: {e}")
            return self._risk_metrics.get("var_95", 0.0)

    async def _calculate_portfolio_beta(self) -> float:
        """Calculate portfolio beta relative to market benchmark."""
        try:
            # Simplified beta calculation (in production, use regression analysis)
            total_value = sum(pos.get("value", 0) for pos in self._portfolio_positions.values())
            if total_value == 0:
                return 1.0

            weighted_beta = 0.0
            beta_mapping = {
                "SPY": 1.0, "QQQ": 1.2, "IWM": 1.3,
                "XLF": 1.1, "XLE": 1.4, "XLK": 1.3,
                "TLT": 0.2, "GLD": 0.1, "VIX": -0.5
            }

            for symbol, position in self._portfolio_positions.items():
                weight = position.get("value", 0) / total_value
                beta = beta_mapping.get(symbol.upper(), 1.0)
                weighted_beta += weight * beta

            return weighted_beta

        except Exception as e:
            logger.error(f"Error calculating portfolio beta: {e}")
            return 1.0

    async def _calculate_factor_exposures(self) -> Dict[str, float]:
        """Calculate factor exposures (style, sector, etc.)."""
        try:
            total_value = sum(pos.get("value", 0) for pos in self._portfolio_positions.values())
            if total_value == 0:
                return {}

            factor_exposures = {
                "value": 0.0, "growth": 0.0, "momentum": 0.0,
                "quality": 0.0, "size": 0.0, "volatility": 0.0
            }

            # Simplified factor mapping
            factor_mapping = {
                "SPY": {"value": 0.0, "growth": 0.0, "momentum": 0.0, "quality": 0.0, "size": 0.0, "volatility": 0.0},
                "QQQ": {"value": -0.5, "growth": 0.8, "momentum": 0.3, "quality": 0.2, "size": -0.3, "volatility": 0.2},
                "IWM": {"value": 0.3, "growth": -0.2, "momentum": 0.1, "quality": -0.3, "size": 0.8, "volatility": 0.4}
            }

            for symbol, position in self._portfolio_positions.items():
                weight = position.get("value", 0) / total_value
                factors = factor_mapping.get(symbol.upper(), {})

                for factor, exposure in factors.items():
                    factor_exposures[factor] += weight * exposure

            return factor_exposures

        except Exception as e:
            logger.error(f"Error calculating factor exposures: {e}")
            return {}

    async def _llm_risk_analysis(self, risk_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Use LLM for comprehensive risk analysis."""
        
        prompt = f"""Analyze the following portfolio risk metrics and provide recommendations:

Risk Metrics:
{json.dumps(risk_metrics, indent=2)}

Risk Limits:
{json.dumps(self._risk_limits, indent=2)}

Recent Alerts:
{json.dumps(self._risk_alerts[-5:], indent=2, default=str)}

Provide analysis including:
1. Overall risk assessment
2. Key risk concerns
3. Recommended actions
4. Risk-return optimization suggestions

Respond in JSON format:
{{
    "overall_risk_level": "low/medium/high/critical",
    "key_concerns": ["concern1", "concern2"],
    "recommended_actions": ["action1", "action2"],
    "risk_score": 0.0-1.0,
    "optimization_suggestions": ["suggestion1", "suggestion2"]
}}"""
        
        try:
            response = await self._call_model([UserMessage(content=prompt, source="user")])
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error in LLM risk analysis: {e}")
            return {"error": str(e)}

    async def calculate_advanced_risk_metrics(self) -> Dict[str, Any]:
        """Calculate advanced portfolio risk metrics using the analytics engine."""

        try:
            # Get current portfolio positions
            positions = self._portfolio_positions
            portfolio_value = sum(pos.get("market_value", 0) for pos in positions.values())

            if not positions or portfolio_value == 0:
                logger.warning("No positions available for advanced risk calculation")
                return {"error": "No positions available"}

            # Calculate comprehensive risk metrics
            risk_metrics = await self._advanced_risk.calculate_portfolio_risk_metrics(
                positions, portfolio_value
            )

            # Convert to dictionary for messaging
            return {
                "timestamp": risk_metrics.timestamp.isoformat(),
                "portfolio_value": risk_metrics.portfolio_value,
                "var_metrics": {
                    "var_95": risk_metrics.var_95,
                    "var_99": risk_metrics.var_99,
                    "expected_shortfall_95": risk_metrics.expected_shortfall_95,
                    "expected_shortfall_99": risk_metrics.expected_shortfall_99
                },
                "volatility_metrics": {
                    "realized_volatility": risk_metrics.realized_volatility,
                    "implied_volatility": risk_metrics.implied_volatility,
                    "volatility_of_volatility": risk_metrics.volatility_of_volatility
                },
                "correlation_metrics": {
                    "avg_correlation": risk_metrics.avg_correlation,
                    "max_correlation": risk_metrics.max_correlation,
                    "correlation_stability": risk_metrics.correlation_stability
                },
                "concentration_metrics": {
                    "herfindahl_index": risk_metrics.herfindahl_index,
                    "max_position_weight": risk_metrics.max_position_weight,
                    "effective_positions": risk_metrics.effective_positions
                },
                "factor_exposures": {
                    "market_beta": risk_metrics.market_beta,
                    "size_factor": risk_metrics.size_factor,
                    "value_factor": risk_metrics.value_factor,
                    "momentum_factor": risk_metrics.momentum_factor,
                    "quality_factor": risk_metrics.quality_factor
                },
                "tail_risk": {
                    "skewness": risk_metrics.skewness,
                    "kurtosis": risk_metrics.kurtosis,
                    "tail_ratio": risk_metrics.tail_ratio
                },
                "liquidity_metrics": {
                    "liquidity_score": risk_metrics.liquidity_score,
                    "days_to_liquidate": risk_metrics.days_to_liquidate
                }
            }

        except Exception as e:
            logger.error(f"Error calculating advanced risk metrics: {e}")
            return {"error": str(e)}

    async def run_stress_test(
        self,
        scenario: str = "market_crash",
        test_type: str = "parametric"
    ) -> Dict[str, Any]:
        """Run stress test for specified scenario."""

        try:
            # Convert string parameters to enums
            risk_scenario = RiskScenario(scenario)
            stress_test_type = StressTestType(test_type)

            # Run stress test
            result = await self._advanced_risk.run_stress_test(
                risk_scenario, stress_test_type
            )

            # Convert result to dictionary
            return {
                "scenario": result.scenario.value,
                "test_type": result.test_type.value,
                "timestamp": result.timestamp.isoformat(),
                "portfolio_impact": {
                    "pnl": result.portfolio_pnl,
                    "pnl_percentage": result.portfolio_pnl_pct,
                    "max_drawdown": result.max_drawdown
                },
                "position_impacts": result.position_impacts,
                "worst_positions": result.worst_positions,
                "best_positions": result.best_positions,
                "risk_changes": {
                    "var_change": result.var_change,
                    "correlation_change": result.correlation_change,
                    "volatility_change": result.volatility_change
                },
                "recovery_metrics": {
                    "recovery_time_days": result.recovery_time_days,
                    "probability_of_loss": result.probability_of_loss
                }
            }

        except Exception as e:
            logger.error(f"Error running stress test: {e}")
            return {"error": str(e)}

    async def run_comprehensive_stress_tests(self) -> Dict[str, Any]:
        """Run all stress test scenarios."""

        try:
            results = await self._advanced_risk.run_comprehensive_stress_tests()

            # Convert results to dictionary format
            stress_results = {}
            for scenario, result in results.items():
                stress_results[scenario.value] = {
                    "portfolio_pnl_pct": result.portfolio_pnl_pct,
                    "max_drawdown": result.max_drawdown,
                    "probability_of_loss": result.probability_of_loss,
                    "recovery_time_days": result.recovery_time_days
                }

            return {
                "timestamp": datetime.now().isoformat(),
                "stress_test_results": stress_results,
                "summary": {
                    "worst_scenario": min(stress_results.items(), key=lambda x: x[1]["portfolio_pnl_pct"]),
                    "avg_impact": sum(r["portfolio_pnl_pct"] for r in stress_results.values()) / len(stress_results),
                    "scenarios_with_losses": len([r for r in stress_results.values() if r["portfolio_pnl_pct"] < 0])
                }
            }

        except Exception as e:
            logger.error(f"Error running comprehensive stress tests: {e}")
            return {"error": str(e)}

    async def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary including advanced analytics."""

        try:
            # Get basic risk summary
            basic_summary = await self._assess_portfolio_risk(
                RiskMessage(
                    message_type=MessageType.RISK_ASSESSMENT,
                    sender_id="risk_control",
                    content={"request": "portfolio_risk_summary"}
                )
            )

            # Get advanced risk summary
            advanced_summary = self._advanced_risk.get_risk_summary()

            # Combine summaries
            return {
                "basic_risk_metrics": basic_summary.content if hasattr(basic_summary, 'content') else {},
                "advanced_risk_metrics": advanced_summary,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting risk summary: {e}")
            return {"error": str(e)}
    
    async def _periodic_risk_check(self):
        """Perform periodic risk monitoring."""
        
        now = datetime.now()
        
        # Check if it's time for periodic update
        if (self._last_risk_check is None or 
            now - self._last_risk_check > timedelta(minutes=5)):
            
            await self._update_risk_metrics()
            self._last_risk_check = now
    
    async def get_risk_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive risk dashboard data."""
        
        return {
            "risk_metrics": self._risk_metrics,
            "risk_limits": self._risk_limits,
            "current_positions": self._portfolio_positions,
            "recent_alerts": self._risk_alerts[-10:],
            "risk_summary": await self._calculate_comprehensive_risk(),
            "last_updated": datetime.now().isoformat(),
        }
    
    async def update_risk_limits(self, new_limits: Dict[str, float]):
        """Update risk limits."""
        self._risk_limits.update(new_limits)
        logger.info(f"Updated risk limits: {new_limits}")
    
    async def emergency_stop(self):
        """Emergency stop all trading activities."""
        
        logger.critical("Emergency stop activated by Risk Control Agent")
        
        # Send emergency stop message
        emergency_message = ControlMessage(
            message_type=MessageType.KILL_SWITCH,
            sender_id=self.agent_id,
            command="emergency_stop",
            parameters={
                "reason": "Risk control emergency stop",
                "timestamp": datetime.now().isoformat(),
                "risk_metrics": self._risk_metrics,
            }
        )
        
        # This would be broadcast to all trading agents
        return emergency_message

    # Multi-Account Risk Management Methods

    async def validate_trade_for_account(
        self,
        account_id: str,
        trade_message: TradeMessage
    ) -> Dict[str, Any]:
        """Validate a trade against account-specific risk limits."""

        # Check if account exists
        account = self.account_manager.get_account(account_id)
        if not account:
            return {
                "approved": False,
                "reason": f"Account {account_id} not found",
                "risk_level": "CRITICAL"
            }

        # Check account status
        if account.status.value != "active":
            return {
                "approved": False,
                "reason": f"Account {account_id} is not active (status: {account.status.value})",
                "risk_level": "HIGH"
            }

        # Prepare trade data for risk check
        proposed_trade = {
            "symbol": trade_message.symbol,
            "action": trade_message.action,
            "quantity": trade_message.quantity,
            "price": getattr(trade_message, 'price', 0),
            "order_type": trade_message.order_type
        }

        # Check account-specific risk limits
        risk_check = await self.account_manager.check_risk_limits(account_id, proposed_trade)

        if not risk_check["allowed"]:
            return {
                "approved": False,
                "reason": risk_check["reason"],
                "risk_level": "HIGH",
                "limit_type": risk_check.get("limit_type")
            }

        # Perform advanced risk analysis for the account
        if account.positions:
            # Calculate portfolio-level risk for this account
            portfolio_value = account.current_capital
            for position in account.positions.values():
                portfolio_value += position.get('market_value', 0)

            # Run advanced risk metrics
            risk_metrics = await self._advanced_risk.calculate_portfolio_risk_metrics(
                account.positions, portfolio_value
            )

            # Check if adding this trade would violate advanced risk limits
            position_value = abs(proposed_trade['quantity'] * proposed_trade.get('price', 0))
            new_position_weight = position_value / portfolio_value if portfolio_value > 0 else 0

            # Check concentration risk
            if new_position_weight > account.risk_limits.max_position_size:
                return {
                    "approved": False,
                    "reason": f"Trade would create position weight {new_position_weight:.2%} exceeding limit {account.risk_limits.max_position_size:.2%}",
                    "risk_level": "HIGH",
                    "limit_type": "concentration"
                }

            # Check portfolio VaR
            var_pct = risk_metrics.var_95 / portfolio_value if portfolio_value > 0 else 0
            if var_pct > account.risk_limits.max_portfolio_risk:
                return {
                    "approved": False,
                    "reason": f"Portfolio VaR {var_pct:.2%} exceeds limit {account.risk_limits.max_portfolio_risk:.2%}",
                    "risk_level": "HIGH",
                    "limit_type": "portfolio_var"
                }

        return {
            "approved": True,
            "reason": "Trade approved - within all risk limits",
            "risk_level": "LOW"
        }

    async def monitor_account_risk(self, account_id: str) -> Dict[str, Any]:
        """Monitor risk metrics for a specific account."""

        account = self.account_manager.get_account(account_id)
        if not account:
            return {"error": f"Account {account_id} not found"}

        # Calculate portfolio value
        portfolio_value = account.current_capital
        for position in account.positions.values():
            portfolio_value += position.get('market_value', 0)

        # Calculate advanced risk metrics if positions exist
        risk_metrics = None
        if account.positions:
            risk_metrics = await self._advanced_risk.calculate_portfolio_risk_metrics(
                account.positions, portfolio_value
            )

        # Check risk limit violations
        violations = []

        # Check position concentration
        if account.metrics and account.metrics.largest_position_pct > account.risk_limits.max_position_size:
            violations.append({
                "type": "position_concentration",
                "current": account.metrics.largest_position_pct,
                "limit": account.risk_limits.max_position_size,
                "severity": "HIGH"
            })

        # Check portfolio VaR
        if risk_metrics:
            var_pct = risk_metrics.var_95 / portfolio_value if portfolio_value > 0 else 0
            if var_pct > account.risk_limits.max_portfolio_risk:
                violations.append({
                    "type": "portfolio_var",
                    "current": var_pct,
                    "limit": account.risk_limits.max_portfolio_risk,
                    "severity": "HIGH"
                })

        # Check drawdown
        if account.metrics and account.metrics.max_drawdown > account.risk_limits.max_drawdown:
            violations.append({
                "type": "max_drawdown",
                "current": account.metrics.max_drawdown,
                "limit": account.risk_limits.max_drawdown,
                "severity": "CRITICAL"
            })

        return {
            "account_id": account_id,
            "account_name": account.account_name,
            "portfolio_value": portfolio_value,
            "risk_metrics": risk_metrics,
            "violations": violations,
            "risk_level": "CRITICAL" if any(v["severity"] == "CRITICAL" for v in violations) else
                         "HIGH" if any(v["severity"] == "HIGH" for v in violations) else "LOW",
            "timestamp": datetime.now().isoformat()
        }
