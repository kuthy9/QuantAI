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
        """Calculate comprehensive risk metrics."""
        
        # This would include sophisticated risk calculations
        # For now, return basic metrics
        
        portfolio_value = self._risk_metrics.get("portfolio_value", 0)
        leverage = self._risk_metrics.get("leverage", 0)
        max_position = self._risk_metrics.get("max_position_weight", 0)
        current_drawdown = self._calculate_current_drawdown()
        
        # Simplified VaR calculation (would use historical simulation or Monte Carlo)
        estimated_volatility = 0.15  # 15% annual volatility assumption
        var_95 = portfolio_value * estimated_volatility * 1.65 / np.sqrt(252)  # Daily VaR
        
        return {
            "portfolio_value": portfolio_value,
            "leverage": leverage,
            "max_position_weight": max_position,
            "current_drawdown": current_drawdown,
            "var_95": var_95,
            "estimated_volatility": estimated_volatility,
            "sharpe_ratio": 0.0,  # Would calculate from returns
            "beta": 1.0,          # Would calculate vs benchmark
            "tracking_error": 0.0, # Would calculate vs benchmark
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
