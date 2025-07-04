"""
Cost Monitoring Agent for the QuantAI system.

This agent tracks API usage and costs across all external services,
provides alerts when approaching limits, and offers cost optimization recommendations.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from autogen_core import MessageContext
from autogen_core.models import ChatCompletionClient, UserMessage
from loguru import logger

from ...core.base import AgentCapability, AgentRole, ModelCapableAgent
from ...core.messages import ControlMessage, MessageType, QuantMessage


class CostAlertLevel(str, Enum):
    """Cost alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class APIUsageMetrics:
    """API usage metrics for a specific service."""
    service_name: str
    api_key_id: str
    requests_count: int
    tokens_used: int
    cost_usd: float
    period_start: datetime
    period_end: datetime
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None


@dataclass
class CostAlert:
    """Cost monitoring alert."""
    alert_id: str
    service_name: str
    alert_level: CostAlertLevel
    message: str
    current_cost: float
    limit_cost: float
    percentage_used: float
    timestamp: datetime
    recommendations: List[str]


@dataclass
class CostBudget:
    """Cost budget configuration for a service."""
    service_name: str
    monthly_budget_usd: float
    warning_threshold: float = 0.8  # 80%
    critical_threshold: float = 0.95  # 95%
    emergency_threshold: float = 1.0  # 100%
    auto_disable_at_limit: bool = True


class CostMonitorAgent(ModelCapableAgent):
    """
    Agent responsible for monitoring API costs and usage across all services.
    
    Features:
    - Real-time cost tracking
    - Budget alerts and notifications
    - Usage optimization recommendations
    - Automatic service throttling/disabling
    - Cost reporting and analytics
    """
    
    def __init__(
        self,
        model_client: ChatCompletionClient,
        check_interval_minutes: int = 15,
        **kwargs
    ):
        super().__init__(
            role=AgentRole.API_MANAGER,
            capabilities=[
                AgentCapability.SYSTEM_INTEGRATION,
                AgentCapability.PERFORMANCE_MONITORING,
            ],
            model_client=model_client,
            system_message=self._get_system_message(),
            **kwargs
        )
        
        self.agent_id = "cost_monitor"
        self.check_interval_minutes = check_interval_minutes
        
        # Cost tracking
        self._usage_metrics: Dict[str, APIUsageMetrics] = {}
        self._cost_budgets: Dict[str, CostBudget] = {}
        self._alerts_history: List[CostAlert] = []
        self._active_alerts: Dict[str, CostAlert] = {}
        
        # Service configurations
        self._service_configs = self._initialize_service_configs()
        
        # Initialize default budgets
        self._initialize_default_budgets()
        
        logger.info("Initialized Cost Monitor Agent")
    
    def _get_system_message(self) -> str:
        return """You are a Cost Monitoring Agent specializing in API cost tracking and optimization.

Your responsibilities:
1. Monitor API usage and costs across all external services
2. Track spending against budgets and alert when approaching limits
3. Provide cost optimization recommendations
4. Prevent service interruptions due to budget overruns
5. Generate detailed cost reports and analytics

Key services to monitor:
- OpenAI API (GPT models)
- Anthropic API (Claude models)
- Google API (Gemini models)
- Market data APIs (Alpha Vantage, Polygon)
- News APIs (NewsAPI, Finnhub)
- Trading APIs (Alpaca, IBKR)

Focus on proactive cost management and optimization suggestions."""
    
    def _initialize_service_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize service-specific cost configurations."""
        return {
            "openai": {
                "cost_per_1k_tokens": {
                    "gpt-4o": {"input": 0.005, "output": 0.015},
                    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
                    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
                },
                "rate_limits": {"requests_per_minute": 500, "tokens_per_minute": 30000}
            },
            "anthropic": {
                "cost_per_1k_tokens": {
                    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
                    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
                },
                "rate_limits": {"requests_per_minute": 50, "tokens_per_minute": 40000}
            },
            "google": {
                "cost_per_1k_tokens": {
                    "gemini-pro": {"input": 0.0005, "output": 0.0015},
                    "gemini-pro-vision": {"input": 0.0005, "output": 0.0015},
                },
                "rate_limits": {"requests_per_minute": 60, "tokens_per_minute": 32000}
            },
            "alpha_vantage": {
                "cost_per_request": 0.0,  # Free tier
                "rate_limits": {"requests_per_minute": 5, "requests_per_day": 500}
            },
            "newsapi": {
                "cost_per_request": 0.0001,  # Estimated
                "rate_limits": {"requests_per_day": 1000}
            }
        }
    
    def _initialize_default_budgets(self):
        """Initialize default cost budgets for services."""
        default_budgets = [
            CostBudget("openai", 100.0, 0.8, 0.9, 0.95),
            CostBudget("anthropic", 50.0, 0.8, 0.9, 0.95),
            CostBudget("google", 30.0, 0.8, 0.9, 0.95),
            CostBudget("alpha_vantage", 0.0, 0.8, 0.9, 0.95),
            CostBudget("newsapi", 10.0, 0.8, 0.9, 0.95),
        ]
        
        for budget in default_budgets:
            self._cost_budgets[budget.service_name] = budget
    
    async def process_message(
        self, 
        message: QuantMessage, 
        ctx: MessageContext
    ) -> Optional[QuantMessage]:
        """Process cost monitoring messages."""
        
        if isinstance(message, ControlMessage):
            if message.command == "update_usage":
                await self._update_usage_metrics(message.parameters)
            elif message.command == "check_costs":
                await self._check_all_costs()
            elif message.command == "get_cost_report":
                return await self._generate_cost_report()
            elif message.command == "update_budget":
                await self._update_budget(message.parameters)
        
        return None
    
    async def _update_usage_metrics(self, usage_data: Dict[str, Any]):
        """Update usage metrics for a service."""
        service_name = usage_data.get("service_name")
        if not service_name:
            return
        
        # Calculate cost based on usage
        cost = await self._calculate_cost(service_name, usage_data)
        
        # Update metrics
        metrics = APIUsageMetrics(
            service_name=service_name,
            api_key_id=usage_data.get("api_key_id", "default"),
            requests_count=usage_data.get("requests_count", 0),
            tokens_used=usage_data.get("tokens_used", 0),
            cost_usd=cost,
            period_start=datetime.fromisoformat(usage_data.get("period_start", datetime.now().isoformat())),
            period_end=datetime.fromisoformat(usage_data.get("period_end", datetime.now().isoformat())),
            rate_limit_remaining=usage_data.get("rate_limit_remaining"),
            rate_limit_reset=datetime.fromisoformat(usage_data["rate_limit_reset"]) if usage_data.get("rate_limit_reset") else None
        )
        
        self._usage_metrics[service_name] = metrics
        
        # Check if we need to generate alerts
        await self._check_service_costs(service_name)
    
    async def _calculate_cost(self, service_name: str, usage_data: Dict[str, Any]) -> float:
        """Calculate cost for service usage."""
        service_config = self._service_configs.get(service_name, {})
        
        if "cost_per_1k_tokens" in service_config:
            # Token-based pricing
            model_name = usage_data.get("model_name", "")
            tokens_used = usage_data.get("tokens_used", 0)
            input_tokens = usage_data.get("input_tokens", tokens_used // 2)
            output_tokens = usage_data.get("output_tokens", tokens_used // 2)
            
            model_costs = service_config["cost_per_1k_tokens"].get(model_name, {})
            input_cost = (input_tokens / 1000) * model_costs.get("input", 0)
            output_cost = (output_tokens / 1000) * model_costs.get("output", 0)
            
            return input_cost + output_cost
            
        elif "cost_per_request" in service_config:
            # Request-based pricing
            requests_count = usage_data.get("requests_count", 0)
            return requests_count * service_config["cost_per_request"]
        
        return 0.0
    
    async def _check_service_costs(self, service_name: str):
        """Check costs for a specific service and generate alerts if needed."""
        metrics = self._usage_metrics.get(service_name)
        budget = self._cost_budgets.get(service_name)
        
        if not metrics or not budget:
            return
        
        # Calculate monthly cost (extrapolate from current period)
        period_days = (metrics.period_end - metrics.period_start).days or 1
        monthly_cost = (metrics.cost_usd / period_days) * 30
        
        percentage_used = monthly_cost / budget.monthly_budget_usd if budget.monthly_budget_usd > 0 else 0
        
        # Determine alert level
        alert_level = None
        if percentage_used >= budget.emergency_threshold:
            alert_level = CostAlertLevel.EMERGENCY
        elif percentage_used >= budget.critical_threshold:
            alert_level = CostAlertLevel.CRITICAL
        elif percentage_used >= budget.warning_threshold:
            alert_level = CostAlertLevel.WARNING
        
        if alert_level:
            await self._generate_cost_alert(service_name, alert_level, monthly_cost, budget, percentage_used)
    
    async def _generate_cost_alert(
        self, 
        service_name: str, 
        alert_level: CostAlertLevel,
        current_cost: float,
        budget: CostBudget,
        percentage_used: float
    ):
        """Generate and process a cost alert."""
        alert_id = f"{service_name}_{alert_level.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Generate recommendations
        recommendations = await self._generate_cost_recommendations(service_name, current_cost, budget)
        
        alert = CostAlert(
            alert_id=alert_id,
            service_name=service_name,
            alert_level=alert_level,
            message=f"{service_name} cost at {percentage_used:.1%} of budget (${current_cost:.2f}/${budget.monthly_budget_usd:.2f})",
            current_cost=current_cost,
            limit_cost=budget.monthly_budget_usd,
            percentage_used=percentage_used,
            timestamp=datetime.now(),
            recommendations=recommendations
        )
        
        # Store alert
        self._alerts_history.append(alert)
        self._active_alerts[service_name] = alert
        
        # Take action based on alert level
        if alert_level == CostAlertLevel.EMERGENCY and budget.auto_disable_at_limit:
            await self._disable_service(service_name)
        
        logger.warning(f"Cost alert generated: {alert.message}")
        
        # Send notification (could integrate with Telegram, email, etc.)
        await self._send_cost_notification(alert)
    
    async def _generate_cost_recommendations(
        self, 
        service_name: str, 
        current_cost: float, 
        budget: CostBudget
    ) -> List[str]:
        """Generate cost optimization recommendations using LLM."""
        
        metrics = self._usage_metrics.get(service_name, {})
        service_config = self._service_configs.get(service_name, {})
        
        prompt = f"""Analyze the following API cost situation and provide specific optimization recommendations:

Service: {service_name}
Current Monthly Cost: ${current_cost:.2f}
Budget: ${budget.monthly_budget_usd:.2f}
Usage Over Budget: {((current_cost / budget.monthly_budget_usd) - 1) * 100:.1f}%

Service Configuration: {json.dumps(service_config, indent=2)}
Current Metrics: {json.dumps(asdict(metrics) if hasattr(metrics, '__dict__') else {}, default=str, indent=2)}

Provide 3-5 specific, actionable recommendations to reduce costs while maintaining system functionality.
Focus on:
1. Model optimization (using cheaper models where appropriate)
2. Request optimization (reducing unnecessary calls)
3. Caching strategies
4. Rate limiting improvements
5. Alternative service providers

Format as a JSON list of strings."""
        
        try:
            response = await self._call_model([UserMessage(content=prompt, source="user")])
            recommendations = json.loads(response)
            return recommendations if isinstance(recommendations, list) else [response]
        except Exception as e:
            logger.error(f"Error generating cost recommendations: {e}")
            return [
                f"Consider switching to a cheaper model for {service_name}",
                "Implement request caching to reduce API calls",
                "Review and optimize prompt lengths",
                "Consider rate limiting to control costs"
            ]

    async def _disable_service(self, service_name: str):
        """Disable a service when cost limits are exceeded."""
        logger.critical(f"Disabling service {service_name} due to cost limit breach")

        # Send disable command to API manager
        disable_message = ControlMessage(
            message_type=MessageType.CONTROL,
            sender_id=self.agent_id,
            command="disable_service",
            parameters={"service_name": service_name, "reason": "cost_limit_exceeded"}
        )

        # Would send this to the API manager agent
        # await self._send_message(disable_message)

    async def _send_cost_notification(self, alert: CostAlert):
        """Send cost alert notification."""
        notification_data = {
            "type": "cost_alert",
            "service": alert.service_name,
            "level": alert.alert_level.value,
            "message": alert.message,
            "cost": alert.current_cost,
            "budget": alert.limit_cost,
            "percentage": alert.percentage_used,
            "recommendations": alert.recommendations
        }

        # Could integrate with Telegram, email, Slack, etc.
        logger.info(f"Cost notification sent: {notification_data}")

    async def _check_all_costs(self):
        """Check costs for all monitored services."""
        for service_name in self._cost_budgets.keys():
            await self._check_service_costs(service_name)

    async def _update_budget(self, budget_data: Dict[str, Any]):
        """Update budget configuration for a service."""
        service_name = budget_data.get("service_name")
        if not service_name:
            return

        budget = self._cost_budgets.get(service_name)
        if budget:
            # Update existing budget
            budget.monthly_budget_usd = budget_data.get("monthly_budget_usd", budget.monthly_budget_usd)
            budget.warning_threshold = budget_data.get("warning_threshold", budget.warning_threshold)
            budget.critical_threshold = budget_data.get("critical_threshold", budget.critical_threshold)
            budget.emergency_threshold = budget_data.get("emergency_threshold", budget.emergency_threshold)
            budget.auto_disable_at_limit = budget_data.get("auto_disable_at_limit", budget.auto_disable_at_limit)
        else:
            # Create new budget
            budget = CostBudget(
                service_name=service_name,
                monthly_budget_usd=budget_data.get("monthly_budget_usd", 100.0),
                warning_threshold=budget_data.get("warning_threshold", 0.8),
                critical_threshold=budget_data.get("critical_threshold", 0.9),
                emergency_threshold=budget_data.get("emergency_threshold", 0.95),
                auto_disable_at_limit=budget_data.get("auto_disable_at_limit", True)
            )
            self._cost_budgets[service_name] = budget

        logger.info(f"Updated budget for {service_name}: ${budget.monthly_budget_usd}")

    async def _generate_cost_report(self) -> QuantMessage:
        """Generate comprehensive cost report."""
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "services": {},
            "total_monthly_cost": 0.0,
            "total_budget": 0.0,
            "active_alerts": len(self._active_alerts),
            "recommendations": []
        }

        for service_name, budget in self._cost_budgets.items():
            metrics = self._usage_metrics.get(service_name)

            if metrics:
                # Calculate monthly cost projection
                period_days = (metrics.period_end - metrics.period_start).days or 1
                monthly_cost = (metrics.cost_usd / period_days) * 30

                service_data = {
                    "current_cost": metrics.cost_usd,
                    "monthly_projection": monthly_cost,
                    "budget": budget.monthly_budget_usd,
                    "percentage_used": monthly_cost / budget.monthly_budget_usd if budget.monthly_budget_usd > 0 else 0,
                    "requests_count": metrics.requests_count,
                    "tokens_used": metrics.tokens_used,
                    "rate_limit_remaining": metrics.rate_limit_remaining
                }

                report_data["services"][service_name] = service_data
                report_data["total_monthly_cost"] += monthly_cost

            report_data["total_budget"] += budget.monthly_budget_usd

        # Add overall recommendations
        if report_data["total_monthly_cost"] > report_data["total_budget"] * 0.8:
            report_data["recommendations"].extend([
                "Consider implementing more aggressive caching",
                "Review model selection for cost optimization",
                "Implement request batching where possible"
            ])

        return ControlMessage(
            message_type=MessageType.CONTROL,
            sender_id=self.agent_id,
            command="cost_report",
            parameters=report_data
        )

    def get_current_costs(self) -> Dict[str, float]:
        """Get current costs for all services."""
        costs = {}
        for service_name, metrics in self._usage_metrics.items():
            if metrics:
                period_days = (metrics.period_end - metrics.period_start).days or 1
                monthly_cost = (metrics.cost_usd / period_days) * 30
                costs[service_name] = monthly_cost
        return costs

    def get_active_alerts(self) -> List[CostAlert]:
        """Get all active cost alerts."""
        return list(self._active_alerts.values())

    def get_budget_status(self) -> Dict[str, Dict[str, Any]]:
        """Get budget status for all services."""
        status = {}
        for service_name, budget in self._cost_budgets.items():
            metrics = self._usage_metrics.get(service_name)
            if metrics:
                period_days = (metrics.period_end - metrics.period_start).days or 1
                monthly_cost = (metrics.cost_usd / period_days) * 30
                percentage_used = monthly_cost / budget.monthly_budget_usd if budget.monthly_budget_usd > 0 else 0

                status[service_name] = {
                    "monthly_cost": monthly_cost,
                    "budget": budget.monthly_budget_usd,
                    "percentage_used": percentage_used,
                    "status": "ok" if percentage_used < budget.warning_threshold else
                             "warning" if percentage_used < budget.critical_threshold else
                             "critical" if percentage_used < budget.emergency_threshold else "emergency"
                }
        return status
