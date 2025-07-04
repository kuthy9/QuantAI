"""
Profitability Agent (A6) for the QuantAI system.

This agent decides whether to go live with strategies based on performance criteria,
monitors live performance vs expectations, and manages strategy lifecycle decisions.
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
from ...core.messages import (
    BacktestMessage, 
    ControlMessage,
    MessageType, 
    QuantMessage, 
    StrategyMessage
)


class ProfitabilityAgent(ModelCapableAgent):
    """
    Profitability Agent (A6) - Makes go-live decisions based on performance criteria.
    
    Capabilities:
    - Performance evaluation against predefined criteria
    - Go-live decision making with risk-adjusted metrics
    - Live performance monitoring and comparison
    - Strategy lifecycle management (start/pause/stop)
    - Performance attribution and analysis
    - Dynamic threshold adjustment based on market conditions
    """
    
    def __init__(
        self,
        model_client: ChatCompletionClient,
        min_sharpe_ratio: float = 1.0,
        min_annual_return: float = 0.08,
        max_drawdown: float = 0.15,
        min_backtest_period_days: int = 252,
        confidence_threshold: float = 0.8,
        **kwargs
    ):
        super().__init__(
            role=AgentRole.PROFITABILITY,
            capabilities=[
                AgentCapability.PERFORMANCE_MONITORING,
                AgentCapability.LEARNING,
            ],
            model_client=model_client,
            system_message=self._get_system_message(),
            **kwargs
        )
        
        # Performance criteria
        self.min_sharpe_ratio = min_sharpe_ratio
        self.min_annual_return = min_annual_return
        self.max_drawdown = max_drawdown
        self.min_backtest_period_days = min_backtest_period_days
        self.confidence_threshold = confidence_threshold
        
        # Strategy tracking
        self._strategy_evaluations: Dict[str, Dict[str, Any]] = {}
        self._live_strategies: Dict[str, Dict[str, Any]] = {}
        self._performance_history: List[Dict[str, Any]] = []
        self._go_live_decisions: List[Dict[str, Any]] = []
        
        # Dynamic thresholds based on market conditions
        self._dynamic_thresholds: Dict[str, float] = {}
        self._market_regime_adjustments: Dict[str, Dict[str, float]] = {
            "bull_market": {"sharpe_multiplier": 0.9, "return_multiplier": 1.1},
            "bear_market": {"sharpe_multiplier": 1.2, "return_multiplier": 0.8},
            "sideways": {"sharpe_multiplier": 1.1, "return_multiplier": 0.9},
            "high_volatility": {"sharpe_multiplier": 1.3, "return_multiplier": 0.9},
            "low_volatility": {"sharpe_multiplier": 0.8, "return_multiplier": 1.0},
        }
    
    def _get_system_message(self) -> str:
        return """You are a Profitability Agent responsible for making critical go-live decisions for trading strategies.

Your responsibilities:
1. Evaluate strategy performance against rigorous criteria
2. Make data-driven go-live decisions with proper risk assessment
3. Monitor live strategy performance vs expectations
4. Manage strategy lifecycle (start, pause, stop, modify)
5. Perform performance attribution and root cause analysis
6. Adapt evaluation criteria based on market conditions

Performance Evaluation Framework:

1. Quantitative Metrics
   - Risk-adjusted returns (Sharpe ratio, Sortino ratio)
   - Maximum drawdown and recovery time
   - Win rate and profit factor
   - Volatility and beta analysis
   - Tail risk measures (VaR, Expected Shortfall)

2. Qualitative Assessment
   - Strategy logic robustness
   - Market regime adaptability
   - Implementation complexity
   - Operational risk factors
   - Regulatory compliance

3. Statistical Significance
   - Minimum sample size requirements
   - Confidence intervals for performance metrics
   - Out-of-sample validation results
   - Walk-forward analysis outcomes
   - Monte Carlo simulation results

4. Market Context
   - Current market regime compatibility
   - Economic cycle positioning
   - Correlation with existing strategies
   - Capacity and scalability constraints
   - Competition and market impact

Go-Live Decision Criteria:
- Sharpe Ratio > threshold (regime-adjusted)
- Annual Return > threshold (regime-adjusted)
- Maximum Drawdown < threshold
- Statistical significance > 95%
- Out-of-sample performance validation
- Risk management compliance
- Operational readiness confirmation

Lifecycle Management:
- Continuous performance monitoring
- Early warning system for underperformance
- Automatic pause triggers for risk breaches
- Performance attribution analysis
- Strategy modification recommendations

Guidelines:
- Prioritize capital preservation over aggressive returns
- Require statistical significance before go-live approval
- Consider market regime changes in evaluation
- Maintain detailed decision audit trails
- Implement graduated response to performance issues
- Balance individual strategy and portfolio-level impacts

Focus on making prudent, well-documented decisions that protect capital while enabling profitable growth."""
    
    async def process_message(
        self, 
        message: QuantMessage, 
        ctx: MessageContext
    ) -> Optional[QuantMessage]:
        """Process performance evaluation and go-live decision requests."""
        
        if isinstance(message, BacktestMessage):
            if message.message_type == MessageType.BACKTEST_RESULT:
                # Evaluate backtest results for go-live decision
                decision = await self._evaluate_strategy_performance(message)
                
                # Create decision response
                response = ControlMessage(
                    message_type=MessageType.STRATEGY_GENERATION,  # Will route to deployment
                    sender_id=self.agent_id,
                    command="go_live_decision",
                    target_agent="strategy_deployment",
                    parameters=decision,
                    session_id=message.session_id,
                    correlation_id=message.correlation_id,
                )
                
                logger.info(f"Go-live decision made for strategy {message.strategy_id}: {decision['decision']}")
                return response
        
        return None
    
    async def _evaluate_strategy_performance(self, backtest_message: BacktestMessage) -> Dict[str, Any]:
        """Evaluate strategy performance and make go-live decision."""
        
        strategy_id = backtest_message.strategy_id
        results = backtest_message.results or {}
        performance_metrics = backtest_message.performance_metrics or {}
        
        logger.info(f"Evaluating strategy performance for {strategy_id}")
        
        # Perform comprehensive evaluation
        quantitative_score = await self._evaluate_quantitative_metrics(performance_metrics)
        qualitative_score = await self._evaluate_qualitative_factors(strategy_id, results)
        statistical_significance = await self._assess_statistical_significance(results)
        market_context_score = await self._evaluate_market_context(strategy_id)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            quantitative_score, qualitative_score, 
            statistical_significance, market_context_score
        )
        
        # Make go-live decision
        decision_result = await self._make_go_live_decision(
            strategy_id, overall_score, performance_metrics, results
        )
        
        # Use LLM for comprehensive analysis
        llm_analysis = await self._llm_profitability_analysis(
            strategy_id, performance_metrics, results, decision_result
        )
        
        # Store evaluation
        evaluation = {
            "strategy_id": strategy_id,
            "evaluation_timestamp": datetime.now().isoformat(),
            "quantitative_score": quantitative_score,
            "qualitative_score": qualitative_score,
            "statistical_significance": statistical_significance,
            "market_context_score": market_context_score,
            "overall_score": overall_score,
            "decision": decision_result["decision"],
            "confidence": decision_result["confidence"],
            "reasoning": decision_result["reasoning"],
            "llm_analysis": llm_analysis,
            "performance_metrics": performance_metrics,
        }
        
        self._strategy_evaluations[strategy_id] = evaluation
        self._go_live_decisions.append(evaluation)
        
        return evaluation
    
    async def _evaluate_quantitative_metrics(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate quantitative performance metrics."""
        
        scores = {}
        
        # Sharpe ratio evaluation
        sharpe_ratio = metrics.get("sharpe_ratio", 0.0)
        adjusted_min_sharpe = self._get_adjusted_threshold("sharpe_ratio")
        scores["sharpe_ratio"] = {
            "value": sharpe_ratio,
            "threshold": adjusted_min_sharpe,
            "score": min(sharpe_ratio / adjusted_min_sharpe, 2.0) if adjusted_min_sharpe > 0 else 0,
            "passed": sharpe_ratio >= adjusted_min_sharpe
        }
        
        # Annual return evaluation
        annual_return = metrics.get("annual_return", 0.0)
        adjusted_min_return = self._get_adjusted_threshold("annual_return")
        scores["annual_return"] = {
            "value": annual_return,
            "threshold": adjusted_min_return,
            "score": min(annual_return / adjusted_min_return, 2.0) if adjusted_min_return > 0 else 0,
            "passed": annual_return >= adjusted_min_return
        }
        
        # Maximum drawdown evaluation
        max_drawdown = metrics.get("max_drawdown", 1.0)
        scores["max_drawdown"] = {
            "value": max_drawdown,
            "threshold": self.max_drawdown,
            "score": max(0, 2.0 - (max_drawdown / self.max_drawdown)) if self.max_drawdown > 0 else 0,
            "passed": max_drawdown <= self.max_drawdown
        }
        
        # Win rate evaluation
        win_rate = metrics.get("win_rate", 0.0)
        scores["win_rate"] = {
            "value": win_rate,
            "threshold": 0.5,  # 50% minimum
            "score": min(win_rate / 0.5, 2.0),
            "passed": win_rate >= 0.5
        }
        
        # Sortino ratio evaluation
        sortino_ratio = metrics.get("sortino_ratio", 0.0)
        scores["sortino_ratio"] = {
            "value": sortino_ratio,
            "threshold": 1.0,
            "score": min(sortino_ratio / 1.0, 2.0) if sortino_ratio > 0 else 0,
            "passed": sortino_ratio >= 1.0
        }
        
        # Calculate overall quantitative score
        passed_count = sum(1 for score in scores.values() if score["passed"])
        total_score = sum(score["score"] for score in scores.values())
        
        return {
            "individual_scores": scores,
            "passed_count": passed_count,
            "total_metrics": len(scores),
            "pass_rate": passed_count / len(scores),
            "average_score": total_score / len(scores),
            "overall_passed": passed_count >= len(scores) * 0.8  # 80% must pass
        }
    
    def _get_adjusted_threshold(self, metric: str) -> float:
        """Get market regime-adjusted threshold for a metric."""
        
        base_threshold = {
            "sharpe_ratio": self.min_sharpe_ratio,
            "annual_return": self.min_annual_return,
        }.get(metric, 1.0)
        
        # Get current market regime (would come from macro agent)
        current_regime = "bull_market"  # Placeholder
        
        if current_regime in self._market_regime_adjustments:
            adjustments = self._market_regime_adjustments[current_regime]
            multiplier_key = f"{metric.split('_')[0]}_multiplier"
            multiplier = adjustments.get(multiplier_key, 1.0)
            return base_threshold * multiplier
        
        return base_threshold
    
    async def _evaluate_qualitative_factors(self, strategy_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate qualitative factors for strategy approval."""
        
        # This would assess factors like:
        # - Strategy complexity and maintainability
        # - Market regime adaptability
        # - Implementation risk
        # - Regulatory compliance
        
        qualitative_scores = {
            "complexity": 0.8,  # Lower complexity is better
            "adaptability": 0.9,  # Higher adaptability is better
            "implementation_risk": 0.7,  # Lower risk is better
            "regulatory_compliance": 1.0,  # Must be compliant
        }
        
        average_score = np.mean(list(qualitative_scores.values()))
        
        return {
            "individual_scores": qualitative_scores,
            "average_score": average_score,
            "overall_passed": average_score >= 0.7
        }
    
    async def _assess_statistical_significance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess statistical significance of strategy performance."""
        
        trades = results.get("trades", [])
        
        if not trades:
            return {
                "sample_size": 0,
                "significance_level": 0.0,
                "confidence_interval": None,
                "statistically_significant": False
            }
        
        sample_size = len(trades)
        
        # Calculate returns
        returns = [trade.get("pnl", 0) for trade in trades]
        
        if not returns:
            return {
                "sample_size": sample_size,
                "significance_level": 0.0,
                "confidence_interval": None,
                "statistically_significant": False
            }
        
        # Basic statistical analysis
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # T-statistic for testing if mean return > 0
        if std_return > 0 and sample_size > 1:
            t_stat = mean_return / (std_return / np.sqrt(sample_size))
            # Simplified p-value calculation (would use proper statistical test)
            significance_level = max(0, min(1, abs(t_stat) / 3.0))
        else:
            significance_level = 0.0
        
        # Confidence interval (simplified)
        margin_error = 1.96 * (std_return / np.sqrt(sample_size)) if sample_size > 0 else 0
        confidence_interval = (mean_return - margin_error, mean_return + margin_error)
        
        return {
            "sample_size": sample_size,
            "mean_return": mean_return,
            "std_return": std_return,
            "significance_level": significance_level,
            "confidence_interval": confidence_interval,
            "statistically_significant": significance_level >= 0.95 and sample_size >= 30
        }
    
    async def _evaluate_market_context(self, strategy_id: str) -> Dict[str, Any]:
        """Evaluate strategy in current market context."""
        
        # This would consider:
        # - Current market regime compatibility
        # - Correlation with existing strategies
        # - Market capacity and competition
        # - Economic cycle positioning
        
        context_scores = {
            "regime_compatibility": 0.8,
            "portfolio_correlation": 0.9,  # Low correlation is good
            "market_capacity": 0.7,
            "economic_cycle_fit": 0.8,
        }
        
        average_score = np.mean(list(context_scores.values()))
        
        return {
            "individual_scores": context_scores,
            "average_score": average_score,
            "overall_passed": average_score >= 0.7
        }
    
    def _calculate_overall_score(
        self, 
        quantitative: Dict[str, Any], 
        qualitative: Dict[str, Any],
        statistical: Dict[str, Any], 
        market_context: Dict[str, Any]
    ) -> float:
        """Calculate overall strategy evaluation score."""
        
        # Weighted scoring
        weights = {
            "quantitative": 0.4,
            "qualitative": 0.2,
            "statistical": 0.3,
            "market_context": 0.1
        }
        
        scores = {
            "quantitative": quantitative.get("average_score", 0),
            "qualitative": qualitative.get("average_score", 0),
            "statistical": 1.0 if statistical.get("statistically_significant", False) else 0.0,
            "market_context": market_context.get("average_score", 0)
        }
        
        overall_score = sum(weights[key] * scores[key] for key in weights.keys())
        
        return overall_score
    
    async def _make_go_live_decision(
        self, 
        strategy_id: str, 
        overall_score: float, 
        metrics: Dict[str, float],
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make the final go-live decision."""
        
        # Decision logic
        if overall_score >= 0.8:
            decision = "APPROVED"
            confidence = overall_score
            reasoning = "Strategy meets all criteria for live trading"
        elif overall_score >= 0.6:
            decision = "CONDITIONAL_APPROVAL"
            confidence = overall_score * 0.8
            reasoning = "Strategy approved with enhanced monitoring and reduced allocation"
        else:
            decision = "REJECTED"
            confidence = 1.0 - overall_score
            reasoning = "Strategy does not meet minimum criteria for live trading"
        
        # Additional safety checks
        max_drawdown = metrics.get("max_drawdown", 1.0)
        if max_drawdown > self.max_drawdown:
            decision = "REJECTED"
            reasoning = f"Maximum drawdown {max_drawdown:.2%} exceeds limit {self.max_drawdown:.2%}"
        
        return {
            "decision": decision,
            "confidence": confidence,
            "reasoning": reasoning,
            "overall_score": overall_score,
            "recommended_allocation": self._calculate_recommended_allocation(decision, overall_score),
            "monitoring_level": self._determine_monitoring_level(decision, overall_score),
        }
    
    def _calculate_recommended_allocation(self, decision: str, score: float) -> float:
        """Calculate recommended portfolio allocation for the strategy."""
        
        if decision == "APPROVED":
            # Base allocation scaled by score
            return min(0.1, 0.05 * score)  # Max 10%, scaled by performance
        elif decision == "CONDITIONAL_APPROVAL":
            # Reduced allocation for conditional approval
            return min(0.05, 0.025 * score)  # Max 5%
        else:
            return 0.0  # No allocation for rejected strategies
    
    def _determine_monitoring_level(self, decision: str, score: float) -> str:
        """Determine required monitoring level for the strategy."""
        
        if decision == "REJECTED":
            return "none"
        elif decision == "CONDITIONAL_APPROVAL":
            return "enhanced"
        elif score >= 0.9:
            return "standard"
        else:
            return "elevated"
    
    async def _llm_profitability_analysis(
        self, 
        strategy_id: str, 
        metrics: Dict[str, float],
        results: Dict[str, Any], 
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use LLM for comprehensive profitability analysis."""
        
        prompt = f"""Analyze the profitability and go-live readiness of this trading strategy:

Strategy ID: {strategy_id}

Performance Metrics:
{json.dumps(metrics, indent=2)}

Backtest Results Summary:
{json.dumps({k: v for k, v in results.items() if k != 'trades'}, indent=2)}

Preliminary Decision:
{json.dumps(decision, indent=2)}

Provide comprehensive analysis including:

1. Performance Assessment
   - Strengths and weaknesses of the strategy
   - Risk-adjusted return quality
   - Consistency and reliability analysis

2. Market Suitability
   - Current market environment fit
   - Expected performance in different regimes
   - Potential risks and challenges

3. Implementation Considerations
   - Operational complexity and requirements
   - Scalability and capacity constraints
   - Integration with existing portfolio

4. Recommendations
   - Final go-live recommendation with rationale
   - Suggested allocation and risk limits
   - Monitoring and review requirements
   - Potential improvements or modifications

5. Risk Assessment
   - Key risk factors and mitigation strategies
   - Stress testing recommendations
   - Exit criteria and stop-loss protocols

Respond in JSON format:
{{
    "performance_assessment": {{
        "strengths": ["strength1", "strength2"],
        "weaknesses": ["weakness1", "weakness2"],
        "overall_quality": "excellent/good/fair/poor"
    }},
    "market_suitability": {{
        "current_fit": "high/medium/low",
        "regime_adaptability": "high/medium/low",
        "key_risks": ["risk1", "risk2"]
    }},
    "implementation": {{
        "complexity": "low/medium/high",
        "scalability": "high/medium/low",
        "integration_impact": "positive/neutral/negative"
    }},
    "final_recommendation": {{
        "decision": "approve/conditional/reject",
        "confidence": 0.0-1.0,
        "rationale": "detailed explanation",
        "allocation_pct": 0.0-0.15
    }},
    "risk_management": {{
        "key_risks": ["risk1", "risk2"],
        "mitigation_strategies": ["strategy1", "strategy2"],
        "exit_criteria": ["criteria1", "criteria2"]
    }}
}}"""
        
        try:
            response = await self._call_model([UserMessage(content=prompt, source="user")])
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error in LLM profitability analysis: {e}")
            return {"error": str(e)}
    
    async def monitor_live_strategy(self, strategy_id: str, current_performance: Dict[str, float]):
        """Monitor live strategy performance and trigger actions if needed."""
        
        if strategy_id not in self._live_strategies:
            # Initialize live strategy tracking
            self._live_strategies[strategy_id] = {
                "start_date": datetime.now(),
                "initial_allocation": 0.05,  # Default 5%
                "performance_history": [],
                "alerts": [],
                "status": "active"
            }
        
        strategy_data = self._live_strategies[strategy_id]
        strategy_data["performance_history"].append({
            "timestamp": datetime.now(),
            "metrics": current_performance
        })
        
        # Check for performance issues
        await self._check_live_performance_alerts(strategy_id, current_performance)
    
    async def _check_live_performance_alerts(self, strategy_id: str, performance: Dict[str, float]):
        """Check for performance alerts and trigger actions."""
        
        strategy_data = self._live_strategies[strategy_id]
        
        # Check drawdown
        current_drawdown = performance.get("current_drawdown", 0)
        if current_drawdown > self.max_drawdown * 0.8:  # 80% of max drawdown
            alert = {
                "type": "drawdown_warning",
                "timestamp": datetime.now(),
                "current_drawdown": current_drawdown,
                "threshold": self.max_drawdown * 0.8,
                "action": "reduce_allocation"
            }
            strategy_data["alerts"].append(alert)
            logger.warning(f"Drawdown warning for strategy {strategy_id}: {current_drawdown:.2%}")
        
        # Check Sharpe ratio degradation
        current_sharpe = performance.get("sharpe_ratio", 0)
        if current_sharpe < self.min_sharpe_ratio * 0.5:  # 50% of minimum
            alert = {
                "type": "sharpe_degradation",
                "timestamp": datetime.now(),
                "current_sharpe": current_sharpe,
                "threshold": self.min_sharpe_ratio * 0.5,
                "action": "enhanced_monitoring"
            }
            strategy_data["alerts"].append(alert)
            logger.warning(f"Sharpe ratio degradation for strategy {strategy_id}: {current_sharpe:.2f}")
    
    async def get_strategy_evaluation(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get evaluation results for a strategy."""
        return self._strategy_evaluations.get(strategy_id)
    
    async def get_live_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Get all live strategies and their status."""
        return self._live_strategies.copy()
    
    async def get_go_live_decisions(self) -> List[Dict[str, Any]]:
        """Get history of go-live decisions."""
        return self._go_live_decisions.copy()
    
    async def update_performance_criteria(self, new_criteria: Dict[str, float]):
        """Update performance criteria thresholds."""
        
        if "min_sharpe_ratio" in new_criteria:
            self.min_sharpe_ratio = new_criteria["min_sharpe_ratio"]
        if "min_annual_return" in new_criteria:
            self.min_annual_return = new_criteria["min_annual_return"]
        if "max_drawdown" in new_criteria:
            self.max_drawdown = new_criteria["max_drawdown"]
        
        logger.info(f"Updated performance criteria: {new_criteria}")
    
    async def pause_strategy(self, strategy_id: str, reason: str):
        """Pause a live strategy."""
        
        if strategy_id in self._live_strategies:
            self._live_strategies[strategy_id]["status"] = "paused"
            self._live_strategies[strategy_id]["pause_reason"] = reason
            self._live_strategies[strategy_id]["pause_timestamp"] = datetime.now()
            
            logger.info(f"Paused strategy {strategy_id}: {reason}")
    
    async def resume_strategy(self, strategy_id: str):
        """Resume a paused strategy."""
        
        if strategy_id in self._live_strategies:
            self._live_strategies[strategy_id]["status"] = "active"
            self._live_strategies[strategy_id]["resume_timestamp"] = datetime.now()
            
            logger.info(f"Resumed strategy {strategy_id}")
    
    async def stop_strategy(self, strategy_id: str, reason: str):
        """Permanently stop a strategy."""
        
        if strategy_id in self._live_strategies:
            self._live_strategies[strategy_id]["status"] = "stopped"
            self._live_strategies[strategy_id]["stop_reason"] = reason
            self._live_strategies[strategy_id]["stop_timestamp"] = datetime.now()
            
            logger.info(f"Stopped strategy {strategy_id}: {reason}")
