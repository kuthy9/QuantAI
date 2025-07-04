"""
Feedback Loop Agent (A5) for the QuantAI system.

This agent learns from failed strategies and informs the strategy generation
process for continuous improvement and adaptation.
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
    FeedbackMessage, 
    MessageType, 
    QuantMessage, 
    StrategyMessage
)


class FeedbackLoopAgent(ModelCapableAgent):
    """
    Feedback Loop Agent (A5) - Learns from strategy performance and failures.
    
    Capabilities:
    - Failure analysis and root cause identification
    - Pattern recognition in successful vs failed strategies
    - Learning extraction from performance data
    - Strategy improvement recommendations
    - Continuous learning and adaptation
    - Knowledge transfer to strategy generation
    """
    
    def __init__(
        self,
        model_client: ChatCompletionClient,
        learning_window_days: int = 90,
        min_strategies_for_learning: int = 5,
        pattern_confidence_threshold: float = 0.7,
        **kwargs
    ):
        super().__init__(
            role=AgentRole.FEEDBACK_LOOP,
            capabilities=[
                AgentCapability.LEARNING,
                AgentCapability.PERFORMANCE_MONITORING,
            ],
            model_client=model_client,
            system_message=self._get_system_message(),
            **kwargs
        )
        
        self.learning_window_days = learning_window_days
        self.min_strategies_for_learning = min_strategies_for_learning
        self.pattern_confidence_threshold = pattern_confidence_threshold
        
        # Learning data storage
        self._strategy_outcomes: List[Dict[str, Any]] = []
        self._failure_patterns: Dict[str, Dict[str, Any]] = {}
        self._success_patterns: Dict[str, Dict[str, Any]] = {}
        self._learning_insights: List[Dict[str, Any]] = []
        self._improvement_recommendations: List[Dict[str, Any]] = []
        
        # Pattern categories
        self._pattern_categories = {
            "market_regime": ["bull_market", "bear_market", "sideways", "high_volatility", "low_volatility"],
            "strategy_type": ["momentum", "mean_reversion", "breakout", "arbitrage", "factor"],
            "timeframe": ["intraday", "daily", "weekly", "monthly"],
            "asset_class": ["equity", "fixed_income", "commodity", "currency", "crypto"],
            "complexity": ["low", "medium", "high"],
        }
    
    def _get_system_message(self) -> str:
        return """You are a Feedback Loop Agent responsible for continuous learning and improvement of the trading strategy development process.

Your responsibilities:
1. Analyze failed strategies to identify root causes and patterns
2. Extract lessons learned from both successful and unsuccessful strategies
3. Identify systematic biases and improvement opportunities
4. Generate actionable recommendations for strategy development
5. Maintain institutional knowledge and best practices
6. Adapt learning based on changing market conditions

Learning Framework:

1. Failure Analysis
   - Root cause identification (data, logic, market, implementation)
   - Pattern recognition across failed strategies
   - Common failure modes and their triggers
   - Market regime dependency analysis

2. Success Pattern Recognition
   - Characteristics of successful strategies
   - Market conditions favoring success
   - Optimal parameter ranges and configurations
   - Timing and implementation factors

3. Comparative Analysis
   - Performance attribution across strategy types
   - Market regime effectiveness comparison
   - Risk-adjusted return optimization insights
   - Scalability and capacity learnings

4. Continuous Improvement
   - Strategy development process refinement
   - Parameter optimization insights
   - Risk management enhancement opportunities
   - Market adaptation strategies

Learning Categories:

1. Market Regime Learnings
   - Strategy performance by market condition
   - Regime transition impact analysis
   - Adaptive parameter recommendations
   - Early warning indicators

2. Strategy Type Insights
   - Momentum vs mean reversion effectiveness
   - Factor strategy performance patterns
   - Cross-asset strategy correlations
   - Complexity vs performance trade-offs

3. Implementation Learnings
   - Code quality impact on performance
   - Execution timing optimization
   - Transaction cost minimization
   - Risk management effectiveness

4. Data Quality Insights
   - Data source reliability patterns
   - Signal quality vs performance correlation
   - Overfitting detection improvements
   - Feature engineering effectiveness

Guidelines:
- Focus on actionable insights that improve future strategies
- Maintain statistical rigor in pattern identification
- Consider market evolution and regime changes
- Balance historical learning with forward-looking adaptation
- Provide specific, implementable recommendations
- Track learning effectiveness and adaptation success

Focus on creating a self-improving system that learns from experience and continuously enhances strategy development capabilities."""
    
    async def process_message(
        self, 
        message: QuantMessage, 
        ctx: MessageContext
    ) -> Optional[QuantMessage]:
        """Process feedback and learning requests."""
        
        if isinstance(message, FeedbackMessage):
            if message.message_type == MessageType.FEEDBACK:
                # Process strategy feedback for learning
                await self._process_strategy_feedback(message)
                
                # Generate learning insights
                insights = await self._generate_learning_insights()
                
                # Create feedback response with recommendations
                response = StrategyMessage(
                    message_type=MessageType.STRATEGY_GENERATION,
                    sender_id=self.agent_id,
                    strategy_id=f"learning_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    strategy_name="Learning Insights and Recommendations",
                    strategy_description="Continuous learning insights for strategy improvement",
                    strategy_parameters=insights,
                    session_id=message.session_id,
                    correlation_id=message.correlation_id,
                )
                
                logger.info("Generated learning insights and recommendations")
                return response
        
        return None
    
    async def _process_strategy_feedback(self, feedback_message: FeedbackMessage):
        """Process feedback from a strategy's performance."""
        
        strategy_id = feedback_message.strategy_id
        performance_actual = feedback_message.performance_actual
        performance_expected = feedback_message.performance_expected
        success_factors = feedback_message.success_factors
        failure_factors = feedback_message.failure_factors
        
        logger.info(f"Processing feedback for strategy {strategy_id}")
        
        # Determine if strategy was successful
        success = self._determine_strategy_success(performance_actual, performance_expected)
        
        # Create strategy outcome record
        outcome = {
            "strategy_id": strategy_id,
            "timestamp": datetime.now(),
            "success": success,
            "performance_actual": performance_actual,
            "performance_expected": performance_expected,
            "success_factors": success_factors,
            "failure_factors": failure_factors,
            "performance_gap": self._calculate_performance_gap(performance_actual, performance_expected),
        }
        
        # Store outcome
        self._strategy_outcomes.append(outcome)
        
        # Update patterns
        await self._update_patterns(outcome)
        
        # Generate specific insights for this strategy
        await self._analyze_strategy_specific_insights(outcome)
    
    def _determine_strategy_success(
        self, 
        actual: Dict[str, float], 
        expected: Dict[str, float]
    ) -> bool:
        """Determine if a strategy was successful based on performance."""
        
        # Key metrics for success determination
        key_metrics = ["sharpe_ratio", "annual_return", "max_drawdown"]
        
        success_count = 0
        total_metrics = 0
        
        for metric in key_metrics:
            if metric in actual and metric in expected:
                actual_value = actual[metric]
                expected_value = expected[metric]
                
                if metric == "max_drawdown":
                    # For drawdown, lower is better
                    success = actual_value <= expected_value * 1.2  # 20% tolerance
                else:
                    # For returns and ratios, higher is better
                    success = actual_value >= expected_value * 0.8  # 80% of expected
                
                if success:
                    success_count += 1
                total_metrics += 1
        
        # Strategy is successful if it meets 2/3 of key metrics
        return success_count >= max(2, total_metrics * 0.67) if total_metrics > 0 else False
    
    def _calculate_performance_gap(
        self, 
        actual: Dict[str, float], 
        expected: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate performance gaps between actual and expected."""
        
        gaps = {}
        
        for metric in actual:
            if metric in expected:
                actual_value = actual[metric]
                expected_value = expected[metric]
                
                if expected_value != 0:
                    gap = (actual_value - expected_value) / abs(expected_value)
                else:
                    gap = actual_value
                
                gaps[metric] = gap
        
        return gaps
    
    async def _update_patterns(self, outcome: Dict[str, Any]):
        """Update success and failure patterns based on new outcome."""
        
        strategy_id = outcome["strategy_id"]
        success = outcome["success"]
        
        # Extract strategy characteristics (would come from strategy metadata)
        characteristics = await self._extract_strategy_characteristics(strategy_id)
        
        # Update appropriate pattern store
        pattern_store = self._success_patterns if success else self._failure_patterns
        
        for category, value in characteristics.items():
            if category not in pattern_store:
                pattern_store[category] = {}
            
            if value not in pattern_store[category]:
                pattern_store[category][value] = {
                    "count": 0,
                    "outcomes": [],
                    "avg_performance": {},
                }
            
            pattern_data = pattern_store[category][value]
            pattern_data["count"] += 1
            pattern_data["outcomes"].append(outcome)
            
            # Update average performance
            self._update_average_performance(pattern_data, outcome["performance_actual"])
    
    async def _extract_strategy_characteristics(self, strategy_id: str) -> Dict[str, str]:
        """Extract characteristics of a strategy for pattern analysis."""
        
        # This would typically query the strategy database
        # For now, return mock characteristics
        return {
            "market_regime": "bull_market",
            "strategy_type": "momentum",
            "timeframe": "daily",
            "asset_class": "equity",
            "complexity": "medium",
        }
    
    def _update_average_performance(self, pattern_data: Dict[str, Any], performance: Dict[str, float]):
        """Update average performance metrics for a pattern."""
        
        count = pattern_data["count"]
        avg_perf = pattern_data["avg_performance"]
        
        for metric, value in performance.items():
            if metric in avg_perf:
                # Update running average
                avg_perf[metric] = ((avg_perf[metric] * (count - 1)) + value) / count
            else:
                avg_perf[metric] = value
    
    async def _analyze_strategy_specific_insights(self, outcome: Dict[str, Any]):
        """Analyze specific insights from a strategy outcome."""
        
        strategy_id = outcome["strategy_id"]
        success = outcome["success"]
        performance_gap = outcome["performance_gap"]
        
        # Use LLM for detailed analysis
        insights = await self._llm_strategy_analysis(outcome)
        
        # Store insights
        insight_record = {
            "strategy_id": strategy_id,
            "timestamp": datetime.now(),
            "success": success,
            "key_insights": insights.get("key_insights", []),
            "improvement_areas": insights.get("improvement_areas", []),
            "lessons_learned": insights.get("lessons_learned", []),
            "performance_gap": performance_gap,
        }
        
        self._learning_insights.append(insight_record)
    
    async def _llm_strategy_analysis(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to analyze strategy outcome and extract insights."""
        
        prompt = f"""Analyze this trading strategy outcome and extract learning insights:

Strategy Outcome:
{json.dumps(outcome, indent=2, default=str)}

Provide detailed analysis including:

1. Root Cause Analysis
   - Primary factors contributing to success/failure
   - Secondary contributing factors
   - Market environment impact

2. Performance Analysis
   - Key performance drivers
   - Areas of strength and weakness
   - Comparison to expectations

3. Learning Insights
   - What worked well and why
   - What didn't work and why
   - Unexpected outcomes and their causes

4. Improvement Opportunities
   - Specific areas for enhancement
   - Parameter optimization suggestions
   - Risk management improvements

5. Broader Implications
   - Lessons applicable to similar strategies
   - Market regime insights
   - Strategy development process improvements

Respond in JSON format:
{{
    "root_cause_analysis": {{
        "primary_factors": ["factor1", "factor2"],
        "secondary_factors": ["factor1", "factor2"],
        "market_impact": "description"
    }},
    "performance_analysis": {{
        "strengths": ["strength1", "strength2"],
        "weaknesses": ["weakness1", "weakness2"],
        "key_drivers": ["driver1", "driver2"]
    }},
    "key_insights": ["insight1", "insight2"],
    "improvement_areas": ["area1", "area2"],
    "lessons_learned": ["lesson1", "lesson2"],
    "broader_implications": ["implication1", "implication2"]
}}"""
        
        try:
            response = await self._call_model([UserMessage(content=prompt, source="user")])
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error in LLM strategy analysis: {e}")
            return {"error": str(e)}
    
    async def _generate_learning_insights(self) -> Dict[str, Any]:
        """Generate comprehensive learning insights from accumulated data."""
        
        # Only generate insights if we have sufficient data
        if len(self._strategy_outcomes) < self.min_strategies_for_learning:
            return {
                "status": "insufficient_data",
                "required_strategies": self.min_strategies_for_learning,
                "current_strategies": len(self._strategy_outcomes),
            }
        
        # Analyze patterns
        pattern_analysis = await self._analyze_patterns()
        
        # Generate recommendations
        recommendations = await self._generate_recommendations()
        
        # Use LLM for comprehensive insights
        llm_insights = await self._llm_comprehensive_learning_analysis()
        
        insights = {
            "generation_timestamp": datetime.now().isoformat(),
            "data_window": {
                "start_date": (datetime.now() - timedelta(days=self.learning_window_days)).isoformat(),
                "end_date": datetime.now().isoformat(),
                "strategies_analyzed": len(self._strategy_outcomes),
            },
            "pattern_analysis": pattern_analysis,
            "recommendations": recommendations,
            "llm_insights": llm_insights,
            "success_rate": self._calculate_overall_success_rate(),
            "key_learnings": self._extract_key_learnings(),
        }
        
        return insights
    
    async def _analyze_patterns(self) -> Dict[str, Any]:
        """Analyze success and failure patterns."""
        
        pattern_analysis = {}
        
        for category in self._pattern_categories:
            category_analysis = {
                "success_patterns": {},
                "failure_patterns": {},
                "insights": [],
            }
            
            # Analyze success patterns
            if category in self._success_patterns:
                for value, data in self._success_patterns[category].items():
                    if data["count"] >= 3:  # Minimum sample size
                        category_analysis["success_patterns"][value] = {
                            "count": data["count"],
                            "avg_performance": data["avg_performance"],
                            "confidence": min(data["count"] / 10.0, 1.0),  # Confidence based on sample size
                        }
            
            # Analyze failure patterns
            if category in self._failure_patterns:
                for value, data in self._failure_patterns[category].items():
                    if data["count"] >= 3:
                        category_analysis["failure_patterns"][value] = {
                            "count": data["count"],
                            "avg_performance": data["avg_performance"],
                            "confidence": min(data["count"] / 10.0, 1.0),
                        }
            
            # Generate category insights
            category_analysis["insights"] = self._generate_category_insights(
                category, category_analysis["success_patterns"], category_analysis["failure_patterns"]
            )
            
            pattern_analysis[category] = category_analysis
        
        return pattern_analysis
    
    def _generate_category_insights(
        self, 
        category: str, 
        success_patterns: Dict[str, Any], 
        failure_patterns: Dict[str, Any]
    ) -> List[str]:
        """Generate insights for a specific pattern category."""
        
        insights = []
        
        # Find best performing patterns
        if success_patterns:
            best_pattern = max(
                success_patterns.items(),
                key=lambda x: x[1]["avg_performance"].get("sharpe_ratio", 0)
            )
            insights.append(f"Best performing {category}: {best_pattern[0]} (Sharpe: {best_pattern[1]['avg_performance'].get('sharpe_ratio', 0):.2f})")
        
        # Find worst performing patterns
        if failure_patterns:
            worst_pattern = min(
                failure_patterns.items(),
                key=lambda x: x[1]["avg_performance"].get("sharpe_ratio", 0)
            )
            insights.append(f"Worst performing {category}: {worst_pattern[0]} (Sharpe: {worst_pattern[1]['avg_performance'].get('sharpe_ratio', 0):.2f})")
        
        # Compare success vs failure rates
        success_total = sum(data["count"] for data in success_patterns.values())
        failure_total = sum(data["count"] for data in failure_patterns.values())
        
        if success_total + failure_total > 0:
            success_rate = success_total / (success_total + failure_total)
            insights.append(f"Overall {category} success rate: {success_rate:.1%}")
        
        return insights
    
    async def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on learning."""
        
        recommendations = []
        
        # Analyze recent outcomes for trends
        recent_outcomes = [
            outcome for outcome in self._strategy_outcomes
            if outcome["timestamp"] > datetime.now() - timedelta(days=30)
        ]
        
        if recent_outcomes:
            recent_success_rate = sum(1 for outcome in recent_outcomes if outcome["success"]) / len(recent_outcomes)
            
            if recent_success_rate < 0.5:
                recommendations.append({
                    "type": "process_improvement",
                    "priority": "high",
                    "title": "Low Recent Success Rate",
                    "description": f"Recent success rate is {recent_success_rate:.1%}, below 50% threshold",
                    "action": "Review strategy generation criteria and validation processes",
                    "confidence": 0.9,
                })
        
        # Analyze performance gaps
        large_gaps = [
            outcome for outcome in self._strategy_outcomes[-20:]  # Last 20 strategies
            if any(abs(gap) > 0.3 for gap in outcome["performance_gap"].values())
        ]
        
        if len(large_gaps) > len(self._strategy_outcomes[-20:]) * 0.3:  # More than 30%
            recommendations.append({
                "type": "expectation_calibration",
                "priority": "medium",
                "title": "Performance Expectation Misalignment",
                "description": "Frequent large gaps between expected and actual performance",
                "action": "Recalibrate performance expectations and improve backtesting accuracy",
                "confidence": 0.8,
            })
        
        # Add more recommendation logic based on patterns
        recommendations.extend(await self._generate_pattern_based_recommendations())
        
        return recommendations
    
    async def _generate_pattern_based_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations based on identified patterns."""
        
        recommendations = []
        
        # Analyze market regime patterns
        if "market_regime" in self._failure_patterns:
            regime_failures = self._failure_patterns["market_regime"]
            
            for regime, data in regime_failures.items():
                if data["count"] >= 3 and data["avg_performance"].get("sharpe_ratio", 0) < 0:
                    recommendations.append({
                        "type": "market_regime_adaptation",
                        "priority": "medium",
                        "title": f"Poor Performance in {regime}",
                        "description": f"Strategies consistently underperform in {regime} conditions",
                        "action": f"Develop regime-specific strategies or avoid trading in {regime}",
                        "confidence": min(data["count"] / 10.0, 1.0),
                    })
        
        # Analyze strategy type patterns
        if "strategy_type" in self._success_patterns and "strategy_type" in self._failure_patterns:
            success_types = self._success_patterns["strategy_type"]
            failure_types = self._failure_patterns["strategy_type"]
            
            for strategy_type in success_types:
                if strategy_type in failure_types:
                    success_count = success_types[strategy_type]["count"]
                    failure_count = failure_types[strategy_type]["count"]
                    success_rate = success_count / (success_count + failure_count)
                    
                    if success_rate > 0.7:
                        recommendations.append({
                            "type": "strategy_focus",
                            "priority": "low",
                            "title": f"Focus on {strategy_type} Strategies",
                            "description": f"{strategy_type} strategies have {success_rate:.1%} success rate",
                            "action": f"Increase allocation to {strategy_type} strategy development",
                            "confidence": min((success_count + failure_count) / 20.0, 1.0),
                        })
        
        return recommendations
    
    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall success rate across all strategies."""
        
        if not self._strategy_outcomes:
            return 0.0
        
        successful = sum(1 for outcome in self._strategy_outcomes if outcome["success"])
        return successful / len(self._strategy_outcomes)
    
    def _extract_key_learnings(self) -> List[str]:
        """Extract key learnings from recent insights."""
        
        key_learnings = []
        
        # Extract from recent learning insights
        recent_insights = [
            insight for insight in self._learning_insights
            if insight["timestamp"] > datetime.now() - timedelta(days=30)
        ]
        
        for insight in recent_insights[-10:]:  # Last 10 insights
            key_learnings.extend(insight.get("lessons_learned", []))
        
        # Remove duplicates and return top learnings
        unique_learnings = list(set(key_learnings))
        return unique_learnings[:10]  # Top 10 learnings
    
    async def _llm_comprehensive_learning_analysis(self) -> Dict[str, Any]:
        """Use LLM for comprehensive learning analysis."""
        
        # Prepare data summary for LLM
        data_summary = {
            "total_strategies": len(self._strategy_outcomes),
            "success_rate": self._calculate_overall_success_rate(),
            "recent_outcomes": [
                {
                    "success": outcome["success"],
                    "performance_gap": outcome["performance_gap"],
                    "factors": outcome.get("success_factors", []) + outcome.get("failure_factors", [])
                }
                for outcome in self._strategy_outcomes[-10:]  # Last 10
            ],
            "top_success_patterns": self._get_top_patterns(self._success_patterns),
            "top_failure_patterns": self._get_top_patterns(self._failure_patterns),
        }
        
        prompt = f"""Analyze the learning data from our trading strategy development system:

Learning Data Summary:
{json.dumps(data_summary, indent=2)}

Provide comprehensive analysis including:

1. Overall System Performance
   - Assessment of strategy development effectiveness
   - Key trends and patterns identified
   - Areas of strength and improvement

2. Critical Learning Insights
   - Most important lessons learned
   - Systematic biases or issues identified
   - Market adaptation insights

3. Strategic Recommendations
   - High-impact improvements for strategy development
   - Process optimization opportunities
   - Risk management enhancements

4. Future Focus Areas
   - Research and development priorities
   - Market conditions to monitor
   - Capability gaps to address

Respond in JSON format:
{{
    "system_assessment": {{
        "effectiveness": "high/medium/low",
        "key_trends": ["trend1", "trend2"],
        "strengths": ["strength1", "strength2"],
        "improvement_areas": ["area1", "area2"]
    }},
    "critical_insights": ["insight1", "insight2"],
    "strategic_recommendations": [
        {{
            "area": "strategy_development",
            "recommendation": "specific recommendation",
            "impact": "high/medium/low"
        }}
    ],
    "future_focus": ["focus1", "focus2"]
}}"""
        
        try:
            response = await self._call_model([UserMessage(content=prompt, source="user")])
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error in LLM comprehensive learning analysis: {e}")
            return {"error": str(e)}
    
    def _get_top_patterns(self, pattern_store: Dict[str, Dict[str, Any]], top_n: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """Get top patterns from a pattern store."""
        
        top_patterns = {}
        
        for category, patterns in pattern_store.items():
            # Sort patterns by count and take top N
            sorted_patterns = sorted(
                patterns.items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )[:top_n]
            
            top_patterns[category] = [
                {
                    "pattern": pattern,
                    "count": data["count"],
                    "avg_performance": data["avg_performance"]
                }
                for pattern, data in sorted_patterns
            ]
        
        return top_patterns
    
    async def get_learning_summary(self) -> Dict[str, Any]:
        """Get a summary of current learning state."""
        
        return {
            "total_strategies_analyzed": len(self._strategy_outcomes),
            "overall_success_rate": self._calculate_overall_success_rate(),
            "learning_insights_count": len(self._learning_insights),
            "recommendations_count": len(self._improvement_recommendations),
            "pattern_categories": list(self._pattern_categories.keys()),
            "last_analysis": self._learning_insights[-1]["timestamp"].isoformat() if self._learning_insights else None,
        }
    
    async def add_strategy_outcome(
        self, 
        strategy_id: str, 
        performance_actual: Dict[str, float],
        performance_expected: Dict[str, float],
        success_factors: List[str] = None,
        failure_factors: List[str] = None
    ):
        """Manually add a strategy outcome for learning."""
        
        feedback_message = FeedbackMessage(
            message_type=MessageType.FEEDBACK,
            sender_id="manual_input",
            strategy_id=strategy_id,
            performance_actual=performance_actual,
            performance_expected=performance_expected,
            success_factors=success_factors or [],
            failure_factors=failure_factors or [],
            lessons_learned=[],
            improvement_suggestions=[],
        )
        
        await self._process_strategy_feedback(feedback_message)
