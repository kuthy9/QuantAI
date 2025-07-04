"""
Strategy Validation Agent for the QuantAI multi-agent system.

This agent is responsible for validating trading strategies before deployment,
checking for overfitting, data leakage, and other potential issues.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from autogen_core import Agent, MessageContext
from ...core.base import BaseQuantAgent, AgentRole, AgentCapability
from ...core.messages import QuantMessage, MessageType, StrategyMessage, ValidationMessage


class StrategyValidationAgent(BaseQuantAgent):
    """
    Agent responsible for validating trading strategies.
    
    This agent performs comprehensive validation including:
    - Code quality analysis
    - Overfitting detection
    - Data leakage detection
    - Risk assessment
    - Performance validation
    - Robustness testing
    """
    
    def __init__(self, agent_id: str = "strategy_validation"):
        super().__init__(
            role=AgentRole.STRATEGY_VALIDATION,
            capabilities=[
                AgentCapability.STRATEGY_VALIDATION,
                AgentCapability.CODE_ANALYSIS,
                AgentCapability.RISK_ASSESSMENT,
            ]
        )
        self.agent_id = agent_id
        self.validation_history: List[Dict[str, Any]] = []
        self.validation_criteria = {
            'code_quality': 0.8,
            'overfitting_threshold': 0.3,
            'data_leakage_threshold': 0.1,
            'min_sharpe_ratio': 1.0,
            'max_drawdown_threshold': 0.2
        }
        self.validation_tests = [
            'code_quality_check',
            'overfitting_detection',
            'data_leakage_detection',
            'performance_validation',
            'robustness_testing',
            'risk_assessment'
        ]
    
    async def on_messages(self, messages: List[QuantMessage], ctx: MessageContext) -> str:
        """Handle incoming messages for strategy validation."""
        results = []
        
        for message in messages:
            if isinstance(message, StrategyMessage):
                result = await self._handle_strategy_validation(message)
                results.append(result)
            else:
                result = await self._handle_general_message(message)
                results.append(result)
        
        return f"StrategyValidationAgent processed {len(results)} messages"
    
    async def _handle_strategy_validation(self, message: StrategyMessage) -> ValidationMessage:
        """Handle strategy validation request."""
        try:
            strategy_id = message.strategy_id
            
            # Perform comprehensive validation
            validation_results = await self._validate_strategy(message)
            
            # Determine overall validation status
            validation_passed = validation_results['overall_score'] >= 0.7
            
            # Create validation message
            validation_message = ValidationMessage(
                message_type=MessageType.VALIDATION_RESULT,
                sender_id=self.agent_id,
                strategy_id=strategy_id,
                validation_passed=validation_passed
            )
            
            # Record validation
            self.validation_history.append({
                'strategy_id': strategy_id,
                'validation_time': datetime.now(),
                'validation_passed': validation_passed,
                'results': validation_results
            })
            
            return validation_message
            
        except Exception as e:
            # Create error validation message
            return ValidationMessage(
                message_type=MessageType.VALIDATION_RESULT,
                sender_id=self.agent_id,
                strategy_id=getattr(message, 'strategy_id', 'unknown'),
                validation_passed=False
            )
    
    async def _validate_strategy(self, strategy: StrategyMessage) -> Dict[str, Any]:
        """Perform comprehensive strategy validation."""
        validation_results = {
            'strategy_id': strategy.strategy_id,
            'validation_timestamp': datetime.now(),
            'test_results': {},
            'overall_score': 0.0,
            'issues_found': [],
            'recommendations': []
        }
        
        # Run all validation tests
        test_scores = []
        
        for test_name in self.validation_tests:
            test_result = await self._run_validation_test(test_name, strategy)
            validation_results['test_results'][test_name] = test_result
            test_scores.append(test_result['score'])
            
            # Collect issues and recommendations
            if test_result.get('issues'):
                validation_results['issues_found'].extend(test_result['issues'])
            if test_result.get('recommendations'):
                validation_results['recommendations'].extend(test_result['recommendations'])
        
        # Calculate overall score
        validation_results['overall_score'] = sum(test_scores) / len(test_scores) if test_scores else 0.0
        
        return validation_results
    
    async def _run_validation_test(self, test_name: str, strategy: StrategyMessage) -> Dict[str, Any]:
        """Run a specific validation test."""
        # Simulate test execution time
        await asyncio.sleep(0.05)
        
        if test_name == 'code_quality_check':
            return await self._check_code_quality(strategy)
        elif test_name == 'overfitting_detection':
            return await self._detect_overfitting(strategy)
        elif test_name == 'data_leakage_detection':
            return await self._detect_data_leakage(strategy)
        elif test_name == 'performance_validation':
            return await self._validate_performance(strategy)
        elif test_name == 'robustness_testing':
            return await self._test_robustness(strategy)
        elif test_name == 'risk_assessment':
            return await self._assess_risk(strategy)
        else:
            return {
                'test_name': test_name,
                'score': 0.5,
                'status': 'unknown_test',
                'issues': [f'Unknown test: {test_name}'],
                'recommendations': ['Review test configuration']
            }
    
    async def _check_code_quality(self, strategy: StrategyMessage) -> Dict[str, Any]:
        """Check code quality of the strategy."""
        import random
        
        issues = []
        recommendations = []
        
        # Simulate code quality checks
        if not strategy.strategy_code:
            issues.append('No strategy code provided')
            score = 0.0
        else:
            # Simulate various code quality metrics
            code_length = len(strategy.strategy_code)
            complexity_score = min(1.0, code_length / 1000)  # Normalize by expected length
            
            # Simulate code quality issues
            if random.random() < 0.3:
                issues.append('Code complexity too high')
                recommendations.append('Consider breaking down complex functions')
            
            if random.random() < 0.2:
                issues.append('Missing error handling')
                recommendations.append('Add try-catch blocks for robustness')
            
            if random.random() < 0.1:
                issues.append('Potential security vulnerabilities')
                recommendations.append('Review code for security best practices')
            
            # Calculate score based on issues found
            score = max(0.0, 1.0 - len(issues) * 0.2)
        
        return {
            'test_name': 'code_quality_check',
            'score': score,
            'status': 'passed' if score >= self.validation_criteria['code_quality'] else 'failed',
            'issues': issues,
            'recommendations': recommendations,
            'metrics': {
                'code_length': len(strategy.strategy_code) if strategy.strategy_code else 0,
                'complexity_score': complexity_score if strategy.strategy_code else 0.0
            }
        }
    
    async def _detect_overfitting(self, strategy: StrategyMessage) -> Dict[str, Any]:
        """Detect potential overfitting in the strategy."""
        import random
        
        # Simulate overfitting detection
        overfitting_risk = random.uniform(0.0, 0.5)
        
        issues = []
        recommendations = []
        
        if overfitting_risk > self.validation_criteria['overfitting_threshold']:
            issues.append(f'High overfitting risk detected: {overfitting_risk:.2f}')
            recommendations.append('Consider using cross-validation')
            recommendations.append('Reduce model complexity')
            recommendations.append('Use regularization techniques')
        
        score = max(0.0, 1.0 - overfitting_risk)
        
        return {
            'test_name': 'overfitting_detection',
            'score': score,
            'status': 'passed' if overfitting_risk <= self.validation_criteria['overfitting_threshold'] else 'failed',
            'issues': issues,
            'recommendations': recommendations,
            'metrics': {
                'overfitting_risk': overfitting_risk,
                'threshold': self.validation_criteria['overfitting_threshold']
            }
        }
    
    async def _detect_data_leakage(self, strategy: StrategyMessage) -> Dict[str, Any]:
        """Detect potential data leakage in the strategy."""
        import random
        
        # Simulate data leakage detection
        leakage_risk = random.uniform(0.0, 0.2)
        
        issues = []
        recommendations = []
        
        if leakage_risk > self.validation_criteria['data_leakage_threshold']:
            issues.append(f'Potential data leakage detected: {leakage_risk:.2f}')
            recommendations.append('Review feature engineering process')
            recommendations.append('Ensure proper time-series splitting')
            recommendations.append('Check for look-ahead bias')
        
        score = max(0.0, 1.0 - leakage_risk * 5)  # Amplify penalty for data leakage
        
        return {
            'test_name': 'data_leakage_detection',
            'score': score,
            'status': 'passed' if leakage_risk <= self.validation_criteria['data_leakage_threshold'] else 'failed',
            'issues': issues,
            'recommendations': recommendations,
            'metrics': {
                'leakage_risk': leakage_risk,
                'threshold': self.validation_criteria['data_leakage_threshold']
            }
        }
    
    async def _validate_performance(self, strategy: StrategyMessage) -> Dict[str, Any]:
        """Validate strategy performance metrics."""
        import random
        
        # Simulate performance metrics
        sharpe_ratio = random.uniform(0.5, 2.5)
        max_drawdown = random.uniform(0.05, 0.3)
        annual_return = random.uniform(-0.1, 0.4)
        
        issues = []
        recommendations = []
        
        if sharpe_ratio < self.validation_criteria['min_sharpe_ratio']:
            issues.append(f'Low Sharpe ratio: {sharpe_ratio:.2f}')
            recommendations.append('Improve risk-adjusted returns')
        
        if max_drawdown > self.validation_criteria['max_drawdown_threshold']:
            issues.append(f'High maximum drawdown: {max_drawdown:.2f}')
            recommendations.append('Implement better risk management')
        
        # Calculate performance score
        sharpe_score = min(1.0, sharpe_ratio / self.validation_criteria['min_sharpe_ratio'])
        drawdown_score = max(0.0, 1.0 - max_drawdown / self.validation_criteria['max_drawdown_threshold'])
        return_score = max(0.0, min(1.0, (annual_return + 0.1) / 0.5))  # Normalize return
        
        score = (sharpe_score + drawdown_score + return_score) / 3
        
        return {
            'test_name': 'performance_validation',
            'score': score,
            'status': 'passed' if len(issues) == 0 else 'failed',
            'issues': issues,
            'recommendations': recommendations,
            'metrics': {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'annual_return': annual_return,
                'sharpe_score': sharpe_score,
                'drawdown_score': drawdown_score,
                'return_score': return_score
            }
        }
    
    async def _test_robustness(self, strategy: StrategyMessage) -> Dict[str, Any]:
        """Test strategy robustness across different market conditions."""
        import random
        
        # Simulate robustness testing across different scenarios
        scenarios = ['bull_market', 'bear_market', 'sideways_market', 'high_volatility', 'low_volatility']
        scenario_scores = [random.uniform(0.3, 0.9) for _ in scenarios]
        
        avg_score = sum(scenario_scores) / len(scenario_scores)
        score_variance = sum((s - avg_score) ** 2 for s in scenario_scores) / len(scenario_scores)
        
        issues = []
        recommendations = []
        
        if score_variance > 0.1:
            issues.append(f'High performance variance across scenarios: {score_variance:.3f}')
            recommendations.append('Improve strategy adaptability')
        
        if min(scenario_scores) < 0.4:
            issues.append('Poor performance in some market conditions')
            recommendations.append('Add market regime detection')
        
        return {
            'test_name': 'robustness_testing',
            'score': avg_score,
            'status': 'passed' if len(issues) == 0 else 'warning',
            'issues': issues,
            'recommendations': recommendations,
            'metrics': {
                'scenario_scores': dict(zip(scenarios, scenario_scores)),
                'average_score': avg_score,
                'score_variance': score_variance,
                'min_score': min(scenario_scores),
                'max_score': max(scenario_scores)
            }
        }
    
    async def _assess_risk(self, strategy: StrategyMessage) -> Dict[str, Any]:
        """Assess overall risk of the strategy."""
        import random
        
        # Simulate risk assessment
        var_95 = random.uniform(0.02, 0.08)  # Value at Risk
        expected_shortfall = random.uniform(0.03, 0.12)
        leverage_ratio = random.uniform(1.0, 3.0)
        
        issues = []
        recommendations = []
        
        if var_95 > 0.05:
            issues.append(f'High VaR (95%): {var_95:.3f}')
            recommendations.append('Reduce position sizes')
        
        if leverage_ratio > 2.0:
            issues.append(f'High leverage ratio: {leverage_ratio:.2f}')
            recommendations.append('Consider reducing leverage')
        
        # Calculate risk score (lower risk = higher score)
        var_score = max(0.0, 1.0 - var_95 / 0.1)
        leverage_score = max(0.0, 1.0 - (leverage_ratio - 1.0) / 2.0)
        
        score = (var_score + leverage_score) / 2
        
        return {
            'test_name': 'risk_assessment',
            'score': score,
            'status': 'passed' if len(issues) == 0 else 'warning',
            'issues': issues,
            'recommendations': recommendations,
            'metrics': {
                'var_95': var_95,
                'expected_shortfall': expected_shortfall,
                'leverage_ratio': leverage_ratio,
                'var_score': var_score,
                'leverage_score': leverage_score
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
            if isinstance(message, StrategyMessage):
                result = await self._handle_strategy_validation(message)
                return result  # ValidationMessage is already returned
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
    
    def get_validation_history(self) -> List[Dict[str, Any]]:
        """Get validation history."""
        return self.validation_history.copy()
    
    def get_validation_criteria(self) -> Dict[str, float]:
        """Get current validation criteria."""
        return self.validation_criteria.copy()
    
    def update_validation_criteria(self, criteria: Dict[str, float]) -> None:
        """Update validation criteria."""
        self.validation_criteria.update(criteria)
