"""
Strategy Validator Agent (D5) for the QuantAI system.

This agent audits strategy code for logic issues, overfitting, data leakage,
and other potential problems before deployment.
"""

import ast
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from autogen_core import MessageContext
from autogen_core.models import ChatCompletionClient, UserMessage
from loguru import logger

from ...core.base import AgentCapability, AgentRole, ModelCapableAgent
from ...core.messages import MessageType, QuantMessage, StrategyMessage, ValidationMessage


class StrategyValidatorAgent(ModelCapableAgent):
    """
    Strategy Validator Agent (D5) - Audits strategy code for issues.
    
    Capabilities:
    - Code quality analysis and best practices validation
    - Logic error detection and edge case identification
    - Overfitting and data leakage detection
    - Performance expectation validation
    - Risk management compliance checking
    - Security and safety validation
    """
    
    def __init__(
        self,
        model_client: ChatCompletionClient,
        validation_strictness: str = "medium",
        auto_fix_issues: bool = False,
        **kwargs
    ):
        super().__init__(
            role=AgentRole.STRATEGY_VALIDATION,
            capabilities=[
                AgentCapability.CODE_VALIDATION,
                AgentCapability.RISK_ASSESSMENT,
            ],
            model_client=model_client,
            system_message=self._get_system_message(),
            **kwargs
        )
        
        self.validation_strictness = validation_strictness
        self.auto_fix_issues = auto_fix_issues
        self._validation_rules = self._initialize_validation_rules()
        self._validation_history: List[Dict[str, Any]] = []
    
    def _get_system_message(self) -> str:
        return """You are a Strategy Validator Agent specializing in comprehensive trading strategy code validation.

Your responsibilities:
1. Analyze strategy code for logic errors and edge cases
2. Detect potential overfitting and data leakage issues
3. Validate risk management implementation
4. Check performance expectations for realism
5. Ensure code quality and best practices
6. Identify security and safety concerns

Validation Categories:

1. Code Quality
   - Syntax and structural analysis
   - Best practices compliance
   - Error handling adequacy
   - Code maintainability

2. Logic Validation
   - Entry/exit condition logic
   - Signal generation correctness
   - Position sizing logic
   - Timing and sequencing issues

3. Data Integrity
   - Look-ahead bias detection
   - Data leakage identification
   - Survivorship bias checks
   - Point-in-time data usage

4. Risk Management
   - Stop loss implementation
   - Position size limits
   - Drawdown controls
   - Portfolio risk constraints

5. Performance Validation
   - Realistic return expectations
   - Risk-adjusted metrics
   - Benchmark comparisons
   - Stress testing considerations

6. Overfitting Detection
   - Parameter optimization concerns
   - In-sample vs out-of-sample performance
   - Strategy complexity analysis
   - Robustness testing

Validation Levels:
- Critical: Issues that prevent deployment
- Warning: Issues that need attention
- Info: Suggestions for improvement

Guidelines:
- Be thorough but practical in validation
- Provide specific, actionable feedback
- Suggest concrete fixes for identified issues
- Consider real-world trading constraints
- Validate against industry best practices
- Ensure strategies are production-ready

Focus on identifying issues that could lead to poor performance, excessive risk, or operational problems in live trading."""
    
    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize validation rules and checks."""
        return {
            "code_quality": {
                "syntax_check": {"enabled": True, "severity": "critical"},
                "imports_check": {"enabled": True, "severity": "warning"},
                "error_handling": {"enabled": True, "severity": "warning"},
                "documentation": {"enabled": True, "severity": "info"},
            },
            "logic_validation": {
                "entry_conditions": {"enabled": True, "severity": "critical"},
                "exit_conditions": {"enabled": True, "severity": "critical"},
                "position_sizing": {"enabled": True, "severity": "critical"},
                "signal_logic": {"enabled": True, "severity": "warning"},
            },
            "data_integrity": {
                "lookahead_bias": {"enabled": True, "severity": "critical"},
                "data_leakage": {"enabled": True, "severity": "critical"},
                "survivorship_bias": {"enabled": True, "severity": "warning"},
                "data_quality": {"enabled": True, "severity": "warning"},
            },
            "risk_management": {
                "stop_loss": {"enabled": True, "severity": "critical"},
                "position_limits": {"enabled": True, "severity": "critical"},
                "drawdown_controls": {"enabled": True, "severity": "warning"},
                "risk_metrics": {"enabled": True, "severity": "warning"},
            },
            "performance": {
                "return_expectations": {"enabled": True, "severity": "warning"},
                "sharpe_ratio": {"enabled": True, "severity": "info"},
                "drawdown_limits": {"enabled": True, "severity": "warning"},
                "win_rate": {"enabled": True, "severity": "info"},
            },
            "overfitting": {
                "parameter_count": {"enabled": True, "severity": "warning"},
                "complexity_analysis": {"enabled": True, "severity": "warning"},
                "robustness_check": {"enabled": True, "severity": "info"},
            },
        }
    
    async def process_message(
        self, 
        message: QuantMessage, 
        ctx: MessageContext
    ) -> Optional[QuantMessage]:
        """Process strategy validation requests."""
        
        if isinstance(message, StrategyMessage):
            if message.message_type == MessageType.STRATEGY_VALIDATION:
                # Validate the strategies
                validation_results = await self._validate_strategies(message)
                
                # Create validation response
                response = ValidationMessage(
                    message_type=MessageType.STRATEGY_VALIDATION,
                    sender_id=self.agent_id,
                    strategy_id=message.strategy_id,
                    validation_passed=validation_results["overall_passed"],
                    validation_score=validation_results["overall_score"],
                    issues_found=validation_results["critical_issues"] + validation_results["warnings"],
                    recommendations=validation_results["recommendations"],
                    overfitting_risk=validation_results["overfitting_risk"],
                    data_leakage_detected=validation_results["data_leakage_detected"],
                    session_id=message.session_id,
                    correlation_id=message.correlation_id,
                )
                
                logger.info(f"Validation completed: {validation_results['overall_passed']}")
                return response
        
        return None
    
    async def _validate_strategies(self, message: StrategyMessage) -> Dict[str, Any]:
        """Validate all strategies in the message."""
        
        strategies_data = json.loads(message.strategy_code) if isinstance(message.strategy_code, str) else message.strategy_code
        coded_strategies = strategies_data.get("coded_strategies", [])
        
        all_results = []
        critical_issues = []
        warnings = []
        recommendations = []
        
        for strategy in coded_strategies:
            try:
                result = await self._validate_single_strategy(strategy)
                all_results.append(result)
                
                # Aggregate issues
                critical_issues.extend(result["critical_issues"])
                warnings.extend(result["warnings"])
                recommendations.extend(result["recommendations"])
                
            except Exception as e:
                logger.error(f"Error validating strategy {strategy.get('name', 'Unknown')}: {e}")
                critical_issues.append(f"Validation error for {strategy.get('name', 'Unknown')}: {e}")
        
        # Calculate overall results
        overall_passed = len(critical_issues) == 0
        overall_score = self._calculate_overall_score(all_results)
        overfitting_risk = max([r.get("overfitting_risk", 0.0) for r in all_results], default=0.0)
        data_leakage_detected = any(r.get("data_leakage_detected", False) for r in all_results)
        
        validation_summary = {
            "overall_passed": overall_passed,
            "overall_score": overall_score,
            "critical_issues": critical_issues,
            "warnings": warnings,
            "recommendations": recommendations,
            "overfitting_risk": overfitting_risk,
            "data_leakage_detected": data_leakage_detected,
            "individual_results": all_results,
            "validation_metadata": {
                "validated_at": datetime.now().isoformat(),
                "agent_id": self.agent_id,
                "strictness": self.validation_strictness,
                "strategies_count": len(coded_strategies),
            }
        }
        
        # Store validation history
        self._validation_history.append(validation_summary)
        
        return validation_summary
    
    async def _validate_single_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single strategy."""
        
        strategy_code = strategy.get("code", "")
        strategy_name = strategy.get("name", "Unknown")
        
        logger.info(f"Validating strategy: {strategy_name}")
        
        # Perform different types of validation
        code_quality_results = await self._validate_code_quality(strategy_code)
        logic_results = await self._validate_logic(strategy, strategy_code)
        data_integrity_results = await self._validate_data_integrity(strategy_code)
        risk_mgmt_results = await self._validate_risk_management(strategy, strategy_code)
        performance_results = await self._validate_performance_expectations(strategy)
        overfitting_results = await self._validate_overfitting_risk(strategy, strategy_code)
        
        # Use LLM for comprehensive analysis
        llm_analysis = await self._llm_comprehensive_validation(strategy, strategy_code)
        
        # Aggregate results
        all_issues = []
        all_warnings = []
        all_recommendations = []
        
        for result in [code_quality_results, logic_results, data_integrity_results, 
                      risk_mgmt_results, performance_results, overfitting_results]:
            all_issues.extend(result.get("critical_issues", []))
            all_warnings.extend(result.get("warnings", []))
            all_recommendations.extend(result.get("recommendations", []))
        
        # Add LLM insights
        if llm_analysis:
            all_issues.extend(llm_analysis.get("critical_issues", []))
            all_warnings.extend(llm_analysis.get("warnings", []))
            all_recommendations.extend(llm_analysis.get("recommendations", []))
        
        validation_score = self._calculate_validation_score(
            len(all_issues), len(all_warnings), len(all_recommendations)
        )
        
        return {
            "strategy_id": strategy.get("strategy_id"),
            "strategy_name": strategy_name,
            "validation_passed": len(all_issues) == 0,
            "validation_score": validation_score,
            "critical_issues": all_issues,
            "warnings": all_warnings,
            "recommendations": all_recommendations,
            "overfitting_risk": overfitting_results.get("risk_score", 0.0),
            "data_leakage_detected": data_integrity_results.get("data_leakage_detected", False),
            "detailed_results": {
                "code_quality": code_quality_results,
                "logic_validation": logic_results,
                "data_integrity": data_integrity_results,
                "risk_management": risk_mgmt_results,
                "performance": performance_results,
                "overfitting": overfitting_results,
                "llm_analysis": llm_analysis,
            }
        }
    
    async def _validate_code_quality(self, code: str) -> Dict[str, Any]:
        """Validate code quality and syntax."""
        
        issues = []
        warnings = []
        recommendations = []
        
        try:
            # Parse the code to check syntax
            ast.parse(code)
        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
        
        # Check for basic code quality issues
        lines = code.split('\n')
        
        # Check for proper imports
        has_imports = any(line.strip().startswith(('import ', 'from ')) for line in lines)
        if not has_imports:
            warnings.append("No imports found - strategy may be incomplete")
        
        # Check for error handling
        has_try_catch = any('try:' in line or 'except' in line for line in lines)
        if not has_try_catch:
            warnings.append("No error handling found - consider adding try/except blocks")
        
        # Check for documentation
        has_docstrings = any('"""' in line or "'''" in line for line in lines)
        if not has_docstrings:
            recommendations.append("Add docstrings for better code documentation")
        
        # Check for logging
        has_logging = any('log' in line.lower() for line in lines)
        if not has_logging:
            recommendations.append("Consider adding logging for better monitoring")
        
        return {
            "critical_issues": issues,
            "warnings": warnings,
            "recommendations": recommendations,
        }
    
    async def _validate_logic(self, strategy: Dict[str, Any], code: str) -> Dict[str, Any]:
        """Validate strategy logic and conditions."""
        
        issues = []
        warnings = []
        recommendations = []
        
        # Check for entry conditions
        entry_conditions = strategy.get("entry_conditions", [])
        if not entry_conditions:
            issues.append("No entry conditions specified")
        
        # Check for exit conditions
        exit_conditions = strategy.get("exit_conditions", [])
        if not exit_conditions:
            issues.append("No exit conditions specified")
        
        # Check for position sizing logic
        if "position" not in code.lower() and "size" not in code.lower():
            warnings.append("Position sizing logic not clearly implemented")
        
        # Check for signal generation
        if "signal" not in code.lower():
            warnings.append("Signal generation logic not clearly implemented")
        
        return {
            "critical_issues": issues,
            "warnings": warnings,
            "recommendations": recommendations,
        }
    
    async def _validate_data_integrity(self, code: str) -> Dict[str, Any]:
        """Validate data integrity and check for common biases."""
        
        issues = []
        warnings = []
        recommendations = []
        data_leakage_detected = False
        
        # Check for potential look-ahead bias
        lookahead_patterns = [
            r'\.shift\(-\d+\)',  # Negative shifts
            r'\.iloc\[\d+:\]',   # Future data access
            r'\.loc\[.*:\]',     # Potential future data
        ]
        
        for pattern in lookahead_patterns:
            if re.search(pattern, code):
                issues.append(f"Potential look-ahead bias detected: {pattern}")
                data_leakage_detected = True
        
        # Check for data leakage indicators
        leakage_keywords = ['future', 'tomorrow', 'next_day', 'forward']
        for keyword in leakage_keywords:
            if keyword in code.lower():
                warnings.append(f"Potential data leakage keyword found: {keyword}")
                data_leakage_detected = True
        
        # Check for proper data handling
        if 'dropna' not in code and 'fillna' not in code:
            recommendations.append("Consider handling missing data explicitly")
        
        return {
            "critical_issues": issues,
            "warnings": warnings,
            "recommendations": recommendations,
            "data_leakage_detected": data_leakage_detected,
        }
    
    async def _validate_risk_management(self, strategy: Dict[str, Any], code: str) -> Dict[str, Any]:
        """Validate risk management implementation."""
        
        issues = []
        warnings = []
        recommendations = []
        
        # Check for stop loss implementation
        risk_mgmt = strategy.get("risk_management", {})
        if not risk_mgmt.get("stop_loss"):
            issues.append("No stop loss mechanism specified")
        
        # Check for position limits
        if not risk_mgmt.get("position_limit"):
            issues.append("No position size limits specified")
        
        # Check for drawdown controls
        if not risk_mgmt.get("max_drawdown"):
            warnings.append("No maximum drawdown limit specified")
        
        # Check code for risk management implementation
        if "stop" not in code.lower() and "loss" not in code.lower():
            warnings.append("Stop loss logic not clearly implemented in code")
        
        if "risk" not in code.lower():
            recommendations.append("Consider adding explicit risk calculations")
        
        return {
            "critical_issues": issues,
            "warnings": warnings,
            "recommendations": recommendations,
        }
    
    async def _validate_performance_expectations(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Validate performance expectations for realism."""
        
        issues = []
        warnings = []
        recommendations = []
        
        targets = strategy.get("performance_targets", {})
        
        # Check annual return expectations
        annual_return = targets.get("annual_return", 0)
        if annual_return > 0.5:  # 50% annual return
            warnings.append(f"Very high return expectation: {annual_return:.1%}")
        elif annual_return < 0:
            warnings.append(f"Negative return expectation: {annual_return:.1%}")
        
        # Check Sharpe ratio expectations
        sharpe_ratio = targets.get("sharpe_ratio", 0)
        if sharpe_ratio > 3.0:
            warnings.append(f"Very high Sharpe ratio expectation: {sharpe_ratio:.2f}")
        elif sharpe_ratio < 0:
            warnings.append(f"Negative Sharpe ratio expectation: {sharpe_ratio:.2f}")
        
        # Check drawdown expectations
        max_drawdown = targets.get("max_drawdown", 0)
        if max_drawdown > 0.3:  # 30% drawdown
            warnings.append(f"High drawdown tolerance: {max_drawdown:.1%}")
        
        # Check win rate
        win_rate = targets.get("win_rate", 0)
        if win_rate > 0.8:  # 80% win rate
            warnings.append(f"Very high win rate expectation: {win_rate:.1%}")
        
        return {
            "critical_issues": issues,
            "warnings": warnings,
            "recommendations": recommendations,
        }
    
    async def _validate_overfitting_risk(self, strategy: Dict[str, Any], code: str) -> Dict[str, Any]:
        """Assess overfitting risk."""
        
        issues = []
        warnings = []
        recommendations = []
        
        # Count parameters
        parameters = strategy.get("parameters", {})
        param_count = len(parameters)
        
        risk_score = 0.0
        
        # Assess based on parameter count
        if param_count > 10:
            warnings.append(f"High parameter count ({param_count}) may indicate overfitting")
            risk_score += 0.3
        elif param_count > 5:
            recommendations.append(f"Moderate parameter count ({param_count}) - monitor for overfitting")
            risk_score += 0.1
        
        # Check for optimization keywords
        optimization_keywords = ['optimize', 'grid_search', 'best_params', 'tune']
        for keyword in optimization_keywords:
            if keyword in code.lower():
                warnings.append(f"Optimization keyword found: {keyword} - ensure out-of-sample testing")
                risk_score += 0.2
        
        # Check strategy complexity
        code_lines = len(code.split('\n'))
        if code_lines > 200:
            recommendations.append("High code complexity - consider simplification")
            risk_score += 0.1
        
        return {
            "critical_issues": issues,
            "warnings": warnings,
            "recommendations": recommendations,
            "risk_score": min(risk_score, 1.0),
        }
    
    async def _llm_comprehensive_validation(self, strategy: Dict[str, Any], code: str) -> Optional[Dict[str, Any]]:
        """Use LLM for comprehensive strategy validation."""
        
        prompt = f"""Perform a comprehensive validation of this trading strategy:

Strategy Specification:
{json.dumps(strategy, indent=2)}

Strategy Code:
```python
{code}
```

Analyze for:
1. Logic errors and edge cases
2. Data integrity issues (look-ahead bias, data leakage)
3. Risk management adequacy
4. Performance expectation realism
5. Code quality and best practices
6. Potential overfitting concerns

Provide specific, actionable feedback in JSON format:
{{
    "critical_issues": ["issue1", "issue2"],
    "warnings": ["warning1", "warning2"],
    "recommendations": ["rec1", "rec2"],
    "overall_assessment": "description",
    "deployment_readiness": "ready/needs_work/not_ready"
}}"""
        
        try:
            response = await self._call_model([UserMessage(content=prompt, source="user")])
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error in LLM validation: {e}")
            return None
    
    def _calculate_validation_score(self, critical_count: int, warning_count: int, rec_count: int) -> float:
        """Calculate overall validation score."""
        
        # Start with perfect score
        score = 1.0
        
        # Deduct for issues
        score -= critical_count * 0.3  # Critical issues heavily penalized
        score -= warning_count * 0.1   # Warnings moderately penalized
        score -= rec_count * 0.02      # Recommendations lightly penalized
        
        return max(0.0, min(1.0, score))
    
    def _calculate_overall_score(self, results: List[Dict[str, Any]]) -> float:
        """Calculate overall score across all strategies."""
        
        if not results:
            return 0.0
        
        scores = [r.get("validation_score", 0.0) for r in results]
        return sum(scores) / len(scores)
    
    async def get_validation_history(self) -> List[Dict[str, Any]]:
        """Get validation history."""
        return self._validation_history.copy()
    
    async def update_validation_rules(self, rules: Dict[str, Any]):
        """Update validation rules."""
        self._validation_rules.update(rules)
        logger.info("Updated validation rules")
