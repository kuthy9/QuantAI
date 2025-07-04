"""
Strategy Generation Agent (A1) for the QuantAI system.

This agent generates multiple draft trading strategies based on market inputs,
macro insights, and fused signals from other agents.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from autogen_core import MessageContext
from autogen_core.models import ChatCompletionClient, UserMessage
from loguru import logger

from ...core.base import AgentCapability, AgentRole, ModelCapableAgent
from ...core.messages import MessageType, QuantMessage, StrategyMessage


class StrategyGenerationAgent(ModelCapableAgent):
    """
    Strategy Generation Agent (A1) - Generates trading strategies.
    
    Capabilities:
    - Multi-timeframe strategy generation (intraday, swing, position)
    - Strategy type diversification (momentum, mean reversion, arbitrage)
    - Risk-adjusted strategy optimization
    - Market regime adaptive strategies
    - Multi-asset strategy development
    """
    
    def __init__(
        self,
        model_client: ChatCompletionClient,
        max_strategies_per_request: int = 5,
        strategy_types: List[str] = None,
        **kwargs
    ):
        super().__init__(
            role=AgentRole.STRATEGY_GENERATION,
            capabilities=[
                AgentCapability.STRATEGY_GENERATION,
            ],
            model_client=model_client,
            system_message=self._get_system_message(),
            **kwargs
        )
        
        self.max_strategies_per_request = max_strategies_per_request
        self.strategy_types = strategy_types or [
            "momentum", "mean_reversion", "breakout", "pairs_trading",
            "statistical_arbitrage", "trend_following", "contrarian"
        ]
        self._generated_strategies: Dict[str, Dict[str, Any]] = {}
    
    def _get_system_message(self) -> str:
        return """You are a Strategy Generation Agent specializing in quantitative trading strategy development.

Your responsibilities:
1. Generate diverse trading strategies based on market signals and macro context
2. Adapt strategies to current market regimes and conditions
3. Optimize strategies for risk-adjusted returns
4. Create strategies across multiple timeframes and asset classes
5. Ensure strategy diversification and complementarity
6. Incorporate regime-aware and adaptive elements

Strategy Categories:
- Momentum: Trend-following, breakout, relative strength
- Mean Reversion: Statistical arbitrage, pairs trading, contrarian
- Cross-Asset: Multi-asset momentum, risk parity, carry trades
- Volatility: Vol targeting, dispersion trading, VIX strategies
- Factor: Value, quality, low volatility, size factors
- Alternative: Event-driven, seasonal, calendar effects

Strategy Framework:
1. Signal Generation: Entry and exit conditions
2. Position Sizing: Risk management and capital allocation
3. Risk Controls: Stop losses, position limits, drawdown controls
4. Regime Adaptation: How strategy adapts to market conditions
5. Performance Targets: Expected returns, Sharpe ratio, max drawdown

Guidelines:
- Generate strategies with clear, testable hypotheses
- Include specific entry/exit rules and risk management
- Consider transaction costs and market impact
- Ensure strategies are implementable with available data
- Provide rationale for each strategy based on market conditions
- Include performance expectations and risk metrics
- Design strategies that complement existing portfolio

Focus on creating robust, diversified strategies that can perform across different market environments while maintaining strong risk management."""
    
    async def process_message(
        self, 
        message: QuantMessage, 
        ctx: MessageContext
    ) -> Optional[QuantMessage]:
        """Process strategy generation requests."""
        
        if isinstance(message, StrategyMessage):
            if message.message_type == MessageType.STRATEGY_GENERATION:
                # Generate strategies based on macro context
                strategies = await self._generate_strategies(message)
                
                # Create response with generated strategies
                response = StrategyMessage(
                    message_type=MessageType.STRATEGY_CODE,
                    sender_id=self.agent_id,
                    strategy_id=f"generated_strategies_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    strategy_name="Generated Strategy Portfolio",
                    strategy_description="Portfolio of generated trading strategies",
                    strategy_parameters=strategies,
                    session_id=message.session_id,
                    correlation_id=message.correlation_id,
                )
                
                logger.info(f"Generated {len(strategies.get('strategies', []))} strategies")
                return response
        
        return None
    
    async def _generate_strategies(self, macro_message: StrategyMessage) -> Dict[str, Any]:
        """Generate trading strategies based on macro context."""
        
        macro_context = macro_message.strategy_parameters
        
        # Extract key information from macro context
        market_environment = macro_context.get("market_environment", {})
        regime_data = macro_context.get("metadata", {}).get("regime_data", {})
        
        # Generate strategies using LLM
        strategies = await self._llm_strategy_generation(macro_context)
        
        # Post-process and validate strategies
        validated_strategies = await self._validate_strategies(strategies)
        
        return {
            "strategies": validated_strategies,
            "generation_metadata": {
                "generated_at": datetime.now().isoformat(),
                "agent_id": self.agent_id,
                "macro_context": macro_context,
                "strategy_count": len(validated_strategies),
            }
        }
    
    async def _llm_strategy_generation(self, macro_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use LLM to generate trading strategies."""
        
        prompt = f"""Based on the following macro market analysis, generate {self.max_strategies_per_request} diverse quantitative trading strategies:

Macro Context:
{json.dumps(macro_context, indent=2)}

For each strategy, provide:

1. Strategy Overview
   - Name and type (momentum/mean_reversion/breakout/etc.)
   - Core hypothesis and market rationale
   - Target timeframe (intraday/daily/weekly/monthly)
   - Target assets/universe

2. Signal Generation
   - Entry conditions (specific rules and thresholds)
   - Exit conditions (profit targets, stop losses, time exits)
   - Signal strength/confidence scoring
   - Lookback periods and parameters

3. Position Management
   - Position sizing methodology
   - Maximum position size limits
   - Portfolio allocation percentage
   - Leverage constraints

4. Risk Management
   - Stop loss methodology
   - Maximum drawdown limits
   - Position correlation limits
   - Regime-based risk adjustments

5. Performance Expectations
   - Expected annual return
   - Expected Sharpe ratio
   - Maximum expected drawdown
   - Win rate expectations

6. Implementation Details
   - Required data sources
   - Calculation frequency
   - Transaction cost considerations
   - Market impact considerations

Generate strategies that:
- Are well-suited to the current market environment
- Complement each other (low correlation)
- Have clear risk management rules
- Are implementable with standard market data
- Adapt to changing market regimes

Respond in JSON format:
{{
    "strategies": [
        {{
            "name": "Strategy Name",
            "type": "momentum/mean_reversion/etc",
            "timeframe": "daily/weekly/etc",
            "universe": ["SPY", "QQQ", "etc"],
            "hypothesis": "Core trading hypothesis",
            "entry_conditions": ["condition1", "condition2"],
            "exit_conditions": ["condition1", "condition2"],
            "position_sizing": "methodology description",
            "risk_management": {{
                "stop_loss": "methodology",
                "max_drawdown": 0.15,
                "position_limit": 0.1
            }},
            "performance_targets": {{
                "annual_return": 0.12,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.55
            }},
            "parameters": {{
                "lookback_period": 20,
                "threshold": 0.02,
                "other_params": "value"
            }},
            "implementation": {{
                "data_requirements": ["price", "volume"],
                "frequency": "daily",
                "complexity": "medium"
            }}
        }}
    ]
}}"""
        
        try:
            response = await self._call_model([UserMessage(content=prompt, source="user")])
            strategies_data = json.loads(response)
            return strategies_data.get("strategies", [])
            
        except Exception as e:
            logger.error(f"Error in LLM strategy generation: {e}")
            return []
    
    async def _validate_strategies(self, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and enhance generated strategies."""
        validated = []
        
        for strategy in strategies:
            try:
                # Add unique ID
                strategy["strategy_id"] = str(uuid.uuid4())
                strategy["created_at"] = datetime.now().isoformat()
                
                # Validate required fields
                required_fields = ["name", "type", "hypothesis", "entry_conditions", "exit_conditions"]
                if all(field in strategy for field in required_fields):
                    
                    # Add default values for missing fields
                    strategy = await self._add_default_values(strategy)
                    
                    # Perform basic validation
                    if await self._basic_strategy_validation(strategy):
                        validated.append(strategy)
                        
                        # Store strategy for future reference
                        self._generated_strategies[strategy["strategy_id"]] = strategy
                    else:
                        logger.warning(f"Strategy validation failed: {strategy.get('name', 'Unknown')}")
                else:
                    logger.warning(f"Strategy missing required fields: {strategy.get('name', 'Unknown')}")
                    
            except Exception as e:
                logger.error(f"Error validating strategy: {e}")
        
        return validated
    
    async def _add_default_values(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Add default values for missing strategy fields."""
        
        # Default timeframe
        if "timeframe" not in strategy:
            strategy["timeframe"] = "daily"
        
        # Default universe
        if "universe" not in strategy:
            strategy["universe"] = ["SPY", "QQQ", "IWM"]
        
        # Default risk management
        if "risk_management" not in strategy:
            strategy["risk_management"] = {
                "stop_loss": "5% trailing stop",
                "max_drawdown": 0.15,
                "position_limit": 0.1
            }
        
        # Default performance targets
        if "performance_targets" not in strategy:
            strategy["performance_targets"] = {
                "annual_return": 0.10,
                "sharpe_ratio": 1.0,
                "max_drawdown": 0.10,
                "win_rate": 0.50
            }
        
        # Default parameters
        if "parameters" not in strategy:
            strategy["parameters"] = {
                "lookback_period": 20,
                "rebalance_frequency": "daily"
            }
        
        # Default implementation details
        if "implementation" not in strategy:
            strategy["implementation"] = {
                "data_requirements": ["price", "volume"],
                "frequency": strategy["timeframe"],
                "complexity": "medium"
            }
        
        return strategy
    
    async def _basic_strategy_validation(self, strategy: Dict[str, Any]) -> bool:
        """Perform basic validation on strategy structure."""
        
        try:
            # Check performance targets are reasonable
            targets = strategy.get("performance_targets", {})
            annual_return = targets.get("annual_return", 0)
            sharpe_ratio = targets.get("sharpe_ratio", 0)
            max_drawdown = targets.get("max_drawdown", 0)
            
            # Basic sanity checks
            if annual_return < -0.5 or annual_return > 2.0:  # -50% to 200% annual return
                return False
            
            if sharpe_ratio < -2.0 or sharpe_ratio > 5.0:  # Reasonable Sharpe ratio range
                return False
            
            if max_drawdown < 0 or max_drawdown > 0.5:  # 0% to 50% max drawdown
                return False
            
            # Check risk management
            risk_mgmt = strategy.get("risk_management", {})
            position_limit = risk_mgmt.get("position_limit", 0)
            
            if position_limit <= 0 or position_limit > 1.0:  # 0% to 100% position limit
                return False
            
            # Check universe is not empty
            universe = strategy.get("universe", [])
            if not universe:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in strategy validation: {e}")
            return False
    
    async def get_generated_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Get all generated strategies."""
        return self._generated_strategies.copy()
    
    async def get_strategy_by_id(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific strategy by ID."""
        return self._generated_strategies.get(strategy_id)
    
    async def generate_custom_strategy(
        self,
        strategy_type: str,
        target_assets: List[str],
        timeframe: str,
        additional_requirements: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Generate a custom strategy with specific requirements."""
        
        prompt = f"""Generate a {strategy_type} trading strategy with the following specifications:

Target Assets: {target_assets}
Timeframe: {timeframe}
Additional Requirements: {additional_requirements}

Provide a complete strategy specification including entry/exit rules, risk management, and performance expectations.

Use the same JSON format as previous strategy generation requests."""
        
        try:
            response = await self._call_model([UserMessage(content=prompt, source="user")])
            strategy_data = json.loads(response)
            
            if "strategies" in strategy_data and strategy_data["strategies"]:
                strategy = strategy_data["strategies"][0]
                validated = await self._validate_strategies([strategy])
                return validated[0] if validated else None
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating custom strategy: {e}")
            return None
