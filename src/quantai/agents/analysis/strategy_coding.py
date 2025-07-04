"""
Strategy Coding Agent (A2) for the QuantAI system.

This agent converts strategy ideas to executable code compatible with
trading platforms like QuantConnect Lean, Zipline, or Backtrader.
"""

import asyncio
import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from autogen_core import MessageContext
from autogen_core.models import ChatCompletionClient, UserMessage
from loguru import logger

from ...core.base import AgentCapability, AgentRole, ModelCapableAgent
from ...core.messages import MessageType, QuantMessage, StrategyMessage


class StrategyCodingAgent(ModelCapableAgent):
    """
    Strategy Coding Agent (A2) - Converts strategy ideas to executable code.
    
    Capabilities:
    - Strategy code generation for multiple platforms
    - Code optimization and best practices implementation
    - Risk management code integration
    - Performance monitoring code generation
    - Code documentation and commenting
    """
    
    def __init__(
        self,
        model_client: ChatCompletionClient,
        target_platform: str = "quantconnect",
        code_style: str = "production",
        **kwargs
    ):
        super().__init__(
            role=AgentRole.STRATEGY_CODING,
            capabilities=[
                AgentCapability.CODE_GENERATION,
            ],
            model_client=model_client,
            system_message=self._get_system_message(),
            **kwargs
        )
        
        self.target_platform = target_platform
        self.code_style = code_style
        self._coded_strategies: Dict[str, Dict[str, Any]] = {}
        self._platform_templates = self._initialize_platform_templates()
    
    def _get_system_message(self) -> str:
        return """You are a Strategy Coding Agent specializing in converting trading strategies to executable code.

Your responsibilities:
1. Convert strategy specifications to clean, executable code
2. Implement proper risk management and position sizing
3. Add comprehensive error handling and logging
4. Optimize code for performance and maintainability
5. Include thorough documentation and comments
6. Ensure code follows platform-specific best practices

Supported Platforms:
- QuantConnect Lean: C# and Python algorithms
- Zipline: Python backtesting framework
- Backtrader: Python trading framework
- Custom: Generic Python implementation

Code Quality Standards:
- Clean, readable, and well-documented code
- Proper error handling and edge case management
- Efficient data structures and algorithms
- Modular design with reusable components
- Comprehensive logging and monitoring
- Unit testable code structure

Risk Management Integration:
- Position sizing based on volatility and risk budget
- Stop loss and take profit implementation
- Drawdown monitoring and circuit breakers
- Portfolio-level risk controls
- Real-time risk metric calculation

Performance Monitoring:
- Real-time P&L tracking
- Performance metrics calculation
- Trade logging and analysis
- Benchmark comparison
- Risk-adjusted return metrics

Guidelines:
- Generate production-ready code with proper structure
- Include all necessary imports and dependencies
- Implement robust error handling for market data issues
- Add detailed comments explaining strategy logic
- Ensure code is testable and maintainable
- Follow platform-specific conventions and best practices
- Include configuration parameters for easy tuning

Focus on creating robust, professional-grade trading algorithms that can be deployed in live trading environments."""
    
    def _initialize_platform_templates(self) -> Dict[str, str]:
        """Initialize code templates for different platforms."""
        return {
            "quantconnect": """
# QuantConnect Lean Algorithm Template
from AlgorithmImports import *

class {strategy_name}(QCAlgorithm):
    def Initialize(self):
        # Algorithm initialization
        pass
    
    def OnData(self, data):
        # Main trading logic
        pass
""",
            "zipline": """
# Zipline Algorithm Template
import zipline.api as algo
import pandas as pd
import numpy as np

def initialize(context):
    # Algorithm initialization
    pass

def handle_data(context, data):
    # Main trading logic
    pass
""",
            "backtrader": """
# Backtrader Strategy Template
import backtrader as bt

class {strategy_name}(bt.Strategy):
    def __init__(self):
        # Strategy initialization
        pass
    
    def next(self):
        # Main trading logic
        pass
""",
        }
    
    async def process_message(
        self, 
        message: QuantMessage, 
        ctx: MessageContext
    ) -> Optional[QuantMessage]:
        """Process strategy coding requests."""
        
        if isinstance(message, StrategyMessage):
            if message.message_type == MessageType.STRATEGY_CODE:
                # Code the strategies
                coded_strategies = await self._code_strategies(message)
                
                # Create response with coded strategies
                response = StrategyMessage(
                    message_type=MessageType.STRATEGY_VALIDATION,
                    sender_id=self.agent_id,
                    strategy_id=message.strategy_id,
                    strategy_name=message.strategy_name,
                    strategy_description="Coded trading strategies ready for validation",
                    strategy_code=json.dumps(coded_strategies, indent=2),
                    strategy_parameters=coded_strategies,
                    session_id=message.session_id,
                    correlation_id=message.correlation_id,
                )
                
                logger.info(f"Coded {len(coded_strategies.get('coded_strategies', []))} strategies")
                return response
        
        return None
    
    async def _code_strategies(self, message: StrategyMessage) -> Dict[str, Any]:
        """Code all strategies from the generation message."""
        
        strategies_data = message.strategy_parameters
        strategies = strategies_data.get("strategies", [])
        
        coded_strategies = []
        
        for strategy in strategies:
            try:
                coded_strategy = await self._code_single_strategy(strategy)
                if coded_strategy:
                    coded_strategies.append(coded_strategy)
                    
                    # Store coded strategy
                    self._coded_strategies[coded_strategy["strategy_id"]] = coded_strategy
                    
            except Exception as e:
                logger.error(f"Error coding strategy {strategy.get('name', 'Unknown')}: {e}")
        
        return {
            "coded_strategies": coded_strategies,
            "coding_metadata": {
                "coded_at": datetime.now().isoformat(),
                "agent_id": self.agent_id,
                "platform": self.target_platform,
                "code_style": self.code_style,
                "total_strategies": len(coded_strategies),
            }
        }
    
    async def _code_single_strategy(self, strategy: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Code a single strategy specification."""
        
        strategy_name = strategy.get("name", "UnnamedStrategy")
        strategy_type = strategy.get("type", "momentum")
        
        # Generate code using LLM
        code = await self._generate_strategy_code(strategy)
        
        if not code:
            return None
        
        # Validate and enhance code
        enhanced_code = await self._enhance_code(code, strategy)
        
        # Extract key components
        code_components = self._extract_code_components(enhanced_code)
        
        coded_strategy = {
            "strategy_id": strategy.get("strategy_id", str(uuid.uuid4())),
            "name": strategy_name,
            "type": strategy_type,
            "platform": self.target_platform,
            "code": enhanced_code,
            "components": code_components,
            "original_strategy": strategy,
            "coded_at": datetime.now().isoformat(),
            "code_metadata": {
                "lines_of_code": len(enhanced_code.split('\n')),
                "complexity": self._assess_code_complexity(enhanced_code),
                "dependencies": self._extract_dependencies(enhanced_code),
            }
        }
        
        return coded_strategy
    
    async def _generate_strategy_code(self, strategy: Dict[str, Any]) -> Optional[str]:
        """Generate code for a strategy using LLM."""
        
        prompt = f"""Convert the following trading strategy specification to executable {self.target_platform} code:

Strategy Specification:
{json.dumps(strategy, indent=2)}

Requirements:
1. Generate complete, production-ready code for {self.target_platform}
2. Implement all entry and exit conditions specified
3. Include proper risk management and position sizing
4. Add comprehensive error handling and logging
5. Include detailed comments explaining the logic
6. Follow {self.target_platform} best practices and conventions
7. Make the code modular and maintainable

Code Structure:
- Class/function definitions with proper naming
- Initialization with strategy parameters
- Main trading logic implementation
- Risk management integration
- Performance tracking
- Proper imports and dependencies

Platform-Specific Requirements:
{self._get_platform_requirements()}

Generate clean, well-documented code that can be directly used in a trading environment.
Include all necessary imports and follow the platform's coding conventions.

Respond with only the code, no additional explanation:"""
        
        try:
            response = await self._call_model([UserMessage(content=prompt, source="user")])
            
            # Clean up the response to extract just the code
            code = self._extract_code_from_response(response)
            return code
            
        except Exception as e:
            logger.error(f"Error generating strategy code: {e}")
            return None
    
    def _get_platform_requirements(self) -> str:
        """Get platform-specific requirements."""
        
        requirements = {
            "quantconnect": """
- Use QCAlgorithm base class
- Implement Initialize() and OnData() methods
- Use self.SetStartDate(), self.SetEndDate(), self.SetCash()
- Use self.AddEquity() for universe selection
- Use self.MarketOrder(), self.LimitOrder() for trading
- Use self.Portfolio for position management
- Use self.Log() for logging
""",
            "zipline": """
- Use initialize() and handle_data() functions
- Use algo.order_target_percent() for position sizing
- Use data.current() for current prices
- Use context for state management
- Use algo.record() for performance tracking
""",
            "backtrader": """
- Inherit from bt.Strategy
- Implement __init__() and next() methods
- Use self.buy(), self.sell() for trading
- Use self.data for price data access
- Use self.broker for portfolio information
""",
        }
        
        return requirements.get(self.target_platform, "Generic Python implementation")
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract code from LLM response."""
        
        # Look for code blocks
        code_pattern = r'```(?:python|csharp|cs)?\n(.*?)\n```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks found, return the entire response
        return response.strip()
    
    async def _enhance_code(self, code: str, strategy: Dict[str, Any]) -> str:
        """Enhance generated code with additional features."""
        
        enhancement_prompt = f"""Enhance the following trading strategy code by adding:

1. Comprehensive error handling and edge case management
2. Detailed logging and monitoring
3. Performance metrics calculation
4. Risk management enhancements
5. Code optimization and best practices
6. Additional documentation and comments

Original Code:
```python
{code}
```

Strategy Context:
{json.dumps(strategy, indent=2)}

Return the enhanced code with all improvements integrated:"""
        
        try:
            response = await self._call_model([UserMessage(content=enhancement_prompt, source="user")])
            enhanced_code = self._extract_code_from_response(response)
            return enhanced_code
            
        except Exception as e:
            logger.error(f"Error enhancing code: {e}")
            return code  # Return original code if enhancement fails
    
    def _extract_code_components(self, code: str) -> Dict[str, List[str]]:
        """Extract key components from the code."""
        
        components = {
            "classes": [],
            "functions": [],
            "imports": [],
            "variables": [],
        }
        
        lines = code.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Extract imports
            if line.startswith('import ') or line.startswith('from '):
                components["imports"].append(line)
            
            # Extract class definitions
            elif line.startswith('class '):
                class_match = re.match(r'class\s+(\w+)', line)
                if class_match:
                    components["classes"].append(class_match.group(1))
            
            # Extract function definitions
            elif line.startswith('def '):
                func_match = re.match(r'def\s+(\w+)', line)
                if func_match:
                    components["functions"].append(func_match.group(1))
        
        return components
    
    def _assess_code_complexity(self, code: str) -> str:
        """Assess the complexity of the generated code."""
        
        lines = len(code.split('\n'))
        
        if lines < 50:
            return "low"
        elif lines < 150:
            return "medium"
        else:
            return "high"
    
    def _extract_dependencies(self, code: str) -> List[str]:
        """Extract dependencies from the code."""
        
        dependencies = []
        lines = code.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('import '):
                module = line.replace('import ', '').split(' as ')[0].split('.')[0]
                dependencies.append(module)
            
            elif line.startswith('from '):
                module = line.split(' ')[1].split('.')[0]
                dependencies.append(module)
        
        return list(set(dependencies))  # Remove duplicates
    
    async def get_coded_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Get all coded strategies."""
        return self._coded_strategies.copy()
    
    async def get_strategy_code(self, strategy_id: str) -> Optional[str]:
        """Get code for a specific strategy."""
        strategy = self._coded_strategies.get(strategy_id)
        return strategy.get("code") if strategy else None
    
    async def update_platform_target(self, platform: str):
        """Update the target platform for code generation."""
        if platform in self._platform_templates:
            self.target_platform = platform
            logger.info(f"Updated target platform to {platform}")
        else:
            logger.warning(f"Unsupported platform: {platform}")
    
    async def generate_test_code(self, strategy_id: str) -> Optional[str]:
        """Generate unit test code for a strategy."""
        
        strategy = self._coded_strategies.get(strategy_id)
        if not strategy:
            return None
        
        test_prompt = f"""Generate comprehensive unit tests for the following trading strategy code:

Strategy Code:
```python
{strategy['code']}
```

Generate tests that cover:
1. Strategy initialization
2. Entry condition logic
3. Exit condition logic
4. Risk management functions
5. Edge cases and error handling
6. Performance calculation methods

Use pytest framework and include mock data for testing.
Return only the test code:"""
        
        try:
            response = await self._call_model([UserMessage(content=test_prompt, source="user")])
            test_code = self._extract_code_from_response(response)
            return test_code
            
        except Exception as e:
            logger.error(f"Error generating test code: {e}")
            return None
