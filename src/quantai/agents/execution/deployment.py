"""
Strategy Deployment Agent for the QuantAI multi-agent system.

This agent is responsible for deploying validated trading strategies
to the live trading environment.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional

from autogen_core import Agent, MessageContext
from ...core.base import BaseQuantAgent, AgentRole, AgentCapability
from ...core.messages import QuantMessage, MessageType, StrategyMessage


class StrategyDeploymentAgent(BaseQuantAgent):
    """
    Agent responsible for deploying validated trading strategies.
    
    This agent handles the deployment of strategies from validation
    to live trading environments, including configuration management
    and deployment monitoring.
    """
    
    def __init__(self, agent_id: str = "strategy_deployment"):
        super().__init__(
            role=AgentRole.STRATEGY_DEPLOYMENT,
            capabilities=[
                AgentCapability.STRATEGY_DEPLOYMENT,
                AgentCapability.SYSTEM_INTEGRATION,
            ]
        )
        self.agent_id = agent_id
        self.deployed_strategies: Dict[str, Dict[str, Any]] = {}
        self.deployment_history: List[Dict[str, Any]] = []
    
    async def on_messages(self, messages: List[QuantMessage], ctx: MessageContext) -> str:
        """Handle incoming messages for strategy deployment."""
        results = []
        
        for message in messages:
            if isinstance(message, StrategyMessage):
                result = await self._handle_strategy_deployment(message)
                results.append(result)
            else:
                result = await self._handle_general_message(message)
                results.append(result)
        
        return f"StrategyDeploymentAgent processed {len(results)} messages"
    
    async def _handle_strategy_deployment(self, message: StrategyMessage) -> Dict[str, Any]:
        """Handle strategy deployment request."""
        try:
            strategy_id = message.strategy_id
            
            # Validate strategy for deployment
            if not await self._validate_for_deployment(message):
                return {
                    'status': 'failed',
                    'strategy_id': strategy_id,
                    'error': 'Strategy validation failed'
                }
            
            # Deploy strategy
            deployment_result = await self._deploy_strategy(message)
            
            # Record deployment
            self.deployment_history.append({
                'strategy_id': strategy_id,
                'deployment_time': datetime.now(),
                'status': deployment_result['status'],
                'details': deployment_result
            })
            
            return deployment_result
            
        except Exception as e:
            return {
                'status': 'error',
                'strategy_id': getattr(message, 'strategy_id', 'unknown'),
                'error': str(e)
            }
    
    async def _validate_for_deployment(self, strategy: StrategyMessage) -> bool:
        """Validate strategy is ready for deployment."""
        # Check if strategy has required components
        if not strategy.strategy_code:
            return False
        
        if not strategy.strategy_parameters:
            return False
        
        # Additional validation logic would go here
        return True
    
    async def _deploy_strategy(self, strategy: StrategyMessage) -> Dict[str, Any]:
        """Deploy strategy to live environment."""
        strategy_id = strategy.strategy_id
        
        # Simulate deployment process
        await asyncio.sleep(0.1)  # Simulate deployment time
        
        # Record deployed strategy
        self.deployed_strategies[strategy_id] = {
            'strategy_name': strategy.strategy_name,
            'deployment_time': datetime.now(),
            'status': 'active',
            'parameters': strategy.strategy_parameters,
            'code': strategy.strategy_code
        }
        
        return {
            'status': 'deployed',
            'strategy_id': strategy_id,
            'deployment_time': datetime.now(),
            'message': f'Strategy {strategy_id} deployed successfully'
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
                result = await self._handle_strategy_deployment(message)
                return QuantMessage(
                    message_type=MessageType.DEPLOYMENT_RESPONSE,
                    sender_id=self.agent_id,
                    data_payload=result
                )
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
    
    def get_deployed_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Get list of currently deployed strategies."""
        return self.deployed_strategies.copy()
    
    def get_deployment_history(self) -> List[Dict[str, Any]]:
        """Get deployment history."""
        return self.deployment_history.copy()
