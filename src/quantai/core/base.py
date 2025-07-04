"""
Base agent classes and protocols for the QuantAI system.

This module provides the foundational agent architecture built on AutoGen's
RoutedAgent pattern, with specialized capabilities for financial applications.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

from autogen_core import (
    AgentId,
    MessageContext,
    RoutedAgent,
    TopicId,
    default_subscription,
    message_handler,
)
from autogen_core.models import ChatCompletionClient, LLMMessage, SystemMessage, UserMessage
from loguru import logger
from pydantic import BaseModel

from .messages import QuantMessage, MessageType, Priority


class AgentRole(str, Enum):
    """Roles of agents in the financial system."""
    DATA_INGESTION = "data_ingestion"
    MULTIMODAL_FUSION = "multimodal_fusion"
    MACRO_INSIGHT = "macro_insight"
    STRATEGY_GENERATION = "strategy_generation"
    STRATEGY_CODING = "strategy_coding"
    STRATEGY_VALIDATION = "strategy_validation"
    STRATEGY_DEPLOYMENT = "strategy_deployment"
    EXECUTION = "execution"
    BACKTEST_MONITOR = "backtest_monitor"
    RISK_CONTROL = "risk_control"
    PROFITABILITY = "profitability"
    FEEDBACK_LOOP = "feedback_loop"
    MEMORY = "memory"
    API_MANAGER = "api_manager"
    KILL_SWITCH = "kill_switch"
    DASHBOARD = "dashboard"


class AgentCapability(str, Enum):
    """Capabilities that agents can have."""
    DATA_COLLECTION = "data_collection"
    DATA_PROCESSING = "data_processing"
    DATA_FUSION = "data_fusion"
    MULTIMODAL_ANALYSIS = "multimodal_analysis"
    STRATEGY_GENERATION = "strategy_generation"
    STRATEGY_VALIDATION = "strategy_validation"
    STRATEGY_DEPLOYMENT = "strategy_deployment"
    CODE_GENERATION = "code_generation"
    CODE_VALIDATION = "code_validation"
    CODE_ANALYSIS = "code_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    TRADE_EXECUTION = "trade_execution"
    ORDER_MANAGEMENT = "order_management"
    MARKET_ACCESS = "market_access"
    PERFORMANCE_MONITORING = "performance_monitoring"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    DATA_ANALYSIS = "data_analysis"
    BACKTESTING = "backtesting"
    LEARNING = "learning"
    MEMORY_MANAGEMENT = "memory_management"
    KNOWLEDGE_MANAGEMENT = "knowledge_management"
    SYSTEM_INTEGRATION = "system_integration"


class AgentStatus(str, Enum):
    """Status of agents."""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    STOPPED = "stopped"


@default_subscription
class BaseQuantAgent(RoutedAgent, ABC):
    """
    Base class for all QuantAI agents.
    
    Provides common functionality for financial agents including:
    - Message handling and routing
    - Status management
    - Performance tracking
    - Error handling and recovery
    - Memory integration
    """

    def __init__(
        self,
        role: AgentRole,
        capabilities: List[AgentCapability],
        model_client: Optional[ChatCompletionClient] = None,
        system_message: Optional[str] = None,
        max_retries: int = 3,
        timeout_seconds: int = 300,
    ):
        super().__init__(f"QuantAI {role.value} agent")
        
        self.role = role
        self.capabilities = capabilities
        self.agent_id = str(uuid.uuid4())
        self.status = AgentStatus.INITIALIZING
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        
        # Model and messaging
        self._model_client = model_client
        self._system_message = system_message or self._default_system_message()
        self._max_retries = max_retries
        self._timeout_seconds = timeout_seconds
        
        # Performance tracking
        self._messages_processed = 0
        self._errors_count = 0
        self._average_response_time = 0.0
        
        # Session management
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Initialized {self.role.value} agent with ID {self.agent_id}")
        self.status = AgentStatus.READY

    def _default_system_message(self) -> str:
        """Generate default system message based on agent role."""
        return f"""You are a {self.role.value} agent in a multi-agent financial system.
        
Your role: {self.role.value}
Your capabilities: {', '.join([cap.value for cap in self.capabilities])}

You work collaboratively with other agents to:
- Analyze financial markets and data
- Generate and validate trading strategies
- Manage risk and execute trades
- Learn from performance and improve

Always provide clear, actionable responses and coordinate effectively with other agents.
Be precise with financial data and calculations. Prioritize risk management and compliance.
"""

    async def _update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()

    async def _track_performance(self, start_time: datetime, success: bool):
        """Track agent performance metrics."""
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Update average response time
        self._average_response_time = (
            (self._average_response_time * self._messages_processed + response_time) /
            (self._messages_processed + 1)
        )
        
        self._messages_processed += 1
        if not success:
            self._errors_count += 1

    @abstractmethod
    async def process_message(
        self, 
        message: QuantMessage, 
        ctx: MessageContext
    ) -> Optional[QuantMessage]:
        """
        Process incoming messages. Must be implemented by subclasses.
        
        Args:
            message: The incoming message to process
            ctx: Message context from AutoGen
            
        Returns:
            Optional response message
        """
        pass

    @message_handler
    async def handle_quant_message(self, message: QuantMessage, ctx: MessageContext) -> None:
        """
        Main message handler that routes to specific processing logic.
        """
        start_time = datetime.utcnow()
        success = False
        
        try:
            await self._update_activity()
            self.status = AgentStatus.BUSY
            
            logger.info(
                f"Agent {self.agent_id} ({self.role.value}) processing message "
                f"{message.message_id} of type {message.message_type}"
            )
            
            # Process the message
            response = await self.process_message(message, ctx)
            
            # Send response if generated
            if response:
                await self.publish_message(
                    response, 
                    topic_id=TopicId("default", self.id.key)
                )
            
            success = True
            self.status = AgentStatus.READY
            
        except Exception as e:
            logger.error(f"Error processing message in {self.role.value} agent: {e}")
            self.status = AgentStatus.ERROR
            self._errors_count += 1
            
            # Could implement retry logic here
            raise
            
        finally:
            await self._track_performance(start_time, success)

    async def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics."""
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "messages_processed": self._messages_processed,
            "errors_count": self._errors_count,
            "error_rate": self._errors_count / max(self._messages_processed, 1),
            "average_response_time": self._average_response_time,
            "active_sessions": len(self._active_sessions),
        }

    async def health_check(self) -> bool:
        """Perform health check on the agent."""
        try:
            # Check if agent is responsive
            if self.status == AgentStatus.ERROR:
                return False
                
            # Check error rate
            error_rate = self._errors_count / max(self._messages_processed, 1)
            if error_rate > 0.1:  # More than 10% error rate
                return False
                
            # Check if agent has been active recently (within last hour)
            time_since_activity = (datetime.utcnow() - self.last_activity).total_seconds()
            if time_since_activity > 3600:  # 1 hour
                logger.warning(f"Agent {self.agent_id} has been inactive for {time_since_activity} seconds")
                
            return True
            
        except Exception as e:
            logger.error(f"Health check failed for agent {self.agent_id}: {e}")
            return False

    async def shutdown(self):
        """Gracefully shutdown the agent."""
        logger.info(f"Shutting down agent {self.agent_id} ({self.role.value})")
        self.status = AgentStatus.STOPPED
        
        # Close model client if available
        if self._model_client:
            await self._model_client.close()
            
        # Clear active sessions
        self._active_sessions.clear()


class ModelCapableAgent(BaseQuantAgent):
    """
    Base class for agents that use LLM models.
    
    Provides additional functionality for model interaction and context management.
    """

    def __init__(
        self,
        role: AgentRole,
        capabilities: List[AgentCapability],
        model_client: ChatCompletionClient,
        system_message: Optional[str] = None,
        max_context_length: int = 4000,
        **kwargs
    ):
        super().__init__(role, capabilities, model_client, system_message, **kwargs)
        self._max_context_length = max_context_length
        self._conversation_history: Dict[str, List[LLMMessage]] = {}

    async def _call_model(
        self, 
        messages: List[LLMMessage], 
        session_id: Optional[str] = None
    ) -> str:
        """
        Call the LLM model with context management.
        
        Args:
            messages: Messages to send to the model
            session_id: Optional session ID for conversation tracking
            
        Returns:
            Model response as string
        """
        if not self._model_client:
            raise ValueError("No model client configured for this agent")

        # Add system message if not present
        if not any(isinstance(msg, SystemMessage) for msg in messages):
            messages = [SystemMessage(content=self._system_message)] + messages

        # Manage context length
        if len(messages) > self._max_context_length:
            # Keep system message and recent messages
            system_msgs = [msg for msg in messages if isinstance(msg, SystemMessage)]
            other_msgs = [msg for msg in messages if not isinstance(msg, SystemMessage)]
            messages = system_msgs + other_msgs[-self._max_context_length:]

        # Store conversation history
        if session_id:
            self._conversation_history[session_id] = messages

        # Call model
        result = await self._model_client.create(messages)
        
        if isinstance(result.content, str):
            return result.content
        else:
            # Handle function calls or other content types
            return str(result.content)

    def get_conversation_history(self, session_id: str) -> List[LLMMessage]:
        """Get conversation history for a session."""
        return self._conversation_history.get(session_id, [])

    def clear_conversation_history(self, session_id: Optional[str] = None):
        """Clear conversation history for a session or all sessions."""
        if session_id:
            self._conversation_history.pop(session_id, None)
        else:
            self._conversation_history.clear()
