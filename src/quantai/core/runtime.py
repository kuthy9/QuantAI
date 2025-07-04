"""
Runtime management and orchestration for the QuantAI multi-agent system.

This module provides the central runtime that manages agent lifecycle,
message routing, and system coordination using AutoGen's runtime patterns.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union

from autogen_core import (
    AgentId,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
)
from loguru import logger

from .base import BaseQuantAgent, AgentRole, AgentStatus
from .config import QuantConfig, get_config
from .messages import QuantMessage, MessageType, ControlMessage


class AgentRegistry:
    """Registry for managing agent instances and their metadata."""
    
    def __init__(self):
        self._agents: Dict[str, BaseQuantAgent] = {}
        self._agent_roles: Dict[AgentRole, List[str]] = {}
        self._agent_types: Dict[str, Type[BaseQuantAgent]] = {}
    
    def register_agent_type(self, role: AgentRole, agent_class: Type[BaseQuantAgent]):
        """Register an agent class for a specific role."""
        self._agent_types[role.value] = agent_class
        logger.info(f"Registered agent type {agent_class.__name__} for role {role.value}")
    
    def add_agent(self, agent: BaseQuantAgent) -> str:
        """Add an agent instance to the registry."""
        agent_id = agent.agent_id
        self._agents[agent_id] = agent
        
        # Track by role
        if agent.role not in self._agent_roles:
            self._agent_roles[agent.role] = []
        self._agent_roles[agent.role].append(agent_id)
        
        logger.info(f"Added agent {agent_id} with role {agent.role.value}")
        return agent_id
    
    def remove_agent(self, agent_id: str):
        """Remove an agent from the registry."""
        if agent_id in self._agents:
            agent = self._agents[agent_id]
            self._agents.pop(agent_id)
            
            # Remove from role tracking
            if agent.role in self._agent_roles:
                self._agent_roles[agent.role].remove(agent_id)
                if not self._agent_roles[agent.role]:
                    del self._agent_roles[agent.role]
            
            logger.info(f"Removed agent {agent_id}")
    
    def get_agent(self, agent_id: str) -> Optional[BaseQuantAgent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)
    
    def get_agents_by_role(self, role: AgentRole) -> List[BaseQuantAgent]:
        """Get all agents with a specific role."""
        agent_ids = self._agent_roles.get(role, [])
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]
    
    def get_all_agents(self) -> List[BaseQuantAgent]:
        """Get all registered agents."""
        return list(self._agents.values())
    
    def get_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all agents."""
        return {
            agent_id: asyncio.create_task(agent.get_status())
            for agent_id, agent in self._agents.items()
        }


class MessageRouter:
    """Routes messages between agents based on message types and agent capabilities."""
    
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self._routing_rules: Dict[MessageType, List[AgentRole]] = {}
        self._setup_default_routing()
    
    def _setup_default_routing(self):
        """Setup default message routing rules."""
        self._routing_rules = {
            MessageType.DATA_REQUEST: [AgentRole.DATA_INGESTION],
            MessageType.DATA_RESPONSE: [AgentRole.MULTIMODAL_FUSION, AgentRole.MACRO_INSIGHT],
            MessageType.STRATEGY_GENERATION: [AgentRole.STRATEGY_GENERATION],
            MessageType.STRATEGY_CODE: [AgentRole.STRATEGY_CODING],
            MessageType.STRATEGY_VALIDATION: [AgentRole.STRATEGY_VALIDATION],
            MessageType.RISK_ASSESSMENT: [AgentRole.RISK_CONTROL],
            MessageType.TRADE_SIGNAL: [AgentRole.EXECUTION],
            MessageType.EXECUTION_RESULT: [AgentRole.BACKTEST_MONITOR, AgentRole.RISK_CONTROL],
            MessageType.BACKTEST_REQUEST: [AgentRole.BACKTEST_MONITOR],
            MessageType.BACKTEST_RESULT: [AgentRole.PROFITABILITY, AgentRole.FEEDBACK_LOOP],
            MessageType.FEEDBACK: [AgentRole.FEEDBACK_LOOP, AgentRole.MEMORY],
            MessageType.MEMORY_UPDATE: [AgentRole.MEMORY],
            MessageType.KILL_SWITCH: [AgentRole.KILL_SWITCH],
        }
    
    def add_routing_rule(self, message_type: MessageType, target_roles: List[AgentRole]):
        """Add or update a routing rule."""
        self._routing_rules[message_type] = target_roles
    
    def get_target_agents(self, message: QuantMessage) -> List[BaseQuantAgent]:
        """Get target agents for a message based on routing rules."""
        target_roles = self._routing_rules.get(message.message_type, [])
        target_agents = []
        
        for role in target_roles:
            agents = self.registry.get_agents_by_role(role)
            target_agents.extend(agents)
        
        # If specific recipient is specified, filter to that agent
        if message.recipient_id:
            target_agents = [
                agent for agent in target_agents 
                if agent.agent_id == message.recipient_id
            ]
        
        return target_agents


class QuantRuntime:
    """
    Main runtime for the QuantAI multi-agent system.
    
    Manages agent lifecycle, message routing, and system coordination.
    """
    
    def __init__(self, config: Optional[QuantConfig] = None):
        self.config = config or get_config()
        self.runtime = SingleThreadedAgentRuntime()
        self.registry = AgentRegistry()
        self.router = MessageRouter(self.registry)
        
        self._running = False
        self._startup_time: Optional[datetime] = None
        self._shutdown_time: Optional[datetime] = None
        
        logger.info("Initialized QuantAI runtime")
    
    async def register_agent_type(
        self, 
        role: AgentRole, 
        agent_class: Type[BaseQuantAgent]
    ):
        """Register an agent type with the runtime."""
        self.registry.register_agent_type(role, agent_class)
        
        # Register with AutoGen runtime
        await agent_class.register(
            self.runtime,
            role.value,
            lambda: self._create_agent_instance(role, agent_class)
        )
    
    def _create_agent_instance(
        self, 
        role: AgentRole, 
        agent_class: Type[BaseQuantAgent]
    ) -> BaseQuantAgent:
        """Create an agent instance with proper configuration."""
        agent_config = self.config.get_agent_config(role.value)
        
        # Create model client if needed
        model_client = None
        if hasattr(agent_class, '_requires_model') and agent_class._requires_model:
            from ..models import create_model_client
            model_client = create_model_client(agent_config.model_config)
        
        # Create agent instance
        agent = agent_class(
            role=role,
            model_client=model_client,
            **agent_config.custom_settings
        )
        
        # Register with our registry
        self.registry.add_agent(agent)
        
        return agent
    
    async def start(self):
        """Start the runtime and all agents."""
        if self._running:
            logger.warning("Runtime is already running")
            return
        
        logger.info("Starting QuantAI runtime...")
        self._startup_time = datetime.utcnow()
        
        # Start AutoGen runtime
        self.runtime.start()
        
        # Setup message subscriptions
        await self._setup_subscriptions()
        
        self._running = True
        logger.info("QuantAI runtime started successfully")
    
    async def _setup_subscriptions(self):
        """Setup message subscriptions between agents."""
        # Create subscriptions based on routing rules
        for message_type, target_roles in self.router._routing_rules.items():
            for role in target_roles:
                await self.runtime.add_subscription(
                    TypeSubscription(message_type.value, role.value)
                )
        
        logger.info("Message subscriptions configured")
    
    async def stop(self):
        """Stop the runtime and all agents."""
        if not self._running:
            logger.warning("Runtime is not running")
            return
        
        logger.info("Stopping QuantAI runtime...")
        self._shutdown_time = datetime.utcnow()
        
        # Shutdown all agents
        for agent in self.registry.get_all_agents():
            await agent.shutdown()
        
        # Stop AutoGen runtime
        await self.runtime.stop_when_idle()
        
        self._running = False
        logger.info("QuantAI runtime stopped")
    
    async def send_message(
        self, 
        message: QuantMessage, 
        target_agent_id: Optional[str] = None
    ):
        """Send a message to agents."""
        if not self._running:
            raise RuntimeError("Runtime is not running")
        
        if target_agent_id:
            # Send to specific agent
            agent_id = AgentId("agent", target_agent_id)
            await self.runtime.send_message(message, agent_id)
        else:
            # Broadcast to appropriate agents based on routing
            target_agents = self.router.get_target_agents(message)
            for agent in target_agents:
                agent_id = AgentId(agent.role.value, agent.agent_id)
                await self.runtime.send_message(message, agent_id)
    
    async def broadcast_message(self, message: QuantMessage):
        """Broadcast a message to all agents."""
        await self.runtime.publish_message(
            message, 
            topic_id=TopicId("broadcast", "all")
        )
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        agent_statuses = {}
        for agent in self.registry.get_all_agents():
            agent_statuses[agent.agent_id] = await agent.get_status()
        
        return {
            "runtime_status": "running" if self._running else "stopped",
            "startup_time": self._startup_time.isoformat() if self._startup_time else None,
            "uptime_seconds": (
                (datetime.utcnow() - self._startup_time).total_seconds() 
                if self._startup_time else 0
            ),
            "total_agents": len(self.registry.get_all_agents()),
            "agents_by_role": {
                role.value: len(self.registry.get_agents_by_role(role))
                for role in AgentRole
            },
            "agent_statuses": agent_statuses,
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        if not self._running:
            return {"status": "unhealthy", "reason": "Runtime not running"}
        
        unhealthy_agents = []
        for agent in self.registry.get_all_agents():
            if not await agent.health_check():
                unhealthy_agents.append(agent.agent_id)
        
        status = "healthy" if not unhealthy_agents else "degraded"
        
        return {
            "status": status,
            "unhealthy_agents": unhealthy_agents,
            "total_agents": len(self.registry.get_all_agents()),
            "healthy_agents": len(self.registry.get_all_agents()) - len(unhealthy_agents),
        }
    
    async def emergency_stop(self):
        """Emergency stop - immediately halt all operations."""
        logger.critical("Emergency stop initiated!")
        
        # Send kill switch message to all agents
        kill_message = ControlMessage(
            message_type=MessageType.KILL_SWITCH,
            sender_id="runtime",
            command="emergency_stop",
            parameters={"reason": "Emergency stop initiated"}
        )
        
        await self.broadcast_message(kill_message)
        await self.stop()


# Global runtime instance
_runtime: Optional[QuantRuntime] = None


def get_runtime() -> QuantRuntime:
    """Get the global runtime instance."""
    global _runtime
    if _runtime is None:
        _runtime = QuantRuntime()
    return _runtime


async def initialize_runtime(config: Optional[QuantConfig] = None) -> QuantRuntime:
    """Initialize and start the global runtime."""
    global _runtime
    _runtime = QuantRuntime(config)
    await _runtime.start()
    return _runtime
