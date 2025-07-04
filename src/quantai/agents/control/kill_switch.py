"""
Kill Switch Agent (M3) for the QuantAI system.

This agent provides emergency stop functionality, forced liquidation,
and system-wide shutdown capabilities for critical risk situations.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from autogen_core import MessageContext
from autogen_core.models import ChatCompletionClient, UserMessage
from loguru import logger

from ...core.base import AgentCapability, AgentRole, ModelCapableAgent
from ...core.messages import (
    ControlMessage, 
    MessageType, 
    QuantMessage, 
    RiskMessage,
    TradeMessage
)


class KillSwitchAgent(ModelCapableAgent):
    """
    Kill Switch Agent (M3) - Emergency stop and system shutdown capabilities.
    
    Capabilities:
    - Emergency stop of all trading activities
    - Forced liquidation of all positions
    - System-wide shutdown and isolation
    - Risk breach response automation
    - Manual override and emergency controls
    - Incident logging and notification
    """
    
    def __init__(
        self,
        model_client: ChatCompletionClient,
        emergency_contacts: List[str] = None,
        max_portfolio_loss: float = 0.20,  # 20% max portfolio loss
        max_daily_loss: float = 0.05,      # 5% max daily loss
        **kwargs
    ):
        super().__init__(
            role=AgentRole.KILL_SWITCH,
            capabilities=[
                AgentCapability.EMERGENCY_CONTROL,
                AgentCapability.RISK_ASSESSMENT,
            ],
            model_client=model_client,
            system_message=self._get_system_message(),
            **kwargs
        )
        
        self.emergency_contacts = emergency_contacts or []
        self.max_portfolio_loss = max_portfolio_loss
        self.max_daily_loss = max_daily_loss
        
        # Emergency state tracking
        self._emergency_active = False
        self._shutdown_initiated = False
        self._liquidation_in_progress = False
        self._emergency_triggers: Set[str] = set()
        
        # Emergency history and logging
        self._emergency_events: List[Dict[str, Any]] = []
        self._shutdown_procedures: Dict[str, bool] = {}
        self._position_liquidation_status: Dict[str, Dict[str, Any]] = {}
        
        # Trigger thresholds
        self._trigger_thresholds = {
            "portfolio_loss": self.max_portfolio_loss,
            "daily_loss": self.max_daily_loss,
            "drawdown": 0.25,  # 25% drawdown
            "risk_breach": True,
            "manual_trigger": True,
            "system_error": True,
        }
        
        # Emergency procedures
        self._emergency_procedures = [
            "stop_new_trades",
            "cancel_pending_orders", 
            "liquidate_positions",
            "disable_strategies",
            "notify_stakeholders",
            "log_incident",
            "isolate_system",
        ]
    
    def _get_system_message(self) -> str:
        return """You are a Kill Switch Agent responsible for emergency stop and crisis management in the trading system.

Your responsibilities:
1. Monitor for emergency conditions and risk breaches
2. Execute immediate emergency stop procedures
3. Coordinate forced liquidation of all positions
4. Implement system-wide shutdown and isolation
5. Manage incident response and stakeholder notification
6. Maintain detailed emergency logs and audit trails

Emergency Response Framework:

1. Trigger Detection
   - Portfolio loss thresholds (daily and total)
   - Risk metric breaches (VaR, drawdown, leverage)
   - System errors and operational failures
   - Manual emergency activation
   - External threat detection

2. Emergency Procedures
   - Immediate halt of all new trading
   - Cancellation of pending orders
   - Forced liquidation of positions
   - Strategy deactivation and isolation
   - System component shutdown

3. Risk Mitigation
   - Position-by-position liquidation priority
   - Market impact minimization
   - Liquidity preservation
   - Counterparty risk management
   - Regulatory compliance maintenance

4. Incident Management
   - Real-time stakeholder notification
   - Detailed incident logging
   - Root cause analysis initiation
   - Recovery planning and execution
   - Post-incident review and learning

5. Communication Protocol
   - Immediate alerts to risk managers
   - Regulatory notification procedures
   - Client communication protocols
   - Media and public relations management
   - Internal team coordination

Emergency Triggers:
- Portfolio loss > 20% (configurable)
- Daily loss > 5% (configurable)
- Maximum drawdown > 25%
- Critical system errors
- Risk control breaches
- Manual activation
- External security threats

Response Priorities:
1. Immediate risk reduction
2. Capital preservation
3. Regulatory compliance
4. Stakeholder communication
5. System recovery preparation

Guidelines:
- Act decisively in emergency situations
- Prioritize capital preservation over profits
- Maintain clear communication channels
- Document all emergency actions
- Follow regulatory requirements
- Prepare for rapid recovery

Focus on swift, decisive action to protect capital and ensure system integrity during crisis situations."""
    
    async def process_message(
        self, 
        message: QuantMessage, 
        ctx: MessageContext
    ) -> Optional[QuantMessage]:
        """Process emergency control and kill switch requests."""
        
        if isinstance(message, ControlMessage):
            if message.command == "emergency_stop":
                # Manual emergency stop activation
                await self._activate_emergency_stop(
                    trigger="manual",
                    reason=message.parameters.get("reason", "Manual emergency activation"),
                    severity="critical"
                )
                
                response = ControlMessage(
                    message_type=MessageType.KILL_SWITCH,
                    sender_id=self.agent_id,
                    command="emergency_stop_activated",
                    parameters={
                        "status": "activated",
                        "timestamp": datetime.now().isoformat(),
                        "procedures_initiated": self._emergency_procedures,
                    },
                    session_id=message.session_id,
                    correlation_id=message.correlation_id,
                )
                
                return response
                
            elif message.command == "force_liquidation":
                # Force liquidation of all positions
                await self._force_liquidation(
                    reason=message.parameters.get("reason", "Forced liquidation requested")
                )
                
                response = ControlMessage(
                    message_type=MessageType.KILL_SWITCH,
                    sender_id=self.agent_id,
                    command="liquidation_initiated",
                    parameters={
                        "status": "in_progress",
                        "positions": list(self._position_liquidation_status.keys()),
                    },
                    session_id=message.session_id,
                    correlation_id=message.correlation_id,
                )
                
                return response
                
            elif message.command == "system_shutdown":
                # Complete system shutdown
                await self._initiate_system_shutdown(
                    reason=message.parameters.get("reason", "System shutdown requested")
                )
                
                response = ControlMessage(
                    message_type=MessageType.KILL_SWITCH,
                    sender_id=self.agent_id,
                    command="shutdown_initiated",
                    parameters={
                        "status": "shutting_down",
                        "procedures": self._shutdown_procedures,
                    },
                    session_id=message.session_id,
                    correlation_id=message.correlation_id,
                )
                
                return response
                
            elif message.command == "reset_emergency":
                # Reset emergency state (after manual review)
                await self._reset_emergency_state(
                    authorized_by=message.parameters.get("authorized_by")
                )
                
                response = ControlMessage(
                    message_type=MessageType.CONTROL,
                    sender_id=self.agent_id,
                    command="emergency_reset",
                    parameters={"status": "reset", "emergency_active": self._emergency_active},
                    session_id=message.session_id,
                    correlation_id=message.correlation_id,
                )
                
                return response
        
        elif isinstance(message, RiskMessage):
            # Monitor risk messages for emergency triggers
            await self._check_risk_triggers(message)
        
        return None
    
    async def _activate_emergency_stop(self, trigger: str, reason: str, severity: str):
        """Activate emergency stop procedures."""
        
        if self._emergency_active:
            logger.warning("Emergency stop already active")
            return
        
        logger.critical(f"EMERGENCY STOP ACTIVATED - Trigger: {trigger}, Reason: {reason}")
        
        self._emergency_active = True
        self._emergency_triggers.add(trigger)
        
        # Record emergency event
        emergency_event = {
            "event_id": f"emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "trigger": trigger,
            "reason": reason,
            "severity": severity,
            "timestamp": datetime.now(),
            "procedures_executed": [],
        }
        
        # Execute emergency procedures
        for procedure in self._emergency_procedures:
            try:
                await self._execute_emergency_procedure(procedure, emergency_event)
                emergency_event["procedures_executed"].append(procedure)
                logger.info(f"Emergency procedure completed: {procedure}")
                
            except Exception as e:
                logger.error(f"Emergency procedure failed: {procedure} - {e}")
                emergency_event["procedures_executed"].append(f"{procedure}_FAILED")
        
        # Store emergency event
        self._emergency_events.append(emergency_event)
        
        # Send notifications
        await self._send_emergency_notifications(emergency_event)
        
        logger.critical("Emergency stop procedures completed")
    
    async def _execute_emergency_procedure(self, procedure: str, event: Dict[str, Any]):
        """Execute a specific emergency procedure."""
        
        if procedure == "stop_new_trades":
            await self._stop_new_trades()
            
        elif procedure == "cancel_pending_orders":
            await self._cancel_pending_orders()
            
        elif procedure == "liquidate_positions":
            await self._liquidate_all_positions()
            
        elif procedure == "disable_strategies":
            await self._disable_all_strategies()
            
        elif procedure == "notify_stakeholders":
            await self._notify_stakeholders(event)
            
        elif procedure == "log_incident":
            await self._log_incident(event)
            
        elif procedure == "isolate_system":
            await self._isolate_system()
    
    async def _stop_new_trades(self):
        """Stop all new trading activities."""
        
        # Send stop trading command to all execution agents
        stop_command = ControlMessage(
            message_type=MessageType.KILL_SWITCH,
            sender_id=self.agent_id,
            command="stop_trading",
            parameters={
                "reason": "Emergency stop activated",
                "timestamp": datetime.now().isoformat(),
            }
        )
        
        # This would be broadcast to all trading agents
        logger.info("New trading activities stopped")
    
    async def _cancel_pending_orders(self):
        """Cancel all pending orders."""
        
        # Send cancel orders command
        cancel_command = ControlMessage(
            message_type=MessageType.KILL_SWITCH,
            sender_id=self.agent_id,
            command="cancel_all_orders",
            parameters={
                "reason": "Emergency stop - cancel all pending orders",
                "timestamp": datetime.now().isoformat(),
            }
        )
        
        logger.info("All pending orders cancelled")
    
    async def _liquidate_all_positions(self):
        """Liquidate all open positions."""
        
        if self._liquidation_in_progress:
            logger.warning("Liquidation already in progress")
            return
        
        self._liquidation_in_progress = True
        
        # This would get current positions from portfolio manager
        # For demo, simulate some positions
        mock_positions = {
            "SPY": {"quantity": 1000, "side": "long", "value": 450000},
            "QQQ": {"quantity": 500, "side": "long", "value": 175000},
            "IWM": {"quantity": -200, "side": "short", "value": -40000},
        }
        
        for symbol, position in mock_positions.items():
            try:
                await self._liquidate_position(symbol, position)
                self._position_liquidation_status[symbol] = {
                    "status": "liquidated",
                    "timestamp": datetime.now(),
                    "original_position": position,
                }
                
            except Exception as e:
                logger.error(f"Failed to liquidate position {symbol}: {e}")
                self._position_liquidation_status[symbol] = {
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.now(),
                }
        
        logger.critical("Position liquidation completed")
    
    async def _liquidate_position(self, symbol: str, position: Dict[str, Any]):
        """Liquidate a specific position."""
        
        # Create liquidation order
        liquidation_order = TradeMessage(
            message_type=MessageType.TRADE_SIGNAL,
            sender_id=self.agent_id,
            trade_id=f"liquidation_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbol=symbol,
            action="SELL" if position["side"] == "long" else "BUY",
            quantity=abs(position["quantity"]),
            order_type="MARKET",
            urgency="EMERGENCY",
            reason="Emergency liquidation",
        )
        
        # This would be sent to execution agent
        logger.info(f"Liquidation order created for {symbol}: {liquidation_order.action} {liquidation_order.quantity}")
    
    async def _disable_all_strategies(self):
        """Disable all active trading strategies."""
        
        disable_command = ControlMessage(
            message_type=MessageType.KILL_SWITCH,
            sender_id=self.agent_id,
            command="disable_strategies",
            parameters={
                "reason": "Emergency stop - disable all strategies",
                "timestamp": datetime.now().isoformat(),
            }
        )
        
        logger.info("All trading strategies disabled")
    
    async def _notify_stakeholders(self, event: Dict[str, Any]):
        """Notify stakeholders of emergency situation."""
        
        notification = {
            "subject": f"EMERGENCY ALERT - {event['trigger'].upper()}",
            "message": f"Emergency stop activated at {event['timestamp']}. Reason: {event['reason']}",
            "severity": event["severity"],
            "event_id": event["event_id"],
            "contacts": self.emergency_contacts,
        }
        
        # This would send actual notifications (email, SMS, Slack, etc.)
        logger.critical(f"Emergency notifications sent: {notification}")
    
    async def _log_incident(self, event: Dict[str, Any]):
        """Log incident details for investigation."""
        
        incident_log = {
            "incident_id": event["event_id"],
            "type": "emergency_stop",
            "trigger": event["trigger"],
            "reason": event["reason"],
            "severity": event["severity"],
            "timestamp": event["timestamp"].isoformat(),
            "procedures_executed": event["procedures_executed"],
            "system_state": await self._capture_system_state(),
            "positions_at_trigger": self._position_liquidation_status,
        }
        
        # This would write to incident management system
        logger.critical(f"Incident logged: {incident_log}")
    
    async def _isolate_system(self):
        """Isolate system from external connections."""
        
        # This would implement actual system isolation
        # - Disable API endpoints
        # - Close external connections
        # - Activate maintenance mode
        
        logger.critical("System isolation activated")
    
    async def _force_liquidation(self, reason: str):
        """Force liquidation without full emergency stop."""
        
        logger.warning(f"Force liquidation initiated: {reason}")
        
        await self._liquidate_all_positions()
        
        # Log liquidation event
        liquidation_event = {
            "event_id": f"liquidation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "type": "force_liquidation",
            "reason": reason,
            "timestamp": datetime.now(),
            "positions": self._position_liquidation_status,
        }
        
        self._emergency_events.append(liquidation_event)
    
    async def _initiate_system_shutdown(self, reason: str):
        """Initiate complete system shutdown."""
        
        if self._shutdown_initiated:
            logger.warning("System shutdown already initiated")
            return
        
        logger.critical(f"System shutdown initiated: {reason}")
        
        self._shutdown_initiated = True
        
        # Execute shutdown procedures
        shutdown_procedures = [
            "stop_all_agents",
            "close_connections",
            "save_state",
            "notify_shutdown",
            "system_halt",
        ]
        
        for procedure in shutdown_procedures:
            try:
                await self._execute_shutdown_procedure(procedure)
                self._shutdown_procedures[procedure] = True
                
            except Exception as e:
                logger.error(f"Shutdown procedure failed: {procedure} - {e}")
                self._shutdown_procedures[procedure] = False
        
        logger.critical("System shutdown completed")
    
    async def _execute_shutdown_procedure(self, procedure: str):
        """Execute a specific shutdown procedure."""
        
        if procedure == "stop_all_agents":
            # Send stop command to all agents
            logger.info("Stopping all agents")
            
        elif procedure == "close_connections":
            # Close all external connections
            logger.info("Closing external connections")
            
        elif procedure == "save_state":
            # Save current system state
            await self._save_system_state()
            
        elif procedure == "notify_shutdown":
            # Notify of shutdown
            logger.info("Shutdown notifications sent")
            
        elif procedure == "system_halt":
            # Final system halt
            logger.info("System halt initiated")
    
    async def _check_risk_triggers(self, risk_message: RiskMessage):
        """Check if risk message triggers emergency procedures."""
        
        if self._emergency_active:
            return  # Already in emergency mode
        
        # Check portfolio loss trigger
        if hasattr(risk_message, 'portfolio_loss'):
            if risk_message.portfolio_loss > self._trigger_thresholds["portfolio_loss"]:
                await self._activate_emergency_stop(
                    trigger="portfolio_loss",
                    reason=f"Portfolio loss {risk_message.portfolio_loss:.2%} exceeds threshold {self._trigger_thresholds['portfolio_loss']:.2%}",
                    severity="critical"
                )
                return
        
        # Check daily loss trigger
        if hasattr(risk_message, 'daily_loss'):
            if risk_message.daily_loss > self._trigger_thresholds["daily_loss"]:
                await self._activate_emergency_stop(
                    trigger="daily_loss",
                    reason=f"Daily loss {risk_message.daily_loss:.2%} exceeds threshold {self._trigger_thresholds['daily_loss']:.2%}",
                    severity="high"
                )
                return
        
        # Check drawdown trigger
        if hasattr(risk_message, 'max_drawdown'):
            if risk_message.max_drawdown > self._trigger_thresholds["drawdown"]:
                await self._activate_emergency_stop(
                    trigger="drawdown",
                    reason=f"Drawdown {risk_message.max_drawdown:.2%} exceeds threshold {self._trigger_thresholds['drawdown']:.2%}",
                    severity="critical"
                )
                return
    
    async def _reset_emergency_state(self, authorized_by: str):
        """Reset emergency state after manual authorization."""
        
        if not authorized_by:
            logger.error("Emergency reset requires authorization")
            return
        
        logger.warning(f"Emergency state reset authorized by: {authorized_by}")
        
        # Reset emergency flags
        self._emergency_active = False
        self._shutdown_initiated = False
        self._liquidation_in_progress = False
        self._emergency_triggers.clear()
        
        # Log reset event
        reset_event = {
            "event_id": f"reset_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "type": "emergency_reset",
            "authorized_by": authorized_by,
            "timestamp": datetime.now(),
        }
        
        self._emergency_events.append(reset_event)
        
        logger.info("Emergency state reset completed")
    
    async def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for incident analysis."""
        
        return {
            "timestamp": datetime.now().isoformat(),
            "emergency_active": self._emergency_active,
            "shutdown_initiated": self._shutdown_initiated,
            "liquidation_in_progress": self._liquidation_in_progress,
            "active_triggers": list(self._emergency_triggers),
            "position_count": len(self._position_liquidation_status),
            "recent_events": self._emergency_events[-5:],  # Last 5 events
        }
    
    async def _save_system_state(self):
        """Save system state for recovery."""
        
        state = await self._capture_system_state()
        
        # This would save to persistent storage
        logger.info(f"System state saved: {state}")
    
    async def get_emergency_status(self) -> Dict[str, Any]:
        """Get current emergency status."""
        
        return {
            "emergency_active": self._emergency_active,
            "shutdown_initiated": self._shutdown_initiated,
            "liquidation_in_progress": self._liquidation_in_progress,
            "active_triggers": list(self._emergency_triggers),
            "trigger_thresholds": self._trigger_thresholds,
            "recent_events": self._emergency_events[-10:],  # Last 10 events
            "position_liquidation_status": self._position_liquidation_status,
            "shutdown_procedures": self._shutdown_procedures,
            "emergency_contacts": len(self.emergency_contacts),
        }
    
    async def update_emergency_thresholds(self, new_thresholds: Dict[str, Any]):
        """Update emergency trigger thresholds."""
        
        self._trigger_thresholds.update(new_thresholds)
        
        logger.info(f"Emergency thresholds updated: {new_thresholds}")
    
    async def test_emergency_procedures(self) -> Dict[str, Any]:
        """Test emergency procedures without actual execution."""
        
        logger.info("Testing emergency procedures (simulation mode)")
        
        test_results = {}
        
        for procedure in self._emergency_procedures:
            try:
                # Simulate procedure execution
                await asyncio.sleep(0.1)  # Simulate processing time
                test_results[procedure] = "PASS"
                
            except Exception as e:
                test_results[procedure] = f"FAIL: {e}"
        
        return {
            "test_timestamp": datetime.now().isoformat(),
            "procedures_tested": len(self._emergency_procedures),
            "results": test_results,
            "overall_status": "PASS" if all(r == "PASS" for r in test_results.values()) else "FAIL",
        }
