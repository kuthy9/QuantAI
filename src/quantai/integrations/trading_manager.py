"""
Enhanced Trading Manager for QuantAI system.

Provides comprehensive trading functionality with paper/live mode support,
risk management, order management, and production-grade safety features.
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass

from loguru import logger
from pydantic import BaseModel, Field

from .ibkr import IBKRClient, IBKRConfig, TradingMode, ConnectionStatus
from ..core.messages import TradeMessage, MessageType


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    ERROR = "error"


@dataclass
class TradingSession:
    """Trading session information."""
    session_id: str
    start_time: datetime
    trading_mode: TradingMode
    total_orders: int = 0
    successful_orders: int = 0
    failed_orders: int = 0
    total_volume: float = 0.0
    total_pnl: float = 0.0


class TradingManagerConfig(BaseModel):
    """Configuration for Trading Manager."""
    # IBKR Configuration
    ibkr_config: IBKRConfig = Field(default_factory=IBKRConfig.from_env)
    
    # Trading session settings
    auto_connect: bool = Field(default=True, description="Auto-connect on startup")
    session_timeout: int = Field(default=3600, description="Session timeout in seconds")
    
    # Order management
    order_timeout: int = Field(default=300, description="Order timeout in seconds")
    max_pending_orders: int = Field(default=50, description="Maximum pending orders")
    
    # Risk management
    enable_position_tracking: bool = Field(default=True, description="Enable position tracking")
    enable_pnl_tracking: bool = Field(default=True, description="Enable P&L tracking")
    
    # Notifications
    enable_notifications: bool = Field(default=True, description="Enable trade notifications")
    notification_webhook: Optional[str] = Field(default=None, description="Webhook for notifications")


class TradingManager:
    """
    Enhanced Trading Manager with comprehensive order and risk management.
    
    Provides a high-level interface for trading operations with built-in
    safety features, monitoring, and production-grade error handling.
    """
    
    def __init__(self, config: TradingManagerConfig):
        self.config = config
        self.ibkr_client = IBKRClient(config.ibkr_config)
        
        # Session management
        self.current_session: Optional[TradingSession] = None
        self.session_history: List[TradingSession] = []
        
        # Order tracking
        self.active_orders: Dict[str, Dict[str, Any]] = {}
        self.order_history: List[Dict[str, Any]] = []
        
        # Position tracking
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.position_history: List[Dict[str, Any]] = []
        
        # Event callbacks
        self.on_order_filled: Optional[Callable] = None
        self.on_position_changed: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Performance tracking
        self.performance_metrics = {
            "total_trades": 0,
            "successful_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "average_trade_size": 0.0
        }
        
        logger.info(f"Trading Manager initialized in {config.ibkr_config.trading_mode.value.upper()} mode")
    
    async def start_session(self) -> bool:
        """Start a new trading session."""
        try:
            # Create new session
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.current_session = TradingSession(
                session_id=session_id,
                start_time=datetime.now(),
                trading_mode=self.config.ibkr_config.trading_mode
            )
            
            logger.info(f"Starting trading session: {session_id}")
            
            # Connect to IBKR if auto-connect is enabled
            if self.config.auto_connect:
                success = await self.ibkr_client.connect()
                if not success:
                    logger.error("Failed to connect to IBKR")
                    return False
            
            # Setup event handlers
            self._setup_event_handlers()
            
            logger.info("✅ Trading session started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start trading session: {e}")
            return False
    
    async def stop_session(self):
        """Stop the current trading session."""
        if self.current_session:
            try:
                # Cancel pending orders if configured
                await self._cancel_pending_orders()
                
                # Disconnect from IBKR
                await self.ibkr_client.disconnect()
                
                # Archive session
                self.session_history.append(self.current_session)
                
                logger.info(f"Trading session {self.current_session.session_id} stopped")
                self.current_session = None
                
            except Exception as e:
                logger.error(f"Error stopping trading session: {e}")
    
    async def place_order(self, trade_message: TradeMessage) -> Dict[str, Any]:
        """Place an order with comprehensive tracking and validation."""
        
        if not self.current_session:
            return {
                "success": False,
                "error": "No active trading session",
                "trade_id": trade_message.trade_id
            }
        
        # Check pending order limits
        if len(self.active_orders) >= self.config.max_pending_orders:
            return {
                "success": False,
                "error": f"Maximum pending orders reached ({self.config.max_pending_orders})",
                "trade_id": trade_message.trade_id
            }
        
        try:
            # Place order through IBKR
            result = await self.ibkr_client.place_order(trade_message)
            
            if result["success"]:
                # Track order
                order_info = {
                    "trade_id": trade_message.trade_id,
                    "order_id": result.get("order_id"),
                    "symbol": trade_message.symbol,
                    "action": trade_message.action,
                    "quantity": trade_message.quantity,
                    "order_type": trade_message.order_type,
                    "price": trade_message.price,
                    "status": OrderStatus.SUBMITTED,
                    "submit_time": datetime.now(),
                    "trading_mode": self.config.ibkr_config.trading_mode.value
                }
                
                self.active_orders[trade_message.trade_id] = order_info
                self.current_session.total_orders += 1
                
                # Send notification
                if self.config.enable_notifications:
                    await self._send_notification("order_placed", order_info)
                
                logger.info(f"Order placed successfully: {trade_message.trade_id}")
                
            else:
                self.current_session.failed_orders += 1
                logger.error(f"Order placement failed: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            self.current_session.failed_orders += 1
            
            return {
                "success": False,
                "error": str(e),
                "trade_id": trade_message.trade_id
            }
    
    async def cancel_order(self, trade_id: str) -> bool:
        """Cancel an active order."""
        if trade_id not in self.active_orders:
            logger.warning(f"Order {trade_id} not found in active orders")
            return False
        
        order_info = self.active_orders[trade_id]
        order_id = order_info.get("order_id")
        
        if order_id:
            success = await self.ibkr_client.cancel_order(order_id)
            if success:
                order_info["status"] = OrderStatus.CANCELLED
                order_info["cancel_time"] = datetime.now()
                
                # Move to history
                self.order_history.append(order_info)
                del self.active_orders[trade_id]
                
                logger.info(f"Order {trade_id} cancelled successfully")
                return True
        
        return False
    
    def _setup_event_handlers(self):
        """Setup event handlers for IBKR client."""
        self.ibkr_client.on_order_filled = self._handle_order_filled
        self.ibkr_client.on_connection_lost = self._handle_connection_lost
        self.ibkr_client.on_error = self._handle_error
    
    def _handle_order_filled(self, trade, fill):
        """Handle order fill events."""
        # Find corresponding order
        for trade_id, order_info in self.active_orders.items():
            if order_info.get("order_id") == trade.order.orderId:
                order_info["status"] = OrderStatus.FILLED
                order_info["fill_time"] = datetime.now()
                order_info["fill_price"] = fill.execution.price
                order_info["fill_quantity"] = fill.execution.shares
                
                # Update session metrics
                self.current_session.successful_orders += 1
                self.current_session.total_volume += fill.execution.shares * fill.execution.price
                
                # Move to history
                self.order_history.append(order_info)
                del self.active_orders[trade_id]
                
                # Callback
                if self.on_order_filled:
                    self.on_order_filled(order_info)
                
                logger.info(f"Order {trade_id} filled at ${fill.execution.price}")
                break
    
    def _handle_connection_lost(self):
        """Handle connection loss."""
        logger.warning("IBKR connection lost - monitoring for reconnection")
        
        if self.on_error:
            self.on_error("connection_lost", "IBKR connection lost")
    
    def _handle_error(self, error_code: int, error_string: str):
        """Handle IBKR errors."""
        logger.error(f"IBKR Error {error_code}: {error_string}")
        
        if self.on_error:
            self.on_error(error_code, error_string)
    
    async def _cancel_pending_orders(self):
        """Cancel all pending orders."""
        pending_orders = list(self.active_orders.keys())
        for trade_id in pending_orders:
            await self.cancel_order(trade_id)
    
    async def _send_notification(self, event_type: str, data: Dict[str, Any]):
        """Send notification about trading events."""
        # Implement notification logic (webhook, email, etc.)
        logger.info(f"Notification: {event_type} - {data}")
    
    def get_session_status(self) -> Dict[str, Any]:
        """Get current session status."""
        if not self.current_session:
            return {"active": False}
        
        return {
            "active": True,
            "session_id": self.current_session.session_id,
            "start_time": self.current_session.start_time.isoformat(),
            "trading_mode": self.current_session.trading_mode.value,
            "total_orders": self.current_session.total_orders,
            "successful_orders": self.current_session.successful_orders,
            "failed_orders": self.current_session.failed_orders,
            "active_orders_count": len(self.active_orders),
            "connection_status": self.ibkr_client.get_connection_status().value,
            "daily_trade_count": self.ibkr_client.get_daily_trade_count()
        }
    
    def get_active_orders(self) -> Dict[str, Dict[str, Any]]:
        """Get all active orders."""
        return self.active_orders.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            "session_performance": self.get_session_status(),
            "overall_performance": self.performance_metrics.copy(),
            "ibkr_health": asyncio.create_task(self.ibkr_client.health_check())
        }
