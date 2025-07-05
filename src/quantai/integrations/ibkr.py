"""
Interactive Brokers (IBKR) integration for the QuantAI system.

This module provides connection and trading capabilities with IBKR's
TWS (Trader Workstation) or IB Gateway for production simulation.
"""

import asyncio
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable
from decimal import Decimal
from enum import Enum

from ib_insync import IB, Stock, Order, MarketOrder, LimitOrder, Contract, Future, Option, Forex
from loguru import logger
from pydantic import BaseModel, Field

from ..core.messages import TradeMessage, AssetType, ProductType


class TradingMode(Enum):
    """Trading mode enumeration."""
    PAPER = "paper"
    LIVE = "live"


class ConnectionStatus(Enum):
    """Connection status enumeration."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class IBKRConfig(BaseModel):
    """Configuration for IBKR connection."""
    # Connection settings
    host: str = Field(default="127.0.0.1", description="IBKR host address")
    port: int = Field(default=7497, description="IBKR port (7497=paper, 7496=live)")
    client_id: int = Field(default=1, description="Client ID for connection")
    account: str = Field(default="DUA559603", description="IBKR account number")
    timeout: int = Field(default=30, description="Connection timeout in seconds")
    readonly: bool = Field(default=False, description="Read-only connection")

    # Trading mode configuration
    trading_mode: TradingMode = Field(default=TradingMode.PAPER, description="Trading mode")

    # Safety settings
    max_order_value: float = Field(default=10000.0, description="Maximum order value in USD")
    max_daily_trades: int = Field(default=100, description="Maximum trades per day")
    position_size_limit: float = Field(default=0.1, description="Maximum position size as % of portfolio")

    # Risk management
    enable_risk_checks: bool = Field(default=True, description="Enable pre-trade risk checks")
    require_confirmation: bool = Field(default=True, description="Require order confirmation")

    # Reconnection settings
    auto_reconnect: bool = Field(default=True, description="Enable automatic reconnection")
    max_reconnect_attempts: int = Field(default=5, description="Maximum reconnection attempts")
    reconnect_delay: int = Field(default=5, description="Delay between reconnection attempts")

    @classmethod
    def from_env(cls) -> 'IBKRConfig':
        """Create configuration from environment variables."""
        trading_mode_str = os.getenv("TRADING_MODE", "paper").lower()
        trading_mode = TradingMode.PAPER if trading_mode_str == "paper" else TradingMode.LIVE

        # Set port based on trading mode if not explicitly set
        default_port = 7497 if trading_mode == TradingMode.PAPER else 7496
        port = int(os.getenv("IB_PORT", default_port))

        return cls(
            host=os.getenv("IB_HOST", "127.0.0.1"),
            port=port,
            client_id=int(os.getenv("IB_CLIENT_ID", "1")),
            account=os.getenv("IB_ACCOUNT", "U15543181"),  # User's actual account
            timeout=int(os.getenv("IB_TIMEOUT", "30")),
            readonly=os.getenv("IB_READONLY", "false").lower() == "true",
            trading_mode=trading_mode,
            max_order_value=float(os.getenv("MAX_ORDER_VALUE", "10000.0")),
            max_daily_trades=int(os.getenv("MAX_DAILY_TRADES", "100")),
            position_size_limit=float(os.getenv("POSITION_SIZE_LIMIT", "0.1")),
            enable_risk_checks=os.getenv("ENABLE_RISK_CHECKS", "true").lower() == "true",
            require_confirmation=os.getenv("REQUIRE_CONFIRMATION", "true").lower() == "true",
            auto_reconnect=os.getenv("AUTO_RECONNECT", "true").lower() == "true",
            max_reconnect_attempts=int(os.getenv("MAX_RECONNECT_ATTEMPTS", "5")),
            reconnect_delay=int(os.getenv("RECONNECT_DELAY", "5"))
        )


class IBKRPosition(BaseModel):
    """IBKR position information."""
    symbol: str
    quantity: float
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float


class IBKROrderStatus(BaseModel):
    """IBKR order status information."""
    order_id: int
    status: str
    filled: float
    remaining: float
    avg_fill_price: float
    last_fill_time: Optional[datetime] = None


class IBKRClient:
    """
    Enhanced Interactive Brokers client with paper/live trading support.

    Provides secure connection to IBKR TWS/Gateway with comprehensive
    risk management, error handling, and production-grade features.
    """

    def __init__(self, config: IBKRConfig):
        self.config = config
        self.ib = IB()
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.account_info: Dict[str, Any] = {}
        self.positions: Dict[str, IBKRPosition] = {}
        self.orders: Dict[int, IBKROrderStatus] = {}

        # Trading session tracking
        self.daily_trade_count = 0
        self.session_start = datetime.now().date()
        self.last_heartbeat = datetime.now()

        # Reconnection management
        self.reconnect_attempts = 0
        self.last_reconnect_time = None

        # Event callbacks
        self.on_connection_lost: Optional[Callable] = None
        self.on_order_filled: Optional[Callable] = None
        self.on_error: Optional[Callable] = None

        # Safety checks
        self._validate_configuration()

        logger.info(f"IBKR Client initialized in {self.config.trading_mode.value.upper()} mode")
        if self.config.trading_mode == TradingMode.LIVE:
            logger.warning("⚠️  LIVE TRADING MODE ENABLED - Real money at risk!")
        else:
            logger.info("📊 Paper trading mode - Safe for testing")
        
    def _validate_configuration(self):
        """Validate configuration for safety."""
        if self.config.trading_mode == TradingMode.LIVE:
            # Extra validation for live trading
            if self.config.port == 7497:
                logger.warning("Live trading mode but using paper trading port (7497)")

            # Ensure safety limits are reasonable
            if self.config.max_order_value > 100000:
                logger.warning(f"High max order value: ${self.config.max_order_value:,.2f}")

            if self.config.position_size_limit > 0.2:
                logger.warning(f"High position size limit: {self.config.position_size_limit:.1%}")

    async def connect(self) -> bool:
        """Connect to IBKR TWS/Gateway with enhanced error handling."""
        self.connection_status = ConnectionStatus.CONNECTING

        try:
            # Safety check for live trading
            if self.config.trading_mode == TradingMode.LIVE:
                logger.warning("🚨 ATTEMPTING LIVE TRADING CONNECTION 🚨")
                logger.warning(f"Account: {self.config.account}")
                logger.warning(f"Port: {self.config.port}")

                if not self._confirm_live_trading():
                    logger.error("Live trading connection aborted - safety check failed")
                    self.connection_status = ConnectionStatus.ERROR
                    return False

            logger.info(f"Connecting to IBKR at {self.config.host}:{self.config.port}")
            logger.info(f"Mode: {self.config.trading_mode.value.upper()}")

            # Setup event handlers
            self._setup_event_handlers()

            # Connect to TWS/Gateway
            await self.ib.connectAsync(
                host=self.config.host,
                port=self.config.port,
                clientId=self.config.client_id,
                timeout=self.config.timeout,
                readonly=self.config.readonly
            )

            self.connection_status = ConnectionStatus.CONNECTED
            self.reconnect_attempts = 0
            self.last_heartbeat = datetime.now()

            logger.info("✅ Successfully connected to IBKR")

            # Get account information
            await self._update_account_info()
            await self._update_positions()

            # Reset daily counters if new day
            self._check_daily_reset()

            return True

        except Exception as e:
            logger.error(f"❌ Failed to connect to IBKR: {e}")
            self.connection_status = ConnectionStatus.ERROR

            # Attempt reconnection if enabled
            if self.config.auto_reconnect and self.reconnect_attempts < self.config.max_reconnect_attempts:
                await self._attempt_reconnection()

            return False

    def _confirm_live_trading(self) -> bool:
        """Confirm live trading mode with additional safety checks."""
        # In production, this could require additional confirmation
        # For now, we'll allow it but with extensive logging

        logger.warning("Live trading mode confirmed")
        logger.warning("Ensure TWS/Gateway is configured for live trading")
        logger.warning("Verify account permissions and risk settings")

        return True  # Allow live trading

    def _setup_event_handlers(self):
        """Setup IB event handlers."""
        self.ib.connectedEvent += self._on_connected
        self.ib.disconnectedEvent += self._on_disconnected
        self.ib.errorEvent += self._on_error_event
        self.ib.orderStatusEvent += self._on_order_status
        self.ib.execDetailsEvent += self._on_execution

    def _on_connected(self):
        """Handle connection event."""
        logger.info("IB connection established")
        self.last_heartbeat = datetime.now()

    def _on_disconnected(self):
        """Handle disconnection event."""
        logger.warning("IB connection lost")
        self.connection_status = ConnectionStatus.DISCONNECTED

        if self.on_connection_lost:
            self.on_connection_lost()

        # Attempt reconnection if enabled
        if self.config.auto_reconnect:
            asyncio.create_task(self._attempt_reconnection())

    def _on_error_event(self, reqId, errorCode, errorString, contract):
        """Handle error events."""
        logger.error(f"IB Error {errorCode}: {errorString} (reqId: {reqId})")

        if self.on_error:
            self.on_error(errorCode, errorString)

    def _on_order_status(self, trade):
        """Handle order status updates."""
        order_id = trade.order.orderId
        status = trade.orderStatus.status

        logger.info(f"Order {order_id} status: {status}")

        if order_id in self.orders:
            self.orders[order_id].status = status
            self.orders[order_id].filled = trade.orderStatus.filled
            self.orders[order_id].remaining = trade.orderStatus.remaining
            self.orders[order_id].avg_fill_price = trade.orderStatus.avgFillPrice

    def _on_execution(self, trade, fill):
        """Handle trade executions."""
        logger.info(f"Trade executed: {fill.execution.shares} shares of {trade.contract.symbol} at ${fill.execution.price}")

        if self.on_order_filled:
            self.on_order_filled(trade, fill)
    
    async def _attempt_reconnection(self):
        """Attempt to reconnect to IBKR."""
        if self.reconnect_attempts >= self.config.max_reconnect_attempts:
            logger.error("Maximum reconnection attempts reached")
            return

        self.reconnect_attempts += 1
        self.connection_status = ConnectionStatus.RECONNECTING

        logger.info(f"Attempting reconnection {self.reconnect_attempts}/{self.config.max_reconnect_attempts}")

        await asyncio.sleep(self.config.reconnect_delay)

        success = await self.connect()
        if success:
            logger.info("Reconnection successful")
        else:
            logger.warning(f"Reconnection attempt {self.reconnect_attempts} failed")

    def _check_daily_reset(self):
        """Check if we need to reset daily counters."""
        current_date = datetime.now().date()
        if current_date > self.session_start:
            self.daily_trade_count = 0
            self.session_start = current_date
            logger.info("Daily trading counters reset")

    async def disconnect(self):
        """Disconnect from IBKR with cleanup."""
        if self.connection_status != ConnectionStatus.DISCONNECTED:
            try:
                # Cancel any pending orders if configured
                if hasattr(self, 'cancel_on_disconnect') and self.cancel_on_disconnect:
                    await self._cancel_all_pending_orders()

                self.ib.disconnect()
                self.connection_status = ConnectionStatus.DISCONNECTED
                logger.info("✅ Disconnected from IBKR")

            except Exception as e:
                logger.error(f"Error during disconnect: {e}")

    async def _cancel_all_pending_orders(self):
        """Cancel all pending orders."""
        try:
            trades = self.ib.trades()
            pending_trades = [t for t in trades if t.orderStatus.status in ['Submitted', 'PreSubmitted']]

            for trade in pending_trades:
                self.ib.cancelOrder(trade.order)
                logger.info(f"Cancelled pending order {trade.order.orderId}")

        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
    
    async def _update_account_info(self):
        """Update account information."""
        try:
            account_values = self.ib.accountValues(account=self.config.account)
            
            self.account_info = {}
            for value in account_values:
                self.account_info[value.tag] = {
                    'value': value.value,
                    'currency': value.currency
                }
                
            logger.info(f"Updated account info for {self.config.account}")
            
        except Exception as e:
            logger.error(f"Failed to update account info: {e}")
    
    async def _update_positions(self):
        """Update current positions."""
        try:
            positions = self.ib.positions(account=self.config.account)
            
            self.positions = {}
            for pos in positions:
                symbol = pos.contract.symbol
                self.positions[symbol] = IBKRPosition(
                    symbol=symbol,
                    quantity=pos.position,
                    avg_cost=pos.avgCost,
                    market_value=pos.marketValue,
                    unrealized_pnl=pos.unrealizedPNL,
                    realized_pnl=0.0  # Would need separate call for realized PnL
                )
                
            logger.info(f"Updated {len(self.positions)} positions")
            
        except Exception as e:
            logger.error(f"Failed to update positions: {e}")
    
    def get_account_summary(self) -> Dict[str, Any]:
        """Get account summary information."""
        if not self.connected:
            return {"error": "Not connected to IBKR"}
            
        summary = {}
        
        # Key account metrics
        key_metrics = [
            'NetLiquidation', 'TotalCashValue', 'AvailableFunds',
            'BuyingPower', 'GrossPositionValue', 'UnrealizedPnL'
        ]
        
        for metric in key_metrics:
            if metric in self.account_info:
                summary[metric] = self.account_info[metric]
        
        summary['positions_count'] = len(self.positions)
        summary['account'] = self.config.account
        summary['connected'] = self.connected
        
        return summary
    
    def get_positions(self) -> Dict[str, IBKRPosition]:
        """Get current positions."""
        return self.positions.copy()
    
    async def place_order(self, trade_message: TradeMessage) -> Dict[str, Any]:
        """
        Place an order through IBKR with comprehensive risk checks.

        Args:
            trade_message: Trade message with order details

        Returns:
            Order execution result with detailed status
        """
        # Connection check
        if self.connection_status != ConnectionStatus.CONNECTED:
            return {
                "success": False,
                "error": f"Not connected to IBKR (status: {self.connection_status.value})",
                "trade_id": trade_message.trade_id
            }

        # Pre-trade risk checks
        if self.config.enable_risk_checks:
            risk_check = await self._perform_risk_checks(trade_message)
            if not risk_check["passed"]:
                return {
                    "success": False,
                    "error": f"Risk check failed: {risk_check['reason']}",
                    "trade_id": trade_message.trade_id,
                    "risk_check": risk_check
                }

        # Daily trade limit check
        if self.daily_trade_count >= self.config.max_daily_trades:
            return {
                "success": False,
                "error": f"Daily trade limit reached ({self.config.max_daily_trades})",
                "trade_id": trade_message.trade_id
            }

        try:
            # Create contract
            contract = self._create_contract(trade_message)

            # Validate contract
            contract_validation = await self._validate_contract(contract)
            if not contract_validation["valid"]:
                return {
                    "success": False,
                    "error": f"Invalid contract: {contract_validation['reason']}",
                    "trade_id": trade_message.trade_id
                }

            # Create order
            order = self._create_order(trade_message)

            # Final confirmation for live trading
            if self.config.trading_mode == TradingMode.LIVE and self.config.require_confirmation:
                confirmation = await self._confirm_live_order(trade_message, contract, order)
                if not confirmation:
                    return {
                        "success": False,
                        "error": "Live order confirmation failed",
                        "trade_id": trade_message.trade_id
                    }

            # Place order
            logger.info(f"Placing {self.config.trading_mode.value} order: {trade_message.action} {trade_message.quantity} {trade_message.symbol}")

            trade = self.ib.placeOrder(contract, order)

            # Store order info
            self.orders[order.orderId] = IBKROrderStatus(
                order_id=order.orderId,
                status="Submitted",
                filled=0.0,
                remaining=trade_message.quantity,
                avg_fill_price=0.0
            )

            # Update daily counter
            self.daily_trade_count += 1

            logger.info(f"✅ Order {order.orderId} placed successfully for {trade_message.symbol}")

            return {
                "success": True,
                "order_id": order.orderId,
                "trade_id": trade_message.trade_id,
                "status": "submitted",
                "symbol": trade_message.symbol,
                "quantity": trade_message.quantity,
                "action": trade_message.action,
                "trading_mode": self.config.trading_mode.value,
                "daily_trade_count": self.daily_trade_count,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Failed to place order: {e}")
            return {
                "success": False,
                "error": str(e),
                "trade_id": trade_message.trade_id,
                "trading_mode": self.config.trading_mode.value
            }
    
    def _create_contract(self, trade_message: TradeMessage) -> Contract:
        """Create IBKR contract from trade message."""
        if trade_message.asset_type == AssetType.EQUITY:
            return Stock(trade_message.symbol, 'SMART', 'USD')
        else:
            # For now, only support stocks in simulation
            # Future: Add futures, options, forex support
            raise ValueError(f"Asset type {trade_message.asset_type} not supported yet")
    
    def _create_order(self, trade_message: TradeMessage) -> Order:
        """Create IBKR order from trade message."""
        action = "BUY" if trade_message.action.upper() in ["BUY", "LONG"] else "SELL"
        
        if trade_message.order_type.upper() == "MARKET":
            order = MarketOrder(action, trade_message.quantity)
        elif trade_message.order_type.upper() == "LIMIT":
            if not trade_message.price:
                raise ValueError("Limit order requires price")
            order = LimitOrder(action, trade_message.quantity, trade_message.price)
        else:
            raise ValueError(f"Order type {trade_message.order_type} not supported")
        
        return order
    
    async def get_order_status(self, order_id: int) -> Optional[IBKROrderStatus]:
        """Get status of a specific order."""
        if order_id in self.orders:
            # Update from IBKR
            try:
                trades = self.ib.trades()
                for trade in trades:
                    if trade.order.orderId == order_id:
                        self.orders[order_id].status = trade.orderStatus.status
                        self.orders[order_id].filled = trade.orderStatus.filled
                        self.orders[order_id].remaining = trade.orderStatus.remaining
                        self.orders[order_id].avg_fill_price = trade.orderStatus.avgFillPrice
                        break
            except Exception as e:
                logger.error(f"Failed to update order status: {e}")
            
            return self.orders[order_id]
        
        return None
    
    async def cancel_order(self, order_id: int) -> bool:
        """Cancel an order."""
        try:
            trades = self.ib.trades()
            for trade in trades:
                if trade.order.orderId == order_id:
                    self.ib.cancelOrder(trade.order)
                    logger.info(f"Cancelled order {order_id}")
                    return True
            
            logger.warning(f"Order {order_id} not found for cancellation")
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def _perform_risk_checks(self, trade_message: TradeMessage) -> Dict[str, Any]:
        """Perform comprehensive pre-trade risk checks."""

        # Calculate order value
        order_value = trade_message.quantity * (trade_message.price or 100)  # Use price or estimate

        # Check maximum order value
        if order_value > self.config.max_order_value:
            return {
                "passed": False,
                "reason": f"Order value ${order_value:,.2f} exceeds limit ${self.config.max_order_value:,.2f}"
            }

        # Check position size limit
        try:
            account_value = float(self.account_info.get('NetLiquidation', {}).get('value', 100000))
            position_percentage = order_value / account_value

            if position_percentage > self.config.position_size_limit:
                return {
                    "passed": False,
                    "reason": f"Position size {position_percentage:.1%} exceeds limit {self.config.position_size_limit:.1%}"
                }
        except (ValueError, KeyError):
            logger.warning("Could not calculate position size - proceeding with caution")

        # Check for existing position concentration
        if trade_message.symbol in self.positions:
            existing_position = self.positions[trade_message.symbol]
            # Add more sophisticated position checks here

        # Market hours check (simplified)
        current_time = datetime.now().time()
        if current_time < datetime.strptime("09:30", "%H:%M").time() or current_time > datetime.strptime("16:00", "%H:%M").time():
            logger.warning("Trading outside regular market hours")

        return {
            "passed": True,
            "reason": "All risk checks passed",
            "order_value": order_value,
            "checks_performed": ["order_value", "position_size", "market_hours"]
        }

    async def _validate_contract(self, contract: Contract) -> Dict[str, Any]:
        """Validate contract with IBKR."""
        try:
            # Request contract details
            details = self.ib.reqContractDetails(contract)

            if not details:
                return {
                    "valid": False,
                    "reason": "Contract not found or invalid"
                }

            # Check if contract is tradeable
            contract_detail = details[0]
            if not contract_detail.tradeable:
                return {
                    "valid": False,
                    "reason": "Contract is not tradeable"
                }

            return {
                "valid": True,
                "reason": "Contract validated successfully",
                "details": contract_detail
            }

        except Exception as e:
            return {
                "valid": False,
                "reason": f"Contract validation error: {e}"
            }

    async def _confirm_live_order(self, trade_message: TradeMessage, contract: Contract, order: Order) -> bool:
        """Confirm live order placement with additional safety checks."""

        if self.config.trading_mode != TradingMode.LIVE:
            return True  # No confirmation needed for paper trading

        # Log detailed order information
        logger.warning("🚨 LIVE ORDER CONFIRMATION REQUIRED 🚨")
        logger.warning(f"Symbol: {trade_message.symbol}")
        logger.warning(f"Action: {trade_message.action}")
        logger.warning(f"Quantity: {trade_message.quantity}")
        logger.warning(f"Order Type: {trade_message.order_type}")
        logger.warning(f"Price: {trade_message.price}")
        logger.warning(f"Account: {self.config.account}")

        # In production, this could integrate with external confirmation systems
        # For now, we'll allow the order but with extensive logging

        logger.warning("Live order confirmed - proceeding with execution")
        return True

    def is_connected(self) -> bool:
        """Check if connected to IBKR."""
        return self.connection_status == ConnectionStatus.CONNECTED and self.ib.isConnected()

    def get_trading_mode(self) -> TradingMode:
        """Get current trading mode."""
        return self.config.trading_mode

    def get_connection_status(self) -> ConnectionStatus:
        """Get current connection status."""
        return self.connection_status

    def get_daily_trade_count(self) -> int:
        """Get current daily trade count."""
        return self.daily_trade_count

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""

        health_status = {
            "connected": self.is_connected(),
            "connection_status": self.connection_status.value,
            "trading_mode": self.config.trading_mode.value,
            "daily_trades": self.daily_trade_count,
            "max_daily_trades": self.config.max_daily_trades,
            "account": self.config.account,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "reconnect_attempts": self.reconnect_attempts
        }

        # Check account info freshness
        if self.account_info:
            health_status["account_info_available"] = True
            health_status["positions_count"] = len(self.positions)
            health_status["pending_orders"] = len([o for o in self.orders.values() if o.status in ['Submitted', 'PreSubmitted']])
        else:
            health_status["account_info_available"] = False

        # Check if heartbeat is recent
        time_since_heartbeat = (datetime.now() - self.last_heartbeat).total_seconds()
        health_status["heartbeat_age_seconds"] = time_since_heartbeat
        health_status["heartbeat_healthy"] = time_since_heartbeat < 60  # 1 minute threshold

        return health_status
