"""
Execution Agent (Trader) for the QuantAI multi-agent system.

This agent is responsible for executing trades based on strategy signals
and managing order execution in the live trading environment with IBKR integration.
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from autogen_core import Agent, MessageContext
from loguru import logger

from ...core.base import BaseQuantAgent, AgentRole, AgentCapability
from ...core.messages import QuantMessage, MessageType, TradeMessage
from ...integrations.ibkr import IBKRClient, IBKRConfig, TradingMode
from ...integrations.trading_manager import TradingManager, TradingManagerConfig
from ...integrations.telegram import TelegramNotifier, TelegramConfig


class ExecutionAgent(BaseQuantAgent):
    """
    Agent responsible for executing trades in the live market with IBKR integration.

    This agent receives trade signals from strategies and executes
    them through Interactive Brokers for production simulation,
    with real-time Telegram notifications.
    """

    def __init__(self, agent_id: str = "execution", use_ibkr: bool = True, use_telegram: bool = True):
        super().__init__(
            role=AgentRole.EXECUTION,
            capabilities=[
                AgentCapability.TRADE_EXECUTION,
                AgentCapability.ORDER_MANAGEMENT,
                AgentCapability.MARKET_ACCESS,
            ]
        )
        self.agent_id = agent_id
        self.use_ibkr = use_ibkr
        self.use_telegram = use_telegram

        # Trading state
        self.pending_orders: Dict[str, Dict[str, Any]] = {}
        self.executed_trades: List[Dict[str, Any]] = []
        self.execution_stats = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_volume': 0.0
        }

        # Enhanced Trading Manager integration
        self.use_trading_manager = os.getenv("USE_TRADING_MANAGER", "true").lower() == "true"
        self.trading_manager: Optional[TradingManager] = None
        if self.use_trading_manager:
            self._initialize_trading_manager()

        # Legacy IBKR integration (fallback)
        self.ibkr_client: Optional[IBKRClient] = None
        if self.use_ibkr and not self.use_trading_manager:
            self._initialize_ibkr()

        # Telegram integration
        self.telegram_notifier: Optional[TelegramNotifier] = None
        if self.use_telegram:
            self._initialize_telegram()

    def _initialize_trading_manager(self):
        """Initialize enhanced Trading Manager."""
        try:
            config = TradingManagerConfig()
            self.trading_manager = TradingManager(config)
            logger.info(f"Trading Manager initialized in {config.ibkr_config.trading_mode.value.upper()} mode")

            # Setup event handlers
            self.trading_manager.on_order_filled = self._handle_order_filled
            self.trading_manager.on_error = self._handle_trading_error

        except Exception as e:
            logger.error(f"Failed to initialize Trading Manager: {e}")
            self.use_trading_manager = False

    def _initialize_ibkr(self):
        """Initialize legacy IBKR client."""
        try:
            config = IBKRConfig.from_env()
            self.ibkr_client = IBKRClient(config)
            logger.info(f"Legacy IBKR client initialized in {config.trading_mode.value.upper()} mode")
        except Exception as e:
            logger.error(f"Failed to initialize IBKR client: {e}")
            self.use_ibkr = False

    def _initialize_telegram(self):
        """Initialize Telegram notifier."""
        try:
            bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
            chat_id = os.getenv("TELEGRAM_CHAT_ID")

            if bot_token and chat_id:
                config = TelegramConfig(
                    bot_token=bot_token,
                    chat_id=chat_id
                )
                self.telegram_notifier = TelegramNotifier(config)
                logger.info("Telegram notifier initialized")
            else:
                logger.warning("Telegram credentials not found")
                self.use_telegram = False
        except Exception as e:
            logger.error(f"Failed to initialize Telegram notifier: {e}")
            self.use_telegram = False
    
    async def on_messages(self, messages: List[QuantMessage], ctx: MessageContext) -> str:
        """Handle incoming messages for trade execution."""
        results = []
        
        for message in messages:
            if isinstance(message, TradeMessage):
                result = await self._handle_trade_execution(message)
                results.append(result)
            else:
                result = await self._handle_general_message(message)
                results.append(result)
        
        return f"ExecutionAgent processed {len(results)} messages"
    
    async def _handle_trade_execution(self, message: TradeMessage) -> Dict[str, Any]:
        """Handle trade execution request."""
        try:
            trade_id = message.trade_id
            
            # Validate trade request
            if not await self._validate_trade_request(message):
                return {
                    'status': 'rejected',
                    'trade_id': trade_id,
                    'error': 'Trade validation failed'
                }
            
            # Execute trade
            execution_result = await self._execute_trade(message)
            
            # Record execution
            self.executed_trades.append({
                'trade_id': trade_id,
                'execution_time': datetime.now(),
                'status': execution_result['status'],
                'details': execution_result
            })
            
            # Update statistics
            self._update_execution_stats(execution_result)
            
            return execution_result
            
        except Exception as e:
            return {
                'status': 'error',
                'trade_id': getattr(message, 'trade_id', 'unknown'),
                'error': str(e)
            }
    
    async def _validate_trade_request(self, trade: TradeMessage) -> bool:
        """Validate trade request before execution."""
        # Check required fields
        if not trade.symbol or not trade.action or not trade.quantity:
            return False
        
        # Check action is valid
        if trade.action not in ['BUY', 'SELL']:
            return False
        
        # Check quantity is positive
        if trade.quantity <= 0:
            return False
        
        # Additional validation logic would go here
        # - Risk checks
        # - Position limits
        # - Market hours
        # - Symbol validity
        
        return True
    
    async def _execute_trade(self, trade: TradeMessage) -> Dict[str, Any]:
        """Execute trade in the market."""
        trade_id = trade.trade_id
        
        # Add to pending orders
        self.pending_orders[trade_id] = {
            'symbol': trade.symbol,
            'action': trade.action,
            'quantity': trade.quantity,
            'order_type': trade.order_type,
            'submit_time': datetime.now(),
            'status': 'pending'
        }
        
        try:
            if self.use_trading_manager and self.trading_manager:
                # Execute through enhanced Trading Manager
                result = await self.trading_manager.place_order(trade)
            elif self.use_ibkr and self.ibkr_client and self.ibkr_client.is_connected():
                # Execute through legacy IBKR
                result = await self._execute_ibkr_trade(trade)
            else:
                # Fallback to simulation
                result = await self._execute_simulated_trade(trade)

            # Send Telegram notification
            if self.use_telegram and self.telegram_notifier:
                await self.telegram_notifier.send_trade_notification(trade)

            return result

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")

            # Update order status
            self.pending_orders[trade_id]['status'] = 'failed'
            self.pending_orders[trade_id]['error'] = str(e)

            return {
                'status': 'failed',
                'trade_id': trade_id,
                'error': str(e),
                'execution_time': datetime.now().isoformat()
            }

    async def _execute_ibkr_trade(self, trade: TradeMessage) -> Dict[str, Any]:
        """Execute trade through IBKR."""
        logger.info(f"Executing trade {trade.trade_id} through IBKR")

        # Place order through IBKR
        result = await self.ibkr_client.place_order(trade)

        if result["success"]:
            # Update order status
            self.pending_orders[trade.trade_id]['status'] = 'submitted'
            self.pending_orders[trade.trade_id]['ibkr_order_id'] = result["order_id"]

            # Wait for fill (simplified - in production would use callbacks)
            await asyncio.sleep(1)

            # Check order status
            order_status = await self.ibkr_client.get_order_status(result["order_id"])

            if order_status and order_status.status in ["Filled", "PartiallyFilled"]:
                self.pending_orders[trade.trade_id]['status'] = 'filled'
                self.pending_orders[trade.trade_id]['fill_price'] = order_status.avg_fill_price
                self.pending_orders[trade.trade_id]['fill_quantity'] = order_status.filled
                self.pending_orders[trade.trade_id]['fill_time'] = datetime.now()

                return {
                    'status': 'filled',
                    'trade_id': trade.trade_id,
                    'ibkr_order_id': result["order_id"],
                    'fill_price': order_status.avg_fill_price,
                    'fill_quantity': order_status.filled,
                    'execution_time': datetime.now().isoformat(),
                    'broker': 'IBKR'
                }
            else:
                return {
                    'status': 'submitted',
                    'trade_id': trade.trade_id,
                    'ibkr_order_id': result["order_id"],
                    'execution_time': datetime.now().isoformat(),
                    'broker': 'IBKR'
                }
        else:
            raise Exception(f"IBKR order failed: {result.get('error', 'Unknown error')}")

    async def _execute_simulated_trade(self, trade: TradeMessage) -> Dict[str, Any]:
        """Execute simulated trade (fallback)."""
        logger.info(f"Executing trade {trade.trade_id} in simulation mode")

        # Simulate execution time
        await asyncio.sleep(0.1)

        # Simulate execution result
        import random

        # Simulate fill price (small random variation from requested price)
        base_price = trade.price if trade.price else 100.0
        fill_price = base_price * (1 + random.uniform(-0.001, 0.001))  # 0.1% variation

        # Simulate partial or full fill
        fill_quantity = trade.quantity * random.uniform(0.95, 1.0)  # 95-100% fill

        # Update order status
        self.pending_orders[trade.trade_id]['status'] = 'filled'
        self.pending_orders[trade.trade_id]['fill_price'] = fill_price
        self.pending_orders[trade.trade_id]['fill_quantity'] = fill_quantity
        self.pending_orders[trade.trade_id]['fill_time'] = datetime.now()

        return {
            'status': 'filled',
            'trade_id': trade.trade_id,
            'symbol': trade.symbol,
            'action': trade.action,
            'requested_quantity': trade.quantity,
            'fill_quantity': fill_quantity,
            'fill_price': fill_price,
            'execution_time': datetime.now(),
            'message': f'Trade {trade_id} executed successfully',
            'broker': 'Simulation'
        }

    async def connect_services(self) -> Dict[str, bool]:
        """Connect to external services (IBKR, Telegram)."""
        results = {}

        # Connect to IBKR
        if self.use_ibkr and self.ibkr_client:
            try:
                results['ibkr'] = await self.ibkr_client.connect()
                if results['ibkr']:
                    logger.info("IBKR connection established")
                else:
                    logger.error("Failed to connect to IBKR")
            except Exception as e:
                logger.error(f"IBKR connection error: {e}")
                results['ibkr'] = False
        else:
            results['ibkr'] = False

        # Start Telegram notifier
        if self.use_telegram and self.telegram_notifier:
            try:
                await self.telegram_notifier.start()
                results['telegram'] = self.telegram_notifier.is_running
                if results['telegram']:
                    logger.info("Telegram notifier started")
                else:
                    logger.error("Failed to start Telegram notifier")
            except Exception as e:
                logger.error(f"Telegram connection error: {e}")
                results['telegram'] = False
        else:
            results['telegram'] = False

        return results

    async def disconnect_services(self):
        """Disconnect from external services."""
        # Disconnect IBKR
        if self.ibkr_client:
            try:
                await self.ibkr_client.disconnect()
                logger.info("IBKR disconnected")
            except Exception as e:
                logger.error(f"IBKR disconnect error: {e}")

        # Stop Telegram notifier
        if self.telegram_notifier:
            try:
                await self.telegram_notifier.stop()
                logger.info("Telegram notifier stopped")
            except Exception as e:
                logger.error(f"Telegram stop error: {e}")

    def get_connection_status(self) -> Dict[str, Any]:
        """Get status of external connections."""
        status = {
            'ibkr': {
                'enabled': self.use_ibkr,
                'connected': False,
                'account': None
            },
            'telegram': {
                'enabled': self.use_telegram,
                'running': False
            }
        }

        if self.ibkr_client:
            status['ibkr']['connected'] = self.ibkr_client.is_connected()
            if status['ibkr']['connected']:
                status['ibkr']['account'] = self.ibkr_client.config.account
                status['ibkr']['account_summary'] = self.ibkr_client.get_account_summary()

        if self.telegram_notifier:
            status['telegram']['running'] = self.telegram_notifier.is_running

        return status

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
            if isinstance(message, TradeMessage):
                result = await self._handle_trade_execution(message)
                return QuantMessage(
                    message_type=MessageType.TRADE_RESPONSE,
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
    
    def _update_execution_stats(self, execution_result: Dict[str, Any]) -> None:
        """Update execution statistics."""
        self.execution_stats['total_trades'] += 1
        
        if execution_result['status'] == 'filled':
            self.execution_stats['successful_trades'] += 1
            self.execution_stats['total_volume'] += execution_result.get('fill_quantity', 0)
        else:
            self.execution_stats['failed_trades'] += 1
    
    def get_pending_orders(self) -> Dict[str, Dict[str, Any]]:
        """Get list of pending orders."""
        return {k: v for k, v in self.pending_orders.items() if v['status'] == 'pending'}
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self.executed_trades.copy()
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return self.execution_stats.copy()

    def _handle_order_filled(self, order_info: Dict[str, Any]):
        """Handle order fill events from Trading Manager."""
        logger.info(f"Order filled: {order_info['trade_id']} - {order_info['symbol']}")

        # Update execution stats
        self.execution_stats['successful_trades'] += 1
        self.execution_stats['total_volume'] += order_info.get('fill_quantity', 0) * order_info.get('fill_price', 0)

        # Move from pending to executed
        trade_id = order_info['trade_id']
        if trade_id in self.pending_orders:
            executed_trade = self.pending_orders[trade_id].copy()
            executed_trade.update(order_info)
            executed_trade['execution_time'] = datetime.now()

            self.executed_trades.append(executed_trade)
            del self.pending_orders[trade_id]

    def _handle_trading_error(self, error_code: Any, error_message: str):
        """Handle trading errors from Trading Manager."""
        logger.error(f"Trading error {error_code}: {error_message}")

        # Could implement error recovery logic here
        # For now, just log the error

    async def start_trading_session(self) -> Dict[str, Any]:
        """Start a new trading session."""
        if self.use_trading_manager and self.trading_manager:
            success = await self.trading_manager.start_session()
            if success:
                logger.info("✅ Trading session started successfully")
                return {
                    "status": "success",
                    "message": "Trading session started",
                    "session_info": self.trading_manager.get_session_status()
                }
            else:
                logger.error("❌ Failed to start trading session")
                return {
                    "status": "error",
                    "message": "Failed to start trading session"
                }
        else:
            return {
                "status": "error",
                "message": "Trading Manager not available"
            }

    async def stop_trading_session(self) -> Dict[str, Any]:
        """Stop the current trading session."""
        if self.use_trading_manager and self.trading_manager:
            await self.trading_manager.stop_session()
            logger.info("Trading session stopped")
            return {
                "status": "success",
                "message": "Trading session stopped"
            }
        else:
            return {
                "status": "error",
                "message": "Trading Manager not available"
            }

    def get_trading_status(self) -> Dict[str, Any]:
        """Get comprehensive trading status."""
        status = {
            "agent_id": self.agent_id,
            "use_trading_manager": self.use_trading_manager,
            "use_ibkr": self.use_ibkr,
            "use_telegram": self.use_telegram,
            "execution_stats": self.execution_stats.copy(),
            "pending_orders_count": len(self.pending_orders),
            "executed_trades_count": len(self.executed_trades)
        }

        # Add Trading Manager status if available
        if self.use_trading_manager and self.trading_manager:
            status["trading_manager"] = self.trading_manager.get_session_status()

        # Add IBKR status if available
        if self.use_ibkr and self.ibkr_client:
            status["ibkr_connected"] = self.ibkr_client.is_connected()
            if hasattr(self.ibkr_client, 'get_trading_mode'):
                status["trading_mode"] = self.ibkr_client.get_trading_mode().value

        return status
