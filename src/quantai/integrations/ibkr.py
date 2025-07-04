"""
Interactive Brokers (IBKR) integration for the QuantAI system.

This module provides connection and trading capabilities with IBKR's
TWS (Trader Workstation) or IB Gateway for production simulation.
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal

from ib_insync import IB, Stock, Order, MarketOrder, LimitOrder, Contract
from loguru import logger
from pydantic import BaseModel

from ..core.messages import TradeMessage, AssetType, ProductType


class IBKRConfig(BaseModel):
    """Configuration for IBKR connection."""
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    account: str = "DUA559603"
    timeout: int = 30
    readonly: bool = False


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
    Interactive Brokers client for production simulation.
    
    Provides connection to IBKR TWS/Gateway and trading functionality
    for the QuantAI system's production simulation environment.
    """
    
    def __init__(self, config: IBKRConfig):
        self.config = config
        self.ib = IB()
        self.connected = False
        self.account_info: Dict[str, Any] = {}
        self.positions: Dict[str, IBKRPosition] = {}
        self.orders: Dict[int, IBKROrderStatus] = {}
        
    async def connect(self) -> bool:
        """Connect to IBKR TWS/Gateway."""
        try:
            logger.info(f"Connecting to IBKR at {self.config.host}:{self.config.port}")
            
            # Connect to TWS/Gateway
            await self.ib.connectAsync(
                host=self.config.host,
                port=self.config.port,
                clientId=self.config.client_id,
                timeout=self.config.timeout,
                readonly=self.config.readonly
            )
            
            self.connected = True
            logger.info("Successfully connected to IBKR")
            
            # Get account information
            await self._update_account_info()
            await self._update_positions()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from IBKR."""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IBKR")
    
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
        Place an order through IBKR.
        
        Args:
            trade_message: Trade message with order details
            
        Returns:
            Order execution result
        """
        if not self.connected:
            return {
                "success": False,
                "error": "Not connected to IBKR",
                "trade_id": trade_message.trade_id
            }
        
        try:
            # Create contract
            contract = self._create_contract(trade_message)
            
            # Create order
            order = self._create_order(trade_message)
            
            # Place order
            trade = self.ib.placeOrder(contract, order)
            
            # Store order info
            self.orders[order.orderId] = IBKROrderStatus(
                order_id=order.orderId,
                status="Submitted",
                filled=0.0,
                remaining=trade_message.quantity,
                avg_fill_price=0.0
            )
            
            logger.info(f"Placed order {order.orderId} for {trade_message.symbol}")
            
            return {
                "success": True,
                "order_id": order.orderId,
                "trade_id": trade_message.trade_id,
                "status": "submitted",
                "symbol": trade_message.symbol,
                "quantity": trade_message.quantity,
                "action": trade_message.action
            }
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return {
                "success": False,
                "error": str(e),
                "trade_id": trade_message.trade_id
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
    
    def is_connected(self) -> bool:
        """Check if connected to IBKR."""
        return self.connected and self.ib.isConnected()
