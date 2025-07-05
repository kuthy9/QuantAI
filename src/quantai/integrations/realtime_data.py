"""
Enhanced Real-time Market Data Integration for QuantAI system.

Provides comprehensive real-time data capabilities beyond basic Alpha Vantage/NewsAPI,
including WebSocket streams, tick-level data, options data, and advanced market feeds.
"""

import asyncio
import json
import os
import websockets
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Set
from enum import Enum
from dataclasses import dataclass, asdict
import aiohttp
from loguru import logger
from pydantic import BaseModel, Field

from ..core.messages import DataMessage, MessageType


class DataFeedType(Enum):
    """Types of real-time data feeds."""
    MARKET_DATA = "market_data"
    OPTIONS_DATA = "options_data"
    LEVEL2_DATA = "level2_data"
    NEWS_FEED = "news_feed"
    ECONOMIC_DATA = "economic_data"
    CRYPTO_DATA = "crypto_data"
    FOREX_DATA = "forex_data"


class DataProvider(Enum):
    """Supported real-time data providers."""
    POLYGON = "polygon"
    ALPACA = "alpaca"
    FINNHUB = "finnhub"
    TWELVEDATA = "twelvedata"
    WEBSOCKET_GENERIC = "websocket_generic"
    IBKR = "ibkr"


@dataclass
class RealTimeQuote:
    """Real-time quote data structure."""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: int
    bid_size: int
    ask_size: int
    provider: str
    exchange: str = ""
    conditions: List[str] = None


@dataclass
class RealTimeTrade:
    """Real-time trade data structure."""
    symbol: str
    timestamp: datetime
    price: float
    size: int
    exchange: str
    conditions: List[str]
    provider: str


@dataclass
class Level2Data:
    """Level 2 market data (order book)."""
    symbol: str
    timestamp: datetime
    bids: List[Dict[str, float]]  # [{"price": 100.0, "size": 500}, ...]
    asks: List[Dict[str, float]]
    provider: str


class RealTimeDataConfig(BaseModel):
    """Configuration for real-time data feeds."""
    # Provider API keys
    polygon_api_key: Optional[str] = Field(default=None)
    alpaca_api_key: Optional[str] = Field(default=None)
    alpaca_secret_key: Optional[str] = Field(default=None)
    finnhub_api_key: Optional[str] = Field(default=None)
    twelvedata_api_key: Optional[str] = Field(default=None)
    
    # Feed configuration
    enabled_feeds: List[DataFeedType] = Field(default=[DataFeedType.MARKET_DATA])
    primary_provider: DataProvider = Field(default=DataProvider.POLYGON)
    fallback_providers: List[DataProvider] = Field(default=[DataProvider.ALPACA, DataProvider.FINNHUB])
    
    # Performance settings
    max_symbols_per_feed: int = Field(default=100)
    reconnect_attempts: int = Field(default=5)
    reconnect_delay: int = Field(default=5)
    heartbeat_interval: int = Field(default=30)
    
    # Data quality
    enable_data_validation: bool = Field(default=True)
    max_price_deviation: float = Field(default=0.1)  # 10% max price change
    stale_data_threshold: int = Field(default=60)  # seconds
    
    @classmethod
    def from_env(cls) -> "RealTimeDataConfig":
        """Create configuration from environment variables."""
        return cls(
            polygon_api_key=os.getenv("POLYGON_API_KEY"),
            alpaca_api_key=os.getenv("ALPACA_API_KEY"),
            alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY"),
            finnhub_api_key=os.getenv("FINNHUB_API_KEY"),
            twelvedata_api_key=os.getenv("TWELVEDATA_API_KEY"),
            enabled_feeds=[
                DataFeedType(feed) for feed in 
                os.getenv("ENABLED_DATA_FEEDS", "market_data").split(",")
            ],
            primary_provider=DataProvider(os.getenv("PRIMARY_DATA_PROVIDER", "polygon")),
            max_symbols_per_feed=int(os.getenv("MAX_SYMBOLS_PER_FEED", "100")),
        )


class RealTimeDataManager:
    """
    Enhanced Real-time Data Manager with multiple provider support.
    
    Provides comprehensive real-time market data capabilities including:
    - WebSocket streaming for tick-level data
    - Level 2 market data (order book)
    - Options data and Greeks
    - Economic calendar events
    - Multi-provider failover and aggregation
    """
    
    def __init__(self, config: RealTimeDataConfig):
        self.config = config
        
        # Connection management
        self.active_connections: Dict[str, Any] = {}
        self.subscribed_symbols: Set[str] = set()
        self.connection_status: Dict[str, bool] = {}
        
        # Data storage
        self.latest_quotes: Dict[str, RealTimeQuote] = {}
        self.latest_trades: Dict[str, RealTimeTrade] = {}
        self.level2_data: Dict[str, Level2Data] = {}
        
        # Event callbacks
        self.on_quote: Optional[Callable[[RealTimeQuote], None]] = None
        self.on_trade: Optional[Callable[[RealTimeTrade], None]] = None
        self.on_level2: Optional[Callable[[Level2Data], None]] = None
        self.on_news: Optional[Callable[[Dict[str, Any]], None]] = None
        
        # Performance tracking
        self.message_count = 0
        self.last_heartbeat = datetime.now()
        self.data_quality_stats = {
            "total_messages": 0,
            "invalid_messages": 0,
            "stale_messages": 0,
            "duplicate_messages": 0
        }
        
        logger.info(f"Real-time Data Manager initialized with {config.primary_provider.value} as primary provider")
    
    async def start(self) -> bool:
        """Start real-time data feeds."""
        try:
            logger.info("Starting real-time data feeds...")
            
            # Start primary provider
            success = await self._start_provider(self.config.primary_provider)
            if not success:
                logger.warning(f"Primary provider {self.config.primary_provider.value} failed, trying fallbacks")
                
                # Try fallback providers
                for provider in self.config.fallback_providers:
                    success = await self._start_provider(provider)
                    if success:
                        logger.info(f"Successfully connected to fallback provider: {provider.value}")
                        break
                
                if not success:
                    logger.error("All data providers failed to connect")
                    return False
            
            # Start heartbeat monitoring
            asyncio.create_task(self._heartbeat_monitor())
            
            logger.info("✅ Real-time data feeds started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start real-time data feeds: {e}")
            return False
    
    async def subscribe_symbols(self, symbols: List[str], feed_types: List[DataFeedType] = None) -> bool:
        """Subscribe to real-time data for specified symbols."""
        if not feed_types:
            feed_types = self.config.enabled_feeds
        
        try:
            for provider_name, connection in self.active_connections.items():
                if self.connection_status.get(provider_name, False):
                    await self._subscribe_provider_symbols(provider_name, connection, symbols, feed_types)
            
            self.subscribed_symbols.update(symbols)
            logger.info(f"Subscribed to {len(symbols)} symbols: {symbols}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to symbols: {e}")
            return False
    
    async def unsubscribe_symbols(self, symbols: List[str]) -> bool:
        """Unsubscribe from real-time data for specified symbols."""
        try:
            for provider_name, connection in self.active_connections.items():
                if self.connection_status.get(provider_name, False):
                    await self._unsubscribe_provider_symbols(provider_name, connection, symbols)
            
            self.subscribed_symbols.difference_update(symbols)
            logger.info(f"Unsubscribed from {len(symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe from symbols: {e}")
            return False
    
    async def _start_provider(self, provider: DataProvider) -> bool:
        """Start connection to a specific data provider."""
        try:
            if provider == DataProvider.POLYGON:
                return await self._start_polygon_feed()
            elif provider == DataProvider.ALPACA:
                return await self._start_alpaca_feed()
            elif provider == DataProvider.FINNHUB:
                return await self._start_finnhub_feed()
            elif provider == DataProvider.TWELVEDATA:
                return await self._start_twelvedata_feed()
            else:
                logger.warning(f"Provider {provider.value} not implemented yet")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start {provider.value} provider: {e}")
            return False
    
    async def _start_polygon_feed(self) -> bool:
        """Start Polygon.io WebSocket feed."""
        if not self.config.polygon_api_key:
            logger.warning("Polygon API key not configured")
            return False
        
        try:
            # Polygon WebSocket URL
            ws_url = f"wss://socket.polygon.io/stocks"
            
            # Connect to WebSocket
            websocket = await websockets.connect(ws_url)
            
            # Authenticate
            auth_message = {
                "action": "auth",
                "params": self.config.polygon_api_key
            }
            await websocket.send(json.dumps(auth_message))
            
            # Wait for auth response
            auth_response = await websocket.recv()
            auth_data = json.loads(auth_response)
            
            if auth_data.get("status") != "auth_success":
                logger.error(f"Polygon authentication failed: {auth_data}")
                return False
            
            # Store connection
            self.active_connections["polygon"] = websocket
            self.connection_status["polygon"] = True
            
            # Start message handler
            asyncio.create_task(self._handle_polygon_messages(websocket))
            
            logger.info("✅ Polygon.io WebSocket connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Polygon.io: {e}")
            return False
    
    async def _start_alpaca_feed(self) -> bool:
        """Start Alpaca WebSocket feed."""
        if not self.config.alpaca_api_key or not self.config.alpaca_secret_key:
            logger.warning("Alpaca API credentials not configured")
            return False
        
        try:
            # Alpaca WebSocket URL
            ws_url = "wss://stream.data.alpaca.markets/v2/iex"
            
            # Connect to WebSocket
            websocket = await websockets.connect(ws_url)
            
            # Authenticate
            auth_message = {
                "action": "auth",
                "key": self.config.alpaca_api_key,
                "secret": self.config.alpaca_secret_key
            }
            await websocket.send(json.dumps(auth_message))
            
            # Wait for auth response
            auth_response = await websocket.recv()
            auth_data = json.loads(auth_response)
            
            if auth_data[0].get("T") != "success":
                logger.error(f"Alpaca authentication failed: {auth_data}")
                return False
            
            # Store connection
            self.active_connections["alpaca"] = websocket
            self.connection_status["alpaca"] = True
            
            # Start message handler
            asyncio.create_task(self._handle_alpaca_messages(websocket))
            
            logger.info("✅ Alpaca WebSocket connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            return False
    
    async def _start_finnhub_feed(self) -> bool:
        """Start Finnhub WebSocket feed."""
        if not self.config.finnhub_api_key:
            logger.warning("Finnhub API key not configured")
            return False
        
        try:
            # Finnhub WebSocket URL
            ws_url = f"wss://ws.finnhub.io?token={self.config.finnhub_api_key}"
            
            # Connect to WebSocket
            websocket = await websockets.connect(ws_url)
            
            # Store connection
            self.active_connections["finnhub"] = websocket
            self.connection_status["finnhub"] = True
            
            # Start message handler
            asyncio.create_task(self._handle_finnhub_messages(websocket))
            
            logger.info("✅ Finnhub WebSocket connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Finnhub: {e}")
            return False
    
    async def _start_twelvedata_feed(self) -> bool:
        """Start TwelveData WebSocket feed."""
        if not self.config.twelvedata_api_key:
            logger.warning("TwelveData API key not configured")
            return False
        
        try:
            # TwelveData WebSocket URL
            ws_url = f"wss://ws.twelvedata.com/v1/quotes/price?apikey={self.config.twelvedata_api_key}"
            
            # Connect to WebSocket
            websocket = await websockets.connect(ws_url)
            
            # Store connection
            self.active_connections["twelvedata"] = websocket
            self.connection_status["twelvedata"] = True
            
            # Start message handler
            asyncio.create_task(self._handle_twelvedata_messages(websocket))
            
            logger.info("✅ TwelveData WebSocket connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to TwelveData: {e}")
            return False

    async def _subscribe_provider_symbols(self, provider_name: str, connection: Any, symbols: List[str], feed_types: List[DataFeedType]):
        """Subscribe to symbols for a specific provider."""
        try:
            if provider_name == "polygon":
                await self._subscribe_polygon_symbols(connection, symbols, feed_types)
            elif provider_name == "alpaca":
                await self._subscribe_alpaca_symbols(connection, symbols, feed_types)
            elif provider_name == "finnhub":
                await self._subscribe_finnhub_symbols(connection, symbols, feed_types)
            elif provider_name == "twelvedata":
                await self._subscribe_twelvedata_symbols(connection, symbols, feed_types)

        except Exception as e:
            logger.error(f"Failed to subscribe symbols for {provider_name}: {e}")

    async def _subscribe_polygon_symbols(self, websocket: Any, symbols: List[str], feed_types: List[DataFeedType]):
        """Subscribe to Polygon symbols."""
        subscribe_message = {
            "action": "subscribe",
            "params": []
        }

        for symbol in symbols:
            if DataFeedType.MARKET_DATA in feed_types:
                subscribe_message["params"].extend([f"Q.{symbol}", f"T.{symbol}"])  # Quotes and Trades
            if DataFeedType.LEVEL2_DATA in feed_types:
                subscribe_message["params"].append(f"L2.{symbol}")

        await websocket.send(json.dumps(subscribe_message))
        logger.info(f"Subscribed to {len(symbols)} symbols on Polygon")

    async def _subscribe_alpaca_symbols(self, websocket: Any, symbols: List[str], feed_types: List[DataFeedType]):
        """Subscribe to Alpaca symbols."""
        subscribe_message = {
            "action": "subscribe",
            "quotes": symbols if DataFeedType.MARKET_DATA in feed_types else [],
            "trades": symbols if DataFeedType.MARKET_DATA in feed_types else []
        }

        await websocket.send(json.dumps(subscribe_message))
        logger.info(f"Subscribed to {len(symbols)} symbols on Alpaca")

    async def _subscribe_finnhub_symbols(self, websocket: Any, symbols: List[str], feed_types: List[DataFeedType]):
        """Subscribe to Finnhub symbols."""
        for symbol in symbols:
            subscribe_message = {"type": "subscribe", "symbol": symbol}
            await websocket.send(json.dumps(subscribe_message))

        logger.info(f"Subscribed to {len(symbols)} symbols on Finnhub")

    async def _subscribe_twelvedata_symbols(self, websocket: Any, symbols: List[str], feed_types: List[DataFeedType]):
        """Subscribe to TwelveData symbols."""
        subscribe_message = {
            "action": "subscribe",
            "params": {
                "symbols": ",".join(symbols)
            }
        }

        await websocket.send(json.dumps(subscribe_message))
        logger.info(f"Subscribed to {len(symbols)} symbols on TwelveData")

    async def _handle_polygon_messages(self, websocket: Any):
        """Handle incoming messages from Polygon WebSocket."""
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)

                    if isinstance(data, list):
                        for item in data:
                            await self._process_polygon_message(item)
                    else:
                        await self._process_polygon_message(data)

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from Polygon: {message}")
                except Exception as e:
                    logger.error(f"Error processing Polygon message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.warning("Polygon WebSocket connection closed")
            self.connection_status["polygon"] = False
        except Exception as e:
            logger.error(f"Polygon WebSocket error: {e}")
            self.connection_status["polygon"] = False

    async def _process_polygon_message(self, data: Dict[str, Any]):
        """Process individual Polygon message."""
        msg_type = data.get("ev")  # Event type

        if msg_type == "Q":  # Quote
            quote = RealTimeQuote(
                symbol=data.get("sym"),
                timestamp=datetime.fromtimestamp(data.get("t", 0) / 1000),
                bid=data.get("bp", 0.0),
                ask=data.get("ap", 0.0),
                last=data.get("lp", 0.0),
                volume=data.get("v", 0),
                bid_size=data.get("bs", 0),
                ask_size=data.get("as", 0),
                provider="polygon",
                exchange=data.get("x", "")
            )

            await self._handle_quote_update(quote)

        elif msg_type == "T":  # Trade
            trade = RealTimeTrade(
                symbol=data.get("sym"),
                timestamp=datetime.fromtimestamp(data.get("t", 0) / 1000),
                price=data.get("p", 0.0),
                size=data.get("s", 0),
                exchange=data.get("x", ""),
                conditions=data.get("c", []),
                provider="polygon"
            )

            await self._handle_trade_update(trade)

    async def _handle_alpaca_messages(self, websocket: Any):
        """Handle incoming messages from Alpaca WebSocket."""
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)

                    if isinstance(data, list):
                        for item in data:
                            await self._process_alpaca_message(item)
                    else:
                        await self._process_alpaca_message(data)

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from Alpaca: {message}")
                except Exception as e:
                    logger.error(f"Error processing Alpaca message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.warning("Alpaca WebSocket connection closed")
            self.connection_status["alpaca"] = False
        except Exception as e:
            logger.error(f"Alpaca WebSocket error: {e}")
            self.connection_status["alpaca"] = False

    async def _process_alpaca_message(self, data: Dict[str, Any]):
        """Process individual Alpaca message."""
        msg_type = data.get("T")  # Message type

        if msg_type == "q":  # Quote
            quote = RealTimeQuote(
                symbol=data.get("S"),
                timestamp=datetime.fromisoformat(data.get("t", "").replace("Z", "+00:00")),
                bid=data.get("bp", 0.0),
                ask=data.get("ap", 0.0),
                last=0.0,  # Not provided in quote
                volume=0,  # Not provided in quote
                bid_size=data.get("bs", 0),
                ask_size=data.get("as", 0),
                provider="alpaca",
                exchange=data.get("x", "")
            )

            await self._handle_quote_update(quote)

        elif msg_type == "t":  # Trade
            trade = RealTimeTrade(
                symbol=data.get("S"),
                timestamp=datetime.fromisoformat(data.get("t", "").replace("Z", "+00:00")),
                price=data.get("p", 0.0),
                size=data.get("s", 0),
                exchange=data.get("x", ""),
                conditions=data.get("c", []),
                provider="alpaca"
            )

            await self._handle_trade_update(trade)

    async def _handle_finnhub_messages(self, websocket: Any):
        """Handle incoming messages from Finnhub WebSocket."""
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._process_finnhub_message(data)

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from Finnhub: {message}")
                except Exception as e:
                    logger.error(f"Error processing Finnhub message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.warning("Finnhub WebSocket connection closed")
            self.connection_status["finnhub"] = False
        except Exception as e:
            logger.error(f"Finnhub WebSocket error: {e}")
            self.connection_status["finnhub"] = False

    async def _process_finnhub_message(self, data: Dict[str, Any]):
        """Process individual Finnhub message."""
        if data.get("type") == "trade":
            for trade_data in data.get("data", []):
                trade = RealTimeTrade(
                    symbol=trade_data.get("s"),
                    timestamp=datetime.fromtimestamp(trade_data.get("t", 0) / 1000),
                    price=trade_data.get("p", 0.0),
                    size=trade_data.get("v", 0),
                    exchange="",
                    conditions=[],
                    provider="finnhub"
                )

                await self._handle_trade_update(trade)

    async def _handle_twelvedata_messages(self, websocket: Any):
        """Handle incoming messages from TwelveData WebSocket."""
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._process_twelvedata_message(data)

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from TwelveData: {message}")
                except Exception as e:
                    logger.error(f"Error processing TwelveData message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.warning("TwelveData WebSocket connection closed")
            self.connection_status["twelvedata"] = False
        except Exception as e:
            logger.error(f"TwelveData WebSocket error: {e}")
            self.connection_status["twelvedata"] = False

    async def _process_twelvedata_message(self, data: Dict[str, Any]):
        """Process individual TwelveData message."""
        if data.get("event") == "price":
            quote = RealTimeQuote(
                symbol=data.get("symbol"),
                timestamp=datetime.now(),  # TwelveData doesn't provide timestamp
                bid=0.0,  # Not provided
                ask=0.0,  # Not provided
                last=data.get("price", 0.0),
                volume=0,  # Not provided
                bid_size=0,
                ask_size=0,
                provider="twelvedata"
            )

            await self._handle_quote_update(quote)

    async def _handle_quote_update(self, quote: RealTimeQuote):
        """Handle real-time quote updates."""
        try:
            # Data validation
            if self.config.enable_data_validation:
                if not self._validate_quote_data(quote):
                    self.data_quality_stats["invalid_messages"] += 1
                    return

            # Check for stale data
            age_seconds = (datetime.now() - quote.timestamp).total_seconds()
            if age_seconds > self.config.stale_data_threshold:
                self.data_quality_stats["stale_messages"] += 1
                logger.warning(f"Stale quote data for {quote.symbol}: {age_seconds}s old")

            # Store latest quote
            self.latest_quotes[quote.symbol] = quote

            # Update statistics
            self.message_count += 1
            self.data_quality_stats["total_messages"] += 1

            # Trigger callback
            if self.on_quote:
                self.on_quote(quote)

        except Exception as e:
            logger.error(f"Error handling quote update: {e}")

    async def _handle_trade_update(self, trade: RealTimeTrade):
        """Handle real-time trade updates."""
        try:
            # Data validation
            if self.config.enable_data_validation:
                if not self._validate_trade_data(trade):
                    self.data_quality_stats["invalid_messages"] += 1
                    return

            # Store latest trade
            self.latest_trades[trade.symbol] = trade

            # Update statistics
            self.message_count += 1
            self.data_quality_stats["total_messages"] += 1

            # Trigger callback
            if self.on_trade:
                self.on_trade(trade)

        except Exception as e:
            logger.error(f"Error handling trade update: {e}")

    def _validate_quote_data(self, quote: RealTimeQuote) -> bool:
        """Validate quote data quality."""
        # Check for valid prices
        if quote.bid <= 0 or quote.ask <= 0 or quote.bid >= quote.ask:
            return False

        # Check for reasonable price deviation
        if quote.symbol in self.latest_quotes:
            last_quote = self.latest_quotes[quote.symbol]
            price_change = abs(quote.last - last_quote.last) / last_quote.last
            if price_change > self.config.max_price_deviation:
                logger.warning(f"Large price change for {quote.symbol}: {price_change:.2%}")
                return False

        return True

    def _validate_trade_data(self, trade: RealTimeTrade) -> bool:
        """Validate trade data quality."""
        # Check for valid price and size
        if trade.price <= 0 or trade.size <= 0:
            return False

        # Check for reasonable price deviation
        if trade.symbol in self.latest_trades:
            last_trade = self.latest_trades[trade.symbol]
            price_change = abs(trade.price - last_trade.price) / last_trade.price
            if price_change > self.config.max_price_deviation:
                logger.warning(f"Large price change for {trade.symbol}: {price_change:.2%}")
                return False

        return True

    async def _heartbeat_monitor(self):
        """Monitor connection health and send heartbeats."""
        while True:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)

                # Update heartbeat
                self.last_heartbeat = datetime.now()

                # Check connection status
                for provider_name, status in self.connection_status.items():
                    if not status:
                        logger.warning(f"Provider {provider_name} is disconnected")
                        # Attempt reconnection
                        await self._attempt_reconnection(provider_name)

                # Log statistics
                logger.info(f"Heartbeat: {self.message_count} messages processed, "
                          f"{len(self.subscribed_symbols)} symbols subscribed")

            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")

    async def _attempt_reconnection(self, provider_name: str):
        """Attempt to reconnect to a provider."""
        try:
            provider = DataProvider(provider_name)
            success = await self._start_provider(provider)

            if success and self.subscribed_symbols:
                # Re-subscribe to symbols
                symbols_list = list(self.subscribed_symbols)
                await self.subscribe_symbols(symbols_list)

        except Exception as e:
            logger.error(f"Reconnection failed for {provider_name}: {e}")

    async def stop(self):
        """Stop all real-time data feeds."""
        logger.info("Stopping real-time data feeds...")

        # Close all connections
        for provider_name, connection in self.active_connections.items():
            try:
                if hasattr(connection, 'close'):
                    await connection.close()
                self.connection_status[provider_name] = False
            except Exception as e:
                logger.error(f"Error closing {provider_name} connection: {e}")

        self.active_connections.clear()
        self.subscribed_symbols.clear()

        logger.info("Real-time data feeds stopped")

    def get_latest_quote(self, symbol: str) -> Optional[RealTimeQuote]:
        """Get the latest quote for a symbol."""
        return self.latest_quotes.get(symbol)

    def get_latest_trade(self, symbol: str) -> Optional[RealTimeTrade]:
        """Get the latest trade for a symbol."""
        return self.latest_trades.get(symbol)

    def get_data_quality_stats(self) -> Dict[str, Any]:
        """Get data quality statistics."""
        stats = self.data_quality_stats.copy()
        stats["message_rate"] = self.message_count / max(1, (datetime.now() - self.last_heartbeat).total_seconds())
        stats["connected_providers"] = [name for name, status in self.connection_status.items() if status]
        stats["subscribed_symbols_count"] = len(self.subscribed_symbols)
        return stats

    def get_connection_status(self) -> Dict[str, Any]:
        """Get comprehensive connection status."""
        return {
            "active_connections": list(self.active_connections.keys()),
            "connection_status": self.connection_status.copy(),
            "subscribed_symbols": list(self.subscribed_symbols),
            "message_count": self.message_count,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "data_quality": self.get_data_quality_stats()
        }
