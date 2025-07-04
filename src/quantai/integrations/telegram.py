"""
Telegram Bot integration for the QuantAI system.

This module provides real-time notifications and alerts through Telegram
for trading activities, system status, and performance updates.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

import aiohttp
from loguru import logger
from pydantic import BaseModel

from ..core.messages import TradeMessage, RiskMessage, ControlMessage


class NotificationLevel(str, Enum):
    """Notification severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class TelegramConfig(BaseModel):
    """Configuration for Telegram bot."""
    bot_token: str
    chat_id: str
    api_base: str = "https://api.telegram.org"
    timeout: int = 30
    max_retries: int = 3


class TelegramMessage(BaseModel):
    """Telegram message structure."""
    text: str
    level: NotificationLevel = NotificationLevel.INFO
    parse_mode: str = "Markdown"
    disable_notification: bool = False


class TelegramNotifier:
    """
    Telegram bot client for QuantAI notifications.
    
    Provides real-time notifications for trading activities, system alerts,
    performance updates, and emergency situations.
    """
    
    def __init__(self, config: TelegramConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.message_queue: List[TelegramMessage] = []
        self.is_running = False
        
        # Message templates
        self.templates = {
            "trade_executed": "🔄 **Trade Executed**\n"
                            "Symbol: `{symbol}`\n"
                            "Action: `{action}`\n"
                            "Quantity: `{quantity}`\n"
                            "Price: `${price:.2f}`\n"
                            "Time: `{timestamp}`",
            
            "system_start": "🚀 **QuantAI System Started**\n"
                          "Environment: `{environment}`\n"
                          "Agents: `{agent_count}`\n"
                          "Time: `{timestamp}`",
            
            "system_stop": "🛑 **QuantAI System Stopped**\n"
                         "Reason: `{reason}`\n"
                         "Time: `{timestamp}`",
            
            "risk_alert": "⚠️ **Risk Alert**\n"
                        "Type: `{risk_type}`\n"
                        "Level: `{level}`\n"
                        "Message: `{message}`\n"
                        "Time: `{timestamp}`",
            
            "performance_update": "📊 **Performance Update**\n"
                                "Return: `{return_pct:.2f}%`\n"
                                "Sharpe: `{sharpe:.2f}`\n"
                                "Drawdown: `{drawdown:.2f}%`\n"
                                "Trades: `{trade_count}`\n"
                                "Time: `{timestamp}`",
            
            "emergency_stop": "🚨 **EMERGENCY STOP ACTIVATED**\n"
                            "Trigger: `{trigger}`\n"
                            "Reason: `{reason}`\n"
                            "All positions liquidated\n"
                            "Time: `{timestamp}`",
            
            "agent_error": "❌ **Agent Error**\n"
                         "Agent: `{agent_id}`\n"
                         "Error: `{error}`\n"
                         "Time: `{timestamp}`",
            
            "strategy_deployed": "🎯 **Strategy Deployed**\n"
                               "Strategy: `{strategy_name}`\n"
                               "Mode: `{mode}`\n"
                               "Capital: `${capital:,.2f}`\n"
                               "Time: `{timestamp}`"
        }
    
    async def start(self):
        """Start the Telegram notifier."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        self.is_running = True
        
        # Test connection
        if await self._test_connection():
            logger.info("Telegram notifier started successfully")
            await self.send_system_notification("system_start", {
                "environment": "Production Simulation",
                "agent_count": 16,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        else:
            logger.error("Failed to start Telegram notifier")
            self.is_running = False
    
    async def stop(self):
        """Stop the Telegram notifier."""
        if self.is_running:
            await self.send_system_notification("system_stop", {
                "reason": "Normal shutdown",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
        if self.session:
            await self.session.close()
        
        self.is_running = False
        logger.info("Telegram notifier stopped")
    
    async def _test_connection(self) -> bool:
        """Test Telegram bot connection."""
        try:
            url = f"{self.config.api_base}/bot{self.config.bot_token}/getMe"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("ok"):
                        bot_info = data.get("result", {})
                        logger.info(f"Connected to Telegram bot: {bot_info.get('username', 'Unknown')}")
                        return True
                
                logger.error(f"Telegram API error: {response.status}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to test Telegram connection: {e}")
            return False
    
    async def send_message(self, message: TelegramMessage) -> bool:
        """Send a message to Telegram."""
        if not self.is_running or not self.session:
            logger.warning("Telegram notifier not running")
            return False
        
        try:
            url = f"{self.config.api_base}/bot{self.config.bot_token}/sendMessage"
            
            # Add emoji based on level
            emoji_map = {
                NotificationLevel.INFO: "ℹ️",
                NotificationLevel.WARNING: "⚠️",
                NotificationLevel.ERROR: "❌",
                NotificationLevel.CRITICAL: "🚨"
            }
            
            text = f"{emoji_map.get(message.level, '')} {message.text}"
            
            payload = {
                "chat_id": self.config.chat_id,
                "text": text,
                "parse_mode": message.parse_mode,
                "disable_notification": message.disable_notification
            }
            
            for attempt in range(self.config.max_retries):
                try:
                    async with self.session.post(url, json=payload) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get("ok"):
                                return True
                            else:
                                logger.error(f"Telegram API error: {data}")
                        else:
                            logger.error(f"HTTP error {response.status}")
                
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(1)
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    async def send_trade_notification(self, trade_message: TradeMessage) -> bool:
        """Send trade execution notification."""
        text = self.templates["trade_executed"].format(
            symbol=trade_message.symbol,
            action=trade_message.action.upper(),
            quantity=trade_message.quantity,
            price=trade_message.price or 0.0,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        message = TelegramMessage(
            text=text,
            level=NotificationLevel.INFO
        )
        
        return await self.send_message(message)
    
    async def send_risk_alert(self, risk_message: RiskMessage) -> bool:
        """Send risk management alert."""
        level = NotificationLevel.WARNING
        if risk_message.severity == "high":
            level = NotificationLevel.ERROR
        elif risk_message.severity == "critical":
            level = NotificationLevel.CRITICAL
        
        text = self.templates["risk_alert"].format(
            risk_type=risk_message.risk_type,
            level=risk_message.severity.upper(),
            message=risk_message.message,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        message = TelegramMessage(
            text=text,
            level=level,
            disable_notification=(level == NotificationLevel.INFO)
        )
        
        return await self.send_message(message)
    
    async def send_performance_update(self, performance_data: Dict[str, Any]) -> bool:
        """Send performance update notification."""
        text = self.templates["performance_update"].format(
            return_pct=performance_data.get("return_pct", 0.0),
            sharpe=performance_data.get("sharpe", 0.0),
            drawdown=performance_data.get("drawdown", 0.0),
            trade_count=performance_data.get("trade_count", 0),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        message = TelegramMessage(
            text=text,
            level=NotificationLevel.INFO
        )
        
        return await self.send_message(message)
    
    async def send_emergency_alert(self, control_message: ControlMessage) -> bool:
        """Send emergency stop alert."""
        text = self.templates["emergency_stop"].format(
            trigger=control_message.control_type,
            reason=control_message.message,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        message = TelegramMessage(
            text=text,
            level=NotificationLevel.CRITICAL,
            disable_notification=False  # Always notify for emergencies
        )
        
        return await self.send_message(message)
    
    async def send_system_notification(self, template_name: str, data: Dict[str, Any]) -> bool:
        """Send system notification using template."""
        if template_name not in self.templates:
            logger.error(f"Unknown template: {template_name}")
            return False
        
        try:
            text = self.templates[template_name].format(**data)
            
            level = NotificationLevel.INFO
            if "error" in template_name or "emergency" in template_name:
                level = NotificationLevel.CRITICAL
            elif "alert" in template_name:
                level = NotificationLevel.WARNING
            
            message = TelegramMessage(
                text=text,
                level=level
            )
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Failed to send system notification: {e}")
            return False
    
    async def send_custom_message(self, text: str, level: NotificationLevel = NotificationLevel.INFO) -> bool:
        """Send custom message."""
        message = TelegramMessage(
            text=text,
            level=level
        )
        
        return await self.send_message(message)
