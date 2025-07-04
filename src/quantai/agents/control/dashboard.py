"""
Dashboard Agent (V0) for the QuantAI system.

This agent provides web-based monitoring, control interfaces, and
real-time visualization of the entire trading system.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from autogen_core import MessageContext
from autogen_core.models import ChatCompletionClient
from loguru import logger

from ...core.base import AgentCapability, AgentRole, ModelCapableAgent
from ...core.messages import ControlMessage, MessageType, QuantMessage


class DashboardAgent(ModelCapableAgent):
    """
    Dashboard Agent (V0) - Web-based monitoring and control interface.
    
    Capabilities:
    - Real-time system monitoring and visualization
    - Interactive control panels for manual intervention
    - Performance metrics and analytics dashboards
    - Risk monitoring and alert management
    - Strategy performance tracking
    - System health and status monitoring
    """
    
    def __init__(
        self,
        model_client: ChatCompletionClient,
        dashboard_port: int = 8501,
        update_interval_seconds: int = 5,
        max_history_points: int = 1000,
        **kwargs
    ):
        super().__init__(
            role=AgentRole.DASHBOARD,
            capabilities=[
                AgentCapability.VISUALIZATION,
                AgentCapability.MONITORING,
            ],
            model_client=model_client,
            system_message=self._get_system_message(),
            **kwargs
        )
        
        self.dashboard_port = dashboard_port
        self.update_interval_seconds = update_interval_seconds
        self.max_history_points = max_history_points
        
        # Dashboard data storage
        self._system_metrics: Dict[str, Any] = {}
        self._performance_data: Dict[str, List[Dict[str, Any]]] = {}
        self._risk_metrics: Dict[str, Any] = {}
        self._strategy_status: Dict[str, Dict[str, Any]] = {}
        self._alerts: List[Dict[str, Any]] = []
        self._trade_history: List[Dict[str, Any]] = []
        
        # Dashboard state
        self._dashboard_active = False
        self._connected_clients: Set[str] = set()
        self._last_update: Optional[datetime] = None
        
        # Dashboard configuration
        self._dashboard_config = {
            "theme": "dark",
            "auto_refresh": True,
            "refresh_interval": self.update_interval_seconds,
            "charts": {
                "performance": True,
                "risk": True,
                "positions": True,
                "trades": True,
                "system": True,
            },
            "alerts": {
                "enabled": True,
                "sound": False,
                "popup": True,
            }
        }
    
    def _get_system_message(self) -> str:
        return """You are a Dashboard Agent responsible for providing comprehensive monitoring and control interfaces for the trading system.

Your responsibilities:
1. Provide real-time system monitoring and visualization
2. Create interactive control panels for manual intervention
3. Display performance metrics and analytics
4. Monitor risk metrics and generate alerts
5. Track strategy performance and status
6. Maintain system health dashboards

Dashboard Components:

1. System Overview
   - Overall system health and status
   - Active agents and their states
   - Resource utilization metrics
   - Network connectivity status
   - Error rates and performance metrics

2. Trading Dashboard
   - Live portfolio performance
   - Position monitoring and P&L
   - Trade execution status
   - Order book and market data
   - Strategy performance comparison

3. Risk Management
   - Real-time risk metrics (VaR, drawdown, leverage)
   - Risk limit monitoring and alerts
   - Exposure analysis by asset/sector
   - Correlation matrices and heat maps
   - Stress testing results

4. Strategy Monitoring
   - Individual strategy performance
   - Strategy allocation and weights
   - Signal strength and confidence
   - Backtest vs live performance
   - Strategy lifecycle status

5. Control Panels
   - Manual trading controls
   - Strategy start/stop/pause controls
   - Risk limit adjustments
   - Emergency stop and kill switch
   - System configuration management

6. Analytics and Reporting
   - Performance attribution analysis
   - Risk-adjusted return metrics
   - Benchmark comparisons
   - Historical trend analysis
   - Custom report generation

Visualization Features:
- Real-time charts and graphs
- Interactive data exploration
- Customizable dashboards
- Alert and notification system
- Export and sharing capabilities

User Interface Design:
- Intuitive and responsive design
- Role-based access control
- Mobile-friendly interface
- Dark/light theme options
- Customizable layouts

Guidelines:
- Prioritize critical information visibility
- Ensure real-time data accuracy
- Implement proper access controls
- Maintain responsive performance
- Provide clear visual indicators
- Enable quick decision making

Focus on creating an intuitive, comprehensive interface that enables effective monitoring and control of the entire trading system."""
    
    async def process_message(
        self, 
        message: QuantMessage, 
        ctx: MessageContext
    ) -> Optional[QuantMessage]:
        """Process dashboard requests and system updates."""
        
        if isinstance(message, ControlMessage):
            if message.command == "get_dashboard_data":
                # Return current dashboard data
                dashboard_data = await self._get_dashboard_data()
                
                response = ControlMessage(
                    message_type=MessageType.CONTROL,
                    sender_id=self.agent_id,
                    command="dashboard_data_response",
                    parameters={"dashboard_data": dashboard_data},
                    session_id=message.session_id,
                    correlation_id=message.correlation_id,
                )
                
                return response
                
            elif message.command == "update_dashboard_config":
                # Update dashboard configuration
                new_config = message.parameters.get("config", {})
                await self._update_dashboard_config(new_config)
                
                response = ControlMessage(
                    message_type=MessageType.CONTROL,
                    sender_id=self.agent_id,
                    command="config_updated",
                    parameters={"status": "success", "config": self._dashboard_config},
                    session_id=message.session_id,
                    correlation_id=message.correlation_id,
                )
                
                return response
                
            elif message.command == "add_alert":
                # Add new alert to dashboard
                alert = message.parameters.get("alert", {})
                await self._add_alert(alert)
                
                response = ControlMessage(
                    message_type=MessageType.CONTROL,
                    sender_id=self.agent_id,
                    command="alert_added",
                    parameters={"status": "success", "alert_id": alert.get("id")},
                    session_id=message.session_id,
                    correlation_id=message.correlation_id,
                )
                
                return response
        
        # Auto-update dashboard data from other messages
        await self._process_system_update(message)
        
        return None
    
    async def _get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": self._system_metrics,
            "performance_data": self._performance_data,
            "risk_metrics": self._risk_metrics,
            "strategy_status": self._strategy_status,
            "alerts": self._alerts[-50:],  # Last 50 alerts
            "trade_history": self._trade_history[-100:],  # Last 100 trades
            "dashboard_config": self._dashboard_config,
            "system_status": await self._get_system_status(),
        }
    
    async def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        
        return {
            "uptime": self._calculate_uptime(),
            "agents_active": len(self._strategy_status),
            "emergency_active": False,  # Would check with kill switch agent
            "trading_active": True,     # Would check with execution agents
            "data_feeds_active": True,  # Would check with data agents
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "health_score": self._calculate_health_score(),
        }
    
    def _calculate_uptime(self) -> str:
        """Calculate system uptime."""
        # This would track actual system start time
        uptime_seconds = 3600  # Mock 1 hour uptime
        
        hours = uptime_seconds // 3600
        minutes = (uptime_seconds % 3600) // 60
        
        return f"{hours}h {minutes}m"
    
    def _calculate_health_score(self) -> float:
        """Calculate overall system health score."""
        # This would aggregate various health metrics
        # For now, return a mock score
        return 0.95
    
    async def _process_system_update(self, message: QuantMessage):
        """Process system updates for dashboard display."""
        
        # Update system metrics based on message type
        if hasattr(message, 'performance_metrics'):
            await self._update_performance_data(message)
        
        if hasattr(message, 'risk_metrics'):
            await self._update_risk_data(message)
        
        if hasattr(message, 'strategy_id'):
            await self._update_strategy_status(message)
        
        # Update last update timestamp
        self._last_update = datetime.now()
    
    async def _update_performance_data(self, message: QuantMessage):
        """Update performance data from message."""
        
        if hasattr(message, 'performance_metrics'):
            timestamp = datetime.now()
            
            performance_point = {
                "timestamp": timestamp.isoformat(),
                "metrics": message.performance_metrics,
                "source": message.sender_id,
            }
            
            # Add to performance history
            if "portfolio" not in self._performance_data:
                self._performance_data["portfolio"] = []
            
            self._performance_data["portfolio"].append(performance_point)
            
            # Limit history size
            if len(self._performance_data["portfolio"]) > self.max_history_points:
                self._performance_data["portfolio"].pop(0)
    
    async def _update_risk_data(self, message: QuantMessage):
        """Update risk metrics from message."""
        
        if hasattr(message, 'risk_metrics'):
            self._risk_metrics.update({
                "timestamp": datetime.now().isoformat(),
                "metrics": message.risk_metrics,
                "source": message.sender_id,
            })
            
            # Check for risk alerts
            await self._check_risk_alerts(message.risk_metrics)
    
    async def _update_strategy_status(self, message: QuantMessage):
        """Update strategy status from message."""
        
        if hasattr(message, 'strategy_id'):
            strategy_id = message.strategy_id
            
            if strategy_id not in self._strategy_status:
                self._strategy_status[strategy_id] = {}
            
            self._strategy_status[strategy_id].update({
                "last_update": datetime.now().isoformat(),
                "status": getattr(message, 'status', 'active'),
                "performance": getattr(message, 'performance_metrics', {}),
                "source": message.sender_id,
            })
    
    async def _check_risk_alerts(self, risk_metrics: Dict[str, Any]):
        """Check risk metrics for alert conditions."""
        
        alerts_generated = []
        
        # Check VaR threshold
        var_95 = risk_metrics.get("var_95", 0)
        if var_95 > 0.02:  # 2% VaR threshold
            alert = {
                "id": f"risk_var_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "type": "risk",
                "severity": "warning",
                "title": "VaR Threshold Exceeded",
                "message": f"95% VaR is {var_95:.2%}, above 2% threshold",
                "timestamp": datetime.now().isoformat(),
                "metric": "var_95",
                "value": var_95,
                "threshold": 0.02,
            }
            alerts_generated.append(alert)
        
        # Check drawdown
        max_drawdown = risk_metrics.get("max_drawdown", 0)
        if max_drawdown > 0.15:  # 15% drawdown threshold
            alert = {
                "id": f"risk_dd_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "type": "risk",
                "severity": "critical",
                "title": "High Drawdown Alert",
                "message": f"Maximum drawdown is {max_drawdown:.2%}, above 15% threshold",
                "timestamp": datetime.now().isoformat(),
                "metric": "max_drawdown",
                "value": max_drawdown,
                "threshold": 0.15,
            }
            alerts_generated.append(alert)
        
        # Add alerts to dashboard
        for alert in alerts_generated:
            await self._add_alert(alert)
    
    async def _add_alert(self, alert: Dict[str, Any]):
        """Add alert to dashboard."""
        
        # Ensure alert has required fields
        if "id" not in alert:
            alert["id"] = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if "timestamp" not in alert:
            alert["timestamp"] = datetime.now().isoformat()
        
        if "severity" not in alert:
            alert["severity"] = "info"
        
        # Add to alerts list
        self._alerts.append(alert)
        
        # Limit alerts history
        if len(self._alerts) > 1000:
            self._alerts.pop(0)
        
        logger.info(f"Dashboard alert added: {alert['title']}")
    
    async def _update_dashboard_config(self, new_config: Dict[str, Any]):
        """Update dashboard configuration."""
        
        # Merge new configuration
        self._dashboard_config.update(new_config)
        
        logger.info(f"Dashboard configuration updated: {new_config}")
    
    async def start_dashboard_server(self):
        """Start the dashboard web server."""
        
        logger.info(f"Starting dashboard server on port {self.dashboard_port}")
        
        # This would start the actual web server (Streamlit, FastAPI, etc.)
        # For now, just mark as active
        self._dashboard_active = True
        
        # Start periodic updates
        asyncio.create_task(self._periodic_dashboard_update())
        
        logger.info("Dashboard server started successfully")
    
    async def _periodic_dashboard_update(self):
        """Periodically update dashboard data."""
        
        while self._dashboard_active:
            try:
                # Update system metrics
                await self._update_system_metrics()
                
                # Clean old data
                await self._cleanup_old_data()
                
                # Wait for next update
                await asyncio.sleep(self.update_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in periodic dashboard update: {e}")
                await asyncio.sleep(30)  # Wait 30 seconds before retrying
    
    async def _update_system_metrics(self):
        """Update system-level metrics."""
        
        current_time = datetime.now()
        
        # Mock system metrics
        self._system_metrics.update({
            "timestamp": current_time.isoformat(),
            "cpu_usage": 45.2,
            "memory_usage": 62.8,
            "disk_usage": 34.1,
            "network_io": {
                "bytes_sent": 1024000,
                "bytes_received": 2048000,
            },
            "agent_count": len(self._strategy_status),
            "active_strategies": sum(
                1 for status in self._strategy_status.values()
                if status.get("status") == "active"
            ),
            "total_alerts": len(self._alerts),
            "unread_alerts": sum(
                1 for alert in self._alerts
                if not alert.get("read", False)
            ),
        })
    
    async def _cleanup_old_data(self):
        """Clean up old dashboard data."""
        
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Clean old alerts
        self._alerts = [
            alert for alert in self._alerts
            if datetime.fromisoformat(alert["timestamp"]) > cutoff_time
        ]
        
        # Clean old trade history
        self._trade_history = [
            trade for trade in self._trade_history
            if datetime.fromisoformat(trade.get("timestamp", "1970-01-01")) > cutoff_time
        ]
    
    async def generate_dashboard_report(self, report_type: str = "daily") -> Dict[str, Any]:
        """Generate dashboard report."""
        
        if report_type == "daily":
            return await self._generate_daily_report()
        elif report_type == "weekly":
            return await self._generate_weekly_report()
        elif report_type == "monthly":
            return await self._generate_monthly_report()
        else:
            return {"error": f"Unknown report type: {report_type}"}
    
    async def _generate_daily_report(self) -> Dict[str, Any]:
        """Generate daily performance report."""
        
        today = datetime.now().date()
        
        # Filter today's data
        today_performance = [
            point for point in self._performance_data.get("portfolio", [])
            if datetime.fromisoformat(point["timestamp"]).date() == today
        ]
        
        today_alerts = [
            alert for alert in self._alerts
            if datetime.fromisoformat(alert["timestamp"]).date() == today
        ]
        
        today_trades = [
            trade for trade in self._trade_history
            if datetime.fromisoformat(trade.get("timestamp", "1970-01-01")).date() == today
        ]
        
        return {
            "report_type": "daily",
            "date": today.isoformat(),
            "generated_at": datetime.now().isoformat(),
            "performance_summary": {
                "data_points": len(today_performance),
                "latest_metrics": today_performance[-1]["metrics"] if today_performance else {},
            },
            "alerts_summary": {
                "total_alerts": len(today_alerts),
                "critical_alerts": sum(1 for a in today_alerts if a.get("severity") == "critical"),
                "warning_alerts": sum(1 for a in today_alerts if a.get("severity") == "warning"),
            },
            "trading_summary": {
                "total_trades": len(today_trades),
                "trade_volume": sum(trade.get("quantity", 0) for trade in today_trades),
            },
            "system_summary": {
                "uptime": self._calculate_uptime(),
                "health_score": self._calculate_health_score(),
                "active_strategies": len(self._strategy_status),
            }
        }
    
    async def _generate_weekly_report(self) -> Dict[str, Any]:
        """Generate weekly performance report."""
        # Similar to daily but for week period
        return {"report_type": "weekly", "status": "not_implemented"}
    
    async def _generate_monthly_report(self) -> Dict[str, Any]:
        """Generate monthly performance report."""
        # Similar to daily but for month period
        return {"report_type": "monthly", "status": "not_implemented"}
    
    async def get_dashboard_status(self) -> Dict[str, Any]:
        """Get dashboard status and statistics."""
        
        return {
            "dashboard_active": self._dashboard_active,
            "port": self.dashboard_port,
            "connected_clients": len(self._connected_clients),
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "data_points": {
                "performance": sum(len(data) for data in self._performance_data.values()),
                "alerts": len(self._alerts),
                "trades": len(self._trade_history),
                "strategies": len(self._strategy_status),
            },
            "configuration": self._dashboard_config,
            "update_interval": self.update_interval_seconds,
        }
    
    async def export_dashboard_data(self, format: str = "json") -> Dict[str, Any]:
        """Export dashboard data in specified format."""
        
        dashboard_data = await self._get_dashboard_data()
        
        if format == "json":
            return {
                "format": "json",
                "data": dashboard_data,
                "exported_at": datetime.now().isoformat(),
            }
        else:
            return {"error": f"Unsupported export format: {format}"}
    
    async def stop_dashboard_server(self):
        """Stop the dashboard server."""
        
        logger.info("Stopping dashboard server")
        
        self._dashboard_active = False
        self._connected_clients.clear()
        
        logger.info("Dashboard server stopped")
