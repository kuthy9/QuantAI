"""
Pydantic models for the QuantAI REST API.

Defines request/response models for all API endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Health and Status Models
class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="System health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    version: str = Field(..., description="System version")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional health details")


class SystemStatusResponse(BaseModel):
    """System status response model."""
    runtime_status: str = Field(..., description="Runtime status")
    total_agents: int = Field(..., description="Total number of agents")
    agents_by_role: Dict[str, int] = Field(..., description="Agent count by role")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    memory_usage: Dict[str, Any] = Field(default_factory=dict, description="Memory usage metrics")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    last_updated: datetime = Field(..., description="Last update timestamp")


# Strategy Models
class StrategyInfo(BaseModel):
    """Basic strategy information model."""
    strategy_id: str = Field(..., description="Unique strategy identifier")
    name: str = Field(..., description="Strategy name")
    status: str = Field(..., description="Strategy status")
    performance: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    created_at: datetime = Field(..., description="Strategy creation timestamp")
    last_updated: datetime = Field(..., description="Last update timestamp")


class StrategyDetails(BaseModel):
    """Detailed strategy information model."""
    strategy_id: str = Field(..., description="Unique strategy identifier")
    name: str = Field(..., description="Strategy name")
    description: str = Field(..., description="Strategy description")
    status: str = Field(..., description="Strategy status")
    code: str = Field(..., description="Strategy code")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    performance: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    risk_metrics: Dict[str, float] = Field(default_factory=dict, description="Risk metrics")
    created_at: datetime = Field(..., description="Strategy creation timestamp")
    last_updated: datetime = Field(..., description="Last update timestamp")


class StrategyPerformance(BaseModel):
    """Strategy performance metrics model."""
    strategy_id: str = Field(..., description="Strategy identifier")
    name: str = Field(..., description="Strategy name")
    annual_return: float = Field(..., description="Annualized return")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    win_rate: float = Field(..., description="Win rate")
    total_trades: int = Field(..., description="Total number of trades")
    last_updated: datetime = Field(..., description="Last update timestamp")


# Risk Models
class RiskStatusResponse(BaseModel):
    """Risk status response model."""
    portfolio_var: float = Field(..., description="Portfolio Value at Risk")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    leverage: float = Field(..., description="Portfolio leverage")
    exposure: Dict[str, float] = Field(default_factory=dict, description="Exposure by asset class")
    risk_limits: Dict[str, float] = Field(default_factory=dict, description="Risk limits")
    alerts_count: int = Field(..., description="Number of active risk alerts")
    last_updated: datetime = Field(..., description="Last update timestamp")


class RiskAlert(BaseModel):
    """Risk alert model."""
    alert_id: str = Field(..., description="Alert identifier")
    type: str = Field(..., description="Alert type")
    severity: str = Field(..., description="Alert severity")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    metric: str = Field(..., description="Risk metric that triggered alert")
    value: float = Field(..., description="Current metric value")
    threshold: float = Field(..., description="Alert threshold")
    timestamp: datetime = Field(..., description="Alert timestamp")


# Performance Models
class PortfolioPerformance(BaseModel):
    """Portfolio performance model."""
    total_value: float = Field(..., description="Total portfolio value")
    daily_pnl: float = Field(..., description="Daily P&L")
    total_return: float = Field(..., description="Total return")
    annual_return: float = Field(..., description="Annualized return")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    volatility: float = Field(..., description="Portfolio volatility")
    beta: float = Field(..., description="Portfolio beta")
    last_updated: datetime = Field(..., description="Last update timestamp")


# Emergency Control Models
class EmergencyStopRequest(BaseModel):
    """Emergency stop request model."""
    reason: str = Field(..., description="Reason for emergency stop")


class LiquidationRequest(BaseModel):
    """Liquidation request model."""
    reason: str = Field(..., description="Reason for liquidation")
    symbols: Optional[List[str]] = Field(None, description="Specific symbols to liquidate (all if None)")


class EmergencyResetRequest(BaseModel):
    """Emergency reset request model."""
    authorized_by: str = Field(..., description="Person authorizing the reset")


# Configuration Models
class SystemConfig(BaseModel):
    """System configuration model."""
    trading_enabled: bool = Field(..., description="Whether trading is enabled")
    risk_limits: Dict[str, float] = Field(..., description="Risk limits configuration")
    strategy_limits: Dict[str, Any] = Field(..., description="Strategy limits configuration")
    data_sources: List[str] = Field(..., description="Enabled data sources")
    notification_settings: Dict[str, Any] = Field(..., description="Notification settings")
    last_updated: datetime = Field(..., description="Last configuration update")


class SystemConfigUpdate(BaseModel):
    """System configuration update model."""
    trading_enabled: Optional[bool] = Field(None, description="Whether trading is enabled")
    risk_limits: Optional[Dict[str, float]] = Field(None, description="Risk limits configuration")
    strategy_limits: Optional[Dict[str, Any]] = Field(None, description="Strategy limits configuration")
    data_sources: Optional[List[str]] = Field(None, description="Enabled data sources")
    notification_settings: Optional[Dict[str, Any]] = Field(None, description="Notification settings")


# Agent Models
class AgentInfo(BaseModel):
    """Agent information model."""
    agent_id: str = Field(..., description="Agent identifier")
    role: str = Field(..., description="Agent role")
    status: str = Field(..., description="Agent status")
    capabilities: List[str] = Field(..., description="Agent capabilities")
    last_heartbeat: datetime = Field(..., description="Last heartbeat timestamp")


class AgentDetails(BaseModel):
    """Detailed agent information model."""
    agent_id: str = Field(..., description="Agent identifier")
    role: str = Field(..., description="Agent role")
    status: str = Field(..., description="Agent status")
    capabilities: List[str] = Field(..., description="Agent capabilities")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Agent configuration")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Agent performance metrics")
    last_heartbeat: datetime = Field(..., description="Last heartbeat timestamp")
    created_at: datetime = Field(..., description="Agent creation timestamp")


# Dashboard Models
class DashboardData(BaseModel):
    """Dashboard data model."""
    timestamp: datetime = Field(..., description="Data timestamp")
    system_metrics: Dict[str, Any] = Field(default_factory=dict, description="System metrics")
    performance_data: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict, description="Performance data")
    risk_metrics: Dict[str, Any] = Field(default_factory=dict, description="Risk metrics")
    strategy_status: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Strategy status")
    alerts: List[Dict[str, Any]] = Field(default_factory=list, description="Recent alerts")
    trade_history: List[Dict[str, Any]] = Field(default_factory=list, description="Recent trades")


# Authentication Models
class LoginRequest(BaseModel):
    """Login request model."""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class LoginResponse(BaseModel):
    """Login response model."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user_info: Dict[str, Any] = Field(..., description="User information")


class UserInfo(BaseModel):
    """User information model."""
    user_id: str = Field(..., description="User identifier")
    username: str = Field(..., description="Username")
    role: str = Field(..., description="User role")
    permissions: List[str] = Field(..., description="User permissions")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")


# Trade Models
class TradeInfo(BaseModel):
    """Trade information model."""
    trade_id: str = Field(..., description="Trade identifier")
    strategy_id: str = Field(..., description="Strategy identifier")
    symbol: str = Field(..., description="Trading symbol")
    action: str = Field(..., description="Trade action (BUY/SELL)")
    quantity: float = Field(..., description="Trade quantity")
    price: float = Field(..., description="Trade price")
    timestamp: datetime = Field(..., description="Trade timestamp")
    status: str = Field(..., description="Trade status")
    pnl: Optional[float] = Field(None, description="Trade P&L")


# Position Models
class PositionInfo(BaseModel):
    """Position information model."""
    symbol: str = Field(..., description="Position symbol")
    quantity: float = Field(..., description="Position quantity")
    average_price: float = Field(..., description="Average entry price")
    current_price: float = Field(..., description="Current market price")
    market_value: float = Field(..., description="Current market value")
    unrealized_pnl: float = Field(..., description="Unrealized P&L")
    realized_pnl: float = Field(..., description="Realized P&L")
    last_updated: datetime = Field(..., description="Last update timestamp")


# Market Data Models
class MarketDataInfo(BaseModel):
    """Market data information model."""
    symbol: str = Field(..., description="Symbol")
    price: float = Field(..., description="Current price")
    change: float = Field(..., description="Price change")
    change_percent: float = Field(..., description="Price change percentage")
    volume: float = Field(..., description="Trading volume")
    timestamp: datetime = Field(..., description="Data timestamp")


# Backtest Models
class BacktestRequest(BaseModel):
    """Backtest request model."""
    strategy_id: str = Field(..., description="Strategy to backtest")
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    initial_capital: float = Field(default=100000.0, description="Initial capital")
    benchmark: str = Field(default="SPY", description="Benchmark symbol")


class BacktestResult(BaseModel):
    """Backtest result model."""
    backtest_id: str = Field(..., description="Backtest identifier")
    strategy_id: str = Field(..., description="Strategy identifier")
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    initial_capital: float = Field(..., description="Initial capital")
    final_value: float = Field(..., description="Final portfolio value")
    total_return: float = Field(..., description="Total return")
    annual_return: float = Field(..., description="Annualized return")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    win_rate: float = Field(..., description="Win rate")
    total_trades: int = Field(..., description="Total number of trades")
    completed_at: datetime = Field(..., description="Backtest completion timestamp")


# Error Models
class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


# Pagination Models
class PaginationParams(BaseModel):
    """Pagination parameters model."""
    page: int = Field(default=1, ge=1, description="Page number")
    size: int = Field(default=50, ge=1, le=1000, description="Page size")


class PaginatedResponse(BaseModel):
    """Paginated response model."""
    items: List[Any] = Field(..., description="Response items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    size: int = Field(..., description="Page size")
    pages: int = Field(..., description="Total number of pages")


# WebSocket Models
class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")


class SubscriptionRequest(BaseModel):
    """WebSocket subscription request model."""
    topics: List[str] = Field(..., description="Topics to subscribe to")
    filters: Optional[Dict[str, Any]] = Field(None, description="Subscription filters")
