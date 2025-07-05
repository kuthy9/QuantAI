"""
Message protocols for inter-agent communication in the QuantAI system.

This module defines the standardized message types used for communication
between agents in the multi-agent financial system, following AutoGen's
message handling patterns.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum
import uuid

from pydantic import BaseModel, Field

# Import financial products support
try:
    from .financial_products import AssetType, ProductType
except ImportError:
    # Fallback if financial_products module is not available
    from enum import Enum

    class AssetType(str, Enum):
        EQUITY = "equity"
        FUTURES = "futures"
        OPTIONS = "options"
        FOREX = "forex"
        COMMODITY = "commodity"
        BOND = "bond"
        CRYPTO = "crypto"
        ETF = "etf"
        REIT = "reit"

    class ProductType(str, Enum):
        STOCK = "stock"
        ETF = "etf"
        REIT = "reit"
        EQUITY_FUTURES = "equity_futures"
        COMMODITY_FUTURES = "commodity_futures"
        BOND_FUTURES = "bond_futures"
        CURRENCY_FUTURES = "currency_futures"
        EQUITY_OPTIONS = "equity_options"
        INDEX_OPTIONS = "index_options"
        COMMODITY_OPTIONS = "commodity_options"
        FOREX_SPOT = "forex_spot"
        CRYPTO_SPOT = "crypto_spot"


class MessageType(str, Enum):
    """Types of messages in the system."""
    DATA_REQUEST = "data_request"
    DATA_RESPONSE = "data_response"
    STRATEGY_GENERATION = "strategy_generation"
    STRATEGY_REQUEST = "strategy_request"
    STRATEGY_CODE = "strategy_code"
    STRATEGY_VALIDATION = "strategy_validation"
    RISK_ASSESSMENT = "risk_assessment"
    TRADE_SIGNAL = "trade_signal"
    TRADE_REQUEST = "trade_request"
    TRADE_RESPONSE = "trade_response"
    EXECUTION_RESULT = "execution_result"
    BACKTEST_REQUEST = "backtest_request"
    BACKTEST_RESPONSE = "backtest_response"
    BACKTEST_RESULT = "backtest_result"
    PROFITABILITY_ANALYSIS = "profitability_analysis"
    FEEDBACK = "feedback"
    FEEDBACK_RESPONSE = "feedback_response"
    MEMORY_UPDATE = "memory_update"
    KILL_SWITCH = "kill_switch"
    DEPLOYMENT_RESPONSE = "deployment_response"
    VALIDATION_RESULT = "validation_result"
    GENERAL_RESPONSE = "general_response"
    ERROR = "error"

    # Performance Attribution Message Types
    PERFORMANCE_ATTRIBUTION_REQUEST = "performance_attribution_request"
    PERFORMANCE_ATTRIBUTION_RESPONSE = "performance_attribution_response"
    PERFORMANCE_SUMMARY_REQUEST = "performance_summary_request"
    PERFORMANCE_SUMMARY_RESPONSE = "performance_summary_response"
    ROLLING_ATTRIBUTION_REQUEST = "rolling_attribution_request"
    ROLLING_ATTRIBUTION_RESPONSE = "rolling_attribution_response"
    FACTOR_ATTRIBUTION_REQUEST = "factor_attribution_request"
    FACTOR_ATTRIBUTION_RESPONSE = "factor_attribution_response"
    SECTOR_ATTRIBUTION_REQUEST = "sector_attribution_request"
    SECTOR_ATTRIBUTION_RESPONSE = "sector_attribution_response"
    RETURNS_DATA_UPDATE = "returns_data_update"
    DATA_UPDATE_CONFIRMATION = "data_update_confirmation"

    # Compliance Message Types
    COMPLIANCE_CHECK_REQUEST = "compliance_check_request"
    COMPLIANCE_CHECK_RESPONSE = "compliance_check_response"
    AUDIT_LOG_REQUEST = "audit_log_request"
    AUDIT_LOG_RESPONSE = "audit_log_response"
    TRADE_REPORT_REQUEST = "trade_report_request"
    TRADE_REPORT_RESPONSE = "trade_report_response"
    REGULATORY_REPORT_REQUEST = "regulatory_report_request"
    REGULATORY_REPORT_RESPONSE = "regulatory_report_response"
    COMPLIANCE_VIOLATION_ALERT = "compliance_violation_alert"
    COMPLIANCE_DASHBOARD_REQUEST = "compliance_dashboard_request"
    COMPLIANCE_DASHBOARD_RESPONSE = "compliance_dashboard_response"

    # Additional system messages
    TRADE_EXECUTION = "trade_execution"
    POSITION_UPDATE = "position_update"


class Priority(str, Enum):
    """Message priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class QuantMessage:
    """Base message class for all agent communications."""
    def __init__(
        self,
        message_type: MessageType,
        sender_id: str,
        message_id: str = None,
        recipient_id: Optional[str] = None,
        timestamp: datetime = None,
        priority: Priority = Priority.NORMAL,
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        metadata: Dict[str, Any] = None,
        data_payload: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ):
        self.message_type = message_type
        self.sender_id = sender_id
        self.message_id = message_id or str(uuid.uuid4())
        self.recipient_id = recipient_id
        self.timestamp = timestamp or datetime.utcnow()
        self.priority = priority
        self.session_id = session_id
        self.correlation_id = correlation_id
        self.metadata = metadata or {}
        self.data_payload = data_payload
        self.error_message = error_message


class DataMessage(QuantMessage):
    """Message for data requests and responses."""
    def __init__(
        self,
        message_type: MessageType,
        sender_id: str,
        data_type: str,  # "market", "news", "earnings", "sentiment", etc.
        symbols: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        data_payload: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        quality_score: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message_type, sender_id, **kwargs)
        self.data_type = data_type
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data_payload = data_payload
        self.source = source
        self.quality_score = quality_score


class StrategyMessage(QuantMessage):
    """Message for strategy generation and coding."""
    def __init__(
        self,
        message_type: MessageType,
        sender_id: str,
        strategy_id: str,
        strategy_name: Optional[str] = None,
        strategy_description: Optional[str] = None,
        strategy_code: Optional[str] = None,
        strategy_parameters: Optional[Dict[str, Any]] = None,
        market_regime: Optional[str] = None,
        target_assets: Optional[List[str]] = None,
        expected_return: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message_type, sender_id, **kwargs)
        self.strategy_id = strategy_id
        self.strategy_name = strategy_name
        self.strategy_description = strategy_description
        self.strategy_code = strategy_code
        self.strategy_parameters = strategy_parameters
        self.market_regime = market_regime
        self.target_assets = target_assets
        self.expected_return = expected_return
        self.max_drawdown = max_drawdown


class ValidationMessage(QuantMessage):
    """Message for strategy validation results."""
    def __init__(
        self,
        message_type: MessageType,
        sender_id: str,
        strategy_id: str,
        validation_passed: bool,
        validation_score: Optional[float] = None,
        validation_results: Optional[Dict[str, Any]] = None,
        issues_found: Optional[List[str]] = None,
        recommendations: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message_type, sender_id, **kwargs)
        self.strategy_id = strategy_id
        self.validation_passed = validation_passed
        self.validation_score = validation_score
        self.validation_results = validation_results
        self.issues_found = issues_found or []
        self.recommendations = recommendations or []


class RiskMessage(QuantMessage):
    """Message for risk assessment and monitoring."""
    def __init__(
        self,
        message_type: MessageType,
        sender_id: str,
        risk_level: str,  # "low", "medium", "high", "critical"
        risk_score: float,
        risk_factors: Optional[List[str]] = None,
        mitigation_strategies: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message_type, sender_id, **kwargs)
        self.risk_level = risk_level
        self.risk_score = risk_score
        self.risk_factors = risk_factors or []
        self.mitigation_strategies = mitigation_strategies or []


class TradeMessage(QuantMessage):
    """Message for trade signals and execution."""
    def __init__(
        self,
        message_type: MessageType,
        sender_id: str,
        trade_id: str,
        symbol: str,
        action: str,  # "BUY", "SELL", "LONG", "SHORT"
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "MARKET",  # "MARKET", "LIMIT", "STOP"
        strategy_id: Optional[str] = None,

        # Multi-product support
        asset_type: AssetType = AssetType.EQUITY,
        product_type: ProductType = ProductType.STOCK,

        # Futures specific fields
        contract_month: Optional[str] = None,
        margin_requirement: Optional[float] = None,

        # Options specific fields
        strike_price: Optional[float] = None,
        expiration_date: Optional[datetime] = None,
        option_type: Optional[str] = None,  # "CALL", "PUT"

        # Forex specific fields
        base_currency: Optional[str] = None,
        quote_currency: Optional[str] = None,

        **kwargs
    ):
        super().__init__(message_type, sender_id, **kwargs)
        self.trade_id = trade_id
        self.symbol = symbol
        self.action = action
        self.quantity = quantity
        self.price = price
        self.order_type = order_type
        self.strategy_id = strategy_id

        # Multi-product fields
        self.asset_type = asset_type
        self.product_type = product_type

        # Product-specific fields
        self.contract_month = contract_month
        self.margin_requirement = margin_requirement
        self.strike_price = strike_price
        self.expiration_date = expiration_date
        self.option_type = option_type
        self.base_currency = base_currency
        self.quote_currency = quote_currency


class BacktestMessage(QuantMessage):
    """Message for backtesting requests and results."""
    def __init__(
        self,
        message_type: MessageType,
        sender_id: str,
        backtest_id: str,
        strategy_id: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float,
        symbols: List[str],
        backtest_results: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(message_type, sender_id, **kwargs)
        self.backtest_id = backtest_id
        self.strategy_id = strategy_id
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.symbols = symbols
        self.backtest_results = backtest_results


class FeedbackMessage(QuantMessage):
    """Message for learning and feedback loops."""
    def __init__(
        self,
        message_type: MessageType,
        sender_id: str,
        strategy_id: str,
        performance_actual: Dict[str, float],
        performance_expected: Dict[str, float],
        success_factors: Optional[List[str]] = None,
        failure_factors: Optional[List[str]] = None,
        lessons_learned: Optional[List[str]] = None,
        improvement_suggestions: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message_type, sender_id, **kwargs)
        self.strategy_id = strategy_id
        self.performance_actual = performance_actual
        self.performance_expected = performance_expected
        self.success_factors = success_factors or []
        self.failure_factors = failure_factors or []
        self.lessons_learned = lessons_learned or []
        self.improvement_suggestions = improvement_suggestions or []


class MemoryMessage(QuantMessage):
    """Message for memory updates and retrieval."""
    def __init__(
        self,
        message_type: MessageType,
        sender_id: str,
        memory_type: str,  # "short_term", "long_term", "episodic"
        memory_content: Dict[str, Any],
        memory_key: Optional[str] = None,
        expiry_time: Optional[datetime] = None,
        **kwargs
    ):
        super().__init__(message_type, sender_id, **kwargs)
        self.memory_type = memory_type
        self.memory_content = memory_content
        self.memory_key = memory_key
        self.expiry_time = expiry_time


class ControlMessage(QuantMessage):
    """Message for system control and monitoring."""
    def __init__(
        self,
        message_type: MessageType,
        sender_id: str,
        control_action: str,  # "start", "stop", "pause", "resume", "kill"
        target_component: Optional[str] = None,
        control_parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(message_type, sender_id, **kwargs)
        self.control_action = control_action
        self.target_component = target_component
        self.control_parameters = control_parameters or {}


# Compliance Message Classes

class ComplianceCheckRequest(QuantMessage):
    """Request for compliance checking."""

    def __init__(
        self,
        sender_id: str,
        event_type: str,
        event_data: Dict[str, Any],
        account_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(MessageType.COMPLIANCE_CHECK_REQUEST, sender_id, **kwargs)
        self.event_type = event_type
        self.event_data = event_data
        self.account_id = account_id
        self.strategy_id = strategy_id


class ComplianceCheckResponse(QuantMessage):
    """Response to compliance check request."""

    def __init__(
        self,
        sender_id: str,
        compliance_status: str,
        violations: List[Dict[str, Any]],
        check_timestamp: str,
        **kwargs
    ):
        super().__init__(MessageType.COMPLIANCE_CHECK_RESPONSE, sender_id, **kwargs)
        self.compliance_status = compliance_status
        self.violations = violations
        self.check_timestamp = check_timestamp


class AuditLogRequest(QuantMessage):
    """Request to log an audit event."""

    def __init__(
        self,
        sender_id: str,
        event_type: str,
        description: str,
        event_data: Dict[str, Any],
        severity: str = "medium",
        user_id: Optional[str] = None,
        account_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(MessageType.AUDIT_LOG_REQUEST, sender_id, **kwargs)
        self.event_type = event_type
        self.description = description
        self.event_data = event_data
        self.severity = severity
        self.user_id = user_id
        self.account_id = account_id


class AuditLogResponse(QuantMessage):
    """Response to audit log request."""

    def __init__(
        self,
        sender_id: str,
        audit_event_id: str,
        status: str,
        timestamp: str,
        **kwargs
    ):
        super().__init__(MessageType.AUDIT_LOG_RESPONSE, sender_id, **kwargs)
        self.audit_event_id = audit_event_id
        self.status = status
        self.timestamp = timestamp


class TradeReportRequest(QuantMessage):
    """Request to report a trade execution."""

    def __init__(
        self,
        sender_id: str,
        trade_id: str,
        account_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        execution_time: str,
        strategy_id: Optional[str] = None,
        market_data: Optional[Dict[str, float]] = None,
        execution_quality: Optional[Dict[str, float]] = None,
        costs: Optional[Dict[str, float]] = None,
        regulatory_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(MessageType.TRADE_REPORT_REQUEST, sender_id, **kwargs)
        self.trade_id = trade_id
        self.account_id = account_id
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.price = price
        self.execution_time = execution_time
        self.strategy_id = strategy_id
        self.market_data = market_data
        self.execution_quality = execution_quality
        self.costs = costs
        self.regulatory_info = regulatory_info


class TradeReportResponse(QuantMessage):
    """Response to trade report request."""

    def __init__(
        self,
        sender_id: str,
        report_id: str,
        status: str,
        trade_id: str,
        total_cost: float,
        slippage: Optional[float] = None,
        **kwargs
    ):
        super().__init__(MessageType.TRADE_REPORT_RESPONSE, sender_id, **kwargs)
        self.report_id = report_id
        self.status = status
        self.trade_id = trade_id
        self.total_cost = total_cost
        self.slippage = slippage


class RegulatoryReportRequest(QuantMessage):
    """Request to generate a regulatory report."""

    def __init__(
        self,
        sender_id: str,
        template_id: str,
        start_date: str,
        end_date: str,
        **kwargs
    ):
        super().__init__(MessageType.REGULATORY_REPORT_REQUEST, sender_id, **kwargs)
        self.template_id = template_id
        self.start_date = start_date
        self.end_date = end_date


class RegulatoryReportResponse(QuantMessage):
    """Response to regulatory report request."""

    def __init__(
        self,
        sender_id: str,
        report_id: str,
        status: str,
        output_path: str,
        generated_at: str,
        **kwargs
    ):
        super().__init__(MessageType.REGULATORY_REPORT_RESPONSE, sender_id, **kwargs)
        self.report_id = report_id
        self.status = status
        self.output_path = output_path
        self.generated_at = generated_at


class ComplianceViolationAlert(QuantMessage):
    """Alert for compliance violations."""

    def __init__(
        self,
        sender_id: str,
        violation_id: str,
        rule_type: str,
        status: str,
        description: str,
        current_value: Optional[float] = None,
        threshold_value: Optional[float] = None,
        account_id: Optional[str] = None,
        symbol: Optional[str] = None,
        detected_at: str = None,
        **kwargs
    ):
        super().__init__(MessageType.COMPLIANCE_VIOLATION_ALERT, sender_id, **kwargs)
        self.violation_id = violation_id
        self.rule_type = rule_type
        self.status = status
        self.description = description
        self.current_value = current_value
        self.threshold_value = threshold_value
        self.account_id = account_id
        self.symbol = symbol
        self.detected_at = detected_at


# Message type mapping for easy access
MESSAGE_TYPES = {
    MessageType.DATA_REQUEST: DataMessage,
    MessageType.DATA_RESPONSE: DataMessage,
    MessageType.STRATEGY_GENERATION: StrategyMessage,
    MessageType.STRATEGY_CODE: StrategyMessage,
    MessageType.STRATEGY_VALIDATION: ValidationMessage,
    MessageType.RISK_ASSESSMENT: RiskMessage,
    MessageType.TRADE_SIGNAL: TradeMessage,
    MessageType.EXECUTION_RESULT: TradeMessage,
    MessageType.BACKTEST_REQUEST: BacktestMessage,
    MessageType.BACKTEST_RESULT: BacktestMessage,
    MessageType.FEEDBACK: FeedbackMessage,
    MessageType.MEMORY_UPDATE: MemoryMessage,
    MessageType.KILL_SWITCH: ControlMessage,
}
