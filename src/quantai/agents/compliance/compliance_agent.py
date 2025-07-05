"""
Compliance Agent for QuantAI Multi-Agent System.

This agent provides comprehensive regulatory compliance capabilities including:
- Real-time compliance monitoring
- Audit trail management
- Trade reporting
- Regulatory report generation
- Violation detection and remediation
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from loguru import logger

from ...core.base_agent import BaseAgent
from ...core.messages import (
    Message, MessageType, ComplianceCheckRequest, ComplianceCheckResponse,
    AuditLogRequest, AuditLogResponse, TradeReportRequest, TradeReportResponse,
    RegulatoryReportRequest, RegulatoryReportResponse, ComplianceViolationAlert
)
from ...compliance.audit_trail import AuditTrailManager, AuditEventType, AuditSeverity
from ...compliance.trade_reporting import TradeReportingEngine, TradeReportType
from ...compliance.compliance_monitor import ComplianceMonitor, ComplianceRule, ComplianceRuleType, ComplianceStatus
from ...compliance.regulatory_reporting import RegulatoryReportGenerator, RegulatoryReportTemplate, ReportingFrequency


class ComplianceAgent(BaseAgent):
    """
    Compliance Agent for regulatory compliance management.
    
    Handles all compliance-related activities including monitoring,
    reporting, and regulatory submissions.
    """
    
    def __init__(self, agent_id: str = "compliance_agent", config: Optional[Dict[str, Any]] = None):
        """Initialize Compliance Agent."""
        super().__init__(agent_id, config)
        
        self.agent_type = "compliance"
        self.capabilities = [
            "compliance_monitoring",
            "audit_trail_management", 
            "trade_reporting",
            "regulatory_reporting",
            "violation_detection",
            "compliance_dashboard"
        ]
        
        # Initialize compliance components
        self.audit_manager = AuditTrailManager(config.get('audit_config', {}))
        self.trade_reporting = TradeReportingEngine(config.get('trade_reporting_config', {}))
        self.compliance_monitor = ComplianceMonitor(config.get('compliance_monitor_config', {}))
        self.regulatory_reporting = RegulatoryReportGenerator(
            self.audit_manager,
            self.trade_reporting,
            self.compliance_monitor,
            config.get('regulatory_reporting_config', {})
        )
        
        # Set up violation callbacks
        self.compliance_monitor.add_violation_callback(self._handle_compliance_violation)
        
        # Initialize default compliance rules
        asyncio.create_task(self._setup_default_rules())
        
        logger.info(f"Compliance Agent {self.agent_id} initialized")
    
    async def _setup_default_rules(self):
        """Set up default compliance rules."""
        
        # Position limit rule
        position_rule = ComplianceRule(
            rule_id="default_position_limit",
            rule_type=ComplianceRuleType.POSITION_LIMIT,
            name="Default Position Limit",
            description="Maximum position size per symbol",
            parameters={"max_position_size": 10000},
            warning_threshold=8000,
            violation_threshold=10000,
            severity="high",
            regulatory_reference="Internal Risk Management"
        )
        await self.compliance_monitor.add_rule(position_rule)
        
        # Daily loss limit rule
        loss_rule = ComplianceRule(
            rule_id="default_daily_loss_limit",
            rule_type=ComplianceRuleType.DAILY_LOSS_LIMIT,
            name="Daily Loss Limit",
            description="Maximum daily loss per account",
            parameters={"max_daily_loss": 50000},
            violation_threshold=50000,
            severity="critical",
            regulatory_reference="Risk Management Policy"
        )
        await self.compliance_monitor.add_rule(loss_rule)
        
        # Trading hours rule
        hours_rule = ComplianceRule(
            rule_id="default_trading_hours",
            rule_type=ComplianceRuleType.TRADING_HOURS,
            name="Trading Hours Compliance",
            description="Allowed trading hours",
            parameters={
                "trading_start_time": "09:30",
                "trading_end_time": "16:00"
            },
            severity="medium",
            regulatory_reference="Market Hours Regulation"
        )
        await self.compliance_monitor.add_rule(hours_rule)
        
        logger.info("Default compliance rules configured")
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """Process incoming compliance-related messages."""
        
        try:
            if message.message_type == MessageType.COMPLIANCE_CHECK_REQUEST:
                return await self._handle_compliance_check(message)
            
            elif message.message_type == MessageType.AUDIT_LOG_REQUEST:
                return await self._handle_audit_log_request(message)
            
            elif message.message_type == MessageType.TRADE_REPORT_REQUEST:
                return await self._handle_trade_report_request(message)
            
            elif message.message_type == MessageType.REGULATORY_REPORT_REQUEST:
                return await self._handle_regulatory_report_request(message)
            
            elif message.message_type == MessageType.TRADE_EXECUTION:
                # Monitor trade executions for compliance
                await self._monitor_trade_execution(message)
                return None
            
            elif message.message_type == MessageType.POSITION_UPDATE:
                # Monitor position updates for compliance
                await self._monitor_position_update(message)
                return None
            
            else:
                logger.warning(f"Unhandled message type: {message.message_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing message {message.message_id}: {e}")
            return self._create_error_response(message, str(e))
    
    async def _handle_compliance_check(self, message: Message) -> Message:
        """Handle compliance check requests."""
        
        request_data = message.data
        event_type = request_data.get('event_type', 'general')
        event_data = request_data.get('event_data', {})
        account_id = request_data.get('account_id')
        strategy_id = request_data.get('strategy_id')
        
        # Perform compliance check
        violations = await self.compliance_monitor.check_compliance(
            event_type, event_data, account_id, strategy_id
        )
        
        # Log compliance check
        await self.audit_manager.log_event(
            event_type=AuditEventType.COMPLIANCE_CHECK,
            description=f"Compliance check performed for {event_type}",
            event_data={
                'event_type': event_type,
                'violations_found': len(violations),
                'account_id': account_id,
                'strategy_id': strategy_id
            },
            severity=AuditSeverity.MEDIUM if not violations else AuditSeverity.HIGH,
            account_id=account_id,
            agent_id=self.agent_id
        )
        
        return Message(
            message_id=f"compliance_check_response_{message.message_id}",
            message_type=MessageType.COMPLIANCE_CHECK_RESPONSE,
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            data={
                'compliance_status': 'violation' if violations else 'compliant',
                'violations': [
                    {
                        'violation_id': v.violation_id,
                        'rule_type': v.rule_type,
                        'status': v.status,
                        'description': v.description,
                        'current_value': v.current_value,
                        'threshold_value': v.threshold_value
                    }
                    for v in violations
                ],
                'check_timestamp': datetime.now().isoformat()
            },
            correlation_id=message.correlation_id
        )
    
    async def _handle_audit_log_request(self, message: Message) -> Message:
        """Handle audit log requests."""
        
        request_data = message.data
        event_type = AuditEventType(request_data.get('event_type', AuditEventType.SYSTEM_ACCESS))
        description = request_data.get('description', '')
        event_data = request_data.get('event_data', {})
        severity = AuditSeverity(request_data.get('severity', AuditSeverity.MEDIUM))
        user_id = request_data.get('user_id')
        account_id = request_data.get('account_id')
        
        # Log audit event
        audit_event = await self.audit_manager.log_event(
            event_type=event_type,
            description=description,
            event_data=event_data,
            severity=severity,
            user_id=user_id,
            agent_id=message.sender_id,
            account_id=account_id
        )
        
        return Message(
            message_id=f"audit_log_response_{message.message_id}",
            message_type=MessageType.AUDIT_LOG_RESPONSE,
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            data={
                'audit_event_id': audit_event.event_id,
                'status': 'logged',
                'timestamp': audit_event.timestamp.isoformat()
            },
            correlation_id=message.correlation_id
        )
    
    async def _handle_trade_report_request(self, message: Message) -> Message:
        """Handle trade reporting requests."""
        
        request_data = message.data
        
        # Report trade execution
        trade_report = await self.trade_reporting.report_trade(
            trade_id=request_data['trade_id'],
            account_id=request_data['account_id'],
            strategy_id=request_data.get('strategy_id', 'unknown'),
            symbol=request_data['symbol'],
            side=request_data['side'],
            quantity=request_data['quantity'],
            price=request_data['price'],
            execution_time=datetime.fromisoformat(request_data['execution_time']),
            market_data=request_data.get('market_data'),
            execution_quality=request_data.get('execution_quality'),
            costs=request_data.get('costs'),
            regulatory_info=request_data.get('regulatory_info')
        )
        
        return Message(
            message_id=f"trade_report_response_{message.message_id}",
            message_type=MessageType.TRADE_REPORT_RESPONSE,
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            data={
                'report_id': trade_report.report_id,
                'status': 'reported',
                'trade_id': trade_report.trade_id,
                'total_cost': trade_report.total_cost,
                'slippage': trade_report.slippage
            },
            correlation_id=message.correlation_id
        )
    
    async def _handle_regulatory_report_request(self, message: Message) -> Message:
        """Handle regulatory report generation requests."""
        
        request_data = message.data
        template_id = request_data.get('template_id')
        start_date = datetime.fromisoformat(request_data['start_date'])
        end_date = datetime.fromisoformat(request_data['end_date'])
        
        try:
            # Generate regulatory report
            report_result = await self.regulatory_reporting.generate_report(
                template_id, start_date, end_date
            )
            
            return Message(
                message_id=f"regulatory_report_response_{message.message_id}",
                message_type=MessageType.REGULATORY_REPORT_RESPONSE,
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                data={
                    'report_id': report_result['report_id'],
                    'status': report_result['status'],
                    'output_path': report_result['output_path'],
                    'generated_at': report_result['generated_at']
                },
                correlation_id=message.correlation_id
            )
            
        except Exception as e:
            return Message(
                message_id=f"regulatory_report_error_{message.message_id}",
                message_type=MessageType.ERROR,
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                data={
                    'error': str(e),
                    'error_type': 'regulatory_report_generation_failed'
                },
                correlation_id=message.correlation_id
            )
    
    async def _monitor_trade_execution(self, message: Message):
        """Monitor trade executions for compliance."""
        
        trade_data = message.data
        
        # Check compliance for trade execution
        event_data = {
            'symbol': trade_data.get('symbol'),
            'order_size': trade_data.get('quantity', 0),
            'position_size': trade_data.get('new_position_size', 0),
            'position_value': trade_data.get('position_value', 0),
            'portfolio_value': trade_data.get('portfolio_value', 1),
            'daily_pnl': trade_data.get('daily_pnl', 0)
        }
        
        violations = await self.compliance_monitor.check_compliance(
            'trade_execution',
            event_data,
            trade_data.get('account_id'),
            trade_data.get('strategy_id')
        )
        
        # Log trade execution
        await self.audit_manager.log_event(
            event_type=AuditEventType.TRADE_EXECUTION,
            description=f"Trade executed: {trade_data.get('symbol')} {trade_data.get('side')} {trade_data.get('quantity')}",
            event_data=trade_data,
            severity=AuditSeverity.HIGH if violations else AuditSeverity.MEDIUM,
            account_id=trade_data.get('account_id'),
            agent_id=message.sender_id
        )
    
    async def _monitor_position_update(self, message: Message):
        """Monitor position updates for compliance."""
        
        position_data = message.data
        
        # Check compliance for position update
        event_data = {
            'symbol': position_data.get('symbol'),
            'position_size': position_data.get('position_size', 0),
            'position_value': position_data.get('position_value', 0),
            'portfolio_value': position_data.get('portfolio_value', 1)
        }
        
        await self.compliance_monitor.check_compliance(
            'position_update',
            event_data,
            position_data.get('account_id'),
            position_data.get('strategy_id')
        )
    
    async def _handle_compliance_violation(self, violation):
        """Handle compliance violations by sending alerts."""
        
        # Create violation alert message
        alert_message = Message(
            message_id=f"compliance_violation_alert_{violation.violation_id}",
            message_type=MessageType.COMPLIANCE_VIOLATION_ALERT,
            sender_id=self.agent_id,
            recipient_id="risk_manager",  # Send to risk manager
            data={
                'violation_id': violation.violation_id,
                'rule_type': violation.rule_type,
                'status': violation.status,
                'description': violation.description,
                'current_value': violation.current_value,
                'threshold_value': violation.threshold_value,
                'account_id': violation.account_id,
                'symbol': violation.symbol,
                'detected_at': violation.detected_at.isoformat()
            }
        )
        
        # Send alert to message bus
        await self.send_message(alert_message)
        
        logger.warning(f"Compliance violation alert sent: {violation.violation_id}")
    
    async def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive compliance dashboard data."""
        return await self.compliance_monitor.get_compliance_dashboard()
    
    async def get_audit_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get audit trail summary for specified days."""
        
        start_date = datetime.now() - timedelta(days=days)
        events = await self.audit_manager.get_events(start_date=start_date, limit=1000)
        
        # Generate summary statistics
        event_counts = {}
        severity_counts = {}
        
        for event in events:
            event_type = event.event_type
            severity = event.severity
            
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'period_days': days,
            'total_events': len(events),
            'event_type_breakdown': event_counts,
            'severity_breakdown': severity_counts,
            'generated_at': datetime.now().isoformat()
        }
