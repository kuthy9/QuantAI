"""
Compliance Monitoring System for QuantAI Regulatory Compliance.

This module provides real-time compliance monitoring including:
- Rule-based compliance checking
- Violation detection and alerting
- Compliance metrics tracking
- Automated remediation actions
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import sqlite3
import json
from loguru import logger


class ComplianceRuleType(str, Enum):
    """Types of compliance rules."""
    POSITION_LIMIT = "position_limit"
    CONCENTRATION_LIMIT = "concentration_limit"
    LEVERAGE_LIMIT = "leverage_limit"
    TRADING_HOURS = "trading_hours"
    MARKET_ACCESS = "market_access"
    ORDER_SIZE_LIMIT = "order_size_limit"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    SECTOR_EXPOSURE = "sector_exposure"
    LIQUIDITY_REQUIREMENT = "liquidity_requirement"
    BEST_EXECUTION = "best_execution"
    TRADE_REPORTING = "trade_reporting"
    RECORD_KEEPING = "record_keeping"
    RISK_MANAGEMENT = "risk_management"
    CLIENT_SUITABILITY = "client_suitability"
    ANTI_MONEY_LAUNDERING = "anti_money_laundering"


class ComplianceStatus(str, Enum):
    """Compliance status levels."""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    CRITICAL = "critical"


@dataclass
class ComplianceRule:
    """Compliance rule definition."""
    rule_id: str
    rule_type: ComplianceRuleType
    name: str
    description: str
    
    # Rule parameters
    parameters: Dict[str, Any]
    
    # Thresholds
    warning_threshold: Optional[float] = None
    violation_threshold: Optional[float] = None
    
    # Rule configuration
    enabled: bool = True
    severity: str = "medium"
    regulatory_reference: Optional[str] = None
    
    # Metadata
    created_at: datetime = None
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_updated is None:
            self.last_updated = datetime.now()


@dataclass
class ComplianceViolation:
    """Compliance violation record."""
    violation_id: str
    rule_id: str
    rule_type: ComplianceRuleType
    status: ComplianceStatus
    
    # Violation details
    description: str
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    
    # Context
    account_id: Optional[str] = None
    strategy_id: Optional[str] = None
    trade_id: Optional[str] = None
    symbol: Optional[str] = None
    
    # Violation data
    violation_data: Dict[str, Any] = None
    
    # Resolution
    resolved: bool = False
    resolution_action: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    # Metadata
    detected_at: datetime = None
    
    def __post_init__(self):
        if self.detected_at is None:
            self.detected_at = datetime.now()
        if self.violation_data is None:
            self.violation_data = {}


class ComplianceMonitor:
    """
    Real-time compliance monitoring system.
    
    Monitors trading activities against regulatory rules,
    detects violations, and triggers appropriate responses.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize compliance monitor."""
        self.config = config or {}
        
        # Database configuration
        self.db_path = Path(self.config.get('compliance_db_path', 'data/compliance.db'))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Monitoring configuration
        self.monitoring_enabled = self.config.get('monitoring_enabled', True)
        self.real_time_alerts = self.config.get('real_time_alerts', True)
        self.auto_remediation = self.config.get('auto_remediation', False)
        
        # Rule storage
        self.rules: Dict[str, ComplianceRule] = {}
        self.active_violations: Dict[str, ComplianceViolation] = {}
        
        # Callbacks for violations
        self.violation_callbacks: List[Callable] = []
        
        # Initialize database and load rules
        self._init_database()
        asyncio.create_task(self._load_rules())
        
        logger.info("Compliance Monitor initialized")
    
    def _init_database(self):
        """Initialize compliance database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS compliance_rules (
                    rule_id TEXT PRIMARY KEY,
                    rule_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    warning_threshold REAL,
                    violation_threshold REAL,
                    enabled BOOLEAN NOT NULL,
                    severity TEXT NOT NULL,
                    regulatory_reference TEXT,
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS compliance_violations (
                    violation_id TEXT PRIMARY KEY,
                    rule_id TEXT NOT NULL,
                    rule_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    description TEXT NOT NULL,
                    current_value REAL,
                    threshold_value REAL,
                    account_id TEXT,
                    strategy_id TEXT,
                    trade_id TEXT,
                    symbol TEXT,
                    violation_data TEXT NOT NULL,
                    resolved BOOLEAN NOT NULL,
                    resolution_action TEXT,
                    resolved_at TEXT,
                    detected_at TEXT NOT NULL,
                    INDEX(rule_id),
                    INDEX(detected_at),
                    INDEX(account_id),
                    INDEX(resolved)
                )
            ''')
            
            conn.commit()
    
    async def _load_rules(self):
        """Load compliance rules from database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM compliance_rules WHERE enabled = 1")
            
            for row in cursor.fetchall():
                rule = ComplianceRule(
                    rule_id=row['rule_id'],
                    rule_type=ComplianceRuleType(row['rule_type']),
                    name=row['name'],
                    description=row['description'],
                    parameters=json.loads(row['parameters']),
                    warning_threshold=row['warning_threshold'],
                    violation_threshold=row['violation_threshold'],
                    enabled=bool(row['enabled']),
                    severity=row['severity'],
                    regulatory_reference=row['regulatory_reference'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    last_updated=datetime.fromisoformat(row['last_updated'])
                )
                self.rules[rule.rule_id] = rule
        
        logger.info(f"Loaded {len(self.rules)} compliance rules")
    
    async def add_rule(self, rule: ComplianceRule) -> bool:
        """Add a new compliance rule."""
        try:
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO compliance_rules (
                        rule_id, rule_type, name, description, parameters,
                        warning_threshold, violation_threshold, enabled, severity,
                        regulatory_reference, created_at, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    rule.rule_id, rule.rule_type, rule.name, rule.description,
                    json.dumps(rule.parameters), rule.warning_threshold,
                    rule.violation_threshold, rule.enabled, rule.severity,
                    rule.regulatory_reference, rule.created_at.isoformat(),
                    rule.last_updated.isoformat()
                ))
                conn.commit()
            
            # Add to memory
            self.rules[rule.rule_id] = rule
            
            logger.info(f"Added compliance rule: {rule.rule_id} ({rule.rule_type})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding compliance rule {rule.rule_id}: {e}")
            return False
    
    async def check_compliance(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        account_id: Optional[str] = None,
        strategy_id: Optional[str] = None
    ) -> List[ComplianceViolation]:
        """Check compliance against all applicable rules."""
        
        if not self.monitoring_enabled:
            return []
        
        violations = []
        
        # Check each rule
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            try:
                violation = await self._check_rule(rule, event_type, event_data, account_id, strategy_id)
                if violation:
                    violations.append(violation)
                    
                    # Store violation
                    await self._store_violation(violation)
                    
                    # Trigger alerts
                    if self.real_time_alerts:
                        await self._trigger_alert(violation)
                    
                    # Auto-remediation if enabled
                    if self.auto_remediation and violation.status == ComplianceStatus.CRITICAL:
                        await self._auto_remediate(violation)
                        
            except Exception as e:
                logger.error(f"Error checking rule {rule.rule_id}: {e}")
        
        return violations
    
    async def _check_rule(
        self,
        rule: ComplianceRule,
        event_type: str,
        event_data: Dict[str, Any],
        account_id: Optional[str],
        strategy_id: Optional[str]
    ) -> Optional[ComplianceViolation]:
        """Check a specific compliance rule."""
        
        # Rule-specific checking logic
        if rule.rule_type == ComplianceRuleType.POSITION_LIMIT:
            return await self._check_position_limit(rule, event_data, account_id)
        
        elif rule.rule_type == ComplianceRuleType.CONCENTRATION_LIMIT:
            return await self._check_concentration_limit(rule, event_data, account_id)
        
        elif rule.rule_type == ComplianceRuleType.LEVERAGE_LIMIT:
            return await self._check_leverage_limit(rule, event_data, account_id)
        
        elif rule.rule_type == ComplianceRuleType.DAILY_LOSS_LIMIT:
            return await self._check_daily_loss_limit(rule, event_data, account_id)
        
        elif rule.rule_type == ComplianceRuleType.ORDER_SIZE_LIMIT:
            return await self._check_order_size_limit(rule, event_data, account_id)
        
        elif rule.rule_type == ComplianceRuleType.TRADING_HOURS:
            return await self._check_trading_hours(rule, event_data)
        
        # Add more rule types as needed
        return None
    
    async def _check_position_limit(
        self,
        rule: ComplianceRule,
        event_data: Dict[str, Any],
        account_id: Optional[str]
    ) -> Optional[ComplianceViolation]:
        """Check position limit compliance."""
        
        symbol = event_data.get('symbol')
        position_size = event_data.get('position_size', 0)
        max_position = rule.parameters.get('max_position_size', float('inf'))
        
        if abs(position_size) > max_position:
            return ComplianceViolation(
                violation_id=f"POS_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                rule_id=rule.rule_id,
                rule_type=rule.rule_type,
                status=ComplianceStatus.VIOLATION,
                description=f"Position size {position_size} exceeds limit {max_position} for {symbol}",
                current_value=abs(position_size),
                threshold_value=max_position,
                account_id=account_id,
                symbol=symbol,
                violation_data=event_data.copy()
            )
        
        return None
    
    async def _check_concentration_limit(
        self,
        rule: ComplianceRule,
        event_data: Dict[str, Any],
        account_id: Optional[str]
    ) -> Optional[ComplianceViolation]:
        """Check concentration limit compliance."""
        
        symbol = event_data.get('symbol')
        position_value = event_data.get('position_value', 0)
        portfolio_value = event_data.get('portfolio_value', 1)
        max_concentration = rule.parameters.get('max_concentration_pct', 1.0)
        
        concentration = abs(position_value) / portfolio_value if portfolio_value > 0 else 0
        
        if concentration > max_concentration:
            return ComplianceViolation(
                violation_id=f"CONC_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                rule_id=rule.rule_id,
                rule_type=rule.rule_type,
                status=ComplianceStatus.VIOLATION,
                description=f"Concentration {concentration:.2%} exceeds limit {max_concentration:.2%} for {symbol}",
                current_value=concentration,
                threshold_value=max_concentration,
                account_id=account_id,
                symbol=symbol,
                violation_data=event_data.copy()
            )
        
        return None

    async def _check_leverage_limit(
        self,
        rule: ComplianceRule,
        event_data: Dict[str, Any],
        account_id: Optional[str]
    ) -> Optional[ComplianceViolation]:
        """Check leverage limit compliance."""

        current_leverage = event_data.get('leverage', 0)
        max_leverage = rule.parameters.get('max_leverage', float('inf'))

        if current_leverage > max_leverage:
            return ComplianceViolation(
                violation_id=f"LEV_{account_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                rule_id=rule.rule_id,
                rule_type=rule.rule_type,
                status=ComplianceStatus.VIOLATION,
                description=f"Leverage {current_leverage:.2f} exceeds limit {max_leverage:.2f}",
                current_value=current_leverage,
                threshold_value=max_leverage,
                account_id=account_id,
                violation_data=event_data.copy()
            )

        return None

    async def _check_daily_loss_limit(
        self,
        rule: ComplianceRule,
        event_data: Dict[str, Any],
        account_id: Optional[str]
    ) -> Optional[ComplianceViolation]:
        """Check daily loss limit compliance."""

        daily_pnl = event_data.get('daily_pnl', 0)
        max_daily_loss = rule.parameters.get('max_daily_loss', float('inf'))

        if daily_pnl < -max_daily_loss:
            return ComplianceViolation(
                violation_id=f"LOSS_{account_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                rule_id=rule.rule_id,
                rule_type=rule.rule_type,
                status=ComplianceStatus.CRITICAL,
                description=f"Daily loss {daily_pnl:.2f} exceeds limit {max_daily_loss:.2f}",
                current_value=abs(daily_pnl),
                threshold_value=max_daily_loss,
                account_id=account_id,
                violation_data=event_data.copy()
            )

        return None

    async def _check_order_size_limit(
        self,
        rule: ComplianceRule,
        event_data: Dict[str, Any],
        account_id: Optional[str]
    ) -> Optional[ComplianceViolation]:
        """Check order size limit compliance."""

        order_size = event_data.get('order_size', 0)
        max_order_size = rule.parameters.get('max_order_size', float('inf'))
        symbol = event_data.get('symbol')

        if abs(order_size) > max_order_size:
            return ComplianceViolation(
                violation_id=f"ORDER_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                rule_id=rule.rule_id,
                rule_type=rule.rule_type,
                status=ComplianceStatus.VIOLATION,
                description=f"Order size {order_size} exceeds limit {max_order_size} for {symbol}",
                current_value=abs(order_size),
                threshold_value=max_order_size,
                account_id=account_id,
                symbol=symbol,
                violation_data=event_data.copy()
            )

        return None

    async def _check_trading_hours(
        self,
        rule: ComplianceRule,
        event_data: Dict[str, Any]
    ) -> Optional[ComplianceViolation]:
        """Check trading hours compliance."""

        current_time = datetime.now().time()
        trading_start = rule.parameters.get('trading_start_time', '09:30')
        trading_end = rule.parameters.get('trading_end_time', '16:00')

        # Parse time strings
        start_time = datetime.strptime(trading_start, '%H:%M').time()
        end_time = datetime.strptime(trading_end, '%H:%M').time()

        if not (start_time <= current_time <= end_time):
            return ComplianceViolation(
                violation_id=f"HOURS_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                rule_id=rule.rule_id,
                rule_type=rule.rule_type,
                status=ComplianceStatus.VIOLATION,
                description=f"Trading outside allowed hours {trading_start}-{trading_end}",
                violation_data=event_data.copy()
            )

        return None

    async def _store_violation(self, violation: ComplianceViolation):
        """Store compliance violation in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO compliance_violations (
                    violation_id, rule_id, rule_type, status, description,
                    current_value, threshold_value, account_id, strategy_id,
                    trade_id, symbol, violation_data, resolved, resolution_action,
                    resolved_at, detected_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                violation.violation_id, violation.rule_id, violation.rule_type,
                violation.status, violation.description, violation.current_value,
                violation.threshold_value, violation.account_id, violation.strategy_id,
                violation.trade_id, violation.symbol, json.dumps(violation.violation_data),
                violation.resolved, violation.resolution_action,
                violation.resolved_at.isoformat() if violation.resolved_at else None,
                violation.detected_at.isoformat()
            ))
            conn.commit()

        # Add to active violations
        self.active_violations[violation.violation_id] = violation

    async def _trigger_alert(self, violation: ComplianceViolation):
        """Trigger compliance violation alert."""
        logger.warning(f"Compliance violation detected: {violation.violation_id} - {violation.description}")

        # Call registered callbacks
        for callback in self.violation_callbacks:
            try:
                await callback(violation)
            except Exception as e:
                logger.error(f"Error in violation callback: {e}")

    async def _auto_remediate(self, violation: ComplianceViolation):
        """Attempt automatic remediation of critical violations."""
        logger.critical(f"Auto-remediation triggered for violation: {violation.violation_id}")

        # Implement auto-remediation logic based on violation type
        if violation.rule_type == ComplianceRuleType.DAILY_LOSS_LIMIT:
            # Stop all trading for the account
            logger.critical(f"Stopping all trading for account {violation.account_id} due to daily loss limit breach")
            # In production, this would trigger emergency stop

        elif violation.rule_type == ComplianceRuleType.POSITION_LIMIT:
            # Reduce position size
            logger.warning(f"Position limit breach detected for {violation.symbol}, consider reducing position")

        # Mark as auto-remediated
        violation.resolution_action = "auto_remediation_attempted"

    def add_violation_callback(self, callback: Callable):
        """Add callback function for violation notifications."""
        self.violation_callbacks.append(callback)

    async def get_violations(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        account_id: Optional[str] = None,
        rule_types: Optional[List[ComplianceRuleType]] = None,
        resolved: Optional[bool] = None,
        limit: int = 1000
    ) -> List[ComplianceViolation]:
        """Retrieve compliance violations with filtering."""

        query = "SELECT * FROM compliance_violations WHERE 1=1"
        params = []

        if start_date:
            query += " AND detected_at >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND detected_at <= ?"
            params.append(end_date.isoformat())

        if account_id:
            query += " AND account_id = ?"
            params.append(account_id)

        if rule_types:
            placeholders = ','.join(['?' for _ in rule_types])
            query += f" AND rule_type IN ({placeholders})"
            params.extend(rule_types)

        if resolved is not None:
            query += " AND resolved = ?"
            params.append(resolved)

        query += " ORDER BY detected_at DESC LIMIT ?"
        params.append(limit)

        violations = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)

            for row in cursor.fetchall():
                violation = ComplianceViolation(
                    violation_id=row['violation_id'],
                    rule_id=row['rule_id'],
                    rule_type=ComplianceRuleType(row['rule_type']),
                    status=ComplianceStatus(row['status']),
                    description=row['description'],
                    current_value=row['current_value'],
                    threshold_value=row['threshold_value'],
                    account_id=row['account_id'],
                    strategy_id=row['strategy_id'],
                    trade_id=row['trade_id'],
                    symbol=row['symbol'],
                    violation_data=json.loads(row['violation_data']),
                    resolved=bool(row['resolved']),
                    resolution_action=row['resolution_action'],
                    resolved_at=datetime.fromisoformat(row['resolved_at']) if row['resolved_at'] else None,
                    detected_at=datetime.fromisoformat(row['detected_at'])
                )
                violations.append(violation)

        return violations

    async def resolve_violation(
        self,
        violation_id: str,
        resolution_action: str
    ) -> bool:
        """Mark a violation as resolved."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE compliance_violations
                    SET resolved = 1, resolution_action = ?, resolved_at = ?
                    WHERE violation_id = ?
                ''', (resolution_action, datetime.now().isoformat(), violation_id))
                conn.commit()

            # Update in memory if present
            if violation_id in self.active_violations:
                violation = self.active_violations[violation_id]
                violation.resolved = True
                violation.resolution_action = resolution_action
                violation.resolved_at = datetime.now()
                del self.active_violations[violation_id]

            logger.info(f"Violation {violation_id} resolved: {resolution_action}")
            return True

        except Exception as e:
            logger.error(f"Error resolving violation {violation_id}: {e}")
            return False

    async def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive compliance dashboard data."""

        # Get recent violations
        recent_violations = await self.get_violations(
            start_date=datetime.now() - timedelta(days=7),
            limit=100
        )

        # Calculate metrics
        total_violations = len(recent_violations)
        unresolved_violations = len([v for v in recent_violations if not v.resolved])
        critical_violations = len([v for v in recent_violations if v.status == ComplianceStatus.CRITICAL])

        # Violation breakdown by type
        violation_by_type = {}
        for violation in recent_violations:
            rule_type = violation.rule_type
            if rule_type not in violation_by_type:
                violation_by_type[rule_type] = 0
            violation_by_type[rule_type] += 1

        # Daily violation counts
        daily_counts = {}
        for violation in recent_violations:
            date_key = violation.detected_at.date().isoformat()
            if date_key not in daily_counts:
                daily_counts[date_key] = 0
            daily_counts[date_key] += 1

        return {
            'compliance_summary': {
                'total_rules': len(self.rules),
                'active_rules': len([r for r in self.rules.values() if r.enabled]),
                'total_violations_7d': total_violations,
                'unresolved_violations': unresolved_violations,
                'critical_violations': critical_violations,
                'compliance_score': max(0, 100 - (unresolved_violations * 10) - (critical_violations * 20))
            },
            'violation_breakdown': violation_by_type,
            'daily_violation_counts': daily_counts,
            'recent_violations': [
                {
                    'violation_id': v.violation_id,
                    'rule_type': v.rule_type,
                    'status': v.status,
                    'description': v.description,
                    'detected_at': v.detected_at.isoformat(),
                    'resolved': v.resolved
                }
                for v in recent_violations[:10]
            ],
            'generated_at': datetime.now().isoformat()
        }
