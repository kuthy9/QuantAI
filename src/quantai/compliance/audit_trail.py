"""
Audit Trail Management for QuantAI Regulatory Compliance.

This module provides comprehensive audit trail functionality including:
- Event logging and tracking
- Immutable audit records
- Compliance audit trails
- Data integrity verification
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import sqlite3
import asyncio
from loguru import logger


class AuditEventType(str, Enum):
    """Types of audit events."""
    TRADE_EXECUTION = "trade_execution"
    ORDER_PLACEMENT = "order_placement"
    ORDER_MODIFICATION = "order_modification"
    ORDER_CANCELLATION = "order_cancellation"
    POSITION_CHANGE = "position_change"
    RISK_LIMIT_BREACH = "risk_limit_breach"
    SYSTEM_ACCESS = "system_access"
    CONFIGURATION_CHANGE = "configuration_change"
    DATA_ACCESS = "data_access"
    COMPLIANCE_VIOLATION = "compliance_violation"
    REGULATORY_REPORT = "regulatory_report"
    ACCOUNT_MODIFICATION = "account_modification"
    STRATEGY_DEPLOYMENT = "strategy_deployment"
    SYSTEM_ERROR = "system_error"
    SECURITY_EVENT = "security_event"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Immutable audit event record."""
    event_id: str
    event_type: AuditEventType
    severity: AuditSeverity
    timestamp: datetime
    user_id: Optional[str]
    agent_id: Optional[str]
    account_id: Optional[str]
    
    # Event details
    description: str
    event_data: Dict[str, Any]
    
    # Compliance fields
    regulatory_category: Optional[str] = None
    retention_period_years: int = 7
    
    # Integrity fields
    data_hash: Optional[str] = None
    previous_hash: Optional[str] = None
    
    def __post_init__(self):
        """Calculate data hash for integrity verification."""
        if self.data_hash is None:
            self.data_hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash of event data."""
        # Create deterministic string representation
        data_str = json.dumps({
            'event_id': self.event_id,
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'agent_id': self.agent_id,
            'account_id': self.account_id,
            'description': self.description,
            'event_data': self.event_data,
            'previous_hash': self.previous_hash
        }, sort_keys=True)
        
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify the integrity of this audit event."""
        expected_hash = self._calculate_hash()
        return self.data_hash == expected_hash


class AuditTrailManager:
    """
    Comprehensive audit trail management system.
    
    Provides immutable audit logging with integrity verification,
    compliance categorization, and regulatory retention policies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize audit trail manager."""
        self.config = config or {}
        
        # Database configuration
        self.db_path = Path(self.config.get('audit_db_path', 'data/audit_trail.db'))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Retention policies
        self.default_retention_years = self.config.get('default_retention_years', 7)
        self.max_retention_years = self.config.get('max_retention_years', 10)
        
        # Integrity chain
        self.last_hash: Optional[str] = None
        
        # Initialize database
        self._init_database()
        
        logger.info("Audit Trail Manager initialized")
    
    def _init_database(self):
        """Initialize audit trail database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_id TEXT,
                    agent_id TEXT,
                    account_id TEXT,
                    description TEXT NOT NULL,
                    event_data TEXT NOT NULL,
                    regulatory_category TEXT,
                    retention_period_years INTEGER NOT NULL,
                    data_hash TEXT NOT NULL,
                    previous_hash TEXT,
                    created_at TEXT NOT NULL,
                    INDEX(timestamp),
                    INDEX(event_type),
                    INDEX(account_id),
                    INDEX(regulatory_category)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS audit_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            # Initialize last hash if not exists
            cursor = conn.execute(
                "SELECT value FROM audit_metadata WHERE key = 'last_hash'"
            )
            result = cursor.fetchone()
            if result:
                self.last_hash = result[0]
            
            conn.commit()
    
    async def log_event(
        self,
        event_type: AuditEventType,
        description: str,
        event_data: Dict[str, Any],
        severity: AuditSeverity = AuditSeverity.MEDIUM,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        account_id: Optional[str] = None,
        regulatory_category: Optional[str] = None,
        retention_period_years: Optional[int] = None
    ) -> AuditEvent:
        """Log an audit event with integrity verification."""
        
        # Generate unique event ID
        event_id = f"{event_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Create audit event
        audit_event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            timestamp=datetime.now(),
            user_id=user_id,
            agent_id=agent_id,
            account_id=account_id,
            description=description,
            event_data=event_data.copy(),
            regulatory_category=regulatory_category,
            retention_period_years=retention_period_years or self.default_retention_years,
            previous_hash=self.last_hash
        )
        
        # Store in database
        await self._store_event(audit_event)
        
        # Update hash chain
        self.last_hash = audit_event.data_hash
        await self._update_metadata('last_hash', self.last_hash)
        
        logger.info(f"Audit event logged: {event_id} ({event_type})")
        return audit_event
    
    async def _store_event(self, event: AuditEvent):
        """Store audit event in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO audit_events (
                    event_id, event_type, severity, timestamp, user_id, agent_id,
                    account_id, description, event_data, regulatory_category,
                    retention_period_years, data_hash, previous_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.event_id,
                event.event_type,
                event.severity,
                event.timestamp.isoformat(),
                event.user_id,
                event.agent_id,
                event.account_id,
                event.description,
                json.dumps(event.event_data),
                event.regulatory_category,
                event.retention_period_years,
                event.data_hash,
                event.previous_hash,
                datetime.now().isoformat()
            ))
            conn.commit()
    
    async def _update_metadata(self, key: str, value: str):
        """Update metadata in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO audit_metadata (key, value, updated_at)
                VALUES (?, ?, ?)
            ''', (key, value, datetime.now().isoformat()))
            conn.commit()
    
    async def get_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        account_id: Optional[str] = None,
        regulatory_category: Optional[str] = None,
        limit: int = 1000
    ) -> List[AuditEvent]:
        """Retrieve audit events with filtering."""
        
        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        
        if event_types:
            placeholders = ','.join(['?' for _ in event_types])
            query += f" AND event_type IN ({placeholders})"
            params.extend(event_types)
        
        if account_id:
            query += " AND account_id = ?"
            params.append(account_id)
        
        if regulatory_category:
            query += " AND regulatory_category = ?"
            params.append(regulatory_category)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        events = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            
            for row in cursor.fetchall():
                event = AuditEvent(
                    event_id=row['event_id'],
                    event_type=AuditEventType(row['event_type']),
                    severity=AuditSeverity(row['severity']),
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    user_id=row['user_id'],
                    agent_id=row['agent_id'],
                    account_id=row['account_id'],
                    description=row['description'],
                    event_data=json.loads(row['event_data']),
                    regulatory_category=row['regulatory_category'],
                    retention_period_years=row['retention_period_years'],
                    data_hash=row['data_hash'],
                    previous_hash=row['previous_hash']
                )
                events.append(event)
        
        return events
    
    async def verify_integrity_chain(self, start_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Verify the integrity of the audit trail chain."""
        
        events = await self.get_events(start_date=start_date, limit=10000)
        events.reverse()  # Process in chronological order
        
        verification_result = {
            'total_events': len(events),
            'verified_events': 0,
            'integrity_violations': [],
            'chain_valid': True,
            'verification_timestamp': datetime.now().isoformat()
        }
        
        previous_hash = None
        
        for event in events:
            # Verify individual event integrity
            if not event.verify_integrity():
                verification_result['integrity_violations'].append({
                    'event_id': event.event_id,
                    'violation_type': 'data_corruption',
                    'timestamp': event.timestamp.isoformat()
                })
                verification_result['chain_valid'] = False
                continue
            
            # Verify chain integrity
            if event.previous_hash != previous_hash:
                verification_result['integrity_violations'].append({
                    'event_id': event.event_id,
                    'violation_type': 'chain_break',
                    'expected_previous_hash': previous_hash,
                    'actual_previous_hash': event.previous_hash,
                    'timestamp': event.timestamp.isoformat()
                })
                verification_result['chain_valid'] = False
            
            verification_result['verified_events'] += 1
            previous_hash = event.data_hash
        
        return verification_result
    
    async def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        regulatory_categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate compliance audit report."""
        
        events = await self.get_events(
            start_date=start_date,
            end_date=end_date,
            regulatory_category=regulatory_categories[0] if regulatory_categories else None
        )
        
        # Categorize events
        event_summary = {}
        severity_summary = {}
        daily_counts = {}
        
        for event in events:
            # Event type summary
            event_type = event.event_type
            if event_type not in event_summary:
                event_summary[event_type] = 0
            event_summary[event_type] += 1
            
            # Severity summary
            severity = event.severity
            if severity not in severity_summary:
                severity_summary[severity] = 0
            severity_summary[severity] += 1
            
            # Daily counts
            date_key = event.timestamp.date().isoformat()
            if date_key not in daily_counts:
                daily_counts[date_key] = 0
            daily_counts[date_key] += 1
        
        # Identify high-risk events
        high_risk_events = [
            {
                'event_id': event.event_id,
                'event_type': event.event_type,
                'severity': event.severity,
                'timestamp': event.timestamp.isoformat(),
                'description': event.description
            }
            for event in events
            if event.severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL]
        ]
        
        return {
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'total_events': len(events),
            'event_type_summary': event_summary,
            'severity_summary': severity_summary,
            'daily_event_counts': daily_counts,
            'high_risk_events': high_risk_events,
            'regulatory_categories': regulatory_categories,
            'generated_at': datetime.now().isoformat()
        }
    
    async def cleanup_expired_records(self) -> Dict[str, Any]:
        """Clean up audit records that have exceeded retention period."""
        
        cleanup_summary = {
            'total_records_before': 0,
            'expired_records_removed': 0,
            'total_records_after': 0,
            'cleanup_timestamp': datetime.now().isoformat()
        }
        
        with sqlite3.connect(self.db_path) as conn:
            # Count total records before cleanup
            cursor = conn.execute("SELECT COUNT(*) FROM audit_events")
            cleanup_summary['total_records_before'] = cursor.fetchone()[0]
            
            # Calculate cutoff dates for each retention period
            current_date = datetime.now()
            
            # Remove expired records
            cursor = conn.execute('''
                DELETE FROM audit_events 
                WHERE datetime(timestamp) < datetime(?, '-' || retention_period_years || ' years')
            ''', (current_date.isoformat(),))
            
            cleanup_summary['expired_records_removed'] = cursor.rowcount
            
            # Count total records after cleanup
            cursor = conn.execute("SELECT COUNT(*) FROM audit_events")
            cleanup_summary['total_records_after'] = cursor.fetchone()[0]
            
            conn.commit()
        
        logger.info(f"Audit trail cleanup completed: {cleanup_summary['expired_records_removed']} records removed")
        return cleanup_summary

    async def export_audit_trail(
        self,
        start_date: datetime,
        end_date: datetime,
        export_format: str = 'json',
        output_path: Optional[Path] = None
    ) -> str:
        """Export audit trail for regulatory submission."""

        events = await self.get_events(start_date=start_date, end_date=end_date)

        # Convert events to serializable format
        export_data = {
            'export_metadata': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'total_events': len(events),
                'export_timestamp': datetime.now().isoformat(),
                'format': export_format
            },
            'audit_events': [
                {
                    'event_id': event.event_id,
                    'event_type': event.event_type,
                    'severity': event.severity,
                    'timestamp': event.timestamp.isoformat(),
                    'user_id': event.user_id,
                    'agent_id': event.agent_id,
                    'account_id': event.account_id,
                    'description': event.description,
                    'event_data': event.event_data,
                    'regulatory_category': event.regulatory_category,
                    'data_hash': event.data_hash
                }
                for event in events
            ]
        }

        # Generate output filename if not provided
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = Path(f'audit_export_{timestamp}.{export_format}')

        # Export data
        if export_format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")

        logger.info(f"Audit trail exported to {output_path}")
        return str(output_path)
