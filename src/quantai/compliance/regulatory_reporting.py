"""
Regulatory Reporting System for QuantAI Compliance Framework.

This module provides automated regulatory reporting capabilities including:
- Scheduled report generation
- Multi-format report output
- Regulatory submission tracking
- Compliance report templates
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import sqlite3
from loguru import logger

from .audit_trail import AuditTrailManager, AuditEventType
from .trade_reporting import TradeReportingEngine, TradeReportType
from .compliance_monitor import ComplianceMonitor, ComplianceRuleType


class ReportingFrequency(str, Enum):
    """Regulatory reporting frequencies."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"
    ON_DEMAND = "on_demand"


class ReportingFormat(str, Enum):
    """Report output formats."""
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    PDF = "pdf"
    XLSX = "xlsx"


@dataclass
class RegulatoryReportTemplate:
    """Template for regulatory reports."""
    template_id: str
    template_name: str
    regulatory_regime: str  # e.g., "SEC", "FINRA", "MiFID II"
    report_type: str
    frequency: ReportingFrequency
    
    # Template configuration
    required_fields: List[str]
    optional_fields: List[str] = None
    data_sources: List[str] = None  # audit_trail, trade_reporting, compliance_monitor
    
    # Output configuration
    output_format: ReportingFormat = ReportingFormat.JSON
    submission_method: str = "manual"  # manual, api, email
    
    # Scheduling
    enabled: bool = True
    next_due_date: Optional[datetime] = None
    
    # Metadata
    created_at: datetime = None
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_updated is None:
            self.last_updated = datetime.now()
        if self.optional_fields is None:
            self.optional_fields = []
        if self.data_sources is None:
            self.data_sources = ["audit_trail", "trade_reporting", "compliance_monitor"]


class RegulatoryReportGenerator:
    """
    Automated regulatory report generation system.
    
    Generates regulatory reports from audit trails, trade data,
    and compliance monitoring data according to regulatory requirements.
    """
    
    def __init__(
        self,
        audit_manager: AuditTrailManager,
        trade_reporting: TradeReportingEngine,
        compliance_monitor: ComplianceMonitor,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize regulatory report generator."""
        self.audit_manager = audit_manager
        self.trade_reporting = trade_reporting
        self.compliance_monitor = compliance_monitor
        self.config = config or {}
        
        # Database configuration
        self.db_path = Path(self.config.get('regulatory_db_path', 'data/regulatory_reports.db'))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Output configuration
        self.output_dir = Path(self.config.get('output_dir', 'reports/regulatory'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Report templates
        self.templates: Dict[str, RegulatoryReportTemplate] = {}
        
        # Initialize database and load templates
        self._init_database()
        asyncio.create_task(self._load_templates())
        
        logger.info("Regulatory Report Generator initialized")
    
    def _init_database(self):
        """Initialize regulatory reporting database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS report_templates (
                    template_id TEXT PRIMARY KEY,
                    template_name TEXT NOT NULL,
                    regulatory_regime TEXT NOT NULL,
                    report_type TEXT NOT NULL,
                    frequency TEXT NOT NULL,
                    required_fields TEXT NOT NULL,
                    optional_fields TEXT,
                    data_sources TEXT NOT NULL,
                    output_format TEXT NOT NULL,
                    submission_method TEXT NOT NULL,
                    enabled BOOLEAN NOT NULL,
                    next_due_date TEXT,
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS generated_reports (
                    report_id TEXT PRIMARY KEY,
                    template_id TEXT NOT NULL,
                    report_period_start TEXT NOT NULL,
                    report_period_end TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    output_format TEXT NOT NULL,
                    generation_status TEXT NOT NULL,
                    submission_status TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    submitted_at TEXT,
                    error_message TEXT,
                    FOREIGN KEY (template_id) REFERENCES report_templates (template_id)
                )
            ''')
            
            conn.commit()
    
    async def _load_templates(self):
        """Load report templates from database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM report_templates WHERE enabled = 1")
            
            for row in cursor.fetchall():
                template = RegulatoryReportTemplate(
                    template_id=row['template_id'],
                    template_name=row['template_name'],
                    regulatory_regime=row['regulatory_regime'],
                    report_type=row['report_type'],
                    frequency=ReportingFrequency(row['frequency']),
                    required_fields=json.loads(row['required_fields']),
                    optional_fields=json.loads(row['optional_fields']) if row['optional_fields'] else [],
                    data_sources=json.loads(row['data_sources']),
                    output_format=ReportingFormat(row['output_format']),
                    submission_method=row['submission_method'],
                    enabled=bool(row['enabled']),
                    next_due_date=datetime.fromisoformat(row['next_due_date']) if row['next_due_date'] else None,
                    created_at=datetime.fromisoformat(row['created_at']),
                    last_updated=datetime.fromisoformat(row['last_updated'])
                )
                self.templates[template.template_id] = template
        
        logger.info(f"Loaded {len(self.templates)} regulatory report templates")
    
    async def add_template(self, template: RegulatoryReportTemplate) -> bool:
        """Add a new regulatory report template."""
        try:
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO report_templates (
                        template_id, template_name, regulatory_regime, report_type,
                        frequency, required_fields, optional_fields, data_sources,
                        output_format, submission_method, enabled, next_due_date,
                        created_at, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    template.template_id, template.template_name, template.regulatory_regime,
                    template.report_type, template.frequency, json.dumps(template.required_fields),
                    json.dumps(template.optional_fields), json.dumps(template.data_sources),
                    template.output_format, template.submission_method, template.enabled,
                    template.next_due_date.isoformat() if template.next_due_date else None,
                    template.created_at.isoformat(), template.last_updated.isoformat()
                ))
                conn.commit()
            
            # Add to memory
            self.templates[template.template_id] = template
            
            logger.info(f"Added regulatory report template: {template.template_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding report template {template.template_id}: {e}")
            return False
    
    async def generate_report(
        self,
        template_id: str,
        start_date: datetime,
        end_date: datetime,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Generate a regulatory report from template."""
        
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")
        
        template = self.templates[template_id]
        
        # Generate report ID
        report_id = f"{template_id}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        
        try:
            # Collect data from configured sources
            report_data = await self._collect_report_data(template, start_date, end_date)
            
            # Generate output path if not provided
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{template.template_id}_{timestamp}.{template.output_format}"
                output_path = self.output_dir / filename
            
            # Generate report in specified format
            await self._generate_report_output(template, report_data, output_path)
            
            # Record generation
            await self._record_report_generation(
                report_id, template_id, start_date, end_date,
                output_path, template.output_format, "completed"
            )
            
            logger.info(f"Generated regulatory report: {report_id}")
            
            return {
                'report_id': report_id,
                'template_id': template_id,
                'output_path': str(output_path),
                'status': 'completed',
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            # Record failure
            await self._record_report_generation(
                report_id, template_id, start_date, end_date,
                output_path or Path("failed"), template.output_format, "failed", str(e)
            )
            
            logger.error(f"Error generating report {report_id}: {e}")
            raise
    
    async def _collect_report_data(
        self,
        template: RegulatoryReportTemplate,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Collect data from configured sources for report generation."""
        
        report_data = {
            'report_metadata': {
                'template_id': template.template_id,
                'template_name': template.template_name,
                'regulatory_regime': template.regulatory_regime,
                'report_type': template.report_type,
                'period_start': start_date.isoformat(),
                'period_end': end_date.isoformat(),
                'generated_at': datetime.now().isoformat()
            }
        }
        
        # Collect audit trail data
        if 'audit_trail' in template.data_sources:
            audit_events = await self.audit_manager.get_events(
                start_date=start_date,
                end_date=end_date,
                limit=10000
            )
            
            report_data['audit_events'] = [
                {
                    'event_id': event.event_id,
                    'event_type': event.event_type,
                    'severity': event.severity,
                    'timestamp': event.timestamp.isoformat(),
                    'description': event.description,
                    'user_id': event.user_id,
                    'agent_id': event.agent_id,
                    'account_id': event.account_id
                }
                for event in audit_events
            ]
        
        # Collect trade reporting data
        if 'trade_reporting' in template.data_sources:
            trade_reports = await self.trade_reporting.get_trade_reports(
                start_date=start_date,
                end_date=end_date,
                limit=10000
            )
            
            report_data['trade_reports'] = [
                {
                    'report_id': report.report_id,
                    'trade_id': report.trade_id,
                    'account_id': report.account_id,
                    'symbol': report.symbol,
                    'side': report.side,
                    'quantity': report.quantity,
                    'price': report.price,
                    'execution_time': report.execution_time.isoformat(),
                    'commission': report.commission,
                    'fees': report.fees,
                    'venue': report.venue
                }
                for report in trade_reports
            ]
        
        # Collect compliance monitoring data
        if 'compliance_monitor' in template.data_sources:
            violations = await self.compliance_monitor.get_violations(
                start_date=start_date,
                end_date=end_date,
                limit=10000
            )
            
            report_data['compliance_violations'] = [
                {
                    'violation_id': violation.violation_id,
                    'rule_type': violation.rule_type,
                    'status': violation.status,
                    'description': violation.description,
                    'detected_at': violation.detected_at.isoformat(),
                    'resolved': violation.resolved,
                    'account_id': violation.account_id
                }
                for violation in violations
            ]
        
        return report_data

    async def _generate_report_output(
        self,
        template: RegulatoryReportTemplate,
        report_data: Dict[str, Any],
        output_path: Path
    ):
        """Generate report output in specified format."""

        if template.output_format == ReportingFormat.JSON:
            await self._generate_json_report(report_data, output_path)
        elif template.output_format == ReportingFormat.CSV:
            await self._generate_csv_report(report_data, output_path)
        elif template.output_format == ReportingFormat.XML:
            await self._generate_xml_report(report_data, output_path)
        else:
            # Default to JSON for unsupported formats
            await self._generate_json_report(report_data, output_path)

    async def _generate_json_report(self, report_data: Dict[str, Any], output_path: Path):
        """Generate JSON format report."""
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

    async def _generate_csv_report(self, report_data: Dict[str, Any], output_path: Path):
        """Generate CSV format report."""
        import csv

        # Create CSV with flattened data structure
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write metadata
            writer.writerow(['Report Metadata'])
            for key, value in report_data['report_metadata'].items():
                writer.writerow([key, value])
            writer.writerow([])

            # Write audit events if present
            if 'audit_events' in report_data and report_data['audit_events']:
                writer.writerow(['Audit Events'])
                events = report_data['audit_events']
                if events:
                    writer.writerow(events[0].keys())
                    for event in events:
                        writer.writerow(event.values())
                writer.writerow([])

            # Write trade reports if present
            if 'trade_reports' in report_data and report_data['trade_reports']:
                writer.writerow(['Trade Reports'])
                trades = report_data['trade_reports']
                if trades:
                    writer.writerow(trades[0].keys())
                    for trade in trades:
                        writer.writerow(trade.values())
                writer.writerow([])

            # Write compliance violations if present
            if 'compliance_violations' in report_data and report_data['compliance_violations']:
                writer.writerow(['Compliance Violations'])
                violations = report_data['compliance_violations']
                if violations:
                    writer.writerow(violations[0].keys())
                    for violation in violations:
                        writer.writerow(violation.values())

    async def _generate_xml_report(self, report_data: Dict[str, Any], output_path: Path):
        """Generate XML format report."""
        import xml.etree.ElementTree as ET

        root = ET.Element("RegulatoryReport")

        # Add metadata
        metadata = ET.SubElement(root, "Metadata")
        for key, value in report_data['report_metadata'].items():
            elem = ET.SubElement(metadata, key.replace('_', ''))
            elem.text = str(value)

        # Add audit events
        if 'audit_events' in report_data:
            audit_section = ET.SubElement(root, "AuditEvents")
            for event in report_data['audit_events']:
                event_elem = ET.SubElement(audit_section, "AuditEvent")
                for key, value in event.items():
                    elem = ET.SubElement(event_elem, key.replace('_', ''))
                    elem.text = str(value)

        # Add trade reports
        if 'trade_reports' in report_data:
            trade_section = ET.SubElement(root, "TradeReports")
            for trade in report_data['trade_reports']:
                trade_elem = ET.SubElement(trade_section, "TradeReport")
                for key, value in trade.items():
                    elem = ET.SubElement(trade_elem, key.replace('_', ''))
                    elem.text = str(value)

        # Add compliance violations
        if 'compliance_violations' in report_data:
            compliance_section = ET.SubElement(root, "ComplianceViolations")
            for violation in report_data['compliance_violations']:
                violation_elem = ET.SubElement(compliance_section, "ComplianceViolation")
                for key, value in violation.items():
                    elem = ET.SubElement(violation_elem, key.replace('_', ''))
                    elem.text = str(value)

        # Write to file
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)

    async def _record_report_generation(
        self,
        report_id: str,
        template_id: str,
        start_date: datetime,
        end_date: datetime,
        output_path: Path,
        output_format: ReportingFormat,
        status: str,
        error_message: Optional[str] = None
    ):
        """Record report generation in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO generated_reports (
                    report_id, template_id, report_period_start, report_period_end,
                    output_path, output_format, generation_status, submission_status,
                    generated_at, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                report_id, template_id, start_date.isoformat(), end_date.isoformat(),
                str(output_path), output_format, status, "pending",
                datetime.now().isoformat(), error_message
            ))
            conn.commit()

    async def get_due_reports(self) -> List[RegulatoryReportTemplate]:
        """Get reports that are due for generation."""
        due_reports = []
        current_time = datetime.now()

        for template in self.templates.values():
            if not template.enabled:
                continue

            if template.next_due_date and template.next_due_date <= current_time:
                due_reports.append(template)

        return due_reports

    async def schedule_reports(self):
        """Schedule automatic report generation for due reports."""
        due_reports = await self.get_due_reports()

        for template in due_reports:
            try:
                # Calculate reporting period based on frequency
                end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

                if template.frequency == ReportingFrequency.DAILY:
                    start_date = end_date - timedelta(days=1)
                elif template.frequency == ReportingFrequency.WEEKLY:
                    start_date = end_date - timedelta(weeks=1)
                elif template.frequency == ReportingFrequency.MONTHLY:
                    start_date = end_date.replace(day=1) - timedelta(days=1)
                    start_date = start_date.replace(day=1)
                elif template.frequency == ReportingFrequency.QUARTERLY:
                    # Calculate quarter start
                    quarter = (end_date.month - 1) // 3 + 1
                    start_date = datetime(end_date.year, (quarter - 1) * 3 + 1, 1)
                    if quarter > 1:
                        start_date = datetime(end_date.year, (quarter - 2) * 3 + 1, 1)
                else:
                    # Default to monthly
                    start_date = end_date.replace(day=1) - timedelta(days=1)
                    start_date = start_date.replace(day=1)

                # Generate report
                await self.generate_report(template.template_id, start_date, end_date)

                # Update next due date
                await self._update_next_due_date(template)

                logger.info(f"Scheduled report generated: {template.template_id}")

            except Exception as e:
                logger.error(f"Error generating scheduled report {template.template_id}: {e}")

    async def _update_next_due_date(self, template: RegulatoryReportTemplate):
        """Update the next due date for a template."""
        current_time = datetime.now()

        if template.frequency == ReportingFrequency.DAILY:
            next_due = current_time + timedelta(days=1)
        elif template.frequency == ReportingFrequency.WEEKLY:
            next_due = current_time + timedelta(weeks=1)
        elif template.frequency == ReportingFrequency.MONTHLY:
            # Next month, same day
            if current_time.month == 12:
                next_due = current_time.replace(year=current_time.year + 1, month=1)
            else:
                next_due = current_time.replace(month=current_time.month + 1)
        elif template.frequency == ReportingFrequency.QUARTERLY:
            # Next quarter
            next_due = current_time + timedelta(days=90)  # Approximate
        else:
            next_due = current_time + timedelta(days=30)  # Default monthly

        # Update in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE report_templates
                SET next_due_date = ?, last_updated = ?
                WHERE template_id = ?
            ''', (next_due.isoformat(), datetime.now().isoformat(), template.template_id))
            conn.commit()

        # Update in memory
        template.next_due_date = next_due
        template.last_updated = datetime.now()

    async def get_report_history(
        self,
        template_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get history of generated reports."""

        query = "SELECT * FROM generated_reports WHERE 1=1"
        params = []

        if template_id:
            query += " AND template_id = ?"
            params.append(template_id)

        if start_date:
            query += " AND generated_at >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND generated_at <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY generated_at DESC LIMIT ?"
        params.append(limit)

        reports = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)

            for row in cursor.fetchall():
                reports.append({
                    'report_id': row['report_id'],
                    'template_id': row['template_id'],
                    'report_period_start': row['report_period_start'],
                    'report_period_end': row['report_period_end'],
                    'output_path': row['output_path'],
                    'output_format': row['output_format'],
                    'generation_status': row['generation_status'],
                    'submission_status': row['submission_status'],
                    'generated_at': row['generated_at'],
                    'submitted_at': row['submitted_at'],
                    'error_message': row['error_message']
                })

        return reports
