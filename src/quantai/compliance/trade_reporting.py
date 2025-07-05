"""
Trade Reporting Engine for QuantAI Regulatory Compliance.

This module provides comprehensive trade reporting functionality including:
- Real-time trade reporting
- Regulatory trade reports
- Transaction cost analysis
- Best execution reporting
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import sqlite3
import asyncio
from loguru import logger


class TradeReportType(str, Enum):
    """Types of trade reports."""
    EXECUTION_REPORT = "execution_report"
    TRANSACTION_COST_ANALYSIS = "transaction_cost_analysis"
    BEST_EXECUTION = "best_execution"
    REGULATORY_FILING = "regulatory_filing"
    POSITION_REPORT = "position_report"
    RISK_REPORT = "risk_report"
    DAILY_SUMMARY = "daily_summary"
    MONTHLY_SUMMARY = "monthly_summary"


@dataclass
class TradeReport:
    """Comprehensive trade report record."""
    report_id: str
    report_type: TradeReportType
    trade_id: str
    account_id: str
    strategy_id: str
    
    # Trade details
    symbol: str
    side: str  # BUY/SELL
    quantity: float
    price: float
    execution_time: datetime
    
    # Market data
    market_price: Optional[float] = None
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    spread: Optional[float] = None
    
    # Execution quality
    slippage: Optional[float] = None
    implementation_shortfall: Optional[float] = None
    market_impact: Optional[float] = None
    
    # Costs
    commission: float = 0.0
    fees: float = 0.0
    total_cost: float = 0.0
    
    # Regulatory fields
    venue: Optional[str] = None
    order_type: Optional[str] = None
    time_in_force: Optional[str] = None
    regulatory_flags: List[str] = None
    
    # Metadata
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.regulatory_flags is None:
            self.regulatory_flags = []
        
        # Calculate derived fields
        if self.market_price and self.price:
            self.slippage = abs(self.price - self.market_price) / self.market_price
        
        if self.bid_price and self.ask_price:
            self.spread = self.ask_price - self.bid_price
        
        self.total_cost = self.commission + self.fees


@dataclass
class RegulatoryReport:
    """Regulatory compliance report."""
    report_id: str
    report_type: str
    reporting_period_start: datetime
    reporting_period_end: datetime
    
    # Report content
    trade_reports: List[TradeReport]
    summary_statistics: Dict[str, Any]
    compliance_metrics: Dict[str, Any]
    
    # Regulatory metadata
    regulatory_regime: str  # e.g., "MiFID II", "SEC", "CFTC"
    submission_deadline: datetime
    report_status: str = "draft"
    
    # Metadata
    generated_at: datetime = None
    submitted_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.generated_at is None:
            self.generated_at = datetime.now()


class TradeReportingEngine:
    """
    Comprehensive trade reporting engine for regulatory compliance.
    
    Provides real-time trade reporting, regulatory submissions,
    and transaction cost analysis capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize trade reporting engine."""
        self.config = config or {}
        
        # Database configuration
        self.db_path = Path(self.config.get('reporting_db_path', 'data/trade_reporting.db'))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Reporting configuration
        self.real_time_reporting = self.config.get('real_time_reporting', True)
        self.regulatory_regimes = self.config.get('regulatory_regimes', ['SEC', 'FINRA'])
        
        # Initialize database
        self._init_database()
        
        logger.info("Trade Reporting Engine initialized")
    
    def _init_database(self):
        """Initialize trade reporting database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trade_reports (
                    report_id TEXT PRIMARY KEY,
                    report_type TEXT NOT NULL,
                    trade_id TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    execution_time TEXT NOT NULL,
                    market_price REAL,
                    bid_price REAL,
                    ask_price REAL,
                    spread REAL,
                    slippage REAL,
                    implementation_shortfall REAL,
                    market_impact REAL,
                    commission REAL NOT NULL,
                    fees REAL NOT NULL,
                    total_cost REAL NOT NULL,
                    venue TEXT,
                    order_type TEXT,
                    time_in_force TEXT,
                    regulatory_flags TEXT,
                    created_at TEXT NOT NULL,
                    INDEX(trade_id),
                    INDEX(account_id),
                    INDEX(execution_time),
                    INDEX(symbol)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS regulatory_reports (
                    report_id TEXT PRIMARY KEY,
                    report_type TEXT NOT NULL,
                    reporting_period_start TEXT NOT NULL,
                    reporting_period_end TEXT NOT NULL,
                    regulatory_regime TEXT NOT NULL,
                    submission_deadline TEXT NOT NULL,
                    report_status TEXT NOT NULL,
                    report_data TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    submitted_at TEXT,
                    INDEX(report_type),
                    INDEX(regulatory_regime),
                    INDEX(submission_deadline)
                )
            ''')
            
            conn.commit()
    
    async def report_trade(
        self,
        trade_id: str,
        account_id: str,
        strategy_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        execution_time: datetime,
        market_data: Optional[Dict[str, float]] = None,
        execution_quality: Optional[Dict[str, float]] = None,
        costs: Optional[Dict[str, float]] = None,
        regulatory_info: Optional[Dict[str, Any]] = None
    ) -> TradeReport:
        """Report a trade execution for compliance tracking."""
        
        # Generate report ID
        report_id = f"TR_{trade_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Extract optional data
        market_data = market_data or {}
        execution_quality = execution_quality or {}
        costs = costs or {}
        regulatory_info = regulatory_info or {}
        
        # Create trade report
        trade_report = TradeReport(
            report_id=report_id,
            report_type=TradeReportType.EXECUTION_REPORT,
            trade_id=trade_id,
            account_id=account_id,
            strategy_id=strategy_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            execution_time=execution_time,
            
            # Market data
            market_price=market_data.get('market_price'),
            bid_price=market_data.get('bid_price'),
            ask_price=market_data.get('ask_price'),
            
            # Execution quality
            implementation_shortfall=execution_quality.get('implementation_shortfall'),
            market_impact=execution_quality.get('market_impact'),
            
            # Costs
            commission=costs.get('commission', 0.0),
            fees=costs.get('fees', 0.0),
            
            # Regulatory info
            venue=regulatory_info.get('venue'),
            order_type=regulatory_info.get('order_type'),
            time_in_force=regulatory_info.get('time_in_force'),
            regulatory_flags=regulatory_info.get('flags', [])
        )
        
        # Store in database
        await self._store_trade_report(trade_report)
        
        # Real-time regulatory notifications if enabled
        if self.real_time_reporting:
            await self._send_real_time_notifications(trade_report)
        
        logger.info(f"Trade reported: {trade_id} ({symbol} {side} {quantity})")
        return trade_report
    
    async def _store_trade_report(self, report: TradeReport):
        """Store trade report in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO trade_reports (
                    report_id, report_type, trade_id, account_id, strategy_id,
                    symbol, side, quantity, price, execution_time,
                    market_price, bid_price, ask_price, spread, slippage,
                    implementation_shortfall, market_impact, commission, fees, total_cost,
                    venue, order_type, time_in_force, regulatory_flags, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                report.report_id, report.report_type, report.trade_id, report.account_id,
                report.strategy_id, report.symbol, report.side, report.quantity, report.price,
                report.execution_time.isoformat(), report.market_price, report.bid_price,
                report.ask_price, report.spread, report.slippage, report.implementation_shortfall,
                report.market_impact, report.commission, report.fees, report.total_cost,
                report.venue, report.order_type, report.time_in_force,
                json.dumps(report.regulatory_flags), report.created_at.isoformat()
            ))
            conn.commit()
    
    async def _send_real_time_notifications(self, report: TradeReport):
        """Send real-time regulatory notifications."""
        # Check for regulatory flags that require immediate notification
        urgent_flags = ['large_trade', 'unusual_activity', 'cross_border']
        
        if any(flag in report.regulatory_flags for flag in urgent_flags):
            logger.warning(f"Urgent regulatory notification required for trade {report.trade_id}")
            # In production, this would send notifications to regulatory systems
    
    async def get_trade_reports(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        account_id: Optional[str] = None,
        symbol: Optional[str] = None,
        report_types: Optional[List[TradeReportType]] = None,
        limit: int = 1000
    ) -> List[TradeReport]:
        """Retrieve trade reports with filtering."""
        
        query = "SELECT * FROM trade_reports WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND execution_time >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND execution_time <= ?"
            params.append(end_date.isoformat())
        
        if account_id:
            query += " AND account_id = ?"
            params.append(account_id)
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if report_types:
            placeholders = ','.join(['?' for _ in report_types])
            query += f" AND report_type IN ({placeholders})"
            params.extend(report_types)
        
        query += " ORDER BY execution_time DESC LIMIT ?"
        params.append(limit)
        
        reports = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            
            for row in cursor.fetchall():
                report = TradeReport(
                    report_id=row['report_id'],
                    report_type=TradeReportType(row['report_type']),
                    trade_id=row['trade_id'],
                    account_id=row['account_id'],
                    strategy_id=row['strategy_id'],
                    symbol=row['symbol'],
                    side=row['side'],
                    quantity=row['quantity'],
                    price=row['price'],
                    execution_time=datetime.fromisoformat(row['execution_time']),
                    market_price=row['market_price'],
                    bid_price=row['bid_price'],
                    ask_price=row['ask_price'],
                    spread=row['spread'],
                    slippage=row['slippage'],
                    implementation_shortfall=row['implementation_shortfall'],
                    market_impact=row['market_impact'],
                    commission=row['commission'],
                    fees=row['fees'],
                    total_cost=row['total_cost'],
                    venue=row['venue'],
                    order_type=row['order_type'],
                    time_in_force=row['time_in_force'],
                    regulatory_flags=json.loads(row['regulatory_flags']) if row['regulatory_flags'] else [],
                    created_at=datetime.fromisoformat(row['created_at'])
                )
                reports.append(report)
        
        return reports

    async def generate_transaction_cost_analysis(
        self,
        start_date: datetime,
        end_date: datetime,
        account_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive transaction cost analysis."""

        reports = await self.get_trade_reports(
            start_date=start_date,
            end_date=end_date,
            account_id=account_id
        )

        if not reports:
            return {
                'period': {'start': start_date.isoformat(), 'end': end_date.isoformat()},
                'total_trades': 0,
                'analysis': 'No trades found for the specified period'
            }

        # Calculate aggregate metrics
        total_volume = sum(r.quantity * r.price for r in reports)
        total_commission = sum(r.commission for r in reports)
        total_fees = sum(r.fees for r in reports)
        total_costs = sum(r.total_cost for r in reports)

        # Calculate slippage statistics
        slippage_values = [r.slippage for r in reports if r.slippage is not None]
        avg_slippage = sum(slippage_values) / len(slippage_values) if slippage_values else 0

        # Calculate implementation shortfall
        shortfall_values = [r.implementation_shortfall for r in reports if r.implementation_shortfall is not None]
        avg_shortfall = sum(shortfall_values) / len(shortfall_values) if shortfall_values else 0

        # Analyze by symbol
        symbol_analysis = {}
        for report in reports:
            symbol = report.symbol
            if symbol not in symbol_analysis:
                symbol_analysis[symbol] = {
                    'trade_count': 0,
                    'total_volume': 0,
                    'total_costs': 0,
                    'avg_slippage': 0,
                    'slippage_count': 0
                }

            symbol_analysis[symbol]['trade_count'] += 1
            symbol_analysis[symbol]['total_volume'] += report.quantity * report.price
            symbol_analysis[symbol]['total_costs'] += report.total_cost

            if report.slippage is not None:
                symbol_analysis[symbol]['avg_slippage'] += report.slippage
                symbol_analysis[symbol]['slippage_count'] += 1

        # Calculate average slippage per symbol
        for symbol_data in symbol_analysis.values():
            if symbol_data['slippage_count'] > 0:
                symbol_data['avg_slippage'] /= symbol_data['slippage_count']
            symbol_data['cost_per_dollar'] = symbol_data['total_costs'] / symbol_data['total_volume'] if symbol_data['total_volume'] > 0 else 0

        return {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'summary': {
                'total_trades': len(reports),
                'total_volume': total_volume,
                'total_commission': total_commission,
                'total_fees': total_fees,
                'total_costs': total_costs,
                'cost_per_dollar_traded': total_costs / total_volume if total_volume > 0 else 0,
                'average_slippage': avg_slippage,
                'average_implementation_shortfall': avg_shortfall
            },
            'symbol_analysis': symbol_analysis,
            'generated_at': datetime.now().isoformat()
        }

    async def generate_best_execution_report(
        self,
        start_date: datetime,
        end_date: datetime,
        account_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate best execution compliance report."""

        reports = await self.get_trade_reports(
            start_date=start_date,
            end_date=end_date,
            account_id=account_id
        )

        # Analyze execution quality
        execution_metrics = {
            'total_trades': len(reports),
            'trades_with_slippage_data': 0,
            'average_slippage': 0,
            'slippage_distribution': {'<0.1%': 0, '0.1-0.5%': 0, '0.5-1%': 0, '>1%': 0},
            'venue_analysis': {},
            'order_type_analysis': {},
            'time_of_day_analysis': {}
        }

        slippage_sum = 0
        for report in reports:
            if report.slippage is not None:
                execution_metrics['trades_with_slippage_data'] += 1
                slippage_sum += report.slippage

                # Categorize slippage
                slippage_pct = report.slippage * 100
                if slippage_pct < 0.1:
                    execution_metrics['slippage_distribution']['<0.1%'] += 1
                elif slippage_pct < 0.5:
                    execution_metrics['slippage_distribution']['0.1-0.5%'] += 1
                elif slippage_pct < 1.0:
                    execution_metrics['slippage_distribution']['0.5-1%'] += 1
                else:
                    execution_metrics['slippage_distribution']['>1%'] += 1

            # Venue analysis
            venue = report.venue or 'Unknown'
            if venue not in execution_metrics['venue_analysis']:
                execution_metrics['venue_analysis'][venue] = {'count': 0, 'avg_slippage': 0, 'slippage_sum': 0}
            execution_metrics['venue_analysis'][venue]['count'] += 1
            if report.slippage is not None:
                execution_metrics['venue_analysis'][venue]['slippage_sum'] += report.slippage

            # Order type analysis
            order_type = report.order_type or 'Unknown'
            if order_type not in execution_metrics['order_type_analysis']:
                execution_metrics['order_type_analysis'][order_type] = {'count': 0, 'avg_slippage': 0, 'slippage_sum': 0}
            execution_metrics['order_type_analysis'][order_type]['count'] += 1
            if report.slippage is not None:
                execution_metrics['order_type_analysis'][order_type]['slippage_sum'] += report.slippage

            # Time of day analysis
            hour = report.execution_time.hour
            time_bucket = f"{hour:02d}:00-{hour+1:02d}:00"
            if time_bucket not in execution_metrics['time_of_day_analysis']:
                execution_metrics['time_of_day_analysis'][time_bucket] = {'count': 0, 'avg_slippage': 0, 'slippage_sum': 0}
            execution_metrics['time_of_day_analysis'][time_bucket]['count'] += 1
            if report.slippage is not None:
                execution_metrics['time_of_day_analysis'][time_bucket]['slippage_sum'] += report.slippage

        # Calculate averages
        if execution_metrics['trades_with_slippage_data'] > 0:
            execution_metrics['average_slippage'] = slippage_sum / execution_metrics['trades_with_slippage_data']

        # Calculate venue averages
        for venue_data in execution_metrics['venue_analysis'].values():
            if venue_data['count'] > 0 and venue_data['slippage_sum'] > 0:
                venue_data['avg_slippage'] = venue_data['slippage_sum'] / venue_data['count']

        # Calculate order type averages
        for order_data in execution_metrics['order_type_analysis'].values():
            if order_data['count'] > 0 and order_data['slippage_sum'] > 0:
                order_data['avg_slippage'] = order_data['slippage_sum'] / order_data['count']

        # Calculate time of day averages
        for time_data in execution_metrics['time_of_day_analysis'].values():
            if time_data['count'] > 0 and time_data['slippage_sum'] > 0:
                time_data['avg_slippage'] = time_data['slippage_sum'] / time_data['count']

        return {
            'report_type': 'best_execution',
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'execution_metrics': execution_metrics,
            'compliance_summary': {
                'best_execution_achieved': execution_metrics['average_slippage'] < 0.005,  # < 0.5%
                'venue_diversification': len(execution_metrics['venue_analysis']) > 1,
                'order_type_optimization': len(execution_metrics['order_type_analysis']) > 1
            },
            'generated_at': datetime.now().isoformat()
        }
