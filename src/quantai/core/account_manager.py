"""
Multi-Account Management System for QuantAI.

Provides support for multiple trading accounts with separate risk limits,
position tracking, and performance monitoring across different account types.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AccountType(str, Enum):
    """Types of trading accounts."""
    PAPER = "paper"
    LIVE = "live"
    DEMO = "demo"
    BACKTESTING = "backtesting"


class AccountStatus(str, Enum):
    """Account status types."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    CLOSED = "closed"


@dataclass
class RiskLimits:
    """Risk limits for an account."""
    max_portfolio_risk: float = 0.02  # 2% daily VaR
    max_position_size: float = 0.1    # 10% per position
    max_leverage: float = 2.0         # 2x leverage
    max_drawdown: float = 0.15        # 15% max drawdown
    max_daily_loss: float = 0.05      # 5% daily loss limit
    max_sector_exposure: float = 0.3  # 30% per sector
    max_correlation: float = 0.8      # Max correlation between positions
    min_liquidity_days: float = 5.0   # Minimum days to liquidate
    max_volatility: float = 0.25      # 25% annualized volatility
    
    # Asset-specific limits
    max_options_exposure: float = 0.2  # 20% options exposure
    max_futures_exposure: float = 0.3  # 30% futures exposure
    max_forex_exposure: float = 0.25   # 25% forex exposure
    
    # Order limits
    max_order_value: float = 100000    # $100k max single order
    max_daily_trades: int = 500        # Max trades per day
    max_pending_orders: int = 50       # Max pending orders


@dataclass
class AccountMetrics:
    """Performance and risk metrics for an account."""
    timestamp: datetime
    
    # Portfolio metrics
    total_value: float = 0.0
    cash_balance: float = 0.0
    equity_value: float = 0.0
    buying_power: float = 0.0
    
    # Performance metrics
    total_return: float = 0.0
    daily_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Risk metrics
    var_95: float = 0.0
    var_99: float = 0.0
    portfolio_beta: float = 1.0
    volatility: float = 0.0
    
    # Trading metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_trade_return: float = 0.0
    
    # Position metrics
    total_positions: int = 0
    long_positions: int = 0
    short_positions: int = 0
    largest_position_pct: float = 0.0


@dataclass
class TradingAccount:
    """Trading account with comprehensive tracking."""
    account_id: str
    account_name: str
    account_type: AccountType
    status: AccountStatus
    
    # Account configuration
    initial_capital: float
    current_capital: float
    risk_limits: RiskLimits
    
    # Broker configuration
    broker: str  # "ibkr", "alpaca", "paper"
    broker_account_id: Optional[str] = None
    api_credentials: Optional[Dict[str, str]] = None
    
    # Tracking data
    positions: Dict[str, Dict[str, Any]] = None
    orders: Dict[str, Dict[str, Any]] = None
    trades: List[Dict[str, Any]] = None
    metrics: Optional[AccountMetrics] = None
    
    # Metadata
    created_at: datetime = None
    last_updated: datetime = None
    tags: List[str] = None
    description: str = ""
    
    def __post_init__(self):
        """Initialize default values."""
        if self.positions is None:
            self.positions = {}
        if self.orders is None:
            self.orders = {}
        if self.trades is None:
            self.trades = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_updated is None:
            self.last_updated = datetime.now()
        if self.tags is None:
            self.tags = []
        if self.metrics is None:
            self.metrics = AccountMetrics(timestamp=datetime.now())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data['created_at'] = self.created_at.isoformat()
        data['last_updated'] = self.last_updated.isoformat()
        if self.metrics:
            data['metrics']['timestamp'] = self.metrics.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingAccount':
        """Create from dictionary."""
        # Convert ISO strings back to datetime objects
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        
        # Handle risk limits
        if isinstance(data['risk_limits'], dict):
            data['risk_limits'] = RiskLimits(**data['risk_limits'])
        
        # Handle metrics
        if data.get('metrics') and isinstance(data['metrics'], dict):
            metrics_data = data['metrics']
            metrics_data['timestamp'] = datetime.fromisoformat(metrics_data['timestamp'])
            data['metrics'] = AccountMetrics(**metrics_data)
        
        return cls(**data)


class MultiAccountManager:
    """
    Multi-Account Management System.
    
    Manages multiple trading accounts with separate risk limits,
    position tracking, and performance monitoring.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the multi-account manager."""
        self.config_file = config_file or "accounts_config.json"
        
        # Account storage
        self.accounts: Dict[str, TradingAccount] = {}
        self.active_account_id: Optional[str] = None
        
        # Performance tracking
        self.account_metrics_history: Dict[str, List[AccountMetrics]] = {}
        
        # Load existing accounts
        self._load_accounts()
        
        logger.info(f"Multi-Account Manager initialized with {len(self.accounts)} accounts")
    
    def create_account(
        self,
        account_id: str,
        account_name: str,
        account_type: AccountType,
        initial_capital: float,
        broker: str = "paper",
        risk_limits: Optional[RiskLimits] = None,
        **kwargs
    ) -> TradingAccount:
        """Create a new trading account."""
        
        if account_id in self.accounts:
            raise ValueError(f"Account {account_id} already exists")
        
        if risk_limits is None:
            risk_limits = RiskLimits()
        
        account = TradingAccount(
            account_id=account_id,
            account_name=account_name,
            account_type=account_type,
            status=AccountStatus.ACTIVE,
            initial_capital=initial_capital,
            current_capital=initial_capital,
            risk_limits=risk_limits,
            broker=broker,
            **kwargs
        )
        
        self.accounts[account_id] = account
        self.account_metrics_history[account_id] = []
        
        # Set as active if it's the first account
        if self.active_account_id is None:
            self.active_account_id = account_id
        
        self._save_accounts()
        
        logger.info(f"Created account {account_id}: {account_name} ({account_type.value})")
        return account
    
    def get_account(self, account_id: str) -> Optional[TradingAccount]:
        """Get account by ID."""
        return self.accounts.get(account_id)
    
    def get_active_account(self) -> Optional[TradingAccount]:
        """Get the currently active account."""
        if self.active_account_id:
            return self.accounts.get(self.active_account_id)
        return None
    
    def set_active_account(self, account_id: str) -> bool:
        """Set the active account."""
        if account_id not in self.accounts:
            logger.error(f"Account {account_id} not found")
            return False
        
        account = self.accounts[account_id]
        if account.status != AccountStatus.ACTIVE:
            logger.error(f"Account {account_id} is not active (status: {account.status.value})")
            return False
        
        self.active_account_id = account_id
        logger.info(f"Set active account to {account_id}")
        return True
    
    def list_accounts(self, account_type: Optional[AccountType] = None) -> List[TradingAccount]:
        """List all accounts, optionally filtered by type."""
        accounts = list(self.accounts.values())
        
        if account_type:
            accounts = [acc for acc in accounts if acc.account_type == account_type]
        
        return sorted(accounts, key=lambda x: x.created_at)
    
    def update_account_status(self, account_id: str, status: AccountStatus) -> bool:
        """Update account status."""
        if account_id not in self.accounts:
            logger.error(f"Account {account_id} not found")
            return False
        
        self.accounts[account_id].status = status
        self.accounts[account_id].last_updated = datetime.now()
        
        # If deactivating the active account, clear active account
        if status != AccountStatus.ACTIVE and account_id == self.active_account_id:
            self.active_account_id = None
        
        self._save_accounts()
        logger.info(f"Updated account {account_id} status to {status.value}")
        return True

    async def update_account_positions(
        self,
        account_id: str,
        positions: Dict[str, Dict[str, Any]]
    ) -> bool:
        """Update positions for an account."""
        if account_id not in self.accounts:
            logger.error(f"Account {account_id} not found")
            return False

        account = self.accounts[account_id]
        account.positions = positions.copy()
        account.last_updated = datetime.now()

        # Update account metrics
        await self._update_account_metrics(account_id)

        self._save_accounts()
        return True

    async def add_trade(
        self,
        account_id: str,
        trade: Dict[str, Any]
    ) -> bool:
        """Add a trade to an account."""
        if account_id not in self.accounts:
            logger.error(f"Account {account_id} not found")
            return False

        account = self.accounts[account_id]
        trade['timestamp'] = datetime.now().isoformat()
        account.trades.append(trade)
        account.last_updated = datetime.now()

        # Update account metrics
        await self._update_account_metrics(account_id)

        self._save_accounts()
        return True

    async def check_risk_limits(
        self,
        account_id: str,
        proposed_trade: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if a proposed trade violates risk limits."""
        if account_id not in self.accounts:
            return {
                "allowed": False,
                "reason": f"Account {account_id} not found"
            }

        account = self.accounts[account_id]
        risk_limits = account.risk_limits

        # Calculate current portfolio metrics
        portfolio_value = account.current_capital
        position_value = abs(proposed_trade.get('quantity', 0) * proposed_trade.get('price', 0))

        # Check position size limit
        if portfolio_value > 0:
            position_weight = position_value / portfolio_value
            if position_weight > risk_limits.max_position_size:
                return {
                    "allowed": False,
                    "reason": f"Position size {position_weight:.2%} exceeds limit {risk_limits.max_position_size:.2%}",
                    "limit_type": "position_size"
                }

        # Check order value limit
        if position_value > risk_limits.max_order_value:
            return {
                "allowed": False,
                "reason": f"Order value ${position_value:,.0f} exceeds limit ${risk_limits.max_order_value:,.0f}",
                "limit_type": "order_value"
            }

        # Check daily trades limit
        today_trades = len([
            t for t in account.trades
            if datetime.fromisoformat(t['timestamp']).date() == datetime.now().date()
        ])

        if today_trades >= risk_limits.max_daily_trades:
            return {
                "allowed": False,
                "reason": f"Daily trades limit {risk_limits.max_daily_trades} reached",
                "limit_type": "daily_trades"
            }

        # Check pending orders limit
        pending_orders = len([
            o for o in account.orders.values()
            if o.get('status') == 'pending'
        ])

        if pending_orders >= risk_limits.max_pending_orders:
            return {
                "allowed": False,
                "reason": f"Pending orders limit {risk_limits.max_pending_orders} reached",
                "limit_type": "pending_orders"
            }

        return {"allowed": True}

    async def _update_account_metrics(self, account_id: str):
        """Update metrics for an account."""
        if account_id not in self.accounts:
            return

        account = self.accounts[account_id]

        # Calculate portfolio value
        total_value = account.current_capital
        for position in account.positions.values():
            total_value += position.get('market_value', 0)

        # Calculate performance metrics
        total_return = (total_value - account.initial_capital) / account.initial_capital if account.initial_capital > 0 else 0.0

        # Calculate trading metrics
        total_trades = len(account.trades)
        winning_trades = len([t for t in account.trades if t.get('pnl', 0) > 0])
        losing_trades = len([t for t in account.trades if t.get('pnl', 0) < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # Calculate position metrics
        total_positions = len(account.positions)
        long_positions = len([p for p in account.positions.values() if p.get('quantity', 0) > 0])
        short_positions = len([p for p in account.positions.values() if p.get('quantity', 0) < 0])

        # Calculate largest position percentage
        largest_position_pct = 0.0
        if total_value > 0:
            position_values = [abs(p.get('market_value', 0)) for p in account.positions.values()]
            if position_values:
                largest_position_pct = max(position_values) / total_value

        # Update metrics
        metrics = AccountMetrics(
            timestamp=datetime.now(),
            total_value=total_value,
            cash_balance=account.current_capital,
            equity_value=total_value - account.current_capital,
            total_return=total_return,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_positions=total_positions,
            long_positions=long_positions,
            short_positions=short_positions,
            largest_position_pct=largest_position_pct
        )

        account.metrics = metrics

        # Add to history
        if account_id not in self.account_metrics_history:
            self.account_metrics_history[account_id] = []

        self.account_metrics_history[account_id].append(metrics)

        # Keep only last 1000 metrics entries
        if len(self.account_metrics_history[account_id]) > 1000:
            self.account_metrics_history[account_id] = self.account_metrics_history[account_id][-1000:]

    def get_account_summary(self) -> Dict[str, Any]:
        """Get summary of all accounts."""
        summary = {
            "total_accounts": len(self.accounts),
            "active_accounts": len([a for a in self.accounts.values() if a.status == AccountStatus.ACTIVE]),
            "total_capital": sum(a.current_capital for a in self.accounts.values()),
            "total_value": sum(a.metrics.total_value if a.metrics else a.current_capital for a in self.accounts.values()),
            "accounts": []
        }

        for account in self.accounts.values():
            account_info = {
                "account_id": account.account_id,
                "account_name": account.account_name,
                "account_type": account.account_type.value,
                "status": account.status.value,
                "broker": account.broker,
                "current_capital": account.current_capital,
                "total_value": account.metrics.total_value if account.metrics else account.current_capital,
                "total_return": account.metrics.total_return if account.metrics else 0.0,
                "total_positions": account.metrics.total_positions if account.metrics else 0,
                "total_trades": account.metrics.total_trades if account.metrics else 0,
                "is_active": account.account_id == self.active_account_id
            }
            summary["accounts"].append(account_info)

        return summary

    def _load_accounts(self):
        """Load accounts from configuration file."""
        try:
            config_path = Path(self.config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    data = json.load(f)

                # Load accounts
                for account_data in data.get('accounts', []):
                    account = TradingAccount.from_dict(account_data)
                    self.accounts[account.account_id] = account

                # Load active account
                self.active_account_id = data.get('active_account_id')

                # Load metrics history
                for account_id, metrics_list in data.get('metrics_history', {}).items():
                    self.account_metrics_history[account_id] = [
                        AccountMetrics(**{
                            **metrics_data,
                            'timestamp': datetime.fromisoformat(metrics_data['timestamp'])
                        })
                        for metrics_data in metrics_list
                    ]

                logger.info(f"Loaded {len(self.accounts)} accounts from {self.config_file}")

        except Exception as e:
            logger.warning(f"Could not load accounts from {self.config_file}: {e}")

    def _save_accounts(self):
        """Save accounts to configuration file."""
        try:
            data = {
                'accounts': [account.to_dict() for account in self.accounts.values()],
                'active_account_id': self.active_account_id,
                'metrics_history': {}
            }

            # Save metrics history
            for account_id, metrics_list in self.account_metrics_history.items():
                data['metrics_history'][account_id] = [
                    {
                        **asdict(metrics),
                        'timestamp': metrics.timestamp.isoformat()
                    }
                    for metrics in metrics_list
                ]

            config_path = Path(self.config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Saved {len(self.accounts)} accounts to {self.config_file}")

        except Exception as e:
            logger.error(f"Could not save accounts to {self.config_file}: {e}")

    async def close(self):
        """Close the account manager and save state."""
        self._save_accounts()
        logger.info("Multi-Account Manager closed")
