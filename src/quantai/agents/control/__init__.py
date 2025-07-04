"""
Control layer agents for the QuantAI system.

These agents handle system management, security, emergency controls,
and external interfaces for the quantitative trading system.
"""

from .api_manager import APIManagerAgent
from .kill_switch import KillSwitchAgent
from .dashboard import DashboardAgent

__all__ = [
    "APIManagerAgent",
    "KillSwitchAgent",
    "DashboardAgent",
]
