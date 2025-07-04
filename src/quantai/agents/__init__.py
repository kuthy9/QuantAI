"""
Agent implementations for the QuantAI multi-agent system.

This module contains all the specialized agents that make up the financial
quantitative trading system, organized by their functional areas.
"""

# Data layer agents
try:
    from .data import *
except ImportError as e:
    print(f"Warning: Data layer agents import failed: {e}")

# Analysis layer agents
try:
    from .analysis import *
except ImportError as e:
    print(f"Warning: Analysis layer agents import failed: {e}")

# Validation layer agents
try:
    from .validation import *
except ImportError as e:
    print(f"Warning: Validation layer agents import failed: {e}")

# Execution layer agents
try:
    from .execution import *
except ImportError as e:
    print(f"Warning: Execution layer agents import failed: {e}")

# Learning layer agents
try:
    from .learning import *
except ImportError as e:
    print(f"Warning: Learning layer agents import failed: {e}")

# Control layer agents
try:
    from .control import *
except ImportError as e:
    print(f"Warning: Control layer agents import failed: {e}")

__all__ = [
    # Data layer
    "DataIngestionAgent",

    # Analysis layer
    "MacroInsightAgent",
    "MultimodalFusionAgent",
    "StrategyGenerationAgent",
    "StrategyCodingAgent",

    # Validation layer
    "StrategyValidationAgent",
    "RiskControlAgent",

    # Execution layer
    "StrategyDeploymentAgent",
    "ExecutionAgent",
    "BacktestMonitorAgent",

    # Learning layer
    "ProfitabilityAgent",
    "FeedbackLoopAgent",
    "MemoryAgent",

    # Control layer
    "APIManagerAgent",
    "KillSwitchAgent",
    "DashboardAgent",
]
