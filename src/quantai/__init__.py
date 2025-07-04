"""
QuantAI AutoGen - Multi-agent AI system for automated financial quantitative algorithm generation.

This package provides a comprehensive multi-agent system built on Microsoft AutoGen for:
- Automated financial data ingestion and processing
- Multimodal signal fusion (text, images, structured data)
- Quantitative strategy generation and coding
- Risk management and validation
- Backtesting and live execution
- Continuous learning and optimization

Architecture:
- Data Layer: D1 (Data Ingestion), D4 (Multimodal Fusion)
- Analysis Layer: A0 (Macro Insight), A1 (Strategy Generation), A2 (Strategy Coding)
- Validation Layer: D5 (Strategy Validator), D2 (Risk Control)
- Execution Layer: A3 (Deployment), D3 (Execution), A4 (Backtest Monitor)
- Learning Layer: A6 (Profitability), A5 (Feedback Loop), D6 (Memory)
- Control Layer: M1 (API Manager), M3 (Kill Switch), V0 (Dashboard)
"""

__version__ = "0.1.0"
__author__ = "QuantAI Team"
__email__ = "team@quantai.dev"

from .core import *
from .agents import *
from .models import *
from .utils import *

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
]
