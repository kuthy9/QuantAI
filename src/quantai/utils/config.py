"""
Configuration utilities for the QuantAI AutoGen system.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from environment variables and config files.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        Configuration dictionary
    """
    # Load environment variables
    load_dotenv()
    
    config = {
        # API Keys
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        "google_api_key": os.getenv("GOOGLE_API_KEY"),
        "alpha_vantage_api_key": os.getenv("ALPHA_VANTAGE_API_KEY"),
        "news_api_key": os.getenv("NEWS_API_KEY"),
        
        # System settings
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "debug_mode": os.getenv("DEBUG_MODE", "false").lower() == "true",
        
        # Trading settings
        "paper_trading": os.getenv("PAPER_TRADING", "true").lower() == "true",
        "max_position_size": float(os.getenv("MAX_POSITION_SIZE", "0.1")),
        "risk_limit": float(os.getenv("RISK_LIMIT", "0.02")),
    }
    
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration settings.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if configuration is valid
    """
    required_keys = ["openai_api_key", "anthropic_api_key"]
    
    for key in required_keys:
        if not config.get(key):
            print(f"Warning: {key} not configured")
            return False
    
    return True


__all__ = ["load_config", "validate_config"]
