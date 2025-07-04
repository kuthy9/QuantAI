"""
Validation utilities for the QuantAI AutoGen system.
"""

from typing import Any, Dict, Optional
from datetime import datetime


def validate_message(message: Dict[str, Any]) -> bool:
    """
    Validate message structure.
    
    Args:
        message: Message dictionary
        
    Returns:
        True if message is valid
    """
    required_fields = ["message_type", "sender_id", "timestamp"]
    
    for field in required_fields:
        if field not in message:
            return False
    
    return True


def validate_agent_config(config: Dict[str, Any]) -> bool:
    """
    Validate agent configuration.
    
    Args:
        config: Agent configuration dictionary
        
    Returns:
        True if configuration is valid
    """
    required_fields = ["role", "capabilities"]
    
    for field in required_fields:
        if field not in config:
            return False
    
    return True


__all__ = ["validate_message", "validate_agent_config"]
