"""
Configuration management for the QuantAI system.

This module provides centralized configuration management for all agents,
models, trading parameters, and system settings.
"""

import os
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    """Configuration for AI models."""
    provider: str = "openai"  # "openai", "anthropic", "google"
    model_name: str = "gpt-4o"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    timeout: int = 60
    max_retries: int = 3


class TradingConfig(BaseModel):
    """Configuration for trading operations."""
    broker: str = "alpaca"  # "alpaca", "interactive_brokers", "paper"
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    base_url: Optional[str] = None
    paper_trading: bool = True
    
    # Risk management
    max_position_size: float = 0.1  # 10% of portfolio
    max_daily_loss: float = 0.02  # 2% daily loss limit
    max_drawdown: float = 0.15  # 15% max drawdown
    
    # Execution settings
    default_order_type: str = "market"
    execution_timeout: int = 30
    slippage_tolerance: float = 0.001  # 0.1%


class DataConfig(BaseModel):
    """Configuration for data sources."""
    # Market data
    market_data_provider: str = "yfinance"  # "yfinance", "alpha_vantage", "polygon"
    alpha_vantage_key: Optional[str] = None
    polygon_key: Optional[str] = None
    
    # News data
    news_provider: str = "newsapi"  # "newsapi", "finnhub", "alpha_vantage"
    newsapi_key: Optional[str] = None
    finnhub_key: Optional[str] = None
    
    # Data storage
    database_url: str = "sqlite:///quantai.db"
    redis_url: str = "redis://localhost:6379"
    
    # Data quality
    min_data_quality_score: float = 0.8
    max_data_age_hours: int = 24


class MemoryConfig(BaseModel):
    """Configuration for agent memory systems."""
    vector_db_provider: str = "chromadb"  # "chromadb", "faiss", "pinecone"
    embedding_model: str = "text-embedding-ada-002"
    collection_name: str = "quantai_memory"
    max_memory_items: int = 10000
    similarity_threshold: float = 0.7
    
    # Memory persistence
    persist_directory: str = "./data/memory"
    backup_interval_hours: int = 24


class AgentConfig(BaseModel):
    """Configuration for individual agents."""
    enabled: bool = True
    ai_model_config: ModelConfig
    max_concurrent_tasks: int = 5
    task_timeout_seconds: int = 300
    retry_attempts: int = 3
    
    # Agent-specific settings
    custom_settings: Dict[str, Any] = Field(default_factory=dict)


class SystemConfig(BaseModel):
    """Configuration for system-wide settings."""
    # Runtime settings
    max_agents: int = 50
    message_queue_size: int = 1000
    heartbeat_interval: int = 30
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/quantai.log"
    log_rotation: str = "1 day"
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 8080
    dashboard_port: int = 8501
    
    # Security
    enable_auth: bool = False
    jwt_secret: Optional[str] = None
    api_rate_limit: int = 100  # requests per minute


class QuantConfig(BaseSettings):
    """Main configuration class for the QuantAI system."""
    
    # Environment
    environment: str = Field(default="development", env="QUANTAI_ENV")
    debug: bool = Field(default=False, env="QUANTAI_DEBUG")
    
    # System configuration
    system: SystemConfig = Field(default_factory=SystemConfig)
    
    # Data configuration
    data: DataConfig = Field(default_factory=DataConfig)
    
    # Memory configuration
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    
    # Trading configuration
    trading: TradingConfig = Field(default_factory=TradingConfig)
    
    # Model configurations for different agents
    models: Dict[str, ModelConfig] = Field(default_factory=lambda: {
        "claude": ModelConfig(
            provider="anthropic",
            model_name="claude-3-sonnet-20240229",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        ),
        "gpt4": ModelConfig(
            provider="openai", 
            model_name="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        "gemini": ModelConfig(
            provider="google",
            model_name="gemini-pro",
            api_key=os.getenv("GOOGLE_API_KEY"),
        ),
    })
    
    # Agent configurations
    agents: Dict[str, AgentConfig] = Field(default_factory=lambda: {
        "data_ingestion": AgentConfig(
            ai_model_config=ModelConfig(provider="claude", model_name="claude-3-sonnet-20240229"),
            custom_settings={
                "crawl_interval_minutes": 15,
                "max_concurrent_crawls": 10,
            }
        ),
        "multimodal_fusion": AgentConfig(
            ai_model_config=ModelConfig(provider="gemini", model_name="gemini-pro"),
            custom_settings={
                "fusion_algorithms": ["weighted_average", "attention", "ensemble"],
            }
        ),
        "macro_insight": AgentConfig(
            ai_model_config=ModelConfig(provider="gemini", model_name="gemini-pro"),
        ),
        "strategy_generation": AgentConfig(
            ai_model_config=ModelConfig(provider="gemini", model_name="gemini-pro"),
        ),
        "strategy_coding": AgentConfig(
            ai_model_config=ModelConfig(provider="claude", model_name="claude-3-sonnet-20240229"),
        ),
        "strategy_validation": AgentConfig(
            ai_model_config=ModelConfig(provider="claude", model_name="claude-3-sonnet-20240229"),
        ),
        "risk_control": AgentConfig(
            ai_model_config=ModelConfig(provider="claude", model_name="claude-3-sonnet-20240229"),
        ),
        "execution": AgentConfig(
            ai_model_config=ModelConfig(provider="gemini", model_name="gemini-pro"),
        ),
        "backtest_monitor": AgentConfig(
            ai_model_config=ModelConfig(provider="claude", model_name="claude-3-sonnet-20240229"),
        ),
        "profitability": AgentConfig(
            ai_model_config=ModelConfig(provider="gemini", model_name="gemini-pro"),
        ),
        "feedback_loop": AgentConfig(
            ai_model_config=ModelConfig(provider="claude", model_name="claude-3-sonnet-20240229"),
        ),
        "memory": AgentConfig(
            ai_model_config=ModelConfig(provider="claude", model_name="claude-3-sonnet-20240229"),
        ),
    })

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"  # Allow extra fields from environment
    )

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        if v not in ["development", "staging", "production"]:
            raise ValueError("Environment must be development, staging, or production")
        return v

    def get_agent_config(self, agent_role: str) -> AgentConfig:
        """Get configuration for a specific agent."""
        return self.agents.get(agent_role, AgentConfig(
            ai_model_config=self.models["gpt4"]
        ))

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model."""
        return self.models.get(model_name, self.models["gpt4"])

    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from a dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.dict()

    def save_to_file(self, file_path: Union[str, Path]):
        """Save configuration to a file."""
        import json
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> "QuantConfig":
        """Load configuration from a file."""
        import json
        
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)


# Global configuration instance
config = QuantConfig()


def get_config() -> QuantConfig:
    """Get the global configuration instance."""
    return config


def update_config(config_dict: Dict[str, Any]):
    """Update the global configuration."""
    global config
    config.update_from_dict(config_dict)
