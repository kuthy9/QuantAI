"""
Memory management system for the QuantAI agents.

This module provides persistent and vector-based memory capabilities
for agents to store and retrieve strategies, market insights, and performance data.
"""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None
    Settings = None

import numpy as np
from loguru import logger
from pydantic import BaseModel

from .config import MemoryConfig


class MemoryEntry(BaseModel):
    """Base class for memory entries."""
    id: str = None
    content: str
    metadata: Dict[str, Any] = {}
    embedding: Optional[List[float]] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __init__(self, **data):
        if data.get('id') is None:
            data['id'] = str(uuid.uuid4())
        if data.get('created_at') is None:
            data['created_at'] = datetime.utcnow()
        if data.get('updated_at') is None:
            data['updated_at'] = datetime.utcnow()
        super().__init__(**data)


class StrategyMemory(MemoryEntry):
    """Memory entry for trading strategies."""
    strategy_id: str
    strategy_name: str
    strategy_code: str
    performance_metrics: Dict[str, float] = {}
    market_conditions: Dict[str, Any] = {}
    success_factors: List[str] = []
    failure_factors: List[str] = []


class MarketMemory(MemoryEntry):
    """Memory entry for market insights."""
    symbol: str
    timeframe: str
    market_regime: str
    key_insights: List[str] = []
    technical_indicators: Dict[str, float] = {}
    sentiment_score: Optional[float] = None


class PerformanceMemory(MemoryEntry):
    """Memory entry for performance tracking."""
    strategy_id: str
    period_start: datetime
    period_end: datetime
    returns: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    lessons_learned: List[str] = []


class BaseMemoryStore(ABC):
    """Abstract base class for memory storage backends."""
    
    @abstractmethod
    async def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry and return its ID."""
        pass
    
    @abstractmethod
    async def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID."""
        pass
    
    @abstractmethod
    async def search(
        self, 
        query: str, 
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MemoryEntry]:
        """Search for memory entries using semantic similarity."""
        pass
    
    @abstractmethod
    async def update(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory entry."""
        pass
    
    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        pass
    
    @abstractmethod
    async def list_entries(
        self, 
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MemoryEntry]:
        """List memory entries with optional filtering."""
        pass


class ChromaMemoryStore(BaseMemoryStore):
    """ChromaDB-based memory store with vector similarity search."""

    def __init__(self, config: MemoryConfig):
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is not available. Please install it with: pip install chromadb")

        self.config = config
        self.client = None
        self.collection = None
        self._embedding_function = None
    
    async def initialize(self):
        """Initialize the ChromaDB client and collection."""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.config.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"description": "QuantAI agent memory store"}
            )
            
            logger.info(f"Initialized ChromaDB memory store with collection: {self.config.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB memory store: {e}")
            raise
    
    async def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry in ChromaDB."""
        if not self.collection:
            await self.initialize()
        
        try:
            # Prepare metadata
            metadata = {
                "type": entry.__class__.__name__,
                "created_at": entry.created_at.isoformat(),
                "updated_at": entry.updated_at.isoformat(),
                **entry.metadata
            }
            
            # Add to collection
            self.collection.add(
                ids=[entry.id],
                documents=[entry.content],
                metadatas=[metadata],
                embeddings=[entry.embedding] if entry.embedding else None
            )
            
            logger.debug(f"Stored memory entry {entry.id} of type {entry.__class__.__name__}")
            return entry.id
            
        except Exception as e:
            logger.error(f"Failed to store memory entry: {e}")
            raise
    
    async def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID."""
        if not self.collection:
            await self.initialize()
        
        try:
            results = self.collection.get(ids=[entry_id])
            
            if not results['ids']:
                return None
            
            # Reconstruct memory entry
            metadata = results['metadatas'][0]
            entry_type = metadata.pop('type', 'MemoryEntry')
            
            entry_data = {
                'id': results['ids'][0],
                'content': results['documents'][0],
                'metadata': metadata,
                'embedding': results['embeddings'][0] if results['embeddings'] else None,
                'created_at': datetime.fromisoformat(metadata.pop('created_at')),
                'updated_at': datetime.fromisoformat(metadata.pop('updated_at')),
            }
            
            # Create appropriate memory entry type
            if entry_type == 'StrategyMemory':
                return StrategyMemory(**entry_data)
            elif entry_type == 'MarketMemory':
                return MarketMemory(**entry_data)
            elif entry_type == 'PerformanceMemory':
                return PerformanceMemory(**entry_data)
            else:
                return MemoryEntry(**entry_data)
                
        except Exception as e:
            logger.error(f"Failed to retrieve memory entry {entry_id}: {e}")
            return None
    
    async def search(
        self, 
        query: str, 
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MemoryEntry]:
        """Search for memory entries using semantic similarity."""
        if not self.collection:
            await self.initialize()
        
        try:
            # Build where clause for filtering
            where_clause = {}
            if filters:
                where_clause.update(filters)
            
            # Perform similarity search
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_clause if where_clause else None
            )
            
            entries = []
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                entry_type = metadata.pop('type', 'MemoryEntry')
                
                entry_data = {
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': metadata,
                    'created_at': datetime.fromisoformat(metadata.pop('created_at')),
                    'updated_at': datetime.fromisoformat(metadata.pop('updated_at')),
                }
                
                # Create appropriate memory entry type
                if entry_type == 'StrategyMemory':
                    entries.append(StrategyMemory(**entry_data))
                elif entry_type == 'MarketMemory':
                    entries.append(MarketMemory(**entry_data))
                elif entry_type == 'PerformanceMemory':
                    entries.append(PerformanceMemory(**entry_data))
                else:
                    entries.append(MemoryEntry(**entry_data))
            
            return entries
            
        except Exception as e:
            logger.error(f"Failed to search memory entries: {e}")
            return []
    
    async def update(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory entry."""
        try:
            # Get existing entry
            entry = await self.retrieve(entry_id)
            if not entry:
                return False
            
            # Update fields
            for key, value in updates.items():
                if hasattr(entry, key):
                    setattr(entry, key, value)
            
            entry.updated_at = datetime.utcnow()
            
            # Delete old entry and store updated one
            await self.delete(entry_id)
            await self.store(entry)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update memory entry {entry_id}: {e}")
            return False
    
    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        if not self.collection:
            await self.initialize()
        
        try:
            self.collection.delete(ids=[entry_id])
            logger.debug(f"Deleted memory entry {entry_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete memory entry {entry_id}: {e}")
            return False
    
    async def list_entries(
        self, 
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MemoryEntry]:
        """List memory entries with optional filtering."""
        if not self.collection:
            await self.initialize()
        
        try:
            where_clause = filters if filters else None
            
            results = self.collection.get(
                limit=limit,
                where=where_clause
            )
            
            entries = []
            for i in range(len(results['ids'])):
                metadata = results['metadatas'][i]
                entry_type = metadata.pop('type', 'MemoryEntry')
                
                entry_data = {
                    'id': results['ids'][i],
                    'content': results['documents'][i],
                    'metadata': metadata,
                    'created_at': datetime.fromisoformat(metadata.pop('created_at')),
                    'updated_at': datetime.fromisoformat(metadata.pop('updated_at')),
                }
                
                # Create appropriate memory entry type
                if entry_type == 'StrategyMemory':
                    entries.append(StrategyMemory(**entry_data))
                elif entry_type == 'MarketMemory':
                    entries.append(MarketMemory(**entry_data))
                elif entry_type == 'PerformanceMemory':
                    entries.append(PerformanceMemory(**entry_data))
                else:
                    entries.append(MemoryEntry(**entry_data))
            
            return entries
            
        except Exception as e:
            logger.error(f"Failed to list memory entries: {e}")
            return []


class QuantMemory:
    """
    Main memory interface for QuantAI agents.
    
    Provides high-level memory operations with automatic categorization
    and intelligent retrieval capabilities.
    """
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.store = ChromaMemoryStore(config)
        self._initialized = False
    
    async def initialize(self):
        """Initialize the memory system."""
        if not self._initialized:
            await self.store.initialize()
            self._initialized = True
            logger.info("QuantMemory system initialized")
    
    async def store_strategy(
        self, 
        strategy_id: str,
        strategy_name: str,
        strategy_code: str,
        performance_metrics: Dict[str, float] = None,
        market_conditions: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Store a trading strategy in memory."""
        if not self._initialized:
            await self.initialize()
        
        content = f"Strategy: {strategy_name}\n\nCode:\n{strategy_code}"
        if performance_metrics:
            content += f"\n\nPerformance: {json.dumps(performance_metrics, indent=2)}"
        
        entry = StrategyMemory(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            strategy_code=strategy_code,
            content=content,
            performance_metrics=performance_metrics or {},
            market_conditions=market_conditions or {},
            metadata=metadata or {}
        )
        
        return await self.store.store(entry)
    
    async def store_market_insight(
        self,
        symbol: str,
        timeframe: str,
        market_regime: str,
        insights: List[str],
        technical_indicators: Dict[str, float] = None,
        sentiment_score: float = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Store market insights in memory."""
        if not self._initialized:
            await self.initialize()
        
        content = f"Market Analysis for {symbol} ({timeframe})\n"
        content += f"Regime: {market_regime}\n\n"
        content += "Key Insights:\n" + "\n".join([f"- {insight}" for insight in insights])
        
        if technical_indicators:
            content += f"\n\nTechnical Indicators:\n{json.dumps(technical_indicators, indent=2)}"
        
        entry = MarketMemory(
            symbol=symbol,
            timeframe=timeframe,
            market_regime=market_regime,
            key_insights=insights,
            content=content,
            technical_indicators=technical_indicators or {},
            sentiment_score=sentiment_score,
            metadata=metadata or {}
        )
        
        return await self.store.store(entry)
    
    async def store_performance(
        self,
        strategy_id: str,
        period_start: datetime,
        period_end: datetime,
        returns: float,
        sharpe_ratio: float,
        max_drawdown: float,
        win_rate: float,
        lessons_learned: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Store performance data in memory."""
        if not self._initialized:
            await self.initialize()
        
        content = f"Performance Report for Strategy {strategy_id}\n"
        content += f"Period: {period_start.date()} to {period_end.date()}\n"
        content += f"Returns: {returns:.2%}\n"
        content += f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
        content += f"Max Drawdown: {max_drawdown:.2%}\n"
        content += f"Win Rate: {win_rate:.2%}\n"
        
        if lessons_learned:
            content += "\nLessons Learned:\n" + "\n".join([f"- {lesson}" for lesson in lessons_learned])
        
        entry = PerformanceMemory(
            strategy_id=strategy_id,
            period_start=period_start,
            period_end=period_end,
            returns=returns,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            content=content,
            lessons_learned=lessons_learned or [],
            metadata=metadata or {}
        )
        
        return await self.store.store(entry)
    
    async def search_strategies(
        self, 
        query: str, 
        limit: int = 10
    ) -> List[StrategyMemory]:
        """Search for strategies based on query."""
        if not self._initialized:
            await self.initialize()
        
        results = await self.store.search(
            query=query,
            limit=limit,
            filters={"type": "StrategyMemory"}
        )
        
        return [entry for entry in results if isinstance(entry, StrategyMemory)]
    
    async def search_market_insights(
        self, 
        query: str, 
        symbol: str = None,
        limit: int = 10
    ) -> List[MarketMemory]:
        """Search for market insights."""
        if not self._initialized:
            await self.initialize()
        
        filters = {"type": "MarketMemory"}
        if symbol:
            filters["symbol"] = symbol
        
        results = await self.store.search(
            query=query,
            limit=limit,
            filters=filters
        )
        
        return [entry for entry in results if isinstance(entry, MarketMemory)]
    
    async def get_strategy_performance(self, strategy_id: str) -> List[PerformanceMemory]:
        """Get performance history for a strategy."""
        if not self._initialized:
            await self.initialize()
        
        results = await self.store.list_entries(
            filters={
                "type": "PerformanceMemory",
                "strategy_id": strategy_id
            }
        )
        
        return [entry for entry in results if isinstance(entry, PerformanceMemory)]
    
    async def get_recent_insights(self, limit: int = 20) -> List[MemoryEntry]:
        """Get recent insights across all categories."""
        if not self._initialized:
            await self.initialize()
        
        return await self.store.list_entries(limit=limit)
    
    async def cleanup_old_entries(self, days_old: int = 90):
        """Clean up old memory entries to manage storage."""
        if not self._initialized:
            await self.initialize()
        
        # This would need to be implemented based on the specific storage backend
        # For now, we'll just log the intent
        logger.info(f"Cleanup requested for entries older than {days_old} days")
        # TODO: Implement cleanup logic
