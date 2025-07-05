#!/usr/bin/env python3
"""
Health check script for QuantAI application.

This script performs comprehensive health checks including:
- API server responsiveness
- Database connectivity
- Redis connectivity
- ChromaDB connectivity
- Agent system status
"""

import asyncio
import json
import os
import sys
import time
from typing import Dict, Any

import aiohttp
import asyncpg
import redis.asyncio as redis
import chromadb
from loguru import logger


class HealthChecker:
    """Comprehensive health checker for QuantAI system."""
    
    def __init__(self):
        self.api_url = "http://localhost:8000"
        self.db_url = os.getenv("DATABASE_URL", "postgresql://quantai:quantai@postgres:5432/quantai")
        self.redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        self.chroma_host = os.getenv("CHROMA_HOST", "chroma")
        self.chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
        
    async def check_api_server(self) -> Dict[str, Any]:
        """Check API server health."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/health", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {"status": "healthy", "response_time": data.get("response_time", 0)}
                    else:
                        return {"status": "unhealthy", "error": f"HTTP {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def check_database(self) -> Dict[str, Any]:
        """Check PostgreSQL database connectivity."""
        try:
            conn = await asyncpg.connect(self.db_url)
            result = await conn.fetchval("SELECT 1")
            await conn.close()
            
            if result == 1:
                return {"status": "healthy", "connection": "active"}
            else:
                return {"status": "unhealthy", "error": "Invalid query result"}
                
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity."""
        try:
            r = redis.from_url(self.redis_url)
            await r.ping()
            info = await r.info()
            await r.close()
            
            return {
                "status": "healthy",
                "memory_usage": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0)
            }
            
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def check_chroma(self) -> Dict[str, Any]:
        """Check ChromaDB connectivity."""
        try:
            client = chromadb.HttpClient(host=self.chroma_host, port=self.chroma_port)
            collections = client.list_collections()
            
            return {
                "status": "healthy",
                "collections_count": len(collections)
            }
            
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "status": "healthy" if cpu_percent < 90 and memory.percent < 90 else "warning",
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent
            }
            
        except ImportError:
            return {"status": "unknown", "error": "psutil not available"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        start_time = time.time()
        
        checks = {
            "api_server": await self.check_api_server(),
            "database": await self.check_database(),
            "redis": await self.check_redis(),
            "chroma": await self.check_chroma(),
            "system_resources": await self.check_system_resources()
        }
        
        # Determine overall health
        unhealthy_services = [name for name, check in checks.items() if check["status"] == "unhealthy"]
        warning_services = [name for name, check in checks.items() if check["status"] == "warning"]
        
        if unhealthy_services:
            overall_status = "unhealthy"
        elif warning_services:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        return {
            "overall_status": overall_status,
            "timestamp": time.time(),
            "check_duration": time.time() - start_time,
            "services": checks,
            "unhealthy_services": unhealthy_services,
            "warning_services": warning_services
        }


async def main():
    """Main health check function."""
    checker = HealthChecker()
    
    try:
        results = await checker.run_health_checks()
        
        # Print results for logging
        print(json.dumps(results, indent=2))
        
        # Exit with appropriate code
        if results["overall_status"] == "healthy":
            sys.exit(0)
        elif results["overall_status"] == "warning":
            sys.exit(0)  # Still considered healthy for Docker
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        print(json.dumps({"overall_status": "unhealthy", "error": str(e)}, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
