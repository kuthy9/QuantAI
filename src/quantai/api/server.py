"""
FastAPI server for the QuantAI REST API control plane.

Provides HTTP endpoints for system control, monitoring, and external integration.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from loguru import logger
from pydantic import BaseModel

from ..core.runtime import QuantRuntime
from ..core.messages import ControlMessage, MessageType
from .auth import AuthManager


class APIServer:
    """
    REST API server for QuantAI system control and monitoring.
    
    Provides endpoints for:
    - System status and health monitoring
    - Strategy management and control
    - Risk monitoring and alerts
    - Performance analytics
    - Emergency controls
    - Configuration management
    """
    
    def __init__(
        self,
        runtime: QuantRuntime,
        host: str = "0.0.0.0",
        port: int = 8000,
        enable_auth: bool = True,
        cors_origins: List[str] = None,
    ):
        self.runtime = runtime
        self.host = host
        self.port = port
        self.enable_auth = enable_auth
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="QuantAI Control Plane API",
            description="REST API for QuantAI multi-agent trading system",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins or ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize auth manager
        self.auth_manager = AuthManager() if enable_auth else None
        self.security = HTTPBearer() if enable_auth else None
        
        # Setup routes
        self._setup_routes()
        
        # Server state
        self._server_running = False
        self._start_time = None
    
    def _setup_routes(self):
        """Setup API routes."""
        
        # Health and status endpoints
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """System health check endpoint."""
            return await self._get_health_status()
        
        @self.app.get("/status", response_model=SystemStatusResponse)
        async def system_status(auth: HTTPAuthorizationCredentials = Depends(self.security)):
            """Get comprehensive system status."""
            if self.enable_auth:
                await self._verify_auth(auth)
            
            return await self._get_system_status()
        
        # Strategy management endpoints
        @self.app.get("/strategies", response_model=List[StrategyInfo])
        async def list_strategies(auth: HTTPAuthorizationCredentials = Depends(self.security)):
            """List all strategies."""
            if self.enable_auth:
                await self._verify_auth(auth)
            
            return await self._list_strategies()
        
        @self.app.get("/strategies/{strategy_id}", response_model=StrategyDetails)
        async def get_strategy(
            strategy_id: str,
            auth: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Get detailed strategy information."""
            if self.enable_auth:
                await self._verify_auth(auth)
            
            return await self._get_strategy_details(strategy_id)
        
        @self.app.post("/strategies/{strategy_id}/start")
        async def start_strategy(
            strategy_id: str,
            auth: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Start a strategy."""
            if self.enable_auth:
                await self._verify_auth(auth)
            
            return await self._start_strategy(strategy_id)
        
        @self.app.post("/strategies/{strategy_id}/stop")
        async def stop_strategy(
            strategy_id: str,
            auth: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Stop a strategy."""
            if self.enable_auth:
                await self._verify_auth(auth)
            
            return await self._stop_strategy(strategy_id)
        
        @self.app.post("/strategies/{strategy_id}/pause")
        async def pause_strategy(
            strategy_id: str,
            auth: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Pause a strategy."""
            if self.enable_auth:
                await self._verify_auth(auth)
            
            return await self._pause_strategy(strategy_id)
        
        # Risk management endpoints
        @self.app.get("/risk/status", response_model=RiskStatusResponse)
        async def risk_status(auth: HTTPAuthorizationCredentials = Depends(self.security)):
            """Get current risk status."""
            if self.enable_auth:
                await self._verify_auth(auth)
            
            return await self._get_risk_status()
        
        @self.app.get("/risk/alerts", response_model=List[RiskAlert])
        async def risk_alerts(
            limit: int = 50,
            auth: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Get recent risk alerts."""
            if self.enable_auth:
                await self._verify_auth(auth)
            
            return await self._get_risk_alerts(limit)
        
        # Performance endpoints
        @self.app.get("/performance/portfolio", response_model=PortfolioPerformance)
        async def portfolio_performance(auth: HTTPAuthorizationCredentials = Depends(self.security)):
            """Get portfolio performance metrics."""
            if self.enable_auth:
                await self._verify_auth(auth)
            
            return await self._get_portfolio_performance()
        
        @self.app.get("/performance/strategies", response_model=List[StrategyPerformance])
        async def strategies_performance(auth: HTTPAuthorizationCredentials = Depends(self.security)):
            """Get performance metrics for all strategies."""
            if self.enable_auth:
                await self._verify_auth(auth)
            
            return await self._get_strategies_performance()
        
        # Emergency control endpoints
        @self.app.post("/emergency/stop")
        async def emergency_stop(
            request: EmergencyStopRequest,
            auth: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Activate emergency stop."""
            if self.enable_auth:
                await self._verify_auth(auth, required_role="admin")
            
            return await self._emergency_stop(request.reason)
        
        @self.app.post("/emergency/liquidate")
        async def force_liquidation(
            request: LiquidationRequest,
            auth: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Force liquidation of positions."""
            if self.enable_auth:
                await self._verify_auth(auth, required_role="admin")
            
            return await self._force_liquidation(request.reason, request.symbols)
        
        @self.app.post("/emergency/reset")
        async def reset_emergency(
            request: EmergencyResetRequest,
            auth: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Reset emergency state."""
            if self.enable_auth:
                await self._verify_auth(auth, required_role="admin")
            
            return await self._reset_emergency(request.authorized_by)
        
        # Configuration endpoints
        @self.app.get("/config", response_model=SystemConfig)
        async def get_config(auth: HTTPAuthorizationCredentials = Depends(self.security)):
            """Get system configuration."""
            if self.enable_auth:
                await self._verify_auth(auth)
            
            return await self._get_system_config()
        
        @self.app.put("/config")
        async def update_config(
            config: SystemConfigUpdate,
            auth: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Update system configuration."""
            if self.enable_auth:
                await self._verify_auth(auth, required_role="admin")
            
            return await self._update_system_config(config)
        
        # Agent management endpoints
        @self.app.get("/agents", response_model=List[AgentInfo])
        async def list_agents(auth: HTTPAuthorizationCredentials = Depends(self.security)):
            """List all agents."""
            if self.enable_auth:
                await self._verify_auth(auth)
            
            return await self._list_agents()
        
        @self.app.get("/agents/{agent_id}", response_model=AgentDetails)
        async def get_agent(
            agent_id: str,
            auth: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Get detailed agent information."""
            if self.enable_auth:
                await self._verify_auth(auth)
            
            return await self._get_agent_details(agent_id)
        
        # Dashboard data endpoints
        @self.app.get("/dashboard/data", response_model=DashboardData)
        async def dashboard_data(auth: HTTPAuthorizationCredentials = Depends(self.security)):
            """Get dashboard data."""
            if self.enable_auth:
                await self._verify_auth(auth)
            
            return await self._get_dashboard_data()
        
        # WebSocket endpoint for real-time updates
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket):
            """WebSocket endpoint for real-time updates."""
            await self._handle_websocket(websocket)
    
    async def _verify_auth(self, auth: HTTPAuthorizationCredentials, required_role: str = "user"):
        """Verify authentication and authorization."""
        if not self.auth_manager:
            return  # Auth disabled
        
        try:
            user_info = await self.auth_manager.verify_token(auth.credentials)
            
            if not user_info:
                raise HTTPException(status_code=401, detail="Invalid authentication token")
            
            if required_role == "admin" and user_info.get("role") != "admin":
                raise HTTPException(status_code=403, detail="Admin access required")
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise HTTPException(status_code=401, detail="Authentication failed")
    
    async def _get_health_status(self) -> HealthResponse:
        """Get system health status."""
        
        try:
            # Check runtime health
            runtime_health = await self.runtime.health_check()
            
            return HealthResponse(
                status="healthy" if runtime_health["status"] == "healthy" else "unhealthy",
                timestamp=datetime.now(),
                uptime_seconds=(datetime.now() - self._start_time).total_seconds() if self._start_time else 0,
                version="1.0.0",
                details=runtime_health
            )
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return HealthResponse(
                status="unhealthy",
                timestamp=datetime.now(),
                uptime_seconds=0,
                version="1.0.0",
                details={"error": str(e)}
            )
    
    async def _get_system_status(self) -> SystemStatusResponse:
        """Get comprehensive system status."""
        
        try:
            status = await self.runtime.get_system_status()
            
            return SystemStatusResponse(
                runtime_status=status["runtime_status"],
                total_agents=status["total_agents"],
                agents_by_role=status["agents_by_role"],
                uptime_seconds=status["uptime_seconds"],
                memory_usage=status.get("memory_usage", {}),
                performance_metrics=status.get("performance_metrics", {}),
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"System status error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get system status: {e}")
    
    async def _list_strategies(self) -> List[StrategyInfo]:
        """List all strategies."""
        
        try:
            # Send request to get strategies
            request = ControlMessage(
                message_type=MessageType.CONTROL,
                sender_id="api_server",
                command="list_strategies",
                parameters={}
            )
            
            response = await self.runtime.send_message(request)
            
            # Convert to API response format
            strategies = []
            if response and "strategies" in response.parameters:
                for strategy_data in response.parameters["strategies"]:
                    strategies.append(StrategyInfo(
                        strategy_id=strategy_data["strategy_id"],
                        name=strategy_data["name"],
                        status=strategy_data["status"],
                        performance=strategy_data.get("performance", {}),
                        created_at=datetime.fromisoformat(strategy_data["created_at"]),
                        last_updated=datetime.fromisoformat(strategy_data["last_updated"])
                    ))
            
            return strategies
            
        except Exception as e:
            logger.error(f"List strategies error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to list strategies: {e}")
    
    async def _get_strategy_details(self, strategy_id: str) -> StrategyDetails:
        """Get detailed strategy information."""
        
        try:
            request = ControlMessage(
                message_type=MessageType.CONTROL,
                sender_id="api_server",
                command="get_strategy_details",
                parameters={"strategy_id": strategy_id}
            )
            
            response = await self.runtime.send_message(request)
            
            if not response or "strategy" not in response.parameters:
                raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
            
            strategy_data = response.parameters["strategy"]
            
            return StrategyDetails(
                strategy_id=strategy_data["strategy_id"],
                name=strategy_data["name"],
                description=strategy_data["description"],
                status=strategy_data["status"],
                code=strategy_data.get("code", ""),
                parameters=strategy_data.get("parameters", {}),
                performance=strategy_data.get("performance", {}),
                risk_metrics=strategy_data.get("risk_metrics", {}),
                created_at=datetime.fromisoformat(strategy_data["created_at"]),
                last_updated=datetime.fromisoformat(strategy_data["last_updated"])
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Get strategy details error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get strategy details: {e}")
    
    async def _start_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """Start a strategy."""
        
        try:
            request = ControlMessage(
                message_type=MessageType.CONTROL,
                sender_id="api_server",
                command="start_strategy",
                parameters={"strategy_id": strategy_id}
            )
            
            response = await self.runtime.send_message(request)
            
            return {
                "strategy_id": strategy_id,
                "action": "start",
                "status": "success" if response else "failed",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Start strategy error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to start strategy: {e}")
    
    async def _stop_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """Stop a strategy."""
        
        try:
            request = ControlMessage(
                message_type=MessageType.CONTROL,
                sender_id="api_server",
                command="stop_strategy",
                parameters={"strategy_id": strategy_id}
            )
            
            response = await self.runtime.send_message(request)
            
            return {
                "strategy_id": strategy_id,
                "action": "stop",
                "status": "success" if response else "failed",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Stop strategy error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to stop strategy: {e}")
    
    async def _pause_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """Pause a strategy."""
        
        try:
            request = ControlMessage(
                message_type=MessageType.CONTROL,
                sender_id="api_server",
                command="pause_strategy",
                parameters={"strategy_id": strategy_id}
            )
            
            response = await self.runtime.send_message(request)
            
            return {
                "strategy_id": strategy_id,
                "action": "pause",
                "status": "success" if response else "failed",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Pause strategy error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to pause strategy: {e}")
    
    async def _get_risk_status(self) -> RiskStatusResponse:
        """Get current risk status."""
        
        try:
            request = ControlMessage(
                message_type=MessageType.CONTROL,
                sender_id="api_server",
                command="get_risk_status",
                parameters={}
            )
            
            response = await self.runtime.send_message(request)
            
            if response and "risk_status" in response.parameters:
                risk_data = response.parameters["risk_status"]
                
                return RiskStatusResponse(
                    portfolio_var=risk_data.get("portfolio_var", 0.0),
                    max_drawdown=risk_data.get("max_drawdown", 0.0),
                    leverage=risk_data.get("leverage", 0.0),
                    exposure=risk_data.get("exposure", {}),
                    risk_limits=risk_data.get("risk_limits", {}),
                    alerts_count=risk_data.get("alerts_count", 0),
                    last_updated=datetime.now()
                )
            
            return RiskStatusResponse(
                portfolio_var=0.0,
                max_drawdown=0.0,
                leverage=0.0,
                exposure={},
                risk_limits={},
                alerts_count=0,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Get risk status error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get risk status: {e}")
    
    async def _emergency_stop(self, reason: str) -> Dict[str, Any]:
        """Activate emergency stop."""
        
        try:
            request = ControlMessage(
                message_type=MessageType.KILL_SWITCH,
                sender_id="api_server",
                command="emergency_stop",
                parameters={"reason": reason}
            )
            
            response = await self.runtime.send_message(request)
            
            return {
                "action": "emergency_stop",
                "reason": reason,
                "status": "activated" if response else "failed",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Emergency stop error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to activate emergency stop: {e}")
    
    async def _handle_websocket(self, websocket):
        """Handle WebSocket connections for real-time updates."""
        
        await websocket.accept()
        
        try:
            while True:
                # Send periodic updates
                update_data = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "system_update",
                    "data": await self._get_dashboard_data()
                }
                
                await websocket.send_text(json.dumps(update_data))
                await asyncio.sleep(5)  # Update every 5 seconds
                
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            await websocket.close()
    
    async def start_server(self):
        """Start the API server."""
        
        logger.info(f"Starting API server on {self.host}:{self.port}")
        
        self._start_time = datetime.now()
        self._server_running = True
        
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=True,
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    async def stop_server(self):
        """Stop the API server."""
        
        logger.info("Stopping API server")
        self._server_running = False
