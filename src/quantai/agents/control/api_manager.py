"""
API Manager Agent (M1) for the QuantAI system.

This agent manages API keys, broker credentials, and external service
authentication with secure storage and automatic rotation capabilities.
"""

import asyncio
import json
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import aiofiles
from autogen_core import MessageContext
from autogen_core.models import ChatCompletionClient
from cryptography.fernet import Fernet
from loguru import logger

from ...core.base import AgentCapability, AgentRole, ModelCapableAgent
from ...core.messages import ControlMessage, MessageType, QuantMessage


class APIManagerAgent(ModelCapableAgent):
    """
    API Manager Agent (M1) - Manages credentials and external service access.
    
    Capabilities:
    - Secure API key storage and retrieval
    - Broker account credential management
    - Automatic credential rotation
    - Access control and rate limiting
    - Service health monitoring
    - Audit logging for security compliance
    """
    
    def __init__(
        self,
        model_client: ChatCompletionClient,
        encryption_key: Optional[str] = None,
        credential_rotation_days: int = 90,
        max_api_calls_per_minute: int = 100,
        **kwargs
    ):
        super().__init__(
            role=AgentRole.API_MANAGER,
            capabilities=[
                AgentCapability.SECURITY_MANAGEMENT,
                AgentCapability.EXTERNAL_INTEGRATION,
            ],
            model_client=model_client,
            system_message=self._get_system_message(),
            **kwargs
        )
        
        # Security configuration
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.credential_rotation_days = credential_rotation_days
        self.max_api_calls_per_minute = max_api_calls_per_minute
        
        # Credential storage
        self._credentials: Dict[str, Dict[str, Any]] = {}
        self._api_usage: Dict[str, List[datetime]] = {}
        self._service_health: Dict[str, Dict[str, Any]] = {}
        self._access_tokens: Dict[str, Dict[str, Any]] = {}
        
        # Rate limiting
        self._rate_limits: Dict[str, Dict[str, Any]] = {}
        self._last_rotation_check: Optional[datetime] = None
        
        # Supported services
        self._supported_services = {
            "trading": {
                "alpaca": {"type": "broker", "auth_type": "api_key"},
                "interactive_brokers": {"type": "broker", "auth_type": "credentials"},
                "td_ameritrade": {"type": "broker", "auth_type": "oauth"},
            },
            "data": {
                "alpha_vantage": {"type": "data_provider", "auth_type": "api_key"},
                "polygon": {"type": "data_provider", "auth_type": "api_key"},
                "newsapi": {"type": "data_provider", "auth_type": "api_key"},
                "finnhub": {"type": "data_provider", "auth_type": "api_key"},
            },
            "ai": {
                "openai": {"type": "ai_model", "auth_type": "api_key"},
                "anthropic": {"type": "ai_model", "auth_type": "api_key"},
                "google": {"type": "ai_model", "auth_type": "api_key"},
            }
        }
    
    def _get_system_message(self) -> str:
        return """You are an API Manager Agent responsible for secure credential management and external service integration.

Your responsibilities:
1. Securely store and manage API keys and credentials
2. Handle broker account authentication and access
3. Implement automatic credential rotation policies
4. Monitor service health and availability
5. Enforce rate limiting and usage quotas
6. Maintain security audit logs and compliance

Security Framework:

1. Credential Management
   - Encrypted storage of all sensitive data
   - Secure key derivation and rotation
   - Access control with role-based permissions
   - Audit logging for all credential operations

2. Service Integration
   - Multi-broker support (Alpaca, Interactive Brokers, TD Ameritrade)
   - Data provider integration (Alpha Vantage, Polygon, NewsAPI)
   - AI model service management (OpenAI, Anthropic, Google)
   - OAuth and API key authentication flows

3. Rate Limiting
   - Per-service rate limit enforcement
   - Usage monitoring and quota management
   - Automatic throttling and backoff
   - Fair usage distribution across agents

4. Health Monitoring
   - Service availability checking
   - Performance metric tracking
   - Error rate monitoring
   - Automatic failover capabilities

5. Compliance and Auditing
   - Comprehensive access logging
   - Credential usage tracking
   - Security event monitoring
   - Regulatory compliance reporting

Security Best Practices:
- Never log or expose raw credentials
- Use encryption for all stored sensitive data
- Implement proper access controls
- Regular credential rotation
- Monitor for suspicious activity
- Maintain detailed audit trails

Guidelines:
- Prioritize security over convenience
- Implement defense in depth
- Use principle of least privilege
- Regular security assessments
- Incident response procedures
- Compliance with financial regulations

Focus on maintaining the highest security standards while enabling seamless integration with external services."""
    
    async def process_message(
        self, 
        message: QuantMessage, 
        ctx: MessageContext
    ) -> Optional[QuantMessage]:
        """Process credential management and API access requests."""
        
        if isinstance(message, ControlMessage):
            if message.command == "get_credentials":
                # Retrieve credentials for a service
                credentials = await self._get_credentials(
                    message.parameters.get("service"),
                    message.parameters.get("requester_id")
                )
                
                response = ControlMessage(
                    message_type=MessageType.CONTROL,
                    sender_id=self.agent_id,
                    command="credentials_response",
                    parameters={
                        "service": message.parameters.get("service"),
                        "credentials": credentials,
                        "expires_at": self._get_credential_expiry(message.parameters.get("service")),
                    },
                    session_id=message.session_id,
                    correlation_id=message.correlation_id,
                )
                
                return response
                
            elif message.command == "store_credentials":
                # Store new credentials
                await self._store_credentials(
                    message.parameters.get("service"),
                    message.parameters.get("credentials"),
                    message.parameters.get("metadata", {})
                )
                
                response = ControlMessage(
                    message_type=MessageType.CONTROL,
                    sender_id=self.agent_id,
                    command="credentials_stored",
                    parameters={"service": message.parameters.get("service"), "status": "success"},
                    session_id=message.session_id,
                    correlation_id=message.correlation_id,
                )
                
                return response
                
            elif message.command == "rotate_credentials":
                # Rotate credentials for a service
                await self._rotate_credentials(message.parameters.get("service"))
                
                response = ControlMessage(
                    message_type=MessageType.CONTROL,
                    sender_id=self.agent_id,
                    command="credentials_rotated",
                    parameters={"service": message.parameters.get("service"), "status": "success"},
                    session_id=message.session_id,
                    correlation_id=message.correlation_id,
                )
                
                return response
                
            elif message.command == "check_service_health":
                # Check health of external services
                health_status = await self._check_service_health(
                    message.parameters.get("service")
                )
                
                response = ControlMessage(
                    message_type=MessageType.CONTROL,
                    sender_id=self.agent_id,
                    command="service_health_response",
                    parameters={"health_status": health_status},
                    session_id=message.session_id,
                    correlation_id=message.correlation_id,
                )
                
                return response
        
        return None
    
    async def _get_credentials(self, service: str, requester_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve credentials for a service with access control."""
        
        if not service or service not in self._credentials:
            logger.warning(f"Credentials not found for service: {service}")
            return None
        
        # Check access permissions
        if not await self._check_access_permission(service, requester_id):
            logger.warning(f"Access denied for {requester_id} to service {service}")
            return None
        
        # Check rate limits
        if not await self._check_rate_limit(service, requester_id):
            logger.warning(f"Rate limit exceeded for {requester_id} on service {service}")
            return None
        
        # Decrypt and return credentials
        encrypted_creds = self._credentials[service]
        decrypted_creds = self._decrypt_credentials(encrypted_creds)
        
        # Log access
        await self._log_credential_access(service, requester_id)
        
        # Update usage tracking
        await self._update_usage_tracking(service, requester_id)
        
        return decrypted_creds
    
    async def _store_credentials(
        self, 
        service: str, 
        credentials: Dict[str, Any], 
        metadata: Dict[str, Any]
    ):
        """Store credentials securely with encryption."""
        
        if not service or not credentials:
            raise ValueError("Service and credentials are required")
        
        # Encrypt credentials
        encrypted_creds = self._encrypt_credentials(credentials)
        
        # Store with metadata
        self._credentials[service] = {
            "encrypted_data": encrypted_creds,
            "created_at": datetime.now(),
            "last_rotated": datetime.now(),
            "metadata": metadata,
            "access_count": 0,
        }
        
        # Initialize rate limiting
        self._rate_limits[service] = {
            "calls_per_minute": metadata.get("rate_limit", self.max_api_calls_per_minute),
            "current_usage": [],
        }
        
        # Log storage
        await self._log_credential_storage(service)
        
        logger.info(f"Credentials stored for service: {service}")
    
    def _encrypt_credentials(self, credentials: Dict[str, Any]) -> str:
        """Encrypt credentials using Fernet encryption."""
        
        credential_json = json.dumps(credentials)
        encrypted_data = self.cipher_suite.encrypt(credential_json.encode())
        return encrypted_data.decode()
    
    def _decrypt_credentials(self, encrypted_creds: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt credentials."""
        
        encrypted_data = encrypted_creds["encrypted_data"].encode()
        decrypted_data = self.cipher_suite.decrypt(encrypted_data)
        return json.loads(decrypted_data.decode())
    
    async def _check_access_permission(self, service: str, requester_id: str) -> bool:
        """Check if requester has permission to access service credentials."""
        
        # Implement role-based access control
        # For now, allow all authenticated agents
        if not requester_id:
            return False
        
        # Check if requester is a valid agent
        valid_agent_prefixes = [
            "data_ingestion_", "execution_", "strategy_", 
            "risk_control_", "profitability_", "dashboard_"
        ]
        
        return any(requester_id.startswith(prefix) for prefix in valid_agent_prefixes)
    
    async def _check_rate_limit(self, service: str, requester_id: str) -> bool:
        """Check rate limits for service access."""
        
        if service not in self._rate_limits:
            return True
        
        rate_config = self._rate_limits[service]
        current_time = datetime.now()
        
        # Clean old usage records (older than 1 minute)
        cutoff_time = current_time - timedelta(minutes=1)
        rate_config["current_usage"] = [
            usage_time for usage_time in rate_config["current_usage"]
            if usage_time > cutoff_time
        ]
        
        # Check if under limit
        current_usage = len(rate_config["current_usage"])
        limit = rate_config["calls_per_minute"]
        
        return current_usage < limit
    
    async def _update_usage_tracking(self, service: str, requester_id: str):
        """Update usage tracking for rate limiting."""
        
        current_time = datetime.now()
        
        if service in self._rate_limits:
            self._rate_limits[service]["current_usage"].append(current_time)
        
        # Update credential access count
        if service in self._credentials:
            self._credentials[service]["access_count"] += 1
    
    async def _rotate_credentials(self, service: str):
        """Rotate credentials for a service."""
        
        if service not in self._credentials:
            logger.warning(f"Cannot rotate credentials for unknown service: {service}")
            return
        
        # This would implement actual credential rotation
        # For now, just update the rotation timestamp
        self._credentials[service]["last_rotated"] = datetime.now()
        
        logger.info(f"Credentials rotated for service: {service}")
        await self._log_credential_rotation(service)
    
    def _get_credential_expiry(self, service: str) -> Optional[datetime]:
        """Get credential expiry time."""
        
        if service not in self._credentials:
            return None
        
        last_rotated = self._credentials[service]["last_rotated"]
        expiry = last_rotated + timedelta(days=self.credential_rotation_days)
        
        return expiry
    
    async def _check_service_health(self, service: str = None) -> Dict[str, Any]:
        """Check health of external services."""
        
        health_status = {}
        
        services_to_check = [service] if service else list(self._credentials.keys())
        
        for svc in services_to_check:
            if svc in self._credentials:
                # Perform actual health check
                health = await self._perform_health_check(svc)
                health_status[svc] = health
        
        return health_status
    
    async def _perform_health_check(self, service: str) -> Dict[str, Any]:
        """Perform actual health check for a service."""
        
        # This would implement actual health checks
        # For now, return mock health status
        
        health = {
            "status": "healthy",
            "response_time_ms": 150,
            "last_check": datetime.now().isoformat(),
            "error_rate": 0.01,
            "availability": 0.999,
        }
        
        # Store health status
        self._service_health[service] = health
        
        return health
    
    async def _log_credential_access(self, service: str, requester_id: str):
        """Log credential access for audit purposes."""
        
        log_entry = {
            "event": "credential_access",
            "service": service,
            "requester_id": requester_id,
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id,
        }
        
        # In production, this would write to a secure audit log
        logger.info(f"Credential access: {log_entry}")
    
    async def _log_credential_storage(self, service: str):
        """Log credential storage for audit purposes."""
        
        log_entry = {
            "event": "credential_storage",
            "service": service,
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id,
        }
        
        logger.info(f"Credential storage: {log_entry}")
    
    async def _log_credential_rotation(self, service: str):
        """Log credential rotation for audit purposes."""
        
        log_entry = {
            "event": "credential_rotation",
            "service": service,
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id,
        }
        
        logger.info(f"Credential rotation: {log_entry}")
    
    async def start_periodic_tasks(self):
        """Start periodic maintenance tasks."""
        
        logger.info("Starting API Manager periodic tasks")
        
        # Start credential rotation check
        asyncio.create_task(self._periodic_rotation_check())
        
        # Start health monitoring
        asyncio.create_task(self._periodic_health_check())
        
        # Start usage cleanup
        asyncio.create_task(self._periodic_usage_cleanup())
    
    async def _periodic_rotation_check(self):
        """Periodically check for credentials that need rotation."""
        
        while True:
            try:
                current_time = datetime.now()
                
                for service, cred_data in self._credentials.items():
                    last_rotated = cred_data["last_rotated"]
                    days_since_rotation = (current_time - last_rotated).days
                    
                    if days_since_rotation >= self.credential_rotation_days:
                        logger.warning(f"Credentials for {service} need rotation")
                        # In production, this would trigger automatic rotation
                
                # Check every hour
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in periodic rotation check: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def _periodic_health_check(self):
        """Periodically check service health."""
        
        while True:
            try:
                await self._check_service_health()
                
                # Check every 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in periodic health check: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _periodic_usage_cleanup(self):
        """Periodically clean up old usage records."""
        
        while True:
            try:
                current_time = datetime.now()
                cutoff_time = current_time - timedelta(hours=1)
                
                for service in self._rate_limits:
                    self._rate_limits[service]["current_usage"] = [
                        usage_time for usage_time in self._rate_limits[service]["current_usage"]
                        if usage_time > cutoff_time
                    ]
                
                # Clean every 10 minutes
                await asyncio.sleep(600)
                
            except Exception as e:
                logger.error(f"Error in periodic usage cleanup: {e}")
                await asyncio.sleep(300)
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        
        return {
            "total_services": len(self._credentials),
            "service_health": self._service_health,
            "rate_limits": {
                service: {
                    "limit": config["calls_per_minute"],
                    "current_usage": len(config["current_usage"])
                }
                for service, config in self._rate_limits.items()
            },
            "credential_status": {
                service: {
                    "last_rotated": data["last_rotated"].isoformat(),
                    "access_count": data["access_count"],
                    "expires_at": self._get_credential_expiry(service).isoformat() if self._get_credential_expiry(service) else None
                }
                for service, data in self._credentials.items()
            },
            "supported_services": self._supported_services,
        }
    
    async def emergency_revoke_credentials(self, service: str):
        """Emergency revocation of credentials."""
        
        if service in self._credentials:
            # Mark credentials as revoked
            self._credentials[service]["revoked"] = True
            self._credentials[service]["revoked_at"] = datetime.now()
            
            logger.critical(f"Emergency revocation of credentials for service: {service}")
            
            # Log security event
            await self._log_security_event("credential_revocation", service)
    
    async def _log_security_event(self, event_type: str, service: str):
        """Log security events."""
        
        log_entry = {
            "event": "security_event",
            "event_type": event_type,
            "service": service,
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id,
            "severity": "critical",
        }
        
        logger.critical(f"Security event: {log_entry}")
