"""
Authentication and authorization manager for the QuantAI REST API.

Handles JWT token generation, validation, and role-based access control.
"""

import asyncio
import json
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import jwt
from passlib.context import CryptContext
from loguru import logger

from .models import UserInfo


class AuthManager:
    """
    Authentication and authorization manager.
    
    Provides:
    - JWT token generation and validation
    - Password hashing and verification
    - Role-based access control
    - User session management
    - API key management
    """
    
    def __init__(
        self,
        secret_key: str = None,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 60,
        refresh_token_expire_days: int = 7,
    ):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        
        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # User database (in production, this would be a real database)
        self._users: Dict[str, Dict[str, Any]] = {}
        self._api_keys: Dict[str, Dict[str, Any]] = {}
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default admin user
        self._initialize_default_users()
        
        # Role permissions
        self._role_permissions = {
            "admin": [
                "read:all", "write:all", "delete:all",
                "emergency:stop", "emergency:liquidate", "emergency:reset",
                "config:read", "config:write",
                "users:manage", "api_keys:manage"
            ],
            "trader": [
                "read:strategies", "write:strategies",
                "read:performance", "read:risk",
                "strategies:start", "strategies:stop", "strategies:pause"
            ],
            "viewer": [
                "read:strategies", "read:performance", "read:risk",
                "read:dashboard"
            ],
            "api": [
                "read:strategies", "read:performance", "read:risk",
                "strategies:start", "strategies:stop"
            ]
        }
    
    def _initialize_default_users(self):
        """Initialize default users for the system."""
        
        # Default admin user
        admin_password_hash = self.get_password_hash("admin123")
        self._users["admin"] = {
            "user_id": "admin",
            "username": "admin",
            "password_hash": admin_password_hash,
            "role": "admin",
            "email": "admin@quantai.com",
            "created_at": datetime.now(),
            "last_login": None,
            "active": True,
        }
        
        # Default trader user
        trader_password_hash = self.get_password_hash("trader123")
        self._users["trader"] = {
            "user_id": "trader",
            "username": "trader",
            "password_hash": trader_password_hash,
            "role": "trader",
            "email": "trader@quantai.com",
            "created_at": datetime.now(),
            "last_login": None,
            "active": True,
        }
        
        # Default viewer user
        viewer_password_hash = self.get_password_hash("viewer123")
        self._users["viewer"] = {
            "user_id": "viewer",
            "username": "viewer",
            "password_hash": viewer_password_hash,
            "role": "viewer",
            "email": "viewer@quantai.com",
            "created_at": datetime.now(),
            "last_login": None,
            "active": True,
        }
        
        logger.info("Default users initialized")
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password."""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    async def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate a user with username and password."""
        
        user = self._users.get(username)
        if not user:
            logger.warning(f"Authentication failed: user {username} not found")
            return None
        
        if not user.get("active", True):
            logger.warning(f"Authentication failed: user {username} is inactive")
            return None
        
        if not self.verify_password(password, user["password_hash"]):
            logger.warning(f"Authentication failed: invalid password for user {username}")
            return None
        
        # Update last login
        user["last_login"] = datetime.now()
        
        logger.info(f"User {username} authenticated successfully")
        return user
    
    def create_access_token(self, user_data: Dict[str, Any]) -> str:
        """Create a JWT access token."""
        
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        payload = {
            "sub": user_data["username"],
            "user_id": user_data["user_id"],
            "role": user_data["role"],
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        # Store active session
        session_id = secrets.token_urlsafe(16)
        self._active_sessions[session_id] = {
            "user_id": user_data["user_id"],
            "username": user_data["username"],
            "role": user_data["role"],
            "token": token,
            "created_at": datetime.utcnow(),
            "expires_at": expire,
        }
        
        return token
    
    def create_refresh_token(self, user_data: Dict[str, Any]) -> str:
        """Create a JWT refresh token."""
        
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        payload = {
            "sub": user_data["username"],
            "user_id": user_data["user_id"],
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token."""
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            username = payload.get("sub")
            if not username:
                return None
            
            # Check if user still exists and is active
            user = self._users.get(username)
            if not user or not user.get("active", True):
                return None
            
            # Return user info from token
            return {
                "user_id": payload.get("user_id"),
                "username": username,
                "role": payload.get("role"),
                "token_type": payload.get("type", "access"),
                "exp": payload.get("exp"),
            }
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.JWTError as e:
            logger.warning(f"Token validation error: {e}")
            return None
    
    async def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Refresh an access token using a refresh token."""
        
        token_data = await self.verify_token(refresh_token)
        if not token_data or token_data.get("token_type") != "refresh":
            return None
        
        user = self._users.get(token_data["username"])
        if not user:
            return None
        
        return self.create_access_token(user)
    
    def check_permission(self, user_role: str, required_permission: str) -> bool:
        """Check if a user role has a specific permission."""
        
        role_permissions = self._role_permissions.get(user_role, [])
        
        # Check exact permission match
        if required_permission in role_permissions:
            return True
        
        # Check wildcard permissions
        permission_parts = required_permission.split(":")
        if len(permission_parts) == 2:
            action, resource = permission_parts
            
            # Check for action:all or read:all, write:all, etc.
            if f"{action}:all" in role_permissions:
                return True
            
            # Check for all:resource
            if f"all:{resource}" in role_permissions:
                return True
        
        return False
    
    async def create_api_key(
        self, 
        name: str, 
        role: str = "api",
        expires_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create an API key for programmatic access."""
        
        api_key = f"qai_{secrets.token_urlsafe(32)}"
        
        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)
        
        api_key_data = {
            "key": api_key,
            "name": name,
            "role": role,
            "created_at": datetime.utcnow(),
            "expires_at": expires_at,
            "last_used": None,
            "usage_count": 0,
            "active": True,
        }
        
        self._api_keys[api_key] = api_key_data
        
        logger.info(f"API key created: {name} with role {role}")
        
        return {
            "api_key": api_key,
            "name": name,
            "role": role,
            "expires_at": expires_at.isoformat() if expires_at else None,
        }
    
    async def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify an API key."""
        
        key_data = self._api_keys.get(api_key)
        if not key_data:
            return None
        
        if not key_data.get("active", True):
            return None
        
        # Check expiration
        if key_data.get("expires_at"):
            if datetime.utcnow() > key_data["expires_at"]:
                logger.warning(f"API key {key_data['name']} has expired")
                return None
        
        # Update usage
        key_data["last_used"] = datetime.utcnow()
        key_data["usage_count"] += 1
        
        return {
            "name": key_data["name"],
            "role": key_data["role"],
            "api_key": True,
        }
    
    async def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        
        if api_key in self._api_keys:
            self._api_keys[api_key]["active"] = False
            self._api_keys[api_key]["revoked_at"] = datetime.utcnow()
            
            logger.info(f"API key revoked: {self._api_keys[api_key]['name']}")
            return True
        
        return False
    
    async def list_api_keys(self) -> List[Dict[str, Any]]:
        """List all API keys (without the actual key values)."""
        
        return [
            {
                "name": data["name"],
                "role": data["role"],
                "created_at": data["created_at"].isoformat(),
                "expires_at": data["expires_at"].isoformat() if data.get("expires_at") else None,
                "last_used": data["last_used"].isoformat() if data.get("last_used") else None,
                "usage_count": data["usage_count"],
                "active": data["active"],
            }
            for data in self._api_keys.values()
        ]
    
    async def create_user(
        self,
        username: str,
        password: str,
        role: str,
        email: str = None
    ) -> Dict[str, Any]:
        """Create a new user."""
        
        if username in self._users:
            raise ValueError(f"User {username} already exists")
        
        if role not in self._role_permissions:
            raise ValueError(f"Invalid role: {role}")
        
        user_data = {
            "user_id": username,
            "username": username,
            "password_hash": self.get_password_hash(password),
            "role": role,
            "email": email,
            "created_at": datetime.now(),
            "last_login": None,
            "active": True,
        }
        
        self._users[username] = user_data
        
        logger.info(f"User created: {username} with role {role}")
        
        return {
            "user_id": username,
            "username": username,
            "role": role,
            "email": email,
            "created_at": user_data["created_at"].isoformat(),
        }
    
    async def update_user(
        self,
        username: str,
        updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update user information."""
        
        user = self._users.get(username)
        if not user:
            return None
        
        # Update allowed fields
        allowed_fields = ["email", "role", "active"]
        for field in allowed_fields:
            if field in updates:
                user[field] = updates[field]
        
        # Handle password update
        if "password" in updates:
            user["password_hash"] = self.get_password_hash(updates["password"])
        
        logger.info(f"User updated: {username}")
        
        return {
            "user_id": user["user_id"],
            "username": user["username"],
            "role": user["role"],
            "email": user.get("email"),
            "active": user["active"],
        }
    
    async def delete_user(self, username: str) -> bool:
        """Delete a user."""
        
        if username in self._users:
            del self._users[username]
            logger.info(f"User deleted: {username}")
            return True
        
        return False
    
    async def list_users(self) -> List[Dict[str, Any]]:
        """List all users (without password hashes)."""
        
        return [
            {
                "user_id": user["user_id"],
                "username": user["username"],
                "role": user["role"],
                "email": user.get("email"),
                "created_at": user["created_at"].isoformat(),
                "last_login": user["last_login"].isoformat() if user.get("last_login") else None,
                "active": user["active"],
            }
            for user in self._users.values()
        ]
    
    async def logout_user(self, token: str) -> bool:
        """Logout a user by invalidating their session."""
        
        # Find and remove the session
        for session_id, session in list(self._active_sessions.items()):
            if session["token"] == token:
                del self._active_sessions[session_id]
                logger.info(f"User {session['username']} logged out")
                return True
        
        return False
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        
        current_time = datetime.utcnow()
        expired_sessions = []
        
        for session_id, session in self._active_sessions.items():
            if session["expires_at"] < current_time:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self._active_sessions[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    async def get_user_info(self, username: str) -> Optional[UserInfo]:
        """Get user information."""
        
        user = self._users.get(username)
        if not user:
            return None
        
        permissions = self._role_permissions.get(user["role"], [])
        
        return UserInfo(
            user_id=user["user_id"],
            username=user["username"],
            role=user["role"],
            permissions=permissions,
            last_login=user.get("last_login")
        )
