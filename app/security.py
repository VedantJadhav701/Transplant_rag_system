#!/usr/bin/env python3
"""
JWT Authentication & Security
==============================

Handles token creation, verification, and password hashing.
"""

import os
from datetime import datetime, timedelta
from typing import Optional

from jose import jwt, JWTError
from passlib.context import CryptContext

# Load from environment or use development default
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Password hashing with argon2 (more modern and secure than bcrypt)
pwd_context = CryptContext(
    schemes=["argon2"],
    deprecated="auto"
)


def hash_password_safe(password: str) -> str:
    """Hash password using argon2"""
    return pwd_context.hash(password)


def verify_password_safe(plain_password: str, hashed_password: str) -> bool:
    """Verify password using argon2"""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[int] = None) -> str:
    """
    Create JWT access token
    
    Args:
        data: Payload to encode (e.g., {"sub": "user@example.com"})
        expires_delta: Token lifetime in minutes (default: 60)
    
    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + timedelta(minutes=expires_delta)
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt


def verify_token(token: str) -> Optional[dict]:
    """
    Verify and decode JWT token
    
    Args:
        token: JWT token string
    
    Returns:
        Decoded payload if valid, None if invalid/expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return verify_password_safe(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash password"""
    return hash_password_safe(password)


# Pre-computed argon2 hashes for demo users
_ADMIN_HASH = "$argon2id$v=19$m=65536,t=3,p=4$2HuPUaoVIiRkLMXY+98bIw$Gx4HrEf/GZBqwKVTCkpN78/La/K4LsnKck4w4gZpbwA"  # admin123
_RESEARCHER_HASH = "$argon2id$v=19$m=65536,t=3,p=4$slaqde5dS0lJyVkLYUyJUQ$B6EBpsP6ArqO7ww27VmmIDbieExrB2ej+rAMpW03dV4"  # research123

# Simple in-memory user database (replace with real DB in production)
DEMO_USERS = {
    "admin@transplant.ai": {
        "username": "admin@transplant.ai",
        "hashed_password": _ADMIN_HASH,
        "role": "admin"
    },
    "researcher@transplant.ai": {
        "username": "researcher@transplant.ai",
        "hashed_password": _RESEARCHER_HASH,
        "role": "researcher"
    }
}


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """
    Authenticate user with username/password
    
    Returns:
        User dict if valid, None if invalid
    """
    user = DEMO_USERS.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user
