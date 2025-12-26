#!/usr/bin/env python3
"""
FastAPI Dependencies
====================

Authentication dependencies for protected endpoints.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.security import verify_token

# HTTP Bearer token security scheme
security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    Dependency to extract and verify JWT token
    
    Usage:
        @router.get("/protected")
        def protected_route(user=Depends(get_current_user)):
            return {"user": user["sub"]}
    
    Raises:
        HTTPException: 401 if token is invalid or expired
    """
    token = credentials.credentials
    payload = verify_token(token)
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return payload


def get_admin_user(user: dict = Depends(get_current_user)) -> dict:
    """
    Dependency to verify admin role
    
    Raises:
        HTTPException: 403 if user is not admin
    """
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return user
