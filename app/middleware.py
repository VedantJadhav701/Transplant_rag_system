#!/usr/bin/env python3
"""
FastAPI Middleware
==================

Request logging and monitoring.
"""

import time
import logging
from fastapi import Request

logger = logging.getLogger("api")


async def log_requests(request: Request, call_next):
    """
    Middleware to log all HTTP requests
    
    Logs:
        - Client IP
        - HTTP method
        - URL path
        - Status code
        - Response time
    """
    start_time = time.time()
    
    # Get client IP (handle proxy headers)
    client_ip = request.client.host if request.client else "unknown"
    if "x-forwarded-for" in request.headers:
        client_ip = request.headers["x-forwarded-for"].split(",")[0].strip()
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = round(time.time() - start_time, 3)
    
    # Log request
    logger.info(
        f"{client_ip} | {request.method} {request.url.path} | "
        f"Status {response.status_code} | {duration}s"
    )
    
    # Add custom header with response time
    response.headers["X-Process-Time"] = str(duration)
    
    return response
