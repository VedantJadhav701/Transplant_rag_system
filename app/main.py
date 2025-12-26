#!/usr/bin/env python3
"""
FastAPI Main Application
========================

Production-ready Medical RAG API

Usage:
    uvicorn app.main:app --reload
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import router
from app.middleware import log_requests
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/api.log', mode='a')
    ]
)

# Suppress ChromaDB telemetry warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["POSTHOG_DISABLED"] = "1"

# Create FastAPI app
app = FastAPI(
    title="Medical Transplant RAG API",
    version="1.0.0",
    description="Clinical-grade Retrieval-Augmented Generation for Transplant Medicine",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Request logging middleware
app.middleware("http")(log_requests)

# CORS middleware (configure for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["RAG"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Medical Transplant RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.on_event("startup")
async def startup_event():
    """Startup tasks"""
    logging.info("Starting Medical RAG API...")
    logging.info("Documentation available at /docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logging.info("Shutting down Medical RAG API...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
