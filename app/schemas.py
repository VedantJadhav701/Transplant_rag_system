#!/usr/bin/env python3
"""
Pydantic Schemas for API Request/Response
==========================================
"""

import os
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class QueryRequest(BaseModel):
    """Request schema for RAG query"""
    query: str = Field(..., min_length=5, max_length=500, description="Medical question")
    top_k: int = Field(default=int(os.getenv("DEFAULT_TOP_K", "4")), ge=1, le=10, description="Number of context chunks")
    max_tokens: int = Field(default=int(os.getenv("DEFAULT_MAX_TOKENS", "512")), ge=128, le=2048, description="Max response tokens")
    model: str = Field(default=os.getenv("DEFAULT_MODEL", "gemma3:1b"), description="Ollama model name")
    temperature: float = Field(default=float(os.getenv("DEFAULT_TEMPERATURE", "0.1")), ge=0.0, le=1.0, description="Sampling temperature")
    answer_mode: str = Field(default="clinical", description="Answer style: brief, clinical, or detailed")
    confidence_threshold: float = Field(default=0.50, ge=0.0, le=1.0, description="Minimum confidence to return answer")
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query doesn't contain prohibited content"""
        v = v.strip()
        if len(v) < 5:
            raise ValueError("Query must be at least 5 characters")
        
        # Medical safety: flag potentially dangerous queries
        prohibited_terms = [
            "diagnose me", "prescribe", "what dose", "how much drug",
            "should i take", "medical advice", "replace my doctor"
        ]
        
        query_lower = v.lower()
        for term in prohibited_terms:
            if term in query_lower:
                raise ValueError(
                    f"This system provides information only, not medical advice. "
                    f"Please consult a healthcare professional."
                )
        
        return v


class SourceInfo(BaseModel):
    """Source citation information"""
    document: str
    section: str
    organ_type: str
    similarity_score: float
    token_count: int
    text_preview: Optional[str] = None


class QueryResponse(BaseModel):
    """Response schema for RAG query"""
    query: str
    answer: str
    confidence: str  # "High", "Medium", "Low"
    confidence_score: float
    sources: List[SourceInfo]
    retrieval_time: float
    generation_time: float
    total_time: float
    chunks_used: int
    total_tokens: int
    model: str
    user: Optional[str] = None  # User who made the query


class HealthStatus(BaseModel):
    """Health check response"""
    status: str
    chroma_connected: bool
    model_available: bool
    chunks_indexed: int


class TokenRequest(BaseModel):
    """Login request"""
    username: str = Field(..., description="User email")
    password: str = Field(..., description="User password")


class TokenResponse(BaseModel):
    """Login response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600
