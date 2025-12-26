#!/usr/bin/env python3
"""
FastAPI Endpoints
=================
"""

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import StreamingResponse
from app.schemas import QueryRequest, QueryResponse, HealthStatus, TokenRequest, TokenResponse
from app.pipeline import HealthcareRAG
from app.deps import get_current_user
from app.security import authenticate_user, create_access_token
import logging
import json

router = APIRouter()

# Initialize RAG pipeline (singleton)
try:
    rag = HealthcareRAG("./data/chroma", "rag_config.toml")
    logging.info("RAG pipeline initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize RAG: {e}")
    rag = None


@router.post("/token", response_model=TokenResponse, status_code=status.HTTP_200_OK)
async def login(payload: TokenRequest):
    """
    Get JWT access token
    
    - **username**: User email
    - **password**: User password
    
    Demo credentials:
    - admin@transplant.ai / admin123
    - researcher@transplant.ai / research123
    """
    user = authenticate_user(payload.username, payload.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]}
    )
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=3600
    )


@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_rag(
    payload: QueryRequest,
    user: dict = Depends(get_current_user)
):
    """
    Query the medical RAG system (Protected endpoint)
    
    - **query**: Medical question (5-500 characters)
    - **top_k**: Number of context chunks to retrieve (1-10)
    - **max_tokens**: Maximum tokens in response (128-2048)
    - **model**: Ollama model name (default: phi3:mini)
    
    Requires: Bearer token in Authorization header
    """
    if rag is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized"
        )
    
    try:
        result = rag.answer(
            query=payload.query,
            top_k=payload.top_k,
            max_tokens=payload.max_tokens,
            model=payload.model,
            temperature=payload.temperature,
            answer_mode=payload.answer_mode,
            confidence_threshold=payload.confidence_threshold
        )
        
        # Add user info to response
        result["user"] = user.get("sub")
        
        return QueryResponse(**result)
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logging.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}"
        )


@router.post("/query/stream", status_code=status.HTTP_200_OK)
async def query_rag_stream(
    payload: QueryRequest,
    user: dict = Depends(get_current_user)
):
    """
    Query the medical RAG system with streaming response (Protected endpoint)
    
    Returns Server-Sent Events (SSE) stream with real-time token generation
    """
    if rag is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized"
        )
    
    async def generate():
        try:
            # Stream tokens from RAG pipeline
            async for chunk in rag.answer_stream(
                query=payload.query,
                top_k=payload.top_k,
                max_tokens=payload.max_tokens,
                model=payload.model,
                temperature=payload.temperature
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
        
        except Exception as e:
            logging.error(f"Streaming query failed: {e}")
            error_data = {"error": str(e), "done": True}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/health", response_model=HealthStatus, status_code=status.HTTP_200_OK)
async def health_check():
    """
    Check system health
    
    Returns status of ChromaDB connection and Ollama availability
    """
    if rag is None:
        return HealthStatus(
            status="unhealthy",
            chroma_connected=False,
            model_available=False,
            chunks_indexed=0
        )
    
    health_data = rag.health_check()
    return HealthStatus(**health_data)
