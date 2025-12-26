#!/usr/bin/env python3
"""
Production RAG Pipeline
=======================
Wraps the RAG logic with confidence scoring, logging, and validation.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Disable telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["POSTHOG_DISABLED"] = "1"

logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from retrieval import MedicalRetriever, RetrievedChunk

try:
    import ollama
except ImportError:
    raise ImportError("Ollama not installed. Run: uv pip install ollama")

# Configure Ollama client with environment variable support
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ollama_client = ollama.Client(host=OLLAMA_BASE_URL)


class HealthcareRAG:
    """Production-grade RAG pipeline with confidence scoring and logging"""
    
    def __init__(
        self, 
        chroma_path: str = "./data/chroma",
        config_path: str = "rag_config.toml",
        log_dir: str = "./logs"
    ):
        self.chroma_path = chroma_path
        self.config_path = config_path
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize retriever
        self.retriever = MedicalRetriever(
            chroma_path=chroma_path,
            config_path=config_path
        )
        
    def compute_confidence(self, chunks: List[RetrievedChunk]) -> tuple[str, float]:
        """
        Compute confidence score based on retrieval quality
        
        Returns:
            (confidence_label, confidence_score)
        """
        if not chunks:
            return "Low", 0.0
        
        # Use similarity scores
        scores = [c.similarity_score for c in chunks]
        avg_score = sum(scores) / len(scores)
        
        # Additional factors
        top_score = scores[0] if scores else 0.0
        score_variance = max(scores) - min(scores) if len(scores) > 1 else 0.0
        
        # Weighted confidence
        confidence_score = (
            0.6 * avg_score +           # Average similarity
            0.3 * top_score +            # Best match quality
            0.1 * (1 - score_variance)   # Consistency (low variance = better)
        )
        
        # Categorize confidence levels
        if confidence_score > 0.70 and top_score > 0.75:
            confidence_label = "High"
        elif confidence_score > 0.50:  # Lowered from 0.55
            confidence_label = "Medium"
        else:
            confidence_label = "Low"
        
        return confidence_label, round(confidence_score, 3)
    
    def build_prompt(self, query: str, chunks: List[RetrievedChunk], answer_mode: str = "clinical") -> str:
        """Build RAG prompt with context and answer mode"""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk.doc_title} - {chunk.section_title}]\n"
                f"{chunk.text}\n"
            )
        
        context = "\n---\n\n".join(context_parts)
        
        # Answer mode instructions
        mode_instructions = {
            "brief": "Provide a concise 2-3 sentence answer focusing on the most critical information.",
            "clinical": "Provide a clear, clinical answer with bullet points for clarity. Include key criteria, typical ranges, and acknowledge institutional variability when appropriate.",
            "detailed": "Provide a comprehensive answer covering mechanisms, clinical implications, variations, and relevant context. Use structured formatting."
        }
        
        instruction = mode_instructions.get(answer_mode, mode_instructions["clinical"])
        
        prompt = f"""You are a medical transplant expert assistant. Answer the question based ONLY on the provided medical context.

Context from Medical Transplant Knowledge Base:

{context}

---

Question: {query}

Instructions:
- {instruction}
- Include inline citations in format: (Source: [Document Name] - [Section])
- Be precise but recognize that practices may vary across institutions"""
        
        return prompt
    
    def log_query(self, data: Dict):
        """Log query to JSONL file"""
        log_file = self.log_dir / "queries.jsonl"
        
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    **data
                }
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logging.error(f"Failed to log query: {e}")
    
    def answer(
        self,
        query: str,
        top_k: int = 5,
        max_tokens: int = 512,
        model: str = "phi3:mini",
        temperature: float = 0.05,
        answer_mode: str = "clinical",
        confidence_threshold: float = 0.50  # Lowered from 0.55 to reduce false negatives
    ) -> Dict:
        """
        Complete RAG pipeline with confidence scoring
        
        Returns:
            Dictionary with answer, sources, confidence, and timing
        """
        start_time = time.time()
        
        # Step 1: Retrieve
        retrieval_start = time.time()
        result = self.retriever.retrieve(query, top_k=top_k)
        chunks = result.chunks
        retrieval_time = time.time() - retrieval_start
        
        if not chunks:
            return {
                "query": query,
                "answer": "No relevant information found in the knowledge base.",
                "confidence": "Low",
                "confidence_score": 0.0,
                "sources": [],
                "retrieval_time": retrieval_time,
                "generation_time": 0.0,
                "total_time": time.time() - start_time,
                "chunks_used": 0,
                "total_tokens": 0,
                "model": model
            }
        
        # Step 2: Compute confidence
        confidence_label, confidence_score = self.compute_confidence(chunks)
        
        # Step 2.5: Confidence gating (production safety)
        if confidence_score < confidence_threshold:
            return {
                "query": query,
                "answer": f"⚠️ Insufficient evidence in knowledge base (confidence: {confidence_score:.2f} < {confidence_threshold:.2f}). Please consult medical documentation or a specialist for this specific query.",
                "confidence": "Low",
                "confidence_score": confidence_score,
                "sources": [],
                "retrieval_time": retrieval_time,
                "generation_time": 0.0,
                "total_time": time.time() - start_time,
                "chunks_used": len(chunks),
                "total_tokens": result.total_tokens,
                "model": model,
                "gated": True
            }
        
        # Step 3: Build prompt
        prompt = self.build_prompt(query, chunks, answer_mode)
        
        # Step 4: Generate
        generation_start = time.time()
        try:
            response = ollama_client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            answer = response['message']['content'].strip()
            generation_time = time.time() - generation_start
            
        except Exception as e:
            logging.error(f"Generation error: {e}")
            return {
                "query": query,
                "answer": f"Error generating response: {str(e)}",
                "confidence": "Low",
                "confidence_score": 0.0,
                "sources": [],
                "retrieval_time": retrieval_time,
                "generation_time": 0.0,
                "total_time": time.time() - start_time,
                "chunks_used": len(chunks),
                "total_tokens": result.total_tokens,
                "model": model
            }
        
        # Step 5: Format sources
        sources = []
        for chunk in chunks[:3]:  # Top 3 sources
            sources.append({
                "document": chunk.doc_title,
                "section": chunk.section_title,
                "organ_type": chunk.organ_type,
                "similarity_score": round(chunk.similarity_score, 3),
                "token_count": chunk.token_count,
                "text_preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
            })
        
        total_time = time.time() - start_time
        
        # Step 6: Build response
        response_data = {
            "query": query,
            "answer": answer,
            "confidence": confidence_label,
            "confidence_score": confidence_score,
            "sources": sources,
            "retrieval_time": round(retrieval_time, 3),
            "generation_time": round(generation_time, 3),
            "total_time": round(total_time, 3),
            "chunks_used": len(chunks),
            "total_tokens": result.total_tokens,
            "model": model
        }
        
        # Step 7: Log query
        self.log_query({
            "query": query,
            "confidence": confidence_label,
            "confidence_score": confidence_score,
            "chunks_used": len(chunks),
            "total_time": total_time,
            "model": model
        })
        
        return response_data
    
    async def answer_stream(
        self,
        query: str,
        top_k: int = 4,
        max_tokens: int = 512,
        model: str = "gemma3:1b",
        temperature: float = 0.1
    ):
        """
        Streaming RAG pipeline - yields tokens as they're generated
        
        Yields:
            Dictionary chunks with streaming tokens and metadata
        """
        start_time = time.time()
        
        # Step 1: Retrieve (non-streaming)
        retrieval_start = time.time()
        result = self.retriever.retrieve(query, top_k=top_k)
        chunks = result.chunks
        retrieval_time = time.time() - retrieval_start
        
        if not chunks:
            yield {
                "type": "error",
                "message": "No relevant information found in the knowledge base.",
                "done": True
            }
            return
        
        # Step 2: Compute confidence
        confidence_label, confidence_score = self.compute_confidence(chunks)
        
        # Step 3: Send metadata first
        sources = []
        for chunk in chunks[:3]:
            sources.append({
                "document": chunk.doc_title,
                "section": chunk.section_title,
                "organ_type": chunk.organ_type,
                "similarity_score": round(chunk.similarity_score, 3)
            })
        
        yield {
            "type": "metadata",
            "query": query,
            "confidence": confidence_label,
            "confidence_score": confidence_score,
            "sources": sources,
            "retrieval_time": round(retrieval_time, 3),
            "chunks_used": len(chunks),
            "model": model
        }
        
        # Step 4: Build prompt
        prompt = self.build_prompt(query, chunks)
        
        # Step 5: Stream generation
        generation_start = time.time()
        full_answer = ""
        
        try:
            stream = ollama_client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                },
                stream=True
            )
            
            for chunk in stream:
                token = chunk['message']['content']
                full_answer += token
                
                yield {
                    "type": "token",
                    "content": token
                }
        
        except Exception as e:
            logging.error(f"Streaming generation error: {e}")
            yield {
                "type": "error",
                "message": str(e),
                "done": True
            }
            return
        
        generation_time = time.time() - generation_start
        total_time = time.time() - start_time
        
        # Step 6: Send completion
        yield {
            "type": "done",
            "generation_time": round(generation_time, 3),
            "total_time": round(total_time, 3),
            "total_tokens": result.total_tokens,
            "done": True
        }
        
        # Step 7: Log query
        self.log_query({
            "query": query,
            "confidence": confidence_label,
            "confidence_score": confidence_score,
            "chunks_used": len(chunks),
            "total_time": total_time,
            "model": model,
            "streamed": True
        })
    
    def health_check(self) -> Dict:
        """Check system health"""
        try:
            # Check ChromaDB
            collection_count = self.retriever.collection.count()
            chroma_ok = collection_count > 0
            
            # Check Ollama
            try:
                ollama_client.list()
                ollama_ok = True
            except:
                ollama_ok = False
            
            return {
                "status": "healthy" if (chroma_ok and ollama_ok) else "degraded",
                "chroma_connected": chroma_ok,
                "model_available": ollama_ok,
                "chunks_indexed": collection_count
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "chroma_connected": False,
                "model_available": False,
                "chunks_indexed": 0,
                "error": str(e)
            }
