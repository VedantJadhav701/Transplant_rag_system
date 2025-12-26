#!/usr/bin/env python3
"""
RAG Retrieval Module - Medical Knowledge Base Query System
==========================================================

Optimized for RTX 3050 4GB with healthcare-grade quality.

Features:
- Semantic search with ChromaDB
- Duplicate removal (token overlap-based)
- Context budget enforcement (2500 tokens max)
- Metadata-rich results with citations

Usage:
    from retrieval import MedicalRetriever
    
    retriever = MedicalRetriever("./data/chroma", "rag_config.toml")
    results = retriever.retrieve("What causes kidney rejection?", top_k=8)

Author: Healthcare RAG System
Version: 1.0
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

# Disable telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["POSTHOG_DISABLED"] = "1"
logging.getLogger("chromadb").setLevel(logging.ERROR)

import numpy as np
import toml
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RetrievedChunk:
    """Retrieved chunk with metadata and relevance score"""
    chunk_id: str
    text: str
    doc_id: str
    doc_title: str
    section_title: str
    organ_type: str
    tier: str
    token_count: int
    similarity_score: float
    rank: int
    
    def format_citation(self) -> str:
        """Format as citation string"""
        return f"{self.doc_title} - {self.section_title}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "doc_title": self.doc_title,
            "section_title": self.section_title,
            "organ_type": self.organ_type,
            "tier": self.tier,
            "token_count": self.token_count,
            "similarity_score": self.similarity_score,
            "rank": self.rank,
            "citation": self.format_citation()
        }


@dataclass
class RetrievalResult:
    """Complete retrieval result with metadata"""
    query: str
    chunks: List[RetrievedChunk]
    total_tokens: int
    retrieval_time: float
    
    def format_context(self) -> str:
        """Format chunks as context for LLM"""
        context_parts = []
        
        for chunk in self.chunks:
            context_parts.append(
                f"[Source {chunk.rank}: {chunk.format_citation()}]\n"
                f"{chunk.text}\n"
            )
        
        return "\n".join(context_parts)
    
    def format_citations(self) -> List[str]:
        """Get list of unique citations"""
        citations = []
        seen = set()
        
        for chunk in self.chunks:
            citation = chunk.format_citation()
            if citation not in seen:
                citations.append(citation)
                seen.add(citation)
        
        return citations


# ============================================================================
# MEDICAL RETRIEVER
# ============================================================================

class MedicalRetriever:
    """
    Production-grade retriever for medical RAG system.
    
    Features:
    - Semantic search with all-mpnet-base-v2
    - Token overlap-based deduplication
    - Context budget enforcement
    - Metadata filtering (organ, tier)
    """
    
    def __init__(self, chroma_path: str, config_path: str):
        """
        Initialize retriever.
        
        Args:
            chroma_path: Path to ChromaDB directory
            config_path: Path to rag_config.toml
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = toml.load(f)
        
        # Load embedding model (for query encoding)
        embedding_config = self.config["embeddings"]
        self.model = SentenceTransformer(
            embedding_config["model_name"],
            device="cpu"  # Use CPU for retrieval to save VRAM for LLM
        )
        
        # Load ChromaDB WITHOUT embedding function (we handle embeddings ourselves)
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection(
            name=self.config["chroma"]["collection_name"]
        )
        
        # Retrieval parameters
        self.default_top_k = 8
        self.context_budget = 2500  # tokens
        self.dedup_threshold = 0.85  # 85% token overlap
        self.hybrid_mode = False  # Disable hybrid search (RRF scoring bug)
        self.bm25_weight = 0.3  # BM25 contribution (0.3 = 30% keyword, 70% semantic)
        
        print(f"✓ Retriever initialized")
        print(f"  Model: {embedding_config['model_name']}")
        print(f"  Collection: {self.collection.count()} chunks")
        print(f"  Hybrid search: {'Enabled' if self.hybrid_mode else 'Disabled'}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        organ_filter: Optional[str] = None,
        tier_filter: Optional[str] = None,
        use_hybrid: bool = None
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve (default: 8)
            organ_filter: Filter by organ (e.g., "kidney", "liver")
            tier_filter: Filter by tier (e.g., "Tier 2: Kidney")
            use_hybrid: Enable BM25 + vector hybrid search (default: self.hybrid_mode)
        
        Returns:
            RetrievalResult with chunks and metadata
        """
        import time
        start_time = time.time()
        
        if top_k is None:
            top_k = self.default_top_k
        
        if use_hybrid is None:
            use_hybrid = self.hybrid_mode
        
        # Use hybrid search if enabled
        if use_hybrid:
            chunks = self._hybrid_retrieve(query, top_k, organ_filter, tier_filter)
        else:
            chunks = self._vector_only_retrieve(query, top_k, organ_filter, tier_filter)
        
        # Deduplicate
        chunks = self._deduplicate(chunks)
        
        # Enforce context budget
        chunks = self._enforce_budget(chunks)
        
        # Assign ranks
        for i, chunk in enumerate(chunks, 1):
            chunk.rank = i
        
        elapsed = time.time() - start_time
        
        return RetrievalResult(
            query=query,
            chunks=chunks,
            total_tokens=sum(c.token_count for c in chunks),
            retrieval_time=elapsed
        )
    
    def _vector_only_retrieve(
        self,
        query: str,
        top_k: int,
        organ_filter: Optional[str],
        tier_filter: Optional[str]
    ) -> List[RetrievedChunk]:
        """Original vector-only retrieval"""
        # Embed query manually (don't use ChromaDB's embedding function)
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Build filter
        where_clause = self._build_filter(organ_filter, tier_filter)
        
        # Query ChromaDB with our embeddings
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        
        # Parse results
        return self._parse_results(results)
    
    def _hybrid_retrieve(
        self,
        query: str,
        top_k: int,
        organ_filter: Optional[str],
        tier_filter: Optional[str]
    ) -> List[RetrievedChunk]:
        """
        Hybrid BM25 + vector search with Reciprocal Rank Fusion (RRF)
        
        Algorithm:
        1. Get top-k results from vector search
        2. Get top-k results from BM25
        3. Combine using RRF: score = sum(1 / (rank + 60))
        4. Return top-k by combined score
        """
        # Get all documents from ChromaDB for BM25
        all_results = self.collection.get(
            include=["documents", "metadatas"],
            where=self._build_filter(organ_filter, tier_filter)
        )
        
        if not all_results["documents"]:
            return []
        
        # Step 1: Vector search with manual embeddings
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        vector_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k * 2, len(all_results["documents"])),  # Get 2x for fusion
            where=self._build_filter(organ_filter, tier_filter),
            include=["documents", "metadatas", "distances"]
        )
        
        vector_chunks = self._parse_results(vector_results)
        
        # Step 2: BM25 search
        tokenized_corpus = [doc.lower().split() for doc in all_results["documents"]]
        bm25 = BM25Okapi(tokenized_corpus)
        
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Get top-k BM25 results
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        
        # Step 3: Reciprocal Rank Fusion (RRF)
        rrf_k = 60  # RRF constant
        chunk_scores = {}  # chunk_id -> RRF score
        
        # Add vector scores
        for rank, chunk in enumerate(vector_chunks, 1):
            chunk_id = f"{chunk.doc_id}:{chunk.text[:50]}"  # Unique ID
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + (1 / (rank + rrf_k))
            
            if chunk_id not in chunk_scores or "chunk_obj" not in str(type(chunk_scores.get(chunk_id))):
                chunk_scores[f"{chunk_id}_obj"] = chunk
        
        # Add BM25 scores
        for rank, idx in enumerate(bm25_top_indices, 1):
            doc = all_results["documents"][idx]
            metadata = all_results["metadatas"][idx]
            
            chunk_id = f"{metadata.get('doc_id', 'unknown')}:{doc[:50]}"
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + (self.bm25_weight / (rank + rrf_k))
            
            # Store chunk object if not already stored
            if f"{chunk_id}_obj" not in chunk_scores:
                chunk = RetrievedChunk(
                    chunk_id=metadata.get("doc_id", "unknown") + f":chunk_{metadata.get('chunk_index', 0)}",
                    text=doc,
                    doc_id=metadata.get("doc_id", "unknown"),
                    doc_title=metadata.get("doc_title", "Unknown Document"),
                    section_title=metadata.get("section_title", "Unknown Section"),
                    organ_type=metadata.get("organ_type", "unknown"),
                    tier=metadata.get("tier", "unknown"),
                    token_count=metadata.get("token_count", len(doc.split())),
                    similarity_score=0.0,  # Will be updated
                    rank=0
                )
                chunk_scores[f"{chunk_id}_obj"] = chunk
        
        # Step 4: Sort by RRF score and return top-k
        sorted_chunks = []
        for chunk_id, score in sorted(chunk_scores.items(), key=lambda x: x[1] if not x[0].endswith("_obj") else 0, reverse=True):
            if not chunk_id.endswith("_obj"):
                chunk_obj = chunk_scores.get(f"{chunk_id}_obj")
                if chunk_obj:
                    chunk_obj.similarity_score = score  # Use RRF score
                    sorted_chunks.append(chunk_obj)
        
        return sorted_chunks[:top_k]
    
    def _build_filter(
        self,
        organ_filter: Optional[str],
        tier_filter: Optional[str]
    ) -> Optional[Dict]:
        """Build ChromaDB where clause for filtering"""
        if not organ_filter and not tier_filter:
            return None
        
        where_clause = {}
        
        if organ_filter:
            where_clause["organ_type"] = organ_filter
        
        if tier_filter:
            where_clause["tier"] = tier_filter
        
        return where_clause
    
    def _parse_results(self, results: Dict) -> List[RetrievedChunk]:
        """Parse ChromaDB results into RetrievedChunk objects"""
        chunks = []
        
        for doc, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            similarity = 1 - distance  # Convert distance to similarity
            
            chunk = RetrievedChunk(
                chunk_id=metadata.get("doc_id", "unknown") + f":chunk_{metadata.get('chunk_index', 0)}",
                text=doc,
                doc_id=metadata.get("doc_id", "unknown"),
                doc_title=metadata.get("doc_title", "Unknown Document"),
                section_title=metadata.get("section_title", "Unknown Section"),
                organ_type=metadata.get("organ_type", "unknown"),
                tier=metadata.get("tier", "unknown"),
                token_count=metadata.get("token_count", len(doc.split())),
                similarity_score=float(similarity),
                rank=0  # Will be assigned later
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _deduplicate(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Remove highly similar chunks based on token overlap.
        
        Algorithm:
        - Keep first chunk always
        - For each subsequent chunk, check token overlap with kept chunks
        - If overlap > threshold, skip (it's a duplicate)
        """
        if not chunks:
            return chunks
        
        kept_chunks = [chunks[0]]
        
        for chunk in chunks[1:]:
            is_duplicate = False
            chunk_tokens = set(chunk.text.lower().split())
            
            for kept_chunk in kept_chunks:
                kept_tokens = set(kept_chunk.text.lower().split())
                
                # Compute Jaccard similarity (intersection / union)
                intersection = len(chunk_tokens & kept_tokens)
                union = len(chunk_tokens | kept_tokens)
                
                if union > 0:
                    overlap = intersection / union
                    
                    if overlap > self.dedup_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                kept_chunks.append(chunk)
        
        return kept_chunks
    
    def _enforce_budget(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Enforce context budget by truncating chunks.
        
        Strategy:
        - Keep highest-scoring chunks first
        - Stop when token budget is exceeded
        """
        kept_chunks = []
        total_tokens = 0
        
        for chunk in chunks:
            if total_tokens + chunk.token_count <= self.context_budget:
                kept_chunks.append(chunk)
                total_tokens += chunk.token_count
            else:
                # Budget exceeded, stop
                break
        
        return kept_chunks
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = None
    ) -> List[RetrievalResult]:
        """Retrieve for multiple queries (useful for evaluation)"""
        return [self.retrieve(q, top_k) for q in queries]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_results_for_display(result: RetrievalResult) -> str:
    """Format retrieval results for console display"""
    output = []
    
    output.append("=" * 80)
    output.append(f"Query: {result.query}")
    output.append(f"Retrieved: {len(result.chunks)} chunks ({result.total_tokens} tokens)")
    output.append(f"Time: {result.retrieval_time:.3f}s")
    output.append("=" * 80)
    output.append("")
    
    for chunk in result.chunks:
        output.append(f"[Rank {chunk.rank}] Similarity: {chunk.similarity_score:.3f}")
        output.append(f"Source: {chunk.format_citation()}")
        output.append(f"Organ: {chunk.organ_type} | Tier: {chunk.tier} | Tokens: {chunk.token_count}")
        output.append("-" * 80)
        
        # Truncate text for display
        text_preview = chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text
        output.append(text_preview)
        output.append("")
    
    return "\n".join(output)


def format_results_for_llm(result: RetrievalResult) -> str:
    """Format retrieval results as context for LLM"""
    return result.format_context()


# ============================================================================
# MAIN (for testing)
# ============================================================================

def main():
    """Test retrieval module"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test medical retrieval system")
    parser.add_argument("query", type=str, help="Query string")
    parser.add_argument("--top_k", type=int, default=8, help="Number of results")
    parser.add_argument("--organ", type=str, help="Filter by organ")
    parser.add_argument("--tier", type=str, help="Filter by tier")
    parser.add_argument("--chroma", type=str, default="./data/chroma", help="ChromaDB path")
    parser.add_argument("--config", type=str, default="rag_config.toml", help="Config path")
    
    args = parser.parse_args()
    
    # Initialize retriever
    print("Initializing retriever...")
    retriever = MedicalRetriever(args.chroma, args.config)
    print()
    
    # Retrieve
    print(f"Searching for: {args.query}")
    if args.organ:
        print(f"  Organ filter: {args.organ}")
    if args.tier:
        print(f"  Tier filter: {args.tier}")
    print()
    
    result = retriever.retrieve(
        args.query,
        top_k=args.top_k,
        organ_filter=args.organ,
        tier_filter=args.tier
    )
    
    # Display
    print(format_results_for_display(result))
    
    # Also show LLM-formatted context
    print("=" * 80)
    print("CONTEXT FOR LLM:")
    print("=" * 80)
    print(format_results_for_llm(result))


if __name__ == "__main__":
    main()