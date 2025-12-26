#!/usr/bin/env python3
"""
Simple RAG Pipeline with Ollama
================================

Uses Ollama for local LLM inference (easiest Windows setup).

Usage:
    python simple_rag.py "What are immunosuppressive drugs?"
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Disable telemetry before any imports
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["POSTHOG_DISABLED"] = "1"

# Suppress ChromaDB warnings
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

# Import from retrieval.py
from retrieval import MedicalRetriever

try:
    import ollama
except ImportError:
    print("Error: Ollama not installed. Run: uv pip install ollama")
    sys.exit(1)


def build_prompt(query: str, chunks: list) -> str:
    """Build RAG prompt with retrieved context"""
    
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        # chunks are RetrievedChunk objects
        context_parts.append(
            f"[Source {i}: {chunk.doc_title} - {chunk.section_title}]\n"
            f"{chunk.text}\n"
        )
    
    context = "\n---\n\n".join(context_parts)
    
    prompt = f"""You are a medical transplant expert assistant. Answer the question based ONLY on the provided medical context.

Context from Medical Transplant Knowledge Base:

{context}

---

Question: {query}

Instructions:
- Provide a clear, clinical answer based on the context above
- Use bullet points for clarity
- When citing specific criteria (age limits, thresholds, protocols), acknowledge institutional variability when appropriate (e.g., "typically", "generally", "depending on center policy")
- Include inline citations in format: (Source: [Document Name] - [Section])
- Be precise but recognize that practices may vary across institutions"""
    
    return prompt


def answer_question(question: str, top_k: int = 5, model: str = "phi3:mini"):
    """Complete RAG pipeline"""
    
    print(f"\n{'='*80}")
    print(f"Question: {question}")
    print(f"{'='*80}\n")
    
    # Step 1: Retrieve
    print(f"[1/2] Retrieving context...")
    retriever = MedicalRetriever(
        chroma_path="./data/chroma",
        config_path="rag_config.toml"
    )
    
    result = retriever.retrieve(question, top_k=top_k)
    chunks = result.chunks
    
    if not chunks:
        print("No relevant information found.")
        return
    
    print(f"      Found {len(chunks)} relevant chunks ({result.total_tokens} tokens)\n")
    
    # Step 2: Build prompt
    prompt = build_prompt(question, chunks)
    
    # Step 3: Generate
    print(f"[2/2] Generating answer with {model}...")
    print(f"{'='*80}\n")
    
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.05}  # Very low for clinical accuracy
        )
        
        answer = response['message']['content']
        
        # Display answer with better formatting
        print("\n📌 ANSWER:")
        print(f"{'-'*80}")
        print(answer.strip())
        print(f"{'-'*80}\n")
        
        # Show sources with detailed attribution
        print("📚 SOURCES:")
        print(f"{'-'*80}")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\n[{i}] Document: {chunk.doc_title}")
            print(f"    Section: {chunk.section_title}")
            print(f"    Organ: {chunk.organ_type}")
            print(f"    Relevance: {chunk.similarity_score:.3f} ({chunk.token_count} tokens)")
        
        print(f"\n{'-'*80}")
        print(f"\nℹ️  Inline citations format: (Source: [Document] - [Section])")
        print()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure Ollama is running:")
        print("  1. Install Ollama: https://ollama.ai")
        print(f"  2. Pull model: ollama pull {model}")
        print("  3. Run this script again")


def main():
    parser = argparse.ArgumentParser(description="Medical RAG Q&A with Ollama")
    
    parser.add_argument(
        'question',
        type=str,
        help='Medical question to answer'
    )
    
    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help='Number of context chunks (default: 5)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='phi3:mini',
        help='Ollama model name (default: phi3:mini)'
    )
    
    args = parser.parse_args()
    
    answer_question(args.question, args.top_k, args.model)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\nMedical RAG Q&A System")
        print("="*80)
        print("\nUsage:")
        print("  python simple_rag.py \"What are immunosuppressive drugs?\"")
        print("  python simple_rag.py \"Kidney rejection signs\" --top_k 8")
        print("  python simple_rag.py \"Liver transplant\" --model llama3.2")
        print("\nFirst time setup:")
        print("  1. Install Ollama: https://ollama.ai (or winget install Ollama.Ollama)")
        print("  2. Pull model: ollama pull llama3.2")
        print("  3. Run this script")
        print()
        sys.exit(1)
    
    main()
