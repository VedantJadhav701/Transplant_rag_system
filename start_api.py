#!/usr/bin/env python3
"""
Start RAG API Server
====================

Quick start script for the FastAPI server.
"""

import subprocess
import sys
import time
from pathlib import Path

def main():
    print("\n" + "="*80)
    print("🚀 Starting Medical RAG API Server")
    print("="*80)
    
    # Check if virtual environment is activated
    venv_path = Path(".venv/Scripts/python.exe")
    if not venv_path.exists():
        print("\n❌ Error: Virtual environment not found")
        print("Run: python -m venv .venv")
        return
    
    print("\n📋 Server Information:")
    print("  - URL: http://localhost:8000")
    print("  - Docs: http://localhost:8000/docs")
    print("  - Health: http://localhost:8000/api/v1/health")
    print("  - Query: POST http://localhost:8000/api/v1/query")
    
    print("\n⚡ Starting uvicorn server...")
    print("  - Press CTRL+C to stop")
    print("  - Auto-reload enabled for development")
    print("="*80 + "\n")
    
    try:
        # Run uvicorn
        subprocess.run([
            ".venv/Scripts/uvicorn.exe",
            "app.main:app",
            "--reload",
            "--host", "0.0.0.0",
            "--port", "8000"
        ])
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("🛑 Server stopped")
        print("="*80)
    except FileNotFoundError:
        print("\n❌ Error: uvicorn not found")
        print("Install with: uv pip install uvicorn[standard]")

if __name__ == "__main__":
    main()
