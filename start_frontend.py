#!/usr/bin/env python3
"""
Launch script for Medical RAG Streamlit frontend
"""

import subprocess
import sys
from pathlib import Path

def main():
    frontend_path = Path(__file__).parent / "frontend.py"
    
    print("🚀 Starting Medical RAG Frontend...")
    print("📍 URL: http://localhost:8501")
    print("Press Ctrl+C to stop\n")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(frontend_path),
            "--server.port=8501",
            "--server.headless=true"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Frontend stopped")

if __name__ == "__main__":
    main()
