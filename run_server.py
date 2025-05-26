#!/usr/bin/env python3
"""
FastAPI Server Startup Script
============================

Simple script to start the Intent Detection API server.
Handles basic configuration and provides helpful startup information.

Usage:
    python run_server.py
    python run_server.py --port 8080
    python run_server.py --host 0.0.0.0 --port 9000

Author: Intent Detection System
Date: 2025
"""

import argparse
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def check_requirements():
    """Check if required files exist before starting server."""
    print("🔍 Checking requirements...")
    
    # Check if model files exist
    model_files = [
        "models/best_model.pkl",
        "models/best_pipeline.pkl"
    ]
    
    missing_files = []
    for file_path in model_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n💡 Please train a model first using the notebooks in nbs/")
        print("   Run: jupyter notebook nbs/experiment_template.ipynb")
        return False
    
    # Check if OpenAI API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-openai-api-key-here":
        print("⚠️  OpenAI API key not set!")
        print("   Set environment variable: export OPENAI_API_KEY='your-key'")
        print("   Or create .env file with: OPENAI_API_KEY=your-key")
        print("   API will work with ML-only fallback, but LLM features disabled.")
    
    print("✅ Requirements check completed!")
    return True

def print_startup_info(host: str, port: int):
    """Print helpful startup information."""
    print("\n" + "="*60)
    print("🚀 INTENT DETECTION API SERVER")
    print("="*60)
    print(f"🌐 Server URL: http://{host}:{port}")
    print(f"📚 API Docs: http://{host}:{port}/docs")
    print(f"🔍 Redoc: http://{host}:{port}/redoc")
    print(f"❤️  Health Check: http://{host}:{port}/health")
    print("\n📖 API Endpoints:")
    print(f"   POST /detect    - Main intent detection")
    print(f"   GET  /health    - System health status")
    print(f"   GET  /          - API information")
    print("\n📋 Example Usage:")
    print(f"   curl -X POST http://{host}:{port}/detect \\")
    print(f'     -H "Content-Type: application/json" \\')
    print(f'     -d \'{{"query": "I want to cancel my order"}}\'')
    print("\n🔧 Configuration:")
    print(f"   Model: SentenceBERT + Logistic Regression")
    print(f"   Confidence Threshold: 0.75")
    print(f"   LLM: OpenAI GPT-3.5-turbo")
    print("="*60)

def main():
    """Main function to start the FastAPI server."""
    parser = argparse.ArgumentParser(description="Start Intent Detection API server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to") 
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    
    args = parser.parse_args()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Server startup aborted due to missing requirements.")
        print("💡 Please fix the issues above and try again.")
        sys.exit(1)
    
    # Print startup information
    print_startup_info(args.host, args.port)
    
    try:
        import uvicorn
        from app.main import app
        
        print(f"\n🚀 Starting server on {args.host}:{args.port}...")
        print("   Press Ctrl+C to stop the server")
        print("-" * 60)
        
        # Start the server
        uvicorn.run(
            "app.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers,
            log_level=args.log_level,
            access_log=True
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()