#!/usr/bin/env python3
"""
Development server starter
Starts both the Python backend and React frontend
"""

import subprocess
import sys
import os
import time
import signal
from threading import Thread

# Define the project directory
PROJECT_DIR = "project-bolt-github-v82hbkom/project"

def start_backend():
    """Start the Python Flask backend"""
    print("🐍 Starting Python backend...")
    try:
        subprocess.run([sys.executable, "web_server.py"], cwd=PROJECT_DIR, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Backend stopped")
    except Exception as e:
        print(f"❌ Backend error: {e}")

def start_frontend():
    """Start the React frontend"""
    print("⚛️  Starting React frontend...")
    try:
        subprocess.run(["npm", "run", "dev"], cwd=PROJECT_DIR, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Frontend stopped")
    except Exception as e:
        print(f"❌ Frontend error: {e}")

def main():
    """Start both servers"""
    print("🚀 Starting RAG File Processor Development Environment")
    print("=" * 60)
    
    # Check if npm is available
    try:
        subprocess.run(["npm", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ npm not found. Please install Node.js and npm first.")
        sys.exit(1)
    
    # Check if dependencies are installed
    if not os.path.exists(os.path.join(PROJECT_DIR, "node_modules")):
        print("📦 Installing npm dependencies...")
        subprocess.run(["npm", "install"], cwd=PROJECT_DIR, check=True)
    
    try:
        # Start backend in a separate thread
        backend_thread = Thread(target=start_backend, daemon=True)
        backend_thread.start()
        
        # Give backend time to start
        time.sleep(2)
        
        # Start frontend (this will block)
        start_frontend()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down development servers...")
        sys.exit(0)

if __name__ == "__main__":
    main()