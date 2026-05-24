#!/bin/bash

# Exit immediately if any command fails
set -e

# Trap CTRL+C (SIGINT) and exit signals to cleanly kill background processes
trap "echo -e '\nShutting down system processes...'; kill 0" EXIT

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================================="
echo "      Facial Expression Recognition System Launcher       "
echo "=========================================================="

# 1. Start FastAPI Backend
echo -e "\n[1/2] Starting FastAPI Backend on http://localhost:8000 ..."
cd "$PROJECT_ROOT/backend"
# Run Uvicorn inside the Conda environment
/opt/anaconda3/bin/conda run --no-capture-output -n fer-backend uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload &
cd "$PROJECT_ROOT"

# 2. Start Vite Frontend
echo -e "\n[2/2] Starting Vite React Frontend..."
cd "$PROJECT_ROOT/frontend"
# Run Vite dev server in the background using Conda Node environment
/opt/anaconda3/bin/conda run -n fer-frontend npm run dev &
cd "$PROJECT_ROOT"

echo -e "\nBoth servers are running! Access the frontend at the URL printed above."
echo "Press Ctrl+C to terminate both servers cleanly."

# Keep parent script alive to receive signals and keep background jobs active
wait
