#!/bin/bash
set -e

echo "🚀 Executing Tricked AI Docker Supervisor..."

# Start Python API Data Service in the background
echo "Booting Python Backend (Port 8080)..."
cd /app
python3 src/tricked_web/server.py &

# Start SvelteKit UI in the foreground
echo "Booting SvelteKit Frontend (Port 5173)..."
cd /app/ui
npm run dev -- --host 0.0.0.0
