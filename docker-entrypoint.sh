#!/bin/bash
set -e

echo "🚀 Executing Tricked AI Backend..."

echo "Booting Python Backend API (Port 8080)..."
cd /app
python3 src/tricked_web/server.py
