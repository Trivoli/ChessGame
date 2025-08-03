#!/bin/bash

# Chess Game Launcher
# This script activates the virtual environment and runs the chess game

cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install chess pygame
    echo "Virtual environment created and dependencies installed!"
else
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo "Starting Chess Game..."
python chess-ai.py 