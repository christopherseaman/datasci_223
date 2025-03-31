#!/bin/bash

# Function to show usage
show_usage() {
    echo "Usage: $0 [build|serve]"
    echo "  build: Build the documentation (default)"
    echo "  serve: Build and serve the documentation locally"
    exit 1
}

# Check if virtual environment exists and activate it
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Handle command line arguments
if [ $# -eq 0 ] || [ "$1" = "build" ]; then
    # Build site
    echo "Building documentation..."
    mkdocs build
elif [ "$1" = "serve" ]; then
    # Build and serve site
    echo "Building and serving documentation..."
    mkdocs serve
else
    show_usage
fi 