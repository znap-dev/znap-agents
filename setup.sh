#!/bin/bash
#
# ZNAP Autonomous Agents - Setup Script
# =====================================
# Quick setup for Ubuntu/macOS with Nvidia GPU support
#

set -e

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║         ZNAP Autonomous Agents - Setup                    ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# =====================================================
# Step 1: Check Python
# =====================================================
echo -e "${YELLOW}[1/4] Checking Python...${NC}"

if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Please install Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

# =====================================================
# Step 2: Setup Virtual Environment
# =====================================================
echo -e "${YELLOW}[2/4] Setting up virtual environment...${NC}"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "Dependencies installed"

# =====================================================
# Step 3: Check Ollama
# =====================================================
echo -e "${YELLOW}[3/4] Checking Ollama...${NC}"

if ! command -v ollama &> /dev/null; then
    echo ""
    echo "Ollama not found. Install it:"
    echo "  curl -fsSL https://ollama.com/install.sh | sh"
    echo ""
    echo "Then run: ollama pull kimi-k2.5:cloud"
    echo ""
else
    echo "Ollama found: $(ollama --version 2>/dev/null || echo 'installed')"
    
    # Check if model exists
    if ollama list 2>/dev/null | grep -q "kimi-k2.5"; then
        echo "Model kimi-k2.5:cloud ready"
    else
        echo ""
        read -p "Pull kimi-k2.5:cloud model? (Y/n): " pull_model
        if [[ ! "$pull_model" =~ ^[Nn]$ ]]; then
            ollama pull kimi-k2.5:cloud
        fi
    fi
fi

# =====================================================
# Step 4: Setup Environment File
# =====================================================
echo -e "${YELLOW}[4/4] Setting up configuration...${NC}"

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ".env file created from .env.example"
else
    echo ".env file already exists"
fi

# Create directories
mkdir -p .keys .memory

# =====================================================
# Done!
# =====================================================
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Setup Complete!                        ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "To run the agents:"
echo ""
echo "  1. Activate virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Make sure Ollama is running:"
echo "     ollama serve"
echo ""
echo "  3. Run the agents:"
echo "     python run_autonomous.py              # All agents"
echo "     python run_autonomous.py -a nexus     # Single agent"
echo "     python run_autonomous.py --list       # List agents"
echo ""
