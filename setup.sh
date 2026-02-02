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
RED='\033[0;31m'
NC='\033[0m'

# =====================================================
# Step 1: Check/Install NVIDIA GPU Drivers (Linux only)
# =====================================================
echo -e "${YELLOW}[1/5] Checking GPU drivers...${NC}"

install_nvidia_drivers() {
    echo "Installing NVIDIA drivers and CUDA toolkit..."
    
    # Detect OS
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
    fi
    
    if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
        # Ubuntu/Debian
        sudo apt-get update
        sudo apt-get install -y ubuntu-drivers-common
        
        # Install recommended driver
        sudo ubuntu-drivers autoinstall || {
            # Fallback: manual install
            echo "Auto-install failed, trying manual install..."
            sudo apt-get install -y nvidia-driver-535 nvidia-utils-535
        }
        
        # Install CUDA toolkit (for Ollama GPU support)
        if ! command -v nvcc &> /dev/null; then
            sudo apt-get install -y nvidia-cuda-toolkit
        fi
        
        echo -e "${GREEN}NVIDIA drivers installed. Reboot may be required.${NC}"
        
    elif [ "$OS" = "centos" ] || [ "$OS" = "rhel" ] || [ "$OS" = "rocky" ]; then
        # CentOS/RHEL/Rocky
        sudo dnf install -y epel-release
        sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
        sudo dnf install -y nvidia-driver nvidia-driver-cuda
        
    else
        echo -e "${RED}Unsupported OS for automatic driver install: $OS${NC}"
        echo "Please install NVIDIA drivers manually:"
        echo "  https://docs.nvidia.com/cuda/cuda-installation-guide-linux/"
        return 1
    fi
}

# Check if running on Linux with GPU
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Check for NVIDIA GPU
    if lspci 2>/dev/null | grep -i nvidia > /dev/null; then
        echo "NVIDIA GPU detected"
        
        # Check if driver is installed
        if command -v nvidia-smi &> /dev/null; then
            echo "NVIDIA driver installed:"
            nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "  (driver loaded)"
        else
            echo -e "${RED}NVIDIA GPU found but drivers not installed!${NC}"
            echo ""
            read -p "Install NVIDIA drivers now? (Y/n): " install_drivers
            if [[ ! "$install_drivers" =~ ^[Nn]$ ]]; then
                install_nvidia_drivers
                echo ""
                echo -e "${YELLOW}Please reboot and run setup.sh again after driver installation.${NC}"
                exit 0
            else
                echo -e "${YELLOW}Warning: Without GPU drivers, Ollama will use CPU (very slow)${NC}"
            fi
        fi
    else
        echo "No NVIDIA GPU detected (will use CPU)"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macOS detected (Apple Silicon or Intel)"
    # macOS uses Metal, no NVIDIA drivers needed
fi

# =====================================================
# Step 2: Check Python
# =====================================================
echo -e "${YELLOW}[2/5] Checking Python...${NC}"

if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Please install Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

# =====================================================
# Step 3: Setup Virtual Environment
# =====================================================
echo -e "${YELLOW}[3/5] Setting up virtual environment...${NC}"

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
# Step 4: Check Ollama
# =====================================================
echo -e "${YELLOW}[4/5] Checking Ollama...${NC}"

if ! command -v ollama &> /dev/null; then
    echo ""
    echo "Ollama not found. Install it:"
    echo "  curl -fsSL https://ollama.com/install.sh | sh"
    echo ""
    echo "Then run: ollama pull glm-4.7-flash:latest"
    echo ""
else
    echo "Ollama found: $(ollama --version 2>/dev/null || echo 'installed')"
    
    # Check if model exists
    if ollama list 2>/dev/null | grep -q "kimi-k2.5"; then
        echo "Model glm-4.7-flash:latest ready"
    else
        echo ""
        read -p "Pull glm-4.7-flash:latest model? (Y/n): " pull_model
        if [[ ! "$pull_model" =~ ^[Nn]$ ]]; then
            ollama pull glm-4.7-flash:latest
        fi
    fi
fi

# =====================================================
# Step 5: Setup Environment File
# =====================================================
echo -e "${YELLOW}[5/5] Setting up configuration...${NC}"

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

# Show GPU status
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | while read line; do
        echo "  ✓ $line"
    done
    echo ""
fi

echo "To run the agents:"
echo ""
echo "  1. Activate virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Make sure Ollama is running:"
echo "     ollama serve"
echo ""
echo "  3. Run the agents:"
echo "     python run_autonomous.py                    # Default agents"
echo "     python run_autonomous.py -a elon satoshi    # Custom agents"
echo "     python run_autonomous.py --list             # Show info"
echo ""
echo "Model requirements:"
echo "  • glm-4.7-flash (default): 24GB+ VRAM (L4, RTX 3090, A100)"
echo "  • qwen2.5:14b:             16GB+ VRAM (T4, RTX 3060)"
echo "  • llama3.1:8b:              8GB+ VRAM (any GPU)"
echo ""
