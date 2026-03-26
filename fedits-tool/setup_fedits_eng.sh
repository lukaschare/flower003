#!/usr/bin/env bash
# ==============================================================================
# FedITS One-click Setup Script
# Target Environment: Instant Veins VirtualBox VM (Ubuntu/Debian based)
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status

echo "============================================================"
echo "🚀 Starting automated deployment for the FedITS Framework..."
echo "============================================================"

# Get the absolute path of the project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VEINS_DIR="$HOME/src/veins"
TARGET_FEDITSTOOL="$HOME/fedits-tool"

# ---------------------------------------------------------
echo -e "\n>>> [1/5] Installing system-level dependencies (Docker, JSON, Python-venv)..."
# ---------------------------------------------------------
sudo apt-get update
sudo apt-get install -y \
    docker-ce docker-ce-cli containerd.io docker-compose-plugin \
    nlohmann-json3-dev \
    python3.9-venv

# Automatically add the current user to the docker group
if ! groups | grep -q "\bdocker\b"; then
    echo ">>> Adding current user ($USER) to the 'docker' group..."
    sudo usermod -aG docker "$USER"
    NEED_RELOGIN=1
else
    NEED_RELOGIN=0
fi

# ---------------------------------------------------------
echo -e "\n>>> [2/5] Deploying project files..."
# ---------------------------------------------------------
echo "Copying fedits-tool to $HOME/"
cp -r "$PROJECT_ROOT/fedits-tool" "$HOME/"

if [ -d "$VEINS_DIR" ]; then
    echo "Copying fedits_veins_rsu to $VEINS_DIR/examples/"
    cp -r "$PROJECT_ROOT/fedits_veins_rsu" "$VEINS_DIR/examples/"
else
    echo "❌ Error: Veins source directory not found at $VEINS_DIR. Ensure you are on the Instant Veins VM."
    exit 1
fi

# ---------------------------------------------------------
echo -e "\n>>> [3/5] Compiling Veins C++ simulation module (this may take a few minutes)..."
# ---------------------------------------------------------
cd "$VEINS_DIR"
# Run configure if Makefile doesn't exist
if [ ! -f "Makefile" ]; then
    echo ">>> Running ./configure..."
    ./configure
fi
echo ">>> Running make -j$(nproc)..."
make -j$(nproc)

# ---------------------------------------------------------
echo -e "\n>>> [4/5] Installing Python dependencies for the Dashboard..."
# ---------------------------------------------------------
cd "$TARGET_FEDITSTOOL"
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
else
    echo "⚠️ Warning: requirements.txt not found, skipping Python dependency installation."
fi

# ---------------------------------------------------------
echo -e "\n>>> [5/5] Deployment Complete! 🎉"
# ---------------------------------------------------------
echo "============================================================"
echo "✅ FedITS environment is ready!"
echo ""
if [ "$NEED_RELOGIN" -eq 1 ]; then
    echo "⚠️ NOTICE: Your user was just added to the docker group."
    echo "⚠️ You MUST log out and log back in (or reboot the VM) for Docker permissions to take effect before running the simulation."
    echo ""
fi
echo "▶ Command to run the simulation:"
echo "   cd ~/fedits-tool"
echo "   ./run_fedits_scenario_v2.sh gui"
echo "============================================================"