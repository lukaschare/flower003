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

# 1. Update the package list and install essential system tools
sudo apt-get update
sudo apt-get install -y curl ca-certificates gnupg

# 2. Check and Install Docker (Using Docker Official One-Click Installation Script)
if ! command -v docker &> /dev/null; then
    echo ">>> Installing Docker via official convenience script..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    rm get-docker.sh
else
    echo ">>> Docker is already installed, skipping."
fi

# 3. Install other specified dependency packages
sudo apt-get install -y nlohmann-json3-dev python3.9-venv

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
rm -rf "$TARGET_FEDITSTOOL"
cp -r "$PROJECT_ROOT/fedits-tool" "$HOME/"

if [ -d "$VEINS_DIR" ]; then
    echo "Deploying fedits_veins_rsu to $VEINS_DIR/examples/"
    rm -rf "$VEINS_DIR/examples/fedits_veins_rsu"
    cp -r "$PROJECT_ROOT/fedits_veins_rsu" "$VEINS_DIR/examples/"

    # 
    rm -rf "$VEINS_DIR/examples/fedits_veins_rsu/out"
    rm -f  "$VEINS_DIR/examples/fedits_veins_rsu/fedits_veins_rsu"
    rm -f  "$VEINS_DIR/examples/fedits_veins_rsu/libfedits_veins_rsu.so"
else
    echo "? Error: Veins source directory not found at $VEINS_DIR. Ensure you are on the Instant Veins VM."
    exit 1
fi

# ---------------------------------------------------------
echo -e "\n>>> [3/5] Compiling Veins C++ simulation module..."
# ---------------------------------------------------------
cd "$VEINS_DIR"
if [ ! -f "Makefile" ]; then
    echo ">>> Running ./configure..."
    ./configure
fi

cd "$VEINS_DIR/examples/fedits_veins_rsu"
echo ">>> Running make clean..."
make clean || true

echo ">>> Running make -j$(nproc)..."
make -j"$(nproc)"

# solve veins Permission denied
if [ -f "./out/clang-release/fedits_veins_rsu" ]; then
    chmod +x ./out/clang-release/fedits_veins_rsu
fi
if [ -f "./fedits_veins_rsu" ]; then
    chmod +x ./fedits_veins_rsu
fi

# ---------------------------------------------------------
echo -e "\n>>> [4/5] Installing Python dependencies for the Dashboard..."
# ---------------------------------------------------------
cd "$TARGET_FEDITSTOOL"
if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
    echo ">>> Found requirements.txt in project root, installing globally..."
    pip3 install -r "$PROJECT_ROOT/requirements.txt"
else
    echo "?? Warning: requirements.txt not found, skipping Python dependency installation."
fi

# ---------------------------------------------------------
echo -e "\n>>> [4.5/5] Fixing executable permissions..."
# ---------------------------------------------------------
if [ -d "$TARGET_FEDITSTOOL/scripts" ]; then
    find "$TARGET_FEDITSTOOL/scripts" -maxdepth 1 -type f -name "*.sh" -exec chmod +x {} \;
fi

if [ -f "$VEINS_DIR/examples/fedits_veins_rsu/out/clang-release/fedits_veins_rsu" ]; then
    chmod +x "$VEINS_DIR/examples/fedits_veins_rsu/out/clang-release/fedits_veins_rsu"
fi

# ---------------------------------------------------------
echo -e "\n>>> [5/5] Deployment Complete! ??"
# ---------------------------------------------------------
echo "============================================================"
echo "? FedITS environment is ready!"
echo ""
if [ "$NEED_RELOGIN" -eq 1 ]; then
    echo "?? NOTICE: Your user was just added to the docker group."
    echo "?? You MUST log out and log back in (or reboot the VM) for Docker permissions to take effect before running the simulation."
    echo ""
fi
echo "? Command to run the simulation:"
echo "   cd ~/fedits-tool/scripts"
echo "   ./run_fedits_scenario_v2.sh gui"
echo "============================================================"
