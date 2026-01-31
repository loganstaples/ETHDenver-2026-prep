#!/bin/bash
#
# HELIX Quick Start Script
# =========================
# Gets HELIX up and running in seconds
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

log() {
    echo -e "${CYAN}[HELIX]${NC} $1"
}

success() {
    echo -e "  ${GREEN}✓${NC} $1"
}

error() {
    echo -e "  ${RED}✗${NC} $1"
    exit 1
}

warn() {
    echo -e "  ${YELLOW}!${NC} $1"
}

# Banner
cat << 'EOF'

   ██╗  ██╗███████╗██╗     ██╗██╗  ██╗
   ██║  ██║██╔════╝██║     ██║╚██╗██╔╝
   ███████║█████╗  ██║     ██║ ╚███╔╝
   ██╔══██║██╔══╝  ██║     ██║ ██╔██╗
   ██║  ██║███████╗███████╗██║██╔╝ ██╗
   ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚═╝  ╚═╝

   Quick Start - Get Running in 60 Seconds
   =========================================

EOF

#######################################
# Step 1: Check Dependencies
#######################################
log "Step 1: Checking dependencies..."

MISSING=()

check_dep() {
    if command -v $1 &> /dev/null; then
        success "$1 found"
    else
        MISSING+=($1)
        warn "$1 not found"
    fi
}

check_dep "cargo"
check_dep "forge"
check_dep "anvil"
check_dep "cast"
check_dep "node"

if [ ${#MISSING[@]} -ne 0 ]; then
    echo ""
    echo -e "${YELLOW}Missing dependencies: ${MISSING[*]}${NC}"
    echo ""
    echo "Install instructions:"
    echo "  - Rust/Cargo: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo "  - Foundry (forge, anvil, cast): curl -L https://foundry.paradigm.xyz | bash && foundryup"
    echo "  - Node.js: https://nodejs.org/"
    echo ""
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

#######################################
# Step 2: Build Projects
#######################################
echo ""
log "Step 2: Building projects..."

# Navigate to helix root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || error "Cannot find helix directory"

# Build Rust projects
log "Building Rust crates..."
cargo build --release -p helix-core -p helix-client 2>&1 | tail -3
success "Rust crates built"

# Build contracts
log "Building smart contracts..."
cd contracts
forge build --silent 2>/dev/null || forge build
success "Contracts built"
cd ..

# Build dashboard (if exists)
if [ -d "dashboard" ] && [ -f "dashboard/package.json" ]; then
    log "Building dashboard..."
    cd dashboard
    npm install --silent 2>/dev/null || npm install
    npm run build --silent 2>/dev/null || true
    success "Dashboard built"
    cd ..
fi

#######################################
# Step 3: Start Services
#######################################
echo ""
log "Step 3: Starting services..."

# Start Anvil
log "Starting local blockchain..."
anvil --port 8545 --silent > /tmp/anvil-quickstart.log 2>&1 &
ANVIL_PID=$!
sleep 2

if kill -0 $ANVIL_PID 2>/dev/null; then
    success "Anvil running (PID: $ANVIL_PID)"
else
    error "Failed to start Anvil"
fi

# Deploy contracts
log "Deploying contracts..."
export RPC_URL=http://localhost:8545
export PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80

cd contracts

VERIFIER=$(forge create src/mocks/MockVerifier.sol:MockVerifier \
    --rpc-url $RPC_URL \
    --private-key $PRIVATE_KEY 2>&1 | grep "Deployed to:" | awk '{print $3}')
success "MockVerifier: $VERIFIER"

COORDINATOR=$(forge create src/core/HelixCoordinator.sol:HelixCoordinator \
    --rpc-url $RPC_URL \
    --private-key $PRIVATE_KEY \
    --constructor-args $VERIFIER 2>&1 | grep "Deployed to:" | awk '{print $3}')
success "HelixCoordinator: $COORDINATOR"

cd ..

export COORDINATOR_ADDRESS=$COORDINATOR
export VERIFIER_ADDRESS=$VERIFIER

# Register a model
log "Registering demo model..."
cast send $COORDINATOR \
    "registerModel(string,uint256)" \
    "QmQuickStart" 0 \
    --rpc-url $RPC_URL \
    --private-key $PRIVATE_KEY \
    --quiet
success "Model registered"

#######################################
# Step 4: Display Info
#######################################
echo ""
cat << EOF
${GREEN}╔═══════════════════════════════════════════════════════════╗
║                 HELIX is Ready!                           ║
╚═══════════════════════════════════════════════════════════╝${NC}

${BOLD}Services Running:${NC}
  ├── Anvil (local blockchain): http://localhost:8545
  └── Contracts deployed

${BOLD}Contract Addresses:${NC}
  ├── HelixCoordinator: $COORDINATOR
  └── MockVerifier:     $VERIFIER

${BOLD}Environment Variables:${NC}
  export COORDINATOR_ADDRESS=$COORDINATOR
  export VERIFIER_ADDRESS=$VERIFIER
  export RPC_URL=http://localhost:8545
  export PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80

${BOLD}Try These Commands:${NC}
  # Check node status
  ./target/release/helix status

  # Run the 90-second demo
  ./scripts/demo-90s.sh

  # Run integration tests
  ./scripts/integration_test.sh

  # Check health
  ./scripts/health-check.sh

${BOLD}Stop Services:${NC}
  kill $ANVIL_PID

EOF

# Save environment
cat > .env.local << EOF
# HELIX Environment (generated $(date))
export COORDINATOR_ADDRESS=$COORDINATOR
export VERIFIER_ADDRESS=$VERIFIER
export RPC_URL=http://localhost:8545
export PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
export ANVIL_PID=$ANVIL_PID
EOF

echo -e "${CYAN}Environment saved to .env.local${NC}"
echo -e "Run: ${BOLD}source .env.local${NC} to load"
echo ""
echo "Press Ctrl+C to stop Anvil and exit..."

# Wait
wait $ANVIL_PID
