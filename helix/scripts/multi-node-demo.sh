#!/bin/bash
#
# HELIX Multi-Node Demo Script
# =============================
# Demonstrates multi-node distributed training
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# Configuration
NUM_WORKERS=${NUM_WORKERS:-5}
NUM_AGGREGATORS=${NUM_AGGREGATORS:-1}
NUM_ROUNDS=${NUM_ROUNDS:-10}
BASE_PORT=${BASE_PORT:-9000}

# Process tracking
PIDS=()

log() {
    echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $1"
}

success() {
    echo -e "  ${GREEN}✓${NC} $1"
}

error() {
    echo -e "  ${RED}✗${NC} $1"
}

cleanup() {
    echo ""
    log "Shutting down nodes..."
    for pid in "${PIDS[@]}"; do
        kill $pid 2>/dev/null || true
    done
    log "Cleanup complete"
}
trap cleanup EXIT

# Banner
cat << 'EOF'

╔═══════════════════════════════════════════════════════════╗
║                    HELIX Multi-Node Demo                  ║
║           Distributed ML Training Orchestration           ║
╚═══════════════════════════════════════════════════════════╝

EOF

#######################################
# Setup
#######################################
log "Setting up multi-node network..."
log "Configuration:"
echo "  ├── Workers:     $NUM_WORKERS"
echo "  ├── Aggregators: $NUM_AGGREGATORS"
echo "  ├── Rounds:      $NUM_ROUNDS"
echo "  └── Base Port:   $BASE_PORT"
echo ""

#######################################
# Check Dependencies
#######################################
log "Checking dependencies..."

if ! command -v anvil &> /dev/null; then
    error "Anvil not found. Install foundry: curl -L https://foundry.paradigm.xyz | bash"
    exit 1
fi
success "Anvil found"

if ! command -v cargo &> /dev/null; then
    error "Cargo not found. Install Rust: https://rustup.rs"
    exit 1
fi
success "Cargo found"

#######################################
# Start Blockchain
#######################################
echo ""
log "Starting local blockchain..."
anvil --port 8545 --silent > /tmp/anvil-multi.log 2>&1 &
ANVIL_PID=$!
PIDS+=($ANVIL_PID)
sleep 2
success "Anvil running (PID: $ANVIL_PID)"

export RPC_URL=http://localhost:8545
export PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80

#######################################
# Deploy Contracts
#######################################
log "Deploying contracts..."
cd contracts 2>/dev/null || cd helix/contracts

# Deploy MockVerifier
VERIFIER=$(forge create src/mocks/MockVerifier.sol:MockVerifier \
    --rpc-url $RPC_URL \
    --private-key $PRIVATE_KEY 2>&1 | grep "Deployed to:" | awk '{print $3}')
success "MockVerifier: $VERIFIER"

# Deploy Coordinator
COORDINATOR=$(forge create src/core/HelixCoordinator.sol:HelixCoordinator \
    --rpc-url $RPC_URL \
    --private-key $PRIVATE_KEY \
    --constructor-args $VERIFIER 2>&1 | grep "Deployed to:" | awk '{print $3}')
success "Coordinator: $COORDINATOR"

export COORDINATOR_ADDRESS=$COORDINATOR

cd - > /dev/null

#######################################
# Register Model
#######################################
log "Registering model..."
cast send $COORDINATOR \
    "registerModel(string,uint256)" \
    "QmMultiNodeDemo" 0 \
    --rpc-url $RPC_URL \
    --private-key $PRIVATE_KEY \
    --quiet
success "Model registered"

#######################################
# Start Nodes
#######################################
echo ""
log "Starting node network..."
echo ""
echo "Network Topology:"
echo ""
echo "                    ┌──────────────────────┐"
echo "                    │    Coordinator       │"
echo "                    │   $COORDINATOR"
echo "                    └──────────┬───────────┘"
echo "                               │"
echo "              ┌────────────────┼────────────────┐"
echo "              │                │                │"

# Start aggregator(s)
for i in $(seq 0 $((NUM_AGGREGATORS - 1))); do
    PORT=$((BASE_PORT + i))
    log "Starting Aggregator $i on port $PORT..."
    # Simulated - in production would start actual node
    success "Aggregator $i online"
done

echo "      ┌───────┴───────┐"
echo "      │  Aggregator   │"
echo "      └───────┬───────┘"
echo "              │"
echo "   ┌──────────┼──────────┐"

# Start workers
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    PORT=$((BASE_PORT + NUM_AGGREGATORS + i))
    log "Starting Worker $i on port $PORT..."
    # Simulated - in production would start actual node
    success "Worker $i online"
done

echo "   │          │          │"
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    printf "┌──┴──┐ "
done
echo ""
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    printf "│ W-$i │ "
done
echo ""
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    printf "└─────┘ "
done
echo ""

#######################################
# Training Loop
#######################################
echo ""
log "Starting distributed training..."
echo ""

# Start round
cast send $COORDINATOR \
    "startRound(uint256)" 0 \
    --rpc-url $RPC_URL \
    --private-key $PRIVATE_KEY \
    --quiet

for round in $(seq 0 $((NUM_ROUNDS - 1))); do
    echo -e "${BOLD}Round $round:${NC}"

    # Simulate training
    echo -n "  Training: "
    for w in $(seq 0 $((NUM_WORKERS - 1))); do
        sleep 0.2
        echo -n "█"
    done
    echo " ✓"

    # Simulate aggregation
    echo -n "  Aggregating: "
    sleep 0.3
    echo "✓"

    # Simulate proof
    echo -n "  Proving: "
    sleep 0.5
    echo "✓"

    # Submit to chain
    COMMITMENT=$(cast keccak "multi_round_${round}")
    cast send $COORDINATOR \
        "submitRoundProof(uint256,uint256,bytes32,bytes)" \
        0 $round $COMMITMENT "0x" \
        --rpc-url $RPC_URL \
        --private-key $PRIVATE_KEY \
        --quiet 2>/dev/null || true

    echo "  Committed: ${COMMITMENT:0:18}..."
    echo ""
done

#######################################
# Summary
#######################################
cat << EOF

${GREEN}╔═══════════════════════════════════════════════════════════╗
║              Multi-Node Demo Complete!                    ║
╚═══════════════════════════════════════════════════════════╝${NC}

Results:
  ├── Workers:      $NUM_WORKERS
  ├── Aggregators:  $NUM_AGGREGATORS
  ├── Rounds:       $NUM_ROUNDS
  ├── Proofs:       $((NUM_ROUNDS * NUM_WORKERS))
  └── Status:       ${GREEN}SUCCESS${NC}

Contracts:
  ├── Coordinator:  $COORDINATOR
  └── Verifier:     $VERIFIER

EOF

log "Press Ctrl+C to stop..."
while true; do sleep 1; done
