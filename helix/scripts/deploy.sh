#!/bin/bash
#
# HELIX Contract Deployment Script
# =================================
# Deploys the HELIX smart contracts to a target network
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# Default configuration
NETWORK=${NETWORK:-local}
RPC_URL=${RPC_URL:-http://localhost:8545}
PRIVATE_KEY=${PRIVATE_KEY:-0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80}
VERIFY=${VERIFY:-false}

# Output files
DEPLOY_LOG="deploy_$(date +%Y%m%d_%H%M%S).log"
ADDRESSES_FILE="deployed_addresses.json"

log() {
    echo -e "${CYAN}[DEPLOY]${NC} $1" | tee -a $DEPLOY_LOG
}

success() {
    echo -e "  ${GREEN}✓${NC} $1" | tee -a $DEPLOY_LOG
}

error() {
    echo -e "  ${RED}✗${NC} $1" | tee -a $DEPLOY_LOG
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --network)
            NETWORK="$2"
            shift 2
            ;;
        --rpc-url)
            RPC_URL="$2"
            shift 2
            ;;
        --private-key)
            PRIVATE_KEY="$2"
            shift 2
            ;;
        --verify)
            VERIFY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Network presets
case $NETWORK in
    local)
        RPC_URL=${RPC_URL:-http://localhost:8545}
        ;;
    sepolia)
        RPC_URL=${RPC_URL:-https://rpc.sepolia.org}
        VERIFY=true
        ;;
    mainnet)
        RPC_URL=${RPC_URL:-https://eth.llamarpc.com}
        VERIFY=true
        ;;
esac

# Banner
cat << 'EOF'
╔═══════════════════════════════════════════════════════════╗
║              HELIX Contract Deployment                    ║
╚═══════════════════════════════════════════════════════════╝
EOF

echo ""
log "Configuration:"
echo "  ├── Network:     $NETWORK"
echo "  ├── RPC URL:     $RPC_URL"
echo "  ├── Verify:      $VERIFY"
echo "  └── Log:         $DEPLOY_LOG"
echo ""

# Check RPC connection
log "Testing RPC connection..."
CHAIN_ID=$(cast chain-id --rpc-url $RPC_URL 2>/dev/null) || error "Cannot connect to RPC"
success "Connected to chain ID: $CHAIN_ID"

# Check deployer balance
log "Checking deployer balance..."
DEPLOYER=$(cast wallet address --private-key $PRIVATE_KEY)
BALANCE=$(cast balance $DEPLOYER --rpc-url $RPC_URL 2>/dev/null)
success "Deployer: $DEPLOYER"
success "Balance: $(cast from-wei $BALANCE) ETH"

# Navigate to contracts
cd "$(dirname "$0")/../contracts" || error "Contracts directory not found"

# Build contracts
log "Building contracts..."
forge build --silent || error "Build failed"
success "Contracts compiled"

#######################################
# Deploy MockVerifier
#######################################
log "Deploying MockVerifier..."
VERIFIER_OUTPUT=$(forge create src/mocks/MockVerifier.sol:MockVerifier \
    --rpc-url $RPC_URL \
    --private-key $PRIVATE_KEY \
    --json 2>/dev/null) || VERIFIER_OUTPUT=""

if [ -n "$VERIFIER_OUTPUT" ]; then
    VERIFIER_ADDRESS=$(echo $VERIFIER_OUTPUT | jq -r '.deployedTo')
else
    VERIFIER_ADDRESS=$(forge create src/mocks/MockVerifier.sol:MockVerifier \
        --rpc-url $RPC_URL \
        --private-key $PRIVATE_KEY 2>&1 | grep "Deployed to:" | awk '{print $3}')
fi

[ -z "$VERIFIER_ADDRESS" ] && error "Failed to deploy MockVerifier"
success "MockVerifier: $VERIFIER_ADDRESS"

#######################################
# Deploy HelixCoordinator
#######################################
log "Deploying HelixCoordinator..."
COORDINATOR_OUTPUT=$(forge create src/core/HelixCoordinator.sol:HelixCoordinator \
    --rpc-url $RPC_URL \
    --private-key $PRIVATE_KEY \
    --constructor-args $VERIFIER_ADDRESS \
    --json 2>/dev/null) || COORDINATOR_OUTPUT=""

if [ -n "$COORDINATOR_OUTPUT" ]; then
    COORDINATOR_ADDRESS=$(echo $COORDINATOR_OUTPUT | jq -r '.deployedTo')
else
    COORDINATOR_ADDRESS=$(forge create src/core/HelixCoordinator.sol:HelixCoordinator \
        --rpc-url $RPC_URL \
        --private-key $PRIVATE_KEY \
        --constructor-args $VERIFIER_ADDRESS 2>&1 | grep "Deployed to:" | awk '{print $3}')
fi

[ -z "$COORDINATOR_ADDRESS" ] && error "Failed to deploy HelixCoordinator"
success "HelixCoordinator: $COORDINATOR_ADDRESS"

#######################################
# Verify Contracts (if enabled)
#######################################
if [ "$VERIFY" = true ]; then
    log "Verifying contracts on Etherscan..."

    forge verify-contract $VERIFIER_ADDRESS \
        src/mocks/MockVerifier.sol:MockVerifier \
        --chain-id $CHAIN_ID \
        --watch 2>/dev/null || log "Verification skipped or failed"

    forge verify-contract $COORDINATOR_ADDRESS \
        src/core/HelixCoordinator.sol:HelixCoordinator \
        --chain-id $CHAIN_ID \
        --constructor-args $(cast abi-encode "constructor(address)" $VERIFIER_ADDRESS) \
        --watch 2>/dev/null || log "Verification skipped or failed"
fi

#######################################
# Write Addresses File
#######################################
cat > $ADDRESSES_FILE << EOF
{
    "network": "$NETWORK",
    "chainId": $CHAIN_ID,
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "deployer": "$DEPLOYER",
    "contracts": {
        "MockVerifier": "$VERIFIER_ADDRESS",
        "HelixCoordinator": "$COORDINATOR_ADDRESS"
    }
}
EOF

log "Addresses written to: $ADDRESSES_FILE"

#######################################
# Summary
#######################################
echo ""
cat << EOF
${GREEN}╔═══════════════════════════════════════════════════════════╗
║              Deployment Complete!                         ║
╚═══════════════════════════════════════════════════════════╝${NC}

Deployed Contracts:
  ├── MockVerifier:      $VERIFIER_ADDRESS
  └── HelixCoordinator:  $COORDINATOR_ADDRESS

Network: $NETWORK (Chain ID: $CHAIN_ID)
Deployer: $DEPLOYER

Environment Variables:
  export COORDINATOR_ADDRESS=$COORDINATOR_ADDRESS
  export VERIFIER_ADDRESS=$VERIFIER_ADDRESS
  export RPC_URL=$RPC_URL

EOF
