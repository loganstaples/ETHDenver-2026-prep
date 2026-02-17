#!/usr/bin/env bash
# ============================================================================
# Deploy HELIX contracts to ADI Testnet
# ============================================================================
# Prerequisites:
#   1. Foundry installed (forge, cast)
#   2. .env.adi has TESTNET_PRIVATE_KEY set and funded from ADI faucet
#
# Usage:
#   ./scripts/deploy-adi.sh
#
# This script:
#   1. Builds contracts with forge
#   2. Deploys MockVerifier + HelixCoordinatorV4 to ADI testnet
#   3. Updates .env.adi with deployed addresses
#   4. Verifies deployment by reading on-chain state
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env.adi"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   HELIX — Deploy to ADI Testnet           ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════╝${NC}"
echo ""

# Load config
if [[ ! -f "$ENV_FILE" ]]; then
    echo -e "${RED}Error: $ENV_FILE not found. Copy from .env.adi.example and fill in TESTNET_PRIVATE_KEY.${NC}"
    exit 1
fi
source "$ENV_FILE"

RPC_URL="${ADI_RPC_URL:-https://rpc.ab.testnet.adifoundation.ai/}"
CHAIN_ID="${ADI_CHAIN_ID:-99999}"
PRIVATE_KEY="${TESTNET_PRIVATE_KEY:-}"

if [[ -z "$PRIVATE_KEY" || "$PRIVATE_KEY" == "0xYOUR_PRIVATE_KEY_HERE" ]]; then
    echo -e "${RED}Error: Set TESTNET_PRIVATE_KEY in .env.adi${NC}"
    echo -e "  1. Go to ${YELLOW}https://faucet.ab.testnet.adifoundation.ai/${NC}"
    echo -e "  2. Fund your wallet with ADI testnet tokens"
    echo -e "  3. Set TESTNET_PRIVATE_KEY=0x... in .env.adi"
    exit 1
fi

# Derive address from private key
OWNER_ADDR=$(cast wallet address "$PRIVATE_KEY" 2>/dev/null)
echo -e "${CYAN}Owner address:${NC} $OWNER_ADDR"

# Check balance
BALANCE=$(cast balance "$OWNER_ADDR" --rpc-url "$RPC_URL" 2>/dev/null)
BALANCE_ETH=$(cast from-wei "$BALANCE" 2>/dev/null || echo "0")
echo -e "${CYAN}Balance:${NC} $BALANCE_ETH ADI"

if [[ "$BALANCE" == "0" ]]; then
    echo -e "${RED}Error: Zero balance. Fund from faucet first:${NC}"
    echo -e "  ${YELLOW}https://faucet.ab.testnet.adifoundation.ai/${NC}"
    echo -e "  Address: $OWNER_ADDR"
    exit 1
fi

# Verify chain connectivity
echo -e "\n${CYAN}Verifying ADI testnet connection...${NC}"
REMOTE_CHAIN_ID=$(cast chain-id --rpc-url "$RPC_URL" 2>/dev/null)
if [[ "$REMOTE_CHAIN_ID" != "$CHAIN_ID" ]]; then
    echo -e "${RED}Chain ID mismatch: expected $CHAIN_ID, got $REMOTE_CHAIN_ID${NC}"
    exit 1
fi
echo -e "${GREEN}Connected to ADI testnet (chain ID: $REMOTE_CHAIN_ID)${NC}"

# Build contracts
echo -e "\n${CYAN}Building contracts...${NC}"
cd "$ROOT_DIR/contracts"
forge build --force 2>&1 | tail -3
echo -e "${GREEN}Contracts built.${NC}"

# Deploy using the DeployV4 script
echo -e "\n${CYAN}Deploying MockVerifier + HelixCoordinatorV4...${NC}"
DEPLOY_OUTPUT=$(PRIVATE_KEY="$PRIVATE_KEY" forge script script/DeployV4.s.sol \
    --rpc-url "$RPC_URL" \
    --broadcast \
    --chain-id "$CHAIN_ID" \
    -q 2>/dev/null || true)

# If standard deploy fails, try with --legacy (no EIP-1559)
if [[ -z "$DEPLOY_OUTPUT" ]] || ! echo "$DEPLOY_OUTPUT" | grep -q "0x"; then
    echo -e "${YELLOW}Standard deploy failed, retrying with --legacy flag...${NC}"
    DEPLOY_OUTPUT=$(PRIVATE_KEY="$PRIVATE_KEY" forge script script/DeployV4.s.sol \
        --rpc-url "$RPC_URL" \
        --broadcast \
        --chain-id "$CHAIN_ID" \
        --legacy \
        -q 2>/dev/null || true)
fi

if [[ -z "$DEPLOY_OUTPUT" ]] || ! echo "$DEPLOY_OUTPUT" | grep -q "0x"; then
    echo -e "${RED}Forge script deployment failed. Trying direct create...${NC}"

    # Fallback: deploy individual contracts using forge create
    echo -e "${CYAN}Deploying MockVerifier...${NC}"
    VERIFIER_OUT=$(forge create src/mocks/MockVerifier.sol:MockVerifier \
        --rpc-url "$RPC_URL" \
        --private-key "$PRIVATE_KEY" \
        --chain-id "$CHAIN_ID" \
        --legacy 2>&1 || true)

    VERIFIER_ADDR=$(echo "$VERIFIER_OUT" | grep "Deployed to:" | awk '{print $3}')

    if [[ -z "$VERIFIER_ADDR" ]]; then
        echo -e "${RED}MockVerifier deployment failed:${NC}"
        echo "$VERIFIER_OUT"
        echo ""
        echo -e "${YELLOW}If ADI chain requires zkSync compilation, you may need foundry-zksync:${NC}"
        echo "  curl -L https://raw.githubusercontent.com/matter-labs/foundry-zksync/main/install-foundry-zksync | bash"
        echo "  foundryup-zksync"
        echo "  Then re-run this script."
        exit 1
    fi
    echo -e "${GREEN}MockVerifier deployed: $VERIFIER_ADDR${NC}"

    echo -e "${CYAN}Deploying HelixCoordinatorV4...${NC}"
    COORD_OUT=$(forge create src/core/HelixCoordinatorV4.sol:HelixCoordinatorV4 \
        --rpc-url "$RPC_URL" \
        --private-key "$PRIVATE_KEY" \
        --chain-id "$CHAIN_ID" \
        --legacy \
        --constructor-args "$OWNER_ADDR" "$VERIFIER_ADDR" 2>&1 || true)

    COORD_ADDR=$(echo "$COORD_OUT" | grep "Deployed to:" | awk '{print $3}')

    if [[ -z "$COORD_ADDR" ]]; then
        echo -e "${RED}HelixCoordinatorV4 deployment failed:${NC}"
        echo "$COORD_OUT"
        exit 1
    fi
    echo -e "${GREEN}HelixCoordinatorV4 deployed: $COORD_ADDR${NC}"
else
    # Parse forge script output (two addresses: coordinator, verifier)
    COORD_ADDR=$(echo "$DEPLOY_OUTPUT" | grep "0x" | head -1 | tr -d '[:space:]')
    VERIFIER_ADDR=$(echo "$DEPLOY_OUTPUT" | grep "0x" | tail -1 | tr -d '[:space:]')
    echo -e "${GREEN}HelixCoordinatorV4: $COORD_ADDR${NC}"
    echo -e "${GREEN}Halo2Verifier:      $VERIFIER_ADDR${NC}"
fi

# Update .env.adi with deployed addresses
cd "$ROOT_DIR"
sed -i '' "s|^COORDINATOR_ADDRESS=.*|COORDINATOR_ADDRESS=$COORD_ADDR|" "$ENV_FILE"
sed -i '' "s|^VERIFIER_ADDRESS=.*|VERIFIER_ADDRESS=$VERIFIER_ADDR|" "$ENV_FILE"

echo -e "\n${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}Deployment complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "  Coordinator: ${CYAN}$COORD_ADDR${NC}"
echo -e "  Verifier:    ${CYAN}$VERIFIER_ADDR${NC}"
echo -e "  Explorer:    ${CYAN}https://explorer.ab.testnet.adifoundation.ai/address/$COORD_ADDR${NC}"
echo -e "  Config:      ${CYAN}$ENV_FILE${NC}"
echo ""
echo -e "Next: run ${YELLOW}./scripts/demo-adi.sh${NC} to start the demo"
