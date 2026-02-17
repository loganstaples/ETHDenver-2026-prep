#!/usr/bin/env bash
# ============================================================================
# HELIX Demo — ADI Testnet (Two-Laptop or Single-Laptop)
# ============================================================================
# Two-laptop setup:
#   Laptop A (user):   ./scripts/demo-adi.sh
#   Laptop B (workers): ./scripts/demo-adi.sh workers --public-addr <B-IP> --api-url http://<A-IP>:3001
#
# Single-laptop with ADI chain:
#   Terminal 1: ./scripts/demo-adi.sh
#   Terminal 2: ./scripts/demo-adi.sh workers
#
# Prerequisites:
#   1. Run ./scripts/deploy-adi.sh first (deploys contracts)
#   2. .env.adi has TESTNET_PRIVATE_KEY and COORDINATOR_ADDRESS set
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env.adi"
HELIX="$ROOT_DIR/target/debug/helix"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Load config
if [[ ! -f "$ENV_FILE" ]]; then
    echo -e "${RED}Error: $ENV_FILE not found. Run ./scripts/deploy-adi.sh first.${NC}"
    exit 1
fi
source "$ENV_FILE"

RPC_URL="${ADI_RPC_URL:-https://rpc.ab.testnet.adifoundation.ai/}"
PRIVATE_KEY="${TESTNET_PRIVATE_KEY:-}"
COORDINATOR="${COORDINATOR_ADDRESS:-}"

if [[ -z "$PRIVATE_KEY" || "$PRIVATE_KEY" == "0xYOUR_PRIVATE_KEY_HERE" ]]; then
    echo -e "${RED}Error: Set TESTNET_PRIVATE_KEY in .env.adi${NC}"
    exit 1
fi

# Build if needed
if [[ ! -f "$HELIX" ]]; then
    echo -e "${YELLOW}Building helix binary...${NC}"
    cd "$ROOT_DIR" && cargo build -p helix-client 2>&1 | tail -3
fi

# ============================================================================
# Mode: workers
# ============================================================================
if [[ "${1:-}" == "workers" ]]; then
    shift
    NUM_WORKERS="${NUM_WORKERS:-6}"
    BASE_PORT="${BASE_PORT:-9001}"
    SEED="${WORKER_SEED:-42}"
    PUBLIC_ADDR=""
    API_URL="http://127.0.0.1:3001"

    # Parse remaining args
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --public-addr) PUBLIC_ADDR="$2"; shift 2 ;;
            --api-url) API_URL="$2"; shift 2 ;;
            --workers) NUM_WORKERS="$2"; shift 2 ;;
            --base-port) BASE_PORT="$2"; shift 2 ;;
            *) echo "Unknown arg: $1"; exit 1 ;;
        esac
    done

    echo -e "${CYAN}╔═══════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║   HELIX Workers — ADI Testnet             ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  Workers:     ${CYAN}$NUM_WORKERS${NC}"
    echo -e "  Base port:   ${CYAN}$BASE_PORT${NC}"
    echo -e "  Public addr: ${CYAN}${PUBLIC_ADDR:-auto}${NC}"
    echo -e "  Dashboard:   ${CYAN}$API_URL${NC}"
    echo ""

    WORKER_CMD="$HELIX spawn-workers --count $NUM_WORKERS --base-port $BASE_PORT --bind 0.0.0.0 --seed $SEED --api-url $API_URL"
    if [[ -n "$PUBLIC_ADDR" ]]; then
        WORKER_CMD="$WORKER_CMD --public-addr $PUBLIC_ADDR"
    fi

    echo -e "${CYAN}Starting workers...${NC}"
    exec $WORKER_CMD
fi

# ============================================================================
# Mode: dashboard (default)
# ============================================================================
echo -e "${CYAN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   HELIX Demo — ADI Testnet                ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════╝${NC}"
echo ""

if [[ -z "$COORDINATOR" ]]; then
    echo -e "${YELLOW}Warning: No COORDINATOR_ADDRESS in .env.adi${NC}"
    echo -e "${YELLOW}Contracts will be deployed on first training run (costs ADI gas).${NC}"
    echo -e "${YELLOW}Run ./scripts/deploy-adi.sh to pre-deploy for faster demo.${NC}"
    echo ""
fi

OWNER_ADDR=$(cast wallet address "$PRIVATE_KEY" 2>/dev/null)
BALANCE=$(cast balance "$OWNER_ADDR" --rpc-url "$RPC_URL" 2>/dev/null || echo "0")
BALANCE_DISPLAY=$(cast from-wei "$BALANCE" 2>/dev/null || echo "?")
echo -e "  Owner:       ${CYAN}$OWNER_ADDR${NC}"
echo -e "  Balance:     ${CYAN}$BALANCE_DISPLAY ADI${NC}"
echo -e "  RPC:         ${CYAN}$RPC_URL${NC}"
echo -e "  Coordinator: ${CYAN}${COORDINATOR:-will deploy on first run}${NC}"
echo -e "  Explorer:    ${CYAN}${ADI_EXPLORER:-https://explorer.ab.testnet.adifoundation.ai/}${NC}"
echo ""

# Check dashboard deps
if [[ ! -d "$ROOT_DIR/dashboard/node_modules" ]]; then
    echo -e "${YELLOW}Installing dashboard dependencies...${NC}"
    cd "$ROOT_DIR/dashboard" && npm install
fi

# Export for the Rust process
export TESTNET_PRIVATE_KEY="$PRIVATE_KEY"

cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    kill $DASHBOARD_PID $FRONTEND_PID 2>/dev/null || true
    wait $DASHBOARD_PID $FRONTEND_PID 2>/dev/null || true
    echo -e "${GREEN}Done.${NC}"
}
trap cleanup EXIT

# Start dashboard API with ADI testnet config
echo -e "${CYAN}Starting dashboard API on port 3001...${NC}"
DASHBOARD_CMD="$HELIX dashboard --port 3001 --host 0.0.0.0 --rpc-url $RPC_URL"
if [[ -n "$COORDINATOR" ]]; then
    DASHBOARD_CMD="$DASHBOARD_CMD --coordinator $COORDINATOR"
fi
cd "$ROOT_DIR"
$DASHBOARD_CMD &
DASHBOARD_PID=$!
sleep 1

# Start Next.js frontend
echo -e "${CYAN}Starting Next.js frontend on port 3000...${NC}"
cd "$ROOT_DIR/dashboard"
npm run dev &
FRONTEND_PID=$!
sleep 2

echo ""
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}  HELIX Demo Ready (ADI Testnet)!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "  Frontend:  ${CYAN}http://localhost:3000${NC}"
echo -e "  API:       ${CYAN}http://localhost:3001${NC}"
echo -e "  Chain:     ${CYAN}ADI Testnet (chain ID 99999)${NC}"
echo ""
echo -e "  ${YELLOW}Two-laptop mode:${NC}"
echo -e "    On worker laptop: ${CYAN}./scripts/demo-adi.sh workers --public-addr <WORKER-IP> --api-url http://<THIS-IP>:3001${NC}"
echo ""
echo -e "  ${YELLOW}Single-laptop mode:${NC}"
echo -e "    In another terminal: ${CYAN}./scripts/demo-adi.sh workers${NC}"
echo -e "    Or just click '${CYAN}Start Training (Local MPC)${NC}' in the web app (no workers needed)."
echo ""
echo -e "  Press ${YELLOW}Ctrl+C${NC} to stop."
echo ""

wait
