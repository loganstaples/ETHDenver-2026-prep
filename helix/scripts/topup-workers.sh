#!/usr/bin/env bash
# ============================================================================
# Top up HELIX worker balances on ADI Testnet
# ============================================================================
# Usage:
#   ./scripts/topup-workers.sh              # Default 0.5 ADI per worker
#   ./scripts/topup-workers.sh 1.0          # Custom amount per worker
# ============================================================================
set -euo pipefail

AMOUNT="${1:-0.5}"
RPC_URL="https://rpc.ab.testnet.adifoundation.ai/"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$ROOT_DIR/.env.adi"
OWNER_KEY="${TESTNET_PRIVATE_KEY}"
OWNER_ADDR=$(cast wallet address "$OWNER_KEY" 2>/dev/null)

WORKER_KEYS=(
  "0x8d684f8cf3b6a00d8a4deba9c06176132056facc7a66f7d2fa8cd31127b09806"
  "0x23944b2afce56b6332c0fc74a1d11ca3508ea7a9f4b821f7a710a5f95eb6f772"
  "0x67fe620ca54f907b42a2585753d2aa2a902e1933a3152d8e30255e6a9f1dacd4"
  "0x880a8a9b8424533bfdaa5c03dacad0ab8f32a69d6f233b9a8301bcb2741f9867"
  "0xb26bf4ed563907b04f6738e33e1c5fb9b61a5097ad63859f9d4537dae3cb6012"
  "0x9c1b1d65d0bb28cb85acc34215a21a94c74171952e33c3b5749d4abb186f34a4"
)

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${CYAN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   HELIX — Top Up Worker Balances          ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════╝${NC}"
echo ""

OWNER_BAL=$(cast balance "$OWNER_ADDR" --rpc-url "$RPC_URL" 2>/dev/null)
OWNER_BAL_ETH=$(cast from-wei "$OWNER_BAL" 2>/dev/null)
echo -e "Owner: ${CYAN}$OWNER_ADDR${NC}"
echo -e "Owner balance: ${CYAN}$OWNER_BAL_ETH ADI${NC}"

NUM_WORKERS=${#WORKER_KEYS[@]}
TOTAL_NEEDED=$(echo "$AMOUNT * $NUM_WORKERS" | bc)
echo -e "Sending: ${YELLOW}$AMOUNT ADI${NC} x $NUM_WORKERS workers = ${YELLOW}$TOTAL_NEEDED ADI${NC} total"
echo ""

# Check if owner has enough
AMOUNT_WEI=$(cast to-wei "$AMOUNT" 2>/dev/null)
TOTAL_WEI=$(echo "$AMOUNT_WEI * $NUM_WORKERS" | bc)
if [ "$(echo "$OWNER_BAL < $TOTAL_WEI" | bc)" -eq 1 ]; then
  echo -e "${RED}Error: Owner balance ($OWNER_BAL_ETH ADI) is less than needed ($TOTAL_NEEDED ADI)${NC}"
  exit 1
fi

for i in "${!WORKER_KEYS[@]}"; do
  WORKER_ADDR=$(cast wallet address "${WORKER_KEYS[$i]}" 2>/dev/null)
  BEFORE=$(cast balance "$WORKER_ADDR" --rpc-url "$RPC_URL" 2>/dev/null)
  BEFORE_ETH=$(cast from-wei "$BEFORE" 2>/dev/null)

  # Skip if worker already has enough
  if [ "$(echo "$BEFORE >= $AMOUNT_WEI" | bc)" -eq 1 ]; then
    echo -e "  ${GREEN}✓${NC} [worker-$i] $WORKER_ADDR already has ${GREEN}$BEFORE_ETH ADI${NC} — skipped"
    continue
  fi

  TX=$(cast send "$WORKER_ADDR" \
    --value "${AMOUNT}ether" \
    --rpc-url "$RPC_URL" \
    --private-key "$OWNER_KEY" \
    --legacy 2>&1)

  AFTER=$(cast balance "$WORKER_ADDR" --rpc-url "$RPC_URL" 2>/dev/null)
  AFTER_ETH=$(cast from-wei "$AFTER" 2>/dev/null)
  echo -e "  ${GREEN}✓${NC} [worker-$i] $WORKER_ADDR: ${RED}$BEFORE_ETH${NC} → ${GREEN}$AFTER_ETH ADI${NC}"
done

echo ""
NEW_OWNER_BAL=$(cast balance "$OWNER_ADDR" --rpc-url "$RPC_URL" 2>/dev/null)
echo -e "Owner remaining: ${CYAN}$(cast from-wei "$NEW_OWNER_BAL" 2>/dev/null) ADI${NC}"
echo -e "${GREEN}Done.${NC}"
