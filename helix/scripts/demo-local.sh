#!/usr/bin/env bash
# ============================================================================
# HELIX Demo — Single Laptop (Local Anvil)
# ============================================================================
# Runs everything on one machine:
#   - Anvil (local Ethereum devnet, auto-started by orchestrator)
#   - Dashboard API (Rust, port 3001)
#   - Next.js frontend (port 3000)
#   - Training runs in-process (Local MPC, no spawn-workers needed)
#
# Usage:
#   ./scripts/demo-local.sh
#
# Then open http://localhost:3000 in your browser.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HELIX="$ROOT_DIR/target/debug/helix"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   HELIX Demo — Single Laptop (Local)      ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════╝${NC}"
echo ""

# Build if needed
if [[ ! -f "$HELIX" ]]; then
    echo -e "${YELLOW}Building helix binary...${NC}"
    cd "$ROOT_DIR" && cargo build -p helix-client 2>&1 | tail -3
fi

# Check dashboard deps
if [[ ! -d "$ROOT_DIR/dashboard/node_modules" ]]; then
    echo -e "${YELLOW}Installing dashboard dependencies...${NC}"
    cd "$ROOT_DIR/dashboard" && npm install
fi

# Cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    kill $DASHBOARD_PID $FRONTEND_PID 2>/dev/null || true
    wait $DASHBOARD_PID $FRONTEND_PID 2>/dev/null || true
    echo -e "${GREEN}Done.${NC}"
}
trap cleanup EXIT

# Start dashboard API (Anvil auto-starts during training)
echo -e "${CYAN}Starting dashboard API on port 3001...${NC}"
cd "$ROOT_DIR"
"$HELIX" dashboard --port 3001 --host 0.0.0.0 --cors &
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
echo -e "${GREEN}  HELIX Demo Ready!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "  Frontend:  ${CYAN}http://localhost:3000${NC}"
echo -e "  API:       ${CYAN}http://localhost:3001${NC}"
echo -e "  Mode:      ${CYAN}Local MPC (in-process, Anvil on-chain)${NC}"
echo ""
echo -e "  Click ${YELLOW}'Start Training (Local MPC)'${NC} in the web app."
echo -e "  Press ${YELLOW}Ctrl+C${NC} to stop."
echo ""

wait
