#!/usr/bin/env bash
# ==============================================================================
# HELIX MPC Demo — Full Pipeline
# ==============================================================================
# One-command launch for the HELIX ETHDenver demo.
#
# Starts:
#   1. Anvil (local chain on port 8545)
#   2. Backend API (Axum on port 3001)
#   3. MPC Workers (auto-register with API)
#   4. Next.js Dashboard (port 3000)
#
# Then open http://localhost:3000/train and click "Start Training".
#
# Usage:
#   ./scripts/mpc-demo.sh                    # Full demo (dashboard + API + 6 workers)
#   ./scripts/mpc-demo.sh --workers 10       # Use 10 workers
#   ./scripts/mpc-demo.sh --cli-only         # CLI training only (no web UI)
#   ./scripts/mpc-demo.sh --skip-build       # Skip Rust/JS build (fast restart)
#   ./scripts/mpc-demo.sh --help
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELIX_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

log()     { echo -e "${CYAN}[helix]${NC} $1"; }
ok()      { echo -e "  ${GREEN}✓${NC} $1"; }
err()     { echo -e "  ${RED}✗${NC} $1"; }
warn()    { echo -e "  ${YELLOW}!${NC} $1"; }

# PIDs to clean up
ANVIL_PID=""
API_PID=""
WORKERS_PID=""
DASH_PID=""

cleanup() {
    echo ""
    log "Shutting down..."
    [ -n "$DASH_PID" ]    && kill "$DASH_PID"    2>/dev/null && echo "  Dashboard stopped"
    [ -n "$WORKERS_PID" ] && kill "$WORKERS_PID" 2>/dev/null && echo "  Workers stopped"
    [ -n "$API_PID" ]     && kill "$API_PID"     2>/dev/null && echo "  API stopped"
    [ -n "$ANVIL_PID" ]   && kill "$ANVIL_PID"   2>/dev/null && echo "  Anvil stopped"
    log "Done."
}
trap cleanup EXIT INT TERM

# ============================================================================
# Arguments
# ============================================================================
CLI_ONLY=false
SKIP_BUILD=false
NUM_WORKERS=6

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cli-only)   CLI_ONLY=true;   shift ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        --workers)    NUM_WORKERS="$2"; shift 2 ;;
        --help|-h)
            cat <<'EOF'
HELIX MPC Demo

Usage: ./scripts/mpc-demo.sh [OPTIONS]

Options:
    --workers N     Number of MPC workers to launch (default: 6)
    --cli-only      Run CLI training only (no web dashboard)
    --skip-build    Skip Rust and JS builds (fast restart)
    --help, -h      Show this help

Full Demo (open browser):
    ./scripts/mpc-demo.sh
    # Then open http://localhost:3000/train

Full Demo with more workers:
    ./scripts/mpc-demo.sh --workers 10

CLI-Only Demo (no browser):
    ./scripts/mpc-demo.sh --cli-only
    # Runs MNIST 784→128→10 with 6 MPC workers, 500 steps
EOF
            exit 0
            ;;
        *) err "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Banner
# ============================================================================
echo ""
echo -e "${BOLD}${BLUE}"
cat << 'BANNER'
   ██╗  ██╗███████╗██╗     ██╗██╗  ██╗
   ██║  ██║██╔════╝██║     ██║╚██╗██╔╝
   ███████║█████╗  ██║     ██║ ╚███╔╝
   ██╔══██║██╔══╝  ██║     ██║ ██╔██╗
   ██║  ██║███████╗███████╗██║██╔╝ ██╗
   ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚═╝  ╚═╝
BANNER
echo -e "${NC}"
echo -e "   ${BOLD}MPC-Primary Training Demo${NC}"
echo -e "   ${DIM}Information-theoretic security • Zero weight leakage${NC}"
echo ""

# ============================================================================
# Prerequisites
# ============================================================================
log "Checking prerequisites..."

MISSING=0
for cmd in cargo rustc anvil forge; do
    if command -v "$cmd" &>/dev/null; then
        ok "$cmd"
    else
        err "$cmd not found"
        MISSING=1
    fi
done

if ! $CLI_ONLY; then
    if command -v node &>/dev/null; then
        ok "node $(node --version)"
    else
        warn "node not found (needed for dashboard)"
        warn "Install Node.js 18+ or run with --cli-only"
        CLI_ONLY=true
    fi
fi

if [[ $MISSING -ne 0 ]]; then
    err "Missing prerequisites. Install:"
    echo "  Rust:    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo "  Foundry: curl -L https://foundry.paradigm.xyz | bash && foundryup"
    exit 1
fi

# ============================================================================
# Build
# ============================================================================
if ! $SKIP_BUILD; then
    log "Building contracts..."
    cd "$HELIX_ROOT/contracts"
    forge build --quiet 2>/dev/null || forge build
    ok "Contracts compiled"

    log "Building helix-client (release, chain feature)..."
    cd "$HELIX_ROOT"
    cargo build -p helix-client --features chain --release 2>&1 | while IFS= read -r line; do
        [[ "$line" == *"Compiling"* ]] && echo -e "  ${DIM}$line${NC}"
    done
    ok "helix-client built"

    if ! $CLI_ONLY; then
        log "Installing dashboard dependencies..."
        cd "$HELIX_ROOT/dashboard"
        if [ ! -d "node_modules" ]; then
            npm install --silent 2>/dev/null || npm install
        fi
        ok "Dashboard ready"
    fi
else
    ok "Skipping builds (--skip-build)"
fi

cd "$HELIX_ROOT"

# Create .helix directory for logs
mkdir -p "$HELIX_ROOT/.helix"

# ============================================================================
# 1. Start Anvil
# ============================================================================
log "Starting Anvil (local chain)..."

if curl -s http://localhost:8545 -X POST -H "Content-Type: application/json" \
    --data '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}' &>/dev/null; then
    warn "Anvil already running on port 8545"
else
    anvil --accounts 20 --balance 10000 --silent &>/dev/null &
    ANVIL_PID=$!
    sleep 2
    if curl -s http://localhost:8545 -X POST -H "Content-Type: application/json" \
        --data '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}' &>/dev/null; then
        ok "Anvil running (PID: $ANVIL_PID)"
    else
        err "Anvil failed to start"
        exit 1
    fi
fi

# ============================================================================
# 2. Start Backend API (port 3001)
# ============================================================================
if ! $CLI_ONLY; then
    log "Starting backend API on port 3001..."
    "$HELIX_ROOT/target/release/helix" dashboard --port 3001 --host 0.0.0.0 --cors \
        > "$HELIX_ROOT/.helix/api.log" 2>&1 &
    API_PID=$!
    sleep 1

    if kill -0 "$API_PID" 2>/dev/null; then
        ok "API server running (PID: $API_PID)"
    else
        err "API server failed to start. Check .helix/api.log"
        exit 1
    fi

    # ========================================================================
    # 3. Spawn MPC Workers (register with API)
    # ========================================================================
    log "Launching $NUM_WORKERS MPC workers..."
    "$HELIX_ROOT/target/release/helix" spawn-workers \
        --count "$NUM_WORKERS" \
        --api-url http://localhost:3001 \
        > "$HELIX_ROOT/.helix/workers.log" 2>&1 &
    WORKERS_PID=$!
    sleep 2

    if kill -0 "$WORKERS_PID" 2>/dev/null; then
        ok "$NUM_WORKERS workers launched and registered (PID: $WORKERS_PID)"
    else
        err "Workers failed to start. Check .helix/workers.log"
        exit 1
    fi

    # ========================================================================
    # 4. Start Dashboard (port 3000)
    # ========================================================================
    log "Starting Next.js dashboard on port 3000..."
    cd "$HELIX_ROOT/dashboard"
    NEXT_PUBLIC_API_URL=http://localhost:3001 \
    NEXT_PUBLIC_WS_URL=ws://localhost:3001/ws \
    npm run dev > "$HELIX_ROOT/.helix/dashboard.log" 2>&1 &
    DASH_PID=$!
    cd "$HELIX_ROOT"

    # Wait for dashboard
    RETRIES=0
    while [ $RETRIES -lt 30 ]; do
        if curl -s http://localhost:3000 &>/dev/null; then
            ok "Dashboard running"
            break
        fi
        RETRIES=$((RETRIES + 1))
        sleep 1
    done

    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}  HELIX Demo Ready${NC}"
    echo ""
    echo -e "  ${BOLD}Dashboard:${NC}  http://localhost:3000/train"
    echo -e "  ${BOLD}API:${NC}        http://localhost:3001"
    echo -e "  ${BOLD}Chain:${NC}      http://localhost:8545"
    echo -e "  ${BOLD}Workers:${NC}    $NUM_WORKERS MPC workers registered"
    echo ""
    echo -e "  Open the dashboard and click ${BOLD}Start Training${NC}"
    echo -e "  Workers will be auto-assigned when training starts."
    echo ""
    echo -e "  ${DIM}Press Ctrl+C to stop everything${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # Keep running until Ctrl+C
    wait $API_PID 2>/dev/null || true
else
    # ========================================================================
    # CLI-Only mode: Run mpc-train directly
    # ========================================================================
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}  HELIX CLI Training Demo${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    exec "$HELIX_ROOT/target/release/helix" mpc-train \
        --architecture 784,128,10 \
        --steps 500 \
        --checkpoint-freq 50 \
        --mac-interval 1 \
        --train-size 1000 \
        --test-size 200 \
        --num-workers "$NUM_WORKERS" \
        --seed 42
fi
