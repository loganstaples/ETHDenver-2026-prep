#!/usr/bin/env bash
# ==============================================================================
# HELIX ETHDenver Demo Runner
# ==============================================================================
# Builds and runs the complete HELIX end-to-end demo.
#
# This script:
#   1. Checks all prerequisites (Rust, Foundry, Node.js)
#   2. Compiles the Solidity contracts
#   3. Builds the helix-demo binary
#   4. Runs the full E2E demo (Anvil + contracts + training + proofs + slashing)
#
# Usage:
#   ./scripts/run_demo.sh              # Full demo (20 steps/worker, 3 workers)
#   ./scripts/run_demo.sh --quick      # Quick demo (5 steps, offline)
#   ./scripts/run_demo.sh --bench      # Full demo + performance benchmark
#   ./scripts/run_demo.sh --offline    # Skip on-chain (no Anvil needed)
#   ./scripts/run_demo.sh --help       # Show all options
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
NC='\033[0m'

log() { echo -e "${BLUE}[helix]${NC} $1"; }
ok()  { echo -e "${GREEN}  [ok]${NC} $1"; }
err() { echo -e "${RED}  [!!]${NC} $1"; }

# ============================================================================
# Argument parsing
# ============================================================================
DEMO_ARGS=()
QUICK=false
BUILD_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            QUICK=true
            shift
            ;;
        --bench)
            DEMO_ARGS+=("--bench")
            shift
            ;;
        --offline)
            DEMO_ARGS+=("--offline")
            shift
            ;;
        --verbose|-v)
            DEMO_ARGS+=("--verbose")
            shift
            ;;
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --steps|-n)
            DEMO_ARGS+=("--steps" "$2")
            shift 2
            ;;
        --workers|-w)
            DEMO_ARGS+=("--workers" "$2")
            shift 2
            ;;
        --help|-h)
            cat << 'EOF'
HELIX ETHDenver Demo Runner

Usage: ./scripts/run_demo.sh [OPTIONS]

Options:
    --quick         Quick demo (5 steps, offline mode)
    --bench         Run performance benchmark after training
    --offline       Skip on-chain deployment (no Anvil/Foundry needed)
    --verbose, -v   Enable verbose logging
    --build-only    Only build, don't run
    --steps N       Training steps per worker (default: 20)
    --workers N     Number of worker threads (default: 3)
    --help, -h      Show this help

Examples:
    ./scripts/run_demo.sh                    # Full demo
    ./scripts/run_demo.sh --quick            # Quick test
    ./scripts/run_demo.sh --bench --steps 50 # Benchmark with 50 steps
    ./scripts/run_demo.sh --offline -v       # Offline with verbose logs
EOF
            exit 0
            ;;
        *)
            DEMO_ARGS+=("$1")
            shift
            ;;
    esac
done

if $QUICK; then
    DEMO_ARGS+=("--steps" "5" "--offline")
fi

# ============================================================================
# Prerequisites check
# ============================================================================
log "Checking prerequisites..."

check_cmd() {
    if command -v "$1" &> /dev/null; then
        ok "$1 found: $(command -v "$1")"
        return 0
    else
        err "$1 not found"
        return 1
    fi
}

MISSING=0
check_cmd "cargo" || MISSING=1
check_cmd "rustc" || MISSING=1

# Foundry is only needed for on-chain mode
if [[ ! " ${DEMO_ARGS[*]} " =~ " --offline " ]]; then
    check_cmd "forge" || MISSING=1
    check_cmd "anvil" || MISSING=1
fi

if [[ $MISSING -ne 0 ]]; then
    err "Missing prerequisites. Install:"
    echo "  Rust:    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo "  Foundry: curl -L https://foundry.paradigm.xyz | bash && foundryup"
    exit 1
fi

ok "All prerequisites met"

# ============================================================================
# Build contracts (if not offline)
# ============================================================================
if [[ ! " ${DEMO_ARGS[*]} " =~ " --offline " ]]; then
    log "Building Solidity contracts..."
    cd "$HELIX_ROOT/contracts"

    if [[ -f "out/HelixCoordinatorV2.sol/HelixCoordinatorV2.json" ]]; then
        ok "Contracts already compiled (skipping)"
    else
        forge build --quiet
        ok "Contracts compiled"
    fi
    cd "$HELIX_ROOT"
fi

# ============================================================================
# Build Rust binary
# ============================================================================
log "Building helix-demo binary (release)..."
cd "$HELIX_ROOT"

cargo build --release -p helix-demo 2>&1 | while IFS= read -r line; do
    # Show progress for Compiling lines
    if [[ "$line" == *"Compiling"* ]]; then
        echo -e "  ${CYAN}..${NC} $line"
    fi
done

DEMO_BIN="$HELIX_ROOT/target/release/helix-demo"
if [[ ! -f "$DEMO_BIN" ]]; then
    err "Build failed: $DEMO_BIN not found"
    exit 1
fi
ok "helix-demo built: $DEMO_BIN"

if $BUILD_ONLY; then
    log "Build complete (--build-only specified)"
    exit 0
fi

# ============================================================================
# Run demo
# ============================================================================
echo ""
log "${BOLD}Starting HELIX E2E Demo${NC}"
echo "=============================================="
echo ""

exec "$DEMO_BIN" "${DEMO_ARGS[@]}"
