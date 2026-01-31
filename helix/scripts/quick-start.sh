#!/bin/bash
#
# HELIX Quick Start Script
# =========================
# One-command setup for HELIX - gets you running in seconds
#
# Features:
# - Automatic dependency checking
# - Pre-warming for instant demo startup
# - Support for mock or real verifier
# - Integrated with recovery system
# - Demo mode for immediate demonstration
#
# Usage:
#   ./quick-start.sh           # Standard setup
#   ./quick-start.sh demo      # Setup + run demo
#   ./quick-start.sh minimal   # Minimal setup (contracts only)
#   ./quick-start.sh full      # Full setup with real verifier
#   ./quick-start.sh prewarm   # Pre-warm only (for fast demos later)
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
DIM='\033[2m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELIX_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEMO_DIR="$SCRIPT_DIR/demo"

# Configuration
MODE=${1:-"standard"}
USE_REAL_VERIFIER=${USE_REAL_VERIFIER:-false}
RUN_DEMO=${RUN_DEMO:-false}
SKIP_BUILD=${SKIP_BUILD:-false}
VERBOSE=${VERBOSE:-false}
DASHBOARD_PORT=${DASHBOARD_PORT:-3000}

# State tracking
ANVIL_PID=""
DASHBOARD_PID=""
STATE_FILE="$HELIX_ROOT/.helix/quickstart_state.json"

# Ensure .helix directory exists
mkdir -p "$HELIX_ROOT/.helix"

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

info() {
    echo -e "  ${DIM}→${NC} $1"
}

spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "      \b\b\b\b\b\b"
}

# Banner
show_banner() {
    echo ""
    echo -e "${BOLD}${BLUE}"
    cat << 'EOF'
   ██╗  ██╗███████╗██╗     ██╗██╗  ██╗
   ██║  ██║██╔════╝██║     ██║╚██╗██╔╝
   ███████║█████╗  ██║     ██║ ╚███╔╝
   ██╔══██║██╔══╝  ██║     ██║ ██╔██╗
   ██║  ██║███████╗███████╗██║██╔╝ ██╗
   ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚═╝  ╚═╝
EOF
    echo -e "${NC}"
    echo -e "   ${BOLD}Trustless AI Training Protocol${NC}"
    echo -e "   ${DIM}Quick Start - One Command Setup${NC}"
    echo ""
}

#######################################
# Dependency Checking
#######################################
check_dependencies() {
    log "Checking dependencies..."

    local missing=()
    local optional_missing=()

    # Required dependencies
    for cmd in cargo forge anvil cast; do
        if command -v $cmd &> /dev/null; then
            success "$cmd $(command -v $cmd | head -1)"
        else
            missing+=($cmd)
            warn "$cmd not found"
        fi
    done

    # Optional dependencies
    if command -v node &> /dev/null; then
        success "node $(node --version)"
    else
        optional_missing+=("node")
        info "node not found (optional, for dashboard)"
    fi

    if command -v jq &> /dev/null; then
        success "jq found"
    else
        optional_missing+=("jq")
        info "jq not found (optional, for JSON parsing)"
    fi

    if [ ${#missing[@]} -ne 0 ]; then
        echo ""
        echo -e "${YELLOW}Missing required dependencies: ${missing[*]}${NC}"
        echo ""
        echo "Install instructions:"
        echo "  Rust/Cargo: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        echo "  Foundry:    curl -L https://foundry.paradigm.xyz | bash && foundryup"
        echo ""

        if [ "$MODE" != "check" ]; then
            read -p "Continue anyway? [y/N] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        else
            return 1
        fi
    fi

    return 0
}

#######################################
# Pre-warming
#######################################
run_prewarm() {
    log "Pre-warming for fast startup..."

    if [ -f "$DEMO_DIR/prewarm.sh" ]; then
        if [ "$VERBOSE" = true ]; then
            bash "$DEMO_DIR/prewarm.sh"
        else
            bash "$DEMO_DIR/prewarm.sh" > /dev/null 2>&1 &
            spinner $!
            success "Pre-warming complete"
        fi
    else
        # Fallback: do manual prewarm
        cd "$HELIX_ROOT"

        # Build Rust crates
        info "Building Rust crates..."
        if [ "$SKIP_BUILD" != true ]; then
            cargo build --release -p helix-core -p helix-client 2>&1 | tail -3 || true
        fi

        # Build contracts
        info "Building contracts..."
        cd "$HELIX_ROOT/contracts"
        forge build --silent 2>/dev/null || forge build

        success "Manual pre-warming complete"
    fi
}

#######################################
# Start Anvil
#######################################
start_anvil() {
    log "Starting local blockchain..."

    # Check if anvil is already running
    if curl -s http://localhost:8545 -X POST -H "Content-Type: application/json" \
        --data '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}' > /dev/null 2>&1; then
        warn "Anvil already running on port 8545"
        return 0
    fi

    # Start Anvil with optimized settings
    anvil \
        --port 8545 \
        --chain-id 31337 \
        --accounts 10 \
        --balance 10000 \
        --block-time 1 \
        --silent \
        > "$HELIX_ROOT/.helix/anvil.log" 2>&1 &

    ANVIL_PID=$!
    echo "$ANVIL_PID" > "$HELIX_ROOT/.helix/anvil.pid"

    # Wait for startup
    local retries=0
    while [ $retries -lt 30 ]; do
        if curl -s http://localhost:8545 -X POST -H "Content-Type: application/json" \
            --data '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}' > /dev/null 2>&1; then
            success "Anvil running (PID: $ANVIL_PID)"
            return 0
        fi
        retries=$((retries + 1))
        sleep 0.2
    done

    error "Failed to start Anvil"
}

#######################################
# Deploy Contracts
#######################################
deploy_contracts() {
    log "Deploying smart contracts..."

    cd "$HELIX_ROOT/contracts"

    export RPC_URL=${RPC_URL:-http://localhost:8545}
    export PRIVATE_KEY=${PRIVATE_KEY:-0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80}

    if [ "$USE_REAL_VERIFIER" = true ]; then
        # Deploy with real Halo2 verifier
        log "Deploying with real Halo2Verifier..."

        local deploy_output=$(forge script script/Deploy.s.sol \
            --rpc-url "$RPC_URL" \
            --private-key "$PRIVATE_KEY" \
            --broadcast 2>&1)

        VERIFIER_ADDRESS=$(echo "$deploy_output" | grep -o "Halo2Verifier[^:]*: 0x[a-fA-F0-9]*" | head -1 | awk '{print $NF}')
        COORDINATOR_ADDRESS=$(echo "$deploy_output" | grep -o "HelixCoordinatorV2[^:]*: 0x[a-fA-F0-9]*" | head -1 | awk '{print $NF}')

        if [ -n "$COORDINATOR_ADDRESS" ]; then
            success "Real verifier deployed"
        else
            warn "Real verifier deployment failed, falling back to mock"
            USE_REAL_VERIFIER=false
        fi
    fi

    if [ "$USE_REAL_VERIFIER" != true ]; then
        # Deploy with mock verifier (faster for development)

        # Deploy MockVerifier
        VERIFIER_ADDRESS=$(forge create src/mocks/MockVerifier.sol:MockVerifier \
            --rpc-url "$RPC_URL" \
            --private-key "$PRIVATE_KEY" 2>&1 | grep "Deployed to:" | awk '{print $3}')

        if [ -z "$VERIFIER_ADDRESS" ]; then
            # Try alternative deployment
            local output=$(forge script script/Deploy.s.sol --sig "runWithMock()" \
                --rpc-url "$RPC_URL" \
                --private-key "$PRIVATE_KEY" \
                --broadcast 2>&1)

            VERIFIER_ADDRESS=$(echo "$output" | grep -o "MockVerifier[^:]*: 0x[a-fA-F0-9]*" | head -1 | awk '{print $NF}')
            COORDINATOR_ADDRESS=$(echo "$output" | grep -o "HelixCoordinator[^:]*: 0x[a-fA-F0-9]*" | head -1 | awk '{print $NF}')
        else
            success "MockVerifier: $VERIFIER_ADDRESS"

            # Deploy HelixCoordinator (try V2 first, fall back to V1)
            COORDINATOR_ADDRESS=$(forge create src/core/HelixCoordinatorV2.sol:HelixCoordinatorV2 \
                --rpc-url "$RPC_URL" \
                --private-key "$PRIVATE_KEY" \
                --constructor-args "$VERIFIER_ADDRESS" 2>&1 | grep "Deployed to:" | awk '{print $3}')

            if [ -z "$COORDINATOR_ADDRESS" ]; then
                COORDINATOR_ADDRESS=$(forge create src/core/HelixCoordinator.sol:HelixCoordinator \
                    --rpc-url "$RPC_URL" \
                    --private-key "$PRIVATE_KEY" \
                    --constructor-args "$VERIFIER_ADDRESS" 2>&1 | grep "Deployed to:" | awk '{print $3}')
            fi
        fi
    fi

    if [ -z "$COORDINATOR_ADDRESS" ]; then
        error "Contract deployment failed"
    fi

    success "HelixCoordinator: $COORDINATOR_ADDRESS"

    # Export for use
    export COORDINATOR_ADDRESS
    export VERIFIER_ADDRESS

    # Save to state file
    cat > "$STATE_FILE" << EOF
{
    "coordinator_address": "$COORDINATOR_ADDRESS",
    "verifier_address": "$VERIFIER_ADDRESS",
    "rpc_url": "$RPC_URL",
    "anvil_pid": $ANVIL_PID,
    "use_real_verifier": $USE_REAL_VERIFIER,
    "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
}

#######################################
# Register Demo Model
#######################################
register_model() {
    log "Registering demo model..."

    cd "$HELIX_ROOT/contracts"

    # Register a demo model
    cast send "$COORDINATOR_ADDRESS" \
        "registerModel(string,uint256)" \
        "QmHelixDemo" 0 \
        --rpc-url "$RPC_URL" \
        --private-key "$PRIVATE_KEY" \
        --quiet 2>/dev/null || true

    success "Demo model registered (ID: 0)"
}

#######################################
# Start Dashboard
#######################################
start_dashboard() {
    if [ ! -d "$HELIX_ROOT/dashboard" ] || [ ! -f "$HELIX_ROOT/dashboard/package.json" ]; then
        info "Dashboard not found, skipping"
        return 0
    fi

    log "Starting dashboard..."

    cd "$HELIX_ROOT/dashboard"

    # Check if already running
    if curl -s "http://localhost:$DASHBOARD_PORT" > /dev/null 2>&1; then
        warn "Dashboard already running on port $DASHBOARD_PORT"
        return 0
    fi

    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        npm install --silent 2>/dev/null || npm install
    fi

    # Start in background
    COORDINATOR_ADDRESS="$COORDINATOR_ADDRESS" \
    VERIFIER_ADDRESS="$VERIFIER_ADDRESS" \
    RPC_URL="$RPC_URL" \
    npm run dev > "$HELIX_ROOT/.helix/dashboard.log" 2>&1 &

    DASHBOARD_PID=$!
    echo "$DASHBOARD_PID" > "$HELIX_ROOT/.helix/dashboard.pid"

    # Wait for startup
    local retries=0
    while [ $retries -lt 30 ]; do
        if curl -s "http://localhost:$DASHBOARD_PORT" > /dev/null 2>&1; then
            success "Dashboard running: http://localhost:$DASHBOARD_PORT"
            return 0
        fi
        retries=$((retries + 1))
        sleep 0.5
    done

    warn "Dashboard may take a moment to start"
}

#######################################
# Run Demo
#######################################
run_demo() {
    log "Running 90-second demo..."
    echo ""

    # Export variables for demo
    export COORDINATOR_ADDRESS
    export VERIFIER_ADDRESS
    export RPC_URL
    export PRIVATE_KEY
    export USE_REAL_PROOFS=${USE_REAL_VERIFIER}
    export USE_REAL_VERIFIER

    if [ -f "$SCRIPT_DIR/demo-90s.sh" ]; then
        bash "$SCRIPT_DIR/demo-90s.sh"
    elif [ -f "$DEMO_DIR/demo-90s.sh" ]; then
        bash "$DEMO_DIR/demo-90s.sh"
    else
        warn "Demo script not found"
    fi
}

#######################################
# Display Summary
#######################################
show_summary() {
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║${NC}                   ${BOLD}HELIX is Ready!${NC}                        ${GREEN}║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""

    echo -e "${BOLD}Services Running:${NC}"
    echo "  ├── Anvil (blockchain): http://localhost:8545"
    if [ -n "$DASHBOARD_PID" ] && kill -0 "$DASHBOARD_PID" 2>/dev/null; then
        echo "  ├── Dashboard:          http://localhost:$DASHBOARD_PORT"
    fi
    echo "  └── Contracts deployed"
    echo ""

    echo -e "${BOLD}Contract Addresses:${NC}"
    echo "  ├── Coordinator: $COORDINATOR_ADDRESS"
    echo "  └── Verifier:    $VERIFIER_ADDRESS"
    if [ "$USE_REAL_VERIFIER" = true ]; then
        echo "      (Real Halo2 Verifier)"
    else
        echo "      (Mock Verifier - for development)"
    fi
    echo ""

    echo -e "${BOLD}Environment:${NC}"
    echo "  export COORDINATOR_ADDRESS=$COORDINATOR_ADDRESS"
    echo "  export VERIFIER_ADDRESS=$VERIFIER_ADDRESS"
    echo "  export RPC_URL=http://localhost:8545"
    echo ""

    echo -e "${BOLD}Next Steps:${NC}"
    echo "  # Run the 90-second demo"
    echo "  ./scripts/demo-90s.sh"
    echo ""
    echo "  # Run multi-node demo"
    echo "  ./scripts/multi-node-demo.sh"
    echo ""
    echo "  # Check system health"
    echo "  ./scripts/demo/recovery.sh check"
    echo ""
    echo "  # Record a demo video"
    echo "  ./scripts/demo/record.sh record"
    echo ""

    # Save environment file
    cat > "$HELIX_ROOT/.env.local" << EOF
# HELIX Environment (generated $(date))
export COORDINATOR_ADDRESS=$COORDINATOR_ADDRESS
export VERIFIER_ADDRESS=$VERIFIER_ADDRESS
export RPC_URL=http://localhost:8545
export PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
export ANVIL_PID=$ANVIL_PID
export DASHBOARD_PID=${DASHBOARD_PID:-""}
export USE_REAL_VERIFIER=$USE_REAL_VERIFIER
EOF

    echo -e "${CYAN}Environment saved to .env.local${NC}"
    echo -e "Run: ${BOLD}source .env.local${NC} to load in new terminals"
    echo ""
}

#######################################
# Cleanup
#######################################
cleanup() {
    log "Shutting down..."

    if [ -n "$ANVIL_PID" ] && kill -0 "$ANVIL_PID" 2>/dev/null; then
        kill "$ANVIL_PID" 2>/dev/null || true
        info "Anvil stopped"
    fi

    if [ -n "$DASHBOARD_PID" ] && kill -0 "$DASHBOARD_PID" 2>/dev/null; then
        kill "$DASHBOARD_PID" 2>/dev/null || true
        info "Dashboard stopped"
    fi

    log "Shutdown complete"
}

#######################################
# Main
#######################################

# Parse mode
case "$MODE" in
    "demo")
        RUN_DEMO=true
        ;;
    "full")
        USE_REAL_VERIFIER=true
        ;;
    "minimal")
        SKIP_BUILD=true
        ;;
    "prewarm")
        show_banner
        run_prewarm
        success "Pre-warming complete. Run './scripts/quick-start.sh' for instant startup."
        exit 0
        ;;
    "check")
        show_banner
        check_dependencies
        exit $?
        ;;
    "help"|"-h"|"--help")
        cat << EOF
HELIX Quick Start

Usage: $0 [mode]

Modes:
  standard    Standard setup with mock verifier (default)
  demo        Setup + run 90-second demo immediately
  full        Full setup with real Halo2 verifier
  minimal     Minimal setup (skip builds, contracts only)
  prewarm     Pre-warm caches for fast demos later
  check       Check dependencies only

Environment Variables:
  USE_REAL_VERIFIER    Use real Halo2 verifier (true/false)
  SKIP_BUILD           Skip Rust/contract builds (true/false)
  VERBOSE              Show detailed output (true/false)
  DASHBOARD_PORT       Dashboard port (default: 3000)

Examples:
  $0                          # Standard setup
  $0 demo                     # Setup and run demo
  USE_REAL_VERIFIER=true $0   # Setup with real verifier
  SKIP_BUILD=true $0          # Quick restart (no builds)

EOF
        exit 0
        ;;
    "standard"|"")
        # Default mode
        ;;
    *)
        warn "Unknown mode: $MODE"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac

# Trap cleanup
trap cleanup EXIT INT TERM

# Run
show_banner
check_dependencies

# Pre-warm unless skipping builds
if [ "$SKIP_BUILD" != true ]; then
    run_prewarm
fi

# Start services
start_anvil
deploy_contracts
register_model

# Start dashboard for non-minimal modes
if [ "$MODE" != "minimal" ]; then
    start_dashboard
fi

# Show summary
show_summary

# Run demo if requested
if [ "$RUN_DEMO" = true ]; then
    run_demo
else
    echo -e "${BOLD}Press Ctrl+C to stop all services...${NC}"
    echo ""

    # Wait for Anvil
    wait $ANVIL_PID 2>/dev/null || true
fi
