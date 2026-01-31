#!/bin/bash
#
# HELIX Demo Recovery Script
# ===========================
# Provides automatic recovery and reliability features for demos
#
# Features:
# - Health checks for all services
# - Automatic restart of failed components
# - Checkpoint/resume for training state
# - Graceful degradation
# - Fallback modes
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

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELIX_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
MAX_RETRIES=${MAX_RETRIES:-3}
RETRY_DELAY=${RETRY_DELAY:-1000}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"$HELIX_ROOT/.helix/checkpoints"}
LOG_FILE=${LOG_FILE:-"$HELIX_ROOT/.helix/recovery.log"}
HEALTH_CHECK_INTERVAL=${HEALTH_CHECK_INTERVAL:-5}

# Mode: "check", "recover", "checkpoint", "resume", "monitor"
MODE=${1:-"check"}

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$CHECKPOINT_DIR"

log() {
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo -e "${CYAN}[$timestamp]${NC} $1"
    echo "[$timestamp] $1" >> "$LOG_FILE"
}

success() {
    echo -e "  ${GREEN}✓${NC} $1"
}

error() {
    echo -e "  ${RED}✗${NC} $1"
}

warning() {
    echo -e "  ${YELLOW}!${NC} $1"
}

#######################################
# Health Checks
#######################################
check_anvil() {
    if cast chain-id --rpc-url "${RPC_URL:-http://localhost:8545}" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

check_contracts() {
    local coordinator="${COORDINATOR_ADDRESS:-}"
    if [ -z "$coordinator" ]; then
        return 1
    fi

    if cast call "$coordinator" "nextModelId()" --rpc-url "${RPC_URL:-http://localhost:8545}" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

check_dashboard() {
    local port="${DASHBOARD_PORT:-3000}"
    if curl -s "http://localhost:$port" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

check_prover() {
    if [ -f "$HELIX_ROOT/target/release/helix-node" ]; then
        return 0
    else
        return 1
    fi
}

run_all_checks() {
    local all_ok=true

    echo -e "${BOLD}Running health checks...${NC}"
    echo ""

    echo -n "  Anvil (blockchain)... "
    if check_anvil; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        all_ok=false
    fi

    echo -n "  Smart contracts... "
    if check_contracts; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${YELLOW}NOT DEPLOYED${NC}"
    fi

    echo -n "  Dashboard... "
    if check_dashboard; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${YELLOW}NOT RUNNING${NC}"
    fi

    echo -n "  Prover binary... "
    if check_prover; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}NOT BUILT${NC}"
        all_ok=false
    fi

    echo ""

    if [ "$all_ok" = true ]; then
        success "All critical checks passed"
        return 0
    else
        error "Some checks failed"
        return 1
    fi
}

#######################################
# Recovery Actions
#######################################
restart_anvil() {
    log "Restarting Anvil..."

    # Kill existing Anvil
    pkill -f "anvil" 2>/dev/null || true
    sleep 1

    # Start new instance
    anvil --port "${ANVIL_PORT:-8545}" --silent > /tmp/anvil_recovery.log 2>&1 &
    ANVIL_PID=$!

    sleep 2

    if kill -0 $ANVIL_PID 2>/dev/null; then
        success "Anvil restarted (PID: $ANVIL_PID)"
        echo "$ANVIL_PID" > "$CHECKPOINT_DIR/anvil.pid"
        return 0
    else
        error "Failed to restart Anvil"
        return 1
    fi
}

redeploy_contracts() {
    log "Redeploying contracts..."

    cd "$HELIX_ROOT/contracts"

    # Deploy with real verifier for production, mock for quick demo
    local use_mock=${USE_MOCK_VERIFIER:-true}

    if [ "$use_mock" = true ]; then
        local deploy_output=$(forge script script/Deploy.s.sol --sig "runWithMock()" \
            --rpc-url "${RPC_URL:-http://localhost:8545}" \
            --private-key "${PRIVATE_KEY:-0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80}" \
            --broadcast 2>&1)
    else
        local deploy_output=$(forge script script/Deploy.s.sol \
            --rpc-url "${RPC_URL:-http://localhost:8545}" \
            --private-key "${PRIVATE_KEY:-0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80}" \
            --broadcast 2>&1)
    fi

    # Extract addresses
    local coordinator=$(echo "$deploy_output" | grep -o "HelixCoordinator[^:]*: 0x[a-fA-F0-9]*" | head -1 | awk '{print $NF}')
    local verifier=$(echo "$deploy_output" | grep -o "Verifier[^:]*: 0x[a-fA-F0-9]*" | head -1 | awk '{print $NF}')

    if [ -n "$coordinator" ]; then
        export COORDINATOR_ADDRESS="$coordinator"
        echo "$coordinator" > "$CHECKPOINT_DIR/coordinator.addr"
        success "Contracts redeployed"
        success "Coordinator: $coordinator"
        return 0
    else
        error "Contract deployment failed"
        return 1
    fi
}

rebuild_prover() {
    log "Rebuilding prover..."

    cd "$HELIX_ROOT"
    cargo build --release -p helix-node -p helix-prover 2>&1 | tail -5

    if check_prover; then
        success "Prover rebuilt"
        return 0
    else
        error "Prover build failed"
        return 1
    fi
}

full_recovery() {
    log "Starting full recovery sequence..."
    echo ""

    local retry=0
    while [ $retry -lt $MAX_RETRIES ]; do
        retry=$((retry + 1))
        log "Recovery attempt $retry of $MAX_RETRIES..."

        # Check and fix Anvil
        if ! check_anvil; then
            restart_anvil || continue
        fi

        # Check and fix contracts
        if ! check_contracts; then
            redeploy_contracts || continue
        fi

        # Check prover
        if ! check_prover; then
            rebuild_prover || continue
        fi

        # All checks passed
        if run_all_checks; then
            success "Recovery successful"
            return 0
        fi

        warning "Recovery attempt $retry failed, retrying in ${RETRY_DELAY}ms..."
        sleep $(echo "scale=2; $RETRY_DELAY / 1000" | bc)
    done

    error "Recovery failed after $MAX_RETRIES attempts"
    return 1
}

#######################################
# Checkpoint/Resume
#######################################
save_checkpoint() {
    local checkpoint_name=${1:-"auto_$(date +%s)"}
    local checkpoint_path="$CHECKPOINT_DIR/$checkpoint_name"

    log "Saving checkpoint: $checkpoint_name"

    mkdir -p "$checkpoint_path"

    # Save training state
    if [ -f "$HELIX_ROOT/.helix/training_output/weights.bin" ]; then
        cp "$HELIX_ROOT/.helix/training_output/weights.bin" "$checkpoint_path/"
    fi

    # Save loss curve
    if [ -f "$HELIX_ROOT/.helix/training_output/loss_curve.csv" ]; then
        cp "$HELIX_ROOT/.helix/training_output/loss_curve.csv" "$checkpoint_path/"
    fi

    # Save contract addresses
    cat > "$checkpoint_path/state.json" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "coordinator_address": "${COORDINATOR_ADDRESS:-}",
    "verifier_address": "${VERIFIER_ADDRESS:-}",
    "model_id": ${MODEL_ID:-0},
    "current_round": ${CURRENT_ROUND:-0},
    "current_step": ${CURRENT_STEP:-0},
    "rpc_url": "${RPC_URL:-http://localhost:8545}",
    "anvil_pid": $(cat "$CHECKPOINT_DIR/anvil.pid" 2>/dev/null || echo "null")
}
EOF

    # Save environment
    env | grep -E "^(HELIX_|COORDINATOR_|VERIFIER_|RPC_|MODEL_)" > "$checkpoint_path/env.sh" 2>/dev/null || true

    success "Checkpoint saved: $checkpoint_path"
    echo "$checkpoint_path"
}

load_checkpoint() {
    local checkpoint_name=${1:-$(ls -t "$CHECKPOINT_DIR" | head -1)}

    if [ -z "$checkpoint_name" ]; then
        error "No checkpoint specified and none found"
        return 1
    fi

    local checkpoint_path="$CHECKPOINT_DIR/$checkpoint_name"

    if [ ! -d "$checkpoint_path" ]; then
        error "Checkpoint not found: $checkpoint_path"
        return 1
    fi

    log "Loading checkpoint: $checkpoint_name"

    # Load state
    if [ -f "$checkpoint_path/state.json" ]; then
        export COORDINATOR_ADDRESS=$(jq -r '.coordinator_address' "$checkpoint_path/state.json")
        export VERIFIER_ADDRESS=$(jq -r '.verifier_address' "$checkpoint_path/state.json")
        export MODEL_ID=$(jq -r '.model_id' "$checkpoint_path/state.json")
        export CURRENT_ROUND=$(jq -r '.current_round' "$checkpoint_path/state.json")
        export CURRENT_STEP=$(jq -r '.current_step' "$checkpoint_path/state.json")
        export RPC_URL=$(jq -r '.rpc_url' "$checkpoint_path/state.json")
    fi

    # Load environment
    if [ -f "$checkpoint_path/env.sh" ]; then
        source "$checkpoint_path/env.sh"
    fi

    # Restore training state
    if [ -f "$checkpoint_path/weights.bin" ]; then
        mkdir -p "$HELIX_ROOT/.helix/training_output"
        cp "$checkpoint_path/weights.bin" "$HELIX_ROOT/.helix/training_output/"
    fi

    success "Checkpoint loaded"
    echo ""
    echo "  Coordinator: $COORDINATOR_ADDRESS"
    echo "  Model ID: $MODEL_ID"
    echo "  Current Step: $CURRENT_STEP"
}

list_checkpoints() {
    echo -e "${BOLD}Available Checkpoints:${NC}"
    echo ""

    if [ ! -d "$CHECKPOINT_DIR" ] || [ -z "$(ls -A "$CHECKPOINT_DIR" 2>/dev/null)" ]; then
        echo "  (no checkpoints found)"
        return
    fi

    for cp in "$CHECKPOINT_DIR"/*; do
        if [ -d "$cp" ]; then
            local name=$(basename "$cp")
            local timestamp=$(jq -r '.timestamp // "unknown"' "$cp/state.json" 2>/dev/null || echo "unknown")
            local step=$(jq -r '.current_step // 0' "$cp/state.json" 2>/dev/null || echo "0")
            echo "  • $name"
            echo "    Time: $timestamp"
            echo "    Step: $step"
            echo ""
        fi
    done
}

#######################################
# Continuous Monitoring
#######################################
monitor_loop() {
    log "Starting continuous monitoring (interval: ${HEALTH_CHECK_INTERVAL}s)..."
    echo "Press Ctrl+C to stop"
    echo ""

    while true; do
        local status="OK"
        local issues=""

        if ! check_anvil; then
            status="DEGRADED"
            issues="$issues Anvil"
            restart_anvil &
        fi

        if ! check_contracts; then
            if [ -z "$issues" ]; then
                issues="Contracts"
            else
                issues="$issues, Contracts"
            fi
        fi

        local timestamp=$(date +"%H:%M:%S")

        if [ "$status" = "OK" ]; then
            echo -e "\r${CYAN}[$timestamp]${NC} Status: ${GREEN}$status${NC}     "
        else
            echo -e "\r${CYAN}[$timestamp]${NC} Status: ${YELLOW}$status${NC} (Issues: $issues)"
        fi

        sleep $HEALTH_CHECK_INTERVAL
    done
}

#######################################
# Graceful Shutdown
#######################################
cleanup() {
    log "Initiating graceful shutdown..."

    # Save checkpoint before exit
    save_checkpoint "shutdown_$(date +%s)" > /dev/null 2>&1

    # Stop processes
    if [ -f "$CHECKPOINT_DIR/anvil.pid" ]; then
        local anvil_pid=$(cat "$CHECKPOINT_DIR/anvil.pid")
        if kill -0 $anvil_pid 2>/dev/null; then
            kill $anvil_pid
            log "Anvil stopped"
        fi
    fi

    log "Shutdown complete"
}

trap cleanup EXIT

#######################################
# Main
#######################################
case "$MODE" in
    "check")
        run_all_checks
        ;;
    "recover")
        full_recovery
        ;;
    "checkpoint")
        save_checkpoint "$2"
        ;;
    "resume")
        load_checkpoint "$2"
        ;;
    "list")
        list_checkpoints
        ;;
    "monitor")
        monitor_loop
        ;;
    "restart-anvil")
        restart_anvil
        ;;
    "redeploy")
        redeploy_contracts
        ;;
    "rebuild")
        rebuild_prover
        ;;
    *)
        echo "HELIX Demo Recovery Tool"
        echo ""
        echo "Usage: $0 <mode> [options]"
        echo ""
        echo "Modes:"
        echo "  check         Run health checks"
        echo "  recover       Attempt automatic recovery"
        echo "  checkpoint    Save current state checkpoint"
        echo "  resume        Resume from checkpoint"
        echo "  list          List available checkpoints"
        echo "  monitor       Continuous monitoring with auto-recovery"
        echo "  restart-anvil Restart Anvil blockchain"
        echo "  redeploy      Redeploy smart contracts"
        echo "  rebuild       Rebuild prover binary"
        echo ""
        echo "Environment variables:"
        echo "  MAX_RETRIES       Maximum recovery attempts (default: 3)"
        echo "  RETRY_DELAY       Delay between retries in ms (default: 1000)"
        echo "  CHECKPOINT_DIR    Checkpoint storage directory"
        echo ""
        exit 1
        ;;
esac
