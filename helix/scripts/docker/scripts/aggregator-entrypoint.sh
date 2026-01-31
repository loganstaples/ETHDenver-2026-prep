#!/bin/bash
# HELIX Aggregator Entrypoint Script
# ===================================
# Initializes and starts the HELIX aggregator with proper configuration.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() {
    echo -e "${CYAN}[AGGREGATOR]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[AGGREGATOR]${NC} $1"
}

error() {
    echo -e "${RED}[AGGREGATOR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[AGGREGATOR]${NC} $1"
}

# Wait for dependencies
wait_for_service() {
    local host="$1"
    local port="$2"
    local service="$3"
    local max_attempts="${4:-60}"
    local attempt=1

    log "Waiting for $service at $host:$port..."

    while ! nc -z "$host" "$port" 2>/dev/null; do
        if [ $attempt -ge $max_attempts ]; then
            error "Timeout waiting for $service after $max_attempts attempts"
            return 1
        fi
        sleep 1
        attempt=$((attempt + 1))
    done

    success "$service is available"
    return 0
}

# Validate environment
validate_environment() {
    log "Validating environment..."

    # Required variables
    if [ -z "$HELIX_ETH_RPC" ]; then
        warn "HELIX_ETH_RPC not set, using default: http://ethereum:8545"
        export HELIX_ETH_RPC="http://ethereum:8545"
    fi

    # Extract host and port from RPC URL for health check
    local rpc_host=$(echo "$HELIX_ETH_RPC" | sed -E 's|https?://([^:/]+).*|\1|')
    local rpc_port=$(echo "$HELIX_ETH_RPC" | sed -E 's|https?://[^:]+:([0-9]+).*|\1|')
    rpc_port=${rpc_port:-8545}

    # Wait for Ethereum RPC
    wait_for_service "$rpc_host" "$rpc_port" "Ethereum RPC" 120 || exit 1

    success "Environment validated"
}

# Initialize aggregator state
initialize_state() {
    log "Initializing aggregator state..."

    # Ensure directories exist
    mkdir -p "$HELIX_DATA_DIR" "$HELIX_LOG_DIR" "$HELIX_GRADIENTS_DIR" "$HELIX_PROOFS_DIR"

    # Generate node ID if not set
    if [ -z "$HELIX_NODE_ID" ]; then
        export HELIX_NODE_ID="aggregator-$(cat /proc/sys/kernel/random/uuid | cut -d'-' -f1)"
        log "Generated node ID: $HELIX_NODE_ID"
    fi

    success "State initialized"
}

# Health check for self
self_health_check() {
    log "Running self health check..."

    # Check disk space
    local disk_usage=$(df -h /app/data | awk 'NR==2 {gsub(/%/,""); print $5}')
    if [ "$disk_usage" -gt 90 ]; then
        warn "Disk usage is at ${disk_usage}%"
    fi

    # Check memory (if available)
    if [ -f /proc/meminfo ]; then
        local mem_available=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
        local mem_total=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
        local mem_percent=$((100 - (mem_available * 100 / mem_total)))
        if [ "$mem_percent" -gt 90 ]; then
            warn "Memory usage is at ${mem_percent}%"
        fi
    fi

    success "Health check passed"
}

# Cleanup handler
cleanup() {
    log "Shutting down aggregator..."
    # Add any cleanup logic here
    exit 0
}

trap cleanup SIGTERM SIGINT

# Main execution
main() {
    log "Starting HELIX Aggregator..."
    log "Node ID: ${HELIX_NODE_ID:-auto}"
    log "Listen: ${HELIX_LISTEN_ADDR}"
    log "API: ${HELIX_API_ADDR}"
    log "Metrics: ${HELIX_METRICS_ADDR}"
    log "Min Workers: ${HELIX_MIN_WORKERS}"

    validate_environment
    initialize_state
    self_health_check

    success "Aggregator initialization complete"
    log "Starting helix-aggregator process..."

    # Execute the aggregator binary
    exec /app/helix-aggregator \
        --config /app/config/aggregator.toml \
        --role aggregator \
        --listen "${HELIX_LISTEN_ADDR}" \
        --api "${HELIX_API_ADDR}" \
        --metrics "${HELIX_METRICS_ADDR}" \
        --eth-rpc "${HELIX_ETH_RPC}" \
        --min-workers "${HELIX_MIN_WORKERS}" \
        --collection-timeout "${HELIX_COLLECTION_TIMEOUT}" \
        "$@"
}

main "$@"
