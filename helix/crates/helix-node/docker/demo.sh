#!/bin/bash
# HELIX 3-Worker Demo Script
# Starts and manages a local 3-worker HELIX training network

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[HELIX]${NC} $1"
}

success() {
    echo -e "${GREEN}[HELIX]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[HELIX]${NC} $1"
}

error() {
    echo -e "${RED}[HELIX]${NC} $1"
}

# Check dependencies
check_deps() {
    log "Checking dependencies..."

    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    success "All dependencies found."
}

# Build the Docker image
build() {
    log "Building HELIX Docker image..."
    docker compose build
    success "Build complete."
}

# Start the demo network
start() {
    log "Starting HELIX 3-worker demo network..."

    # Start infrastructure first
    docker compose up -d ethereum
    log "Waiting for Ethereum node to be ready..."
    sleep 5

    # Deploy contracts (if needed)
    # log "Deploying contracts..."
    # docker compose exec ethereum cast ...

    # Start aggregator
    docker compose up -d aggregator
    log "Waiting for aggregator to be ready..."
    sleep 5

    # Start workers
    docker compose up -d worker1 worker2 worker3
    log "Waiting for workers to connect..."
    sleep 5

    # Start monitoring
    docker compose up -d prometheus grafana

    success "HELIX demo network is running!"
    echo ""
    echo "Services:"
    echo "  - Aggregator API: http://localhost:9001"
    echo "  - Ethereum RPC: http://localhost:8545"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Grafana: http://localhost:3000 (admin/admin)"
    echo ""
    echo "Use 'docker compose logs -f' to view logs"
}

# Stop the demo network
stop() {
    log "Stopping HELIX demo network..."
    docker compose down
    success "Demo network stopped."
}

# Clean up everything
clean() {
    log "Cleaning up HELIX demo resources..."
    docker compose down -v --rmi local
    success "Cleanup complete."
}

# Show status
status() {
    log "HELIX Demo Network Status:"
    docker compose ps
    echo ""

    log "Worker Health:"
    for service in aggregator worker1 worker2 worker3; do
        health=$(docker inspect --format='{{.State.Health.Status}}' helix-$service 2>/dev/null || echo "not running")
        echo "  - $service: $health"
    done
}

# View logs
logs() {
    service=${1:-}
    if [ -z "$service" ]; then
        docker compose logs -f
    else
        docker compose logs -f "$service"
    fi
}

# Run a training round
train() {
    log "Triggering a training round..."
    curl -X POST http://localhost:9001/api/v1/round/start \
        -H "Content-Type: application/json" \
        -d '{"model_hash": "0x0000000000000000000000000000000000000000000000000000000000000001"}'
    success "Training round triggered."
}

# Show help
help() {
    echo "HELIX 3-Worker Demo Script"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  build   - Build the Docker image"
    echo "  start   - Start the demo network"
    echo "  stop    - Stop the demo network"
    echo "  clean   - Remove all containers and volumes"
    echo "  status  - Show network status"
    echo "  logs    - View logs (optionally: logs <service>)"
    echo "  train   - Trigger a training round"
    echo "  help    - Show this help message"
}

# Main
case "${1:-help}" in
    build)
        check_deps
        build
        ;;
    start)
        check_deps
        start
        ;;
    stop)
        stop
        ;;
    clean)
        clean
        ;;
    status)
        status
        ;;
    logs)
        logs "$2"
        ;;
    train)
        train
        ;;
    help)
        help
        ;;
    *)
        error "Unknown command: $1"
        help
        exit 1
        ;;
esac
