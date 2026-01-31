#!/bin/bash
# ==============================================================================
# HELIX Chaos Testing: Kill Node
# ==============================================================================
# Simulates node failures to test network resilience and fault tolerance.
#
# Usage:
#   ./kill-node.sh random           # Kill a random worker
#   ./kill-node.sh worker-1         # Kill specific worker
#   ./kill-node.sh aggregator       # Kill aggregator (dangerous!)
#   ./kill-node.sh --cascade 2      # Kill 2 workers in sequence
#   ./kill-node.sh --restore        # Restart all killed nodes
#
# This script tests:
#   - Worker failure recovery
#   - Gradient re-assignment
#   - Aggregator failover (if HA)
#   - Network partition handling
# ==============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# Configuration
DOCKER_COMPOSE="${DOCKER_COMPOSE:-docker-compose}"
COMPOSE_FILE="${COMPOSE_FILE:-../docker/docker-compose.yml}"
KUBECTL="${KUBECTL:-kubectl}"
NAMESPACE="${NAMESPACE:-helix}"
MODE="${MODE:-docker}"  # docker or kubernetes

log() {
    echo -e "${CYAN}[CHAOS]${NC} $1"
}

success() {
    echo -e "${GREEN}[CHAOS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[CHAOS]${NC} $1"
}

error() {
    echo -e "${RED}[CHAOS]${NC} $1" >&2
}

banner() {
    cat << 'EOF'
   ██╗  ██╗███████╗██╗     ██╗██╗  ██╗     ██████╗██╗  ██╗ █████╗  ██████╗ ███████╗
   ██║  ██║██╔════╝██║     ██║╚██╗██╔╝    ██╔════╝██║  ██║██╔══██╗██╔═══██╗██╔════╝
   ███████║█████╗  ██║     ██║ ╚███╔╝     ██║     ███████║███████║██║   ██║███████╗
   ██╔══██║██╔══╝  ██║     ██║ ██╔██╗     ██║     ██╔══██║██╔══██║██║   ██║╚════██║
   ██║  ██║███████╗███████╗██║██╔╝ ██╗    ╚██████╗██║  ██║██║  ██║╚██████╔╝███████║
   ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚═╝  ╚═╝     ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝

   Node Failure Simulation
EOF
    echo ""
}

get_workers_docker() {
    docker ps --filter "label=helix.role=worker" --format "{{.Names}}"
}

get_workers_k8s() {
    $KUBECTL get pods -n "$NAMESPACE" -l app.kubernetes.io/component=worker -o jsonpath='{.items[*].metadata.name}'
}

get_aggregator_docker() {
    docker ps --filter "label=helix.role=aggregator" --format "{{.Names}}"
}

get_aggregator_k8s() {
    $KUBECTL get pods -n "$NAMESPACE" -l app.kubernetes.io/component=aggregator -o jsonpath='{.items[*].metadata.name}'
}

kill_docker_container() {
    local container="$1"
    local signal="${2:-SIGKILL}"

    log "Killing container: $container (signal: $signal)"

    # Record container info for potential restore
    docker inspect "$container" > "/tmp/helix-chaos-${container}.json" 2>/dev/null || true

    case "$signal" in
        SIGKILL)
            docker kill "$container"
            ;;
        SIGTERM)
            docker stop "$container"
            ;;
        SIGSTOP)
            docker pause "$container"
            ;;
        *)
            docker kill --signal="$signal" "$container"
            ;;
    esac

    success "Container $container killed"
}

kill_k8s_pod() {
    local pod="$1"
    local grace_period="${2:-0}"

    log "Deleting pod: $pod (grace period: ${grace_period}s)"

    $KUBECTL delete pod "$pod" -n "$NAMESPACE" --grace-period="$grace_period"

    success "Pod $pod deleted (will be recreated by controller)"
}

kill_random_worker() {
    log "Selecting random worker to kill..."

    if [ "$MODE" = "docker" ]; then
        local workers=($(get_workers_docker))
    else
        local workers=($(get_workers_k8s))
    fi

    if [ ${#workers[@]} -eq 0 ]; then
        error "No workers found!"
        exit 1
    fi

    local random_index=$((RANDOM % ${#workers[@]}))
    local target="${workers[$random_index]}"

    log "Selected: $target"

    if [ "$MODE" = "docker" ]; then
        kill_docker_container "$target"
    else
        kill_k8s_pod "$target"
    fi
}

kill_specific_node() {
    local target="$1"

    log "Targeting node: $target"

    if [ "$MODE" = "docker" ]; then
        if docker ps --format "{{.Names}}" | grep -q "^${target}$"; then
            kill_docker_container "$target"
        elif docker ps --format "{{.Names}}" | grep -q "$target"; then
            # Partial match
            local full_name=$(docker ps --format "{{.Names}}" | grep "$target" | head -1)
            kill_docker_container "$full_name"
        else
            error "Container not found: $target"
            exit 1
        fi
    else
        if $KUBECTL get pod "$target" -n "$NAMESPACE" &>/dev/null; then
            kill_k8s_pod "$target"
        else
            error "Pod not found: $target"
            exit 1
        fi
    fi
}

kill_aggregator() {
    warn "WARNING: Killing the aggregator will halt training!"
    read -p "Are you sure? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        log "Aborted"
        exit 0
    fi

    if [ "$MODE" = "docker" ]; then
        local aggregator=$(get_aggregator_docker)
    else
        local aggregator=$(get_aggregator_k8s | awk '{print $1}')
    fi

    if [ -z "$aggregator" ]; then
        error "Aggregator not found!"
        exit 1
    fi

    log "Killing aggregator: $aggregator"

    if [ "$MODE" = "docker" ]; then
        kill_docker_container "$aggregator"
    else
        kill_k8s_pod "$aggregator"
    fi
}

cascade_kill() {
    local count="${1:-2}"
    local delay="${2:-5}"

    log "Cascade kill: $count workers with ${delay}s delay"

    if [ "$MODE" = "docker" ]; then
        local workers=($(get_workers_docker))
    else
        local workers=($(get_workers_k8s))
    fi

    if [ ${#workers[@]} -lt "$count" ]; then
        error "Not enough workers! Have ${#workers[@]}, need $count"
        exit 1
    fi

    for ((i=0; i<count; i++)); do
        local target="${workers[$i]}"
        log "[$((i+1))/$count] Killing: $target"

        if [ "$MODE" = "docker" ]; then
            kill_docker_container "$target"
        else
            kill_k8s_pod "$target"
        fi

        if [ $i -lt $((count-1)) ]; then
            log "Waiting ${delay}s before next kill..."
            sleep "$delay"
        fi
    done

    success "Cascade kill complete: $count workers killed"
}

restore_all() {
    log "Restoring all killed nodes..."

    if [ "$MODE" = "docker" ]; then
        # Restart stopped containers
        local stopped=$(docker ps -a --filter "status=exited" --filter "label=helix.service" --format "{{.Names}}")
        for container in $stopped; do
            log "Restarting: $container"
            docker start "$container"
            success "Restarted: $container"
        done

        # Unpause paused containers
        local paused=$(docker ps --filter "status=paused" --filter "label=helix.service" --format "{{.Names}}")
        for container in $paused; do
            log "Unpausing: $container"
            docker unpause "$container"
            success "Unpaused: $container"
        done
    else
        # For Kubernetes, pods are automatically recreated
        # Just check status
        log "Kubernetes pods are automatically recreated"
        log "Current pod status:"
        $KUBECTL get pods -n "$NAMESPACE"
    fi
}

monitor_recovery() {
    local duration="${1:-60}"

    log "Monitoring recovery for ${duration}s..."

    local end_time=$(($(date +%s) + duration))

    while [ $(date +%s) -lt $end_time ]; do
        echo ""
        echo "--- $(date) ---"

        if [ "$MODE" = "docker" ]; then
            echo "Workers:"
            docker ps --filter "label=helix.role=worker" --format "  {{.Names}}: {{.Status}}"
            echo "Aggregator:"
            docker ps --filter "label=helix.role=aggregator" --format "  {{.Names}}: {{.Status}}"
        else
            echo "Pods:"
            $KUBECTL get pods -n "$NAMESPACE" -o wide
        fi

        sleep 5
    done

    success "Monitoring complete"
}

status() {
    log "Current cluster status:"
    echo ""

    if [ "$MODE" = "docker" ]; then
        echo "${BOLD}Docker Containers:${NC}"
        docker ps --filter "label=helix.service" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    else
        echo "${BOLD}Kubernetes Pods:${NC}"
        $KUBECTL get pods -n "$NAMESPACE" -o wide
    fi
}

usage() {
    cat << EOF
HELIX Chaos Testing: Kill Node

Usage: $(basename "$0") [OPTIONS] <command>

Commands:
    random              Kill a random worker
    <node-name>         Kill specific node by name
    aggregator          Kill the aggregator (dangerous!)
    --cascade <n>       Kill n workers in sequence
    --restore           Restart all killed nodes
    --monitor [secs]    Monitor recovery (default: 60s)
    status              Show current cluster status
    help                Show this help

Options:
    --delay <secs>      Delay between cascade kills (default: 5)
    --mode <docker|k8s> Execution mode (default: docker)
    --namespace <ns>    Kubernetes namespace (default: helix)

Examples:
    $(basename "$0") random
    $(basename "$0") helix-worker1
    $(basename "$0") --cascade 2 --delay 10
    $(basename "$0") --restore
    $(basename "$0") --mode k8s random

EOF
}

# Parse arguments
CASCADE=0
DELAY=5
MONITOR=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --cascade)
            CASCADE="$2"
            shift 2
            ;;
        --delay)
            DELAY="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --restore)
            banner
            restore_all
            exit 0
            ;;
        --monitor)
            MONITOR="${2:-60}"
            shift
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                shift
            fi
            ;;
        random)
            banner
            kill_random_worker
            if [ "$MONITOR" -gt 0 ]; then
                monitor_recovery "$MONITOR"
            fi
            exit 0
            ;;
        aggregator)
            banner
            kill_aggregator
            if [ "$MONITOR" -gt 0 ]; then
                monitor_recovery "$MONITOR"
            fi
            exit 0
            ;;
        status)
            banner
            status
            exit 0
            ;;
        help|--help|-h)
            usage
            exit 0
            ;;
        -*)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
        *)
            # Specific node name
            banner
            kill_specific_node "$1"
            if [ "$MONITOR" -gt 0 ]; then
                monitor_recovery "$MONITOR"
            fi
            exit 0
            ;;
    esac
done

# Handle cascade mode
if [ "$CASCADE" -gt 0 ]; then
    banner
    cascade_kill "$CASCADE" "$DELAY"
    if [ "$MONITOR" -gt 0 ]; then
        monitor_recovery "$MONITOR"
    fi
    exit 0
fi

# Default: show usage
usage
