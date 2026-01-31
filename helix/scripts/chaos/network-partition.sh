#!/bin/bash
# ==============================================================================
# HELIX Chaos Testing: Network Partition
# ==============================================================================
# Simulates network partitions to test distributed consensus and fault tolerance.
#
# Usage:
#   ./network-partition.sh isolate worker-1        # Isolate single node
#   ./network-partition.sh split 2                 # Split into 2 partitions
#   ./network-partition.sh latency worker-1 500    # Add 500ms latency
#   ./network-partition.sh loss worker-1 30        # 30% packet loss
#   ./network-partition.sh heal                    # Restore all connectivity
#
# This script tests:
#   - Network partition tolerance
#   - Byzantine fault handling
#   - Gradient synchronization under degraded conditions
#   - MPC protocol resilience
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
NAMESPACE="${NAMESPACE:-helix}"
MODE="${MODE:-docker}"  # docker or kubernetes

log() {
    echo -e "${CYAN}[NETWORK-CHAOS]${NC} $1"
}

success() {
    echo -e "${GREEN}[NETWORK-CHAOS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[NETWORK-CHAOS]${NC} $1"
}

error() {
    echo -e "${RED}[NETWORK-CHAOS]${NC} $1" >&2
}

banner() {
    cat << 'EOF'
   ██╗  ██╗███████╗██╗     ██╗██╗  ██╗    ███╗   ██╗███████╗████████╗██╗    ██╗ ██████╗ ██████╗ ██╗  ██╗
   ██║  ██║██╔════╝██║     ██║╚██╗██╔╝    ████╗  ██║██╔════╝╚══██╔══╝██║    ██║██╔═══██╗██╔══██╗██║ ██╔╝
   ███████║█████╗  ██║     ██║ ╚███╔╝     ██╔██╗ ██║█████╗     ██║   ██║ █╗ ██║██║   ██║██████╔╝█████╔╝
   ██╔══██║██╔══╝  ██║     ██║ ██╔██╗     ██║╚██╗██║██╔══╝     ██║   ██║███╗██║██║   ██║██╔══██╗██╔═██╗
   ██║  ██║███████╗███████╗██║██╔╝ ██╗    ██║ ╚████║███████╗   ██║   ╚███╔███╔╝╚██████╔╝██║  ██║██║  ██╗
   ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚═╝  ╚═╝    ╚═╝  ╚═══╝╚══════╝   ╚═╝    ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝

   Network Partition Simulation
EOF
    echo ""
}

check_requirements() {
    # Check for tc (traffic control)
    if ! command -v tc &> /dev/null; then
        warn "tc (traffic control) not found. Some features may not work."
    fi

    # Check for iptables
    if ! command -v iptables &> /dev/null; then
        warn "iptables not found. Some features may not work."
    fi
}

get_container_pid() {
    local container="$1"
    docker inspect --format '{{.State.Pid}}' "$container"
}

get_container_ip() {
    local container="$1"
    docker inspect --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$container"
}

# Execute command in container's network namespace
nsenter_exec() {
    local container="$1"
    shift
    local pid=$(get_container_pid "$container")
    nsenter -t "$pid" -n "$@"
}

isolate_container() {
    local container="$1"
    local duration="${2:-0}"

    log "Isolating container: $container"

    local ip=$(get_container_ip "$container")
    if [ -z "$ip" ]; then
        error "Could not get IP for container: $container"
        return 1
    fi

    # Block all traffic to/from this container
    docker exec "$container" iptables -A INPUT -j DROP 2>/dev/null || \
        log "Note: iptables not available in container, using host rules"

    # Alternative: use host iptables
    iptables -I DOCKER-USER -s "$ip" -j DROP 2>/dev/null || true
    iptables -I DOCKER-USER -d "$ip" -j DROP 2>/dev/null || true

    success "Container $container ($ip) isolated"

    if [ "$duration" -gt 0 ]; then
        log "Will heal after ${duration}s..."
        sleep "$duration"
        heal_container "$container"
    fi
}

heal_container() {
    local container="$1"

    log "Healing container: $container"

    local ip=$(get_container_ip "$container")
    if [ -z "$ip" ]; then
        error "Could not get IP for container: $container"
        return 1
    fi

    # Remove isolation rules
    iptables -D DOCKER-USER -s "$ip" -j DROP 2>/dev/null || true
    iptables -D DOCKER-USER -d "$ip" -j DROP 2>/dev/null || true

    docker exec "$container" iptables -D INPUT -j DROP 2>/dev/null || true

    success "Container $container ($ip) healed"
}

add_latency() {
    local container="$1"
    local delay_ms="$2"
    local jitter_ms="${3:-10}"

    log "Adding ${delay_ms}ms latency (jitter: ${jitter_ms}ms) to: $container"

    # Use tc (traffic control) to add delay
    local pid=$(get_container_pid "$container")

    # Get the interface inside the container
    nsenter -t "$pid" -n tc qdisc add dev eth0 root netem delay "${delay_ms}ms" "${jitter_ms}ms" 2>/dev/null || \
        nsenter -t "$pid" -n tc qdisc change dev eth0 root netem delay "${delay_ms}ms" "${jitter_ms}ms"

    success "Latency added to $container"
}

add_packet_loss() {
    local container="$1"
    local loss_percent="$2"

    log "Adding ${loss_percent}% packet loss to: $container"

    local pid=$(get_container_pid "$container")

    nsenter -t "$pid" -n tc qdisc add dev eth0 root netem loss "${loss_percent}%" 2>/dev/null || \
        nsenter -t "$pid" -n tc qdisc change dev eth0 root netem loss "${loss_percent}%"

    success "Packet loss added to $container"
}

add_bandwidth_limit() {
    local container="$1"
    local rate="$2"  # e.g., "1mbit"

    log "Limiting bandwidth to $rate for: $container"

    local pid=$(get_container_pid "$container")

    nsenter -t "$pid" -n tc qdisc add dev eth0 root tbf rate "$rate" burst 32kbit latency 400ms 2>/dev/null || \
        nsenter -t "$pid" -n tc qdisc change dev eth0 root tbf rate "$rate" burst 32kbit latency 400ms

    success "Bandwidth limited to $rate for $container"
}

clear_tc() {
    local container="$1"

    log "Clearing traffic control rules for: $container"

    local pid=$(get_container_pid "$container")
    nsenter -t "$pid" -n tc qdisc del dev eth0 root 2>/dev/null || true

    success "Traffic control rules cleared for $container"
}

split_network() {
    local partitions="$1"

    log "Creating network split into $partitions partitions..."

    # Get all helix containers
    local containers=($(docker ps --filter "label=helix.service" --format "{{.Names}}"))

    if [ ${#containers[@]} -eq 0 ]; then
        error "No HELIX containers found!"
        return 1
    fi

    # Calculate partition sizes
    local total=${#containers[@]}
    local per_partition=$((total / partitions))

    log "Total containers: $total"
    log "Partition size: ~$per_partition"

    # Create partitions
    local partition_members=()
    for ((p=0; p<partitions; p++)); do
        local start=$((p * per_partition))
        local end
        if [ $p -eq $((partitions - 1)) ]; then
            end=$total
        else
            end=$((start + per_partition))
        fi

        partition_members[$p]=""
        for ((i=start; i<end; i++)); do
            partition_members[$p]+="${containers[$i]} "
        done

        log "Partition $p: ${partition_members[$p]}"
    done

    # Block traffic between partitions
    for ((p1=0; p1<partitions; p1++)); do
        for container1 in ${partition_members[$p1]}; do
            local ip1=$(get_container_ip "$container1")

            for ((p2=p1+1; p2<partitions; p2++)); do
                for container2 in ${partition_members[$p2]}; do
                    local ip2=$(get_container_ip "$container2")

                    # Block traffic between these containers
                    iptables -I DOCKER-USER -s "$ip1" -d "$ip2" -j DROP 2>/dev/null || true
                    iptables -I DOCKER-USER -s "$ip2" -d "$ip1" -j DROP 2>/dev/null || true
                done
            done
        done
    done

    success "Network split into $partitions partitions"

    # Store partition info for healing
    echo "${partition_members[@]}" > /tmp/helix-network-partitions.txt
}

heal_all() {
    log "Healing all network partitions..."

    # Flush DOCKER-USER chain (reset to default)
    iptables -F DOCKER-USER 2>/dev/null || true
    iptables -A DOCKER-USER -j RETURN 2>/dev/null || true

    # Clear tc rules on all containers
    local containers=($(docker ps --filter "label=helix.service" --format "{{.Names}}"))
    for container in "${containers[@]}"; do
        clear_tc "$container" 2>/dev/null || true
    done

    rm -f /tmp/helix-network-partitions.txt

    success "All network chaos healed"
}

simulate_byzantine() {
    local container="$1"

    log "Simulating Byzantine behavior for: $container"

    # Combination of delay + packet loss + reordering
    local pid=$(get_container_pid "$container")

    nsenter -t "$pid" -n tc qdisc add dev eth0 root netem \
        delay 100ms 50ms distribution normal \
        loss 10% \
        corrupt 5% \
        reorder 25% 50% 2>/dev/null || \
    nsenter -t "$pid" -n tc qdisc change dev eth0 root netem \
        delay 100ms 50ms distribution normal \
        loss 10% \
        corrupt 5% \
        reorder 25% 50%

    success "Byzantine simulation applied to $container"
}

monitor_network() {
    local duration="${1:-60}"

    log "Monitoring network for ${duration}s..."

    local end_time=$(($(date +%s) + duration))

    while [ $(date +%s) -lt $end_time ]; do
        echo ""
        echo "--- $(date) ---"

        # Show container connectivity
        local containers=($(docker ps --filter "label=helix.service" --format "{{.Names}}"))

        echo "Connectivity matrix:"
        for c1 in "${containers[@]}"; do
            printf "%-20s" "$c1"
            for c2 in "${containers[@]}"; do
                if [ "$c1" != "$c2" ]; then
                    local ip2=$(get_container_ip "$c2")
                    if docker exec "$c1" ping -c 1 -W 1 "$ip2" &>/dev/null; then
                        printf "${GREEN}O${NC} "
                    else
                        printf "${RED}X${NC} "
                    fi
                else
                    printf "- "
                fi
            done
            echo ""
        done

        sleep 10
    done
}

status() {
    log "Network status:"
    echo ""

    echo "${BOLD}Container Network Info:${NC}"
    docker ps --filter "label=helix.service" --format "table {{.Names}}\t{{.Networks}}"

    echo ""
    echo "${BOLD}DOCKER-USER iptables rules:${NC}"
    iptables -L DOCKER-USER -n 2>/dev/null || echo "  (no custom rules)"

    echo ""
    echo "${BOLD}Container TC rules:${NC}"
    local containers=($(docker ps --filter "label=helix.service" --format "{{.Names}}"))
    for container in "${containers[@]}"; do
        echo "  $container:"
        local pid=$(get_container_pid "$container")
        nsenter -t "$pid" -n tc qdisc show dev eth0 2>/dev/null | grep -v "pfifo_fast" | sed 's/^/    /' || echo "    (no rules)"
    done
}

usage() {
    cat << EOF
HELIX Chaos Testing: Network Partition

Usage: $(basename "$0") <command> [args]

Commands:
    isolate <container> [duration]    Isolate container from network
    split <n>                         Split network into n partitions
    latency <container> <ms> [jitter] Add latency to container
    loss <container> <percent>        Add packet loss to container
    bandwidth <container> <rate>      Limit bandwidth (e.g., 1mbit)
    byzantine <container>             Simulate Byzantine behavior
    heal [container]                  Heal specific container or all
    monitor [seconds]                 Monitor network connectivity
    status                            Show current network status
    help                              Show this help

Examples:
    $(basename "$0") isolate helix-worker1 30      # Isolate for 30s
    $(basename "$0") split 2                       # Create 2 partitions
    $(basename "$0") latency helix-worker1 200 50  # 200ms +/- 50ms
    $(basename "$0") loss helix-worker1 30         # 30% packet loss
    $(basename "$0") bandwidth helix-worker1 1mbit # Limit to 1Mbps
    $(basename "$0") byzantine helix-worker1       # Byzantine behavior
    $(basename "$0") heal                          # Heal all

Notes:
    - Requires root/sudo for iptables and tc commands
    - Docker containers must have NET_ADMIN capability for in-container rules
    - Use 'heal' to restore normal network before ending tests

EOF
}

# Main
banner
check_requirements

case "${1:-help}" in
    isolate)
        isolate_container "${2:-}" "${3:-0}"
        ;;
    split)
        split_network "${2:-2}"
        ;;
    latency)
        add_latency "${2:-}" "${3:-100}" "${4:-10}"
        ;;
    loss)
        add_packet_loss "${2:-}" "${3:-10}"
        ;;
    bandwidth)
        add_bandwidth_limit "${2:-}" "${3:-1mbit}"
        ;;
    byzantine)
        simulate_byzantine "${2:-}"
        ;;
    heal)
        if [ -n "${2:-}" ]; then
            heal_container "$2"
            clear_tc "$2"
        else
            heal_all
        fi
        ;;
    monitor)
        monitor_network "${2:-60}"
        ;;
    status)
        status
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        error "Unknown command: $1"
        usage
        exit 1
        ;;
esac
