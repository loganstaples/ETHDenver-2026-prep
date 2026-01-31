#!/bin/bash
# ==============================================================================
# HELIX Health Check Script
# ==============================================================================
# Comprehensive health checks for HELIX distributed ML training infrastructure.
#
# Usage:
#   ./health-check.sh              # Full health check
#   ./health-check.sh --quick      # Quick check (essential only)
#   ./health-check.sh --docker     # Docker-specific checks
#   ./health-check.sh --kubernetes # Kubernetes-specific checks
#   ./health-check.sh --json       # Output as JSON
#   ./health-check.sh --watch      # Continuous monitoring
# ==============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'
DIM='\033[2m'

# Configuration
RPC_URL=${RPC_URL:-http://localhost:8545}
COORDINATOR_ADDRESS=${COORDINATOR_ADDRESS:-}
API_URL=${API_URL:-http://localhost:9001}
AGGREGATOR_URL=${AGGREGATOR_URL:-http://localhost:9001}
DASHBOARD_URL=${DASHBOARD_URL:-http://localhost:3000}
PROMETHEUS_URL=${PROMETHEUS_URL:-http://localhost:9090}
GRAFANA_URL=${GRAFANA_URL:-http://localhost:3001}
NAMESPACE=${NAMESPACE:-helix}

# Modes
QUICK_MODE=false
DOCKER_MODE=false
K8S_MODE=false
JSON_MODE=false
WATCH_MODE=false

# Counters
PASSED=0
FAILED=0
WARNINGS=0

check() {
    local name="$1"
    local status="$2"
    local message="$3"

    if [ "$status" = "ok" ]; then
        echo -e "  ${GREEN}✓${NC} $name: $message"
        ((PASSED++))
    elif [ "$status" = "warn" ]; then
        echo -e "  ${YELLOW}⚠${NC} $name: $message"
        ((WARNINGS++))
    else
        echo -e "  ${RED}✗${NC} $name: $message"
        ((FAILED++))
    fi
}

check_rpc() {
    echo -e "\n${BOLD}RPC Node:${NC}"

    # Connection test
    if CHAIN_ID=$(cast chain-id --rpc-url $RPC_URL 2>/dev/null); then
        check "Connection" "ok" "Connected to chain $CHAIN_ID"
    else
        check "Connection" "fail" "Cannot connect to $RPC_URL"
        return
    fi

    # Block number
    if BLOCK=$(cast block-number --rpc-url $RPC_URL 2>/dev/null); then
        check "Block Height" "ok" "$BLOCK"
    else
        check "Block Height" "fail" "Cannot get block number"
    fi

    # Gas price
    if GAS=$(cast gas-price --rpc-url $RPC_URL 2>/dev/null); then
        GAS_GWEI=$(echo "scale=2; $GAS / 1000000000" | bc)
        check "Gas Price" "ok" "${GAS_GWEI} gwei"
    else
        check "Gas Price" "warn" "Cannot get gas price"
    fi
}

check_contracts() {
    echo -e "\n${BOLD}Smart Contracts:${NC}"

    if [ -z "$COORDINATOR_ADDRESS" ]; then
        check "Coordinator" "warn" "Address not set (export COORDINATOR_ADDRESS=...)"
        return
    fi

    # Check coordinator code
    if CODE=$(cast code $COORDINATOR_ADDRESS --rpc-url $RPC_URL 2>/dev/null); then
        if [ "$CODE" != "0x" ] && [ -n "$CODE" ]; then
            check "Coordinator" "ok" "${COORDINATOR_ADDRESS:0:10}..."
        else
            check "Coordinator" "fail" "No code at address"
        fi
    else
        check "Coordinator" "fail" "Cannot query contract"
    fi

    # Try to get next model ID
    if NEXT_MODEL=$(cast call $COORDINATOR_ADDRESS "nextModelId()(uint256)" --rpc-url $RPC_URL 2>/dev/null); then
        check "Contract State" "ok" "nextModelId = $NEXT_MODEL"
    else
        check "Contract State" "warn" "Cannot read state"
    fi
}

check_api() {
    echo -e "\n${BOLD}API Server:${NC}"

    # Health endpoint
    if RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $API_URL/health 2>/dev/null); then
        if [ "$RESPONSE" = "200" ]; then
            check "Health Endpoint" "ok" "HTTP 200"
        else
            check "Health Endpoint" "warn" "HTTP $RESPONSE"
        fi
    else
        check "Health Endpoint" "fail" "Not reachable"
    fi

    # API info
    if INFO=$(curl -s $API_URL/ 2>/dev/null | jq -r '.version // empty'); then
        if [ -n "$INFO" ]; then
            check "API Version" "ok" "$INFO"
        fi
    fi
}

check_dashboard() {
    echo -e "\n${BOLD}Dashboard:${NC}"

    if RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $DASHBOARD_URL 2>/dev/null); then
        if [ "$RESPONSE" = "200" ]; then
            check "Web UI" "ok" "HTTP 200"
        else
            check "Web UI" "warn" "HTTP $RESPONSE"
        fi
    else
        check "Web UI" "fail" "Not reachable"
    fi
}

check_docker() {
    echo -e "\n${BOLD}Docker Containers:${NC}"

    if ! command -v docker &> /dev/null; then
        check "Docker" "warn" "Docker not installed"
        return
    fi

    if ! docker info &> /dev/null; then
        check "Docker" "fail" "Docker daemon not running"
        return
    fi
    check "Docker" "ok" "Daemon running"

    # Check HELIX containers
    local containers=$(docker ps --filter "label=helix.service" --format "{{.Names}}" 2>/dev/null)
    if [ -z "$containers" ]; then
        check "HELIX Containers" "warn" "No HELIX containers found"
        return
    fi

    for container in $containers; do
        local status=$(docker inspect --format '{{.State.Status}}' "$container" 2>/dev/null)
        local health=$(docker inspect --format '{{.State.Health.Status}}' "$container" 2>/dev/null || echo "none")

        if [ "$status" = "running" ]; then
            if [ "$health" = "healthy" ] || [ "$health" = "none" ]; then
                check "$container" "ok" "Running ($health)"
            else
                check "$container" "warn" "Running but $health"
            fi
        else
            check "$container" "fail" "$status"
        fi
    done

    # Check aggregator specifically
    local agg_container=$(docker ps --filter "label=helix.role=aggregator" --format "{{.Names}}" 2>/dev/null | head -1)
    if [ -n "$agg_container" ]; then
        local agg_ip=$(docker inspect --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$agg_container" 2>/dev/null)
        check "Aggregator IP" "ok" "$agg_ip"
    fi

    # Count workers
    local worker_count=$(docker ps --filter "label=helix.role=worker" --format "{{.Names}}" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$worker_count" -ge 3 ]; then
        check "Worker Count" "ok" "$worker_count workers running"
    elif [ "$worker_count" -gt 0 ]; then
        check "Worker Count" "warn" "Only $worker_count workers (need 3+)"
    else
        check "Worker Count" "fail" "No workers running"
    fi
}

check_kubernetes() {
    echo -e "\n${BOLD}Kubernetes:${NC}"

    if ! command -v kubectl &> /dev/null; then
        check "kubectl" "warn" "kubectl not installed"
        return
    fi

    if ! kubectl cluster-info &> /dev/null; then
        check "Cluster" "fail" "Cannot connect to cluster"
        return
    fi
    check "Cluster" "ok" "Connected"

    # Check namespace
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        check "Namespace" "ok" "$NAMESPACE exists"
    else
        check "Namespace" "fail" "Namespace $NAMESPACE not found"
        return
    fi

    # Check pods
    local pods=$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)
    if [ -z "$pods" ]; then
        check "Pods" "warn" "No pods in namespace"
        return
    fi

    # Check pod status
    local ready=0
    local not_ready=0
    for pod in $pods; do
        local phase=$(kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null)
        if [ "$phase" = "Running" ]; then
            ((ready++))
        else
            ((not_ready++))
            check "$pod" "warn" "$phase"
        fi
    done
    check "Ready Pods" "ok" "$ready pods running"

    # Check aggregator
    local agg_ready=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/component=aggregator -o jsonpath='{.items[0].status.containerStatuses[0].ready}' 2>/dev/null)
    if [ "$agg_ready" = "true" ]; then
        check "Aggregator" "ok" "Ready"
    else
        check "Aggregator" "fail" "Not ready"
    fi

    # Check worker statefulset
    local worker_replicas=$(kubectl get statefulset -n "$NAMESPACE" -l app.kubernetes.io/component=worker -o jsonpath='{.items[0].status.readyReplicas}' 2>/dev/null)
    local worker_desired=$(kubectl get statefulset -n "$NAMESPACE" -l app.kubernetes.io/component=worker -o jsonpath='{.items[0].spec.replicas}' 2>/dev/null)
    if [ "$worker_replicas" = "$worker_desired" ]; then
        check "Workers" "ok" "$worker_replicas/$worker_desired ready"
    else
        check "Workers" "warn" "$worker_replicas/$worker_desired ready"
    fi

    # Check PVCs
    local pvc_count=$(kubectl get pvc -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
    check "PVCs" "ok" "$pvc_count persistent volume claims"

    # Check services
    local svc_count=$(kubectl get svc -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
    check "Services" "ok" "$svc_count services"
}

check_monitoring() {
    echo -e "\n${BOLD}Monitoring:${NC}"

    # Prometheus
    if RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$PROMETHEUS_URL/-/ready" 2>/dev/null); then
        if [ "$RESPONSE" = "200" ]; then
            check "Prometheus" "ok" "Ready"

            # Check target count
            local targets=$(curl -s "$PROMETHEUS_URL/api/v1/targets" 2>/dev/null | jq '.data.activeTargets | length' 2>/dev/null || echo "0")
            check "Prometheus Targets" "ok" "$targets active"
        else
            check "Prometheus" "warn" "HTTP $RESPONSE"
        fi
    else
        check "Prometheus" "warn" "Not reachable at $PROMETHEUS_URL"
    fi

    # Grafana
    if RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$GRAFANA_URL/api/health" 2>/dev/null); then
        if [ "$RESPONSE" = "200" ]; then
            check "Grafana" "ok" "Healthy"
        else
            check "Grafana" "warn" "HTTP $RESPONSE"
        fi
    else
        check "Grafana" "warn" "Not reachable at $GRAFANA_URL"
    fi
}

check_workers() {
    echo -e "\n${BOLD}Worker Nodes:${NC}"

    # Try to get worker status from aggregator API
    local workers_response=$(curl -s "$AGGREGATOR_URL/workers" 2>/dev/null)
    if [ -n "$workers_response" ]; then
        local worker_count=$(echo "$workers_response" | jq 'length' 2>/dev/null || echo "0")
        if [ "$worker_count" -gt 0 ]; then
            check "Registered Workers" "ok" "$worker_count workers"

            # Check each worker
            for i in $(seq 0 $((worker_count - 1))); do
                local worker_id=$(echo "$workers_response" | jq -r ".[$i].id" 2>/dev/null)
                local worker_status=$(echo "$workers_response" | jq -r ".[$i].status" 2>/dev/null)
                if [ "$worker_status" = "active" ]; then
                    check "Worker $worker_id" "ok" "Active"
                else
                    check "Worker $worker_id" "warn" "$worker_status"
                fi
            done
        else
            check "Registered Workers" "warn" "No workers registered"
        fi
    else
        check "Worker API" "warn" "Cannot fetch worker status"
    fi
}

check_training() {
    echo -e "\n${BOLD}Training Status:${NC}"

    # Check training status from aggregator
    local training_response=$(curl -s "$AGGREGATOR_URL/training/status" 2>/dev/null)
    if [ -n "$training_response" ]; then
        local is_active=$(echo "$training_response" | jq -r '.active' 2>/dev/null)
        local current_round=$(echo "$training_response" | jq -r '.currentRound' 2>/dev/null)
        local total_rounds=$(echo "$training_response" | jq -r '.totalRounds' 2>/dev/null)

        if [ "$is_active" = "true" ]; then
            check "Training" "ok" "Active (round $current_round/$total_rounds)"
        else
            check "Training" "ok" "Idle"
        fi

        local error_bound=$(echo "$training_response" | jq -r '.accumulatedError' 2>/dev/null)
        if [ -n "$error_bound" ] && [ "$error_bound" != "null" ]; then
            if (( $(echo "$error_bound < 0.5" | bc -l) )); then
                check "Error Bound" "ok" "$error_bound"
            elif (( $(echo "$error_bound < 0.8" | bc -l) )); then
                check "Error Bound" "warn" "$error_bound (approaching limit)"
            else
                check "Error Bound" "fail" "$error_bound (exceeded limit)"
            fi
        fi
    else
        check "Training API" "warn" "Cannot fetch training status"
    fi
}

check_system() {
    echo -e "\n${BOLD}System:${NC}"

    # Memory
    if [ "$(uname)" = "Darwin" ]; then
        MEM_USED=$(vm_stat | awk '/Pages active/ {print $3}' | tr -d '.')
        MEM_TOTAL=$(sysctl -n hw.memsize)
        MEM_PERCENT=$((MEM_USED * 4096 * 100 / MEM_TOTAL))
    else
        MEM_PERCENT=$(free | awk '/Mem:/ {printf "%.0f", $3/$2 * 100}')
    fi

    if [ "$MEM_PERCENT" -lt 80 ]; then
        check "Memory" "ok" "${MEM_PERCENT}% used"
    elif [ "$MEM_PERCENT" -lt 95 ]; then
        check "Memory" "warn" "${MEM_PERCENT}% used"
    else
        check "Memory" "fail" "${MEM_PERCENT}% used"
    fi

    # Disk
    DISK_PERCENT=$(df -h . | awk 'NR==2 {gsub(/%/,""); print $5}')
    if [ "$DISK_PERCENT" -lt 80 ]; then
        check "Disk" "ok" "${DISK_PERCENT}% used"
    elif [ "$DISK_PERCENT" -lt 95 ]; then
        check "Disk" "warn" "${DISK_PERCENT}% used"
    else
        check "Disk" "fail" "${DISK_PERCENT}% used"
    fi

    # Required commands
    for cmd in cargo forge cast anvil curl jq; do
        if command -v $cmd &> /dev/null; then
            check "$cmd" "ok" "$(which $cmd)"
        else
            check "$cmd" "fail" "not found"
        fi
    done
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --docker)
            DOCKER_MODE=true
            shift
            ;;
        --kubernetes|--k8s)
            K8S_MODE=true
            shift
            ;;
        --json)
            JSON_MODE=true
            shift
            ;;
        --watch)
            WATCH_MODE=true
            shift
            ;;
        --help|-h)
            cat << EOF
HELIX Health Check Script

Usage: $(basename "$0") [OPTIONS]

Options:
    --quick          Quick check (essential only)
    --docker         Docker-specific checks
    --kubernetes     Kubernetes-specific checks
    --json           Output as JSON
    --watch          Continuous monitoring (every 30s)
    --help           Show this help

Environment Variables:
    RPC_URL              Ethereum RPC URL
    COORDINATOR_ADDRESS  Coordinator contract address
    API_URL              Aggregator API URL
    DASHBOARD_URL        Dashboard URL
    PROMETHEUS_URL       Prometheus URL
    NAMESPACE            Kubernetes namespace

EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

run_checks() {
    PASSED=0
    FAILED=0
    WARNINGS=0

    # Banner
    if ! $JSON_MODE; then
        cat << 'EOF'
╔═══════════════════════════════════════════════════════════╗
║                   HELIX Health Check                      ║
╚═══════════════════════════════════════════════════════════╝
EOF

        echo -e "\n${CYAN}Configuration:${NC}"
        echo "  RPC URL:     $RPC_URL"
        echo "  Coordinator: ${COORDINATOR_ADDRESS:-<not set>}"
        echo "  Aggregator:  $AGGREGATOR_URL"
        echo "  Dashboard:   $DASHBOARD_URL"
    fi

    # Run appropriate checks
    if $DOCKER_MODE; then
        check_docker
    elif $K8S_MODE; then
        check_kubernetes
    elif $QUICK_MODE; then
        check_rpc
        check_api
    else
        check_system
        check_rpc
        check_contracts
        check_api
        check_dashboard
        check_docker 2>/dev/null || true
        check_monitoring
        check_workers
        check_training
    fi

    # Summary
    if ! $JSON_MODE; then
        echo ""
        echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
        echo -e "${BOLD}Summary:${NC}"
        echo -e "  ${GREEN}Passed:${NC}   $PASSED"
        echo -e "  ${YELLOW}Warnings:${NC} $WARNINGS"
        echo -e "  ${RED}Failed:${NC}   $FAILED"

        if [ $FAILED -eq 0 ]; then
            echo -e "\n${GREEN}All critical checks passed!${NC}"
        else
            echo -e "\n${RED}Some checks failed.${NC}"
        fi
    else
        cat << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "passed": $PASSED,
    "warnings": $WARNINGS,
    "failed": $FAILED,
    "healthy": $([ $FAILED -eq 0 ] && echo "true" || echo "false")
}
EOF
    fi

    return $FAILED
}

# Main execution
if $WATCH_MODE; then
    while true; do
        clear
        run_checks || true
        echo ""
        echo -e "${DIM}Next check in 30 seconds... (Ctrl+C to stop)${NC}"
        sleep 30
    done
else
    run_checks
    exit $?
fi
