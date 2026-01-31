#!/bin/bash
#
# HELIX Health Check Script
# ==========================
# Checks the health of HELIX components
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# Configuration
RPC_URL=${RPC_URL:-http://localhost:8545}
COORDINATOR_ADDRESS=${COORDINATOR_ADDRESS:-}
API_URL=${API_URL:-http://localhost:8080}
DASHBOARD_URL=${DASHBOARD_URL:-http://localhost:3000}

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

# Banner
cat << 'EOF'
╔═══════════════════════════════════════════════════════════╗
║                   HELIX Health Check                      ║
╚═══════════════════════════════════════════════════════════╝
EOF

echo -e "\n${CYAN}Configuration:${NC}"
echo "  RPC URL:     $RPC_URL"
echo "  Coordinator: ${COORDINATOR_ADDRESS:-<not set>}"
echo "  API URL:     $API_URL"
echo "  Dashboard:   $DASHBOARD_URL"

# Run checks
check_system
check_rpc
check_contracts
check_api
check_dashboard

# Summary
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}Summary:${NC}"
echo -e "  ${GREEN}Passed:${NC}   $PASSED"
echo -e "  ${YELLOW}Warnings:${NC} $WARNINGS"
echo -e "  ${RED}Failed:${NC}   $FAILED"

if [ $FAILED -eq 0 ]; then
    echo -e "\n${GREEN}All critical checks passed!${NC}"
    exit 0
else
    echo -e "\n${RED}Some checks failed.${NC}"
    exit 1
fi
