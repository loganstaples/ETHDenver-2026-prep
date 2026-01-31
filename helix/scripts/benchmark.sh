#!/bin/bash
#
# HELIX Benchmark Script
# =======================
# Performance benchmarks for HELIX components
#

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
ITERATIONS=${ITERATIONS:-100}
WARMUP=${WARMUP:-10}
OUTPUT_DIR=${OUTPUT_DIR:-./benchmark_results}
RPC_URL=${RPC_URL:-http://localhost:8545}

# Results tracking
declare -A RESULTS

log() {
    echo -e "${CYAN}[BENCH]${NC} $1"
}

benchmark() {
    local name="$1"
    local iterations="$2"
    shift 2
    local cmd="$@"

    echo -e "\n${BOLD}Benchmark: $name${NC}"
    echo "  Iterations: $iterations"
    echo "  Warmup:     $WARMUP"

    # Warmup
    echo -n "  Warming up: "
    for ((i=0; i<WARMUP; i++)); do
        eval "$cmd" > /dev/null 2>&1 || true
        echo -n "."
    done
    echo " done"

    # Benchmark
    local times=()
    echo -n "  Running:    "
    for ((i=0; i<iterations; i++)); do
        local start=$(python3 -c "import time; print(int(time.time() * 1000))")
        eval "$cmd" > /dev/null 2>&1 || true
        local end=$(python3 -c "import time; print(int(time.time() * 1000))")
        times+=($((end - start)))

        if ((i % 10 == 0)); then
            echo -n "█"
        fi
    done
    echo " done"

    # Calculate statistics
    local sum=0
    local min=${times[0]}
    local max=${times[0]}

    for t in "${times[@]}"; do
        sum=$((sum + t))
        ((t < min)) && min=$t
        ((t > max)) && max=$t
    done

    local mean=$((sum / iterations))

    # Calculate percentiles (simplified)
    IFS=$'\n' sorted=($(sort -n <<<"${times[*]}"))
    unset IFS
    local p50=${sorted[$((iterations / 2))]}
    local p95=${sorted[$((iterations * 95 / 100))]}
    local p99=${sorted[$((iterations * 99 / 100))]}

    # Store results
    RESULTS["$name.mean"]=$mean
    RESULTS["$name.min"]=$min
    RESULTS["$name.max"]=$max
    RESULTS["$name.p50"]=$p50
    RESULTS["$name.p95"]=$p95
    RESULTS["$name.p99"]=$p99

    # Display
    echo ""
    echo "  ${BOLD}Results:${NC}"
    echo "  ┌────────────┬────────────────┐"
    echo "  │ Metric     │ Value          │"
    echo "  ├────────────┼────────────────┤"
    printf "  │ Mean       │ %10d ms  │\n" $mean
    printf "  │ Min        │ %10d ms  │\n" $min
    printf "  │ Max        │ %10d ms  │\n" $max
    printf "  │ P50        │ %10d ms  │\n" $p50
    printf "  │ P95        │ %10d ms  │\n" $p95
    printf "  │ P99        │ %10d ms  │\n" $p99
    printf "  │ Throughput │ %10.1f op/s│\n" $(echo "scale=1; 1000 / $mean" | bc)
    echo "  └────────────┴────────────────┘"
}

benchmark_rpc() {
    log "Benchmarking RPC operations..."

    # Chain ID query
    benchmark "RPC: chain-id" 50 "cast chain-id --rpc-url $RPC_URL"

    # Block number query
    benchmark "RPC: block-number" 50 "cast block-number --rpc-url $RPC_URL"

    # Gas price query
    benchmark "RPC: gas-price" 50 "cast gas-price --rpc-url $RPC_URL"
}

benchmark_contract_calls() {
    log "Benchmarking contract calls..."

    if [ -z "$COORDINATOR_ADDRESS" ]; then
        echo -e "${YELLOW}COORDINATOR_ADDRESS not set, skipping contract benchmarks${NC}"
        return
    fi

    # Read nextModelId
    benchmark "Contract: nextModelId()" 50 \
        "cast call $COORDINATOR_ADDRESS 'nextModelId()(uint256)' --rpc-url $RPC_URL"

    # Read model
    benchmark "Contract: getModel(0)" 50 \
        "cast call $COORDINATOR_ADDRESS 'models(uint256)(string,uint256,uint256,address,bool)' 0 --rpc-url $RPC_URL"
}

benchmark_hashing() {
    log "Benchmarking cryptographic operations..."

    # Keccak256
    benchmark "Crypto: keccak256" 100 "cast keccak 'test message'"

    # Signature (requires private key)
    if [ -n "$PRIVATE_KEY" ]; then
        benchmark "Crypto: sign" 50 \
            "cast wallet sign --private-key $PRIVATE_KEY 'test message'"
    fi
}

benchmark_transactions() {
    log "Benchmarking transaction submission..."

    if [ -z "$COORDINATOR_ADDRESS" ] || [ -z "$PRIVATE_KEY" ]; then
        echo -e "${YELLOW}Skipping transaction benchmarks (need COORDINATOR_ADDRESS and PRIVATE_KEY)${NC}"
        return
    fi

    # Self-transfer (no-op)
    SENDER=$(cast wallet address --private-key $PRIVATE_KEY)

    benchmark "TX: self-transfer" 20 \
        "cast send $SENDER --value 0 --rpc-url $RPC_URL --private-key $PRIVATE_KEY"
}

generate_report() {
    mkdir -p $OUTPUT_DIR
    local report_file="$OUTPUT_DIR/benchmark_$(date +%Y%m%d_%H%M%S).json"

    log "Generating report: $report_file"

    # Build JSON
    echo "{" > $report_file
    echo '  "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'",' >> $report_file
    echo '  "iterations": '$ITERATIONS',' >> $report_file
    echo '  "warmup": '$WARMUP',' >> $report_file
    echo '  "results": {' >> $report_file

    local first=true
    for key in "${!RESULTS[@]}"; do
        local test_name=${key%.*}
        local metric=${key##*.}

        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> $report_file
        fi

        echo -n "    \"$key\": ${RESULTS[$key]}" >> $report_file
    done

    echo "" >> $report_file
    echo "  }" >> $report_file
    echo "}" >> $report_file

    echo ""
    echo -e "${GREEN}Report saved to: $report_file${NC}"
}

# Banner
cat << 'EOF'
╔═══════════════════════════════════════════════════════════╗
║                   HELIX Benchmarks                        ║
╚═══════════════════════════════════════════════════════════╝
EOF

echo -e "\n${CYAN}Configuration:${NC}"
echo "  Iterations:   $ITERATIONS"
echo "  Warmup:       $WARMUP"
echo "  RPC URL:      $RPC_URL"
echo "  Output Dir:   $OUTPUT_DIR"
echo "  Coordinator:  ${COORDINATOR_ADDRESS:-<not set>}"

# Check dependencies
log "Checking dependencies..."
for cmd in cast python3 bc; do
    if ! command -v $cmd &> /dev/null; then
        echo -e "${RED}Error: $cmd not found${NC}"
        exit 1
    fi
done
echo -e "  ${GREEN}✓${NC} All dependencies found"

# Run benchmarks
benchmark_rpc
benchmark_hashing
benchmark_contract_calls
benchmark_transactions

# Generate report
generate_report

# Summary
cat << EOF

${GREEN}╔═══════════════════════════════════════════════════════════╗
║              Benchmarks Complete!                         ║
╚═══════════════════════════════════════════════════════════╝${NC}

Run with different configurations:
  ITERATIONS=500 ./benchmark.sh
  WARMUP=20 ./benchmark.sh
  COORDINATOR_ADDRESS=0x... ./benchmark.sh

EOF
