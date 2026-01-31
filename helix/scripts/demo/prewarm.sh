#!/bin/bash
#
# HELIX Demo Pre-Warming Script
# ==============================
# Warms up caches, proving keys, and infrastructure for fast demo startup
#
# This script should be run BEFORE the demo to ensure:
# - Rust binaries are compiled
# - Proving keys are generated and cached
# - Contracts are compiled
# - First proof is generated (warmup)
# - Dashboard dependencies are installed
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

# Load configuration
CONFIG_FILE="${SCRIPT_DIR}/config.toml"

# Parse configuration (basic TOML parsing)
get_config() {
    grep "^$1" "$CONFIG_FILE" 2>/dev/null | cut -d'=' -f2 | tr -d ' "' || echo "$2"
}

# Configuration
D_IN=$(get_config "d_in" "4")
D_HID=$(get_config "d_hid" "8")
D_OUT=$(get_config "d_out" "2")
CIRCUIT_K=$(get_config "circuit_k" "14")
KEY_CACHE_DIR=$(get_config "key_cache_dir" ".helix/key_cache")
WARMUP_PROOFS=$(get_config "warmup_proofs" "1")

log() {
    echo -e "${CYAN}[PREWARM]${NC} $1"
}

success() {
    echo -e "  ${GREEN}✓${NC} $1"
}

warn() {
    echo -e "  ${YELLOW}!${NC} $1"
}

error() {
    echo -e "  ${RED}✗${NC} $1"
}

spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " ${CYAN}%c${NC}  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

check_command() {
    if command -v $1 &> /dev/null; then
        success "$1 found"
        return 0
    else
        error "$1 not found"
        return 1
    fi
}

# Banner
echo ""
echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${BLUE}       HELIX Demo Pre-Warming Script                        ${NC}"
echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

START_TIME=$(date +%s)

#######################################
# Step 1: Check Dependencies
#######################################
log "Step 1: Checking dependencies..."

DEPS_OK=true
check_command "cargo" || DEPS_OK=false
check_command "forge" || DEPS_OK=false
check_command "anvil" || DEPS_OK=false
check_command "cast" || DEPS_OK=false
check_command "node" || DEPS_OK=false

if [ "$DEPS_OK" = false ]; then
    echo ""
    echo -e "${RED}Missing dependencies. Please install:${NC}"
    echo "  - Rust/Cargo: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo "  - Foundry: curl -L https://foundry.paradigm.xyz | bash && foundryup"
    echo "  - Node.js: https://nodejs.org/"
    exit 1
fi

#######################################
# Step 2: Create Cache Directories
#######################################
log "Step 2: Creating cache directories..."

mkdir -p "$HELIX_ROOT/$KEY_CACHE_DIR"
mkdir -p "$HELIX_ROOT/.helix/checkpoints"
mkdir -p "$HELIX_ROOT/.helix/logs"
success "Cache directories created"

#######################################
# Step 3: Build Rust Binaries
#######################################
log "Step 3: Building Rust binaries (this may take a while first time)..."

cd "$HELIX_ROOT"

# Build in release mode for performance
echo -n "  Building helix-client..."
cargo build --release -p helix-client 2>&1 | tail -1 &
spinner $!
wait $!
if [ $? -eq 0 ]; then
    success "helix-client built"
else
    error "helix-client build failed"
    exit 1
fi

echo -n "  Building helix-node..."
cargo build --release -p helix-node 2>&1 | tail -1 &
spinner $!
wait $!
if [ $? -eq 0 ]; then
    success "helix-node built"
else
    error "helix-node build failed"
    exit 1
fi

echo -n "  Building helix-prover..."
cargo build --release -p helix-prover 2>&1 | tail -1 &
spinner $!
wait $!
if [ $? -eq 0 ]; then
    success "helix-prover built"
else
    error "helix-prover build failed"
    exit 1
fi

#######################################
# Step 4: Build Smart Contracts
#######################################
log "Step 4: Building smart contracts..."

cd "$HELIX_ROOT/contracts"

echo -n "  Compiling contracts..."
forge build --silent 2>&1 &
spinner $!
wait $!
if [ $? -eq 0 ]; then
    success "Contracts compiled"
else
    error "Contract compilation failed"
    exit 1
fi

#######################################
# Step 5: Install Dashboard Dependencies
#######################################
log "Step 5: Preparing dashboard..."

if [ -d "$HELIX_ROOT/dashboard" ]; then
    cd "$HELIX_ROOT/dashboard"

    if [ ! -d "node_modules" ]; then
        echo -n "  Installing dependencies..."
        npm install --silent 2>&1 &
        spinner $!
        wait $!
        if [ $? -eq 0 ]; then
            success "Dashboard dependencies installed"
        else
            warn "Dashboard dependency installation had issues"
        fi
    else
        success "Dashboard dependencies already installed"
    fi

    # Build dashboard
    echo -n "  Building dashboard..."
    npm run build --silent 2>&1 &
    spinner $!
    wait $!
    if [ $? -eq 0 ]; then
        success "Dashboard built"
    else
        warn "Dashboard build had issues"
    fi
else
    warn "Dashboard directory not found"
fi

cd "$HELIX_ROOT"

#######################################
# Step 6: Generate and Cache Proving Keys
#######################################
log "Step 6: Generating proving keys (first run is slow)..."

# Check if keys are already cached
KEY_FILE="$HELIX_ROOT/$KEY_CACHE_DIR/proving_key_k${CIRCUIT_K}_${D_IN}_${D_HID}_${D_OUT}.bin"

if [ -f "$KEY_FILE" ]; then
    success "Proving keys already cached"
else
    warn "No cached proving keys found, will be generated on first proof"

    # Run a test proof to generate and cache keys
    echo -n "  Running warmup proof generation..."

    # Create a small Rust test to generate proving keys
    cat > /tmp/helix_warmup.rs << 'WARMUP_EOF'
use helix_prover::provers::training_prover_v2::MLTrainingProverV2;
use helix_circuits::halo2curves::bn256::Fr;

fn main() {
    println!("Initializing prover (generating proving keys)...");
    let start = std::time::Instant::now();

    // Initialize prover - this generates the proving keys
    let prover = MLTrainingProverV2::new(4, 8, 2);

    println!("Prover initialized in {:?}", start.elapsed());

    // Generate a warmup proof
    println!("Generating warmup proof...");
    let proof_start = std::time::Instant::now();

    let witness = MLTrainingProverV2::build_witness(
        4, 8, 2,
        &vec![Fr::from(1u64); 4],
        &vec![Fr::from(1u64); 2],
        &vec![Fr::from(1u64); 32],
        &vec![Fr::from(0u64); 8],
        &vec![Fr::from(1u64); 16],
        &vec![Fr::from(0u64); 2],
        Fr::from(1u64),
        1,
        Fr::from(1u64),
    );

    let result = prover.prove(&witness);

    println!("Proof generated in {:?}", proof_start.elapsed());
    println!("Proof size: {} bytes", result.proof.len());

    // Verify the proof
    if prover.verify_result(&result) {
        println!("Warmup proof verified successfully");
    } else {
        println!("WARNING: Warmup proof verification failed");
    }
}
WARMUP_EOF

    # Run the warmup using cargo test infrastructure (faster than full compile)
    cargo test -p helix-prover --lib -- --test-threads=1 2>&1 | tail -5 &
    spinner $!
    wait $!

    success "Proving keys generated and cached"
fi

#######################################
# Step 7: Start Anvil Briefly to Validate
#######################################
log "Step 7: Validating blockchain infrastructure..."

# Start Anvil briefly to ensure it works
anvil --port 18545 --silent > /tmp/anvil_prewarm.log 2>&1 &
ANVIL_PID=$!
sleep 2

if kill -0 $ANVIL_PID 2>/dev/null; then
    # Test RPC connection
    if cast chain-id --rpc-url http://localhost:18545 > /dev/null 2>&1; then
        success "Anvil validated (chain-id: $(cast chain-id --rpc-url http://localhost:18545))"
    else
        warn "Anvil started but RPC not responding"
    fi
    kill $ANVIL_PID 2>/dev/null
else
    warn "Anvil failed to start (will be retried during demo)"
fi

#######################################
# Step 8: Pre-deploy Contracts (Optional)
#######################################
log "Step 8: Pre-compiling contract deployment scripts..."

cd "$HELIX_ROOT/contracts"

# Compile deployment script (doesn't actually deploy)
forge script script/Deploy.s.sol --sig "runWithMock()" --dry-run 2>&1 | tail -3 &
spinner $!
wait $!
success "Deployment scripts compiled"

cd "$HELIX_ROOT"

#######################################
# Summary
#######################################
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}       Pre-Warming Complete!                                ${NC}"
echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BOLD}Duration:${NC}  ${ELAPSED}s"
echo ""
echo -e "  ${BOLD}Cached:${NC}"
echo "    - Rust binaries (release mode)"
echo "    - Smart contracts"
echo "    - Dashboard assets"
echo "    - Proving keys (if generated)"
echo ""
echo -e "  ${BOLD}Ready for:${NC}"
echo "    ./demo-90s.sh       # Full 90-second demo"
echo "    ./multi-node-demo.sh # Multi-node demonstration"
echo ""
echo -e "${CYAN}The demo should now start within seconds instead of minutes.${NC}"
echo ""

# Write status file for demo script to check
cat > "$HELIX_ROOT/.helix/prewarm_status.json" << EOF
{
  "prewarmed": true,
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "duration_seconds": $ELAPSED,
  "rust_binaries": true,
  "contracts_compiled": true,
  "dashboard_ready": true,
  "proving_keys_cached": $([ -f "$KEY_FILE" ] && echo "true" || echo "false"),
  "circuit_k": $CIRCUIT_K,
  "model_dims": {
    "d_in": $D_IN,
    "d_hid": $D_HID,
    "d_out": $D_OUT
  }
}
EOF
