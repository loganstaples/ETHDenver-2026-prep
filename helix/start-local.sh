#!/usr/bin/env bash
# ============================================================================
# HELIX Local E2E Test — starts Anvil, deploys contracts, launches backend +
# workers, and starts the frontend dashboard.
#
# Usage:
#   ./start-local.sh                    # Single machine, browser on localhost
#   ./start-local.sh --lan              # Host everything here, browser from LAN
#   ./start-local.sh --remote-workers   # Dashboard + Anvil here, workers on LAN
#   ./start-local.sh stop               # Kill all background processes
#
# Processes run in the background and log to ./logs/
# ============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
LOGS_DIR="$SCRIPT_DIR/logs"
PID_FILE="$LOGS_DIR/pids"

# ── Parse flags ──
REMOTE_WORKERS=false
LAN_MODE=false
for arg in "$@"; do
    case "$arg" in
        --remote-workers) REMOTE_WORKERS=true; LAN_MODE=true ;;
        --lan)            LAN_MODE=true ;;
        stop) ;;  # handled below
    esac
done

# ── Stop mode ──
if [[ "${1:-}" == "stop" ]]; then
    if [[ -f "$PID_FILE" ]]; then
        echo "Stopping HELIX processes..."
        while read -r pid name; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null && echo "  Stopped $name (PID $pid)" || true
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    else
        echo "No PID file found. Killing by name..."
        pkill -f "anvil --chain-id 31337" 2>/dev/null || true
        pkill -f "helix dashboard" 2>/dev/null || true
        pkill -f "helix spawn-workers" 2>/dev/null || true
    fi
    echo "Done."
    exit 0
fi

mkdir -p "$LOGS_DIR"
> "$PID_FILE"

# ── Detect LAN IP ──
# Use the default route interface (prefers wired ethernet over WiFi when cable is plugged in)
DEFAULT_IF=$(route -n get default 2>/dev/null | awk '/interface:/{print $2}')
if [[ -n "$DEFAULT_IF" ]]; then
    LAN_IP=$(ipconfig getifaddr "$DEFAULT_IF" 2>/dev/null || echo "")
fi
# Fallback: try any non-loopback interface
if [[ -z "$LAN_IP" ]]; then
    LAN_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}' || echo "")
fi
if [[ -z "$LAN_IP" ]]; then
    LAN_IP="127.0.0.1"
fi

if [[ "$LAN_MODE" == "true" && "$LAN_IP" == "127.0.0.1" ]]; then
    echo "WARNING: --lan/--remote-workers used but no LAN IP detected."
    echo "         Make sure you're connected to a network."
    echo "         Falling back to 127.0.0.1 (localhost only)."
    echo
fi

# Determine which host to use in browser-facing URLs
if [[ "$LAN_MODE" == "true" && "$LAN_IP" != "127.0.0.1" ]]; then
    BROWSER_HOST="$LAN_IP"
else
    BROWSER_HOST="localhost"
fi

# ── Clean up any stale processes from previous runs ──
for port in 8545 3001 9001 9004 9007 9010 9013 9016 3000; do
    pid=$(lsof -ti :"$port" 2>/dev/null || true)
    if [[ -n "$pid" ]]; then
        kill "$pid" 2>/dev/null || true
        echo "Killed stale process on port $port (PID $pid)"
    fi
done
sleep 1

# Anvil default deployer key (account 0)
DEPLOYER_KEY="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
RPC_URL="http://127.0.0.1:8545"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║            HELIX Local E2E Test Environment                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
if [[ "$LAN_IP" != "127.0.0.1" ]]; then
    echo "  LAN IP: $LAN_IP"
fi
if [[ "$REMOTE_WORKERS" == "true" ]]; then
    echo "  Mode:   Remote workers (workers run on another machine)"
elif [[ "$LAN_MODE" == "true" ]]; then
    echo "  Mode:   LAN (everything here, browser accessible from $LAN_IP)"
else
    echo "  Mode:   All-local (everything on this machine)"
fi
echo

# ── 1. Start Anvil ──
echo "▸ Starting Anvil (chain 31337)..."
anvil --chain-id 31337 --block-time 1 --accounts 10 --balance 10000 \
    --host 0.0.0.0 \
    > "$LOGS_DIR/anvil.log" 2>&1 &
ANVIL_PID=$!
echo "$ANVIL_PID anvil" >> "$PID_FILE"
sleep 2

if ! kill -0 "$ANVIL_PID" 2>/dev/null; then
    echo "  ✗ Anvil failed to start. Check $LOGS_DIR/anvil.log"
    exit 1
fi
echo "  ✓ Anvil running (PID $ANVIL_PID)"

# ── 2. Deploy V4 Coordinator ──
echo "▸ Deploying HelixCoordinatorV4 + Halo2Verifier..."
V4_LOG="$LOGS_DIR/deploy-v4.log"
(cd contracts && PRIVATE_KEY="$DEPLOYER_KEY" forge script script/DeployV4.s.sol \
    --rpc-url "$RPC_URL" --broadcast > "$V4_LOG" 2>&1)
# Extract the two 0x addresses from == Logs == section
V4_ADDRESSES=$(grep "^  0x" "$V4_LOG" | sed 's/^  //')
COORDINATOR_ADDRESS=$(echo "$V4_ADDRESSES" | head -1)
VERIFIER_ADDRESS=$(echo "$V4_ADDRESSES" | tail -1)
if [[ -z "$COORDINATOR_ADDRESS" ]]; then
    echo "  ✗ Deploy failed. Check $V4_LOG"
    cat "$V4_LOG" | tail -20
    exit 1
fi
echo "  ✓ Coordinator: $COORDINATOR_ADDRESS"
echo "  ✓ Verifier:    $VERIFIER_ADDRESS"

# ── 3. Deploy ModelStore ──
echo "▸ Deploying HelixModelStore (ERC-721)..."
MS_LOG="$LOGS_DIR/deploy-modelstore.log"
(cd contracts && PRIVATE_KEY="$DEPLOYER_KEY" forge script script/DeployModelStore.s.sol \
    --rpc-url "$RPC_URL" --broadcast > "$MS_LOG" 2>&1)
MODEL_STORE_ADDRESS=$(grep "deployed at:" "$MS_LOG" | grep -o "0x[0-9a-fA-F]*" || true)
if [[ -z "$MODEL_STORE_ADDRESS" ]]; then
    echo "  ✗ Deploy failed. Check $MS_LOG"
    cat "$MS_LOG" | tail -20
    exit 1
fi
echo "  ✓ ModelStore:  $MODEL_STORE_ADDRESS"
echo

# ── 4. Update dashboard .env.local for local chain ──
# NEXT_PUBLIC_* vars are baked into the JS bundle and run in the browser.
# In LAN mode, use the LAN IP so browsers on other machines can reach the backend.
echo "▸ Updating dashboard/.env.local for local chain..."
cat > dashboard/.env.local <<EOF
NEXT_PUBLIC_API_URL=http://${BROWSER_HOST}:3001
NEXT_PUBLIC_WS_URL=ws://${BROWSER_HOST}:3001/ws
NEXT_PUBLIC_ENABLE_TESTNETS=true
NEXT_PUBLIC_DEFAULT_CHAIN_ID=31337
NEXT_PUBLIC_ENABLE_DEMO_MODE=false
NEXT_PUBLIC_ENABLE_MOCK_DATA=false
NEXT_PUBLIC_WALLETCONNECT_PROJECT_ID=placeholder
NEXT_TELEMETRY_DISABLED=1
ZG_PRIVATE_KEY=${ZG_PRIVATE_KEY:-}

# Local Anvil contract addresses (auto-generated by start-local.sh)
NEXT_PUBLIC_COORDINATOR_ADDRESS=$COORDINATOR_ADDRESS
NEXT_PUBLIC_MODEL_STORE_ADDRESS=$MODEL_STORE_ADDRESS
NEXT_PUBLIC_ETH_RPC_URL=http://${BROWSER_HOST}:8545
EOF
echo "  ✓ .env.local updated (chain 31337, browser host: $BROWSER_HOST)"

# ── 5. Build Rust backend (skip if already built) ──
HELIX_BIN="target/release/helix"
if [[ ! -f "$HELIX_BIN" ]]; then
    echo "▸ Building helix-client (release)... this may take a few minutes"
    cargo build -p helix-client --release 2>"$LOGS_DIR/build.log"
    echo "  ✓ Build complete"
else
    echo "▸ Using existing release binary: $HELIX_BIN"
fi

# ── 6. Start Dashboard Backend ──
echo "▸ Starting dashboard backend (port 3001)..."
"$HELIX_BIN" -v dashboard \
    --port 3001 \
    --host 0.0.0.0 \
    --cors \
    --rpc-url "$RPC_URL" \
    --coordinator "$COORDINATOR_ADDRESS" \
    --model-store "$MODEL_STORE_ADDRESS" \
    > "$LOGS_DIR/dashboard.log" 2>&1 &
DASHBOARD_PID=$!
echo "$DASHBOARD_PID dashboard" >> "$PID_FILE"
sleep 2

if ! kill -0 "$DASHBOARD_PID" 2>/dev/null; then
    echo "  ✗ Dashboard failed to start. Check $LOGS_DIR/dashboard.log"
    cat "$LOGS_DIR/dashboard.log" | tail -20
    exit 1
fi
echo "  ✓ Dashboard backend running (PID $DASHBOARD_PID)"

# ── 7. Spawn MPC Workers ──
if [[ "$REMOTE_WORKERS" == "true" ]]; then
    echo "▸ Skipping local workers (--remote-workers mode)"
else
    echo "▸ Spawning 6 MPC workers (ports 9001, 9004, 9007, 9010, 9013, 9016)..."
    "$HELIX_BIN" -v spawn-workers \
        --count 6 \
        --base-port 9001 \
        --api-url http://localhost:3001 \
        --rpc-url "$RPC_URL" \
        --coordinator "$COORDINATOR_ADDRESS" \
        > "$LOGS_DIR/workers.log" 2>&1 &
    WORKERS_PID=$!
    echo "$WORKERS_PID workers" >> "$PID_FILE"
    sleep 3

    if ! kill -0 "$WORKERS_PID" 2>/dev/null; then
        echo "  ✗ Workers failed to start. Check $LOGS_DIR/workers.log"
        cat "$LOGS_DIR/workers.log" | tail -20
        exit 1
    fi
    echo "  ✓ Workers running (PID $WORKERS_PID)"
fi

# ── 8. Start Frontend ──
echo "▸ Starting frontend dashboard (port 3000)..."
(cd dashboard && npm run dev > "$LOGS_DIR/frontend.log" 2>&1) &
FRONTEND_PID=$!
echo "$FRONTEND_PID frontend" >> "$PID_FILE"
sleep 3
echo "  ✓ Frontend starting (PID $FRONTEND_PID)"

echo
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    ALL SERVICES RUNNING                     ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                             ║"
if [[ "$LAN_MODE" == "true" && "$LAN_IP" != "127.0.0.1" ]]; then
printf "║  Frontend:    http://%-40s ║\n" "$LAN_IP:3000"
printf "║  Backend API: http://%-40s ║\n" "$LAN_IP:3001"
printf "║  WebSocket:   ws://%-42s ║\n" "$LAN_IP:3001/ws"
printf "║  Anvil RPC:   http://%-40s ║\n" "$LAN_IP:8545"
else
echo "║  Frontend:    http://localhost:3000                         ║"
echo "║  Backend API: http://localhost:3001                         ║"
echo "║  WebSocket:   ws://localhost:3001/ws                        ║"
echo "║  Anvil RPC:   http://localhost:8545                         ║"
fi
echo "║                                                             ║"
echo "║  Coordinator: $COORDINATOR_ADDRESS  ║"
echo "║  ModelStore:  $MODEL_STORE_ADDRESS  ║"
echo "║  Verifier:    $VERIFIER_ADDRESS  ║"
echo "║                                                             ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Logs: ./logs/{anvil,dashboard,workers,frontend}.log        ║"
echo "║  Stop: ./start-local.sh stop                                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo
echo "MetaMask setup:"
if [[ "$LAN_MODE" == "true" && "$LAN_IP" != "127.0.0.1" ]]; then
echo "  1. Add network: RPC=http://$LAN_IP:8545, Chain ID=31337, Symbol=ETH"
else
echo "  1. Add network: RPC=http://localhost:8545, Chain ID=31337, Symbol=ETH"
fi
echo "  2. Import account with key: $DEPLOYER_KEY"
if [[ "$LAN_MODE" == "true" && "$LAN_IP" != "127.0.0.1" ]]; then
echo "  3. Go to http://$LAN_IP:3000/train and start training!"
else
echo "  3. Go to http://localhost:3000/train and start training!"
fi
echo

if [[ "$REMOTE_WORKERS" == "true" && "$LAN_IP" != "127.0.0.1" ]]; then
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                  REMOTE WORKER SETUP                        ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                             ║"
echo "║  Run this on the other machine to start workers:            ║"
echo "║                                                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo
echo "  $HELIX_BIN -v spawn-workers \\"
echo "      --count 6 \\"
echo "      --base-port 9001 \\"
echo "      --api-url http://$LAN_IP:3001 \\"
echo "      --rpc-url http://$LAN_IP:8545 \\"
echo "      --coordinator $COORDINATOR_ADDRESS"
echo
echo "  (Make sure the helix binary is built on that machine too)"
echo
fi

echo "Tail logs:"
echo "  tail -f logs/dashboard.log   # Backend"
if [[ "$REMOTE_WORKERS" != "true" ]]; then
echo "  tail -f logs/workers.log     # MPC workers"
fi
echo "  tail -f logs/anvil.log       # Blockchain"
echo "  tail -f logs/frontend.log    # Next.js"
