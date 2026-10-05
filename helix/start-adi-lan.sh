#!/usr/bin/env bash
# ============================================================================
# HELIX Demo — ADI Testnet + LAN (Two-Laptop ETHDenver Setup)
# ============================================================================
# Laptop 1 (this machine): Runs dashboard backend, 6 MPC workers, frontend
# Laptop 2 (other machine): Opens http://<LAN-IP>:3000 in browser
#
# Usage:
#   ./start-adi-lan.sh          # Start everything
#   ./start-adi-lan.sh stop     # Kill all background processes
#
# Prerequisites:
#   1. Contracts deployed on ADI testnet (./scripts/deploy-adi.sh)
#   2. .env.adi has TESTNET_PRIVATE_KEY and contract addresses
#   3. Workers funded (./scripts/topup-workers.sh)
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
LOGS_DIR="$SCRIPT_DIR/logs"
PID_FILE="$LOGS_DIR/pids"

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
        pkill -f "helix dashboard" 2>/dev/null || true
        pkill -f "helix spawn-workers" 2>/dev/null || true
        pkill -f "next-server" 2>/dev/null || true
    fi
    echo "Done."
    exit 0
fi

# ── Load ADI testnet config ──
ENV_FILE="$SCRIPT_DIR/.env.adi"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: $ENV_FILE not found. Run ./scripts/deploy-adi.sh first."
    exit 1
fi
source "$ENV_FILE"

RPC_URL="${ADI_RPC_URL:-https://rpc.ab.testnet.adifoundation.ai/}"
PRIVATE_KEY="${TESTNET_PRIVATE_KEY:-}"
COORDINATOR="${COORDINATOR_ADDRESS:-}"
MODEL_STORE="${MODEL_STORE_ADDRESS:-}"

if [[ -z "$PRIVATE_KEY" || "$PRIVATE_KEY" == "0xYOUR_PRIVATE_KEY_HERE" ]]; then
    echo "Error: Set TESTNET_PRIVATE_KEY in .env.adi"
    exit 1
fi
if [[ -z "$COORDINATOR" ]]; then
    echo "Error: Set COORDINATOR_ADDRESS in .env.adi (run ./scripts/deploy-adi.sh)"
    exit 1
fi

mkdir -p "$LOGS_DIR"
> "$PID_FILE"

# ── Detect LAN IP (prefer wired ethernet) ──
LAN_IP=""
DEFAULT_IF=$(route -n get default 2>/dev/null | awk '/interface:/{print $2}')
if [[ -n "$DEFAULT_IF" ]]; then
    LAN_IP=$(ipconfig getifaddr "$DEFAULT_IF" 2>/dev/null || echo "")
fi
if [[ -z "$LAN_IP" ]]; then
    LAN_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}' || echo "")
fi
if [[ -z "$LAN_IP" ]]; then
    echo "WARNING: No LAN IP detected. Make sure you're connected to the ethernet switch."
    echo "Falling back to localhost."
    LAN_IP="127.0.0.1"
fi

BROWSER_HOST="$LAN_IP"
if [[ "$LAN_IP" == "127.0.0.1" ]]; then
    BROWSER_HOST="localhost"
fi

# ── Clean up any stale processes from previous runs ──
for port in 3001 9001 9004 9007 9010 9013 9016 3000; do
    pid=$(lsof -ti :"$port" 2>/dev/null || true)
    if [[ -n "$pid" ]]; then
        kill "$pid" 2>/dev/null || true
        echo "Killed stale process on port $port (PID $pid)"
    fi
done
sleep 1

# ── Verify ADI testnet connectivity ──
OWNER_ADDR=$(cast wallet address "$PRIVATE_KEY" 2>/dev/null)
BALANCE=$(cast balance "$OWNER_ADDR" --rpc-url "$RPC_URL" 2>/dev/null || echo "FAIL")
if [[ "$BALANCE" == "FAIL" ]]; then
    echo "Error: Cannot connect to ADI testnet at $RPC_URL"
    exit 1
fi
BALANCE_ETH=$(cast from-wei "$BALANCE" 2>/dev/null || echo "?")

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       HELIX Demo — ADI Testnet + LAN (ETHDenver)           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  LAN IP:      $LAN_IP"
echo "  Chain:       ADI Testnet (chain ID ${ADI_CHAIN_ID:-99999})"
echo "  RPC:         $RPC_URL"
echo "  Owner:       $OWNER_ADDR"
echo "  Balance:     $BALANCE_ETH ADI"
echo "  Coordinator: $COORDINATOR"
echo "  ModelStore:  $MODEL_STORE"
echo

# ── 1. Update dashboard .env.local for ADI testnet + LAN ──
echo "▸ Updating dashboard/.env.local for ADI testnet + LAN..."
cat > dashboard/.env.local <<EOF
NEXT_PUBLIC_API_URL=http://${BROWSER_HOST}:3001
NEXT_PUBLIC_WS_URL=ws://${BROWSER_HOST}:3001/ws
NEXT_PUBLIC_ENABLE_TESTNETS=true
NEXT_PUBLIC_DEFAULT_CHAIN_ID=${ADI_CHAIN_ID:-99999}
NEXT_PUBLIC_ENABLE_DEMO_MODE=false
NEXT_PUBLIC_ENABLE_MOCK_DATA=false
NEXT_PUBLIC_WALLETCONNECT_PROJECT_ID=placeholder
NEXT_TELEMETRY_DISABLED=1
ZG_PRIVATE_KEY=${PRIVATE_KEY#0x}

# ADI Testnet contract addresses (from .env.adi)
NEXT_PUBLIC_COORDINATOR_ADDRESS=$COORDINATOR
NEXT_PUBLIC_MODEL_STORE_ADDRESS=$MODEL_STORE
NEXT_PUBLIC_ETH_RPC_URL=$RPC_URL
EOF
echo "  ✓ .env.local updated (ADI testnet, browser host: $BROWSER_HOST)"

# ── 2. Build Rust backend (release) ──
HELIX_BIN="target/release/helix"
if [[ ! -f "$HELIX_BIN" ]]; then
    echo "▸ Building helix-client (release)... this may take a few minutes"
    cargo build -p helix-client --release 2>"$LOGS_DIR/build.log"
    echo "  ✓ Build complete"
else
    echo "▸ Using existing release binary: $HELIX_BIN"
fi

# ── 3. Start Dashboard Backend ──
echo "▸ Starting dashboard backend (port 3001, ADI testnet)..."
export TESTNET_PRIVATE_KEY="$PRIVATE_KEY"
"$HELIX_BIN" -v dashboard \
    --port 3001 \
    --host 0.0.0.0 \
    --cors \
    --rpc-url "$RPC_URL" \
    --coordinator "$COORDINATOR" \
    --model-store "$MODEL_STORE" \
    > "$LOGS_DIR/dashboard.log" 2>&1 &
DASHBOARD_PID=$!
echo "$DASHBOARD_PID dashboard" >> "$PID_FILE"
sleep 2

if ! kill -0 "$DASHBOARD_PID" 2>/dev/null; then
    echo "  ✗ Dashboard failed to start. Check $LOGS_DIR/dashboard.log"
    tail -20 "$LOGS_DIR/dashboard.log"
    exit 1
fi
echo "  ✓ Dashboard backend running (PID $DASHBOARD_PID)"

# ── 4. Spawn 6 MPC Workers (with ADI testnet keys — hardcoded defaults) ──
echo "▸ Spawning 6 MPC workers (ports 9001, 9004, 9007, 9010, 9013, 9016)..."
"$HELIX_BIN" -v spawn-workers \
    --count 6 \
    --base-port 9001 \
    --bind 0.0.0.0 \
    --seed 42 \
    --api-url http://localhost:3001 \
    --rpc-url "$RPC_URL" \
    --coordinator "$COORDINATOR" \
    > "$LOGS_DIR/workers.log" 2>&1 &
WORKERS_PID=$!
echo "$WORKERS_PID workers" >> "$PID_FILE"
sleep 3

if ! kill -0 "$WORKERS_PID" 2>/dev/null; then
    echo "  ✗ Workers failed to start. Check $LOGS_DIR/workers.log"
    tail -20 "$LOGS_DIR/workers.log"
    exit 1
fi
echo "  ✓ Workers running (PID $WORKERS_PID)"

# ── 5. Build & Start Frontend (production mode for stability) ──
echo "▸ Building frontend dashboard..."
(cd dashboard && npm run build > "$LOGS_DIR/frontend-build.log" 2>&1)
if [[ $? -ne 0 ]]; then
    echo "  ✗ Frontend build failed. Check $LOGS_DIR/frontend-build.log"
    tail -20 "$LOGS_DIR/frontend-build.log"
    exit 1
fi
echo "  ✓ Build complete"
echo "▸ Starting frontend dashboard (port 3000)..."
(cd dashboard && npm run start > "$LOGS_DIR/frontend.log" 2>&1) &
FRONTEND_PID=$!
echo "$FRONTEND_PID frontend" >> "$PID_FILE"
sleep 3
echo "  ✓ Frontend running (PID $FRONTEND_PID)"

echo
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    ALL SERVICES RUNNING                     ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                             ║"
printf "║  Frontend:    http://%-40s ║\n" "$BROWSER_HOST:3000"
printf "║  Backend API: http://%-40s ║\n" "$BROWSER_HOST:3001"
printf "║  WebSocket:   ws://%-42s ║\n" "$BROWSER_HOST:3001/ws"
echo "║                                                             ║"
echo "║  Chain:       ADI Testnet (ID ${ADI_CHAIN_ID:-99999})                        ║"
printf "║  RPC:         %-48s ║\n" "$RPC_URL"
echo "║                                                             ║"
echo "║  Coordinator: $COORDINATOR  ║"
echo "║  ModelStore:  $MODEL_STORE  ║"
echo "║                                                             ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Logs: ./logs/{dashboard,workers,frontend}.log              ║"
echo "║  Stop: ./start-adi-lan.sh stop                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo
echo "Laptop 2 setup:"
echo "  1. Connect to same ethernet switch"
echo "  2. Open http://$BROWSER_HOST:3000 in browser"
echo "  3. Add MetaMask network: RPC=$RPC_URL, Chain ID=${ADI_CHAIN_ID:-99999}, Symbol=ADI"
echo "  4. Import account with key: $PRIVATE_KEY"
echo "  5. Go to http://$BROWSER_HOST:3000/train and start training!"
echo
echo "Tail logs:"
echo "  tail -f logs/dashboard.log   # Backend"
echo "  tail -f logs/workers.log     # MPC workers"
echo "  tail -f logs/frontend.log    # Next.js"
