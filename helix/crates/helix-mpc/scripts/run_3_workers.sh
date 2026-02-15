#!/usr/bin/env bash
# Launches 3 MPC worker processes on localhost for a demo training session.
#
# Usage:
#   ./scripts/run_3_workers.sh [--steps N] [--config path/to/config.json]
#
# Each worker binds to a different port and connects to the other two.
# Results are written to /tmp/helix-worker-{0,1,2}.json.

set -euo pipefail

STEPS="${1:-20}"
CONFIG_FLAG=""
if [ -n "${2:-}" ]; then
    CONFIG_FLAG="--config $2"
fi

PORT0=19000
PORT1=19001
PORT2=19002

echo "=== HELIX MPC Training: 3-Worker Demo ==="
echo "Ports: $PORT0, $PORT1, $PORT2"
echo "Steps: $STEPS"
echo ""

# Build the worker binary.
echo "Building mpc-worker..."
cargo build -p helix-mpc --features network-mpc --bin mpc-worker --release 2>&1 | tail -1

WORKER="cargo run -p helix-mpc --features network-mpc --bin mpc-worker --release --"

# Launch all 3 workers in parallel.
echo ""
echo "Launching workers..."

$WORKER \
    --party 0 --bind "127.0.0.1:$PORT0" \
    --peer "1=127.0.0.1:$PORT1" --peer "2=127.0.0.1:$PORT2" \
    --steps "$STEPS" --output /tmp/helix-worker-0.json \
    $CONFIG_FLAG &
PID0=$!

$WORKER \
    --party 1 --bind "127.0.0.1:$PORT1" \
    --peer "0=127.0.0.1:$PORT0" --peer "2=127.0.0.1:$PORT2" \
    --steps "$STEPS" --output /tmp/helix-worker-1.json \
    $CONFIG_FLAG &
PID1=$!

$WORKER \
    --party 2 --bind "127.0.0.1:$PORT2" \
    --peer "0=127.0.0.1:$PORT0" --peer "1=127.0.0.1:$PORT1" \
    --steps "$STEPS" --output /tmp/helix-worker-2.json \
    $CONFIG_FLAG &
PID2=$!

echo "Workers launched: PIDs $PID0, $PID1, $PID2"
echo "Waiting for completion..."

# Wait for all workers.
FAILED=0
wait $PID0 || FAILED=1
wait $PID1 || FAILED=1
wait $PID2 || FAILED=1

echo ""
if [ $FAILED -eq 0 ]; then
    echo "=== All workers completed successfully ==="
    echo ""
    echo "Results:"
    for i in 0 1 2; do
        if [ -f "/tmp/helix-worker-$i.json" ]; then
            echo "  Worker $i: $(cat /tmp/helix-worker-$i.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f\"steps={d['steps_completed']}, loss={d['final_loss']:.6f}, mac_checks={d['mac_checks_passed']}, time={d['training_time_ms']}ms\")
" 2>/dev/null || echo "(see /tmp/helix-worker-$i.json)")"
        fi
    done
else
    echo "=== Some workers FAILED ==="
    exit 1
fi
