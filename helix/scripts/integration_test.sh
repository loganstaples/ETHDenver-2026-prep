#!/bin/bash
set -e

echo "Starting Integration Test..."

# 1. Start Anvil
echo "Booting Anvil..."
anvil --port 8545 > anvil_log.txt 2>&1 &
ANVIL_PID=$!
sleep 3 # Wait for Anvil to be ready

Cleanup() {
    echo "Stopping Anvil (PID $ANVIL_PID)..."
    kill $ANVIL_PID
}
trap Cleanup EXIT

# 2. Deploy Contracts
echo "Deploying Contracts..."
# Default Anvil Private Key #0
export PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
export RPC_URL=http://localhost:8545

# Deploy MockVerifier
cd contracts
deploy_log="$(pwd)/../deploy.log"

echo "Deploying MockVerifier..."
forge create src/mocks/MockVerifier.sol:MockVerifier --rpc-url $RPC_URL --private-key $PRIVATE_KEY --broadcast > $deploy_log 2>&1
VERIFIER_ADDRESS=$(cat $deploy_log | grep "Deployed to:" | awk '{print $3}')
if [ -z "$VERIFIER_ADDRESS" ]; then
    echo "Failed to deploy MockVerifier"
    cat $deploy_log
    exit 1
fi
echo "MockVerifier deployed at: $VERIFIER_ADDRESS"

# Deploy HelixCoordinator
echo "Deploying HelixCoordinator..."
# Pass VERIFIER_ADDRESS to constructor
forge create --rpc-url $RPC_URL --private-key $PRIVATE_KEY --broadcast src/core/HelixCoordinator.sol:HelixCoordinator --constructor-args $VERIFIER_ADDRESS --json > $deploy_log 2>&1

cd ..

echo "Deploy finished. Log content (head):"
head -n 20 $deploy_log

# Use grep to extract address from standard output
# "Deployed to: 0x..."
COORDINATOR_ADDRESS=$(cat $deploy_log | grep "Deployed to:" | awk '{print $3}')

if [ -z "$COORDINATOR_ADDRESS" ]; then
    echo "Failed to extract address. Full log:"
    cat $deploy_log
    echo "Failed to deploy contract"
    exit 1
fi

echo "HelixCoordinator deployed at: $COORDINATOR_ADDRESS"

# 2.5 Initialize State (Register Model & Start Round)
echo "Initializing State..."
# Register Model (ipfsHash="Qm...", initialCommitment=0)
cast send $COORDINATOR_ADDRESS "registerModel(string,uint256)" "QmTest" 0 --rpc-url $RPC_URL --private-key $PRIVATE_KEY
# Start Round for Model 0
cast send $COORDINATOR_ADDRESS "startRound(uint256)" 0 --rpc-url $RPC_URL --private-key $PRIVATE_KEY

# 3. Running Training Node
echo "Running Helix Node (One-Shot Mode)..."
export COORDINATOR_ADDRESS=$COORDINATOR_ADDRESS
export RUST_LOG=info

# We expect it to succeed and exit
cargo run -p helix-node -- --one-shot

echo "Integration Test Passed: Node successfully submitted update."
