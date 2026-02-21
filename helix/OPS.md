# HELIX Operations Guide

How to launch the full stack for demo/testing on ADI testnet.

## Prerequisites

- Rust 1.75+ with `cargo build --release` completed
- Node.js 18+ with `npm install` in `dashboard/`
- Foundry (`forge`, `cast`) installed
- `.env.adi` configured with `TESTNET_PRIVATE_KEY`

## Current Contract Addresses (ADI Testnet, Chain 99999)

| Contract | Address |
|----------|---------|
| Coordinator V4 | `0x0b4a2E27dF67f5D8e90405C76817689B4F048c19` |
| ModelStore | `0xB891b392F9d290c2CfEBe8dE2d20d8E51870A7fC` |
| Verifier | `0x76e537b6AD41ad670E126689C7CB4488ebd0598a` |
| Token | `0x73E04552246D905795EBC18F3068a2528f8588a1` |

## Worker Addresses

| Worker | Address |
|--------|---------|
| 0 | `0x706f366023280A8AE127e59E5bf357D4ECfB8A75` |
| 1 | `0x91c3CeaDC59dD5CfB844369fdC60700e48e5c0Ef` |
| 2 | `0xed5BE53a719bBc1645C7AB587dAf06efc4765681` |
| 3 | `0xfc339dF8431a734c07cB67900cB600a4a607cE62` |
| 4 | `0xa6719C6E6c5bb75A36D1b5fDC007Ffc15F779040` |
| 5 | `0x26Fb70f335acc034CE450A90380bbe7e6169444b` |

Owner: `0x64CEE815fA7FDb863f59184EE9266FD6276ab71d`

## Quick Start (copy-paste these 3 commands in 3 terminals)

### Terminal 1 — Backend API

```bash
cd helix
TESTNET_PRIVATE_KEY=0xYOUR_PRIVATE_KEY \
  ./target/release/helix dashboard \
  --port 3001 --host 0.0.0.0 \
  --rpc-url https://rpc.ab.testnet.adifoundation.ai/ \
  --coordinator 0x0b4a2E27dF67f5D8e90405C76817689B4F048c19 \
  --model-store 0xB891b392F9d290c2CfEBe8dE2d20d8E51870A7fC
```

### Terminal 2 — 6 Workers

```bash
cd helix
TESTNET_PRIVATE_KEY=0xYOUR_PRIVATE_KEY \
  ./target/release/helix spawn-workers \
  --count 6 --base-port 9001 --bind 0.0.0.0 --seed 42 \
  --api-url http://localhost:3001 \
  --rpc-url https://rpc.ab.testnet.adifoundation.ai/ \
  --coordinator 0x0b4a2E27dF67f5D8e90405C76817689B4F048c19
```

The 6 real ADI testnet worker keys are baked into the binary as defaults — no `--private-keys` flag needed.

### Terminal 3 — Frontend

```bash
cd helix/dashboard
npm run build && npx next start --hostname 0.0.0.0 --port 3000
```

**Important:** Use `npm run build && npx next start` (production mode), NOT `npm run dev`. The dev server hangs under load.

## Accessing from Another Device on LAN

1. Find this machine's IP: `ifconfig | grep "inet " | grep -v 127.0.0.1`
2. On the other device, open: `http://<THIS_MACHINE_IP>:3000`
3. The frontend `.env.local` must have the LAN IP for API calls:
   ```
   NEXT_PUBLIC_API_URL=http://<THIS_MACHINE_IP>:3001
   NEXT_PUBLIC_WS_URL=ws://<THIS_MACHINE_IP>:3001/ws
   ```
4. After changing `.env.local`, rebuild: `npm run build`

Current LAN IP: `192.168.68.92` (ethernet)

## Top Up Worker Balances

Workers need ADI for gas (staking, registration). Run this before training if workers are low:

```bash
cd helix
./scripts/topup-workers.sh         # Default 0.5 ADI per worker
./scripts/topup-workers.sh 1.0     # Custom amount per worker
```

Skips workers that already have enough. Check balances manually:

```bash
for addr in 0x706f366023280A8AE127e59E5bf357D4ECfB8A75 0x91c3CeaDC59dD5CfB844369fdC60700e48e5c0Ef 0xed5BE53a719bBc1645C7AB587dAf06efc4765681 0xfc339dF8431a734c07cB67900cB600a4a607cE62 0xa6719C6E6c5bb75A36D1b5fDC007Ffc15F779040 0x26Fb70f335acc034CE450A90380bbe7e6169444b; do
  echo "$addr: $(cast from-wei $(cast balance $addr --rpc-url https://rpc.ab.testnet.adifoundation.ai/)) ADI"
done
```

## Gas Economics (ADI Testnet)

ADI testnet gas is ~550 gwei. Key costs per training job:
- Worker `stakeAndJoin`: ~0.067 ADI (0.001 stake + 0.066 gas)
- Owner checkpoint submission: ~0.044 ADI per checkpoint
- Owner `completeTraining`: ~0.15 ADI

Workers get paid from the job payment when `completeTraining` is called. Make sure the payment covers worker gas + profit. With 4 workers and 10 checkpoints, set payment to at least 1.0 ADI.

## Kill Everything

```bash
pkill -9 -f "mpc-worker"
pkill -9 -f "spawn-workers"
pkill -9 -f "helix dashboard"
pkill -9 -f "next start"
```

## Verify Everything is Running

```bash
# Backend
curl http://localhost:3001/

# Workers
curl http://localhost:3001/api/workers | python3 -m json.tool

# Frontend
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/dashboard
```

## Redeploying Contracts

Only do this if contract source code changed. Uses ~2 ADI per contract.

```bash
cd helix
source .env.adi

# Check balance first
cast balance $(cast wallet address $TESTNET_PRIVATE_KEY) --rpc-url https://rpc.ab.testnet.adifoundation.ai/ | xargs cast from-wei

# Deploy (only what changed)
cd contracts

# Coordinator V4
forge create src/core/HelixCoordinatorV4.sol:HelixCoordinatorV4 \
  --rpc-url https://rpc.ab.testnet.adifoundation.ai/ \
  --private-key $TESTNET_PRIVATE_KEY --legacy --broadcast \
  --constructor-args $(cast wallet address $TESTNET_PRIVATE_KEY) $VERIFIER_ADDRESS

# ModelStore
forge create src/core/HelixModelStore.sol:HelixModelStore \
  --rpc-url https://rpc.ab.testnet.adifoundation.ai/ \
  --private-key $TESTNET_PRIVATE_KEY --legacy --broadcast

# Token (only if changed)
forge create src/token/HelixToken.sol:HelixToken \
  --rpc-url https://rpc.ab.testnet.adifoundation.ai/ \
  --private-key $TESTNET_PRIVATE_KEY --legacy --broadcast \
  --constructor-args $(cast wallet address $TESTNET_PRIVATE_KEY)
```

After deploying, update addresses in:
- `helix/.env.adi`
- `helix/dashboard/.env.local`
- The Terminal 1 and 2 commands above

## Rebuilding

```bash
# Rust (after code changes)
cd helix && cargo build --release

# Contracts (after solidity changes)
cd helix/contracts && forge build

# Frontend (after dashboard changes)
cd helix/dashboard && npm run build
```
