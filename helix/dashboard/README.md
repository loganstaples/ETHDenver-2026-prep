# HELIX Dashboard

Next.js frontend for visualizing live training with proofs.

## Features

- **Training Progress** - Loss curves, metrics, real-time updates
- **Status Monitor** - Real-time training and network status page
- **Proof Explorer** - View any training step's ZK proof
- **Error Bounds Visualization** - See bounds tightening over training
- **Model Chat** - Query the trained model
- **Node Network** - Geographic view of participants

## Development

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build
```

## Tech Stack

- Next.js 16 (App Router)
- TypeScript
- Tailwind CSS
- wagmi/viem for contract interaction
- RainbowKit for wallet connection
- React Query for data fetching

## Pages

| Route | Description |
|-------|-------------|
| `/` | Main dashboard overview |
| `/status` | Real-time status monitor with auto-refresh |
| `/models` | Model registry browser |
| `/training` | Training sessions overview |
| `/training/[id]` | Individual training session details |
| `/nodes` | Network node explorer |
| `/proofs` | Proof verification explorer |
| `/demo` | Demo mode for presentations |

---

## API Endpoints Documentation

The dashboard consumes a backend API (default: `http://localhost:3001/api`). Configure via environment variables:

```bash
NEXT_PUBLIC_API_URL=http://localhost:3001/api
NEXT_PUBLIC_WS_URL=ws://localhost:3001/ws
```

### Core Endpoints

#### Nodes

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/nodes` | List all network nodes |
| GET | `/nodes/:id` | Get node details by ID |
| GET | `/nodes?type={compute\|aggregator\|verifier}` | Filter nodes by type |
| GET | `/nodes/:id/metrics` | Get node resource metrics (CPU, memory, GPU) |
| GET | `/nodes/:id/performance` | Get node performance stats |
| GET | `/network/topology` | Get full network topology with connections |

**Node Response Schema:**
```typescript
interface NodeInfo {
    id: string;
    address: string;
    type: 'compute' | 'aggregator' | 'verifier';
    status: 'online' | 'offline' | 'syncing' | 'proving' | 'training';
    lastSeen: number;
    lastHeartbeat: number;
    metrics: NodeMetrics;
    capabilities: NodeCapabilities;
    stake: { amount: bigint; lockedUntil: number; slashed: boolean };
    performance: NodePerformance;
    location?: { region: string; latency: number };
}
```

#### Proofs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/proofs` | List proofs with optional filters |
| GET | `/proofs/:id` | Get proof details by ID |
| GET | `/proofs/:id/progress` | Get proof generation progress |
| GET | `/proofs/timeline/:modelId` | Get proof timeline for a model |
| GET | `/proofs/stats` | Get aggregate proof statistics |

**Query Parameters for `/proofs`:**
- `modelId` - Filter by model ID
- `roundId` - Filter by round ID
- `prover` - Filter by prover address
- `status` - Filter by status (generating|pending|verified|failed|challenged)
- `type` - Filter by type (training|aggregation|gradient|computation|verification)
- `limit` - Number of results (default: 20)
- `offset` - Pagination offset

**Proof Response Schema:**
```typescript
interface ProofInfo {
    id: string;
    hash: string;
    modelId: bigint;
    roundId: bigint;
    prover: string;
    type: 'training' | 'aggregation' | 'gradient' | 'computation' | 'verification';
    status: 'generating' | 'pending' | 'verified' | 'failed' | 'challenged';
    createdAt: number;
    verifiedAt?: number;
    size: number;
    generationTime: number;
    verificationTime?: number;
    errorBound: number;
    publicInputsHash: string;
    commitment: bigint;
    gasUsed?: bigint;
    transactionHash?: string;
    blockNumber?: number;
    circuitType: string;
    constraintCount: number;
}
```

#### Training

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/training` | List training sessions |
| GET | `/training/:id` | Get session details |
| GET | `/training/:id/metrics` | Get training metrics |
| GET | `/training/:id/rounds` | Get training rounds |
| GET | `/training/:id/error-bounds` | Get error bound history |
| GET | `/training/:id/loss-curve` | Get loss curve data points |

**Training Session Schema:**
```typescript
interface TrainingSession {
    id: string;
    modelId: bigint;
    modelName: string;
    status: 'initializing' | 'training' | 'paused' | 'completed' | 'failed';
    startedAt: number;
    updatedAt: number;
    completedAt?: number;
    config: TrainingConfig;
    metrics: TrainingMetrics;
    rounds: TrainingRound[];
    workers: string[];
    errorBounds: ErrorBoundEntry[];
}
```

#### Adversarial Events

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/adversarial` | List adversarial events |
| GET | `/adversarial/:id` | Get event details |
| GET | `/adversarial/slashing` | Get slashing history |

**Query Parameters for `/adversarial`:**
- `modelId` - Filter by model ID
- `type` - Filter by event type
- `severity` - Filter by severity (warning|critical|resolved)
- `resolved` - Filter by resolution status
- `limit`, `offset` - Pagination

#### Network

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/network/stats` | Get network-wide statistics |
| GET | `/health` | Health check endpoint |

**NetworkStats Schema:**
```typescript
interface NetworkStats {
    totalNodes: number;
    activeNodes: number;
    totalStaked: bigint;
    totalProofs: number;
    totalRounds: number;
    totalSlashed: bigint;
    averageProofTime: number;
    networkUptime: number;
    throughput: number;
    activeTrainingSessions: number;
}
```

### Status Page Endpoints

These endpoints are specifically for the `/status` monitoring page:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/status/overview` | Complete status dashboard data |
| GET | `/status/workers` | Worker health status list |
| GET | `/status/round/:modelId` | Current round state for model |
| GET | `/status/logs` | Error log entries |
| GET | `/status/contract` | Contract state snapshot |
| GET | `/status/proofs` | Recent proof status with stats |
| GET | `/status/training/:id/live` | Live training metrics |

**Status Overview Schema:**
```typescript
interface StatusOverview {
    trainingSessions: TrainingSession[];
    workers: NodeInfo[];
    recentProofs: ProofInfo[];
    currentMetrics: TrainingMetrics | null;
    networkStats: NetworkStats;
    errorLogs: StatusErrorLog[];
    contractState: ContractStateSnapshot;
}
```

**Worker Health Status:**
```typescript
interface WorkerHealthStatus {
    workerId: string;
    address: string;
    type: 'compute' | 'aggregator' | 'verifier';
    status: 'online' | 'offline' | 'syncing' | 'proving' | 'training';
    lastHeartbeat: number;
    uptime: number;
    proofsSubmitted: number;
    proofsVerified: number;
    proofsFailed: number;
    averageProofTime: number;
    stake: { amount: string; slashed: boolean };
}
```

**Contract State Snapshot:**
```typescript
interface ContractStateSnapshot {
    nextModelId: string;
    defaultMinStake: string;
    stakeLockPeriod: string;
    slashPercentage: string;
    maxErrorBound: string;
    slashingRecordCount: string;
    activeModels: number;
    totalStaked: string;
}
```

**Round State:**
```typescript
interface RoundState {
    modelId: string;
    currentRound: string;
    roundStatus: 'pending' | 'in_progress' | 'aggregating' | 'proving' | 'completed' | 'failed';
    deadline: number | null;
    participants: string[];
    prover: string | null;
    modelCommitment: string;
    newCommitment: string | null;
    errorBound: number;
    proofSubmitted: boolean;
}
```

### WebSocket Events

Connect to `ws://localhost:3001/ws` for real-time updates.

**Message Types:**
- `node_update` - Node status changed
- `proof_update` - Proof status changed
- `training_update` - Training metrics updated
- `round_update` - Round state changed
- `adversarial_event` - Security event detected
- `network_stats` - Network statistics updated
- `error_bound_update` - Error bounds changed
- `heartbeat` - Keep-alive ping

**Subscribe to channels:**
```json
{ "type": "subscribe", "timestamp": 1234567890, "data": { "channel": "training:1" } }
```

### Error Responses

All endpoints return errors in this format:
```json
{
    "error": true,
    "message": "Error description",
    "statusCode": 400,
    "details": {}
}
```

### Demo Mode

When the backend is unavailable, the dashboard falls back to mock data generators. Toggle demo mode in the status page UI to switch between live and mock data.

---

## Smart Contract Integration

The dashboard reads from the `HelixCoordinatorV2` contract. Supported networks:

| Network | Chain ID | Coordinator Address |
|---------|----------|---------------------|
| Mainnet | 1 | `0x...` (TBD) |
| Sepolia | 11155111 | `0x...` (TBD) |
| Localhost | 31337 | `0x5FbDB2315678afecb367f032d93F642f64180aa3` |

### Contract Read Operations

The dashboard uses these contract view functions:
- `models(uint256)` - Get model data
- `rounds(uint256, uint256)` - Get round data
- `stakes(address, uint256)` - Get stake info
- `getModelState(uint256)` - Get model state
- `getStake(address, uint256)` - Get stake details
- `getAccumulatedErrorBound(uint256)` - Get error bounds
- `getSlashingRecordCount()` - Get slash count
- `slashingRecords(uint256)` - Get slash record
- `nextModelId()` - Get next model ID
- `defaultMinStake()` - Get minimum stake
- `stakeLockPeriod()` - Get lock period
- `slashPercentage()` - Get slash percentage
- `maxErrorBound()` - Get max error bound

### Contract Events

The dashboard listens for:
- `ModelRegistered` - New model registered
- `RoundStarted` - Training round began
- `ProofSubmitted` - Proof submitted
- `RoundCompleted` - Round finished
- `Staked` - Tokens staked
- `Unstaked` - Tokens unstaked
- `Slashed` - Worker slashed
- `InvalidProofDetected` - Bad proof detected
