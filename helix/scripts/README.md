# HELIX Deployment & Operations Scripts

Production-grade deployment infrastructure for HELIX distributed ML training.

## Quick Start

```bash
# Local development with Docker
cd docker && docker-compose up -d

# Run the 90-second demo
./demo/demo-90s.sh

# Check system health
./health-check.sh
```

## Directory Structure

```
scripts/
├── docker/                    # Docker containerization
│   ├── Dockerfile.node        # Worker/verifier node image
│   ├── Dockerfile.aggregator  # Aggregator node image
│   ├── Dockerfile.client      # CLI client image
│   ├── Dockerfile.dashboard   # Next.js dashboard image
│   ├── docker-compose.yml     # Full local stack
│   ├── config/                # Configuration files
│   └── scripts/               # Container entrypoints
│
├── kubernetes/                # Kubernetes deployment
│   └── helm/helix/           # Helm chart
│       ├── Chart.yaml        # Chart metadata
│       ├── values.yaml       # Default values
│       └── templates/        # K8s manifests
│
├── monitoring/               # Observability stack
│   ├── prometheus.yml        # Metrics collection
│   ├── rules/               # Alert rules
│   ├── grafana/             # Dashboards
│   ├── loki-config.yaml     # Log aggregation
│   └── promtail-config.yaml # Log shipping
│
├── deploy/                   # Cloud deployment
│   ├── aws.sh               # Amazon EKS
│   ├── gcp.sh               # Google GKE
│   └── azure.sh             # Azure AKS
│
├── chaos/                    # Chaos testing
│   ├── kill-node.sh         # Node failure simulation
│   └── network-partition.sh # Network chaos
│
├── demo/                     # Demo scripts
│   └── demo-90s.sh          # 90-second pitch demo
│
└── *.sh                      # Utility scripts
    ├── health-check.sh       # Health monitoring
    ├── deploy.sh             # Contract deployment
    ├── benchmark.sh          # Performance testing
    └── log-aggregator.sh     # Log viewing
```

## Docker Deployment

### Building Images

```bash
cd helix

# Build all images
docker-compose -f scripts/docker/docker-compose.yml build

# Build individual images
docker build -t helix-node -f scripts/docker/Dockerfile.node .
docker build -t helix-aggregator -f scripts/docker/Dockerfile.aggregator .
docker build -t helix-client -f scripts/docker/Dockerfile.client .
```

### Running Locally

```bash
# Start full stack
docker-compose -f scripts/docker/docker-compose.yml up -d

# Start with monitoring
docker-compose -f scripts/docker/docker-compose.yml --profile monitoring up -d

# Start with dashboard
docker-compose -f scripts/docker/docker-compose.yml --profile demo up -d

# Scale workers
docker-compose -f scripts/docker/docker-compose.yml up -d --scale worker=5

# View logs
docker-compose -f scripts/docker/docker-compose.yml logs -f aggregator

# Stop everything
docker-compose -f scripts/docker/docker-compose.yml down -v
```

### Ports

| Service     | Port  | Description          |
|-------------|-------|---------------------|
| Ethereum    | 8545  | JSON-RPC            |
| Aggregator  | 9000  | P2P                 |
| Aggregator  | 9001  | HTTP API            |
| Aggregator  | 9002  | Prometheus metrics  |
| Workers     | 901x  | API (per worker)    |
| Dashboard   | 3000  | Web UI              |
| Prometheus  | 9090  | Metrics             |
| Grafana     | 3001  | Dashboards          |

## Kubernetes Deployment

### Prerequisites

- kubectl configured
- Helm 3.x installed
- Cluster access (EKS, GKE, AKS, or local)

### Using Helm

```bash
cd scripts/kubernetes/helm/helix

# Update dependencies
helm dependency update

# Install
helm install helix . \
  --namespace helix \
  --create-namespace \
  --set workers.replicaCount=5

# Upgrade
helm upgrade helix . --namespace helix

# Uninstall
helm uninstall helix --namespace helix
```

### Custom Values

```yaml
# my-values.yaml
workers:
  replicaCount: 10
  resources:
    limits:
      cpu: "4"
      memory: "8Gi"
  gpu:
    enabled: true
    type: nvidia
    count: 1

aggregator:
  resources:
    limits:
      cpu: "8"
      memory: "16Gi"

ethereum:
  external:
    enabled: true
    rpcUrl: "https://mainnet.infura.io/v3/YOUR-KEY"
    chainId: 1

monitoring:
  prometheus:
    enabled: true
  grafana:
    enabled: true
    adminPassword: "secure-password"
```

```bash
helm install helix . -f my-values.yaml --namespace helix
```

## Cloud Deployment

### AWS (EKS)

```bash
# Create cluster
./deploy/aws.sh create-cluster

# Deploy HELIX
./deploy/aws.sh deploy

# Check status
./deploy/aws.sh status

# Destroy
./deploy/aws.sh destroy
```

Environment variables:
- `AWS_REGION` - Region (default: us-east-1)
- `CLUSTER_NAME` - Cluster name (default: helix-production)
- `NODE_COUNT` - Worker nodes (default: 5)
- `ENABLE_GPU` - GPU support (default: false)

### GCP (GKE)

```bash
export GCP_PROJECT=my-project

./deploy/gcp.sh create-cluster
./deploy/gcp.sh deploy
```

### Azure (AKS)

```bash
./deploy/azure.sh create-cluster
./deploy/azure.sh deploy
```

## Monitoring

### Prometheus Metrics

Key metrics:
- `helix_training_rounds_completed_total` - Training progress
- `helix_training_accumulated_error` - Error bound tracking
- `helix_prover_proof_generation_seconds` - Proof latency
- `helix_aggregator_proofs_verified_total` - Verification count
- `helix_network_connected_peers` - Network health

### Grafana Dashboards

Access at http://localhost:3001 (default: admin/helix)

Dashboards:
- **HELIX Overview** - Training progress, error bounds, worker status
- **Worker Performance** - Gradient computation, proof generation
- **Network Health** - Peer connections, message latency
- **System Resources** - CPU, memory, disk usage

### Alerts

Critical alerts:
- `AggregatorDown` - Aggregator unreachable
- `InsufficientWorkers` - Less than 3 workers
- `ErrorBoundExceeded` - Training quality compromised
- `SlashingEvent` - Byzantine worker detected

## Chaos Testing

### Node Failures

```bash
# Kill random worker
./chaos/kill-node.sh random

# Kill specific node
./chaos/kill-node.sh helix-worker1

# Kill multiple workers
./chaos/kill-node.sh --cascade 2

# Monitor recovery
./chaos/kill-node.sh random --monitor 60

# Restore all
./chaos/kill-node.sh --restore
```

### Network Partitions

```bash
# Isolate a node
./chaos/network-partition.sh isolate helix-worker1 30

# Create network split
./chaos/network-partition.sh split 2

# Add latency
./chaos/network-partition.sh latency helix-worker1 200

# Packet loss
./chaos/network-partition.sh loss helix-worker1 30

# Heal all
./chaos/network-partition.sh heal
```

## Demo Script

The 90-second demo showcases:

1. **Infrastructure** (0:00-0:12) - Blockchain + contracts
2. **Model Registration** (0:12-0:22) - MPC weight sharing
3. **Worker Network** (0:22-0:34) - Distributed nodes
4. **Training** (0:34-1:09) - Rounds with ZK proofs
5. **Adversarial** (1:09-1:21) - Attack detection
6. **Summary** (1:21-1:30) - Results

```bash
# Full demo
./demo/demo-90s.sh

# Fast mode (60s)
./demo/demo-90s.sh --fast

# Rehearsal with prompts
./demo/demo-90s.sh --rehearse

# Generate precomputed data
./demo/demo-90s.sh --precompute
```

## Health Checks

```bash
# Full health check
./health-check.sh

# Quick check
./health-check.sh --quick

# Docker-specific
./health-check.sh --docker

# Kubernetes-specific
./health-check.sh --kubernetes

# JSON output
./health-check.sh --json

# Continuous monitoring
./health-check.sh --watch
```

## Troubleshooting

### Common Issues

**Anvil won't start**
```bash
# Check if port is in use
lsof -i :8545
# Kill existing process
pkill anvil
```

**Workers not connecting**
```bash
# Check aggregator logs
docker logs helix-aggregator
# Verify network
docker network inspect helix-network
```

**Proof generation slow**
```bash
# Increase worker resources
docker-compose up -d --scale worker=3 \
  --cpus 4 --memory 8g
```

**Contract deployment fails**
```bash
# Rebuild contracts
cd contracts && forge build --force
# Check gas
cast gas-price --rpc-url http://localhost:8545
```

### Log Viewing

```bash
# View all logs
./log-aggregator.sh -f

# Filter by level
./log-aggregator.sh -l ERROR

# Filter by pattern
./log-aggregator.sh -g "proof.*failed"
```

## Security Considerations

- Never commit private keys
- Use secrets management in production
- Enable TLS for all communications
- Review network policies before deployment
- Run containers as non-root
- Enable audit logging

## License

MIT License - See LICENSE file in repository root.
