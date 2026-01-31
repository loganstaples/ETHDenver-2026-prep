#!/bin/bash
# ==============================================================================
# HELIX AWS Deployment Script
# ==============================================================================
# Production-grade deployment to Amazon Web Services using EKS.
#
# Prerequisites:
#   - AWS CLI v2 installed and configured
#   - kubectl installed
#   - helm 3.x installed
#   - eksctl installed (for cluster creation)
#
# Usage:
#   ./aws.sh create-cluster     # Create new EKS cluster
#   ./aws.sh deploy             # Deploy HELIX to existing cluster
#   ./aws.sh destroy            # Tear down cluster
#   ./aws.sh status             # Check deployment status
#
# Environment variables:
#   AWS_REGION         - AWS region (default: us-east-1)
#   AWS_PROFILE        - AWS CLI profile (default: default)
#   CLUSTER_NAME       - EKS cluster name (default: helix-production)
#   NODE_COUNT         - Number of worker nodes (default: 5)
#   NODE_INSTANCE_TYPE - EC2 instance type (default: c5.2xlarge)
#   ENABLE_GPU         - Enable GPU nodes (default: false)
#   GPU_NODE_COUNT     - Number of GPU nodes (default: 3)
#   GPU_INSTANCE_TYPE  - GPU instance type (default: p3.2xlarge)
# ==============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# Configuration
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_PROFILE="${AWS_PROFILE:-default}"
CLUSTER_NAME="${CLUSTER_NAME:-helix-production}"
NODE_COUNT="${NODE_COUNT:-5}"
NODE_INSTANCE_TYPE="${NODE_INSTANCE_TYPE:-c5.2xlarge}"
ENABLE_GPU="${ENABLE_GPU:-false}"
GPU_NODE_COUNT="${GPU_NODE_COUNT:-3}"
GPU_INSTANCE_TYPE="${GPU_INSTANCE_TYPE:-p3.2xlarge}"
KUBERNETES_VERSION="${KUBERNETES_VERSION:-1.28}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELIX_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HELM_CHART="${HELIX_ROOT}/scripts/kubernetes/helm/helix"

log() {
    echo -e "${CYAN}[HELIX-AWS]${NC} $1"
}

success() {
    echo -e "${GREEN}[HELIX-AWS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[HELIX-AWS]${NC} $1"
}

error() {
    echo -e "${RED}[HELIX-AWS]${NC} $1" >&2
    exit 1
}

banner() {
    cat << 'EOF'
   ██╗  ██╗███████╗██╗     ██╗██╗  ██╗     █████╗ ██╗    ██╗███████╗
   ██║  ██║██╔════╝██║     ██║╚██╗██╔╝    ██╔══██╗██║    ██║██╔════╝
   ███████║█████╗  ██║     ██║ ╚███╔╝     ███████║██║ █╗ ██║███████╗
   ██╔══██║██╔══╝  ██║     ██║ ██╔██╗     ██╔══██║██║███╗██║╚════██║
   ██║  ██║███████╗███████╗██║██╔╝ ██╗    ██║  ██║╚███╔███╔╝███████║
   ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚═╝  ╚═╝    ╚═╝  ╚═╝ ╚══╝╚══╝ ╚══════╝

   Production Deployment to Amazon Web Services
EOF
    echo ""
}

check_prerequisites() {
    log "Checking prerequisites..."

    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        error "AWS CLI not found. Install: https://aws.amazon.com/cli/"
    fi
    success "AWS CLI: $(aws --version | head -1)"

    # Check eksctl
    if ! command -v eksctl &> /dev/null; then
        error "eksctl not found. Install: https://eksctl.io/installation/"
    fi
    success "eksctl: $(eksctl version)"

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        error "kubectl not found. Install: https://kubernetes.io/docs/tasks/tools/"
    fi
    success "kubectl: $(kubectl version --client --short 2>/dev/null || kubectl version --client | head -1)"

    # Check helm
    if ! command -v helm &> /dev/null; then
        error "Helm not found. Install: https://helm.sh/docs/intro/install/"
    fi
    success "Helm: $(helm version --short)"

    # Check AWS credentials
    if ! aws sts get-caller-identity --profile "$AWS_PROFILE" &> /dev/null; then
        error "AWS credentials not configured. Run: aws configure"
    fi
    local identity=$(aws sts get-caller-identity --profile "$AWS_PROFILE" --output text --query 'Arn')
    success "AWS Identity: $identity"

    log "All prerequisites satisfied"
}

create_cluster() {
    log "Creating EKS cluster: $CLUSTER_NAME in $AWS_REGION..."

    # Generate eksctl config
    local config_file="/tmp/helix-eksctl-config.yaml"

    cat > "$config_file" << EOF
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: ${CLUSTER_NAME}
  region: ${AWS_REGION}
  version: "${KUBERNETES_VERSION}"

iam:
  withOIDC: true

addons:
  - name: vpc-cni
    attachPolicyARNs:
      - arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy
  - name: coredns
  - name: kube-proxy
  - name: aws-ebs-csi-driver
    wellKnownPolicies:
      ebsCSIController: true

managedNodeGroups:
  - name: helix-workers
    instanceType: ${NODE_INSTANCE_TYPE}
    desiredCapacity: ${NODE_COUNT}
    minSize: 2
    maxSize: 20
    volumeSize: 100
    volumeType: gp3
    privateNetworking: true
    iam:
      attachPolicyARNs:
        - arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy
        - arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
        - arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
    labels:
      role: worker
      helix.protocol/node-type: worker
    tags:
      Environment: production
      Project: helix
    ssh:
      allow: false

  - name: helix-aggregator
    instanceType: c5.4xlarge
    desiredCapacity: 2
    minSize: 1
    maxSize: 3
    volumeSize: 200
    volumeType: gp3
    privateNetworking: true
    labels:
      role: aggregator
      helix.protocol/node-type: aggregator
    tags:
      Environment: production
      Project: helix
    ssh:
      allow: false
EOF

    # Add GPU node group if enabled
    if [ "$ENABLE_GPU" = "true" ]; then
        cat >> "$config_file" << EOF

  - name: helix-gpu
    instanceType: ${GPU_INSTANCE_TYPE}
    desiredCapacity: ${GPU_NODE_COUNT}
    minSize: 0
    maxSize: 10
    volumeSize: 200
    volumeType: gp3
    privateNetworking: true
    labels:
      role: gpu-worker
      helix.protocol/node-type: gpu-worker
    taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule
    tags:
      Environment: production
      Project: helix
      k8s.io/cluster-autoscaler/enabled: "true"
      k8s.io/cluster-autoscaler/${CLUSTER_NAME}: "owned"
EOF
    fi

    log "Creating cluster with configuration:"
    cat "$config_file"
    echo ""

    # Create cluster
    eksctl create cluster -f "$config_file" --profile "$AWS_PROFILE"

    success "EKS cluster created successfully"

    # Install NVIDIA device plugin if GPU enabled
    if [ "$ENABLE_GPU" = "true" ]; then
        log "Installing NVIDIA device plugin..."
        kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
        success "NVIDIA device plugin installed"
    fi

    # Install metrics server
    log "Installing metrics server..."
    kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
    success "Metrics server installed"

    # Install cluster autoscaler
    log "Installing cluster autoscaler..."
    helm repo add autoscaler https://kubernetes.github.io/autoscaler
    helm install cluster-autoscaler autoscaler/cluster-autoscaler \
        --namespace kube-system \
        --set autoDiscovery.clusterName="$CLUSTER_NAME" \
        --set awsRegion="$AWS_REGION" \
        --set extraArgs.balance-similar-node-groups=true \
        --set extraArgs.skip-nodes-with-system-pods=false
    success "Cluster autoscaler installed"

    # Create storage class
    log "Creating storage class..."
    kubectl apply -f - << EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: helix-ssd
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"
provisioner: ebs.csi.aws.com
volumeBindingMode: WaitForFirstConsumer
parameters:
  type: gp3
  fsType: ext4
  encrypted: "true"
  iopsPerGB: "50"
  throughput: "125"
EOF
    success "Storage class created"

    # Update kubeconfig
    aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$AWS_REGION" --profile "$AWS_PROFILE"
    success "kubeconfig updated"
}

deploy() {
    log "Deploying HELIX to EKS cluster..."

    # Verify cluster access
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster. Update kubeconfig first."
    fi

    # Create namespace
    kubectl create namespace helix --dry-run=client -o yaml | kubectl apply -f -

    # Create secrets (example - should be replaced with actual secrets management)
    log "Creating secrets..."
    kubectl create secret generic helix-eth-credentials \
        --namespace helix \
        --from-literal=private-key="${HELIX_PRIVATE_KEY:-}" \
        --dry-run=client -o yaml | kubectl apply -f -

    # Install monitoring stack
    log "Installing monitoring stack..."
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update

    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --set grafana.adminPassword=helix-admin \
        --set prometheus.prometheusSpec.retention=15d \
        --wait

    # Deploy HELIX
    log "Deploying HELIX..."
    helm dependency update "$HELM_CHART"
    helm upgrade --install helix "$HELM_CHART" \
        --namespace helix \
        --set global.storageClass=helix-ssd \
        --set workers.replicaCount=5 \
        --set workers.resources.requests.cpu=1000m \
        --set workers.resources.requests.memory=2Gi \
        --set workers.resources.limits.cpu=4000m \
        --set workers.resources.limits.memory=8Gi \
        --set aggregator.resources.requests.cpu=2000m \
        --set aggregator.resources.requests.memory=4Gi \
        --set monitoring.prometheus.enabled=true \
        --set monitoring.grafana.enabled=true \
        --wait \
        --timeout 10m

    success "HELIX deployed successfully"

    # Output access information
    echo ""
    log "Deployment complete. Access information:"
    echo ""
    echo "  Dashboard:"
    kubectl get svc -n helix helix-dashboard -o jsonpath='  http://{.status.loadBalancer.ingress[0].hostname}:3000'
    echo ""
    echo ""
    echo "  Aggregator API:"
    kubectl get svc -n helix helix-aggregator -o jsonpath='  http://{.status.loadBalancer.ingress[0].hostname}:9001'
    echo ""
    echo ""
    echo "  Grafana (monitoring):"
    kubectl get svc -n monitoring prometheus-grafana -o jsonpath='  http://{.status.loadBalancer.ingress[0].hostname}'
    echo ""
}

destroy() {
    log "Destroying HELIX deployment and EKS cluster..."

    read -p "Are you sure you want to destroy the cluster? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        log "Aborted"
        exit 0
    fi

    # Uninstall HELIX
    log "Uninstalling HELIX..."
    helm uninstall helix --namespace helix || true

    # Delete namespace
    kubectl delete namespace helix || true

    # Uninstall monitoring
    helm uninstall prometheus --namespace monitoring || true
    kubectl delete namespace monitoring || true

    # Delete cluster
    log "Deleting EKS cluster..."
    eksctl delete cluster --name "$CLUSTER_NAME" --region "$AWS_REGION" --profile "$AWS_PROFILE" --wait

    success "Cluster destroyed"
}

status() {
    log "Checking HELIX deployment status..."

    echo ""
    echo "${BOLD}Cluster Info:${NC}"
    kubectl cluster-info

    echo ""
    echo "${BOLD}Nodes:${NC}"
    kubectl get nodes -o wide

    echo ""
    echo "${BOLD}HELIX Pods:${NC}"
    kubectl get pods -n helix -o wide

    echo ""
    echo "${BOLD}HELIX Services:${NC}"
    kubectl get svc -n helix

    echo ""
    echo "${BOLD}Aggregator Status:${NC}"
    kubectl get pods -n helix -l app.kubernetes.io/component=aggregator

    echo ""
    echo "${BOLD}Worker Status:${NC}"
    kubectl get pods -n helix -l app.kubernetes.io/component=worker

    echo ""
    echo "${BOLD}PersistentVolumeClaims:${NC}"
    kubectl get pvc -n helix
}

usage() {
    cat << EOF
HELIX AWS Deployment Script

Usage: $(basename "$0") <command>

Commands:
    create-cluster    Create new EKS cluster
    deploy            Deploy HELIX to existing cluster
    destroy           Tear down cluster and all resources
    status            Check deployment status
    help              Show this help message

Environment Variables:
    AWS_REGION         AWS region (default: us-east-1)
    AWS_PROFILE        AWS CLI profile (default: default)
    CLUSTER_NAME       EKS cluster name (default: helix-production)
    NODE_COUNT         Number of worker nodes (default: 5)
    NODE_INSTANCE_TYPE EC2 instance type (default: c5.2xlarge)
    ENABLE_GPU         Enable GPU nodes (default: false)
    GPU_NODE_COUNT     Number of GPU nodes (default: 3)
    GPU_INSTANCE_TYPE  GPU instance type (default: p3.2xlarge)

Examples:
    $(basename "$0") create-cluster
    AWS_REGION=eu-west-1 $(basename "$0") create-cluster
    ENABLE_GPU=true $(basename "$0") create-cluster
    $(basename "$0") deploy

EOF
}

# Main
banner

case "${1:-help}" in
    create-cluster)
        check_prerequisites
        create_cluster
        ;;
    deploy)
        check_prerequisites
        deploy
        ;;
    destroy)
        check_prerequisites
        destroy
        ;;
    status)
        status
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        error "Unknown command: $1. Use 'help' for usage."
        ;;
esac
