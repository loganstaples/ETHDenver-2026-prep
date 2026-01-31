#!/bin/bash
# ==============================================================================
# HELIX GCP Deployment Script
# ==============================================================================
# Production-grade deployment to Google Cloud Platform using GKE.
#
# Prerequisites:
#   - gcloud CLI installed and configured
#   - kubectl installed
#   - helm 3.x installed
#
# Usage:
#   ./gcp.sh create-cluster     # Create new GKE cluster
#   ./gcp.sh deploy             # Deploy HELIX to existing cluster
#   ./gcp.sh destroy            # Tear down cluster
#   ./gcp.sh status             # Check deployment status
#
# Environment variables:
#   GCP_PROJECT        - GCP project ID (required)
#   GCP_REGION         - GCP region (default: us-central1)
#   GCP_ZONE           - GCP zone (default: us-central1-a)
#   CLUSTER_NAME       - GKE cluster name (default: helix-production)
#   NODE_COUNT         - Number of worker nodes (default: 5)
#   MACHINE_TYPE       - Machine type (default: n2-standard-8)
#   ENABLE_GPU         - Enable GPU nodes (default: false)
#   GPU_NODE_COUNT     - Number of GPU nodes (default: 3)
#   GPU_TYPE           - GPU type (default: nvidia-tesla-t4)
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
GCP_PROJECT="${GCP_PROJECT:-}"
GCP_REGION="${GCP_REGION:-us-central1}"
GCP_ZONE="${GCP_ZONE:-us-central1-a}"
CLUSTER_NAME="${CLUSTER_NAME:-helix-production}"
NODE_COUNT="${NODE_COUNT:-5}"
MACHINE_TYPE="${MACHINE_TYPE:-n2-standard-8}"
ENABLE_GPU="${ENABLE_GPU:-false}"
GPU_NODE_COUNT="${GPU_NODE_COUNT:-3}"
GPU_TYPE="${GPU_TYPE:-nvidia-tesla-t4}"
KUBERNETES_VERSION="${KUBERNETES_VERSION:-1.28}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELIX_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HELM_CHART="${HELIX_ROOT}/scripts/kubernetes/helm/helix"

log() {
    echo -e "${CYAN}[HELIX-GCP]${NC} $1"
}

success() {
    echo -e "${GREEN}[HELIX-GCP]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[HELIX-GCP]${NC} $1"
}

error() {
    echo -e "${RED}[HELIX-GCP]${NC} $1" >&2
    exit 1
}

banner() {
    cat << 'EOF'
   ██╗  ██╗███████╗██╗     ██╗██╗  ██╗     ██████╗  ██████╗██████╗
   ██║  ██║██╔════╝██║     ██║╚██╗██╔╝    ██╔════╝ ██╔════╝██╔══██╗
   ███████║█████╗  ██║     ██║ ╚███╔╝     ██║  ███╗██║     ██████╔╝
   ██╔══██║██╔══╝  ██║     ██║ ██╔██╗     ██║   ██║██║     ██╔═══╝
   ██║  ██║███████╗███████╗██║██╔╝ ██╗    ╚██████╔╝╚██████╗██║
   ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚═╝  ╚═╝     ╚═════╝  ╚═════╝╚═╝

   Production Deployment to Google Cloud Platform
EOF
    echo ""
}

check_prerequisites() {
    log "Checking prerequisites..."

    # Check gcloud
    if ! command -v gcloud &> /dev/null; then
        error "gcloud CLI not found. Install: https://cloud.google.com/sdk/docs/install"
    fi
    success "gcloud: $(gcloud version | head -1)"

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

    # Check project
    if [ -z "$GCP_PROJECT" ]; then
        GCP_PROJECT=$(gcloud config get-value project 2>/dev/null)
        if [ -z "$GCP_PROJECT" ]; then
            error "GCP_PROJECT not set. Set via environment or 'gcloud config set project'"
        fi
    fi
    success "GCP Project: $GCP_PROJECT"

    # Verify authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
        error "Not authenticated. Run: gcloud auth login"
    fi
    local account=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -1)
    success "GCP Account: $account"

    log "All prerequisites satisfied"
}

enable_apis() {
    log "Enabling required GCP APIs..."

    local apis=(
        "container.googleapis.com"
        "compute.googleapis.com"
        "monitoring.googleapis.com"
        "logging.googleapis.com"
        "cloudresourcemanager.googleapis.com"
    )

    for api in "${apis[@]}"; do
        gcloud services enable "$api" --project="$GCP_PROJECT" --quiet
        success "Enabled: $api"
    done
}

create_cluster() {
    log "Creating GKE cluster: $CLUSTER_NAME..."

    enable_apis

    # Create cluster
    local cluster_cmd="gcloud container clusters create $CLUSTER_NAME \
        --project=$GCP_PROJECT \
        --region=$GCP_REGION \
        --cluster-version=$KUBERNETES_VERSION \
        --release-channel=regular \
        --num-nodes=$NODE_COUNT \
        --machine-type=$MACHINE_TYPE \
        --disk-type=pd-ssd \
        --disk-size=100GB \
        --enable-autoscaling \
        --min-nodes=2 \
        --max-nodes=20 \
        --enable-autorepair \
        --enable-autoupgrade \
        --enable-ip-alias \
        --enable-network-policy \
        --enable-vertical-pod-autoscaling \
        --enable-shielded-nodes \
        --workload-pool=${GCP_PROJECT}.svc.id.goog \
        --logging=SYSTEM,WORKLOAD \
        --monitoring=SYSTEM \
        --node-labels=role=worker,helix.protocol/node-type=worker \
        --addons=HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver"

    eval "$cluster_cmd"
    success "GKE cluster created"

    # Add aggregator node pool
    log "Creating aggregator node pool..."
    gcloud container node-pools create helix-aggregator \
        --cluster="$CLUSTER_NAME" \
        --project="$GCP_PROJECT" \
        --region="$GCP_REGION" \
        --num-nodes=2 \
        --min-nodes=1 \
        --max-nodes=3 \
        --machine-type=n2-standard-16 \
        --disk-type=pd-ssd \
        --disk-size=200GB \
        --enable-autoscaling \
        --enable-autorepair \
        --enable-autoupgrade \
        --node-labels=role=aggregator,helix.protocol/node-type=aggregator
    success "Aggregator node pool created"

    # Add GPU node pool if enabled
    if [ "$ENABLE_GPU" = "true" ]; then
        log "Creating GPU node pool..."
        gcloud container node-pools create helix-gpu \
            --cluster="$CLUSTER_NAME" \
            --project="$GCP_PROJECT" \
            --region="$GCP_REGION" \
            --num-nodes="$GPU_NODE_COUNT" \
            --min-nodes=0 \
            --max-nodes=10 \
            --machine-type=n1-standard-8 \
            --accelerator="type=$GPU_TYPE,count=1" \
            --disk-type=pd-ssd \
            --disk-size=200GB \
            --enable-autoscaling \
            --enable-autorepair \
            --enable-autoupgrade \
            --node-labels=role=gpu-worker,helix.protocol/node-type=gpu-worker \
            --node-taints=nvidia.com/gpu=true:NoSchedule
        success "GPU node pool created"

        # Install NVIDIA driver
        log "Installing NVIDIA GPU driver..."
        kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
        success "NVIDIA driver installer deployed"
    fi

    # Get cluster credentials
    gcloud container clusters get-credentials "$CLUSTER_NAME" \
        --project="$GCP_PROJECT" \
        --region="$GCP_REGION"
    success "Cluster credentials configured"

    # Create storage class
    log "Creating storage class..."
    kubectl apply -f - << EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: helix-ssd
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"
provisioner: pd.csi.storage.gke.io
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
parameters:
  type: pd-ssd
  fsType: ext4
EOF
    success "Storage class created"
}

deploy() {
    log "Deploying HELIX to GKE cluster..."

    # Verify cluster access
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster. Run 'gcloud container clusters get-credentials' first."
    fi

    # Create namespace
    kubectl create namespace helix --dry-run=client -o yaml | kubectl apply -f -

    # Create secrets
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
        --set workers.resources.requests.cpu=2000m \
        --set workers.resources.requests.memory=4Gi \
        --set workers.resources.limits.cpu=6000m \
        --set workers.resources.limits.memory=12Gi \
        --set aggregator.resources.requests.cpu=4000m \
        --set aggregator.resources.requests.memory=8Gi \
        --set monitoring.prometheus.enabled=true \
        --set monitoring.grafana.enabled=true \
        --wait \
        --timeout 10m

    success "HELIX deployed successfully"

    # Output access information
    echo ""
    log "Deployment complete!"
    echo ""
    status
}

destroy() {
    log "Destroying HELIX deployment and GKE cluster..."

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
    log "Deleting GKE cluster..."
    gcloud container clusters delete "$CLUSTER_NAME" \
        --project="$GCP_PROJECT" \
        --region="$GCP_REGION" \
        --quiet

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
    echo "${BOLD}External IPs:${NC}"
    kubectl get svc -n helix -o jsonpath='{range .items[*]}{.metadata.name}: {.status.loadBalancer.ingress[0].ip}{"\n"}{end}'
}

usage() {
    cat << EOF
HELIX GCP Deployment Script

Usage: $(basename "$0") <command>

Commands:
    create-cluster    Create new GKE cluster
    deploy            Deploy HELIX to existing cluster
    destroy           Tear down cluster and all resources
    status            Check deployment status
    help              Show this help message

Environment Variables:
    GCP_PROJECT        GCP project ID (required)
    GCP_REGION         GCP region (default: us-central1)
    GCP_ZONE           GCP zone (default: us-central1-a)
    CLUSTER_NAME       GKE cluster name (default: helix-production)
    NODE_COUNT         Number of worker nodes (default: 5)
    MACHINE_TYPE       Machine type (default: n2-standard-8)
    ENABLE_GPU         Enable GPU nodes (default: false)
    GPU_NODE_COUNT     Number of GPU nodes (default: 3)
    GPU_TYPE           GPU type (default: nvidia-tesla-t4)

Examples:
    GCP_PROJECT=my-project $(basename "$0") create-cluster
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
