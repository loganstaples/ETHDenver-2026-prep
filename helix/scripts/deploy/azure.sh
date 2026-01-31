#!/bin/bash
# ==============================================================================
# HELIX Azure Deployment Script
# ==============================================================================
# Production-grade deployment to Microsoft Azure using AKS.
#
# Prerequisites:
#   - Azure CLI installed and configured
#   - kubectl installed
#   - helm 3.x installed
#
# Usage:
#   ./azure.sh create-cluster     # Create new AKS cluster
#   ./azure.sh deploy             # Deploy HELIX to existing cluster
#   ./azure.sh destroy            # Tear down cluster
#   ./azure.sh status             # Check deployment status
#
# Environment variables:
#   AZURE_SUBSCRIPTION  - Azure subscription ID
#   AZURE_RESOURCE_GROUP- Resource group name (default: helix-rg)
#   AZURE_LOCATION      - Azure region (default: eastus)
#   CLUSTER_NAME        - AKS cluster name (default: helix-production)
#   NODE_COUNT          - Number of worker nodes (default: 5)
#   NODE_SIZE           - VM size (default: Standard_D8s_v3)
#   ENABLE_GPU          - Enable GPU nodes (default: false)
#   GPU_NODE_COUNT      - Number of GPU nodes (default: 3)
#   GPU_SIZE            - GPU VM size (default: Standard_NC6s_v3)
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
AZURE_SUBSCRIPTION="${AZURE_SUBSCRIPTION:-}"
AZURE_RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-helix-rg}"
AZURE_LOCATION="${AZURE_LOCATION:-eastus}"
CLUSTER_NAME="${CLUSTER_NAME:-helix-production}"
NODE_COUNT="${NODE_COUNT:-5}"
NODE_SIZE="${NODE_SIZE:-Standard_D8s_v3}"
ENABLE_GPU="${ENABLE_GPU:-false}"
GPU_NODE_COUNT="${GPU_NODE_COUNT:-3}"
GPU_SIZE="${GPU_SIZE:-Standard_NC6s_v3}"
KUBERNETES_VERSION="${KUBERNETES_VERSION:-1.28}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELIX_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HELM_CHART="${HELIX_ROOT}/scripts/kubernetes/helm/helix"

log() {
    echo -e "${CYAN}[HELIX-AZURE]${NC} $1"
}

success() {
    echo -e "${GREEN}[HELIX-AZURE]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[HELIX-AZURE]${NC} $1"
}

error() {
    echo -e "${RED}[HELIX-AZURE]${NC} $1" >&2
    exit 1
}

banner() {
    cat << 'EOF'
   ██╗  ██╗███████╗██╗     ██╗██╗  ██╗     █████╗ ███████╗██╗   ██╗██████╗ ███████╗
   ██║  ██║██╔════╝██║     ██║╚██╗██╔╝    ██╔══██╗╚══███╔╝██║   ██║██╔══██╗██╔════╝
   ███████║█████╗  ██║     ██║ ╚███╔╝     ███████║  ███╔╝ ██║   ██║██████╔╝█████╗
   ██╔══██║██╔══╝  ██║     ██║ ██╔██╗     ██╔══██║ ███╔╝  ██║   ██║██╔══██╗██╔══╝
   ██║  ██║███████╗███████╗██║██╔╝ ██╗    ██║  ██║███████╗╚██████╔╝██║  ██║███████╗
   ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝

   Production Deployment to Microsoft Azure
EOF
    echo ""
}

check_prerequisites() {
    log "Checking prerequisites..."

    # Check Azure CLI
    if ! command -v az &> /dev/null; then
        error "Azure CLI not found. Install: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    fi
    success "Azure CLI: $(az version --query '"azure-cli"' -o tsv)"

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

    # Check Azure login
    if ! az account show &> /dev/null; then
        error "Not logged in to Azure. Run: az login"
    fi

    # Get subscription
    if [ -z "$AZURE_SUBSCRIPTION" ]; then
        AZURE_SUBSCRIPTION=$(az account show --query id -o tsv)
    fi
    az account set --subscription "$AZURE_SUBSCRIPTION"
    success "Azure Subscription: $(az account show --query name -o tsv)"

    log "All prerequisites satisfied"
}

register_providers() {
    log "Registering Azure resource providers..."

    local providers=(
        "Microsoft.Compute"
        "Microsoft.ContainerService"
        "Microsoft.Network"
        "Microsoft.Storage"
        "Microsoft.OperationsManagement"
        "Microsoft.OperationalInsights"
    )

    for provider in "${providers[@]}"; do
        az provider register --namespace "$provider" --wait || true
        success "Registered: $provider"
    done
}

create_cluster() {
    log "Creating AKS cluster: $CLUSTER_NAME..."

    register_providers

    # Create resource group
    log "Creating resource group..."
    az group create \
        --name "$AZURE_RESOURCE_GROUP" \
        --location "$AZURE_LOCATION" \
        --tags Project=helix Environment=production
    success "Resource group created: $AZURE_RESOURCE_GROUP"

    # Create Log Analytics workspace
    log "Creating Log Analytics workspace..."
    local workspace_name="${CLUSTER_NAME}-logs"
    az monitor log-analytics workspace create \
        --resource-group "$AZURE_RESOURCE_GROUP" \
        --workspace-name "$workspace_name" \
        --location "$AZURE_LOCATION" || true
    local workspace_id=$(az monitor log-analytics workspace show \
        --resource-group "$AZURE_RESOURCE_GROUP" \
        --workspace-name "$workspace_name" \
        --query id -o tsv)
    success "Log Analytics workspace created"

    # Create AKS cluster
    log "Creating AKS cluster..."
    az aks create \
        --resource-group "$AZURE_RESOURCE_GROUP" \
        --name "$CLUSTER_NAME" \
        --location "$AZURE_LOCATION" \
        --kubernetes-version "$KUBERNETES_VERSION" \
        --node-count "$NODE_COUNT" \
        --node-vm-size "$NODE_SIZE" \
        --node-osdisk-type Managed \
        --node-osdisk-size 100 \
        --nodepool-name workers \
        --nodepool-labels role=worker helix.protocol/node-type=worker \
        --enable-cluster-autoscaler \
        --min-count 2 \
        --max-count 20 \
        --enable-addons monitoring \
        --workspace-resource-id "$workspace_id" \
        --enable-managed-identity \
        --enable-aad \
        --enable-azure-rbac \
        --network-plugin azure \
        --network-policy calico \
        --load-balancer-sku standard \
        --zones 1 2 3 \
        --tags Project=helix Environment=production
    success "AKS cluster created"

    # Add aggregator node pool
    log "Creating aggregator node pool..."
    az aks nodepool add \
        --resource-group "$AZURE_RESOURCE_GROUP" \
        --cluster-name "$CLUSTER_NAME" \
        --name aggregator \
        --node-count 2 \
        --node-vm-size Standard_D16s_v3 \
        --node-osdisk-type Managed \
        --node-osdisk-size 200 \
        --labels role=aggregator helix.protocol/node-type=aggregator \
        --enable-cluster-autoscaler \
        --min-count 1 \
        --max-count 3 \
        --zones 1 2 3
    success "Aggregator node pool created"

    # Add GPU node pool if enabled
    if [ "$ENABLE_GPU" = "true" ]; then
        log "Creating GPU node pool..."
        az aks nodepool add \
            --resource-group "$AZURE_RESOURCE_GROUP" \
            --cluster-name "$CLUSTER_NAME" \
            --name gpu \
            --node-count "$GPU_NODE_COUNT" \
            --node-vm-size "$GPU_SIZE" \
            --node-osdisk-type Managed \
            --node-osdisk-size 200 \
            --labels role=gpu-worker helix.protocol/node-type=gpu-worker \
            --node-taints nvidia.com/gpu=true:NoSchedule \
            --enable-cluster-autoscaler \
            --min-count 0 \
            --max-count 10 \
            --zones 1 2
        success "GPU node pool created"

        # Install NVIDIA device plugin
        log "Installing NVIDIA device plugin..."
        kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
        success "NVIDIA device plugin installed"
    fi

    # Get credentials
    az aks get-credentials \
        --resource-group "$AZURE_RESOURCE_GROUP" \
        --name "$CLUSTER_NAME" \
        --overwrite-existing
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
provisioner: disk.csi.azure.com
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
parameters:
  skuName: Premium_LRS
  kind: Managed
  fsType: ext4
EOF
    success "Storage class created"
}

deploy() {
    log "Deploying HELIX to AKS cluster..."

    # Verify cluster access
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster. Run 'az aks get-credentials' first."
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
    log "Destroying HELIX deployment and AKS cluster..."

    read -p "Are you sure you want to destroy the cluster? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        log "Aborted"
        exit 0
    fi

    # Delete cluster
    log "Deleting AKS cluster..."
    az aks delete \
        --resource-group "$AZURE_RESOURCE_GROUP" \
        --name "$CLUSTER_NAME" \
        --yes \
        --no-wait

    # Optionally delete resource group
    read -p "Delete entire resource group? (yes/no): " delete_rg
    if [ "$delete_rg" = "yes" ]; then
        log "Deleting resource group..."
        az group delete \
            --name "$AZURE_RESOURCE_GROUP" \
            --yes \
            --no-wait
    fi

    success "Cluster deletion initiated"
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
HELIX Azure Deployment Script

Usage: $(basename "$0") <command>

Commands:
    create-cluster    Create new AKS cluster
    deploy            Deploy HELIX to existing cluster
    destroy           Tear down cluster and all resources
    status            Check deployment status
    help              Show this help message

Environment Variables:
    AZURE_SUBSCRIPTION   Azure subscription ID
    AZURE_RESOURCE_GROUP Resource group name (default: helix-rg)
    AZURE_LOCATION       Azure region (default: eastus)
    CLUSTER_NAME         AKS cluster name (default: helix-production)
    NODE_COUNT           Number of worker nodes (default: 5)
    NODE_SIZE            VM size (default: Standard_D8s_v3)
    ENABLE_GPU           Enable GPU nodes (default: false)
    GPU_NODE_COUNT       Number of GPU nodes (default: 3)
    GPU_SIZE             GPU VM size (default: Standard_NC6s_v3)

Examples:
    $(basename "$0") create-cluster
    AZURE_LOCATION=westus2 $(basename "$0") create-cluster
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
