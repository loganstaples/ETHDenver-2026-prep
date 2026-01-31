#!/bin/bash
#
# HELIX Log Aggregator Script
# ============================
# Aggregates and displays logs from all HELIX components
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'
DIM='\033[2m'

# Configuration
LOG_DIR=${LOG_DIR:-/tmp/helix-logs}
FOLLOW=${FOLLOW:-false}
LEVEL=${LEVEL:-INFO}
GREP_PATTERN=${GREP:-}
LINES=${LINES:-50}

# Create log directory
mkdir -p $LOG_DIR

usage() {
    cat << EOF
HELIX Log Aggregator

Usage: $(basename $0) [OPTIONS]

Options:
    -f, --follow      Follow logs in real-time
    -l, --level       Filter by log level (DEBUG, INFO, WARN, ERROR)
    -g, --grep        Filter by pattern
    -n, --lines       Number of lines to show (default: 50)
    -d, --dir         Log directory (default: /tmp/helix-logs)
    -h, --help        Show this help

Examples:
    $(basename $0) -f                    # Follow all logs
    $(basename $0) -l ERROR              # Show only errors
    $(basename $0) -g "round.*complete"  # Filter by pattern
    $(basename $0) -n 100 -l WARN        # Last 100 warnings

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--follow)
            FOLLOW=true
            shift
            ;;
        -l|--level)
            LEVEL="$2"
            shift 2
            ;;
        -g|--grep)
            GREP_PATTERN="$2"
            shift 2
            ;;
        -n|--lines)
            LINES="$2"
            shift 2
            ;;
        -d|--dir)
            LOG_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Color by component
color_component() {
    local component="$1"
    case $component in
        coordinator|aggregator)
            echo "${MAGENTA}$component${NC}"
            ;;
        worker*)
            echo "${BLUE}$component${NC}"
            ;;
        verifier)
            echo "${CYAN}$component${NC}"
            ;;
        *)
            echo "$component"
            ;;
    esac
}

# Color by level
color_level() {
    local level="$1"
    case $level in
        ERROR|FATAL)
            echo "${RED}$level${NC}"
            ;;
        WARN|WARNING)
            echo "${YELLOW}$level${NC}"
            ;;
        INFO)
            echo "${GREEN}$level${NC}"
            ;;
        DEBUG|TRACE)
            echo "${DIM}$level${NC}"
            ;;
        *)
            echo "$level"
            ;;
    esac
}

format_log() {
    # Expected format: TIMESTAMP LEVEL [COMPONENT] MESSAGE
    while IFS= read -r line; do
        if [[ "$line" =~ ^([0-9T:-]+)\ ([A-Z]+)\ \[([^\]]+)\]\ (.*)$ ]]; then
            local timestamp="${BASH_REMATCH[1]}"
            local level="${BASH_REMATCH[2]}"
            local component="${BASH_REMATCH[3]}"
            local message="${BASH_REMATCH[4]}"

            # Level filter
            case $LEVEL in
                ERROR)
                    [[ "$level" != "ERROR" && "$level" != "FATAL" ]] && continue
                    ;;
                WARN)
                    [[ "$level" == "DEBUG" || "$level" == "TRACE" || "$level" == "INFO" ]] && continue
                    ;;
                INFO)
                    [[ "$level" == "DEBUG" || "$level" == "TRACE" ]] && continue
                    ;;
            esac

            # Grep filter
            if [ -n "$GREP_PATTERN" ]; then
                echo "$line" | grep -qiE "$GREP_PATTERN" || continue
            fi

            # Format output
            printf "${DIM}%s${NC} %s [%s] %s\n" \
                "$timestamp" \
                "$(color_level $level)" \
                "$(color_component $component)" \
                "$message"
        else
            # Unstructured log
            if [ -n "$GREP_PATTERN" ]; then
                echo "$line" | grep -qiE "$GREP_PATTERN" || continue
            fi
            echo "$line"
        fi
    done
}

# Demo logs generator (for testing)
generate_demo_logs() {
    local components=("coordinator" "worker-0" "worker-1" "worker-2" "aggregator" "verifier")
    local levels=("INFO" "INFO" "INFO" "DEBUG" "WARN" "ERROR")
    local messages=(
        "Round started"
        "Computing gradients for batch 1/32"
        "Gradient computation complete"
        "Sending gradient to aggregator"
        "Received gradient from worker"
        "Aggregating 3 gradients"
        "Generating ZK proof"
        "Proof generated successfully"
        "Verifying proof"
        "Proof verified"
        "Committing round to chain"
        "Round committed"
        "Heartbeat"
        "Peer connected"
        "Peer disconnected"
    )

    while true; do
        local component=${components[$RANDOM % ${#components[@]}]}
        local level=${levels[$RANDOM % ${#levels[@]}]}
        local message=${messages[$RANDOM % ${#messages[@]}]}
        local timestamp=$(date +%Y-%m-%dT%H:%M:%S)

        echo "$timestamp $level [$component] $message"
        sleep 0.$((RANDOM % 5 + 1))
    done
}

# Banner
cat << 'EOF'
╔═══════════════════════════════════════════════════════════╗
║                   HELIX Log Aggregator                    ║
╚═══════════════════════════════════════════════════════════╝
EOF

echo -e "\n${CYAN}Configuration:${NC}"
echo "  Log Directory: $LOG_DIR"
echo "  Level Filter:  $LEVEL"
echo "  Grep Pattern:  ${GREP_PATTERN:-<none>}"
echo "  Follow:        $FOLLOW"
echo ""

# Check for existing logs
LOG_FILES=$(find $LOG_DIR -name "*.log" -type f 2>/dev/null | wc -l | tr -d ' ')

if [ "$LOG_FILES" = "0" ]; then
    echo -e "${YELLOW}No log files found in $LOG_DIR${NC}"
    echo -e "Generating demo logs...\n"

    if [ "$FOLLOW" = true ]; then
        generate_demo_logs | format_log
    else
        generate_demo_logs | head -n $LINES | format_log
    fi
else
    echo -e "${GREEN}Found $LOG_FILES log files${NC}\n"

    if [ "$FOLLOW" = true ]; then
        # Follow all logs
        tail -F $LOG_DIR/*.log 2>/dev/null | format_log
    else
        # Show recent logs
        cat $LOG_DIR/*.log 2>/dev/null | tail -n $LINES | format_log
    fi
fi
