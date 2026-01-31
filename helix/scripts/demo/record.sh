#!/bin/bash
#
# HELIX Demo Recording Script
# =============================
# Creates reproducible demo recordings for presentations and videos
#
# Features:
# - Terminal recording with asciinema
# - Timestamp overlays
# - Automatic pauses at key moments
# - Video export options
# - Playback speed control
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELIX_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
OUTPUT_DIR=${OUTPUT_DIR:-"$HELIX_ROOT/.helix/recordings"}
RECORDING_NAME=${RECORDING_NAME:-"helix_demo_$(date +%Y%m%d_%H%M%S)"}
RECORDER=${RECORDER:-"asciinema"}
PLAYBACK_SPEED=${PLAYBACK_SPEED:-1.0}
IDLE_TIME_LIMIT=${IDLE_TIME_LIMIT:-2}

# Mode: "record", "play", "export", "list"
MODE=${1:-"record"}
DEMO_SCRIPT=${2:-"$SCRIPT_DIR/../demo-90s.sh"}

mkdir -p "$OUTPUT_DIR"

log() {
    echo -e "${CYAN}[RECORD]${NC} $1"
}

success() {
    echo -e "  ${GREEN}✓${NC} $1"
}

error() {
    echo -e "  ${RED}✗${NC} $1"
    exit 1
}

check_dependencies() {
    local missing=()

    if ! command -v asciinema &> /dev/null; then
        missing+=("asciinema")
    fi

    if ! command -v agg &> /dev/null; then
        missing+=("agg (asciinema-agg)")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        echo -e "${YELLOW}Optional dependencies not found: ${missing[*]}${NC}"
        echo ""
        echo "Install with:"
        echo "  brew install asciinema"
        echo "  cargo install agg"
        echo ""
        return 1
    fi

    return 0
}

#######################################
# Record Demo
#######################################
record_demo() {
    local script=${1:-$DEMO_SCRIPT}
    local output_file="$OUTPUT_DIR/${RECORDING_NAME}.cast"

    log "Starting demo recording..."
    echo ""
    echo "  Recording to: $output_file"
    echo "  Script: $script"
    echo "  Idle time limit: ${IDLE_TIME_LIMIT}s"
    echo ""

    # Check if asciinema is available
    if ! command -v asciinema &> /dev/null; then
        log "asciinema not found, using script fallback"

        # Fallback to script command
        script -q "$OUTPUT_DIR/${RECORDING_NAME}.txt" bash -c "$script"
        success "Recording saved (text format)"
        return 0
    fi

    # Pre-recording setup
    echo -e "${BOLD}Recording will start in 3 seconds...${NC}"
    echo "Press Ctrl+C to stop recording early"
    sleep 3

    # Record with asciinema
    asciinema rec \
        --idle-time-limit "$IDLE_TIME_LIMIT" \
        --title "HELIX Demo - $(date +%Y-%m-%d)" \
        --command "$script" \
        "$output_file"

    if [ -f "$output_file" ]; then
        success "Recording saved: $output_file"

        # Generate metadata
        cat > "$OUTPUT_DIR/${RECORDING_NAME}.json" << EOF
{
    "name": "$RECORDING_NAME",
    "file": "$output_file",
    "script": "$script",
    "recorded_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "idle_time_limit": $IDLE_TIME_LIMIT,
    "duration_estimate": "90s"
}
EOF

        echo ""
        echo "  Playback: asciinema play $output_file"
        echo "  Upload: asciinema upload $output_file"
    else
        error "Recording failed"
    fi
}

#######################################
# Record with Automatic Pauses
#######################################
record_with_pauses() {
    local output_file="$OUTPUT_DIR/${RECORDING_NAME}_paused.cast"

    log "Recording with automatic pauses at key moments..."

    # Create a wrapper script that adds pauses
    local wrapper_script=$(mktemp)
    cat > "$wrapper_script" << 'WRAPPER_EOF'
#!/bin/bash

# Source the original script with pause injections
PAUSE_DURATION=${PAUSE_DURATION:-2}

pause_at_phase() {
    echo ""
    echo -e "\033[2m[PAUSE: $1]\033[0m"
    sleep $PAUSE_DURATION
}

# Hook into key moments
original_phase() {
    # Original phase function
    :
}

# Override to add pauses
phase() {
    pause_at_phase "Phase transition"
    echo ""
    echo -e "\033[1m\033[34m═══════════════════════════════════════════════════════════\033[0m"
    echo -e "\033[1m\033[34m $1\033[0m"
    echo -e "\033[1m\033[34m═══════════════════════════════════════════════════════════\033[0m"
}

# Run the demo
WRAPPER_EOF

    cat "$DEMO_SCRIPT" >> "$wrapper_script"
    chmod +x "$wrapper_script"

    # Record
    PAUSE_DURATION=${PAUSE_DURATION:-2}
    export PAUSE_DURATION

    asciinema rec \
        --idle-time-limit 5 \
        --title "HELIX Demo (with pauses)" \
        --command "$wrapper_script" \
        "$output_file"

    rm -f "$wrapper_script"

    if [ -f "$output_file" ]; then
        success "Recording with pauses saved: $output_file"
    fi
}

#######################################
# Play Recording
#######################################
play_recording() {
    local recording=${1:-$(ls -t "$OUTPUT_DIR"/*.cast 2>/dev/null | head -1)}

    if [ -z "$recording" ] || [ ! -f "$recording" ]; then
        error "No recording found. Specify a recording file or record first."
    fi

    log "Playing recording: $(basename "$recording")"
    echo ""

    if command -v asciinema &> /dev/null; then
        asciinema play -s "$PLAYBACK_SPEED" "$recording"
    else
        error "asciinema not installed"
    fi
}

#######################################
# Export to GIF/Video
#######################################
export_recording() {
    local recording=${1:-$(ls -t "$OUTPUT_DIR"/*.cast 2>/dev/null | head -1)}
    local format=${2:-"gif"}

    if [ -z "$recording" ] || [ ! -f "$recording" ]; then
        error "No recording found"
    fi

    local basename=$(basename "$recording" .cast)
    local output_file="$OUTPUT_DIR/${basename}.${format}"

    log "Exporting to $format: $output_file"

    case "$format" in
        "gif")
            if command -v agg &> /dev/null; then
                agg "$recording" "$output_file" \
                    --speed "$PLAYBACK_SPEED" \
                    --font-size 14 \
                    --theme monokai
                success "GIF exported: $output_file"
            else
                error "agg not installed. Install with: cargo install agg"
            fi
            ;;
        "svg")
            if command -v svg-term &> /dev/null; then
                svg-term --in "$recording" --out "$output_file"
                success "SVG exported: $output_file"
            else
                echo "  svg-term not installed. Install with: npm install -g svg-term-cli"
                # Fallback: use asciinema's built-in SVG
                asciinema upload "$recording" || true
            fi
            ;;
        "mp4")
            if command -v agg &> /dev/null && command -v ffmpeg &> /dev/null; then
                # First convert to gif, then to mp4
                local temp_gif=$(mktemp).gif
                agg "$recording" "$temp_gif" --speed "$PLAYBACK_SPEED"
                ffmpeg -i "$temp_gif" -pix_fmt yuv420p "$output_file" -y
                rm -f "$temp_gif"
                success "MP4 exported: $output_file"
            else
                error "agg and ffmpeg required for MP4 export"
            fi
            ;;
        *)
            error "Unsupported format: $format (use gif, svg, or mp4)"
            ;;
    esac
}

#######################################
# List Recordings
#######################################
list_recordings() {
    echo -e "${BOLD}Available Recordings:${NC}"
    echo ""

    if [ ! -d "$OUTPUT_DIR" ] || [ -z "$(ls -A "$OUTPUT_DIR"/*.cast 2>/dev/null)" ]; then
        echo "  (no recordings found)"
        echo ""
        echo "  Create one with: $0 record"
        return
    fi

    for recording in "$OUTPUT_DIR"/*.cast; do
        if [ -f "$recording" ]; then
            local name=$(basename "$recording" .cast)
            local size=$(du -h "$recording" | cut -f1)
            local date=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M" "$recording" 2>/dev/null || stat -c "%y" "$recording" 2>/dev/null | cut -d'.' -f1)

            echo "  • $name"
            echo "    Size: $size"
            echo "    Date: $date"

            # Check for exports
            for ext in gif svg mp4; do
                if [ -f "$OUTPUT_DIR/${name}.${ext}" ]; then
                    echo "    Export: ${name}.${ext}"
                fi
            done
            echo ""
        fi
    done
}

#######################################
# Create Presentation Mode Recording
#######################################
record_presentation() {
    log "Recording in presentation mode..."
    echo ""
    echo "  This mode creates a polished recording suitable for presentations."
    echo "  - Slower pace"
    echo "  - Clear transitions"
    echo "  - Automatic pauses"
    echo ""

    # Set presentation mode variables
    export DEMO_SPEED="slow"
    export PAUSE_DURATION=3
    export HELIX_DEMO_PRESENTATION=true

    RECORDING_NAME="${RECORDING_NAME}_presentation"
    IDLE_TIME_LIMIT=5

    # Create presentation wrapper
    local wrapper=$(mktemp)
    cat > "$wrapper" << 'EOF'
#!/bin/bash

# Presentation introduction
clear
echo ""
echo ""
echo "    ██╗  ██╗███████╗██╗     ██╗██╗  ██╗"
echo "    ██║  ██║██╔════╝██║     ██║╚██╗██╔╝"
echo "    ███████║█████╗  ██║     ██║ ╚███╔╝ "
echo "    ██╔══██║██╔══╝  ██║     ██║ ██╔██╗ "
echo "    ██║  ██║███████╗███████╗██║██╔╝ ██╗"
echo "    ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚═╝  ╚═╝"
echo ""
echo "         Trustless AI Training"
echo ""
echo ""
sleep 4

# Run main demo
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/demo-90s.sh"

# Presentation outro
echo ""
echo ""
echo "    Thank you for watching!"
echo ""
echo "    Learn more: github.com/helix-protocol/helix"
echo ""
sleep 3
EOF
    chmod +x "$wrapper"

    # Record
    asciinema rec \
        --idle-time-limit 5 \
        --title "HELIX - Trustless AI Training" \
        --command "$wrapper" \
        "$OUTPUT_DIR/${RECORDING_NAME}.cast"

    rm -f "$wrapper"

    success "Presentation recording complete"

    # Auto-export to GIF
    log "Auto-exporting to GIF..."
    export_recording "$OUTPUT_DIR/${RECORDING_NAME}.cast" "gif"
}

#######################################
# Quick Demo Recording (no deps needed)
#######################################
record_simple() {
    local output_file="$OUTPUT_DIR/${RECORDING_NAME}.txt"
    local timing_file="$OUTPUT_DIR/${RECORDING_NAME}.timing"

    log "Recording with simple script recorder..."

    # Use script command (available on all Unix systems)
    script -q -t "$timing_file" "$output_file" bash -c "$DEMO_SCRIPT" 2>/dev/null || \
    script -q "$output_file" bash -c "$DEMO_SCRIPT"

    success "Simple recording saved: $output_file"

    if [ -f "$timing_file" ]; then
        echo "  Timing file: $timing_file"
        echo "  Replay with: scriptreplay $timing_file $output_file"
    fi
}

#######################################
# Main
#######################################

# Banner
echo ""
echo -e "${BOLD}${BLUE}HELIX Demo Recording Tool${NC}"
echo ""

case "$MODE" in
    "record")
        check_dependencies || true
        record_demo "${2:-$DEMO_SCRIPT}"
        ;;
    "record-pauses")
        check_dependencies
        record_with_pauses
        ;;
    "record-presentation")
        check_dependencies
        record_presentation
        ;;
    "record-simple")
        record_simple
        ;;
    "play")
        check_dependencies
        play_recording "$2"
        ;;
    "export")
        check_dependencies
        export_recording "$2" "$3"
        ;;
    "list")
        list_recordings
        ;;
    *)
        echo "Usage: $0 <mode> [options]"
        echo ""
        echo "Modes:"
        echo "  record [script]      Record a demo script"
        echo "  record-pauses        Record with automatic pauses"
        echo "  record-presentation  Record polished presentation"
        echo "  record-simple        Record with basic script command"
        echo "  play [file]          Play back a recording"
        echo "  export [file] [fmt]  Export to gif/svg/mp4"
        echo "  list                 List available recordings"
        echo ""
        echo "Environment variables:"
        echo "  OUTPUT_DIR           Output directory (default: .helix/recordings)"
        echo "  RECORDING_NAME       Recording name prefix"
        echo "  PLAYBACK_SPEED       Playback speed multiplier (default: 1.0)"
        echo "  IDLE_TIME_LIMIT      Max idle time in seconds (default: 2)"
        echo ""
        echo "Dependencies:"
        echo "  asciinema           Terminal recording (brew install asciinema)"
        echo "  agg                 GIF/video export (cargo install agg)"
        echo "  svg-term-cli        SVG export (npm install -g svg-term-cli)"
        echo ""
        exit 1
        ;;
esac
