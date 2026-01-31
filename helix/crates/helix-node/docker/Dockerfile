# HELIX Node Docker Image
# Multi-stage build for optimized image size

# Stage 1: Build
FROM rust:1.75-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /helix

# Copy Cargo workspace files
COPY Cargo.toml Cargo.lock ./

# Copy all crate manifests
COPY crates/helix-core/Cargo.toml crates/helix-core/
COPY crates/helix-avm/Cargo.toml crates/helix-avm/
COPY crates/helix-prover/Cargo.toml crates/helix-prover/
COPY crates/helix-mpc/Cargo.toml crates/helix-mpc/
COPY crates/helix-node/Cargo.toml crates/helix-node/
COPY crates/helix-client/Cargo.toml crates/helix-client/

# Create dummy source files for caching dependencies
RUN mkdir -p crates/helix-core/src && echo "pub fn dummy() {}" > crates/helix-core/src/lib.rs \
    && mkdir -p crates/helix-avm/src && echo "pub fn dummy() {}" > crates/helix-avm/src/lib.rs \
    && mkdir -p crates/helix-prover/src && echo "pub fn dummy() {}" > crates/helix-prover/src/lib.rs \
    && mkdir -p crates/helix-mpc/src && echo "pub fn dummy() {}" > crates/helix-mpc/src/lib.rs \
    && mkdir -p crates/helix-node/src && echo "pub fn main() {}" > crates/helix-node/src/main.rs \
    && mkdir -p crates/helix-client/src && echo "pub fn main() {}" > crates/helix-client/src/main.rs

# Build dependencies only (this layer will be cached)
RUN cargo build --release -p helix-node 2>/dev/null || true

# Now copy actual source code
COPY crates/ crates/
COPY contracts/ contracts/

# Remove the dummy files and rebuild
RUN rm -rf target/release/.fingerprint/helix-* \
    && rm -rf target/release/deps/helix* \
    && rm -rf target/release/deps/libhelix*

# Build the actual binaries
RUN cargo build --release -p helix-node

# Stage 2: Runtime
FROM debian:bookworm-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary from builder
COPY --from=builder /helix/target/release/helix-node /app/helix-node

# Create data directory
RUN mkdir -p /app/data /app/logs /app/config

# Create non-root user
RUN useradd -r -u 1000 helix && chown -R helix:helix /app
USER helix

# Environment variables
ENV RUST_LOG=info
ENV HELIX_DATA_DIR=/app/data
ENV HELIX_LOG_DIR=/app/logs
ENV HELIX_CONFIG_DIR=/app/config

# Expose ports
# 9000: P2P communication
# 9001: HTTP API
# 5353: mDNS (UDP)
EXPOSE 9000 9001 5353/udp

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9001/health || exit 1

# Default command
ENTRYPOINT ["/app/helix-node"]
CMD ["--help"]
