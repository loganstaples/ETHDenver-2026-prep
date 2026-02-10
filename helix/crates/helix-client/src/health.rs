//! HELIX Health Check Module
//!
//! Provides real health monitoring for nodes, contracts, and network.
//! Performs actual HTTP/TCP liveness probes instead of returning mocked data.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;
use colored::*;
use serde::{Deserialize, Serialize};

/// Health check target
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthTarget {
    All,
    Nodes,
    Contracts,
    Network,
    Storage,
}

/// Overall health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    pub overall_status: HealthStatus,
    pub checks: Vec<HealthCheckResult>,
    pub duration_ms: u64,
    pub timestamp: String,
}

/// Health status levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

impl HealthStatus {
    /// Get colored string representation
    pub fn colored_string(&self) -> ColoredString {
        match self {
            HealthStatus::Healthy => "HEALTHY".green(),
            HealthStatus::Degraded => "DEGRADED".yellow(),
            HealthStatus::Unhealthy => "UNHEALTHY".red(),
            HealthStatus::Unknown => "UNKNOWN".dimmed(),
        }
    }
}

/// Result of a single health check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    pub name: String,
    pub category: String,
    pub status: HealthStatus,
    pub message: Option<String>,
    pub duration_ms: u64,
    pub details: Option<serde_json::Value>,
}

/// Health checker that performs real liveness probes
pub struct HealthChecker {
    timeout: Duration,
    rpc_url: String,
    http_client: reqwest::Client,
    data_dir: PathBuf,
    ipfs_gateway: Option<String>,
}

impl HealthChecker {
    /// Create a new health checker
    pub fn new(timeout: Duration, rpc_url: &str) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            timeout,
            rpc_url: rpc_url.to_string(),
            http_client,
            data_dir: dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".helix")
                .join("data"),
            ipfs_gateway: None,
        }
    }

    /// Set the IPFS gateway URL for storage health checks
    pub fn with_ipfs_gateway(mut self, gateway: &str) -> Self {
        self.ipfs_gateway = Some(gateway.to_string());
        self
    }

    /// Set the data directory for storage health checks
    pub fn with_data_dir(mut self, dir: PathBuf) -> Self {
        self.data_dir = dir;
        self
    }

    /// Run all health checks
    pub async fn check_all(&self) -> HealthReport {
        let start = Instant::now();
        let mut checks = Vec::new();

        // Node checks
        checks.extend(self.check_nodes().await);

        // Contract checks
        checks.extend(self.check_contracts().await);

        // Network checks
        checks.extend(self.check_network().await);

        // Storage checks
        checks.extend(self.check_storage().await);

        // Determine overall status
        let overall_status = Self::aggregate_status(&checks);

        HealthReport {
            overall_status,
            checks,
            duration_ms: start.elapsed().as_millis() as u64,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Check specific target
    pub async fn check(&self, target: HealthTarget) -> HealthReport {
        let start = Instant::now();

        let checks = match target {
            HealthTarget::All => {
                let mut all = Vec::new();
                all.extend(self.check_nodes().await);
                all.extend(self.check_contracts().await);
                all.extend(self.check_network().await);
                all.extend(self.check_storage().await);
                all
            }
            HealthTarget::Nodes => self.check_nodes().await,
            HealthTarget::Contracts => self.check_contracts().await,
            HealthTarget::Network => self.check_network().await,
            HealthTarget::Storage => self.check_storage().await,
        };

        let overall_status = Self::aggregate_status(&checks);

        HealthReport {
            overall_status,
            checks,
            duration_ms: start.elapsed().as_millis() as u64,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Perform an HTTP liveness ping to the given URL.
    /// Returns (is_alive, latency_ms, error_message).
    async fn http_ping(&self, url: &str) -> (bool, u64, Option<String>) {
        let start = Instant::now();
        match tokio::time::timeout(self.timeout, self.http_client.get(url).send()).await {
            Ok(Ok(resp)) => {
                let latency = start.elapsed().as_millis() as u64;
                let status = resp.status();
                if status.is_success() || status.is_informational() {
                    (true, latency, None)
                } else {
                    (false, latency, Some(format!("HTTP {}", status)))
                }
            }
            Ok(Err(e)) => {
                let latency = start.elapsed().as_millis() as u64;
                (false, latency, Some(e.to_string()))
            }
            Err(_) => {
                let latency = start.elapsed().as_millis() as u64;
                (false, latency, Some("Connection timed out".to_string()))
            }
        }
    }

    /// Perform a TCP liveness probe to the given address.
    /// Returns (is_alive, latency_ms, error_message).
    async fn tcp_ping(&self, addr: &str) -> (bool, u64, Option<String>) {
        let start = Instant::now();
        match tokio::time::timeout(self.timeout, tokio::net::TcpStream::connect(addr)).await {
            Ok(Ok(_)) => {
                let latency = start.elapsed().as_millis() as u64;
                (true, latency, None)
            }
            Ok(Err(e)) => {
                let latency = start.elapsed().as_millis() as u64;
                (false, latency, Some(e.to_string()))
            }
            Err(_) => {
                let latency = start.elapsed().as_millis() as u64;
                (false, latency, Some("Connection timed out".to_string()))
            }
        }
    }

    /// Send a JSON-RPC request and return the result as serde_json::Value.
    async fn json_rpc_call(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1
        });

        let resp = tokio::time::timeout(
            self.timeout,
            self.http_client.post(&self.rpc_url).json(&body).send(),
        )
        .await
        .map_err(|_| "RPC request timed out".to_string())?
        .map_err(|e| format!("RPC request failed: {}", e))?;

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| format!("Invalid RPC response: {}", e))?;

        if let Some(err) = json.get("error") {
            return Err(format!("RPC error: {}", err));
        }

        json.get("result")
            .cloned()
            .ok_or_else(|| "No result in RPC response".to_string())
    }

    /// Check node health via real HTTP liveness ping
    async fn check_nodes(&self) -> Vec<HealthCheckResult> {
        let mut results = Vec::new();

        // Probe the HELIX node RPC endpoint (JSON-RPC liveness)
        let node_rpc_addr = self
            .rpc_url
            .strip_prefix("http://")
            .or_else(|| self.rpc_url.strip_prefix("https://"))
            .unwrap_or(&self.rpc_url);

        let start = Instant::now();
        let (alive, latency, err) = self.tcp_ping(node_rpc_addr).await;
        let duration = start.elapsed().as_millis() as u64;

        results.push(HealthCheckResult {
            name: "RPC Endpoint Liveness".to_string(),
            category: "nodes".to_string(),
            status: if alive {
                HealthStatus::Healthy
            } else {
                HealthStatus::Unhealthy
            },
            message: if alive {
                Some(format!("Reachable ({}ms)", latency))
            } else {
                Some(format!(
                    "Unreachable: {}",
                    err.unwrap_or_else(|| "unknown".to_string())
                ))
            },
            duration_ms: duration,
            details: Some(serde_json::json!({
                "endpoint": self.rpc_url,
                "latency_ms": latency,
                "alive": alive,
            })),
        });

        // Process liveness: check if helix-node processes exist via /proc or ps
        let start = Instant::now();
        let process_check = tokio::task::spawn_blocking(|| {
            // Check for helix-node processes via command
            std::process::Command::new("pgrep")
                .arg("-f")
                .arg("helix-node")
                .output()
                .map(|out| {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    let pids: Vec<&str> = stdout.trim().lines().collect();
                    (out.status.success(), pids.len(), pids.join(","))
                })
                .unwrap_or((false, 0, String::new()))
        })
        .await
        .unwrap_or((false, 0, String::new()));
        let duration = start.elapsed().as_millis() as u64;

        let (found, count, pids) = process_check;
        results.push(HealthCheckResult {
            name: "HELIX Node Processes".to_string(),
            category: "nodes".to_string(),
            status: if found {
                HealthStatus::Healthy
            } else {
                HealthStatus::Degraded
            },
            message: if found {
                Some(format!("{} process(es) running (pids: {})", count, pids))
            } else {
                Some("No helix-node processes found".to_string())
            },
            duration_ms: duration,
            details: Some(serde_json::json!({
                "found": found,
                "count": count,
                "pids": pids,
            })),
        });

        results
    }

    /// Check contract health by verifying RPC responds to eth_chainId
    async fn check_contracts(&self) -> Vec<HealthCheckResult> {
        let mut results = Vec::new();

        // Verify the chain is reachable by calling eth_chainId
        let start = Instant::now();
        let chain_check = self.json_rpc_call("eth_chainId", serde_json::json!([])).await;
        let duration = start.elapsed().as_millis() as u64;

        match chain_check {
            Ok(chain_id_hex) => {
                let chain_id_str = chain_id_hex.as_str().unwrap_or("0x0");
                let chain_id =
                    u64::from_str_radix(chain_id_str.trim_start_matches("0x"), 16).unwrap_or(0);

                results.push(HealthCheckResult {
                    name: "Chain Connectivity".to_string(),
                    category: "contracts".to_string(),
                    status: HealthStatus::Healthy,
                    message: Some(format!("Chain ID: {} ({})", chain_id, self.rpc_url)),
                    duration_ms: duration,
                    details: Some(serde_json::json!({
                        "chain_id": chain_id,
                        "rpc_url": self.rpc_url,
                        "latency_ms": duration,
                    })),
                });
            }
            Err(e) => {
                results.push(HealthCheckResult {
                    name: "Chain Connectivity".to_string(),
                    category: "contracts".to_string(),
                    status: HealthStatus::Unhealthy,
                    message: Some(format!("Cannot reach chain: {}", e)),
                    duration_ms: duration,
                    details: Some(serde_json::json!({
                        "rpc_url": self.rpc_url,
                        "error": e.to_string(),
                    })),
                });
            }
        }

        results
    }

    /// Check network health with real RPC calls (block number, gas price)
    async fn check_network(&self) -> Vec<HealthCheckResult> {
        let mut results = Vec::new();

        // RPC Connection liveness via HTTP ping
        let start = Instant::now();
        let (alive, latency, err) = self.http_ping(&self.rpc_url).await;
        let duration = start.elapsed().as_millis() as u64;

        results.push(HealthCheckResult {
            name: "RPC Connection".to_string(),
            category: "network".to_string(),
            status: if alive {
                HealthStatus::Healthy
            } else {
                HealthStatus::Unhealthy
            },
            message: if alive {
                Some(format!("{} ({}ms)", self.rpc_url, latency))
            } else {
                Some(format!(
                    "{} — {}",
                    self.rpc_url,
                    err.unwrap_or_else(|| "unreachable".to_string())
                ))
            },
            duration_ms: duration,
            details: Some(serde_json::json!({
                "url": self.rpc_url,
                "latency_ms": latency,
                "alive": alive,
            })),
        });

        // Block height via eth_blockNumber
        let start = Instant::now();
        let block_check = self
            .json_rpc_call("eth_blockNumber", serde_json::json!([]))
            .await;
        let duration = start.elapsed().as_millis() as u64;

        match block_check {
            Ok(block_hex) => {
                let block_str = block_hex.as_str().unwrap_or("0x0");
                let block =
                    u64::from_str_radix(block_str.trim_start_matches("0x"), 16).unwrap_or(0);

                results.push(HealthCheckResult {
                    name: "Block Height".to_string(),
                    category: "network".to_string(),
                    status: HealthStatus::Healthy,
                    message: Some(format!("{}", block)),
                    duration_ms: duration,
                    details: Some(serde_json::json!({
                        "height": block,
                        "latency_ms": duration,
                    })),
                });
            }
            Err(e) => {
                results.push(HealthCheckResult {
                    name: "Block Height".to_string(),
                    category: "network".to_string(),
                    status: HealthStatus::Unhealthy,
                    message: Some(format!("Failed: {}", e)),
                    duration_ms: duration,
                    details: None,
                });
            }
        }

        // Gas price via eth_gasPrice
        let start = Instant::now();
        let gas_check = self
            .json_rpc_call("eth_gasPrice", serde_json::json!([]))
            .await;
        let duration = start.elapsed().as_millis() as u64;

        match gas_check {
            Ok(gas_hex) => {
                let gas_str = gas_hex.as_str().unwrap_or("0x0");
                let gas_wei =
                    u64::from_str_radix(gas_str.trim_start_matches("0x"), 16).unwrap_or(0);
                let gas_gwei = gas_wei / 1_000_000_000;

                results.push(HealthCheckResult {
                    name: "Gas Price".to_string(),
                    category: "network".to_string(),
                    status: HealthStatus::Healthy,
                    message: Some(format!("{} gwei", gas_gwei)),
                    duration_ms: duration,
                    details: Some(serde_json::json!({
                        "gwei": gas_gwei,
                        "wei": gas_wei,
                    })),
                });
            }
            Err(e) => {
                results.push(HealthCheckResult {
                    name: "Gas Price".to_string(),
                    category: "network".to_string(),
                    status: HealthStatus::Degraded,
                    message: Some(format!("Failed: {}", e)),
                    duration_ms: duration,
                    details: None,
                });
            }
        }

        results
    }

    /// Check storage health with real filesystem and IPFS probes
    async fn check_storage(&self) -> Vec<HealthCheckResult> {
        let mut results = Vec::new();

        // Data directory existence and writability
        let start = Instant::now();
        let data_dir = self.data_dir.clone();
        let dir_check = tokio::task::spawn_blocking(move || {
            let exists = data_dir.exists();
            let writable = if exists {
                let test_file = data_dir.join(".health_check_probe");
                match std::fs::write(&test_file, b"ok") {
                    Ok(()) => {
                        let _ = std::fs::remove_file(&test_file);
                        true
                    }
                    Err(_) => false,
                }
            } else {
                false
            };
            (exists, writable, data_dir.display().to_string())
        })
        .await
        .unwrap_or((false, false, "unknown".to_string()));
        let duration = start.elapsed().as_millis() as u64;

        let (exists, writable, path_str) = dir_check;
        let status = if exists && writable {
            HealthStatus::Healthy
        } else if exists {
            HealthStatus::Degraded
        } else {
            HealthStatus::Unhealthy
        };

        results.push(HealthCheckResult {
            name: "Data Directory".to_string(),
            category: "storage".to_string(),
            status,
            message: Some(format!(
                "{} (exists={}, writable={})",
                path_str, exists, writable
            )),
            duration_ms: duration,
            details: Some(serde_json::json!({
                "path": path_str,
                "exists": exists,
                "writable": writable,
            })),
        });

        // IPFS gateway probe (if configured)
        if let Some(ref gateway) = self.ipfs_gateway {
            let start = Instant::now();
            // Probe the IPFS API version endpoint
            let probe_url = format!("{}/api/v0/version", gateway.trim_end_matches('/'));
            let (alive, latency, err) = self.http_ping(&probe_url).await;
            let duration = start.elapsed().as_millis() as u64;

            results.push(HealthCheckResult {
                name: "IPFS Gateway".to_string(),
                category: "storage".to_string(),
                status: if alive {
                    HealthStatus::Healthy
                } else {
                    HealthStatus::Degraded
                },
                message: if alive {
                    Some(format!("Connected ({}ms)", latency))
                } else {
                    Some(format!(
                        "Unreachable: {}",
                        err.unwrap_or_else(|| "unknown".to_string())
                    ))
                },
                duration_ms: duration,
                details: Some(serde_json::json!({
                    "gateway": gateway,
                    "alive": alive,
                    "latency_ms": latency,
                })),
            });
        }

        results
    }

    /// Aggregate status from multiple checks
    fn aggregate_status(checks: &[HealthCheckResult]) -> HealthStatus {
        let mut unhealthy = 0;
        let mut degraded = 0;

        for check in checks {
            match check.status {
                HealthStatus::Unhealthy => unhealthy += 1,
                HealthStatus::Degraded => degraded += 1,
                _ => {}
            }
        }

        if unhealthy > 0 {
            HealthStatus::Unhealthy
        } else if degraded > 0 {
            HealthStatus::Degraded
        } else if checks.is_empty() {
            HealthStatus::Unknown
        } else {
            HealthStatus::Healthy
        }
    }
}

/// Display health report to console
pub fn display_health_report(report: &HealthReport, detailed: bool) {
    println!("{}", "═".repeat(60).cyan());
    println!(" {} Health Check Report", "HELIX".cyan().bold());
    println!("{}", "═".repeat(60).cyan());
    println!();
    println!("Overall Status: {}", report.overall_status.colored_string());
    println!("Duration: {} ms", report.duration_ms);
    println!("Timestamp: {}", report.timestamp);
    println!();

    // Group by category
    let mut categories: std::collections::HashMap<&str, Vec<&HealthCheckResult>> =
        std::collections::HashMap::new();

    for check in &report.checks {
        categories
            .entry(&check.category)
            .or_insert_with(Vec::new)
            .push(check);
    }

    for (category, checks) in categories {
        println!("{}:", category.to_uppercase().yellow().bold());

        for check in checks {
            let status_icon = match check.status {
                HealthStatus::Healthy => "✓".green(),
                HealthStatus::Degraded => "⚠".yellow(),
                HealthStatus::Unhealthy => "✗".red(),
                HealthStatus::Unknown => "?".dimmed(),
            };

            let message = check.message.as_deref().unwrap_or("");
            println!("  {} {} {}", status_icon, check.name, message.dimmed());

            if detailed {
                if let Some(details) = &check.details {
                    let json = serde_json::to_string_pretty(details).unwrap_or_default();
                    for line in json.lines() {
                        println!("      {}", line.dimmed());
                    }
                }
            }
        }
        println!();
    }
}
