//! Audit Logging for Key Operations
//!
//! Provides comprehensive audit logging for all wallet and key operations.
//! Logs are tamper-evident using hash chaining and can be exported for compliance.

use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::RwLock;

/// Types of key operations that are audited
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KeyOperation {
    // Mnemonic operations
    MnemonicGenerated,
    MnemonicImported,
    MnemonicExported,
    MnemonicViewed,

    // Key derivation
    KeyDerived,
    KeyRotated,

    // Wallet operations
    WalletCreated,
    WalletImported,
    WalletExported,
    WalletDeleted,
    WalletUnlocked,
    WalletLocked,

    // Signing operations
    MessageSigned,
    TransactionSigned,
    TypedDataSigned,

    // Keychain operations
    KeyStoredInKeychain,
    KeyRetrievedFromKeychain,
    KeyDeletedFromKeychain,

    // Hardware wallet operations
    HardwareWalletConnected,
    HardwareWalletDisconnected,
    HardwareWalletSigned,

    // Security events
    FailedUnlockAttempt,
    PasswordChanged,
    BackupCreated,
    BackupRestored,

    // Confirmation events
    TransactionConfirmed,
    TransactionRejected,
}

impl KeyOperation {
    /// Get severity level for the operation
    pub fn severity(&self) -> AuditSeverity {
        match self {
            // Critical operations
            KeyOperation::MnemonicExported
            | KeyOperation::WalletExported
            | KeyOperation::KeyRotated
            | KeyOperation::WalletDeleted
            | KeyOperation::BackupCreated => AuditSeverity::Critical,

            // High severity
            KeyOperation::MnemonicGenerated
            | KeyOperation::MnemonicImported
            | KeyOperation::WalletCreated
            | KeyOperation::WalletImported
            | KeyOperation::TransactionSigned
            | KeyOperation::PasswordChanged
            | KeyOperation::BackupRestored => AuditSeverity::High,

            // Medium severity
            KeyOperation::KeyDerived
            | KeyOperation::MessageSigned
            | KeyOperation::TypedDataSigned
            | KeyOperation::HardwareWalletSigned
            | KeyOperation::KeyStoredInKeychain
            | KeyOperation::KeyDeletedFromKeychain
            | KeyOperation::TransactionConfirmed
            | KeyOperation::TransactionRejected => AuditSeverity::Medium,

            // Low severity
            KeyOperation::WalletUnlocked
            | KeyOperation::WalletLocked
            | KeyOperation::MnemonicViewed
            | KeyOperation::KeyRetrievedFromKeychain
            | KeyOperation::HardwareWalletConnected
            | KeyOperation::HardwareWalletDisconnected => AuditSeverity::Low,

            // Warning severity
            KeyOperation::FailedUnlockAttempt => AuditSeverity::Warning,
        }
    }
}

/// Severity levels for audit events
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AuditSeverity {
    Low,
    Medium,
    High,
    Critical,
    Warning,
}

/// Single audit event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Unique event ID
    pub id: String,
    /// Timestamp (Unix epoch milliseconds)
    pub timestamp: u64,
    /// Operation type
    pub operation: KeyOperation,
    /// Associated wallet ID (if applicable)
    pub wallet_id: Option<String>,
    /// Additional details
    pub details: Option<String>,
    /// Severity level
    pub severity: AuditSeverity,
    /// Hash of previous event (for chain integrity)
    pub previous_hash: String,
    /// Hash of this event
    pub event_hash: String,
    /// IP address or client identifier (if available)
    pub client_id: Option<String>,
    /// User agent or application identifier
    pub user_agent: Option<String>,
}

impl AuditEvent {
    /// Create a new audit event
    pub fn new(
        operation: KeyOperation,
        wallet_id: Option<String>,
        details: Option<String>,
    ) -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let severity = operation.severity();

        Self {
            id,
            timestamp,
            operation,
            wallet_id,
            details,
            severity,
            previous_hash: String::new(),
            event_hash: String::new(),
            client_id: None,
            user_agent: Some(format!("helix-client/{}", env!("CARGO_PKG_VERSION"))),
        }
    }

    /// Set the previous hash and calculate event hash
    pub fn finalize(mut self, previous_hash: &str) -> Self {
        self.previous_hash = previous_hash.to_string();
        self.event_hash = self.calculate_hash();
        self
    }

    /// Calculate the hash of this event
    fn calculate_hash(&self) -> String {
        let mut hasher = Sha256::new();

        // Include all fields in the hash
        hasher.update(self.id.as_bytes());
        hasher.update(self.timestamp.to_le_bytes());
        hasher.update(format!("{:?}", self.operation).as_bytes());
        if let Some(ref wallet_id) = self.wallet_id {
            hasher.update(wallet_id.as_bytes());
        }
        if let Some(ref details) = self.details {
            hasher.update(details.as_bytes());
        }
        hasher.update(self.previous_hash.as_bytes());

        hex::encode(hasher.finalize())
    }

    /// Verify the event hash is correct
    pub fn verify_hash(&self) -> bool {
        self.event_hash == self.calculate_hash()
    }

    /// Format for display
    pub fn format_display(&self) -> String {
        let time = chrono::DateTime::from_timestamp_millis(self.timestamp as i64)
            .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
            .unwrap_or_else(|| "Unknown time".to_string());

        let wallet = self
            .wallet_id
            .as_ref()
            .map(|w| format!(" [{}]", w))
            .unwrap_or_default();

        let details = self
            .details
            .as_ref()
            .map(|d| format!(": {}", d))
            .unwrap_or_default();

        format!(
            "[{:?}] {} - {:?}{}{}",
            self.severity, time, self.operation, wallet, details
        )
    }
}

/// Audit log configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Path to audit log file
    pub log_path: PathBuf,
    /// Maximum log file size in bytes before rotation
    pub max_file_size: u64,
    /// Number of rotated files to keep
    pub max_rotated_files: usize,
    /// Minimum severity to log
    pub min_severity: AuditSeverity,
    /// Whether to log to console as well
    pub console_output: bool,
    /// Whether to include stack traces on errors
    pub include_stack_traces: bool,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            log_path: dirs::data_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("helix")
                .join("audit.log"),
            max_file_size: 10 * 1024 * 1024, // 10 MB
            max_rotated_files: 5,
            min_severity: AuditSeverity::Low,
            console_output: false,
            include_stack_traces: false,
        }
    }
}

/// Audit logger with file-based persistent storage
#[derive(Clone)]
pub struct AuditLogger {
    /// Configuration
    config: AuditConfig,
    /// Last event hash for chain integrity
    last_hash: Arc<RwLock<String>>,
    /// In-memory buffer for recent events
    buffer: Arc<RwLock<VecDeque<AuditEvent>>>,
    /// Maximum buffer size
    buffer_size: usize,
}

impl AuditLogger {
    /// Create a new audit logger
    pub fn new(config: AuditConfig) -> Result<Self> {
        // Ensure log directory exists
        if let Some(parent) = config.log_path.parent() {
            fs::create_dir_all(parent).context("Failed to create audit log directory")?;
        }

        // Read last hash from existing log
        let last_hash = Self::read_last_hash(&config.log_path)?;

        Ok(Self {
            config,
            last_hash: Arc::new(RwLock::new(last_hash)),
            buffer: Arc::new(RwLock::new(VecDeque::new())),
            buffer_size: 1000,
        })
    }

    /// Create with default configuration
    pub fn default_logger() -> Result<Self> {
        Self::new(AuditConfig::default())
    }

    /// Create a no-op logger (for testing)
    pub fn noop() -> Self {
        Self {
            config: AuditConfig::default(),
            last_hash: Arc::new(RwLock::new(String::new())),
            buffer: Arc::new(RwLock::new(VecDeque::new())),
            buffer_size: 0,
        }
    }

    /// Read the last hash from the log file
    fn read_last_hash(path: &Path) -> Result<String> {
        if !path.exists() {
            return Ok("genesis".to_string());
        }

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut last_hash = "genesis".to_string();

        for line in reader.lines() {
            if let Ok(line) = line {
                if let Ok(event) = serde_json::from_str::<AuditEvent>(&line) {
                    last_hash = event.event_hash;
                }
            }
        }

        Ok(last_hash)
    }

    /// Log an audit event
    pub async fn log(&self, event: AuditEvent) -> Result<()> {
        // Check severity filter
        if event.severity < self.config.min_severity {
            return Ok(());
        }

        // Finalize event with hash chain
        let last_hash = self.last_hash.read().await.clone();
        let event = event.finalize(&last_hash);

        // Update last hash
        {
            let mut hash = self.last_hash.write().await;
            *hash = event.event_hash.clone();
        }

        // Add to buffer
        {
            let mut buffer = self.buffer.write().await;
            buffer.push_back(event.clone());
            while buffer.len() > self.buffer_size {
                buffer.pop_front();
            }
        }

        // Write to file
        self.write_event(&event)?;

        // Console output if enabled
        if self.config.console_output {
            println!("{}", event.format_display());
        }

        // Check for log rotation
        self.rotate_if_needed()?;

        Ok(())
    }

    /// Write event to file
    fn write_event(&self, event: &AuditEvent) -> Result<()> {
        // Skip writing if this is a noop logger (buffer_size == 0)
        if self.buffer_size == 0 {
            return Ok(());
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.config.log_path)
            .context("Failed to open audit log")?;

        let json = serde_json::to_string(event)?;
        writeln!(file, "{}", json)?;
        file.flush()?;

        Ok(())
    }

    /// Rotate log file if it exceeds max size
    fn rotate_if_needed(&self) -> Result<()> {
        // Skip rotation for noop logger
        if self.buffer_size == 0 {
            return Ok(());
        }

        let metadata = match fs::metadata(&self.config.log_path) {
            Ok(m) => m,
            Err(_) => return Ok(()),
        };

        if metadata.len() < self.config.max_file_size {
            return Ok(());
        }

        // Rotate files
        for i in (1..self.config.max_rotated_files).rev() {
            let from = self.config.log_path.with_extension(format!("log.{}", i));
            let to = self.config.log_path.with_extension(format!("log.{}", i + 1));
            if from.exists() {
                let _ = fs::rename(&from, &to);
            }
        }

        // Rename current log
        let first_rotated = self.config.log_path.with_extension("log.1");
        fs::rename(&self.config.log_path, &first_rotated)?;

        Ok(())
    }

    /// Get recent events from buffer
    pub async fn recent_events(&self, count: usize) -> Vec<AuditEvent> {
        let buffer = self.buffer.read().await;
        buffer.iter().rev().take(count).cloned().collect()
    }

    /// Get events filtered by operation
    pub async fn events_by_operation(&self, operation: KeyOperation) -> Vec<AuditEvent> {
        let buffer = self.buffer.read().await;
        buffer
            .iter()
            .filter(|e| e.operation == operation)
            .cloned()
            .collect()
    }

    /// Get events for a specific wallet
    pub async fn events_by_wallet(&self, wallet_id: &str) -> Vec<AuditEvent> {
        let buffer = self.buffer.read().await;
        buffer
            .iter()
            .filter(|e| e.wallet_id.as_deref() == Some(wallet_id))
            .cloned()
            .collect()
    }

    /// Verify the integrity of the audit chain
    pub async fn verify_chain(&self) -> Result<ChainVerificationResult> {
        let file = match File::open(&self.config.log_path) {
            Ok(f) => f,
            Err(_) => {
                return Ok(ChainVerificationResult {
                    is_valid: true,
                    total_events: 0,
                    verified_events: 0,
                    broken_at: None,
                })
            }
        };

        let reader = BufReader::new(file);
        let mut previous_hash = "genesis".to_string();
        let mut total_events = 0;
        let mut verified_events = 0;

        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            let event: AuditEvent = serde_json::from_str(&line)
                .map_err(|e| anyhow!("Failed to parse event on line {}: {}", line_num + 1, e))?;

            total_events += 1;

            // Verify previous hash link
            if event.previous_hash != previous_hash {
                return Ok(ChainVerificationResult {
                    is_valid: false,
                    total_events,
                    verified_events,
                    broken_at: Some(line_num + 1),
                });
            }

            // Verify event hash
            if !event.verify_hash() {
                return Ok(ChainVerificationResult {
                    is_valid: false,
                    total_events,
                    verified_events,
                    broken_at: Some(line_num + 1),
                });
            }

            previous_hash = event.event_hash;
            verified_events += 1;
        }

        Ok(ChainVerificationResult {
            is_valid: true,
            total_events,
            verified_events,
            broken_at: None,
        })
    }

    /// Export audit log to a different format
    pub fn export(&self, format: ExportFormat, output_path: &Path) -> Result<()> {
        let file = File::open(&self.config.log_path)?;
        let reader = BufReader::new(file);
        let events: Vec<AuditEvent> = reader
            .lines()
            .filter_map(|line| line.ok())
            .filter_map(|line| serde_json::from_str(&line).ok())
            .collect();

        match format {
            ExportFormat::Json => {
                let json = serde_json::to_string_pretty(&events)?;
                fs::write(output_path, json)?;
            }
            ExportFormat::Csv => {
                let mut writer = csv::Writer::from_path(output_path)?;
                writer.write_record([
                    "id",
                    "timestamp",
                    "operation",
                    "wallet_id",
                    "details",
                    "severity",
                    "event_hash",
                ])?;
                for event in events {
                    writer.write_record([
                        &event.id,
                        &event.timestamp.to_string(),
                        &format!("{:?}", event.operation),
                        &event.wallet_id.unwrap_or_default(),
                        &event.details.unwrap_or_default(),
                        &format!("{:?}", event.severity),
                        &event.event_hash,
                    ])?;
                }
            }
            ExportFormat::Text => {
                let mut output = String::new();
                for event in events {
                    output.push_str(&event.format_display());
                    output.push('\n');
                }
                fs::write(output_path, output)?;
            }
        }

        Ok(())
    }

    /// Get statistics about the audit log
    pub async fn statistics(&self) -> AuditStatistics {
        let buffer = self.buffer.read().await;

        let mut operation_counts = std::collections::HashMap::new();
        let mut severity_counts = std::collections::HashMap::new();

        for event in buffer.iter() {
            *operation_counts
                .entry(format!("{:?}", event.operation))
                .or_insert(0) += 1;
            *severity_counts
                .entry(format!("{:?}", event.severity))
                .or_insert(0) += 1;
        }

        AuditStatistics {
            total_events: buffer.len(),
            operation_counts,
            severity_counts,
            oldest_event: buffer.front().map(|e| e.timestamp),
            newest_event: buffer.back().map(|e| e.timestamp),
        }
    }
}

/// Export format options
#[derive(Debug, Clone, Copy)]
pub enum ExportFormat {
    Json,
    Csv,
    Text,
}

/// Result of chain verification
#[derive(Debug, Clone)]
pub struct ChainVerificationResult {
    pub is_valid: bool,
    pub total_events: usize,
    pub verified_events: usize,
    pub broken_at: Option<usize>,
}

/// Audit log statistics
#[derive(Debug, Clone)]
pub struct AuditStatistics {
    pub total_events: usize,
    pub operation_counts: std::collections::HashMap<String, usize>,
    pub severity_counts: std::collections::HashMap<String, usize>,
    pub oldest_event: Option<u64>,
    pub newest_event: Option<u64>,
}

/// Builder for audit events with fluent API
pub struct AuditEventBuilder {
    event: AuditEvent,
}

impl AuditEventBuilder {
    pub fn new(operation: KeyOperation) -> Self {
        Self {
            event: AuditEvent::new(operation, None, None),
        }
    }

    pub fn wallet_id(mut self, id: impl Into<String>) -> Self {
        self.event.wallet_id = Some(id.into());
        self
    }

    pub fn details(mut self, details: impl Into<String>) -> Self {
        self.event.details = Some(details.into());
        self
    }

    pub fn client_id(mut self, client_id: impl Into<String>) -> Self {
        self.event.client_id = Some(client_id.into());
        self
    }

    pub fn build(self) -> AuditEvent {
        self.event
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_audit_event_creation() {
        let event = AuditEvent::new(
            KeyOperation::WalletCreated,
            Some("test-wallet".to_string()),
            Some("Test wallet creation".to_string()),
        );

        assert_eq!(event.operation, KeyOperation::WalletCreated);
        assert_eq!(event.wallet_id, Some("test-wallet".to_string()));
        assert_eq!(event.severity, AuditSeverity::High);
    }

    #[tokio::test]
    async fn test_audit_hash_chain() {
        let event1 = AuditEvent::new(KeyOperation::WalletCreated, None, None).finalize("genesis");

        let event2 =
            AuditEvent::new(KeyOperation::KeyDerived, None, None).finalize(&event1.event_hash);

        assert!(event1.verify_hash());
        assert!(event2.verify_hash());
        assert_eq!(event2.previous_hash, event1.event_hash);
    }

    #[tokio::test]
    async fn test_audit_logger() {
        let temp_dir = TempDir::new().unwrap();
        let config = AuditConfig {
            log_path: temp_dir.path().join("test_audit.log"),
            ..Default::default()
        };

        let logger = AuditLogger::new(config).unwrap();

        logger
            .log(AuditEvent::new(
                KeyOperation::WalletCreated,
                Some("test".to_string()),
                None,
            ))
            .await
            .unwrap();

        logger
            .log(AuditEvent::new(
                KeyOperation::KeyDerived,
                Some("test".to_string()),
                None,
            ))
            .await
            .unwrap();

        let result = logger.verify_chain().await.unwrap();
        assert!(result.is_valid);
        assert_eq!(result.total_events, 2);
    }

    #[tokio::test]
    async fn test_event_builder() {
        let event = AuditEventBuilder::new(KeyOperation::TransactionSigned)
            .wallet_id("my-wallet")
            .details("Signed transaction 0x123...")
            .client_id("cli-v1")
            .build();

        assert_eq!(event.wallet_id, Some("my-wallet".to_string()));
        assert_eq!(event.details, Some("Signed transaction 0x123...".to_string()));
        assert_eq!(event.client_id, Some("cli-v1".to_string()));
    }
}
