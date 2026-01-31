//! Hardware Wallet Support
//!
//! Provides integration with hardware wallets like Ledger and Trezor.
//! Hardware wallets keep private keys secure in tamper-resistant hardware
//! and require physical confirmation for signing operations.

use std::fmt;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::audit::{AuditEvent, AuditLogger, KeyOperation};
use super::derivation::{DerivationPath, EthereumAddress, EthereumSignature};

pub mod ledger;

/// Types of supported hardware wallets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HardwareWalletType {
    /// Ledger Nano S/X/S Plus
    Ledger,
    /// Trezor Model T/One
    Trezor,
    /// GridPlus Lattice1
    GridPlus,
    /// KeepKey
    KeepKey,
}

impl HardwareWalletType {
    pub fn name(&self) -> &'static str {
        match self {
            HardwareWalletType::Ledger => "Ledger",
            HardwareWalletType::Trezor => "Trezor",
            HardwareWalletType::GridPlus => "GridPlus",
            HardwareWalletType::KeepKey => "KeepKey",
        }
    }
}

/// Hardware wallet device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareWalletInfo {
    /// Wallet type
    pub wallet_type: HardwareWalletType,
    /// Device model name
    pub model: String,
    /// Firmware version
    pub firmware_version: String,
    /// Device label (user-assigned name)
    pub label: Option<String>,
    /// Whether the device is currently locked
    pub is_locked: bool,
    /// Device serial number or unique identifier
    pub device_id: Option<String>,
}

/// Status of a hardware wallet operation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HardwareWalletStatus {
    /// Device is ready for operation
    Ready,
    /// Waiting for user to unlock device
    WaitingForUnlock,
    /// Waiting for user to confirm on device
    WaitingForConfirmation,
    /// Operation in progress
    InProgress,
    /// Operation completed successfully
    Completed,
    /// User rejected on device
    Rejected,
    /// Device disconnected
    Disconnected,
    /// Error occurred
    Error(String),
}

/// Error types for hardware wallet operations
#[derive(Debug, thiserror::Error)]
pub enum HardwareWalletError {
    #[error("No hardware wallet found")]
    DeviceNotFound,

    #[error("Multiple devices found, please specify which one")]
    MultipleDevices,

    #[error("Device is locked, please unlock it")]
    DeviceLocked,

    #[error("User rejected the operation")]
    UserRejected,

    #[error("Device disconnected")]
    Disconnected,

    #[error("Communication error: {0}")]
    CommunicationError(String),

    #[error("Invalid response from device: {0}")]
    InvalidResponse(String),

    #[error("App not open on device: {0}")]
    AppNotOpen(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Timeout waiting for device")]
    Timeout,

    #[error("Invalid path: {0}")]
    InvalidPath(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Trait for hardware wallet implementations
#[async_trait]
pub trait HardwareWallet: Send + Sync {
    /// Get wallet type
    fn wallet_type(&self) -> HardwareWalletType;

    /// Check if device is connected
    async fn is_connected(&self) -> bool;

    /// Get device information
    async fn get_info(&self) -> Result<HardwareWalletInfo, HardwareWalletError>;

    /// Get address at derivation path
    async fn get_address(
        &self,
        path: &DerivationPath,
        display: bool,
    ) -> Result<EthereumAddress, HardwareWalletError>;

    /// Sign a message hash
    async fn sign_message(
        &self,
        path: &DerivationPath,
        message: &[u8],
    ) -> Result<EthereumSignature, HardwareWalletError>;

    /// Sign a transaction hash
    async fn sign_transaction(
        &self,
        path: &DerivationPath,
        tx_hash: &[u8; 32],
        chain_id: u64,
    ) -> Result<EthereumSignature, HardwareWalletError>;

    /// Sign typed data (EIP-712)
    async fn sign_typed_data(
        &self,
        path: &DerivationPath,
        domain_separator: &[u8; 32],
        struct_hash: &[u8; 32],
    ) -> Result<EthereumSignature, HardwareWalletError>;
}

/// Callback for hardware wallet status updates
pub type StatusCallback = Box<dyn Fn(HardwareWalletStatus) + Send + Sync>;

/// Hardware wallet manager for unified access to different devices
pub struct HardwareWalletManager {
    /// Currently connected device
    device: Option<Box<dyn HardwareWallet>>,
    /// Audit logger
    audit_logger: AuditLogger,
    /// Status callback
    status_callback: Option<StatusCallback>,
    /// Operation timeout
    timeout: Duration,
}

impl HardwareWalletManager {
    /// Create a new hardware wallet manager
    pub fn new(audit_logger: AuditLogger) -> Self {
        Self {
            device: None,
            audit_logger,
            status_callback: None,
            timeout: Duration::from_secs(60),
        }
    }

    /// Set status callback for UI updates
    pub fn set_status_callback(&mut self, callback: StatusCallback) {
        self.status_callback = Some(callback);
    }

    /// Set operation timeout
    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }

    /// Report status to callback
    fn report_status(&self, status: HardwareWalletStatus) {
        if let Some(ref callback) = self.status_callback {
            callback(status);
        }
    }

    /// Detect and connect to a hardware wallet
    pub async fn detect(&mut self) -> Result<HardwareWalletInfo, HardwareWalletError> {
        self.report_status(HardwareWalletStatus::InProgress);

        // Try Ledger first
        #[cfg(feature = "hardware-wallet")]
        {
            if let Ok(ledger) = ledger::LedgerWallet::connect().await {
                if let Ok(info) = ledger.get_info().await {
                    self.device = Some(Box::new(ledger));

                    // Log connection
                    let _ = self
                        .audit_logger
                        .log(AuditEvent::new(
                            KeyOperation::HardwareWalletConnected,
                            None,
                            Some(format!("Connected to {} {}", info.wallet_type.name(), info.model)),
                        ))
                        .await;

                    self.report_status(HardwareWalletStatus::Ready);
                    return Ok(info);
                }
            }
        }

        // No hardware wallet feature - return simulated device for testing
        #[cfg(not(feature = "hardware-wallet"))]
        {
            let simulated = SimulatedHardwareWallet::new();
            let info = simulated.get_info().await?;
            self.device = Some(Box::new(simulated));

            let _ = self
                .audit_logger
                .log(AuditEvent::new(
                    KeyOperation::HardwareWalletConnected,
                    None,
                    Some("Connected to simulated hardware wallet".to_string()),
                ))
                .await;

            self.report_status(HardwareWalletStatus::Ready);
            return Ok(info);
        }

        #[cfg(feature = "hardware-wallet")]
        {
            self.report_status(HardwareWalletStatus::Error("No device found".to_string()));
            Err(HardwareWalletError::DeviceNotFound)
        }
    }

    /// Disconnect from the current device
    pub async fn disconnect(&mut self) {
        if self.device.is_some() {
            let _ = self
                .audit_logger
                .log(AuditEvent::new(
                    KeyOperation::HardwareWalletDisconnected,
                    None,
                    Some("Hardware wallet disconnected".to_string()),
                ))
                .await;
        }
        self.device = None;
        self.report_status(HardwareWalletStatus::Disconnected);
    }

    /// Check if a device is connected
    pub fn is_connected(&self) -> bool {
        self.device.is_some()
    }

    /// Get the connected device
    fn get_device(&self) -> Result<&dyn HardwareWallet, HardwareWalletError> {
        self.device
            .as_ref()
            .map(|d| d.as_ref())
            .ok_or(HardwareWalletError::DeviceNotFound)
    }

    /// Get address at derivation path
    pub async fn get_address(
        &self,
        path: &DerivationPath,
        display: bool,
    ) -> Result<EthereumAddress, HardwareWalletError> {
        let device = self.get_device()?;

        if display {
            self.report_status(HardwareWalletStatus::WaitingForConfirmation);
        }

        device.get_address(path, display).await
    }

    /// Sign a message with user confirmation prompt
    pub async fn sign_message(
        &self,
        path: &DerivationPath,
        message: &[u8],
        wallet_id: Option<&str>,
    ) -> Result<EthereumSignature, HardwareWalletError> {
        let device = self.get_device()?;

        self.report_status(HardwareWalletStatus::WaitingForConfirmation);

        let result = device.sign_message(path, message).await;

        match &result {
            Ok(_) => {
                self.report_status(HardwareWalletStatus::Completed);

                // Log the signing event
                let _ = self
                    .audit_logger
                    .log(AuditEvent::new(
                        KeyOperation::HardwareWalletSigned,
                        wallet_id.map(|s| s.to_string()),
                        Some("Message signed with hardware wallet".to_string()),
                    ))
                    .await;
            }
            Err(HardwareWalletError::UserRejected) => {
                self.report_status(HardwareWalletStatus::Rejected);
            }
            Err(e) => {
                self.report_status(HardwareWalletStatus::Error(e.to_string()));
            }
        }

        result
    }

    /// Sign a transaction with user confirmation prompt
    pub async fn sign_transaction(
        &self,
        path: &DerivationPath,
        tx_hash: &[u8; 32],
        chain_id: u64,
        wallet_id: Option<&str>,
    ) -> Result<EthereumSignature, HardwareWalletError> {
        let device = self.get_device()?;

        self.report_status(HardwareWalletStatus::WaitingForConfirmation);

        let result = device.sign_transaction(path, tx_hash, chain_id).await;

        match &result {
            Ok(_) => {
                self.report_status(HardwareWalletStatus::Completed);

                // Log the signing event
                let _ = self
                    .audit_logger
                    .log(AuditEvent::new(
                        KeyOperation::HardwareWalletSigned,
                        wallet_id.map(|s| s.to_string()),
                        Some(format!(
                            "Transaction signed with hardware wallet (chain_id: {})",
                            chain_id
                        )),
                    ))
                    .await;
            }
            Err(HardwareWalletError::UserRejected) => {
                self.report_status(HardwareWalletStatus::Rejected);
            }
            Err(e) => {
                self.report_status(HardwareWalletStatus::Error(e.to_string()));
            }
        }

        result
    }

    /// Sign typed data (EIP-712)
    pub async fn sign_typed_data(
        &self,
        path: &DerivationPath,
        domain_separator: &[u8; 32],
        struct_hash: &[u8; 32],
        wallet_id: Option<&str>,
    ) -> Result<EthereumSignature, HardwareWalletError> {
        let device = self.get_device()?;

        self.report_status(HardwareWalletStatus::WaitingForConfirmation);

        let result = device
            .sign_typed_data(path, domain_separator, struct_hash)
            .await;

        match &result {
            Ok(_) => {
                self.report_status(HardwareWalletStatus::Completed);

                let _ = self
                    .audit_logger
                    .log(AuditEvent::new(
                        KeyOperation::HardwareWalletSigned,
                        wallet_id.map(|s| s.to_string()),
                        Some("Typed data signed with hardware wallet".to_string()),
                    ))
                    .await;
            }
            Err(HardwareWalletError::UserRejected) => {
                self.report_status(HardwareWalletStatus::Rejected);
            }
            Err(e) => {
                self.report_status(HardwareWalletStatus::Error(e.to_string()));
            }
        }

        result
    }

    /// List available addresses for an account
    pub async fn list_addresses(
        &self,
        account: u32,
        count: u32,
    ) -> Result<Vec<(DerivationPath, EthereumAddress)>, HardwareWalletError> {
        let device = self.get_device()?;

        let mut addresses = Vec::with_capacity(count as usize);
        for i in 0..count {
            let path = DerivationPath::ethereum(account, i);
            let address = device.get_address(&path, false).await?;
            addresses.push((path, address));
        }

        Ok(addresses)
    }
}

impl fmt::Debug for HardwareWalletManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HardwareWalletManager")
            .field("connected", &self.device.is_some())
            .field("timeout", &self.timeout)
            .finish()
    }
}

/// Simulated hardware wallet for testing
pub struct SimulatedHardwareWallet {
    /// Simulated private key (for testing only)
    private_key: [u8; 32],
}

impl SimulatedHardwareWallet {
    pub fn new() -> Self {
        // Use a deterministic key for testing
        let mut key = [0u8; 32];
        key[0] = 0x01;
        Self { private_key: key }
    }
}

impl Default for SimulatedHardwareWallet {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl HardwareWallet for SimulatedHardwareWallet {
    fn wallet_type(&self) -> HardwareWalletType {
        HardwareWalletType::Ledger
    }

    async fn is_connected(&self) -> bool {
        true
    }

    async fn get_info(&self) -> Result<HardwareWalletInfo, HardwareWalletError> {
        Ok(HardwareWalletInfo {
            wallet_type: HardwareWalletType::Ledger,
            model: "Simulated Ledger".to_string(),
            firmware_version: "1.0.0".to_string(),
            label: Some("Test Device".to_string()),
            is_locked: false,
            device_id: Some("simulated-001".to_string()),
        })
    }

    async fn get_address(
        &self,
        path: &DerivationPath,
        _display: bool,
    ) -> Result<EthereumAddress, HardwareWalletError> {
        // Derive a deterministic address from the path
        use sha3::{Digest, Keccak256};

        let mut hasher = Keccak256::new();
        hasher.update(&self.private_key);
        hasher.update(path.as_str().as_bytes());
        let hash = hasher.finalize();

        let mut address = [0u8; 20];
        address.copy_from_slice(&hash[12..]);
        Ok(EthereumAddress::from_bytes(address))
    }

    async fn sign_message(
        &self,
        _path: &DerivationPath,
        message: &[u8],
    ) -> Result<EthereumSignature, HardwareWalletError> {
        // Return a simulated signature
        use sha3::{Digest, Keccak256};

        let mut hasher = Keccak256::new();
        hasher.update(&self.private_key);
        hasher.update(message);
        let hash = hasher.finalize();

        let mut r = [0u8; 32];
        let mut s = [0u8; 32];
        r.copy_from_slice(&hash);
        s.copy_from_slice(&hash);

        Ok(EthereumSignature { r, s, v: 27 })
    }

    async fn sign_transaction(
        &self,
        _path: &DerivationPath,
        tx_hash: &[u8; 32],
        chain_id: u64,
    ) -> Result<EthereumSignature, HardwareWalletError> {
        use sha3::{Digest, Keccak256};

        let mut hasher = Keccak256::new();
        hasher.update(&self.private_key);
        hasher.update(tx_hash);
        let hash = hasher.finalize();

        let mut r = [0u8; 32];
        let mut s = [0u8; 32];
        r.copy_from_slice(&hash);
        s.copy_from_slice(&hash);

        // EIP-155 v value
        let v = (chain_id * 2 + 35) as u8;

        Ok(EthereumSignature { r, s, v })
    }

    async fn sign_typed_data(
        &self,
        _path: &DerivationPath,
        domain_separator: &[u8; 32],
        struct_hash: &[u8; 32],
    ) -> Result<EthereumSignature, HardwareWalletError> {
        use sha3::{Digest, Keccak256};

        let mut hasher = Keccak256::new();
        hasher.update(&self.private_key);
        hasher.update(domain_separator);
        hasher.update(struct_hash);
        let hash = hasher.finalize();

        let mut r = [0u8; 32];
        let mut s = [0u8; 32];
        r.copy_from_slice(&hash);
        s.copy_from_slice(&hash);

        Ok(EthereumSignature { r, s, v: 27 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_simulated_wallet() {
        let wallet = SimulatedHardwareWallet::new();

        assert!(wallet.is_connected().await);

        let info = wallet.get_info().await.unwrap();
        assert_eq!(info.wallet_type, HardwareWalletType::Ledger);
        assert!(!info.is_locked);

        let path = DerivationPath::ethereum(0, 0);
        let address = wallet.get_address(&path, false).await.unwrap();
        assert!(!address.to_hex().is_empty());

        let message = b"test message";
        let sig = wallet.sign_message(&path, message).await.unwrap();
        assert!(sig.v == 27 || sig.v == 28);
    }

    #[tokio::test]
    async fn test_hardware_wallet_manager() {
        let audit_logger = AuditLogger::noop();
        let mut manager = HardwareWalletManager::new(audit_logger);

        // Without hardware-wallet feature, should use simulated device
        let info = manager.detect().await.unwrap();
        assert!(manager.is_connected());

        let path = DerivationPath::ethereum(0, 0);
        let address = manager.get_address(&path, false).await.unwrap();
        assert!(!address.to_hex().is_empty());

        manager.disconnect().await;
        assert!(!manager.is_connected());
    }
}
