//! Ledger Hardware Wallet Support
//!
//! Implements the Ethereum application protocol for Ledger devices.
//! Supports Ledger Nano S, Nano X, and Nano S Plus.
//!
//! Protocol reference:
//! https://github.com/LedgerHQ/app-ethereum/blob/master/doc/ethapp.adoc

#![cfg(feature = "hardware-wallet")]

use std::time::Duration;

use async_trait::async_trait;
use hidapi::{HidApi, HidDevice};

use super::{
    HardwareWallet, HardwareWalletError, HardwareWalletInfo, HardwareWalletType,
};
use crate::wallet::derivation::{DerivationPath, EthereumAddress, EthereumSignature};

/// Ledger vendor ID
const LEDGER_VENDOR_ID: u16 = 0x2c97;

/// Ledger product IDs
const LEDGER_PRODUCT_IDS: &[u16] = &[
    0x0001, // Nano S
    0x0004, // Nano X
    0x0005, // Nano S Plus
    0x4011, // Nano S (newer)
    0x4015, // Nano X (newer)
    0x5011, // Nano S Plus (newer)
];

/// APDU instruction codes for Ethereum app
mod ins {
    pub const GET_PUBLIC_KEY: u8 = 0x02;
    pub const SIGN_TRANSACTION: u8 = 0x04;
    pub const GET_APP_CONFIGURATION: u8 = 0x06;
    pub const SIGN_PERSONAL_MESSAGE: u8 = 0x08;
    pub const SIGN_EIP712_MESSAGE: u8 = 0x0C;
}

/// APDU status codes
mod status {
    pub const OK: u16 = 0x9000;
    pub const USER_REJECTED: u16 = 0x6985;
    pub const WRONG_LENGTH: u16 = 0x6700;
    pub const WRONG_DATA: u16 = 0x6A80;
    pub const INVALID_P1_P2: u16 = 0x6B00;
    pub const INS_NOT_SUPPORTED: u16 = 0x6D00;
    pub const APP_NOT_OPEN: u16 = 0x6E00;
    pub const UNKNOWN: u16 = 0x6F00;
    pub const LOCKED_DEVICE: u16 = 0x6982;
}

/// HID frame size
const HID_FRAME_SIZE: usize = 64;

/// Channel ID for HID
const CHANNEL_ID: u16 = 0x0101;

/// Ledger hardware wallet implementation
pub struct LedgerWallet {
    /// HID device handle
    device: HidDevice,
    /// Device model name
    model: String,
}

impl LedgerWallet {
    /// Connect to a Ledger device
    pub async fn connect() -> Result<Self, HardwareWalletError> {
        let api = HidApi::new().map_err(|e| {
            HardwareWalletError::CommunicationError(format!("Failed to initialize HID: {}", e))
        })?;

        // Find Ledger device
        for device_info in api.device_list() {
            if device_info.vendor_id() == LEDGER_VENDOR_ID
                && LEDGER_PRODUCT_IDS.contains(&device_info.product_id())
            {
                let device = device_info.open_device(&api).map_err(|e| {
                    HardwareWalletError::CommunicationError(format!(
                        "Failed to open device: {}",
                        e
                    ))
                })?;

                let model = Self::detect_model(device_info.product_id());

                return Ok(Self { device, model });
            }
        }

        Err(HardwareWalletError::DeviceNotFound)
    }

    /// Detect device model from product ID
    fn detect_model(product_id: u16) -> String {
        match product_id {
            0x0001 | 0x4011 => "Nano S".to_string(),
            0x0004 | 0x4015 => "Nano X".to_string(),
            0x0005 | 0x5011 => "Nano S Plus".to_string(),
            _ => "Unknown Ledger".to_string(),
        }
    }

    /// Send APDU command to device
    fn send_apdu(&self, cla: u8, ins: u8, p1: u8, p2: u8, data: &[u8]) -> Result<Vec<u8>, HardwareWalletError> {
        // Build APDU
        let mut apdu = Vec::with_capacity(5 + data.len());
        apdu.push(cla);
        apdu.push(ins);
        apdu.push(p1);
        apdu.push(p2);
        apdu.push(data.len() as u8);
        apdu.extend_from_slice(data);

        // Wrap in HID frames
        let frames = self.wrap_apdu(&apdu);

        // Send frames
        for frame in frames {
            self.device.write(&frame).map_err(|e| {
                HardwareWalletError::CommunicationError(format!("Write failed: {}", e))
            })?;
        }

        // Read response
        self.read_response()
    }

    /// Wrap APDU in HID frames
    fn wrap_apdu(&self, apdu: &[u8]) -> Vec<[u8; HID_FRAME_SIZE]> {
        let mut frames = Vec::new();
        let mut offset = 0;
        let mut seq_idx = 0u16;

        while offset < apdu.len() {
            let mut frame = [0u8; HID_FRAME_SIZE];

            // Header
            frame[0] = (CHANNEL_ID >> 8) as u8;
            frame[1] = (CHANNEL_ID & 0xFF) as u8;
            frame[2] = 0x05; // Command tag

            if seq_idx == 0 {
                // First frame includes length
                frame[3] = (seq_idx >> 8) as u8;
                frame[4] = (seq_idx & 0xFF) as u8;
                frame[5] = (apdu.len() >> 8) as u8;
                frame[6] = (apdu.len() & 0xFF) as u8;

                let data_len = std::cmp::min(apdu.len() - offset, HID_FRAME_SIZE - 7);
                frame[7..7 + data_len].copy_from_slice(&apdu[offset..offset + data_len]);
                offset += data_len;
            } else {
                frame[3] = (seq_idx >> 8) as u8;
                frame[4] = (seq_idx & 0xFF) as u8;

                let data_len = std::cmp::min(apdu.len() - offset, HID_FRAME_SIZE - 5);
                frame[5..5 + data_len].copy_from_slice(&apdu[offset..offset + data_len]);
                offset += data_len;
            }

            frames.push(frame);
            seq_idx += 1;
        }

        frames
    }

    /// Read response from device
    fn read_response(&self) -> Result<Vec<u8>, HardwareWalletError> {
        let mut response = Vec::new();
        let mut expected_len: Option<usize> = None;
        let mut seq_idx = 0u16;

        loop {
            let mut frame = [0u8; HID_FRAME_SIZE];
            let read = self.device.read_timeout(&mut frame, 30000).map_err(|e| {
                HardwareWalletError::CommunicationError(format!("Read failed: {}", e))
            })?;

            if read == 0 {
                return Err(HardwareWalletError::Timeout);
            }

            // Verify channel ID
            let channel = ((frame[0] as u16) << 8) | (frame[1] as u16);
            if channel != CHANNEL_ID {
                continue;
            }

            // Verify command tag
            if frame[2] != 0x05 {
                continue;
            }

            let frame_seq = ((frame[3] as u16) << 8) | (frame[4] as u16);
            if frame_seq != seq_idx {
                return Err(HardwareWalletError::InvalidResponse(
                    "Sequence mismatch".to_string(),
                ));
            }

            if seq_idx == 0 {
                // First frame has length
                let len = ((frame[5] as usize) << 8) | (frame[6] as usize);
                expected_len = Some(len);

                let data_len = std::cmp::min(len, HID_FRAME_SIZE - 7);
                response.extend_from_slice(&frame[7..7 + data_len]);
            } else {
                let remaining = expected_len
                    .ok_or_else(|| HardwareWalletError::InvalidResponse("continuation frame before length frame".into()))?
                    - response.len();
                let data_len = std::cmp::min(remaining, HID_FRAME_SIZE - 5);
                response.extend_from_slice(&frame[5..5 + data_len]);
            }

            if let Some(len) = expected_len {
                if response.len() >= len {
                    break;
                }
            }

            seq_idx += 1;
        }

        // Check status code
        if response.len() < 2 {
            return Err(HardwareWalletError::InvalidResponse(
                "Response too short".to_string(),
            ));
        }

        let status =
            ((response[response.len() - 2] as u16) << 8) | (response[response.len() - 1] as u16);

        match status {
            status::OK => {
                response.truncate(response.len() - 2);
                Ok(response)
            }
            status::USER_REJECTED => Err(HardwareWalletError::UserRejected),
            status::LOCKED_DEVICE => Err(HardwareWalletError::DeviceLocked),
            status::APP_NOT_OPEN => {
                Err(HardwareWalletError::AppNotOpen("Ethereum app not open".to_string()))
            }
            status::INS_NOT_SUPPORTED => Err(HardwareWalletError::UnsupportedOperation(
                "Instruction not supported".to_string(),
            )),
            _ => Err(HardwareWalletError::InvalidResponse(format!(
                "Status code: 0x{:04X}",
                status
            ))),
        }
    }

    /// Serialize derivation path for APDU
    fn serialize_path(path: &DerivationPath) -> Vec<u8> {
        let path_str = path.as_str();
        let components: Vec<u32> = path_str
            .strip_prefix("m/")
            .unwrap_or(&path_str)
            .split('/')
            .filter_map(|s| {
                let hardened = s.ends_with('\'');
                let num_str = s.trim_end_matches('\'');
                num_str.parse::<u32>().ok().map(|n| {
                    if hardened {
                        n | 0x80000000
                    } else {
                        n
                    }
                })
            })
            .collect();

        let mut data = Vec::with_capacity(1 + components.len() * 4);
        data.push(components.len() as u8);
        for component in components {
            data.extend_from_slice(&component.to_be_bytes());
        }
        data
    }
}

#[async_trait]
impl HardwareWallet for LedgerWallet {
    fn wallet_type(&self) -> HardwareWalletType {
        HardwareWalletType::Ledger
    }

    async fn is_connected(&self) -> bool {
        // Try to get app configuration to verify connection
        self.send_apdu(0xE0, ins::GET_APP_CONFIGURATION, 0x00, 0x00, &[])
            .is_ok()
    }

    async fn get_info(&self) -> Result<HardwareWalletInfo, HardwareWalletError> {
        let response = self.send_apdu(0xE0, ins::GET_APP_CONFIGURATION, 0x00, 0x00, &[])?;

        let version = if response.len() >= 4 {
            format!("{}.{}.{}", response[1], response[2], response[3])
        } else {
            "Unknown".to_string()
        };

        Ok(HardwareWalletInfo {
            wallet_type: HardwareWalletType::Ledger,
            model: self.model.clone(),
            firmware_version: version,
            label: None,
            is_locked: false,
            device_id: None,
        })
    }

    async fn get_address(
        &self,
        path: &DerivationPath,
        display: bool,
    ) -> Result<EthereumAddress, HardwareWalletError> {
        let path_data = Self::serialize_path(path);
        let p1 = if display { 0x01 } else { 0x00 };

        let response = self.send_apdu(0xE0, ins::GET_PUBLIC_KEY, p1, 0x00, &path_data)?;

        // Response format: public key length (1 byte) + public key + address length (1 byte) + address (hex string)
        if response.len() < 2 {
            return Err(HardwareWalletError::InvalidResponse(
                "Response too short".to_string(),
            ));
        }

        let pk_len = response[0] as usize;
        if response.len() < 1 + pk_len + 1 {
            return Err(HardwareWalletError::InvalidResponse(
                "Invalid public key".to_string(),
            ));
        }

        let addr_len = response[1 + pk_len] as usize;
        if response.len() < 1 + pk_len + 1 + addr_len {
            return Err(HardwareWalletError::InvalidResponse(
                "Invalid address".to_string(),
            ));
        }

        let addr_str = String::from_utf8(response[2 + pk_len..2 + pk_len + addr_len].to_vec())
            .map_err(|e| {
                HardwareWalletError::InvalidResponse(format!("Invalid address encoding: {}", e))
            })?;

        EthereumAddress::from_hex(&addr_str)
            .map_err(|e| HardwareWalletError::InvalidResponse(format!("Invalid address: {}", e)))
    }

    async fn sign_message(
        &self,
        path: &DerivationPath,
        message: &[u8],
    ) -> Result<EthereumSignature, HardwareWalletError> {
        let path_data = Self::serialize_path(path);

        // First chunk: path + message length + first message bytes
        let mut data = path_data.clone();
        data.extend_from_slice(&(message.len() as u32).to_be_bytes());

        let first_chunk_data_len = std::cmp::min(message.len(), 255 - data.len());
        data.extend_from_slice(&message[..first_chunk_data_len]);

        let mut response = self.send_apdu(0xE0, ins::SIGN_PERSONAL_MESSAGE, 0x00, 0x00, &data)?;

        // Send remaining chunks
        let mut offset = first_chunk_data_len;
        while offset < message.len() {
            let chunk_len = std::cmp::min(message.len() - offset, 255);
            let chunk = &message[offset..offset + chunk_len];
            response = self.send_apdu(0xE0, ins::SIGN_PERSONAL_MESSAGE, 0x80, 0x00, chunk)?;
            offset += chunk_len;
        }

        // Parse signature
        if response.len() < 65 {
            return Err(HardwareWalletError::InvalidResponse(
                "Signature too short".to_string(),
            ));
        }

        let v = response[0];
        let mut r = [0u8; 32];
        let mut s = [0u8; 32];
        r.copy_from_slice(&response[1..33]);
        s.copy_from_slice(&response[33..65]);

        Ok(EthereumSignature { r, s, v })
    }

    async fn sign_transaction(
        &self,
        path: &DerivationPath,
        tx_hash: &[u8; 32],
        chain_id: u64,
    ) -> Result<EthereumSignature, HardwareWalletError> {
        let path_data = Self::serialize_path(path);

        // Build RLP-encoded transaction for signing
        // This is a simplified version - full implementation would need RLP encoding
        let mut data = path_data.clone();
        data.extend_from_slice(tx_hash);

        let response = self.send_apdu(0xE0, ins::SIGN_TRANSACTION, 0x00, 0x00, &data)?;

        // Parse signature
        if response.len() < 65 {
            return Err(HardwareWalletError::InvalidResponse(
                "Signature too short".to_string(),
            ));
        }

        let v = response[0];
        let mut r = [0u8; 32];
        let mut s = [0u8; 32];
        r.copy_from_slice(&response[1..33]);
        s.copy_from_slice(&response[33..65]);

        // Apply EIP-155
        let v = if v == 0 || v == 1 {
            (chain_id * 2 + 35 + v as u64) as u8
        } else {
            v
        };

        Ok(EthereumSignature { r, s, v })
    }

    async fn sign_typed_data(
        &self,
        path: &DerivationPath,
        domain_separator: &[u8; 32],
        struct_hash: &[u8; 32],
    ) -> Result<EthereumSignature, HardwareWalletError> {
        let path_data = Self::serialize_path(path);

        let mut data = path_data.clone();
        data.extend_from_slice(domain_separator);
        data.extend_from_slice(struct_hash);

        let response = self.send_apdu(0xE0, ins::SIGN_EIP712_MESSAGE, 0x00, 0x00, &data)?;

        // Parse signature
        if response.len() < 65 {
            return Err(HardwareWalletError::InvalidResponse(
                "Signature too short".to_string(),
            ));
        }

        let v = response[0];
        let mut r = [0u8; 32];
        let mut s = [0u8; 32];
        r.copy_from_slice(&response[1..33]);
        s.copy_from_slice(&response[33..65]);

        Ok(EthereumSignature { r, s, v })
    }
}

/// Enumerate all connected Ledger devices
pub fn enumerate_devices() -> Result<Vec<LedgerDeviceInfo>, HardwareWalletError> {
    let api = HidApi::new().map_err(|e| {
        HardwareWalletError::CommunicationError(format!("Failed to initialize HID: {}", e))
    })?;

    let mut devices = Vec::new();

    for device_info in api.device_list() {
        if device_info.vendor_id() == LEDGER_VENDOR_ID
            && LEDGER_PRODUCT_IDS.contains(&device_info.product_id())
        {
            devices.push(LedgerDeviceInfo {
                model: LedgerWallet::detect_model(device_info.product_id()),
                path: device_info.path().to_string_lossy().to_string(),
                serial: device_info.serial_number().map(|s| s.to_string()),
            });
        }
    }

    Ok(devices)
}

/// Information about a connected Ledger device
#[derive(Debug, Clone)]
pub struct LedgerDeviceInfo {
    pub model: String,
    pub path: String,
    pub serial: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_serialization() {
        let path = DerivationPath::ethereum(0, 0);
        let serialized = LedgerWallet::serialize_path(&path);

        // Should have: length byte + 5 * 4 bytes (for m/44'/60'/0'/0'/0)
        assert_eq!(serialized[0], 5);
        assert_eq!(serialized.len(), 1 + 5 * 4);
    }

    #[test]
    fn test_model_detection() {
        assert_eq!(LedgerWallet::detect_model(0x0001), "Nano S");
        assert_eq!(LedgerWallet::detect_model(0x0004), "Nano X");
        assert_eq!(LedgerWallet::detect_model(0x0005), "Nano S Plus");
    }
}
