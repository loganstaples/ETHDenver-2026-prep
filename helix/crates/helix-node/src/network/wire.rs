//! Wire Format for HELIX Network Messages.
//!
//! Defines the binary wire protocol for network messages including:
//! - Message framing (length prefix, header, payload)
//! - Versioning for protocol evolution
//! - Compression support
//! - Checksums for integrity

use std::io::{self, Read, Write};

use bytes::{Buf, BufMut, BytesMut};
use serde::{Deserialize, Serialize};

use super::messages::NetworkMessage;

/// Current protocol version.
pub const PROTOCOL_VERSION: u8 = 1;

/// Magic bytes for frame identification.
pub const MAGIC: [u8; 4] = [0x48, 0x45, 0x4C, 0x58]; // "HELX"

/// Maximum message size (16 MB).
pub const MAX_MESSAGE_SIZE: usize = 16 * 1024 * 1024;

/// Header size in bytes.
pub const HEADER_SIZE: usize = 16;

/// Wire protocol errors.
#[derive(Debug, thiserror::Error)]
pub enum WireError {
    #[error("Invalid magic bytes")]
    InvalidMagic,
    #[error("Unsupported protocol version: {0}")]
    UnsupportedVersion(u8),
    #[error("Message too large: {size} > {max}")]
    MessageTooLarge { size: usize, max: usize },
    #[error("Checksum mismatch: expected {expected:08x}, got {actual:08x}")]
    ChecksumMismatch { expected: u32, actual: u32 },
    #[error("Decompression failed: {0}")]
    DecompressionFailed(String),
    #[error("Serialization failed: {0}")]
    SerializationFailed(String),
    #[error("Deserialization failed: {0}")]
    DeserializationFailed(String),
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("Incomplete frame: need {need} bytes, have {have}")]
    IncompleteFrame { need: usize, have: usize },
}

/// Message flags.
#[derive(Debug, Clone, Copy, Default)]
pub struct MessageFlags(u8);

impl MessageFlags {
    /// No flags set.
    pub const NONE: u8 = 0;
    /// Message is compressed.
    pub const COMPRESSED: u8 = 1 << 0;
    /// Message requires acknowledgment.
    pub const REQUIRES_ACK: u8 = 1 << 1;
    /// Message is a response.
    pub const IS_RESPONSE: u8 = 1 << 2;
    /// Message is encrypted (above TLS).
    pub const ENCRYPTED: u8 = 1 << 3;
    /// Message has priority.
    pub const PRIORITY: u8 = 1 << 4;

    pub fn new(flags: u8) -> Self {
        Self(flags)
    }

    pub fn is_compressed(&self) -> bool {
        self.0 & Self::COMPRESSED != 0
    }

    pub fn requires_ack(&self) -> bool {
        self.0 & Self::REQUIRES_ACK != 0
    }

    pub fn is_response(&self) -> bool {
        self.0 & Self::IS_RESPONSE != 0
    }

    pub fn is_encrypted(&self) -> bool {
        self.0 & Self::ENCRYPTED != 0
    }

    pub fn is_priority(&self) -> bool {
        self.0 & Self::PRIORITY != 0
    }

    pub fn as_u8(&self) -> u8 {
        self.0
    }
}

/// Wire frame header.
///
/// Layout (16 bytes):
/// - magic: 4 bytes ("HELX")
/// - version: 1 byte
/// - flags: 1 byte
/// - message_type: 2 bytes
/// - payload_len: 4 bytes
/// - checksum: 4 bytes (CRC32 of payload)
#[derive(Debug, Clone, Copy)]
pub struct FrameHeader {
    /// Protocol version.
    pub version: u8,
    /// Message flags.
    pub flags: MessageFlags,
    /// Message type ID.
    pub message_type: u16,
    /// Payload length.
    pub payload_len: u32,
    /// Payload checksum (CRC32).
    pub checksum: u32,
}

impl FrameHeader {
    /// Creates a new frame header.
    pub fn new(flags: MessageFlags, message_type: u16, payload_len: u32, checksum: u32) -> Self {
        Self {
            version: PROTOCOL_VERSION,
            flags,
            message_type,
            payload_len,
            checksum,
        }
    }

    /// Encodes the header to bytes.
    pub fn encode(&self, buf: &mut BytesMut) {
        buf.put_slice(&MAGIC);
        buf.put_u8(self.version);
        buf.put_u8(self.flags.as_u8());
        buf.put_u16(self.message_type);
        buf.put_u32(self.payload_len);
        buf.put_u32(self.checksum);
    }

    /// Decodes a header from bytes.
    pub fn decode(buf: &[u8]) -> Result<Self, WireError> {
        if buf.len() < HEADER_SIZE {
            return Err(WireError::IncompleteFrame {
                need: HEADER_SIZE,
                have: buf.len(),
            });
        }

        // Check magic
        if &buf[0..4] != &MAGIC {
            return Err(WireError::InvalidMagic);
        }

        let version = buf[4];
        if version > PROTOCOL_VERSION {
            return Err(WireError::UnsupportedVersion(version));
        }

        let flags = MessageFlags::new(buf[5]);
        let message_type = u16::from_be_bytes([buf[6], buf[7]]);
        let payload_len = u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]);
        let checksum = u32::from_be_bytes([buf[12], buf[13], buf[14], buf[15]]);

        Ok(Self {
            version,
            flags,
            message_type,
            payload_len,
            checksum,
        })
    }
}

/// Message type IDs.
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageTypeId {
    /// Discovery messages.
    Discovery = 0x0100,
    /// Training coordination messages.
    Training = 0x0200,
    /// Gradient exchange messages.
    Gradient = 0x0300,
    /// State sync messages.
    Sync = 0x0400,
    /// Heartbeat messages.
    Heartbeat = 0x0500,
    /// MPC protocol messages (Beaver triples, secret shares, garbled circuits).
    Mpc = 0x0600,
}

impl From<u16> for MessageTypeId {
    fn from(v: u16) -> Self {
        match v {
            0x0100 => Self::Discovery,
            0x0200 => Self::Training,
            0x0300 => Self::Gradient,
            0x0400 => Self::Sync,
            0x0500 => Self::Heartbeat,
            0x0600 => Self::Mpc,
            _ => Self::Discovery, // Default
        }
    }
}

/// Computes CRC32 checksum of data.
pub fn crc32(data: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(data);
    hasher.finalize()
}

/// Simplified CRC32 implementation (no external dependency).
mod crc32fast {
    pub struct Hasher {
        crc: u32,
    }

    impl Hasher {
        const CRC_TABLE: [u32; 256] = Self::make_crc_table();

        const fn make_crc_table() -> [u32; 256] {
            let mut table = [0u32; 256];
            let mut i = 0;
            while i < 256 {
                let mut crc = i as u32;
                let mut j = 0;
                while j < 8 {
                    if crc & 1 != 0 {
                        crc = 0xedb88320 ^ (crc >> 1);
                    } else {
                        crc >>= 1;
                    }
                    j += 1;
                }
                table[i] = crc;
                i += 1;
            }
            table
        }

        pub fn new() -> Self {
            Self { crc: 0xffffffff }
        }

        pub fn update(&mut self, data: &[u8]) {
            for byte in data {
                let index = ((self.crc ^ (*byte as u32)) & 0xff) as usize;
                self.crc = Self::CRC_TABLE[index] ^ (self.crc >> 8);
            }
        }

        pub fn finalize(self) -> u32 {
            self.crc ^ 0xffffffff
        }
    }
}

/// Wire codec for encoding/decoding messages.
pub struct WireCodec {
    /// Enable compression for large messages.
    compression_threshold: usize,
    /// Enable checksums.
    enable_checksums: bool,
}

impl WireCodec {
    /// Creates a new wire codec with default settings.
    pub fn new() -> Self {
        Self {
            compression_threshold: 1024, // Compress messages > 1KB
            enable_checksums: true,
        }
    }

    /// Creates a new wire codec with custom settings.
    pub fn with_settings(compression_threshold: usize, enable_checksums: bool) -> Self {
        Self {
            compression_threshold,
            enable_checksums,
        }
    }

    /// Encodes a message to wire format.
    pub fn encode(&self, message: &NetworkMessage) -> Result<BytesMut, WireError> {
        // Serialize message to bincode
        let payload = bincode::serialize(message)
            .map_err(|e| WireError::SerializationFailed(e.to_string()))?;

        if payload.len() > MAX_MESSAGE_SIZE {
            return Err(WireError::MessageTooLarge {
                size: payload.len(),
                max: MAX_MESSAGE_SIZE,
            });
        }

        // Determine message type
        let message_type = self.get_message_type(&message.payload);

        // Determine flags
        let mut flags = MessageFlags::NONE;
        if payload.len() > self.compression_threshold {
            flags |= MessageFlags::COMPRESSED;
        }

        // Optionally compress (simplified - just mark the flag, real impl would compress)
        let final_payload = payload; // In production, would compress here

        // Compute checksum
        let checksum = if self.enable_checksums {
            crc32(&final_payload)
        } else {
            0
        };

        // Build header
        let header = FrameHeader::new(
            MessageFlags::new(flags),
            message_type,
            final_payload.len() as u32,
            checksum,
        );

        // Build frame
        let mut buf = BytesMut::with_capacity(HEADER_SIZE + final_payload.len());
        header.encode(&mut buf);
        buf.put_slice(&final_payload);

        Ok(buf)
    }

    /// Decodes a message from wire format.
    pub fn decode(&self, buf: &mut BytesMut) -> Result<Option<NetworkMessage>, WireError> {
        // Check if we have enough for header
        if buf.len() < HEADER_SIZE {
            return Ok(None);
        }

        // Parse header
        let header = FrameHeader::decode(&buf[..HEADER_SIZE])?;

        // Enforce frame size limit on decode to prevent OOM from malicious headers
        if header.payload_len as usize > MAX_MESSAGE_SIZE {
            return Err(WireError::MessageTooLarge {
                size: header.payload_len as usize,
                max: MAX_MESSAGE_SIZE,
            });
        }

        let frame_size = HEADER_SIZE + header.payload_len as usize;
        if buf.len() < frame_size {
            return Ok(None);
        }

        // Extract payload
        buf.advance(HEADER_SIZE);
        let payload = buf.split_to(header.payload_len as usize);

        // Verify checksum
        if self.enable_checksums && header.checksum != 0 {
            let actual_checksum = crc32(&payload);
            if actual_checksum != header.checksum {
                return Err(WireError::ChecksumMismatch {
                    expected: header.checksum,
                    actual: actual_checksum,
                });
            }
        }

        // Decompress if needed (simplified)
        let final_payload = payload.to_vec();

        // Deserialize
        let message: NetworkMessage = bincode::deserialize(&final_payload)
            .map_err(|e| WireError::DeserializationFailed(e.to_string()))?;

        Ok(Some(message))
    }

    fn get_message_type(&self, payload: &super::messages::MessagePayload) -> u16 {
        use super::messages::MessagePayload;
        match payload {
            MessagePayload::Discovery(_) => MessageTypeId::Discovery as u16,
            MessagePayload::Training(_) => MessageTypeId::Training as u16,
            MessagePayload::Gradient(_) => MessageTypeId::Gradient as u16,
            MessagePayload::Sync(_) => MessageTypeId::Sync as u16,
            MessagePayload::Heartbeat(_) => MessageTypeId::Heartbeat as u16,
            MessagePayload::Consensus(_) => MessageTypeId::Training as u16,
            MessagePayload::MpcData(_) => MessageTypeId::Mpc as u16,
            MessagePayload::Registration(_) => MessageTypeId::Discovery as u16,
            MessagePayload::RoundManagement(_) => MessageTypeId::Training as u16,
        }
    }
}

impl Default for WireCodec {
    fn default() -> Self {
        Self::new()
    }
}

/// Frame reader for streaming decoding.
pub struct FrameReader {
    /// Buffer for incoming data.
    buffer: BytesMut,
    /// Codec for decoding.
    codec: WireCodec,
}

impl FrameReader {
    /// Creates a new frame reader.
    pub fn new() -> Self {
        Self {
            buffer: BytesMut::with_capacity(8192),
            codec: WireCodec::new(),
        }
    }

    /// Adds data to the buffer.
    pub fn extend(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
    }

    /// Tries to decode the next message.
    pub fn next_message(&mut self) -> Result<Option<NetworkMessage>, WireError> {
        self.codec.decode(&mut self.buffer)
    }

    /// Returns the amount of buffered data.
    pub fn buffered(&self) -> usize {
        self.buffer.len()
    }
}

impl Default for FrameReader {
    fn default() -> Self {
        Self::new()
    }
}

/// Frame writer for streaming encoding.
pub struct FrameWriter {
    /// Codec for encoding.
    codec: WireCodec,
}

impl FrameWriter {
    /// Creates a new frame writer.
    pub fn new() -> Self {
        Self {
            codec: WireCodec::new(),
        }
    }

    /// Encodes a message.
    pub fn encode(&self, message: &NetworkMessage) -> Result<BytesMut, WireError> {
        self.codec.encode(message)
    }
}

impl Default for FrameWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Binary format for gradient data (more efficient than JSON).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryGradient {
    /// Gradient dimensions.
    pub dimensions: Vec<usize>,
    /// Flattened gradient values (quantized).
    pub values: Vec<i32>,
    /// Quantization scale.
    pub scale: f64,
    /// Zero point.
    pub zero_point: i32,
}

impl BinaryGradient {
    /// Creates a new binary gradient from float values using asymmetric quantization.
    pub fn from_floats(values: &[f64], dimensions: Vec<usize>, bits: u8) -> Self {
        let max_val = (1i32 << bits) - 1; // For 8 bits: 255
        let min_val = 0i32;

        // Find range
        let (vmin, vmax) = values.iter().fold((f64::MAX, f64::MIN), |(min, max), &v| {
            (min.min(v), max.max(v))
        });

        let scale = if vmax - vmin > 1e-10 {
            (vmax - vmin) / (max_val - min_val) as f64
        } else {
            1.0
        };

        // Zero point maps vmin to min_val (0)
        let zero_point = if scale > 1e-10 {
            (min_val as f64 - vmin / scale).round() as i32
        } else {
            0
        };

        let quantized: Vec<i32> = values
            .iter()
            .map(|&v| ((v / scale).round() as i32 + zero_point).clamp(min_val, max_val))
            .collect();

        Self {
            dimensions,
            values: quantized,
            scale,
            zero_point,
        }
    }

    /// Converts back to float values.
    pub fn to_floats(&self) -> Vec<f64> {
        self.values
            .iter()
            .map(|&v| (v - self.zero_point) as f64 * self.scale)
            .collect()
    }

    /// Serializes to binary format.
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Deserializes from binary format.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        bincode::deserialize(data).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::messages::{HeartbeatMessage, MessagePayload, PeerId};

    #[test]
    fn test_header_encode_decode() {
        let header = FrameHeader::new(
            MessageFlags::new(MessageFlags::COMPRESSED),
            0x0100,
            1234,
            0xdeadbeef,
        );

        let mut buf = BytesMut::new();
        header.encode(&mut buf);

        let decoded = FrameHeader::decode(&buf).unwrap();
        assert_eq!(decoded.version, PROTOCOL_VERSION);
        assert!(decoded.flags.is_compressed());
        assert_eq!(decoded.message_type, 0x0100);
        assert_eq!(decoded.payload_len, 1234);
        assert_eq!(decoded.checksum, 0xdeadbeef);
    }

    #[test]
    fn test_wire_codec_roundtrip() {
        let codec = WireCodec::new();

        let message = NetworkMessage::new(
            PeerId::random(),
            MessagePayload::Heartbeat(HeartbeatMessage {
                seq: 42,
                is_pong: false,
                load: 50,
            }),
        );

        let encoded = codec.encode(&message).unwrap();
        let mut buf = encoded;
        let decoded = codec.decode(&mut buf).unwrap().unwrap();

        assert_eq!(decoded.id, message.id);
    }

    #[test]
    fn test_crc32() {
        let data = b"hello world";
        let checksum = crc32(data);
        assert_eq!(checksum, 0x0D4A1185);
    }

    #[test]
    fn test_binary_gradient_roundtrip() {
        let values = vec![0.1, 0.2, 0.3, -0.1, -0.2];
        let dims = vec![5];

        let binary = BinaryGradient::from_floats(&values, dims, 8);
        let bytes = binary.to_bytes();
        let restored = BinaryGradient::from_bytes(&bytes).unwrap();
        let floats = restored.to_floats();

        // Check approximate equality (due to quantization)
        for (orig, restored) in values.iter().zip(floats.iter()) {
            assert!((orig - restored).abs() < 0.01);
        }
    }

    #[test]
    fn test_bincode_smaller_than_json() {
        use crate::network::messages::GradientMessage;

        let message = NetworkMessage::new(
            PeerId::from_string("test-peer"),
            MessagePayload::Gradient(GradientMessage::ShareGradient {
                round_id: 42,
                gradient_commitment: [0xAB; 32],
                commitment_nonce: [0u8; 16],
                error_bound: 0.001,
                proof: vec![1u8; 256],
            }),
        );

        let bincode_bytes = bincode::serialize(&message).unwrap();
        let json_bytes = serde_json::to_vec(&message).unwrap();

        assert!(
            bincode_bytes.len() < json_bytes.len(),
            "bincode ({} bytes) should be smaller than JSON ({} bytes)",
            bincode_bytes.len(),
            json_bytes.len(),
        );
    }

    #[test]
    fn test_decode_rejects_oversized_frame() {
        let codec = WireCodec::new();

        // Build a valid header that claims a payload larger than MAX_MESSAGE_SIZE
        let header = FrameHeader::new(
            MessageFlags::new(MessageFlags::NONE),
            0x0100,
            (MAX_MESSAGE_SIZE + 1) as u32,
            0,
        );

        let mut buf = BytesMut::new();
        header.encode(&mut buf);
        // Add some dummy bytes (not the full payload, just enough to have a header)
        buf.extend_from_slice(&[0u8; 64]);

        let result = codec.decode(&mut buf);
        assert!(result.is_err());
        match result.unwrap_err() {
            WireError::MessageTooLarge { size, max } => {
                assert_eq!(size, MAX_MESSAGE_SIZE + 1);
                assert_eq!(max, MAX_MESSAGE_SIZE);
            }
            e => panic!("Expected MessageTooLarge, got {:?}", e),
        }
    }

    #[test]
    fn test_frame_reader() {
        let codec = WireCodec::new();
        let mut reader = FrameReader::new();

        let message = NetworkMessage::new(
            PeerId::random(),
            MessagePayload::Heartbeat(HeartbeatMessage {
                seq: 1,
                is_pong: false,
                load: 10,
            }),
        );

        let encoded = codec.encode(&message).unwrap();
        reader.extend(&encoded);

        let decoded = reader.next_message().unwrap();
        assert!(decoded.is_some());
    }
}
