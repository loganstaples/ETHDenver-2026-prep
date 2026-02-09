//! Proof Serialization.
//!
//! Serialization and deserialization of proofs for storage and transmission.
//! Supports multiple formats and compression.

use std::io::{Read, Write};
use serde::{Deserialize, Serialize};

use super::aggregation::AggregatedProof;
use super::chunking::ChunkId;
use super::ivc::IVCState;
use super::parallel::ChunkProof;

/// Proof format for serialization.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ProofFormat {
    /// Raw binary format.
    Binary,
    /// JSON format (human-readable).
    Json,
    /// Bincode format (compact binary).
    Bincode,
    /// CBOR format (Ethereum-compatible).
    Cbor,
}

/// Compression mode for proofs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CompressionMode {
    /// No compression.
    None,
    /// Gzip compression.
    Gzip,
    /// LZ4 compression (fast).
    Lz4,
    /// Zstd compression (best ratio).
    Zstd,
}

/// Configuration for serialization.
#[derive(Debug, Clone)]
pub struct SerializationConfig {
    /// Proof format.
    pub format: ProofFormat,
    /// Compression mode.
    pub compression: CompressionMode,
    /// Compression level (0-9).
    pub compression_level: u32,
    /// Whether to include metadata.
    pub include_metadata: bool,
}

impl Default for SerializationConfig {
    fn default() -> Self {
        Self {
            format: ProofFormat::Binary,
            compression: CompressionMode::None,
            compression_level: 6,
            include_metadata: true,
        }
    }
}

/// A serialized proof with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedProof {
    /// Format used.
    pub format: ProofFormat,
    /// Compression used.
    pub compression: CompressionMode,
    /// Version number.
    pub version: u8,
    /// Proof type identifier.
    pub proof_type: ProofType,
    /// Serialized proof data.
    pub data: Vec<u8>,
    /// Uncompressed size.
    pub uncompressed_size: usize,
    /// Checksum.
    pub checksum: [u8; 32],
}

/// Type of proof being serialized.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ProofType {
    /// Single chunk proof.
    Chunk,
    /// Aggregated proof.
    Aggregated,
    /// IVC state/proof.
    IVC,
    /// Full training step proof.
    TrainingStep,
}

/// Result of serialization operations.
pub type SerializeResult<T> = Result<T, SerializeError>;

/// Errors during serialization.
#[derive(Debug)]
pub enum SerializeError {
    /// Format error.
    FormatError(String),
    /// Compression error.
    CompressionError(String),
    /// Checksum mismatch.
    ChecksumMismatch,
    /// IO error.
    IoError(std::io::Error),
    /// Version mismatch.
    VersionMismatch(u8, u8),
}

impl std::fmt::Display for SerializeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SerializeError::FormatError(msg) => write!(f, "Format error: {}", msg),
            SerializeError::CompressionError(msg) => write!(f, "Compression error: {}", msg),
            SerializeError::ChecksumMismatch => write!(f, "Checksum mismatch"),
            SerializeError::IoError(e) => write!(f, "IO error: {}", e),
            SerializeError::VersionMismatch(expected, got) => {
                write!(f, "Version mismatch: expected {}, got {}", expected, got)
            }
        }
    }
}

impl std::error::Error for SerializeError {}

impl From<std::io::Error> for SerializeError {
    fn from(e: std::io::Error) -> Self {
        SerializeError::IoError(e)
    }
}

/// Current serialization version.
const CURRENT_VERSION: u8 = 1;

/// Magic bytes for proof files.
const MAGIC_BYTES: &[u8] = b"HELIX_PROOF";

/// Serializer for proofs.
pub struct ProofSerializer {
    /// Configuration.
    config: SerializationConfig,
}

impl ProofSerializer {
    /// Creates a new serializer with default config.
    pub fn new() -> Self {
        Self::with_config(SerializationConfig::default())
    }

    /// Creates a serializer with custom config.
    pub fn with_config(config: SerializationConfig) -> Self {
        Self { config }
    }

    /// Serializes a chunk proof.
    pub fn serialize_chunk(&self, proof: &ChunkProof) -> SerializeResult<SerializedProof> {
        let json = serde_json::to_vec(proof)
            .map_err(|e| SerializeError::FormatError(e.to_string()))?;
        
        self.wrap_proof(ProofType::Chunk, json)
    }

    /// Serializes an aggregated proof.
    pub fn serialize_aggregated(&self, proof: &AggregatedProof) -> SerializeResult<SerializedProof> {
        let json = serde_json::to_vec(proof)
            .map_err(|e| SerializeError::FormatError(e.to_string()))?;
        
        self.wrap_proof(ProofType::Aggregated, json)
    }

    /// Serializes an IVC state.
    pub fn serialize_ivc(&self, state: &IVCState) -> SerializeResult<SerializedProof> {
        let json = serde_json::to_vec(state)
            .map_err(|e| SerializeError::FormatError(e.to_string()))?;
        
        self.wrap_proof(ProofType::IVC, json)
    }

    /// Deserializes a chunk proof.
    pub fn deserialize_chunk(&self, serialized: &SerializedProof) -> SerializeResult<ChunkProof> {
        if serialized.proof_type != ProofType::Chunk {
            return Err(SerializeError::FormatError(
                format!("Expected Chunk proof, got {:?}", serialized.proof_type)
            ));
        }

        let data = self.unwrap_proof(serialized)?;
        
        serde_json::from_slice(&data)
            .map_err(|e| SerializeError::FormatError(e.to_string()))
    }

    /// Deserializes an aggregated proof.
    pub fn deserialize_aggregated(&self, serialized: &SerializedProof) -> SerializeResult<AggregatedProof> {
        if serialized.proof_type != ProofType::Aggregated {
            return Err(SerializeError::FormatError(
                format!("Expected Aggregated proof, got {:?}", serialized.proof_type)
            ));
        }

        let data = self.unwrap_proof(serialized)?;
        
        serde_json::from_slice(&data)
            .map_err(|e| SerializeError::FormatError(e.to_string()))
    }

    /// Deserializes an IVC state.
    pub fn deserialize_ivc(&self, serialized: &SerializedProof) -> SerializeResult<IVCState> {
        if serialized.proof_type != ProofType::IVC {
            return Err(SerializeError::FormatError(
                format!("Expected IVC proof, got {:?}", serialized.proof_type)
            ));
        }

        let data = self.unwrap_proof(serialized)?;
        
        serde_json::from_slice(&data)
            .map_err(|e| SerializeError::FormatError(e.to_string()))
    }

    /// Writes a serialized proof to a writer.
    pub fn write_to<W: Write>(&self, serialized: &SerializedProof, mut writer: W) -> SerializeResult<()> {
        // Magic bytes
        writer.write_all(MAGIC_BYTES)?;
        
        // Version
        writer.write_all(&[serialized.version])?;
        
        // Proof type
        writer.write_all(&[serialized.proof_type as u8])?;
        
        // Format and compression
        writer.write_all(&[serialized.format as u8, serialized.compression as u8])?;
        
        // Sizes
        writer.write_all(&(serialized.data.len() as u64).to_le_bytes())?;
        writer.write_all(&(serialized.uncompressed_size as u64).to_le_bytes())?;
        
        // Checksum
        writer.write_all(&serialized.checksum)?;
        
        // Data
        writer.write_all(&serialized.data)?;
        
        Ok(())
    }

    /// Reads a serialized proof from a reader.
    pub fn read_from<R: Read>(&self, mut reader: R) -> SerializeResult<SerializedProof> {
        // Magic bytes
        let mut magic = [0u8; MAGIC_BYTES.len()];
        reader.read_exact(&mut magic)?;
        if magic != MAGIC_BYTES {
            return Err(SerializeError::FormatError("Invalid magic bytes".to_string()));
        }
        
        // Version
        let mut version = [0u8; 1];
        reader.read_exact(&mut version)?;
        if version[0] != CURRENT_VERSION {
            return Err(SerializeError::VersionMismatch(CURRENT_VERSION, version[0]));
        }
        
        // Proof type
        let mut proof_type = [0u8; 1];
        reader.read_exact(&mut proof_type)?;
        let proof_type = match proof_type[0] {
            0 => ProofType::Chunk,
            1 => ProofType::Aggregated,
            2 => ProofType::IVC,
            3 => ProofType::TrainingStep,
            _ => return Err(SerializeError::FormatError("Invalid proof type".to_string())),
        };
        
        // Format and compression
        let mut format_compression = [0u8; 2];
        reader.read_exact(&mut format_compression)?;
        let format = match format_compression[0] {
            0 => ProofFormat::Binary,
            1 => ProofFormat::Json,
            2 => ProofFormat::Bincode,
            3 => ProofFormat::Cbor,
            _ => return Err(SerializeError::FormatError("Invalid format".to_string())),
        };
        let compression = match format_compression[1] {
            0 => CompressionMode::None,
            1 => CompressionMode::Gzip,
            2 => CompressionMode::Lz4,
            3 => CompressionMode::Zstd,
            _ => return Err(SerializeError::FormatError("Invalid compression".to_string())),
        };
        
        // Sizes
        let mut sizes = [0u8; 16];
        reader.read_exact(&mut sizes)?;
        let data_len = u64::from_le_bytes(sizes[..8].try_into().expect("invariant: fixed-size slice [..8] is 8 bytes")) as usize;
        let uncompressed_size = u64::from_le_bytes(sizes[8..].try_into().expect("invariant: fixed-size slice [8..] is 8 bytes")) as usize;
        
        // Checksum
        let mut checksum = [0u8; 32];
        reader.read_exact(&mut checksum)?;
        
        // Data
        let mut data = vec![0u8; data_len];
        reader.read_exact(&mut data)?;
        
        // Verify checksum
        let computed_checksum = Self::compute_checksum(&data);
        if computed_checksum != checksum {
            return Err(SerializeError::ChecksumMismatch);
        }
        
        Ok(SerializedProof {
            format,
            compression,
            version: version[0],
            proof_type,
            data,
            uncompressed_size,
            checksum,
        })
    }

    fn wrap_proof(&self, proof_type: ProofType, data: Vec<u8>) -> SerializeResult<SerializedProof> {
        let uncompressed_size = data.len();

        let compressed_data = match self.config.compression {
            CompressionMode::None => data,
            CompressionMode::Gzip => {
                use flate2::write::GzEncoder;
                use flate2::Compression;
                let level = self.config.compression_level.min(9);
                let mut encoder = GzEncoder::new(Vec::new(), Compression::new(level));
                encoder.write_all(&data).map_err(|e| {
                    SerializeError::CompressionError(format!("Gzip compress: {e}"))
                })?;
                encoder.finish().map_err(|e| {
                    SerializeError::CompressionError(format!("Gzip finish: {e}"))
                })?
            }
            CompressionMode::Lz4 => {
                lz4_flex::compress_prepend_size(&data)
            }
            CompressionMode::Zstd => {
                let level = self.config.compression_level.min(9) as i32;
                zstd::encode_all(data.as_slice(), level).map_err(|e| {
                    SerializeError::CompressionError(format!("Zstd compress: {e}"))
                })?
            }
        };

        let checksum = Self::compute_checksum(&compressed_data);

        Ok(SerializedProof {
            format: self.config.format,
            compression: self.config.compression,
            version: CURRENT_VERSION,
            proof_type,
            data: compressed_data,
            uncompressed_size,
            checksum,
        })
    }

    fn unwrap_proof(&self, serialized: &SerializedProof) -> SerializeResult<Vec<u8>> {
        // Verify checksum
        let computed = Self::compute_checksum(&serialized.data);
        if computed != serialized.checksum {
            return Err(SerializeError::ChecksumMismatch);
        }

        let data = match serialized.compression {
            CompressionMode::None => serialized.data.clone(),
            CompressionMode::Gzip => {
                use flate2::read::GzDecoder;
                let mut decoder = GzDecoder::new(serialized.data.as_slice());
                let mut decompressed = Vec::with_capacity(serialized.uncompressed_size);
                decoder.read_to_end(&mut decompressed).map_err(|e| {
                    SerializeError::CompressionError(format!("Gzip decompress: {e}"))
                })?;
                decompressed
            }
            CompressionMode::Lz4 => {
                lz4_flex::decompress_size_prepended(&serialized.data).map_err(|e| {
                    SerializeError::CompressionError(format!("LZ4 decompress: {e}"))
                })?
            }
            CompressionMode::Zstd => {
                zstd::decode_all(serialized.data.as_slice()).map_err(|e| {
                    SerializeError::CompressionError(format!("Zstd decompress: {e}"))
                })?
            }
        };

        Ok(data)
    }

    fn compute_checksum(data: &[u8]) -> [u8; 32] {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.finalize().into()
    }
}

impl Default for ProofSerializer {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to serialize a proof to bytes.
pub fn serialize_to_bytes<T: Serialize>(proof: &T) -> SerializeResult<Vec<u8>> {
    serde_json::to_vec(proof)
        .map_err(|e| SerializeError::FormatError(e.to_string()))
}

/// Convenience function to deserialize a proof from bytes.
pub fn deserialize_from_bytes<T: for<'de> Deserialize<'de>>(data: &[u8]) -> SerializeResult<T> {
    serde_json::from_slice(data)
        .map_err(|e| SerializeError::FormatError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_chunk_proof() -> ChunkProof {
        ChunkProof {
            chunk_id: ChunkId(42),
            proof: vec![1, 2, 3, 4, 5],
            public_inputs: vec![[1; 32], [2; 32]],
            error_bound: 0.001,
            generation_time_ms: 100,
        }
    }

    #[test]
    fn test_serialize_chunk_proof() {
        let serializer = ProofSerializer::new();
        let proof = make_test_chunk_proof();
        
        let serialized = serializer.serialize_chunk(&proof).unwrap();
        
        assert!(!serialized.data.is_empty());
        assert_eq!(serialized.proof_type, ProofType::Chunk);
    }

    #[test]
    fn test_roundtrip_chunk_proof() {
        let serializer = ProofSerializer::new();
        let original = make_test_chunk_proof();
        
        let serialized = serializer.serialize_chunk(&original).unwrap();
        let deserialized = serializer.deserialize_chunk(&serialized).unwrap();
        
        assert_eq!(original.chunk_id.0, deserialized.chunk_id.0);
        assert_eq!(original.proof, deserialized.proof);
    }

    #[test]
    fn test_write_read_proof() {
        let serializer = ProofSerializer::new();
        let proof = make_test_chunk_proof();

        let serialized = serializer.serialize_chunk(&proof).unwrap();

        let mut buffer = Vec::new();
        serializer.write_to(&serialized, &mut buffer).unwrap();

        let read_back = serializer.read_from(&buffer[..]).unwrap();

        assert_eq!(serialized.data, read_back.data);
        assert_eq!(serialized.checksum, read_back.checksum);
    }

    fn make_large_chunk_proof() -> ChunkProof {
        ChunkProof {
            chunk_id: ChunkId(42),
            proof: vec![0xAB; 4096],
            public_inputs: vec![[1; 32]; 8],
            error_bound: 0.001,
            generation_time_ms: 100,
        }
    }

    #[test]
    fn test_gzip_compression_roundtrip() {
        let serializer = ProofSerializer::with_config(SerializationConfig {
            compression: CompressionMode::Gzip,
            compression_level: 6,
            ..Default::default()
        });
        let original = make_large_chunk_proof();

        let serialized = serializer.serialize_chunk(&original).unwrap();
        assert_eq!(serialized.compression, CompressionMode::Gzip);
        assert!(
            serialized.data.len() < serialized.uncompressed_size,
            "Gzip should compress repeated data: {} >= {}",
            serialized.data.len(),
            serialized.uncompressed_size
        );

        let deserialized = serializer.deserialize_chunk(&serialized).unwrap();
        assert_eq!(original.chunk_id.0, deserialized.chunk_id.0);
        assert_eq!(original.proof, deserialized.proof);
    }

    #[test]
    fn test_lz4_compression_roundtrip() {
        let serializer = ProofSerializer::with_config(SerializationConfig {
            compression: CompressionMode::Lz4,
            ..Default::default()
        });
        let original = make_large_chunk_proof();

        let serialized = serializer.serialize_chunk(&original).unwrap();
        assert_eq!(serialized.compression, CompressionMode::Lz4);

        let deserialized = serializer.deserialize_chunk(&serialized).unwrap();
        assert_eq!(original.chunk_id.0, deserialized.chunk_id.0);
        assert_eq!(original.proof, deserialized.proof);
    }

    #[test]
    fn test_zstd_compression_roundtrip() {
        let serializer = ProofSerializer::with_config(SerializationConfig {
            compression: CompressionMode::Zstd,
            compression_level: 3,
            ..Default::default()
        });
        let original = make_large_chunk_proof();

        let serialized = serializer.serialize_chunk(&original).unwrap();
        assert_eq!(serialized.compression, CompressionMode::Zstd);
        assert!(
            serialized.data.len() < serialized.uncompressed_size,
            "Zstd should compress repeated data: {} >= {}",
            serialized.data.len(),
            serialized.uncompressed_size
        );

        let deserialized = serializer.deserialize_chunk(&serialized).unwrap();
        assert_eq!(original.chunk_id.0, deserialized.chunk_id.0);
        assert_eq!(original.proof, deserialized.proof);
    }

    #[test]
    fn test_compression_modes_all_produce_valid_checksums() {
        let original = make_large_chunk_proof();

        for mode in [
            CompressionMode::None,
            CompressionMode::Gzip,
            CompressionMode::Lz4,
            CompressionMode::Zstd,
        ] {
            let serializer = ProofSerializer::with_config(SerializationConfig {
                compression: mode,
                ..Default::default()
            });

            let serialized = serializer.serialize_chunk(&original).unwrap();
            let computed = ProofSerializer::compute_checksum(&serialized.data);
            assert_eq!(
                serialized.checksum, computed,
                "Checksum mismatch for {:?}",
                mode
            );

            let deserialized = serializer.deserialize_chunk(&serialized).unwrap();
            assert_eq!(original.proof, deserialized.proof, "Data mismatch for {:?}", mode);
        }
    }
}
