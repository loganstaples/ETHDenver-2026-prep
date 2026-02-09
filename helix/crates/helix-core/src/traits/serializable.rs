//! Binary serialization for circuit compatibility.

use std::io::{Read, Write};

/// Error type for serialization operations.
#[derive(Debug)]
pub enum SerializeError {
    /// I/O error during serialization.
    Io(std::io::Error),
    /// Invalid data encountered during deserialization.
    InvalidData(String),
    /// Buffer too small for serialization.
    BufferTooSmall { needed: usize, available: usize },
}

impl std::fmt::Display for SerializeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SerializeError::Io(e) => write!(f, "IO error: {}", e),
            SerializeError::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
            SerializeError::BufferTooSmall { needed, available } => {
                write!(f, "Buffer too small: needed {}, available {}", needed, available)
            }
        }
    }
}

impl std::error::Error for SerializeError {}

impl From<std::io::Error> for SerializeError {
    fn from(e: std::io::Error) -> Self {
        SerializeError::Io(e)
    }
}

/// Trait for types that can be serialized to a binary format suitable for circuits.
///
/// This is separate from serde because circuit serialization has specific requirements:
/// - Fixed-size representations
/// - Field-element aligned data
/// - Deterministic byte ordering
pub trait BinarySerializable: Sized {
    /// Returns the number of bytes needed to serialize this value.
    fn serialized_size(&self) -> usize;

    /// Serializes the value to a writer.
    fn serialize<W: Write>(&self, writer: &mut W) -> Result<(), SerializeError>;

    /// Deserializes a value from a reader.
    fn deserialize<R: Read>(reader: &mut R) -> Result<Self, SerializeError>;

    /// Serializes to a byte vector.
    fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.serialized_size());
        self.serialize(&mut buf).expect("Vec write should not fail");
        buf
    }

    /// Deserializes from a byte slice.
    fn from_bytes(bytes: &[u8]) -> Result<Self, SerializeError> {
        let mut cursor = std::io::Cursor::new(bytes);
        Self::deserialize(&mut cursor)
    }

    /// Serializes with zstd compression.
    ///
    /// Requires the `compression` feature. Wraps the binary output in a zstd stream
    /// for reduced storage size.
    #[cfg(feature = "compression")]
    fn serialize_compressed<W: Write>(&self, writer: &mut W) -> Result<(), SerializeError> {
        let mut encoder = zstd::Encoder::new(writer, 3)
            .map_err(|e| SerializeError::Io(e))?;
        self.serialize(&mut encoder)?;
        encoder.finish().map_err(|e| SerializeError::Io(e))?;
        Ok(())
    }

    /// Deserializes from a zstd-compressed stream.
    ///
    /// Requires the `compression` feature.
    #[cfg(feature = "compression")]
    fn deserialize_compressed<R: Read>(reader: &mut R) -> Result<Self, SerializeError> {
        let mut decoder = zstd::Decoder::new(reader)
            .map_err(|e| SerializeError::Io(e))?;
        Self::deserialize(&mut decoder)
    }

    /// Serializes to a compressed byte vector.
    #[cfg(feature = "compression")]
    fn to_compressed_bytes(&self) -> Result<Vec<u8>, SerializeError> {
        let mut buf = Vec::new();
        self.serialize_compressed(&mut buf)?;
        Ok(buf)
    }

    /// Deserializes from a compressed byte slice.
    #[cfg(feature = "compression")]
    fn from_compressed_bytes(bytes: &[u8]) -> Result<Self, SerializeError> {
        let mut cursor = std::io::Cursor::new(bytes);
        Self::deserialize_compressed(&mut cursor)
    }
}

/// Implement BinarySerializable for primitive types.
impl BinarySerializable for f64 {
    fn serialized_size(&self) -> usize {
        8
    }

    fn serialize<W: Write>(&self, writer: &mut W) -> Result<(), SerializeError> {
        writer.write_all(&self.to_le_bytes())?;
        Ok(())
    }

    fn deserialize<R: Read>(reader: &mut R) -> Result<Self, SerializeError> {
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        Ok(f64::from_le_bytes(buf))
    }
}

impl BinarySerializable for f32 {
    fn serialized_size(&self) -> usize {
        4
    }

    fn serialize<W: Write>(&self, writer: &mut W) -> Result<(), SerializeError> {
        writer.write_all(&self.to_le_bytes())?;
        Ok(())
    }

    fn deserialize<R: Read>(reader: &mut R) -> Result<Self, SerializeError> {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        Ok(f32::from_le_bytes(buf))
    }
}

impl BinarySerializable for u64 {
    fn serialized_size(&self) -> usize {
        8
    }

    fn serialize<W: Write>(&self, writer: &mut W) -> Result<(), SerializeError> {
        writer.write_all(&self.to_le_bytes())?;
        Ok(())
    }

    fn deserialize<R: Read>(reader: &mut R) -> Result<Self, SerializeError> {
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }
}

impl BinarySerializable for u32 {
    fn serialized_size(&self) -> usize {
        4
    }

    fn serialize<W: Write>(&self, writer: &mut W) -> Result<(), SerializeError> {
        writer.write_all(&self.to_le_bytes())?;
        Ok(())
    }

    fn deserialize<R: Read>(reader: &mut R) -> Result<Self, SerializeError> {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f64_serialization() {
        let val: f64 = 3.14159;
        let bytes = val.to_bytes();
        let restored = f64::from_bytes(&bytes).unwrap();
        assert_eq!(val, restored);
    }

    #[test]
    fn test_u64_serialization() {
        let val: u64 = 0xDEADBEEF_CAFEBABE;
        let bytes = val.to_bytes();
        let restored = u64::from_bytes(&bytes).unwrap();
        assert_eq!(val, restored);
    }

    #[cfg(feature = "compression")]
    #[test]
    fn test_compressed_f64_roundtrip() {
        let val: f64 = 3.14159;
        let compressed = val.to_compressed_bytes().unwrap();
        let restored = f64::from_compressed_bytes(&compressed).unwrap();
        assert_eq!(val, restored);
    }

    #[cfg(feature = "compression")]
    #[test]
    fn test_compressed_u64_roundtrip() {
        let val: u64 = 0xDEADBEEF_CAFEBABE;
        let compressed = val.to_compressed_bytes().unwrap();
        let restored = u64::from_compressed_bytes(&compressed).unwrap();
        assert_eq!(val, restored);
    }
}
