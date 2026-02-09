//! Serializes and deserializes witness data (tensors) into bytes.
//!
//! Uses a simple binary format:
//! [Shape Rank (u8)] [Shape Dims (u64...)] [Data (f64...)]

use helix_core::types::bounded_value::BoundedValue;
use helix_core::types::BoundedTensor;

/// Serializes a single tensor to a byte vector.
pub fn serialize_tensor(tensor: &BoundedTensor) -> Vec<u8> {
    let mut bytes = Vec::new();

    // 1. Serialize Shape
    let shape = tensor.shape();
    let rank = shape.len() as u8;
    bytes.push(rank);
    for dim in shape {
        bytes.extend_from_slice(&(*dim as u64).to_le_bytes());
    }

    // 2. Serialize Data (Values only for now, or Errors too?)
    // Witness usually needs Values. Error bounds might be part of public input or witness depending on circuit.
    // Let's serialize values as f64 (IEEE 754).
    // In a real ZK system, these would be converted to Field Elements.
    for val in tensor.data() {
        bytes.extend_from_slice(&val.value().to_le_bytes());
    }

    bytes
}

/// Serializes a list of tensors.
pub fn serialize_tensors(tensors: &[BoundedTensor]) -> Vec<u8> {
    let mut bytes = Vec::new();
    // Prefix with count
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    for t in tensors {
        bytes.extend(serialize_tensor(t));
    }
    bytes
}

/// Deserializes a single tensor from a byte slice.
///
/// Returns the deserialized tensor and the number of bytes consumed,
/// or `None` if the data is malformed or too short.
pub fn deserialize_tensor(bytes: &[u8]) -> Option<(BoundedTensor, usize)> {
    if bytes.is_empty() {
        return None;
    }

    let mut cursor = 0;

    // 1. Read rank
    let rank = bytes[cursor] as usize;
    cursor += 1;

    // 2. Read shape dimensions
    let mut shape = Vec::with_capacity(rank);
    for _ in 0..rank {
        if cursor + 8 > bytes.len() {
            return None;
        }
        let dim = u64::from_le_bytes(bytes[cursor..cursor + 8].try_into().ok()?) as usize;
        shape.push(dim);
        cursor += 8;
    }

    // 3. Compute total elements
    let total_elements: usize = if shape.is_empty() {
        0
    } else {
        shape.iter().product()
    };

    // 4. Read f64 values
    let data_bytes_needed = total_elements * 8;
    if cursor + data_bytes_needed > bytes.len() {
        return None;
    }

    let mut data = Vec::with_capacity(total_elements);
    for _ in 0..total_elements {
        let val = f64::from_le_bytes(bytes[cursor..cursor + 8].try_into().ok()?);
        data.push(BoundedValue::exact(val));
        cursor += 8;
    }

    let tensor = BoundedTensor::new(data, shape);
    Some((tensor, cursor))
}

/// Deserializes a list of tensors from a byte slice.
///
/// The format matches `serialize_tensors`: a u64 count prefix followed by
/// that many serialized tensors. Returns `None` if the data is malformed.
pub fn deserialize_tensors(bytes: &[u8]) -> Option<Vec<BoundedTensor>> {
    if bytes.len() < 8 {
        return None;
    }

    let count = u64::from_le_bytes(bytes[0..8].try_into().ok()?) as usize;
    let mut cursor = 8;
    let mut tensors = Vec::with_capacity(count);

    for _ in 0..count {
        if cursor >= bytes.len() && count > 0 {
            return None;
        }
        let (tensor, consumed) = deserialize_tensor(&bytes[cursor..])?;
        tensors.push(tensor);
        cursor += consumed;
    }

    Some(tensors)
}

/// Computes a 32-byte state hash using FNV-1a.
///
/// This is a non-cryptographic hash used for IVC state chaining.
/// It provides uniqueness for boundary state identification but is
/// not intended for security-critical applications.
pub fn compute_state_hash(data: &[u8]) -> [u8; 32] {
    // FNV-1a 64-bit, then extend to 32 bytes by hashing in 4 segments
    // with different initial offsets to fill the full 32-byte output.
    let mut result = [0u8; 32];

    for segment in 0..4u64 {
        let mut hash: u64 = 0xcbf29ce484222325u64.wrapping_add(segment.wrapping_mul(0x100000001b3));
        for &byte in data {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        let offset = (segment as usize) * 8;
        result[offset..offset + 8].copy_from_slice(&hash.to_le_bytes());
    }

    result
}
