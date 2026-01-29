//! Serializes witness data (tensors) into bytes.
//!
//! Uses a simple binary format:
//! [Shape Rank (u8)] [Shape Dims (u64...)] [Data (f64...)]

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
