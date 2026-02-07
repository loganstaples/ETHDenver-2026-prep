//! Trait for types that can generate ZK witnesses.

/// A witness for a ZK proof, containing all intermediate values needed.
pub trait Witness: Sized {
    /// The field element type used in the witness.
    type FieldElement;

    /// Returns the witness as a vector of field elements.
    fn to_field_elements(&self) -> Vec<Self::FieldElement>;

    /// Returns the number of field elements in this witness.
    fn element_count(&self) -> usize {
        self.to_field_elements().len()
    }
}

/// Trait for computations that can produce a ZK witness.
pub trait Provable {
    /// The witness type produced by this computation.
    type Witness: Witness;

    /// Generates a witness for this computation.
    ///
    /// The witness contains all intermediate values needed to verify
    /// the computation in a ZK circuit.
    fn generate_witness(&self) -> Self::Witness;

    /// Returns the public inputs for the proof.
    fn public_inputs(&self) -> Vec<u64>;

    /// Returns an identifier for the circuit type.
    fn circuit_id(&self) -> &'static str;
}

/// A simple witness implementation using u64 field elements.
#[derive(Debug, Clone)]
pub struct SimpleWitness {
    elements: Vec<u64>,
}

impl SimpleWitness {
    /// Creates a new witness from field elements.
    pub fn new(elements: Vec<u64>) -> Self {
        Self { elements }
    }

    /// Creates an empty witness.
    pub fn empty() -> Self {
        Self { elements: vec![] }
    }

    /// Appends a value to the witness.
    pub fn push(&mut self, value: u64) {
        self.elements.push(value);
    }

    /// Appends multiple values to the witness.
    pub fn extend(&mut self, values: impl IntoIterator<Item = u64>) {
        self.elements.extend(values);
    }
}

impl Witness for SimpleWitness {
    type FieldElement = u64;

    fn to_field_elements(&self) -> Vec<u64> {
        self.elements.clone()
    }
}

/// A batch witness that shares common structure across multiple items.
///
/// In production batch training, individual witnesses repeat common data
/// (tensor shape, model metadata). `BatchWitness` factors this out, reducing
/// total serialization overhead.
#[derive(Debug, Clone)]
pub struct BatchWitness {
    /// Shared structure across all items (e.g., shape, metadata).
    pub common: Vec<u64>,
    /// Per-item witness data (values + errors for each item).
    pub items: Vec<Vec<u64>>,
}

impl BatchWitness {
    /// Creates a new batch witness.
    pub fn new(common: Vec<u64>, items: Vec<Vec<u64>>) -> Self {
        Self { common, items }
    }

    /// Returns the number of items in the batch.
    pub fn batch_size(&self) -> usize {
        self.items.len()
    }

    /// Returns the total number of field elements across all items.
    pub fn total_elements(&self) -> usize {
        self.common.len() + self.items.iter().map(|item| item.len()).sum::<usize>()
    }

    /// Serializes to a flat vector of field elements.
    ///
    /// Format: [batch_size, common_len, common..., item_0_len, item_0..., item_1_len, item_1..., ...]
    pub fn to_field_elements(&self) -> Vec<u64> {
        let mut elements = Vec::with_capacity(self.total_elements() + 2 + self.items.len());
        elements.push(self.items.len() as u64);
        elements.push(self.common.len() as u64);
        elements.extend(&self.common);
        for item in &self.items {
            elements.push(item.len() as u64);
            elements.extend(item);
        }
        elements
    }
}

/// Error for batch witness generation.
#[derive(Debug, Clone)]
pub enum BatchError {
    /// Items in the batch have mismatched shapes.
    ShapeMismatch {
        expected: Vec<u64>,
        actual: Vec<u64>,
        index: usize,
    },
    /// Batch is empty.
    EmptyBatch,
}

impl std::fmt::Display for BatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchError::ShapeMismatch { expected, actual, index } => {
                write!(
                    f,
                    "Shape mismatch at index {}: expected {:?}, got {:?}",
                    index, expected, actual
                )
            }
            BatchError::EmptyBatch => write!(f, "Cannot create batch witness from empty batch"),
        }
    }
}

impl std::error::Error for BatchError {}

/// Trait for types that support batch witness generation.
///
/// Batch witnesses share common structure (shape, metadata) across items,
/// reducing serialization overhead compared to generating individual witnesses.
pub trait BatchProvable: Provable {
    /// Generates a batch witness from multiple items of the same shape.
    ///
    /// Returns `Err(BatchError::ShapeMismatch)` if items have different shapes,
    /// or `Err(BatchError::EmptyBatch)` if the slice is empty.
    fn generate_batch_witness(items: &[Self]) -> Result<BatchWitness, BatchError>
    where
        Self: Sized;

    /// Returns combined public inputs for a batch of items.
    fn batch_public_inputs(items: &[Self]) -> Vec<u64>
    where
        Self: Sized;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_witness() {
        let mut w = SimpleWitness::empty();
        w.push(1);
        w.push(2);
        w.push(3);
        assert_eq!(w.element_count(), 3);
        assert_eq!(w.to_field_elements(), vec![1, 2, 3]);
    }

    #[test]
    fn test_batch_witness_serialization() {
        let bw = BatchWitness::new(
            vec![2, 3],           // common: shape [2, 3]
            vec![
                vec![10, 20, 30, 40, 50, 60],  // item 0 values
                vec![11, 21, 31, 41, 51, 61],  // item 1 values
            ],
        );

        assert_eq!(bw.batch_size(), 2);
        assert_eq!(bw.total_elements(), 2 + 6 + 6);

        let flat = bw.to_field_elements();
        // Format: [batch_size=2, common_len=2, 2, 3, item0_len=6, 10..60, item1_len=6, 11..61]
        assert_eq!(flat[0], 2); // batch_size
        assert_eq!(flat[1], 2); // common_len
        assert_eq!(flat[2], 2); // shape dim 0
        assert_eq!(flat[3], 3); // shape dim 1
        assert_eq!(flat[4], 6); // item 0 len
    }

    #[test]
    fn test_batch_witness_empty() {
        let bw = BatchWitness::new(vec![1, 2, 3], vec![]);
        assert_eq!(bw.batch_size(), 0);
        assert_eq!(bw.total_elements(), 3);
    }
}
