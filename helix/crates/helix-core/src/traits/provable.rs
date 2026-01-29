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
}
