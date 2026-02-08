//! Constraint Builder.
//!
//! A high-level API for building Halo2 constraints with named regions
//! and automatic wire allocation.

use halo2_proofs::{
    circuit::Value,
    plonk::{Column, Advice, Selector, ConstraintSystem},
    poly::Rotation,
};
use halo2curves::ff::PrimeField;
use std::collections::HashMap;
use std::marker::PhantomData;

/// Wire identifier for tracking allocations.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct WireId(pub String);

impl WireId {
    /// Creates a new wire ID.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

impl std::fmt::Display for WireId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Configuration for the constraint builder.
#[derive(Debug, Clone)]
pub struct BuilderConfig<F: PrimeField> {
    /// Advice columns.
    pub advice: Vec<Column<Advice>>,
    /// Selectors for gates.
    pub selectors: Vec<Selector>,
    /// Phantom data.
    _marker: PhantomData<F>,
}

impl<F: PrimeField> BuilderConfig<F> {
    /// Creates a new builder configuration.
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        num_advice: usize,
        num_selectors: usize,
    ) -> Self {
        let advice: Vec<_> = (0..num_advice)
            .map(|_| {
                let col = meta.advice_column();
                meta.enable_equality(col);
                col
            })
            .collect();

        let selectors: Vec<_> = (0..num_selectors)
            .map(|_| meta.selector())
            .collect();

        Self {
            advice,
            selectors,
            _marker: PhantomData,
        }
    }

    /// Creates an addition gate: a + b = c.
    pub fn create_add_gate(meta: &mut ConstraintSystem<F>, advice: &[Column<Advice>], selector: Selector) {
        let a = advice[0];
        let b = advice[1];
        let c = advice[2];

        meta.create_gate("add", |meta| {
            let s = meta.query_selector(selector);
            let a_val = meta.query_advice(a, Rotation::cur());
            let b_val = meta.query_advice(b, Rotation::cur());
            let c_val = meta.query_advice(c, Rotation::cur());

            vec![s * (a_val + b_val - c_val)]
        });
    }

    /// Creates a multiplication gate: a * b = c.
    pub fn create_mul_gate(meta: &mut ConstraintSystem<F>, advice: &[Column<Advice>], selector: Selector) {
        let a = advice[0];
        let b = advice[1];
        let c = advice[2];

        meta.create_gate("mul", |meta| {
            let s = meta.query_selector(selector);
            let a_val = meta.query_advice(a, Rotation::cur());
            let b_val = meta.query_advice(b, Rotation::cur());
            let c_val = meta.query_advice(c, Rotation::cur());

            vec![s * (a_val * b_val - c_val)]
        });
    }
}

/// High-level constraint builder that tracks gate creation.
#[derive(Debug)]
pub struct ConstraintBuilder {
    /// Named gates that have been created.
    gates: HashMap<String, GateInfo>,
    /// Gate counter.
    gate_count: usize,
}

/// Information about a created gate.
#[derive(Debug, Clone)]
pub struct GateInfo {
    /// Gate index.
    pub index: usize,
    /// Gate type.
    pub gate_type: GateType,
}

/// Type of gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateType {
    /// Addition gate.
    Add,
    /// Multiplication gate.
    Mul,
    /// Range check gate.
    Range,
    /// Custom gate.
    Custom,
}

impl Default for ConstraintBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstraintBuilder {
    /// Creates a new constraint builder.
    pub fn new() -> Self {
        Self {
            gates: HashMap::new(),
            gate_count: 0,
        }
    }

    /// Records a gate creation.
    pub fn record_gate(&mut self, name: &str, gate_type: GateType) -> usize {
        let index = self.gate_count;
        self.gates.insert(name.to_string(), GateInfo { index, gate_type });
        self.gate_count += 1;
        index
    }

    /// Returns the number of gates created.
    pub fn gate_count(&self) -> usize {
        self.gate_count
    }

    /// Gets gate info by name.
    pub fn get_gate(&self, name: &str) -> Option<&GateInfo> {
        self.gates.get(name)
    }

    /// Lists all gates.
    pub fn gates(&self) -> impl Iterator<Item = (&String, &GateInfo)> {
        self.gates.iter()
    }
}

/// Wire allocator for tracking cell assignments.
#[derive(Debug)]
pub struct WireAllocator<F: PrimeField> {
    /// Allocated wires.
    allocations: HashMap<WireId, WireAllocation<F>>,
    /// Next row to allocate.
    next_row: usize,
    /// Column capacity.
    num_columns: usize,
}

/// A wire allocation.
#[derive(Debug, Clone)]
pub struct WireAllocation<F: PrimeField> {
    /// Column index.
    pub column: usize,
    /// Row index.
    pub row: usize,
    /// Value (if known).
    pub value: Value<F>,
}

impl<F: PrimeField> WireAllocator<F> {
    /// Creates a new wire allocator.
    pub fn new(num_columns: usize) -> Self {
        Self {
            allocations: HashMap::new(),
            next_row: 0,
            num_columns,
        }
    }

    /// Allocates a new wire.
    pub fn alloc(&mut self, id: WireId, value: Value<F>) -> WireAllocation<F> {
        let column = self.allocations.len() % self.num_columns;
        let row = self.next_row;

        if column == self.num_columns - 1 {
            self.next_row += 1;
        }

        let allocation = WireAllocation { column, row, value };
        self.allocations.insert(id.clone(), allocation.clone());
        allocation
    }

    /// Gets an existing wire allocation.
    pub fn get(&self, id: &WireId) -> Option<&WireAllocation<F>> {
        self.allocations.get(id)
    }

    /// Returns the number of rows used.
    pub fn rows_used(&self) -> usize {
        self.next_row + 1
    }

    /// Clears all allocations.
    pub fn clear(&mut self) {
        self.allocations.clear();
        self.next_row = 0;
    }
}

/// Region builder for a specific circuit region.
pub struct RegionBuilder<'a, F: PrimeField> {
    /// Region name.
    name: String,
    /// Wire allocator.
    allocator: &'a mut WireAllocator<F>,
    /// Assigned cells in this region.
    assigned: Vec<(WireId, Value<F>)>,
}

impl<'a, F: PrimeField> RegionBuilder<'a, F> {
    /// Creates a new region builder.
    pub fn new(name: impl Into<String>, allocator: &'a mut WireAllocator<F>) -> Self {
        Self {
            name: name.into(),
            allocator,
            assigned: Vec::new(),
        }
    }

    /// Assigns a value to a named wire.
    pub fn assign(&mut self, name: &str, value: Value<F>) -> WireId {
        let id = WireId::new(format!("{}_{}", self.name, name));
        self.allocator.alloc(id.clone(), value);
        self.assigned.push((id.clone(), value));
        id
    }

    /// Returns the region name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns assigned wire IDs.
    pub fn assigned_wires(&self) -> impl Iterator<Item = &WireId> {
        self.assigned.iter().map(|(id, _)| id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2curves::bn256::Fr;

    #[test]
    fn test_wire_id() {
        let id = WireId::new("test_wire");
        assert_eq!(id.0, "test_wire");
    }

    #[test]
    fn test_wire_allocator() {
        let mut allocator = WireAllocator::<Fr>::new(3);

        let a = allocator.alloc(WireId::new("a"), Value::known(Fr::from(1)));
        let b = allocator.alloc(WireId::new("b"), Value::known(Fr::from(2)));
        let c = allocator.alloc(WireId::new("c"), Value::known(Fr::from(3)));

        assert_eq!(a.column, 0);
        assert_eq!(b.column, 1);
        assert_eq!(c.column, 2);
    }

    #[test]
    fn test_region_builder() {
        let mut allocator = WireAllocator::<Fr>::new(3);
        let mut region = RegionBuilder::new("test", &mut allocator);

        let id = region.assign("input", Value::known(Fr::from(42)));
        assert!(id.0.contains("input"));
    }

    #[test]
    fn test_constraint_builder() {
        let mut builder = ConstraintBuilder::new();

        let idx = builder.record_gate("add_gate", GateType::Add);
        assert_eq!(idx, 0);
        assert_eq!(builder.gate_count(), 1);

        let info = builder.get_gate("add_gate").unwrap();
        assert_eq!(info.gate_type, GateType::Add);
    }
}
