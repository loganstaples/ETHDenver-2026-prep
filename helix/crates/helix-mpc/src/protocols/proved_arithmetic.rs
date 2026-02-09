//! Proved Arithmetic Operations.
//!
//! This module provides MPC arithmetic operations that capture witness data
//! for ZK proof generation. Every operation records its inputs, outputs, and
//! error bounds, enabling later proof that the computation was performed correctly.
//!
//! # Design
//!
//! The `ProvedArithmetic` struct wraps standard `SecureArithmetic` operations
//! while maintaining a `WitnessCapture` that records all intermediate values.
//! After computation, the capture can be converted to a circuit witness.
//!
//! # Witness Structure
//!
//! For each operation, we capture:
//! - Input values (as field elements)
//! - Output values (as field elements)
//! - Error bounds for the operation
//! - Beaver triple data (for multiplications)
//! - Operation metadata (type, timestamp)
//!
//! This enables generating ZK proofs that verify:
//! - The computation was performed correctly
//! - Error bounds were respected
//! - Beaver triples were used correctly

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::beaver::pool::BeaverPool;
use crate::beaver::triple::BeaverTriple;
use crate::error::MPCResult;
use crate::field::Fr;
use crate::protocols::arithmetic::SecureArithmetic;
use crate::sharing::tensor::TensorShare;
use sha2::{Digest, Sha256};
use tracing::{debug, trace, warn, instrument};

/// Types of operations that can be witnessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WitnessedOperation {
    /// Scalar addition (local).
    Add,
    /// Scalar subtraction (local).
    Sub,
    /// Scalar multiplication (Beaver triple).
    Mul,
    /// Addition of public constant.
    AddPublic,
    /// Scaling by public constant.
    Scale,
    /// Vector addition (element-wise).
    VectorAdd,
    /// Vector subtraction (element-wise).
    VectorSub,
    /// Vector scaling.
    VectorScale,
    /// Vector multiplication (element-wise, Beaver).
    VectorMul,
    /// Matrix multiplication (Beaver).
    MatMul,
    /// ReLU activation.
    ReLU,
    /// Weight update step.
    WeightUpdate,
    /// Gradient computation.
    GradientCompute,
    /// Forward pass.
    ForwardPass,
    /// Backward pass.
    BackwardPass,
}

impl std::fmt::Display for WitnessedOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WitnessedOperation::Add => write!(f, "Add"),
            WitnessedOperation::Sub => write!(f, "Sub"),
            WitnessedOperation::Mul => write!(f, "Mul"),
            WitnessedOperation::AddPublic => write!(f, "AddPublic"),
            WitnessedOperation::Scale => write!(f, "Scale"),
            WitnessedOperation::VectorAdd => write!(f, "VectorAdd"),
            WitnessedOperation::VectorSub => write!(f, "VectorSub"),
            WitnessedOperation::VectorScale => write!(f, "VectorScale"),
            WitnessedOperation::VectorMul => write!(f, "VectorMul"),
            WitnessedOperation::MatMul => write!(f, "MatMul"),
            WitnessedOperation::ReLU => write!(f, "ReLU"),
            WitnessedOperation::WeightUpdate => write!(f, "WeightUpdate"),
            WitnessedOperation::GradientCompute => write!(f, "GradientCompute"),
            WitnessedOperation::ForwardPass => write!(f, "ForwardPass"),
            WitnessedOperation::BackwardPass => write!(f, "BackwardPass"),
        }
    }
}

/// A single witnessed operation with all captured data.
#[derive(Debug, Clone)]
pub struct OperationWitness {
    /// Type of operation.
    pub op_type: WitnessedOperation,
    /// Input values.
    pub inputs: Vec<Fr>,
    /// Output values.
    pub outputs: Vec<Fr>,
    /// Error bound for this operation.
    pub error_bound: Fr,
    /// Beaver triple data if applicable (a, b, c shares).
    pub beaver_data: Option<BeaverWitness>,
    /// Operation timestamp (nanoseconds since start).
    pub timestamp_ns: u64,
    /// Party index that performed the operation.
    pub party_index: usize,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl OperationWitness {
    /// Creates a new operation witness.
    pub fn new(
        op_type: WitnessedOperation,
        party_index: usize,
        timestamp_ns: u64,
    ) -> Self {
        Self {
            op_type,
            inputs: Vec::new(),
            outputs: Vec::new(),
            error_bound: Fr::ZERO,
            beaver_data: None,
            timestamp_ns,
            party_index,
            metadata: HashMap::new(),
        }
    }

    /// Adds an input value.
    pub fn add_input(&mut self, value: Fr) {
        self.inputs.push(value);
    }

    /// Adds multiple input values.
    pub fn add_inputs(&mut self, values: &[Fr]) {
        self.inputs.extend(values.iter().cloned());
    }

    /// Sets the output values.
    pub fn set_output(&mut self, value: Fr) {
        self.outputs.push(value);
    }

    /// Sets multiple output values.
    pub fn set_outputs(&mut self, values: &[Fr]) {
        self.outputs.extend(values.iter().cloned());
    }

    /// Sets the error bound.
    pub fn set_error_bound(&mut self, error: Fr) {
        self.error_bound = error;
    }

    /// Sets Beaver triple data.
    pub fn set_beaver(&mut self, beaver: BeaverWitness) {
        self.beaver_data = Some(beaver);
    }

    /// Adds metadata.
    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// Computes a commitment to this witness.
    pub fn commitment(&self, blinding: &[u8; 32]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(&[self.op_type as u8]);
        for v in &self.inputs {
            hasher.update(&v.to_bytes_le());
        }
        for v in &self.outputs {
            hasher.update(&v.to_bytes_le());
        }
        hasher.update(&self.error_bound.to_bytes_le());
        if let Some(ref beaver) = self.beaver_data {
            hasher.update(&beaver.a.to_bytes_le());
            hasher.update(&beaver.b.to_bytes_le());
            hasher.update(&beaver.c.to_bytes_le());
        }
        hasher.update(&self.timestamp_ns.to_le_bytes());
        hasher.update(blinding);
        hasher.finalize().into()
    }

    /// Returns the number of inputs.
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// Returns the number of outputs.
    pub fn num_outputs(&self) -> usize {
        self.outputs.len()
    }
}

/// Captured Beaver triple data for witness.
#[derive(Debug, Clone)]
pub struct BeaverWitness {
    /// Share of first random value.
    pub a: Fr,
    /// Share of second random value.
    pub b: Fr,
    /// Share of product.
    pub c: Fr,
    /// Opened d = x - a value (public).
    pub opened_d: Fr,
    /// Opened e = y - b value (public).
    pub opened_e: Fr,
    /// Whether this triple was verified.
    pub verified: bool,
}

impl BeaverWitness {
    /// Creates from a Beaver triple and opened values.
    pub fn from_triple(triple: &BeaverTriple, opened_d: Fr, opened_e: Fr) -> Self {
        Self {
            a: triple.a.clone(),
            b: triple.b.clone(),
            c: triple.c.clone(),
            opened_d,
            opened_e,
            verified: false,
        }
    }

    /// Marks this triple as verified.
    pub fn mark_verified(&mut self) {
        self.verified = true;
    }
}

/// Collector for capturing witness data from MPC operations.
#[derive(Debug)]
pub struct WitnessCapture {
    /// Collected operation witnesses.
    operations: Vec<OperationWitness>,
    /// Start time for timestamps.
    start_time: Instant,
    /// Party index.
    party_index: usize,
    /// Current operation being built.
    current_op: Option<OperationWitness>,
    /// Total accumulated error.
    total_error: Fr,
    /// Operation counter.
    op_counter: u64,
    /// Whether capturing is enabled.
    enabled: bool,
    /// Base error per operation.
    base_error: Fr,
}

impl WitnessCapture {
    /// Creates a new witness capture for a party.
    pub fn new(party_index: usize) -> Self {
        Self {
            operations: Vec::new(),
            start_time: Instant::now(),
            party_index,
            current_op: None,
            total_error: Fr::ZERO,
            op_counter: 0,
            enabled: true,
            base_error: Fr::from_f64(1e-6),
        }
    }

    /// Sets the base error per operation.
    pub fn set_base_error(&mut self, error: f64) {
        self.base_error = Fr::from_f64(error);
    }

    /// Enables or disables capturing.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Returns whether capturing is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Starts a new operation.
    pub fn start_operation(&mut self, op_type: WitnessedOperation) -> u64 {
        if !self.enabled {
            return 0;
        }

        let timestamp = self.start_time.elapsed().as_nanos() as u64;
        self.current_op = Some(OperationWitness::new(
            op_type,
            self.party_index,
            timestamp,
        ));
        self.op_counter += 1;

        trace!(
            party = self.party_index,
            op_type = %op_type,
            op_id = self.op_counter,
            "Starting operation"
        );

        self.op_counter
    }

    /// Records inputs for the current operation.
    pub fn record_inputs(&mut self, inputs: &[Fr]) {
        if !self.enabled {
            return;
        }
        if let Some(ref mut op) = self.current_op {
            op.add_inputs(inputs);
        }
    }

    /// Records a single input.
    pub fn record_input(&mut self, input: Fr) {
        if !self.enabled {
            return;
        }
        if let Some(ref mut op) = self.current_op {
            op.add_input(input);
        }
    }

    /// Records outputs for the current operation.
    pub fn record_outputs(&mut self, outputs: &[Fr]) {
        if !self.enabled {
            return;
        }
        if let Some(ref mut op) = self.current_op {
            op.set_outputs(outputs);
        }
    }

    /// Records a single output.
    pub fn record_output(&mut self, output: Fr) {
        if !self.enabled {
            return;
        }
        if let Some(ref mut op) = self.current_op {
            op.set_output(output);
        }
    }

    /// Records Beaver triple data.
    pub fn record_beaver(&mut self, beaver: BeaverWitness) {
        if !self.enabled {
            return;
        }
        if let Some(ref mut op) = self.current_op {
            op.set_beaver(beaver);
        }
    }

    /// Sets the error bound for the current operation.
    pub fn record_error(&mut self, error: Fr) {
        if !self.enabled {
            return;
        }
        if let Some(ref mut op) = self.current_op {
            op.set_error_bound(error.clone());
        }
        self.total_error = Fr::add(&self.total_error, &error);
    }

    /// Sets the error bound based on operation type and dimensions.
    pub fn record_default_error(&mut self, multiplier: u64) {
        let error = Fr::mul(&self.base_error, &Fr::from_u64(multiplier));
        self.record_error(error);
    }

    /// Adds metadata to the current operation.
    pub fn record_metadata(&mut self, key: &str, value: &str) {
        if !self.enabled {
            return;
        }
        if let Some(ref mut op) = self.current_op {
            op.add_metadata(key, value);
        }
    }

    /// Finishes the current operation and adds it to the collection.
    pub fn finish_operation(&mut self) {
        if !self.enabled {
            return;
        }
        if let Some(op) = self.current_op.take() {
            trace!(
                party = self.party_index,
                op_type = %op.op_type,
                num_inputs = op.num_inputs(),
                num_outputs = op.num_outputs(),
                "Finished operation"
            );
            self.operations.push(op);
        }
    }

    /// Returns all collected operations.
    pub fn operations(&self) -> &[OperationWitness] {
        &self.operations
    }

    /// Returns the total accumulated error.
    pub fn total_error(&self) -> &Fr {
        &self.total_error
    }

    /// Returns the number of operations captured.
    pub fn num_operations(&self) -> usize {
        self.operations.len()
    }

    /// Clears all captured operations.
    pub fn clear(&mut self) {
        self.operations.clear();
        self.current_op = None;
        self.total_error = Fr::ZERO;
        self.op_counter = 0;
    }

    /// Generates a commitment to all captured operations.
    pub fn full_commitment(&self, blinding: &[u8; 32]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(&(self.party_index as u64).to_le_bytes());
        hasher.update(&(self.operations.len() as u64).to_le_bytes());

        for op in &self.operations {
            let op_commit = op.commitment(blinding);
            hasher.update(&op_commit);
        }

        hasher.update(&self.total_error.to_bytes_le());
        hasher.update(blinding);
        hasher.finalize().into()
    }

    /// Extracts all inputs across all operations.
    pub fn all_inputs(&self) -> Vec<Fr> {
        self.operations
            .iter()
            .flat_map(|op| op.inputs.clone())
            .collect()
    }

    /// Extracts all outputs across all operations.
    pub fn all_outputs(&self) -> Vec<Fr> {
        self.operations
            .iter()
            .flat_map(|op| op.outputs.clone())
            .collect()
    }

    /// Extracts all Beaver witnesses.
    pub fn all_beaver_witnesses(&self) -> Vec<&BeaverWitness> {
        self.operations
            .iter()
            .filter_map(|op| op.beaver_data.as_ref())
            .collect()
    }
}

impl Default for WitnessCapture {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Thread-safe witness capture.
pub type SharedWitnessCapture = Arc<Mutex<WitnessCapture>>;

/// Creates a shared witness capture.
pub fn create_shared_capture(party_index: usize) -> SharedWitnessCapture {
    Arc::new(Mutex::new(WitnessCapture::new(party_index)))
}

/// Proved arithmetic operations with witness capture.
///
/// This struct wraps `SecureArithmetic` to automatically capture
/// witness data for ZK proof generation.
pub struct ProvedArithmetic {
    /// Party index.
    party_index: usize,
    /// Witness capture.
    capture: WitnessCapture,
    /// Base error per operation.
    base_error: f64,
}

impl ProvedArithmetic {
    /// Creates a new proved arithmetic instance.
    pub fn new(party_index: usize) -> Self {
        Self {
            party_index,
            capture: WitnessCapture::new(party_index),
            base_error: 1e-6,
        }
    }

    /// Sets the base error per operation.
    pub fn with_base_error(mut self, error: f64) -> Self {
        self.base_error = error;
        self.capture.set_base_error(error);
        self
    }

    /// Enables or disables witness capture.
    pub fn set_capture_enabled(&mut self, enabled: bool) {
        self.capture.set_enabled(enabled);
    }

    /// Returns a reference to the witness capture.
    pub fn capture(&self) -> &WitnessCapture {
        &self.capture
    }

    /// Returns a mutable reference to the witness capture.
    pub fn capture_mut(&mut self) -> &mut WitnessCapture {
        &mut self.capture
    }

    /// Takes ownership of the witness capture, resetting internal state.
    pub fn take_capture(&mut self) -> WitnessCapture {
        std::mem::replace(
            &mut self.capture,
            WitnessCapture::new(self.party_index),
        )
    }

    // ========== LOCAL OPERATIONS (no communication) ==========

    /// Adds two scalar shares with witness capture.
    #[instrument(skip(self, x_share, y_share), level = "trace")]
    pub fn add_shares(&mut self, x_share: &Fr, y_share: &Fr) -> Fr {
        self.capture.start_operation(WitnessedOperation::Add);
        self.capture.record_inputs(&[x_share.clone(), y_share.clone()]);

        let result = SecureArithmetic::add_shares(x_share, y_share);

        self.capture.record_output(result.clone());
        self.capture.record_default_error(1);
        self.capture.finish_operation();

        result
    }

    /// Subtracts two scalar shares with witness capture.
    #[instrument(skip(self, x_share, y_share), level = "trace")]
    pub fn sub_shares(&mut self, x_share: &Fr, y_share: &Fr) -> Fr {
        self.capture.start_operation(WitnessedOperation::Sub);
        self.capture.record_inputs(&[x_share.clone(), y_share.clone()]);

        let result = SecureArithmetic::sub_shares(x_share, y_share);

        self.capture.record_output(result.clone());
        self.capture.record_default_error(1);
        self.capture.finish_operation();

        result
    }

    /// Scales a share by a public constant with witness capture.
    #[instrument(skip(self, x_share, constant), level = "trace")]
    pub fn scale_share(&mut self, x_share: &Fr, constant: &Fr) -> Fr {
        self.capture.start_operation(WitnessedOperation::Scale);
        self.capture.record_inputs(&[x_share.clone(), constant.clone()]);

        let result = SecureArithmetic::scale_share(x_share, constant);

        self.capture.record_output(result.clone());
        self.capture.record_default_error(1);
        self.capture.finish_operation();

        result
    }

    /// Adds a public constant to a share with witness capture.
    #[instrument(skip(self, x_share, constant), level = "trace")]
    pub fn add_public(&mut self, x_share: &Fr, constant: &Fr) -> Fr {
        self.capture.start_operation(WitnessedOperation::AddPublic);
        self.capture.record_inputs(&[x_share.clone(), constant.clone()]);
        self.capture.record_metadata("party_index", &self.party_index.to_string());

        let result = SecureArithmetic::add_public(x_share, constant, self.party_index);

        self.capture.record_output(result.clone());
        self.capture.record_default_error(1);
        self.capture.finish_operation();

        result
    }

    // ========== INTERACTIVE OPERATIONS (require communication) ==========

    /// Secure multiplication with Beaver triple and witness capture.
    ///
    /// This captures the full Beaver protocol execution for the witness.
    #[instrument(skip(self, triple, opened_d, opened_e), level = "debug")]
    pub fn multiply_shares(
        &mut self,
        triple: &BeaverTriple,
        opened_d: &Fr,
        opened_e: &Fr,
    ) -> Fr {
        self.capture.start_operation(WitnessedOperation::Mul);

        // Record Beaver triple data.
        let beaver = BeaverWitness::from_triple(triple, opened_d.clone(), opened_e.clone());
        self.capture.record_beaver(beaver);

        // The inputs to multiplication are derived from d, e and triple values.
        // x = d + a, y = e + b
        let x_share = Fr::add(opened_d, &triple.a);
        let y_share = Fr::add(opened_e, &triple.b);
        self.capture.record_inputs(&[x_share, y_share]);

        let result = SecureArithmetic::multiply_shares(triple, opened_d, opened_e, self.party_index);

        self.capture.record_output(result.clone());
        self.capture.record_default_error(2); // Multiplication has higher error
        self.capture.finish_operation();

        debug!(
            party = self.party_index,
            result_f64 = result.to_f64(),
            "Beaver multiplication complete"
        );

        result
    }

    /// Computes d and e masks for Beaver multiplication with witness capture.
    #[instrument(skip(self, x_share, y_share, triple), level = "trace")]
    pub fn beaver_mask(
        &mut self,
        x_share: &Fr,
        y_share: &Fr,
        triple: &BeaverTriple,
    ) -> (Fr, Fr) {
        let (d_share, e_share) = SecureArithmetic::beaver_mask(x_share, y_share, triple);

        // Record as metadata for the next multiplication operation
        self.capture.record_metadata("d_share", &format!("{:?}", d_share.to_f64()));
        self.capture.record_metadata("e_share", &format!("{:?}", e_share.to_f64()));

        (d_share, e_share)
    }

    // ========== VECTOR OPERATIONS ==========

    /// Adds two tensor shares element-wise with witness capture.
    #[instrument(skip(self, x, y), level = "debug")]
    pub fn add_tensor_shares(&mut self, x: &TensorShare, y: &TensorShare) -> MPCResult<TensorShare> {
        self.capture.start_operation(WitnessedOperation::VectorAdd);
        self.capture.record_inputs(&x.data);
        self.capture.record_inputs(&y.data);
        self.capture.record_metadata("shape", &format!("{:?}", x.shape));

        let result = SecureArithmetic::add_tensor_shares(x, y)?;

        self.capture.record_outputs(&result.data);
        self.capture.record_default_error(result.data.len() as u64);
        self.capture.finish_operation();

        Ok(result)
    }

    /// Subtracts two tensor shares element-wise with witness capture.
    #[instrument(skip(self, x, y), level = "debug")]
    pub fn sub_tensor_shares(&mut self, x: &TensorShare, y: &TensorShare) -> MPCResult<TensorShare> {
        self.capture.start_operation(WitnessedOperation::VectorSub);
        self.capture.record_inputs(&x.data);
        self.capture.record_inputs(&y.data);
        self.capture.record_metadata("shape", &format!("{:?}", x.shape));

        let result = SecureArithmetic::sub_tensor_shares(x, y)?;

        self.capture.record_outputs(&result.data);
        self.capture.record_default_error(result.data.len() as u64);
        self.capture.finish_operation();

        Ok(result)
    }

    /// Scales a tensor share by a public constant with witness capture.
    #[instrument(skip(self, x, scalar), level = "debug")]
    pub fn scale_tensor_share(&mut self, x: &TensorShare, scalar: &Fr) -> TensorShare {
        self.capture.start_operation(WitnessedOperation::VectorScale);
        self.capture.record_inputs(&x.data);
        self.capture.record_input(scalar.clone());
        self.capture.record_metadata("shape", &format!("{:?}", x.shape));

        let result = SecureArithmetic::scale_tensor_share(x, scalar);

        self.capture.record_outputs(&result.data);
        self.capture.record_default_error(result.data.len() as u64);
        self.capture.finish_operation();

        result
    }

    // ========== COMPLEX OPERATIONS ==========

    /// Simulates vector element-wise multiplication with witness capture.
    ///
    /// This is useful for testing the full MPC multiplication protocol.
    #[instrument(skip(self, x_shares, y_shares, pools), level = "debug")]
    pub fn simulate_vector_multiply(
        &mut self,
        x_shares: &[Vec<Fr>],
        y_shares: &[Vec<Fr>],
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Vec<Fr>>> {
        self.capture.start_operation(WitnessedOperation::VectorMul);

        for shares in x_shares {
            self.capture.record_inputs(shares);
        }
        for shares in y_shares {
            self.capture.record_inputs(shares);
        }

        let result = SecureArithmetic::simulate_vector_multiply(x_shares, y_shares, pools)?;

        for shares in &result {
            self.capture.record_outputs(shares);
        }

        let dim = x_shares.get(0).map(|v| v.len()).unwrap_or(0);
        self.capture.record_default_error(dim as u64 * 2);
        self.capture.finish_operation();

        Ok(result)
    }

    // ========== HIGH-LEVEL TRAINING OPERATIONS ==========

    /// Records a forward pass operation with captured intermediates.
    #[instrument(skip(self, input, h_pre, h, y, loss), level = "info")]
    pub fn record_forward_pass(
        &mut self,
        input: &[Fr],
        h_pre: &[Fr],
        h: &[Fr],
        y: &[Fr],
        loss: Fr,
    ) {
        self.capture.start_operation(WitnessedOperation::ForwardPass);
        self.capture.record_inputs(input);
        self.capture.record_outputs(h_pre);
        self.capture.record_outputs(h);
        self.capture.record_outputs(y);
        self.capture.record_output(loss);

        let total_elements = (input.len() + h_pre.len() + h.len() + y.len() + 1) as u64;
        self.capture.record_default_error(total_elements);
        self.capture.finish_operation();
    }

    /// Records a backward pass operation with captured gradients.
    #[instrument(skip(self, dy, dw2, db2, dh, dh_pre, dw1, db1), level = "info")]
    pub fn record_backward_pass(
        &mut self,
        dy: &[Fr],
        dw2: &[Fr],
        db2: &[Fr],
        dh: &[Fr],
        dh_pre: &[Fr],
        dw1: &[Fr],
        db1: &[Fr],
    ) {
        self.capture.start_operation(WitnessedOperation::BackwardPass);
        self.capture.record_inputs(dy);
        self.capture.record_outputs(dw2);
        self.capture.record_outputs(db2);
        self.capture.record_outputs(dh);
        self.capture.record_outputs(dh_pre);
        self.capture.record_outputs(dw1);
        self.capture.record_outputs(db1);

        let total_elements = (dy.len() + dw2.len() + db2.len() + dh.len() + dh_pre.len() + dw1.len() + db1.len()) as u64;
        self.capture.record_default_error(total_elements);
        self.capture.finish_operation();
    }

    /// Records a weight update operation.
    #[instrument(skip(self, old_weights, gradients, new_weights, learning_rate), level = "info")]
    pub fn record_weight_update(
        &mut self,
        old_weights: &[Fr],
        gradients: &[Fr],
        new_weights: &[Fr],
        learning_rate: Fr,
    ) {
        self.capture.start_operation(WitnessedOperation::WeightUpdate);
        self.capture.record_inputs(old_weights);
        self.capture.record_inputs(gradients);
        self.capture.record_input(learning_rate);
        self.capture.record_outputs(new_weights);

        self.capture.record_default_error(old_weights.len() as u64);
        self.capture.finish_operation();
    }

    /// Records a ReLU activation operation.
    #[instrument(skip(self, input, output, mask), level = "debug")]
    pub fn record_relu(
        &mut self,
        input: &[Fr],
        output: &[Fr],
        mask: &[Fr],
    ) {
        self.capture.start_operation(WitnessedOperation::ReLU);
        self.capture.record_inputs(input);
        self.capture.record_outputs(output);
        self.capture.record_metadata("mask_sum", &format!("{}", mask.iter().fold(0, |acc, m| acc + if m.is_zero().to_bool() { 0 } else { 1 })));

        self.capture.record_default_error(input.len() as u64);
        self.capture.finish_operation();
    }
}

/// Summary statistics for captured witnesses.
#[derive(Debug, Clone)]
pub struct WitnessSummary {
    /// Total number of operations.
    pub total_operations: usize,
    /// Operations by type.
    pub operations_by_type: HashMap<WitnessedOperation, usize>,
    /// Total inputs captured.
    pub total_inputs: usize,
    /// Total outputs captured.
    pub total_outputs: usize,
    /// Number of Beaver triples used.
    pub beaver_triples_used: usize,
    /// Total accumulated error.
    pub total_error: f64,
    /// Capture duration in milliseconds.
    pub duration_ms: u64,
}

impl WitnessSummary {
    /// Creates a summary from a witness capture.
    pub fn from_capture(capture: &WitnessCapture) -> Self {
        let mut operations_by_type = HashMap::new();
        let mut total_inputs = 0;
        let mut total_outputs = 0;
        let mut beaver_count = 0;

        for op in capture.operations() {
            *operations_by_type.entry(op.op_type).or_insert(0) += 1;
            total_inputs += op.num_inputs();
            total_outputs += op.num_outputs();
            if op.beaver_data.is_some() {
                beaver_count += 1;
            }
        }

        let duration_ms = if let Some(last_op) = capture.operations().last() {
            last_op.timestamp_ns / 1_000_000
        } else {
            0
        };

        Self {
            total_operations: capture.num_operations(),
            operations_by_type,
            total_inputs,
            total_outputs,
            beaver_triples_used: beaver_count,
            total_error: capture.total_error().to_f64(),
            duration_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_witness_capture_basic() {
        let mut capture = WitnessCapture::new(0);

        capture.start_operation(WitnessedOperation::Add);
        capture.record_inputs(&[Fr::from_f64(1.0), Fr::from_f64(2.0)]);
        capture.record_output(Fr::from_f64(3.0));
        capture.record_error(Fr::from_f64(0.001));
        capture.finish_operation();

        assert_eq!(capture.num_operations(), 1);
        assert_eq!(capture.operations()[0].op_type, WitnessedOperation::Add);
        assert_eq!(capture.operations()[0].num_inputs(), 2);
        assert_eq!(capture.operations()[0].num_outputs(), 1);
    }

    #[test]
    fn test_proved_arithmetic_add() {
        let mut arith = ProvedArithmetic::new(0);

        let x = Fr::from_f64(5.0);
        let y = Fr::from_f64(3.0);
        let result = arith.add_shares(&x, &y);

        let expected = Fr::from_f64(8.0);
        assert!(result.ct_eq(&expected).to_bool());

        assert_eq!(arith.capture().num_operations(), 1);
    }

    #[test]
    fn test_proved_arithmetic_multiply() {
        let mut arith = ProvedArithmetic::new(0);

        let triple = BeaverTriple::new(
            Fr::from_f64(2.0),
            Fr::from_f64(3.0),
            Fr::from_f64(6.0),
        );

        // Simulating x=5, y=7 with Beaver protocol
        // d = x - a = 5 - 2 = 3
        // e = y - b = 7 - 3 = 4
        let opened_d = Fr::from_f64(3.0);
        let opened_e = Fr::from_f64(4.0);

        let result = arith.multiply_shares(&triple, &opened_d, &opened_e);

        // Result should be share of x*y = 35
        // For party 0: c + d*b + e*a + d*e = 6 + 3*3 + 4*2 + 3*4 = 6 + 9 + 8 + 12 = 35
        assert_eq!(arith.capture().num_operations(), 1);
        assert!(arith.capture().all_beaver_witnesses().len() == 1);
    }

    #[test]
    fn test_witness_commitment() {
        let mut capture = WitnessCapture::new(0);

        capture.start_operation(WitnessedOperation::Add);
        capture.record_inputs(&[Fr::from_f64(1.0), Fr::from_f64(2.0)]);
        capture.record_output(Fr::from_f64(3.0));
        capture.finish_operation();

        let blinding = [42u8; 32];
        let commitment1 = capture.full_commitment(&blinding);
        let commitment2 = capture.full_commitment(&blinding);

        // Same blinding should produce same commitment
        assert_eq!(commitment1, commitment2);

        // Different blinding should produce different commitment
        let blinding2 = [43u8; 32];
        let commitment3 = capture.full_commitment(&blinding2);
        assert_ne!(commitment1, commitment3);
    }

    #[test]
    fn test_witness_summary() {
        let mut capture = WitnessCapture::new(0);

        capture.start_operation(WitnessedOperation::Add);
        capture.record_inputs(&[Fr::from_f64(1.0), Fr::from_f64(2.0)]);
        capture.record_output(Fr::from_f64(3.0));
        capture.finish_operation();

        capture.start_operation(WitnessedOperation::Mul);
        capture.record_inputs(&[Fr::from_f64(3.0), Fr::from_f64(4.0)]);
        capture.record_output(Fr::from_f64(12.0));
        let beaver = BeaverWitness {
            a: Fr::from_f64(1.0),
            b: Fr::from_f64(2.0),
            c: Fr::from_f64(2.0),
            opened_d: Fr::from_f64(2.0),
            opened_e: Fr::from_f64(2.0),
            verified: true,
        };
        capture.record_beaver(beaver);
        capture.finish_operation();

        let summary = WitnessSummary::from_capture(&capture);

        assert_eq!(summary.total_operations, 2);
        assert_eq!(summary.total_inputs, 4);
        assert_eq!(summary.total_outputs, 2);
        assert_eq!(summary.beaver_triples_used, 1);
        assert_eq!(*summary.operations_by_type.get(&WitnessedOperation::Add).unwrap(), 1);
        assert_eq!(*summary.operations_by_type.get(&WitnessedOperation::Mul).unwrap(), 1);
    }

    #[test]
    fn test_forward_pass_capture() {
        let mut arith = ProvedArithmetic::new(0);

        let input = vec![Fr::from_f64(1.0), Fr::from_f64(2.0)];
        let h_pre = vec![Fr::from_f64(0.5), Fr::from_f64(0.6)];
        let h = vec![Fr::from_f64(0.5), Fr::from_f64(0.6)];
        let y = vec![Fr::from_f64(0.7)];
        let loss = Fr::from_f64(0.1);

        arith.record_forward_pass(&input, &h_pre, &h, &y, loss);

        assert_eq!(arith.capture().num_operations(), 1);
        assert_eq!(arith.capture().operations()[0].op_type, WitnessedOperation::ForwardPass);
    }

    #[test]
    fn test_capture_enable_disable() {
        let mut capture = WitnessCapture::new(0);

        capture.start_operation(WitnessedOperation::Add);
        capture.record_inputs(&[Fr::from_f64(1.0)]);
        capture.finish_operation();

        assert_eq!(capture.num_operations(), 1);

        // Disable capture
        capture.set_enabled(false);

        capture.start_operation(WitnessedOperation::Add);
        capture.record_inputs(&[Fr::from_f64(2.0)]);
        capture.finish_operation();

        // Should still be 1 (nothing captured while disabled)
        assert_eq!(capture.num_operations(), 1);

        // Re-enable capture
        capture.set_enabled(true);

        capture.start_operation(WitnessedOperation::Sub);
        capture.record_inputs(&[Fr::from_f64(3.0)]);
        capture.finish_operation();

        assert_eq!(capture.num_operations(), 2);
    }
}
