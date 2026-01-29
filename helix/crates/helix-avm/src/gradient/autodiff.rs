//! Automatic differentiation with bounds tracking.
//!
//! This module implements a tape-based (Wengert list) automatic differentiation system.
//! It records operations on `Variable`s (wrappers around `BoundedTensor`) onto a `GradientTape`.
//! During the backward pass, it computes gradients with their associated error bounds.

use helix_core::types::BoundedTensor;
use std::cell::RefCell;
use std::rc::Rc;

/// Index of a node in the computation graph.
pub type NodeIndex = usize;

/// Operations supported by the autodiff system.
///
/// Each variant stores the indices of its input nodes in the tape.
#[derive(Debug, Clone)]
pub enum Operation {
    /// External input or leaf node (no parents).
    Input,
    /// Parameter (learnable weights).
    Parameter,
    /// Element-wise addition: lhs + rhs
    Add(NodeIndex, NodeIndex),
    /// Element-wise subtraction: lhs - rhs
    Sub(NodeIndex, NodeIndex),
    /// Element-wise multiplication: lhs * rhs
    Mul(NodeIndex, NodeIndex),
    /// Element-wise division: lhs / rhs
    Div(NodeIndex, NodeIndex),
    /// Matrix multiplication: lhs @ rhs
    MatMul(NodeIndex, NodeIndex),
    /// Rectified Linear Unit: max(0, input)
    Relu(NodeIndex),
    /// Sigmoid activation: 1 / (1 + exp(-x))
    Sigmoid(NodeIndex),
    /// Softmax activation
    Softmax(NodeIndex),
    /// Sum reduction
    Sum(NodeIndex),
    /// Mean reduction
    Mean(NodeIndex),
    /// Layer Normalization (input, gamma, beta)
    LayerNorm(NodeIndex, Option<NodeIndex>, Option<NodeIndex>),
}

/// Metadata for a node in the computation graph.
#[derive(Debug, Clone)]
pub struct NodeInfo {
    /// The operation that produced this node.
    pub op: Operation,
    /// The shape of the tensor at this node (useful for checks).
    pub shape: Vec<usize>,
    /// Name mapping to trace/debug info.
    pub name: Option<String>,
    /// Cached value of the tensor (required for backward pass of non-linear ops).
    pub cached_value: Option<BoundedTensor>,
}

/// A tape that records operations for reverse-mode automatic differentiation.
#[derive(Debug, Default)]
pub struct GradientTape {
    /// List of nodes in topological order (by construction).
    pub nodes: Vec<NodeInfo>,
}

impl GradientTape {
    /// Creates a new empty gradient tape.
    pub fn new() -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self::default()))
    }

    /// Adds a node to the tape and returns its index.
    pub fn push(
        &mut self,
        op: Operation,
        shape: Vec<usize>,
        name: Option<String>,
        cached_value: Option<BoundedTensor>,
    ) -> NodeIndex {
        let index = self.nodes.len();
        self.nodes.push(NodeInfo {
            op,
            shape,
            name,
            cached_value,
        });
        index
    }
}

/// A wrapper around `BoundedTensor` that tracks computation history.
#[derive(Debug, Clone)]
pub struct Variable {
    /// The actual data with error bounds.
    pub tensor: BoundedTensor,
    /// The index of this variable's node in the tape (if tracked).
    pub node_index: Option<NodeIndex>,
    /// The tape recording the computation graph.
    pub tape: Option<Rc<RefCell<GradientTape>>>,
}

impl Variable {
    /// Creates a generic variable not attached to any graph (leaf).
    pub fn new(tensor: BoundedTensor) -> Self {
        Self {
            tensor,
            node_index: None,
            tape: None,
        }
    }

    /// Creates a variable that is tracked on a tape as an Input.
    pub fn input(tensor: BoundedTensor, tape: Rc<RefCell<GradientTape>>, name: Option<String>) -> Self {
        let node_index = tape.borrow_mut().push(
            Operation::Input,
            tensor.shape().clone(),
            name,
            Some(tensor.clone()),
        );
        Self {
            tensor,
            node_index: Some(node_index),
            tape: Some(tape),
        }
    }

    /// Creates a variable that is tracked on a tape as a Parameter (learnable).
    pub fn param(tensor: BoundedTensor, tape: Rc<RefCell<GradientTape>>, name: Option<String>) -> Self {
        let node_index = tape.borrow_mut().push(
            Operation::Parameter,
            tensor.shape().clone(),
            name,
            Some(tensor.clone()),
        );
        Self {
            tensor,
            node_index: Some(node_index),
            tape: Some(tape),
        }
    }

    /// Helper to create a new variable resulting from an operation.
    ///
    /// The resulting variable shares the same tape as `self`.
    pub fn with_op(
        tensor: BoundedTensor,
        op: Operation,
        tape: Option<Rc<RefCell<GradientTape>>>,
    ) -> Self {
        let node_index = tape.as_ref().map(|t| {
            t.borrow_mut().push(
                op,
                tensor.shape().clone(),
                None,
                Some(tensor.clone()),
            )
        });

        Self {
            tensor,
            node_index,
            tape,
        }
    }

    /// Returns the shape of the underlying tensor.
    pub fn shape(&self) -> &Vec<usize> {
        self.tensor.shape()
    }
}

// Operator overloads for Variable to make it feel like a tensor.
// We only implement a few key ones here for demonstration; 
// in a full impl we'd use macros or thorough impls.

impl Variable {
    pub fn add(&self, other: &Variable) -> Variable {
        let result = self.tensor.add(&other.tensor);
        let tape = merge_tapes(&self.tape, &other.tape);
        
        let op = if let (Some(lhs), Some(rhs), Some(_)) = (self.node_index, other.node_index, &tape) {
            Operation::Add(lhs, rhs)
        } else {
            Operation::Input // Fallback or untracked
        };

        Self::with_op(result, op, tape)
    }

    pub fn matmul(&self, other: &Variable) -> Variable {
        // Use default precision F32 for now, or make it configurable on Variable
        let precision = helix_core::types::Precision::F32;
        
        let result = crate::ops::matmul::matmul(&self.tensor, &other.tensor, precision)
            .expect("MatMul shape mismatch in Variable::matmul"); // In a real lib we'd return Result
        
        let tape = merge_tapes(&self.tape, &other.tape);
        let op = if let (Some(lhs), Some(rhs), Some(_)) = (self.node_index, other.node_index, &tape) {
            Operation::MatMul(lhs, rhs)
        } else {
            Operation::Input
        };

        Self::with_op(result, op, tape)
    }
    
    pub fn relu(&self) -> Variable {
         let result = self.tensor.clone(); // Dummy activation
         let op = if let (Some(idx), Some(_)) = (self.node_index, &self.tape) {
             Operation::Relu(idx)
         } else {
             Operation::Input
         };
         Self::with_op(result, op, self.tape.clone())
    }
}

/// Helper to merge tapes. Only works if they refer to the same tape (same Rc pointer).
/// If they are different tapes, we panic (can't mix graphs).
fn merge_tapes(t1: &Option<Rc<RefCell<GradientTape>>>, t2: &Option<Rc<RefCell<GradientTape>>>) -> Option<Rc<RefCell<GradientTape>>> {
    match (t1, t2) {
        (Some(t1), Some(t2)) => {
            assert!(Rc::ptr_eq(t1, t2), "Variables belong to different calculation graphs!");
            Some(t1.clone())
        },
        (Some(t), None) => Some(t.clone()),
        (None, Some(t)) => Some(t.clone()),
        (None, None) => None,
    }
}
