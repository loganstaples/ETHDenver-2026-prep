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
    /// 1D Convolution: Conv1d(input, kernel, stride, padding)
    Conv1d {
        input: NodeIndex,
        kernel: NodeIndex,
        stride: usize,
        padding: usize,
    },
    /// 2D Convolution: Conv2d(input, kernel, stride, padding, dilation, groups)
    Conv2d {
        input: NodeIndex,
        kernel: NodeIndex,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    },
    /// Max pooling 2D with indices for backward
    MaxPool2d {
        input: NodeIndex,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        /// Indices of max elements for backward pass
        indices: Vec<usize>,
    },
    /// Average pooling 2D
    AvgPool2d {
        input: NodeIndex,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    },
    /// Depthwise 2D convolution
    DepthwiseConv2d {
        input: NodeIndex,
        kernel: NodeIndex,
        stride: (usize, usize),
        padding: (usize, usize),
    },
    /// Embedding lookup: maps token indices to embedding vectors.
    /// `table` is the embedding table (vocab_size x embedding_dim).
    /// `indices` are the token IDs to look up.
    /// `embedding_dim` is the dimension of each embedding vector.
    Embedding {
        table: NodeIndex,
        indices: Vec<usize>,
        embedding_dim: usize,
    },
    /// GELU activation: x * sigmoid(1.702 * x) (approximate form).
    Gelu(NodeIndex),
    /// Multi-head attention: Attention(Q, K, V).
    /// Stores query, key, value node indices and the number of heads.
    Attention {
        query: NodeIndex,
        key: NodeIndex,
        value: NodeIndex,
        num_heads: usize,
    },
    /// MLP block: hidden = activation(input @ W1 + b1), output = hidden @ W2 + b2.
    /// Stores input, weights1, optional bias1, weights2, optional bias2.
    MLP {
        input: NodeIndex,
        weights1: NodeIndex,
        bias1: Option<NodeIndex>,
        weights2: NodeIndex,
        bias2: Option<NodeIndex>,
    },
    /// Grouped 2D convolution: splits input channels into groups and applies
    /// separate convolutions per group.
    GroupedConv2d {
        input: NodeIndex,
        kernel: NodeIndex,
        stride: (usize, usize),
        padding: (usize, usize),
        groups: usize,
    },
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
         let result = crate::ops::relu(&self.tensor);
         let op = if let (Some(idx), Some(_)) = (self.node_index, &self.tape) {
             Operation::Relu(idx)
         } else {
             Operation::Input
         };
         Self::with_op(result, op, self.tape.clone())
    }

    /// 1D convolution with kernel.
    pub fn conv1d(&self, kernel: &Variable, stride: usize, padding: usize) -> Variable {
        let precision = helix_core::types::Precision::F32;

        let result = crate::ops::conv1d(&self.tensor, &kernel.tensor, stride, padding, precision)
            .expect("Conv1d shape mismatch in Variable::conv1d");

        let tape = merge_tapes(&self.tape, &kernel.tape);
        let op = if let (Some(input_idx), Some(kernel_idx), Some(_)) =
            (self.node_index, kernel.node_index, &tape)
        {
            Operation::Conv1d {
                input: input_idx,
                kernel: kernel_idx,
                stride,
                padding,
            }
        } else {
            Operation::Input
        };

        Self::with_op(result, op, tape)
    }

    /// 2D convolution with kernel.
    pub fn conv2d(
        &self,
        kernel: &Variable,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Variable {
        self.conv2d_with_config(kernel, stride, padding, (1, 1), 1)
    }

    /// 2D convolution with full configuration.
    pub fn conv2d_with_config(
        &self,
        kernel: &Variable,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> Variable {
        let precision = helix_core::types::Precision::F32;

        let config = crate::ops::Conv2dConfig {
            stride,
            padding,
            dilation,
            groups,
        };

        let result = crate::ops::conv2d_with_config(&self.tensor, &kernel.tensor, config, precision)
            .expect("Conv2d shape mismatch in Variable::conv2d");

        let tape = merge_tapes(&self.tape, &kernel.tape);
        let op = if let (Some(input_idx), Some(kernel_idx), Some(_)) =
            (self.node_index, kernel.node_index, &tape)
        {
            Operation::Conv2d {
                input: input_idx,
                kernel: kernel_idx,
                stride,
                padding,
                dilation,
                groups,
            }
        } else {
            Operation::Input
        };

        Self::with_op(result, op, tape)
    }

    /// Max pooling 2D.
    pub fn max_pool2d(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Variable {
        let config = crate::ops::Pool2dConfig {
            kernel_size,
            stride,
            padding,
        };

        let pool_result = crate::ops::max_pool2d(&self.tensor, config)
            .expect("MaxPool2d failed in Variable::max_pool2d");

        let op = if let (Some(idx), Some(_)) = (self.node_index, &self.tape) {
            Operation::MaxPool2d {
                input: idx,
                kernel_size,
                stride,
                padding,
                indices: pool_result.indices.clone(),
            }
        } else {
            Operation::Input
        };

        Self::with_op(pool_result.output, op, self.tape.clone())
    }

    /// Average pooling 2D.
    pub fn avg_pool2d(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Variable {
        let precision = helix_core::types::Precision::F32;
        let config = crate::ops::Pool2dConfig {
            kernel_size,
            stride,
            padding,
        };

        let result = crate::ops::avg_pool2d(&self.tensor, config, precision)
            .expect("AvgPool2d failed in Variable::avg_pool2d");

        let op = if let (Some(idx), Some(_)) = (self.node_index, &self.tape) {
            Operation::AvgPool2d {
                input: idx,
                kernel_size,
                stride,
                padding,
            }
        } else {
            Operation::Input
        };

        Self::with_op(result, op, self.tape.clone())
    }

    /// Depthwise 2D convolution.
    pub fn depthwise_conv2d(
        &self,
        kernel: &Variable,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Variable {
        let precision = helix_core::types::Precision::F32;

        let result = crate::ops::depthwise_conv2d(&self.tensor, &kernel.tensor, stride, padding, precision)
            .expect("DepthwiseConv2d shape mismatch in Variable::depthwise_conv2d");

        let tape = merge_tapes(&self.tape, &kernel.tape);
        let op = if let (Some(input_idx), Some(kernel_idx), Some(_)) =
            (self.node_index, kernel.node_index, &tape)
        {
            Operation::DepthwiseConv2d {
                input: input_idx,
                kernel: kernel_idx,
                stride,
                padding,
            }
        } else {
            Operation::Input
        };

        Self::with_op(result, op, tape)
    }

    /// GELU activation: x * sigmoid(1.702 * x) (approximate form).
    pub fn gelu(&self) -> Variable {
        let precision = helix_core::types::Precision::F32;
        let result = crate::ops::gelu(&self.tensor, precision);
        let op = if let (Some(idx), Some(_)) = (self.node_index, &self.tape) {
            Operation::Gelu(idx)
        } else {
            Operation::Input
        };
        Self::with_op(result, op, self.tape.clone())
    }

    /// Embedding lookup from a table variable.
    ///
    /// `table` is the embedding table variable (vocab_size x embedding_dim).
    /// `indices` are the token IDs to look up.
    /// `embedding_dim` is the dimension of each embedding vector.
    pub fn embedding(table: &Variable, indices: &[usize], embedding_dim: usize) -> Variable {
        let seq_len = indices.len();
        let table_data = table.tensor.data();
        let mut result_data = Vec::with_capacity(seq_len * embedding_dim);

        for &token_id in indices {
            for j in 0..embedding_dim {
                let idx = token_id * embedding_dim + j;
                if idx < table_data.len() {
                    result_data.push(table_data[idx]);
                } else {
                    result_data.push(helix_core::types::BoundedValue::exact(0.0));
                }
            }
        }

        let result = helix_core::types::BoundedTensor::new(result_data, vec![seq_len, embedding_dim]);
        let op = if let (Some(table_idx), Some(_)) = (table.node_index, &table.tape) {
            Operation::Embedding {
                table: table_idx,
                indices: indices.to_vec(),
                embedding_dim,
            }
        } else {
            Operation::Input
        };
        Self::with_op(result, op, table.tape.clone())
    }

    /// Multi-head attention: Attention(Q, K, V).
    pub fn attention(
        query: &Variable,
        key: &Variable,
        value: &Variable,
        num_heads: usize,
    ) -> Variable {
        let precision = helix_core::types::Precision::F32;

        // Compute scaled dot-product attention
        let d_k = if query.tensor.is_matrix() {
            query.tensor.shape()[1]
        } else {
            query.tensor.shape().last().copied().unwrap_or(1)
        };
        let scale = 1.0 / (d_k as f64).sqrt();
        let scale_val = helix_core::types::BoundedValue::exact(scale);

        let key_t = key.tensor.transpose();
        let scores = crate::ops::matmul::matmul(&query.tensor, &key_t, precision)
            .expect("Attention Q@K^T shape mismatch");
        let scaled_scores = scores.scale(scale_val);

        // Row-wise softmax
        let softmax_result = crate::ops::softmax::softmax(&scaled_scores, precision)
            .unwrap_or(scaled_scores);

        let result = crate::ops::matmul::matmul(&softmax_result, &value.tensor, precision)
            .expect("Attention attn@V shape mismatch");

        let tape = merge_tapes(&query.tape, &merge_tapes(&key.tape, &value.tape));
        let op = if let (Some(q_idx), Some(k_idx), Some(v_idx), Some(_)) =
            (query.node_index, key.node_index, value.node_index, &tape)
        {
            Operation::Attention {
                query: q_idx,
                key: k_idx,
                value: v_idx,
                num_heads,
            }
        } else {
            Operation::Input
        };
        Self::with_op(result, op, tape)
    }

    /// MLP block: hidden = relu(input @ W1 + b1), output = hidden @ W2 + b2.
    pub fn mlp(
        input: &Variable,
        weights1: &Variable,
        bias1: Option<&Variable>,
        weights2: &Variable,
        bias2: Option<&Variable>,
    ) -> Variable {
        let precision = helix_core::types::Precision::F32;

        // Forward: hidden = input @ W1^T + b1
        let w1_t = weights1.tensor.transpose();
        let mut hidden = crate::ops::matmul::matmul(&input.tensor, &w1_t, precision)
            .expect("MLP input@W1^T shape mismatch");
        if let Some(b1) = bias1 {
            hidden = hidden.add(&b1.tensor);
        }

        // Activation (ReLU)
        let activated = crate::ops::relu(&hidden);

        // Output: output = activated @ W2^T + b2
        let w2_t = weights2.tensor.transpose();
        let mut output = crate::ops::matmul::matmul(&activated, &w2_t, precision)
            .expect("MLP activated@W2^T shape mismatch");
        if let Some(b2) = bias2 {
            output = output.add(&b2.tensor);
        }

        // Merge tapes from all operands
        let mut tape = merge_tapes(&input.tape, &weights1.tape);
        tape = merge_tapes(&tape, &weights2.tape);
        if let Some(b1) = bias1 {
            tape = merge_tapes(&tape, &b1.tape);
        }
        if let Some(b2) = bias2 {
            tape = merge_tapes(&tape, &b2.tape);
        }

        let op = if let (Some(input_idx), Some(w1_idx), Some(w2_idx), Some(_)) =
            (input.node_index, weights1.node_index, weights2.node_index, &tape)
        {
            Operation::MLP {
                input: input_idx,
                weights1: w1_idx,
                bias1: bias1.and_then(|b| b.node_index),
                weights2: w2_idx,
                bias2: bias2.and_then(|b| b.node_index),
            }
        } else {
            Operation::Input
        };
        Self::with_op(output, op, tape)
    }

    /// Grouped 2D convolution.
    pub fn grouped_conv2d(
        &self,
        kernel: &Variable,
        stride: (usize, usize),
        padding: (usize, usize),
        groups: usize,
    ) -> Variable {
        let precision = helix_core::types::Precision::F32;

        let config = crate::ops::Conv2dConfig {
            stride,
            padding,
            dilation: (1, 1),
            groups,
        };

        let result = crate::ops::conv2d_with_config(&self.tensor, &kernel.tensor, config, precision)
            .expect("GroupedConv2d shape mismatch in Variable::grouped_conv2d");

        let tape = merge_tapes(&self.tape, &kernel.tape);
        let op = if let (Some(input_idx), Some(kernel_idx), Some(_)) =
            (self.node_index, kernel.node_index, &tape)
        {
            Operation::GroupedConv2d {
                input: input_idx,
                kernel: kernel_idx,
                stride,
                padding,
                groups,
            }
        } else {
            Operation::Input
        };

        Self::with_op(result, op, tape)
    }

    /// Global average pooling.
    pub fn global_avg_pool2d(&self) -> Variable {
        let precision = helix_core::types::Precision::F32;

        let result = crate::ops::global_avg_pool2d(&self.tensor, precision)
            .expect("GlobalAvgPool2d failed in Variable::global_avg_pool2d");

        // Global pooling is just a special case of avg pooling
        let (h, w) = if self.tensor.ndim() == 4 {
            (self.tensor.shape()[2], self.tensor.shape()[3])
        } else {
            (self.tensor.shape()[1], self.tensor.shape()[2])
        };

        let op = if let (Some(idx), Some(_)) = (self.node_index, &self.tape) {
            Operation::AvgPool2d {
                input: idx,
                kernel_size: (h, w),
                stride: (h, w),
                padding: (0, 0),
            }
        } else {
            Operation::Input
        };

        Self::with_op(result, op, self.tape.clone())
    }
}

// ============================================================================
// Result-returning API variants (avoid panics in public-facing code)
// ============================================================================

impl Variable {
    /// Fallible matrix multiplication. Returns `Err` on shape mismatch.
    pub fn try_matmul(&self, other: &Variable) -> Result<Variable, String> {
        let precision = helix_core::types::Precision::F32;
        let result = crate::ops::matmul::matmul(&self.tensor, &other.tensor, precision)
            .map_err(|e| format!("MatMul error: {}", e))?;
        let tape = try_merge_tapes(&self.tape, &other.tape)?;
        let op = if let (Some(lhs), Some(rhs), Some(_)) = (self.node_index, other.node_index, &tape) {
            Operation::MatMul(lhs, rhs)
        } else {
            Operation::Input
        };
        Ok(Self::with_op(result, op, tape))
    }

    /// Fallible 1D convolution.
    pub fn try_conv1d(&self, kernel: &Variable, stride: usize, padding: usize) -> Result<Variable, String> {
        let precision = helix_core::types::Precision::F32;
        let result = crate::ops::conv1d(&self.tensor, &kernel.tensor, stride, padding, precision)
            .map_err(|e| format!("Conv1d error: {}", e))?;
        let tape = try_merge_tapes(&self.tape, &kernel.tape)?;
        let op = if let (Some(input_idx), Some(kernel_idx), Some(_)) =
            (self.node_index, kernel.node_index, &tape)
        {
            Operation::Conv1d { input: input_idx, kernel: kernel_idx, stride, padding }
        } else {
            Operation::Input
        };
        Ok(Self::with_op(result, op, tape))
    }

    /// Fallible 2D convolution.
    pub fn try_conv2d(
        &self,
        kernel: &Variable,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Variable, String> {
        self.try_conv2d_with_config(kernel, stride, padding, (1, 1), 1)
    }

    /// Fallible 2D convolution with full configuration.
    pub fn try_conv2d_with_config(
        &self,
        kernel: &Variable,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> Result<Variable, String> {
        let precision = helix_core::types::Precision::F32;
        let config = crate::ops::Conv2dConfig { stride, padding, dilation, groups };
        let result = crate::ops::conv2d_with_config(&self.tensor, &kernel.tensor, config, precision)
            .map_err(|e| format!("Conv2d error: {}", e))?;
        let tape = try_merge_tapes(&self.tape, &kernel.tape)?;
        let op = if let (Some(input_idx), Some(kernel_idx), Some(_)) =
            (self.node_index, kernel.node_index, &tape)
        {
            Operation::Conv2d { input: input_idx, kernel: kernel_idx, stride, padding, dilation, groups }
        } else {
            Operation::Input
        };
        Ok(Self::with_op(result, op, tape))
    }

    /// Fallible max pooling 2D.
    pub fn try_max_pool2d(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Variable, String> {
        let config = crate::ops::Pool2dConfig { kernel_size, stride, padding };
        let pool_result = crate::ops::max_pool2d(&self.tensor, config)
            .map_err(|e| format!("MaxPool2d error: {}", e))?;
        let op = if let (Some(idx), Some(_)) = (self.node_index, &self.tape) {
            Operation::MaxPool2d { input: idx, kernel_size, stride, padding, indices: pool_result.indices.clone() }
        } else {
            Operation::Input
        };
        Ok(Self::with_op(pool_result.output, op, self.tape.clone()))
    }

    /// Fallible average pooling 2D.
    pub fn try_avg_pool2d(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Variable, String> {
        let precision = helix_core::types::Precision::F32;
        let config = crate::ops::Pool2dConfig { kernel_size, stride, padding };
        let result = crate::ops::avg_pool2d(&self.tensor, config, precision)
            .map_err(|e| format!("AvgPool2d error: {}", e))?;
        let op = if let (Some(idx), Some(_)) = (self.node_index, &self.tape) {
            Operation::AvgPool2d { input: idx, kernel_size, stride, padding }
        } else {
            Operation::Input
        };
        Ok(Self::with_op(result, op, self.tape.clone()))
    }

    /// Fallible depthwise 2D convolution.
    pub fn try_depthwise_conv2d(
        &self,
        kernel: &Variable,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Variable, String> {
        let precision = helix_core::types::Precision::F32;
        let result = crate::ops::depthwise_conv2d(&self.tensor, &kernel.tensor, stride, padding, precision)
            .map_err(|e| format!("DepthwiseConv2d error: {}", e))?;
        let tape = try_merge_tapes(&self.tape, &kernel.tape)?;
        let op = if let (Some(input_idx), Some(kernel_idx), Some(_)) =
            (self.node_index, kernel.node_index, &tape)
        {
            Operation::DepthwiseConv2d { input: input_idx, kernel: kernel_idx, stride, padding }
        } else {
            Operation::Input
        };
        Ok(Self::with_op(result, op, tape))
    }

    /// Fallible global average pooling.
    pub fn try_global_avg_pool2d(&self) -> Result<Variable, String> {
        let precision = helix_core::types::Precision::F32;
        let result = crate::ops::global_avg_pool2d(&self.tensor, precision)
            .map_err(|e| format!("GlobalAvgPool2d error: {}", e))?;
        let (h, w) = if self.tensor.ndim() == 4 {
            (self.tensor.shape()[2], self.tensor.shape()[3])
        } else {
            (self.tensor.shape()[1], self.tensor.shape()[2])
        };
        let op = if let (Some(idx), Some(_)) = (self.node_index, &self.tape) {
            Operation::AvgPool2d { input: idx, kernel_size: (h, w), stride: (h, w), padding: (0, 0) }
        } else {
            Operation::Input
        };
        Ok(Self::with_op(result, op, self.tape.clone()))
    }

    /// Fallible GELU activation.
    pub fn try_gelu(&self) -> Result<Variable, String> {
        let precision = helix_core::types::Precision::F32;
        let result = crate::ops::gelu(&self.tensor, precision);
        let op = if let (Some(idx), Some(_)) = (self.node_index, &self.tape) {
            Operation::Gelu(idx)
        } else {
            Operation::Input
        };
        Ok(Self::with_op(result, op, self.tape.clone()))
    }

    /// Fallible embedding lookup.
    pub fn try_embedding(table: &Variable, indices: &[usize], embedding_dim: usize) -> Result<Variable, String> {
        let table_data = table.tensor.data();
        let vocab_size = if table.tensor.is_matrix() {
            table.tensor.shape()[0]
        } else {
            return Err("Embedding table must be a 2D tensor".to_string());
        };

        for &idx in indices {
            if idx >= vocab_size {
                return Err(format!("Token index {} out of range (vocab size: {})", idx, vocab_size));
            }
        }

        let seq_len = indices.len();
        let mut result_data = Vec::with_capacity(seq_len * embedding_dim);
        for &token_id in indices {
            for j in 0..embedding_dim {
                let flat_idx = token_id * embedding_dim + j;
                result_data.push(table_data[flat_idx]);
            }
        }

        let result = helix_core::types::BoundedTensor::new(result_data, vec![seq_len, embedding_dim]);
        let op = if let (Some(table_idx), Some(_)) = (table.node_index, &table.tape) {
            Operation::Embedding {
                table: table_idx,
                indices: indices.to_vec(),
                embedding_dim,
            }
        } else {
            Operation::Input
        };
        Ok(Self::with_op(result, op, table.tape.clone()))
    }

    /// Fallible grouped 2D convolution.
    pub fn try_grouped_conv2d(
        &self,
        kernel: &Variable,
        stride: (usize, usize),
        padding: (usize, usize),
        groups: usize,
    ) -> Result<Variable, String> {
        let precision = helix_core::types::Precision::F32;
        let config = crate::ops::Conv2dConfig {
            stride,
            padding,
            dilation: (1, 1),
            groups,
        };
        let result = crate::ops::conv2d_with_config(&self.tensor, &kernel.tensor, config, precision)
            .map_err(|e| format!("GroupedConv2d error: {}", e))?;
        let tape = try_merge_tapes(&self.tape, &kernel.tape)?;
        let op = if let (Some(input_idx), Some(kernel_idx), Some(_)) =
            (self.node_index, kernel.node_index, &tape)
        {
            Operation::GroupedConv2d { input: input_idx, kernel: kernel_idx, stride, padding, groups }
        } else {
            Operation::Input
        };
        Ok(Self::with_op(result, op, tape))
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

/// Fallible version of `merge_tapes` that returns `Err` instead of panicking.
fn try_merge_tapes(
    t1: &Option<Rc<RefCell<GradientTape>>>,
    t2: &Option<Rc<RefCell<GradientTape>>>,
) -> Result<Option<Rc<RefCell<GradientTape>>>, String> {
    match (t1, t2) {
        (Some(t1), Some(t2)) => {
            if !Rc::ptr_eq(t1, t2) {
                return Err("Variables belong to different calculation graphs".to_string());
            }
            Ok(Some(t1.clone()))
        }
        (Some(t), None) => Ok(Some(t.clone())),
        (None, Some(t)) => Ok(Some(t.clone())),
        (None, None) => Ok(None),
    }
}
