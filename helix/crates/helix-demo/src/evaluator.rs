//! MNIST accuracy evaluation in cleartext.
//!
//! After MPC training completes, weight shares are reconstructed by summing
//! across all parties (which only the model owner would do in production).
//! This module runs a plaintext forward pass to evaluate classification
//! accuracy on the held-out test set.
//!
//! Architecture: 784 -> 32 (ReLU) -> 10 (argmax)

/// Evaluates MNIST classification accuracy using reconstructed weights.
///
/// Runs a two-layer MLP forward pass in cleartext:
///   hidden = ReLU(W1 @ x + b1)
///   output = W2 @ hidden + b2
///   predicted_class = argmax(output)
///
/// # Arguments
///
/// * `w1` - Layer 1 weights, shape [d_hid, d_in] flattened row-major
/// * `b1` - Layer 1 biases, shape [d_hid]
/// * `w2` - Layer 2 weights, shape [d_out, d_hid] flattened row-major
/// * `b2` - Layer 2 biases, shape [d_out]
/// * `test_data` - (input, one_hot_target) pairs
///
/// # Returns
///
/// Accuracy as a fraction in [0.0, 1.0].
pub fn evaluate_accuracy(
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    d_in: usize,
    d_hid: usize,
    d_out: usize,
    test_data: &[(Vec<f64>, Vec<f64>)],
) -> f64 {
    if test_data.is_empty() {
        return 0.0;
    }

    let mut correct = 0usize;

    for (input, target) in test_data {
        let predicted = forward_predict(w1, b1, w2, b2, d_in, d_hid, d_out, input);
        let target_class = argmax(target);
        if predicted == target_class {
            correct += 1;
        }
    }

    correct as f64 / test_data.len() as f64
}

/// Runs a forward pass and returns the predicted class index.
fn forward_predict(
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    d_in: usize,
    d_hid: usize,
    d_out: usize,
    input: &[f64],
) -> usize {
    let output = forward_pass(w1, b1, w2, b2, d_in, d_hid, d_out, input);
    argmax(&output)
}

/// Computes the full forward pass, returning raw output logits.
///
/// hidden_pre = W1 @ x + b1
/// hidden = ReLU(hidden_pre)
/// output = W2 @ hidden + b2
fn forward_pass(
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    d_in: usize,
    d_hid: usize,
    d_out: usize,
    input: &[f64],
) -> Vec<f64> {
    // Layer 1: hidden_pre = W1 @ x + b1
    let mut hidden_pre = vec![0.0f64; d_hid];
    for i in 0..d_hid {
        let mut sum = 0.0;
        for j in 0..d_in {
            sum += w1[i * d_in + j] * input[j];
        }
        hidden_pre[i] = sum + b1[i];
    }

    // ReLU activation
    let mut hidden = vec![0.0f64; d_hid];
    for i in 0..d_hid {
        hidden[i] = hidden_pre[i].max(0.0);
    }

    // Layer 2: output = W2 @ hidden + b2
    let mut output = vec![0.0f64; d_out];
    for i in 0..d_out {
        let mut sum = 0.0;
        for j in 0..d_hid {
            sum += w2[i * d_hid + j] * hidden[j];
        }
        output[i] = sum + b2[i];
    }

    output
}

/// Returns the index of the maximum element.
fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Evaluates per-class accuracy and prints a confusion summary.
///
/// Returns (overall_accuracy, per_class_accuracies).
pub fn evaluate_per_class(
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    d_in: usize,
    d_hid: usize,
    d_out: usize,
    test_data: &[(Vec<f64>, Vec<f64>)],
) -> (f64, Vec<f64>) {
    let mut per_class_correct = vec![0usize; d_out];
    let mut per_class_total = vec![0usize; d_out];

    for (input, target) in test_data {
        let predicted = forward_predict(w1, b1, w2, b2, d_in, d_hid, d_out, input);
        let target_class = argmax(target);
        per_class_total[target_class] += 1;
        if predicted == target_class {
            per_class_correct[target_class] += 1;
        }
    }

    let overall_correct: usize = per_class_correct.iter().sum();
    let overall_total: usize = per_class_total.iter().sum();
    let overall_acc = if overall_total > 0 {
        overall_correct as f64 / overall_total as f64
    } else {
        0.0
    };

    let per_class_acc: Vec<f64> = per_class_correct
        .iter()
        .zip(per_class_total.iter())
        .map(|(&c, &t)| if t > 0 { c as f64 / t as f64 } else { 0.0 })
        .collect();

    (overall_acc, per_class_acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
        assert_eq!(argmax(&[0.5, 0.5, 0.6, 0.1]), 2);
        assert_eq!(argmax(&[1.0]), 0);
    }

    #[test]
    fn test_forward_pass_shape() {
        let d_in = 4;
        let d_hid = 3;
        let d_out = 2;

        let w1 = vec![0.1; d_hid * d_in];
        let b1 = vec![0.0; d_hid];
        let w2 = vec![0.1; d_out * d_hid];
        let b2 = vec![0.0; d_out];
        let input = vec![1.0; d_in];

        let output = forward_pass(&w1, &b1, &w2, &b2, d_in, d_hid, d_out, &input);
        assert_eq!(output.len(), d_out);
    }

    #[test]
    fn test_relu_applied() {
        // With negative bias, hidden activations should be zero (ReLU)
        let d_in = 2;
        let d_hid = 2;
        let d_out = 1;

        let w1 = vec![1.0, 0.0, 0.0, 1.0]; // identity-ish
        let b1 = vec![-10.0, -10.0]; // very negative bias forces h_pre < 0
        let w2 = vec![1.0, 1.0];
        let b2 = vec![0.0];
        let input = vec![1.0, 1.0];

        let output = forward_pass(&w1, &b1, &w2, &b2, d_in, d_hid, d_out, &input);
        // h_pre = [1-10, 1-10] = [-9, -9], ReLU -> [0, 0], output = [0]
        assert!((output[0] - 0.0).abs() < 1e-10);
    }
}
