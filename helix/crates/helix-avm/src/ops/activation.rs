//! Activation functions with error tracking.

use helix_core::types::{BoundedTensor, BoundedValue, ErrorMargin, Precision};

/// ReLU activation: max(0, x)
/// ReLU is exact (no error introduced) except at discontinuity.
pub fn relu(input: &BoundedTensor) -> BoundedTensor {
    let data: Vec<_> = input
        .data()
        .iter()
        .map(|v| {
            let val = v.value();
            if val > 0.0 {
                *v
            } else if val < 0.0 {
                BoundedValue::exact(0.0)
            } else {
                // At zero, error is the input error (could be positive or negative)
                BoundedValue::new(0.0, v.error())
            }
        })
        .collect();
    BoundedTensor::new(data, input.shape().clone())
}

/// Leaky ReLU: x if x > 0, alpha * x otherwise
pub fn leaky_relu(input: &BoundedTensor, alpha: f64) -> BoundedTensor {
    let data: Vec<_> = input
        .data()
        .iter()
        .map(|v| {
            let val = v.value();
            if val > 0.0 {
                *v
            } else {
                let new_val = alpha * val;
                let new_error = alpha * v.absolute_error();
                BoundedValue::new(new_val, ErrorMargin::absolute(new_error))
            }
        })
        .collect();
    BoundedTensor::new(data, input.shape().clone())
}

/// Sigmoid activation: 1 / (1 + exp(-x))
pub fn sigmoid(input: &BoundedTensor, precision: Precision) -> BoundedTensor {
    let precision_error = precision.max_relative_error();

    let data: Vec<_> = input
        .data()
        .iter()
        .map(|v| {
            let x = v.value();
            let sig = 1.0 / (1.0 + (-x).exp());

            // Derivative of sigmoid: sig * (1 - sig)
            // Error propagation: |d(sig)/dx| * input_error + precision_error
            let derivative = sig * (1.0 - sig);
            let error = derivative * v.absolute_error() + sig * precision_error;

            BoundedValue::new(sig, ErrorMargin::absolute(error))
        })
        .collect();
    BoundedTensor::new(data, input.shape().clone())
}

/// Tanh activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
pub fn tanh(input: &BoundedTensor, precision: Precision) -> BoundedTensor {
    let precision_error = precision.max_relative_error();

    let data: Vec<_> = input
        .data()
        .iter()
        .map(|v| {
            let x = v.value();
            let th = x.tanh();

            // Derivative: 1 - tanh^2(x)
            let derivative = 1.0 - th * th;
            let error = derivative * v.absolute_error() + th.abs() * precision_error;

            BoundedValue::new(th, ErrorMargin::absolute(error))
        })
        .collect();
    BoundedTensor::new(data, input.shape().clone())
}

/// GELU activation (approximate): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(input: &BoundedTensor, precision: Precision) -> BoundedTensor {
    let precision_error = precision.max_relative_error();
    let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();

    let data: Vec<_> = input
        .data()
        .iter()
        .map(|v| {
            let x = v.value();
            let inner = sqrt_2_over_pi * (x + 0.044715 * x.powi(3));
            let th = inner.tanh();
            let gelu_val = 0.5 * x * (1.0 + th);

            // Approximate error (GELU has complex derivative)
            let error = v.absolute_error() * 0.5 * (1.0 + th).abs()
                + gelu_val.abs() * precision_error;

            BoundedValue::new(gelu_val, ErrorMargin::absolute(error))
        })
        .collect();
    BoundedTensor::new(data, input.shape().clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let input = BoundedTensor::from_exact(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let output = relu(&input);
        let values = output.values();
        assert_eq!(values, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid() {
        let input = BoundedTensor::from_exact(vec![0.0], vec![1]);
        let output = sigmoid(&input, Precision::F32);
        let val = output.values()[0];
        assert!((val - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_tanh() {
        let input = BoundedTensor::from_exact(vec![0.0], vec![1]);
        let output = tanh(&input, Precision::F32);
        let val = output.values()[0];
        assert!(val.abs() < 1e-10);
    }
}
