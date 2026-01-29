//! Loss Functions with error tracking.
//!
//! Provides common loss functions used in machine learning training,
//! with proper error bound propagation.

use helix_core::types::{BoundedTensor, BoundedValue, ErrorMargin, Precision};

/// Cross-entropy loss for classification.
///
/// Computes: -sum(targets * log(predictions))
/// Predictions should be probabilities (e.g., after softmax).
pub fn cross_entropy_loss(
    predictions: &BoundedTensor,
    targets: &BoundedTensor,
    precision: Precision,
) -> BoundedValue<f64> {
    assert_eq!(predictions.len(), targets.len(), "Shape mismatch for cross entropy");
    
    let precision_error = precision.max_relative_error();
    let eps = 1e-10; // Numerical stability
    
    let mut loss = 0.0;
    let mut total_error = 0.0;
    
    for (pred, target) in predictions.data().iter().zip(targets.data().iter()) {
        let p = pred.value().max(eps); // Clamp for log stability
        let t = target.value();
        
        if t > 0.0 {
            let log_p = p.ln();
            loss -= t * log_p;
            
            // Error propagation: d/dp[-t * log(p)] = -t/p
            let error = (t / p).abs() * pred.absolute_error() + precision_error;
            total_error += error;
        }
    }
    
    BoundedValue::new(loss, ErrorMargin::absolute(total_error))
}

/// Mean Squared Error (MSE) loss.
///
/// Computes: mean((predictions - targets)^2)
pub fn mse_loss(
    predictions: &BoundedTensor,
    targets: &BoundedTensor,
    _precision: Precision,
) -> BoundedValue<f64> {
    assert_eq!(predictions.len(), targets.len(), "Shape mismatch for MSE loss");
    
    let n = predictions.len() as f64;
    let mut loss = 0.0;
    let mut total_error = 0.0;
    
    for (pred, target) in predictions.data().iter().zip(targets.data().iter()) {
        let diff = pred.value() - target.value();
        loss += diff * diff;
        
        // Error propagation: d/dp[(p-t)^2] = 2*(p-t)
        let error = 2.0 * diff.abs() * (pred.absolute_error() + target.absolute_error());
        total_error += error;
    }
    
    loss /= n;
    total_error /= n;
    
    BoundedValue::new(loss, ErrorMargin::absolute(total_error))
}

/// Mean Absolute Error (MAE) loss.
///
/// Computes: mean(|predictions - targets|)
pub fn mae_loss(
    predictions: &BoundedTensor,
    targets: &BoundedTensor,
    _precision: Precision,
) -> BoundedValue<f64> {
    assert_eq!(predictions.len(), targets.len(), "Shape mismatch for MAE loss");
    
    let n = predictions.len() as f64;
    let mut loss = 0.0;
    let mut total_error = 0.0;
    
    for (pred, target) in predictions.data().iter().zip(targets.data().iter()) {
        let diff = (pred.value() - target.value()).abs();
        loss += diff;
        
        // Error is sum of input errors
        total_error += pred.absolute_error() + target.absolute_error();
    }
    
    loss /= n;
    total_error /= n;
    
    BoundedValue::new(loss, ErrorMargin::absolute(total_error))
}

/// Huber loss (smooth L1 loss).
///
/// Quadratic for small errors, linear for large errors.
pub fn huber_loss(
    predictions: &BoundedTensor,
    targets: &BoundedTensor,
    delta: f64,
    _precision: Precision,
) -> BoundedValue<f64> {
    assert_eq!(predictions.len(), targets.len(), "Shape mismatch for Huber loss");
    
    let n = predictions.len() as f64;
    let mut loss = 0.0;
    let mut total_error = 0.0;
    
    for (pred, target) in predictions.data().iter().zip(targets.data().iter()) {
        let diff = (pred.value() - target.value()).abs();
        
        if diff <= delta {
            // Quadratic region
            loss += 0.5 * diff * diff;
            total_error += diff * (pred.absolute_error() + target.absolute_error());
        } else {
            // Linear region
            loss += delta * (diff - 0.5 * delta);
            total_error += delta * (pred.absolute_error() + target.absolute_error());
        }
    }
    
    loss /= n;
    total_error /= n;
    
    BoundedValue::new(loss, ErrorMargin::absolute(total_error))
}

/// Binary cross-entropy loss.
///
/// For binary classification with sigmoid outputs.
pub fn binary_cross_entropy_loss(
    predictions: &BoundedTensor,
    targets: &BoundedTensor,
    precision: Precision,
) -> BoundedValue<f64> {
    assert_eq!(predictions.len(), targets.len(), "Shape mismatch for BCE loss");
    
    let precision_error = precision.max_relative_error();
    let eps = 1e-10;
    let n = predictions.len() as f64;
    
    let mut loss = 0.0;
    let mut total_error = 0.0;
    
    for (pred, target) in predictions.data().iter().zip(targets.data().iter()) {
        let p = pred.value().clamp(eps, 1.0 - eps);
        let t = target.value();
        
        // BCE = -[t * log(p) + (1-t) * log(1-p)]
        loss -= t * p.ln() + (1.0 - t) * (1.0 - p).ln();
        
        // Error propagation
        let grad = -t / p + (1.0 - t) / (1.0 - p);
        total_error += grad.abs() * pred.absolute_error() + precision_error;
    }
    
    loss /= n;
    total_error /= n;
    
    BoundedValue::new(loss, ErrorMargin::absolute(total_error))
}

/// Computes the gradient of cross-entropy loss with respect to predictions.
///
/// Returns dL/dp_i for each prediction.
pub fn cross_entropy_grad(
    predictions: &BoundedTensor,
    targets: &BoundedTensor,
) -> BoundedTensor {
    let eps = 1e-10;
    
    let data: Vec<BoundedValue<f64>> = predictions
        .data()
        .iter()
        .zip(targets.data().iter())
        .map(|(pred, target)| {
            let p = pred.value().max(eps);
            let t = target.value();
            
            // d/dp[-t * log(p)] = -t/p
            let grad = -t / p;
            
            // Error in gradient
            let error = (t / (p * p)).abs() * pred.absolute_error();
            
            BoundedValue::new(grad, ErrorMargin::absolute(error))
        })
        .collect();
    
    BoundedTensor::new(data, predictions.shape().clone())
}

/// Computes the gradient of MSE loss with respect to predictions.
pub fn mse_grad(
    predictions: &BoundedTensor,
    targets: &BoundedTensor,
) -> BoundedTensor {
    let n = predictions.len() as f64;
    
    let data: Vec<BoundedValue<f64>> = predictions
        .data()
        .iter()
        .zip(targets.data().iter())
        .map(|(pred, target)| {
            let diff = pred.value() - target.value();
            let grad = 2.0 * diff / n;
            let error = 2.0 * (pred.absolute_error() + target.absolute_error()) / n;
            
            BoundedValue::new(grad, ErrorMargin::absolute(error))
        })
        .collect();
    
    BoundedTensor::new(data, predictions.shape().clone())
}

/// Softmax cross-entropy combined (numerically stable).
///
/// Takes logits and targets, computes softmax internally.
pub fn softmax_cross_entropy_loss(
    logits: &BoundedTensor,
    targets: &BoundedTensor,
    precision: Precision,
) -> BoundedValue<f64> {
    let precision_error = precision.max_relative_error();
    
    // Compute log-softmax for numerical stability
    // log_softmax(x)_i = x_i - log(sum(exp(x_j)))
    let max_logit = logits
        .data()
        .iter()
        .map(|v| v.value())
        .fold(f64::NEG_INFINITY, f64::max);
    
    let exp_sum: f64 = logits
        .data()
        .iter()
        .map(|v| (v.value() - max_logit).exp())
        .sum();
    
    let log_sum = max_logit + exp_sum.ln();
    
    let mut loss = 0.0;
    let mut total_error = 0.0;
    
    for (logit, target) in logits.data().iter().zip(targets.data().iter()) {
        let t = target.value();
        if t > 0.0 {
            let log_softmax = logit.value() - log_sum;
            loss -= t * log_softmax;
            total_error += t.abs() * (logit.absolute_error() + precision_error);
        }
    }
    
    BoundedValue::new(loss, ErrorMargin::absolute(total_error))
}

/// Gradient of softmax cross-entropy (combined for efficiency).
///
/// dL/dlogits = softmax(logits) - targets
pub fn softmax_cross_entropy_grad(
    logits: &BoundedTensor,
    targets: &BoundedTensor,
) -> BoundedTensor {
    // Compute softmax
    let max_logit = logits
        .data()
        .iter()
        .map(|v| v.value())
        .fold(f64::NEG_INFINITY, f64::max);
    
    let exps: Vec<f64> = logits
        .data()
        .iter()
        .map(|v| (v.value() - max_logit).exp())
        .collect();
    
    let exp_sum: f64 = exps.iter().sum();
    
    let data: Vec<BoundedValue<f64>> = exps
        .iter()
        .zip(targets.data().iter())
        .zip(logits.data().iter())
        .map(|((&e, target), logit)| {
            let softmax = e / exp_sum;
            let t = target.value();
            let grad = softmax - t;
            
            // Error bound for softmax gradient
            let error = logit.absolute_error() * softmax * (1.0 - softmax) + target.absolute_error();
            
            BoundedValue::new(grad, ErrorMargin::absolute(error))
        })
        .collect();
    
    BoundedTensor::new(data, logits.shape().clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss() {
        let predictions = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]);
        let targets = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]);
        
        let loss = mse_loss(&predictions, &targets, Precision::F32);
        assert!(loss.value().abs() < 1e-10);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let predictions = BoundedTensor::from_exact(vec![0.9, 0.1], vec![2]);
        let targets = BoundedTensor::from_exact(vec![1.0, 0.0], vec![2]); // One-hot
        
        let loss = cross_entropy_loss(&predictions, &targets, Precision::F32);
        assert!(loss.value() > 0.0);
        assert!(loss.value() < 0.2); // Should be small since prediction is correct
    }

    #[test]
    fn test_softmax_cross_entropy() {
        let logits = BoundedTensor::from_exact(vec![2.0, 1.0, 0.1], vec![3]);
        let targets = BoundedTensor::from_exact(vec![1.0, 0.0, 0.0], vec![3]); // Class 0
        
        let loss = softmax_cross_entropy_loss(&logits, &targets, Precision::F32);
        assert!(loss.value() > 0.0);
    }

    #[test]
    fn test_mse_grad() {
        let predictions = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]);
        let targets = BoundedTensor::from_exact(vec![0.0, 2.0, 4.0], vec![3]);
        
        let grad = mse_grad(&predictions, &targets);
        let values = grad.values();
        
        // Gradients: 2*(1-0)/3, 2*(2-2)/3, 2*(3-4)/3
        assert!((values[0] - 2.0/3.0).abs() < 1e-10);
        assert!(values[1].abs() < 1e-10);
        assert!((values[2] - (-2.0/3.0)).abs() < 1e-10);
    }
}
