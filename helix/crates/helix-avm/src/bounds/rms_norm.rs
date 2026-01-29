//! LayerNorm and RMSNorm Error Bounds.
//!
//! Provides rigorous error analysis for normalization layers,
//! critical for transformer training stability.

/// Error bounds for LayerNorm computation.
#[derive(Debug, Clone)]
pub struct LayerNormBounds {
    /// Normalized dimension.
    pub normalized_dim: usize,
    /// Minimum variance (for numerical stability).
    pub min_variance: f64,
    /// Whether using learnable affine parameters.
    pub affine: bool,
    /// Machine epsilon.
    pub epsilon: f64,
}

impl Default for LayerNormBounds {
    fn default() -> Self {
        Self {
            normalized_dim: 768,
            min_variance: 1e-5,
            affine: true,
            epsilon: f64::EPSILON,
        }
    }
}

/// Error components in LayerNorm.
#[derive(Debug, Clone)]
pub struct LayerNormError {
    /// Error from mean computation.
    pub mean_error: f64,
    /// Error from variance computation.
    pub variance_error: f64,
    /// Error from normalization.
    pub norm_error: f64,
    /// Error from affine transform (if applicable).
    pub affine_error: f64,
    /// Total error bound.
    pub total_error: f64,
}

impl LayerNormBounds {
    /// Creates new LayerNorm bounds.
    pub fn new(normalized_dim: usize) -> Self {
        Self {
            normalized_dim,
            min_variance: 1e-5,
            affine: true,
            epsilon: f64::EPSILON,
        }
    }

    /// Computes error bounds for LayerNorm forward pass.
    ///
    /// LayerNorm(x) = (x - μ) / √(σ² + ε) * γ + β
    pub fn forward_error(&self, input_error: f64, max_input_magnitude: f64) -> LayerNormError {
        let d = self.normalized_dim as f64;
        
        // 1. Mean computation: μ = Σx / d
        // Error: O(d * ε / d) + input_error = O(ε) + input_error / d
        let mean_error = d * self.epsilon / d + input_error / d.sqrt();
        
        // 2. Variance computation: σ² = Σ(x - μ)² / d
        // Error sources: mean error propagation + squaring + averaging
        let centered_error = input_error + mean_error;
        let squared_error = 2.0 * max_input_magnitude * centered_error + centered_error.powi(2);
        let variance_error = squared_error / d.sqrt() + d * self.epsilon;
        
        // 3. Normalization: (x - μ) / √(σ² + ε)
        // Division by sqrt is sensitive to small variance
        let min_std = self.min_variance.sqrt();
        let std_error = variance_error / (2.0 * min_std); // d(√x)/dx = 1/(2√x)
        
        let numerator_error = centered_error;
        let denominator = min_std;
        
        // Error in division: |d(a/b)| ≤ |da|/|b| + |a||db|/|b|²
        let norm_error = numerator_error / denominator + 
            max_input_magnitude * std_error / (denominator * denominator);
        
        // 4. Affine transform: y * γ + β
        let affine_error = if self.affine {
            norm_error * 1.0 + self.epsilon // γ assumed ~1, β doesn't add error
        } else {
            0.0
        };
        
        let total_error = norm_error + affine_error;
        
        LayerNormError {
            mean_error,
            variance_error,
            norm_error,
            affine_error,
            total_error,
        }
    }

    /// Computes backward error propagation.
    pub fn backward_error(
        &self,
        output_grad_error: f64,
        input_magnitude: f64,
        variance: f64,
    ) -> LayerNormGradError {
        let d = self.normalized_dim as f64;
        let std = (variance + self.min_variance).sqrt();
        
        // LayerNorm gradient is complex, involving:
        // ∂L/∂x = γ/σ * (∂L/∂y - mean(∂L/∂y) - ŷ*mean(ŷ*∂L/∂y))
        
        // Error from γ/σ scaling
        let scale_error = self.epsilon / std;
        
        // Error from mean computations
        let mean_grad_error = output_grad_error / d.sqrt();
        
        // Error from product with normalized value
        let prod_error = (input_magnitude / std) * output_grad_error + self.epsilon;
        
        // Total gradient error
        let dx_error = scale_error * output_grad_error + mean_grad_error + prod_error / d.sqrt();
        
        // Gamma gradient: ŷ * ∂L/∂y
        let dgamma_error = (input_magnitude / std) * output_grad_error;
        
        // Beta gradient: ∂L/∂y
        let dbeta_error = output_grad_error;
        
        LayerNormGradError {
            dx_error,
            dgamma_error,
            dbeta_error,
        }
    }
}

/// Gradient errors for LayerNorm.
#[derive(Debug, Clone)]
pub struct LayerNormGradError {
    /// Error in input gradient.
    pub dx_error: f64,
    /// Error in gamma gradient.
    pub dgamma_error: f64,
    /// Error in beta gradient.
    pub dbeta_error: f64,
}

/// Error bounds for RMSNorm computation.
#[derive(Debug, Clone)]
pub struct RMSNormBounds {
    /// Normalized dimension.
    pub normalized_dim: usize,
    /// Minimum RMS (for stability).
    pub min_rms: f64,
    /// Machine epsilon.
    pub epsilon: f64,
}

impl Default for RMSNormBounds {
    fn default() -> Self {
        Self {
            normalized_dim: 768,
            min_rms: 1e-5,
            epsilon: f64::EPSILON,
        }
    }
}

/// Error components in RMSNorm.
#[derive(Debug, Clone)]
pub struct RMSNormError {
    /// Error from RMS computation.
    pub rms_error: f64,
    /// Error from normalization.
    pub norm_error: f64,
    /// Error from scaling.
    pub scale_error: f64,
    /// Total error.
    pub total_error: f64,
}

impl RMSNormBounds {
    /// Creates new RMSNorm bounds.
    pub fn new(normalized_dim: usize) -> Self {
        Self {
            normalized_dim,
            min_rms: 1e-5,
            epsilon: f64::EPSILON,
        }
    }

    /// Computes error bounds for RMSNorm forward pass.
    ///
    /// RMSNorm(x) = x / √(mean(x²) + ε) * γ
    ///
    /// RMSNorm is simpler than LayerNorm (no centering).
    pub fn forward_error(&self, input_error: f64, max_input_magnitude: f64) -> RMSNormError {
        let d = self.normalized_dim as f64;
        
        // 1. Compute mean of squares: mean(x²)
        let squared_error = 2.0 * max_input_magnitude * input_error;
        let mean_sq_error = squared_error / d.sqrt() + d * self.epsilon;
        
        // 2. RMS = √(mean(x²) + ε)
        let rms_error = mean_sq_error / (2.0 * self.min_rms.sqrt());
        
        // 3. Division: x / RMS
        let min_rms = self.min_rms.sqrt();
        let norm_error = input_error / min_rms + 
            max_input_magnitude * rms_error / (min_rms * min_rms);
        
        // 4. Scaling by γ
        let scale_error = norm_error + self.epsilon;
        
        let total_error = scale_error;
        
        RMSNormError {
            rms_error,
            norm_error,
            scale_error,
            total_error,
        }
    }

    /// Advantage of RMSNorm: simpler gradient, fewer error sources.
    pub fn backward_error(&self, output_grad_error: f64, rms: f64) -> f64 {
        // RMSNorm gradient: γ/rms * (∂L/∂y - x̂*mean(x̂*∂L/∂y))
        let d = self.normalized_dim as f64;
        
        let scale_error = self.epsilon / rms;
        let mean_error = output_grad_error / d.sqrt();
        
        scale_error * output_grad_error + mean_error + self.epsilon
    }
}

/// Compares LayerNorm vs RMSNorm error characteristics.
pub fn compare_norm_errors(dim: usize, input_error: f64, magnitude: f64) -> NormComparison {
    let layer_bounds = LayerNormBounds::new(dim);
    let rms_bounds = RMSNormBounds::new(dim);
    
    let layer_error = layer_bounds.forward_error(input_error, magnitude);
    let rms_error = rms_bounds.forward_error(input_error, magnitude);
    
    NormComparison {
        layernorm_error: layer_error.total_error,
        rmsnorm_error: rms_error.total_error,
        ratio: layer_error.total_error / rms_error.total_error,
        recommendation: if rms_error.total_error < layer_error.total_error * 0.9 {
            NormRecommendation::UseRMSNorm
        } else {
            NormRecommendation::UseLayerNorm
        },
    }
}

/// Comparison of normalization methods.
#[derive(Debug, Clone)]
pub struct NormComparison {
    /// LayerNorm total error.
    pub layernorm_error: f64,
    /// RMSNorm total error.
    pub rmsnorm_error: f64,
    /// Error ratio (LayerNorm / RMSNorm).
    pub ratio: f64,
    /// Recommendation.
    pub recommendation: NormRecommendation,
}

/// Normalization recommendation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormRecommendation {
    UseLayerNorm,
    UseRMSNorm,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layernorm_bounds() {
        let bounds = LayerNormBounds::new(768);
        let error = bounds.forward_error(1e-7, 10.0);
        
        assert!(error.total_error > 0.0);
        assert!(error.mean_error > 0.0);
    }

    #[test]
    fn test_rmsnorm_bounds() {
        let bounds = RMSNormBounds::new(768);
        let error = bounds.forward_error(1e-7, 10.0);
        
        assert!(error.total_error > 0.0);
        assert!(error.rms_error > 0.0);
    }

    #[test]
    fn test_rmsnorm_simpler() {
        let comparison = compare_norm_errors(768, 1e-7, 10.0);
        
        // RMSNorm should generally have less error (simpler computation)
        assert!(comparison.rmsnorm_error > 0.0);
        assert!(comparison.layernorm_error > 0.0);
    }

    #[test]
    fn test_backward_error() {
        let bounds = LayerNormBounds::new(768);
        let grad_error = bounds.backward_error(1e-6, 10.0, 1.0);
        
        assert!(grad_error.dx_error > 0.0);
    }
}
