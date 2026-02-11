//! Secret sharing for complete ModelWeights and ModelGradients.
//!
//! Shares every weight tensor in a model layer-by-layer, producing
//! a `ModelShare` per party that has the same structure as the original
//! ModelWeights but contains only that party's share of each tensor.

use std::collections::HashMap;

use crate::error::{MPCError, MPCResult};
use crate::types::PartyId;
use super::AdditiveSharing;
use super::tensor::{TensorShare, TensorSharing};

use serde::{Deserialize, Serialize};

/// A single party's share of an entire model's weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelShare {
    /// Which party holds this share.
    pub party: PartyId,
    /// Share index.
    pub index: usize,
    /// Share of embedding weights.
    pub embeddings: Option<TensorShare>,
    /// Shares of layer weights, keyed by "layer_{idx}_{weight_name}".
    pub layers: Vec<LayerShare>,
    /// Share of the LM head.
    pub lm_head: Option<TensorShare>,
    /// Shares of extra/named weights.
    pub extra_weights: HashMap<String, TensorShare>,
    /// Model metadata (public, not secret-shared).
    pub model_name: String,
    /// Number of layers in the model.
    pub num_layers: usize,
}

/// A single party's share of one layer's weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerShare {
    /// Layer index.
    pub layer_idx: usize,
    /// Shares of weight tensors in this layer.
    pub weights: HashMap<String, TensorShare>,
}

/// A single party's share of a model gradient.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientShare {
    /// Which party holds this share.
    pub party: PartyId,
    /// Share index.
    pub index: usize,
    /// Share of embedding gradient.
    pub embeddings: Option<TensorShare>,
    /// Shares of layer gradients.
    pub layers: Vec<LayerGradientShare>,
    /// Share of LM head gradient.
    pub lm_head: Option<TensorShare>,
    /// Error bound.
    pub error_bound: f64,
}

/// A single party's share of one layer's gradients.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerGradientShare {
    /// Layer index.
    pub layer_idx: usize,
    /// Shares of gradient tensors.
    pub gradients: HashMap<String, TensorShare>,
}

/// Operations for sharing and reconstructing complete models.
pub struct ModelSharing;

impl ModelSharing {
    /// Shares all weights of a model using additive sharing.
    ///
    /// Each weight tensor is shared independently. The result is one `ModelShare`
    /// per party, where each ModelShare contains that party's share of every tensor.
    pub fn share_model_additive(
        embeddings: Option<(&[f32], &[usize], f64)>,
        layers: &[(usize, Vec<(&str, &[f32], &[usize], f64)>)],
        lm_head: Option<(&[f32], &[usize], f64)>,
        model_name: &str,
        parties: &[PartyId],
        sharing: &AdditiveSharing,
    ) -> MPCResult<Vec<ModelShare>> {
        let _n = parties.len();
        let num_layers = layers.len();

        // Initialize empty ModelShares.
        let mut model_shares: Vec<ModelShare> = parties
            .iter()
            .enumerate()
            .map(|(i, p)| ModelShare {
                party: p.clone(),
                index: i,
                embeddings: None,
                layers: (0..num_layers)
                    .map(|idx| LayerShare {
                        layer_idx: idx,
                        weights: HashMap::new(),
                    })
                    .collect(),
                lm_head: None,
                extra_weights: HashMap::new(),
                model_name: model_name.to_string(),
                num_layers,
            })
            .collect();

        // Share embeddings.
        if let Some((data, shape, error)) = embeddings {
            let shares = TensorSharing::share_weight_data_additive(
                data, shape, error, "embeddings", parties, sharing,
            )?;
            for (i, share) in shares.into_iter().enumerate() {
                model_shares[i].embeddings = Some(share);
            }
        }

        // Share each layer's weights.
        for (layer_idx, weight_list) in layers {
            for (weight_name, data, shape, error) in weight_list {
                let secret_id = format!("layer_{layer_idx}_{weight_name}");
                let shares = TensorSharing::share_weight_data_additive(
                    data, shape, *error, &secret_id, parties, sharing,
                )?;
                for (i, share) in shares.into_iter().enumerate() {
                    model_shares[i].layers[*layer_idx]
                        .weights
                        .insert(weight_name.to_string(), share);
                }
            }
        }

        // Share LM head.
        if let Some((data, shape, error)) = lm_head {
            let shares = TensorSharing::share_weight_data_additive(
                data, shape, error, "lm_head", parties, sharing,
            )?;
            for (i, share) in shares.into_iter().enumerate() {
                model_shares[i].lm_head = Some(share);
            }
        }

        Ok(model_shares)
    }

    /// Reconstructs model weight data from additive shares.
    ///
    /// Returns (embeddings, layers, lm_head) where each is the reconstructed data.
    pub fn reconstruct_model_additive(
        shares: &[ModelShare],
    ) -> MPCResult<ReconstructedModel> {
        if shares.is_empty() {
            return Err(MPCError::InsufficientShares {
                required: 2,
                available: 0,
            });
        }

        let num_layers = shares[0].num_layers;

        // Reconstruct embeddings.
        let embeddings = if shares[0].embeddings.is_some() {
            let embed_shares: Vec<TensorShare> = shares
                .iter()
                .filter_map(|s| s.embeddings.clone())
                .collect();
            let (data, error) =
                TensorSharing::reconstruct_weight_data_additive(&embed_shares)?;
            let shape = embed_shares[0].shape.clone();
            Some(ReconstructedWeight { data, shape, error })
        } else {
            None
        };

        // Reconstruct layers.
        let mut layers = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let mut weights = HashMap::new();

            // Collect all weight names from the first share.
            let weight_names: Vec<String> = shares[0].layers[layer_idx]
                .weights
                .keys()
                .cloned()
                .collect();

            for name in weight_names {
                let weight_shares: Vec<TensorShare> = shares
                    .iter()
                    .filter_map(|s| s.layers[layer_idx].weights.get(&name).cloned())
                    .collect();

                if !weight_shares.is_empty() {
                    let shape = weight_shares[0].shape.clone();
                    let (data, error) =
                        TensorSharing::reconstruct_weight_data_additive(&weight_shares)?;
                    weights.insert(
                        name,
                        ReconstructedWeight { data, shape, error },
                    );
                }
            }

            layers.push(ReconstructedLayer { layer_idx, weights });
        }

        // Reconstruct LM head.
        let lm_head = if shares[0].lm_head.is_some() {
            let lm_shares: Vec<TensorShare> = shares
                .iter()
                .filter_map(|s| s.lm_head.clone())
                .collect();
            let (data, error) =
                TensorSharing::reconstruct_weight_data_additive(&lm_shares)?;
            let shape = lm_shares[0].shape.clone();
            Some(ReconstructedWeight { data, shape, error })
        } else {
            None
        };

        Ok(ReconstructedModel {
            embeddings,
            layers,
            lm_head,
        })
    }

    /// Shares a gradient using additive sharing.
    pub fn share_gradient_additive(
        embeddings: Option<(&[f32], &[usize], f64)>,
        layers: &[(usize, Vec<(&str, &[f32], &[usize], f64)>)],
        lm_head: Option<(&[f32], &[usize], f64)>,
        error_bound: f64,
        parties: &[PartyId],
        sharing: &AdditiveSharing,
    ) -> MPCResult<Vec<GradientShare>> {
        let _n = parties.len();
        let num_layers = layers.len();

        let mut grad_shares: Vec<GradientShare> = parties
            .iter()
            .enumerate()
            .map(|(i, p)| GradientShare {
                party: p.clone(),
                index: i,
                embeddings: None,
                layers: (0..num_layers)
                    .map(|idx| LayerGradientShare {
                        layer_idx: idx,
                        gradients: HashMap::new(),
                    })
                    .collect(),
                lm_head: None,
                error_bound,
            })
            .collect();

        if let Some((data, shape, error)) = embeddings {
            let shares = TensorSharing::share_weight_data_additive(
                data, shape, error, "grad_embeddings", parties, sharing,
            )?;
            for (i, share) in shares.into_iter().enumerate() {
                grad_shares[i].embeddings = Some(share);
            }
        }

        for (layer_idx, grad_list) in layers {
            for (name, data, shape, error) in grad_list {
                let secret_id = format!("grad_layer_{layer_idx}_{name}");
                let shares = TensorSharing::share_weight_data_additive(
                    data, shape, *error, &secret_id, parties, sharing,
                )?;
                for (i, share) in shares.into_iter().enumerate() {
                    grad_shares[i].layers[*layer_idx]
                        .gradients
                        .insert(name.to_string(), share);
                }
            }
        }

        if let Some((data, shape, error)) = lm_head {
            let shares = TensorSharing::share_weight_data_additive(
                data, shape, error, "grad_lm_head", parties, sharing,
            )?;
            for (i, share) in shares.into_iter().enumerate() {
                grad_shares[i].lm_head = Some(share);
            }
        }

        Ok(grad_shares)
    }

    /// Reconstructs a gradient from additive shares.
    pub fn reconstruct_gradient_additive(
        shares: &[GradientShare],
    ) -> MPCResult<ReconstructedGradient> {
        if shares.is_empty() {
            return Err(MPCError::InsufficientShares {
                required: 2,
                available: 0,
            });
        }

        let num_layers = shares[0].layers.len();

        let embeddings = if shares[0].embeddings.is_some() {
            let embed_shares: Vec<TensorShare> = shares
                .iter()
                .filter_map(|s| s.embeddings.clone())
                .collect();
            let (data, error) =
                TensorSharing::reconstruct_weight_data_additive(&embed_shares)?;
            let shape = embed_shares[0].shape.clone();
            Some(ReconstructedWeight { data, shape, error })
        } else {
            None
        };

        let mut layers = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let mut gradients = HashMap::new();

            let grad_names: Vec<String> = shares[0].layers[layer_idx]
                .gradients
                .keys()
                .cloned()
                .collect();

            for name in grad_names {
                let grad_shares: Vec<TensorShare> = shares
                    .iter()
                    .filter_map(|s| s.layers[layer_idx].gradients.get(&name).cloned())
                    .collect();

                if !grad_shares.is_empty() {
                    let shape = grad_shares[0].shape.clone();
                    let (data, error) =
                        TensorSharing::reconstruct_weight_data_additive(&grad_shares)?;
                    gradients.insert(name, ReconstructedWeight { data, shape, error });
                }
            }

            layers.push(ReconstructedLayer {
                layer_idx,
                weights: gradients,
            });
        }

        let lm_head = if shares[0].lm_head.is_some() {
            let lm_shares: Vec<TensorShare> = shares
                .iter()
                .filter_map(|s| s.lm_head.clone())
                .collect();
            let (data, error) =
                TensorSharing::reconstruct_weight_data_additive(&lm_shares)?;
            let shape = lm_shares[0].shape.clone();
            Some(ReconstructedWeight { data, shape, error })
        } else {
            None
        };

        let error_bound = shares.iter().map(|s| s.error_bound).fold(0.0, f64::max);

        Ok(ReconstructedGradient {
            embeddings,
            layers,
            lm_head,
            error_bound,
        })
    }

    /// Applies a gradient share to a model share in-place.
    /// This performs: weight_share -= learning_rate * gradient_share
    /// Since both are additive shares, this is a local operation per party.
    pub fn apply_gradient_share(
        model: &mut ModelShare,
        gradient: &GradientShare,
        learning_rate: f64,
    ) -> MPCResult<()> {
        use crate::field::Fr;
        let lr_fr = Fr::from_f64(learning_rate);

        // Apply to embeddings.
        if let (Some(embed), Some(grad)) = (&mut model.embeddings, &gradient.embeddings) {
            if embed.shape != grad.shape {
                return Err(MPCError::ShapeMismatch {
                    expected: embed.shape.clone(),
                    got: grad.shape.clone(),
                });
            }
            for (w, g) in embed.data.iter_mut().zip(&grad.data) {
                // Use fixed_mul: both lr_fr and g are from_f64 encoded values.
                // fixed_mul is correct for local fixed-point × fixed-point operations.
                // (mpc_scale is only for Beaver protocol where linearity over random Fr is needed.)
                let scaled_grad = lr_fr.fixed_mul(g);
                *w = Fr::sub(w, &scaled_grad);
            }
        }

        // Apply to layers.
        for (layer, grad_layer) in model.layers.iter_mut().zip(&gradient.layers) {
            for (name, weight) in layer.weights.iter_mut() {
                if let Some(grad) = grad_layer.gradients.get(name) {
                    if weight.shape != grad.shape {
                        return Err(MPCError::ShapeMismatch {
                            expected: weight.shape.clone(),
                            got: grad.shape.clone(),
                        });
                    }
                    for (w, g) in weight.data.iter_mut().zip(&grad.data) {
                        // Use fixed_mul: both lr_fr and g are from_f64 encoded values.
                        // fixed_mul is correct for local fixed-point × fixed-point operations.
                        let scaled_grad = lr_fr.fixed_mul(g);
                        *w = Fr::sub(w, &scaled_grad);
                    }
                }
            }
        }

        // Apply to LM head.
        if let (Some(lm), Some(grad)) = (&mut model.lm_head, &gradient.lm_head) {
            if lm.shape != grad.shape {
                return Err(MPCError::ShapeMismatch {
                    expected: lm.shape.clone(),
                    got: grad.shape.clone(),
                });
            }
            for (w, g) in lm.data.iter_mut().zip(&grad.data) {
                // Use fixed_mul: both lr_fr and g are from_f64 encoded values.
                // fixed_mul is correct for local fixed-point × fixed-point operations.
                // (mpc_scale is only for Beaver protocol where linearity over random Fr is needed.)
                let scaled_grad = lr_fr.fixed_mul(g);
                *w = Fr::sub(w, &scaled_grad);
            }
        }

        Ok(())
    }
}

/// Reconstructed weight data.
#[derive(Debug, Clone)]
pub struct ReconstructedWeight {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub error: f64,
}

/// Reconstructed layer.
#[derive(Debug, Clone)]
pub struct ReconstructedLayer {
    pub layer_idx: usize,
    pub weights: HashMap<String, ReconstructedWeight>,
}

/// Reconstructed complete model.
#[derive(Debug, Clone)]
pub struct ReconstructedModel {
    pub embeddings: Option<ReconstructedWeight>,
    pub layers: Vec<ReconstructedLayer>,
    pub lm_head: Option<ReconstructedWeight>,
}

/// Reconstructed gradient.
#[derive(Debug, Clone)]
pub struct ReconstructedGradient {
    pub embeddings: Option<ReconstructedWeight>,
    pub layers: Vec<ReconstructedLayer>,
    pub lm_head: Option<ReconstructedWeight>,
    pub error_bound: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_parties(n: usize) -> Vec<PartyId> {
        (0..n).map(PartyId::from_index).collect()
    }

    #[test]
    fn test_share_and_reconstruct_model() {
        let parties = test_parties(3);
        let sharing = AdditiveSharing::with_seed(42);

        let embed_data = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let embed_shape = vec![3, 2];

        let layer0_weights: Vec<(&str, &[f32], &[usize], f64)> = vec![
            ("q_proj", &[0.1, 0.2, 0.3, 0.4], &[2, 2], 0.0),
            ("v_proj", &[0.5, 0.6, 0.7, 0.8], &[2, 2], 0.0),
        ];

        let lm_head_data = vec![0.01f32, 0.02, 0.03, 0.04, 0.05, 0.06];
        let lm_head_shape = vec![2, 3];

        let shares = ModelSharing::share_model_additive(
            Some((&embed_data, &embed_shape, 0.0)),
            &[(0, layer0_weights)],
            Some((&lm_head_data, &lm_head_shape, 0.0)),
            "test-model",
            &parties,
            &sharing,
        )
        .unwrap();

        assert_eq!(shares.len(), 3);
        assert_eq!(shares[0].num_layers, 1);
        assert!(shares[0].embeddings.is_some());
        assert!(shares[0].lm_head.is_some());
        assert_eq!(shares[0].layers[0].weights.len(), 2);

        let recon = ModelSharing::reconstruct_model_additive(&shares).unwrap();

        // Verify embeddings.
        let recon_embed = recon.embeddings.unwrap();
        for (a, b) in recon_embed.data.iter().zip(embed_data.iter()) {
            assert!((a - b).abs() < 1e-5, "Embed recon: {} vs {}", a, b);
        }

        // Verify layer weights.
        let q_proj = recon.layers[0].weights.get("q_proj").unwrap();
        assert_eq!(q_proj.shape, vec![2, 2]);
        assert!((q_proj.data[0] - 0.1).abs() < 1e-5);

        // Verify LM head.
        let recon_lm = recon.lm_head.unwrap();
        for (a, b) in recon_lm.data.iter().zip(lm_head_data.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_gradient_share_and_apply() {
        let parties = test_parties(3);
        let sharing = AdditiveSharing::with_seed(42);

        // Create model shares.
        let weight_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let weight_shape = vec![2, 2];

        let mut model_shares = ModelSharing::share_model_additive(
            None,
            &[(0, vec![("w", &weight_data, &weight_shape, 0.0)])],
            None,
            "test",
            &parties,
            &sharing,
        )
        .unwrap();

        // Create gradient shares.
        let grad_data = vec![0.1f32, 0.2, 0.3, 0.4];
        let sharing_g = AdditiveSharing::with_seed(99);

        let grad_shares = ModelSharing::share_gradient_additive(
            None,
            &[(0, vec![("w", &grad_data, &weight_shape, 0.01)])],
            None,
            0.01,
            &parties,
            &sharing_g,
        )
        .unwrap();

        // Apply gradient to each party's model share locally.
        let lr = 0.1;
        for (ms, gs) in model_shares.iter_mut().zip(&grad_shares) {
            ModelSharing::apply_gradient_share(ms, gs, lr).unwrap();
        }

        // Reconstruct updated model.
        let recon = ModelSharing::reconstruct_model_additive(&model_shares).unwrap();
        let updated_w = recon.layers[0].weights.get("w").unwrap();

        // Expected: w_new = w_old - lr * grad = [1 - 0.01, 2 - 0.02, 3 - 0.03, 4 - 0.04]
        let expected = vec![0.99f32, 1.98, 2.97, 3.96];
        for (a, b) in updated_w.data.iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "Gradient apply failed: {} vs {}",
                a,
                b,
            );
        }
    }
}
