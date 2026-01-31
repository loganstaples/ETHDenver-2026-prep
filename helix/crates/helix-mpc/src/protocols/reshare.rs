//! Re-sharing protocol for periodic share refresh.
//!
//! Over many training steps, an adversary observing gradient updates could
//! potentially accumulate information to reconstruct the model weights.
//! Re-sharing periodically replaces shares with fresh random shares of
//! the same secrets, resetting any accumulated information.
//!
//! # Protocol
//!
//! To re-share [x] among n parties:
//! 1. Each party i generates n shares of zero: 0 = r_{i,1} + r_{i,2} + ... + r_{i,n}
//! 2. Each party i sends r_{i,j} to party j
//! 3. Each party j computes new share: x'_j = x_j + sum_i(r_{i,j})
//!
//! The sum of all new shares still equals x (since the r's sum to zero),
//! but the new shares are independent of the old ones.

use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;

use crate::error::MPCResult;
use crate::field::Fr;
use crate::types::PartyId;
use crate::sharing::tensor::TensorShare;
use crate::sharing::model::ModelShare;

/// Re-sharing operations for refreshing secret shares.
pub struct Resharing;

impl Resharing {
    /// Generates zero-shares for one party.
    ///
    /// Returns a vector of n field elements that sum to zero.
    /// Party i will send value j to party j.
    pub fn generate_zero_shares(n: usize, rng: &mut ChaCha20Rng) -> Vec<Fr> {
        let mut shares = Vec::with_capacity(n);
        let mut sum = Fr::ZERO;

        for _ in 0..n - 1 {
            let r = Fr::random(rng);
            shares.push(r.clone());
            sum = Fr::add(&sum, &r);
        }
        // Last share is -sum so all shares sum to zero.
        shares.push(Fr::neg(&sum));
        shares
    }

    /// Simulates the full re-sharing protocol for scalar shares.
    ///
    /// Takes all parties' current shares and produces new shares of the same value.
    pub fn reshare_scalar(old_shares: &[Fr], seed: u64) -> Vec<Fr> {
        let n = old_shares.len();
        let mut new_shares: Vec<Fr> = old_shares.to_vec();

        // Each party generates zero-shares and distributes.
        for i in 0..n {
            let party_seed = seed.wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15));
            let mut rng = ChaCha20Rng::seed_from_u64(party_seed);
            let zero_shares = Self::generate_zero_shares(n, &mut rng);

            for j in 0..n {
                new_shares[j] = Fr::add(&new_shares[j], &zero_shares[j]);
            }
        }

        new_shares
    }

    /// Simulates re-sharing for a vector.
    pub fn reshare_vector(old_shares: &[Vec<Fr>], seed: u64) -> Vec<Vec<Fr>> {
        let n = old_shares.len();
        let dim = old_shares[0].len();

        let mut new_shares: Vec<Vec<Fr>> = old_shares.to_vec();

        for i in 0..n {
            let party_seed = seed.wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15));
            let mut rng = ChaCha20Rng::seed_from_u64(party_seed);

            for d in 0..dim {
                let zero_shares = Self::generate_zero_shares(n, &mut rng);
                for j in 0..n {
                    new_shares[j][d] = Fr::add(&new_shares[j][d], &zero_shares[j]);
                }
            }
        }

        new_shares
    }

    /// Simulates re-sharing for tensor shares.
    pub fn reshare_tensor(old_shares: &[TensorShare], seed: u64) -> Vec<TensorShare> {
        let n = old_shares.len();
        let dim = old_shares[0].data.len();

        let mut new_data: Vec<Vec<Fr>> = old_shares.iter().map(|s| s.data.clone()).collect();

        for i in 0..n {
            let party_seed = seed.wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15));
            let mut rng = ChaCha20Rng::seed_from_u64(party_seed);

            for d in 0..dim {
                let zero_shares = Self::generate_zero_shares(n, &mut rng);
                for j in 0..n {
                    new_data[j][d] = Fr::add(&new_data[j][d], &zero_shares[j]);
                }
            }
        }

        old_shares
            .iter()
            .enumerate()
            .map(|(i, original)| {
                let mut ts = TensorShare::new(
                    original.id.clone(),
                    new_data[i].clone(),
                    original.shape.clone(),
                );
                ts.sharing_error = original.sharing_error;
                ts.original_error = original.original_error;
                ts
            })
            .collect()
    }

    /// Simulates re-sharing for a complete model share.
    pub fn reshare_model(
        old_shares: &[ModelShare],
        seed: u64,
    ) -> Vec<ModelShare> {
        let n = old_shares.len();
        let mut new_shares = old_shares.to_vec();

        let mut seed_counter: u64 = seed;
        let mut next_seed = || {
            seed_counter = seed_counter.wrapping_add(0xDEADBEEF);
            seed_counter
        };

        // Re-share embeddings.
        let all_have_embeddings = old_shares.iter().all(|s| s.embeddings.is_some());
        if all_have_embeddings {
            let embed_shares: Vec<TensorShare> = old_shares
                .iter()
                .map(|s| s.embeddings.clone().unwrap())
                .collect();
            let new_embed = Self::reshare_tensor(&embed_shares, next_seed());
            for (i, embed) in new_embed.into_iter().enumerate() {
                new_shares[i].embeddings = Some(embed);
            }
        }

        // Re-share each layer's weights.
        let num_layers = old_shares[0].layers.len();
        for layer_idx in 0..num_layers {
            let weight_names: Vec<String> = old_shares[0].layers[layer_idx]
                .weights
                .keys()
                .cloned()
                .collect();

            for name in weight_names {
                let weight_shares: Vec<TensorShare> = old_shares
                    .iter()
                    .map(|s| s.layers[layer_idx].weights.get(&name).unwrap().clone())
                    .collect();

                let new_weights = Self::reshare_tensor(&weight_shares, next_seed());
                for (i, w) in new_weights.into_iter().enumerate() {
                    new_shares[i].layers[layer_idx]
                        .weights
                        .insert(name.clone(), w);
                }
            }
        }

        // Re-share LM head.
        let all_have_lm = old_shares.iter().all(|s| s.lm_head.is_some());
        if all_have_lm {
            let lm_shares: Vec<TensorShare> = old_shares
                .iter()
                .map(|s| s.lm_head.clone().unwrap())
                .collect();
            let new_lm = Self::reshare_tensor(&lm_shares, next_seed());
            for (i, lm) in new_lm.into_iter().enumerate() {
                new_shares[i].lm_head = Some(lm);
            }
        }

        new_shares
    }

    /// Checks whether re-sharing should happen based on the current training step.
    pub fn should_reshare(current_step: u64, reshare_interval: u64) -> bool {
        reshare_interval > 0 && current_step > 0 && current_step % reshare_interval == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::ops::sum;

    fn test_parties(n: usize) -> Vec<PartyId> {
        (0..n).map(PartyId::from_index).collect()
    }

    #[test]
    fn test_zero_shares_sum_to_zero() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let shares = Resharing::generate_zero_shares(5, &mut rng);
        let total = sum(&shares);
        assert!(
            total.ct_eq(&Fr::ZERO).to_bool(),
            "Zero shares don't sum to zero",
        );
    }

    #[test]
    fn test_reshare_preserves_secret() {
        // Create shares that sum to a known value.
        let secret = Fr::from_f64(42.0);
        let share1 = Fr::from_f64(10.0);
        let share2 = Fr::from_f64(15.0);
        let share3 = Fr::sub(&secret, &Fr::add(&share1, &share2)); // Makes sum = 42
        let old_shares = vec![share1.clone(), share2.clone(), share3.clone()];

        let new_shares = Resharing::reshare_scalar(&old_shares, 99);
        let new_sum = sum(&new_shares);

        assert!(
            new_sum.ct_eq(&secret).to_bool(),
            "Reshare changed secret",
        );

        // New shares should be different from old shares.
        let changed = new_shares
            .iter()
            .zip(&old_shares)
            .any(|(n, o)| !n.ct_eq(o).to_bool());
        assert!(changed, "Reshare didn't change shares");
    }

    #[test]
    fn test_reshare_vector_preserves_secret() {
        // Create a simple 3-element vector shared among 3 parties.
        let secrets = vec![Fr::from_f64(1.0), Fr::from_f64(2.0), Fr::from_f64(3.0)];

        // Party 0 holds all values, others hold zeros (simple additive sharing).
        let old_shares = vec![
            secrets.clone(),
            vec![Fr::ZERO, Fr::ZERO, Fr::ZERO],
            vec![Fr::ZERO, Fr::ZERO, Fr::ZERO],
        ];

        let new_shares = Resharing::reshare_vector(&old_shares, 99);

        // Reconstruct and verify each element.
        for d in 0..3 {
            let recon = sum(&new_shares.iter().map(|s| s[d].clone()).collect::<Vec<_>>());
            assert!(
                recon.ct_eq(&secrets[d]).to_bool(),
                "Reshare vector element {} incorrect",
                d,
            );
        }
    }

    #[test]
    fn test_should_reshare() {
        assert!(!Resharing::should_reshare(0, 50));
        assert!(!Resharing::should_reshare(25, 50));
        assert!(Resharing::should_reshare(50, 50));
        assert!(Resharing::should_reshare(100, 50));
        assert!(!Resharing::should_reshare(10, 0)); // interval 0 = disabled
    }
}
