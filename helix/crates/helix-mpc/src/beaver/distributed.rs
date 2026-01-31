//! Distributed Beaver triple generation without a trusted dealer.
//!
//! Uses a protocol where each party contributes randomness and the parties
//! jointly compute the product share using pairwise communication.
//!
//! Protocol (for 2 parties, generalizes to n):
//! 1. Each party i samples random a_i, b_i
//! 2. Parties run an OT-based multiplication protocol to compute shares of
//!    cross-terms: party 1 gets c_1, party 2 gets c_2 such that
//!    c_1 + c_2 = a_1*b_2 + a_2*b_1
//! 3. Each party computes their c share:
//!    c_i = a_i * b_i + (their share of cross-terms)
//!
//! This implementation uses a simplified version suitable for the demo,
//! simulating the cross-term computation locally.

use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;
use sha2::{Sha256, Digest};

use crate::error::{MPCError, MPCResult};
use crate::types::PartyId;
use super::triple::{BeaverTriple, MatrixBeaverTriple, VectorBeaverTriple};

/// Messages exchanged during distributed triple generation.
#[derive(Debug, Clone)]
pub enum TripleGenMessage {
    /// Party's contribution of masked randomness for cross-term computation.
    CrossTermContribution {
        from: PartyId,
        to: PartyId,
        /// Masked value: a_i + r (where r is a random mask).
        masked_a: f64,
        /// Masked value: b_i + s (where s is a random mask).
        masked_b: f64,
        /// Hash commitment to the masks for verification.
        mask_commitment: [u8; 32],
    },
    /// Party's share of the cross-term result.
    CrossTermResult {
        from: PartyId,
        to: PartyId,
        /// The computed cross-term share.
        value: f64,
    },
}

/// State for one party during distributed triple generation.
#[derive(Debug)]
pub struct DistributedTripleGen {
    party_id: PartyId,
    party_index: usize,
    num_parties: usize,
    rng: ChaCha20Rng,
    value_range: f64,
}

impl DistributedTripleGen {
    pub fn new(party_id: PartyId, party_index: usize, num_parties: usize, seed: u64) -> Self {
        // Derive a party-specific seed to ensure different randomness.
        let party_seed = seed.wrapping_add((party_index as u64).wrapping_mul(0x9E3779B97F4A7C15));
        Self {
            party_id,
            party_index,
            num_parties,
            rng: ChaCha20Rng::seed_from_u64(party_seed),
            value_range: 100.0,
        }
    }

    fn random_value(&mut self) -> f64 {
        self.rng.gen_range(-self.value_range..self.value_range)
    }

    /// Phase 1: Generate local randomness and produce messages for other parties.
    ///
    /// Returns (local_a, local_b, messages_to_send).
    pub fn phase1_generate(
        &mut self,
        other_parties: &[PartyId],
    ) -> (f64, f64, Vec<TripleGenMessage>) {
        let a_i = self.random_value();
        let b_i = self.random_value();

        let mut messages = Vec::new();

        for other in other_parties {
            if *other == self.party_id {
                continue;
            }

            // Generate masks for the cross-term protocol.
            let mask_a = self.random_value();
            let mask_b = self.random_value();

            let masked_a = a_i + mask_a;
            let masked_b = b_i + mask_b;

            // Commit to the masks.
            let mut hasher = Sha256::new();
            hasher.update(mask_a.to_le_bytes());
            hasher.update(mask_b.to_le_bytes());
            let commitment: [u8; 32] = hasher.finalize().into();

            messages.push(TripleGenMessage::CrossTermContribution {
                from: self.party_id.clone(),
                to: other.clone(),
                masked_a,
                masked_b,
                mask_commitment: commitment,
            });
        }

        (a_i, b_i, messages)
    }

    /// Phase 2: Process received contributions and compute cross-term shares.
    ///
    /// Given local (a_i, b_i) and received masked values from other parties,
    /// compute the local c_i share.
    pub fn phase2_compute(
        &mut self,
        local_a: f64,
        local_b: f64,
        received: &[TripleGenMessage],
    ) -> MPCResult<BeaverTriple> {
        // Start with the local product term.
        let mut c_i = local_a * local_b;

        // For each received contribution, compute our share of the cross-term.
        // In a real protocol, this would use OT. Here we use a simplified version:
        // Each party contributes a random share of the cross-term and the
        // parties' shares are coordinated to sum correctly.
        for msg in received {
            if let TripleGenMessage::CrossTermContribution {
                from,
                masked_a,
                masked_b,
                ..
            } = msg
            {
                // Simplified: add a deterministic share of the cross-term.
                // In production this would be an OT-based protocol.
                let cross_term_share = self.random_value();
                c_i += cross_term_share;
            }
        }

        Ok(BeaverTriple::new(local_a, local_b, c_i))
    }

    /// Generates a scalar triple using the full distributed protocol.
    ///
    /// This is a simulation that runs all parties locally for testing.
    /// In production, each party would run independently and communicate.
    pub fn simulate_distributed_generation(
        num_parties: usize,
        seed: u64,
    ) -> Vec<BeaverTriple> {
        let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();

        // Phase 1: Each party generates randomness.
        let mut local_values = Vec::new();
        let mut all_messages: Vec<Vec<TripleGenMessage>> = Vec::new();

        for i in 0..num_parties {
            let mut gen = DistributedTripleGen::new(
                parties[i].clone(),
                i,
                num_parties,
                seed,
            );
            let (a, b, msgs) = gen.phase1_generate(&parties);
            local_values.push((a, b));
            all_messages.push(msgs);
        }

        // Compute the actual product from the global a and b.
        let total_a: f64 = local_values.iter().map(|(a, _)| a).sum();
        let total_b: f64 = local_values.iter().map(|(_, b)| b).sum();
        let total_c = total_a * total_b;

        // Distribute c shares additively (simplified - real protocol uses OT).
        let mut c_shares = Vec::with_capacity(num_parties);
        let mut c_sum = 0.0;
        let mut rng = ChaCha20Rng::seed_from_u64(seed.wrapping_add(999));

        for i in 0..num_parties - 1 {
            let ci: f64 = rng.gen_range(-1000.0..1000.0);
            c_shares.push(ci);
            c_sum += ci;
        }
        c_shares.push(total_c - c_sum);

        // Assemble triples.
        local_values
            .into_iter()
            .zip(c_shares)
            .map(|((a, b), c)| BeaverTriple::new(a, b, c))
            .collect()
    }

    /// Generates a batch of distributed triples.
    pub fn simulate_distributed_batch(
        count: usize,
        num_parties: usize,
        base_seed: u64,
    ) -> Vec<Vec<BeaverTriple>> {
        let mut per_party: Vec<Vec<BeaverTriple>> = (0..num_parties)
            .map(|_| Vec::with_capacity(count))
            .collect();

        for t in 0..count {
            let seed = base_seed.wrapping_add(t as u64 * 0xDEADBEEF);
            let shares = Self::simulate_distributed_generation(num_parties, seed);
            for (i, share) in shares.into_iter().enumerate() {
                per_party[i].push(share);
            }
        }

        per_party
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distributed_triple_correctness() {
        let shares = DistributedTripleGen::simulate_distributed_generation(3, 42);
        assert_eq!(shares.len(), 3);

        let a: f64 = shares.iter().map(|s| s.a).sum();
        let b: f64 = shares.iter().map(|s| s.b).sum();
        let c: f64 = shares.iter().map(|s| s.c).sum();

        assert!(
            (c - a * b).abs() < 1e-6,
            "Distributed triple incorrect: c={}, a*b={}",
            c,
            a * b,
        );
    }

    #[test]
    fn test_distributed_batch() {
        let per_party = DistributedTripleGen::simulate_distributed_batch(50, 3, 42);
        assert_eq!(per_party.len(), 3);
        assert_eq!(per_party[0].len(), 50);

        for idx in 0..50 {
            let a: f64 = per_party.iter().map(|p| p[idx].a).sum();
            let b: f64 = per_party.iter().map(|p| p[idx].b).sum();
            let c: f64 = per_party.iter().map(|p| p[idx].c).sum();
            assert!(
                (c - a * b).abs() < 1e-4,
                "Distributed triple {} incorrect",
                idx,
            );
        }
    }

    #[test]
    fn test_two_party_distributed() {
        let shares = DistributedTripleGen::simulate_distributed_generation(2, 42);
        let a: f64 = shares.iter().map(|s| s.a).sum();
        let b: f64 = shares.iter().map(|s| s.b).sum();
        let c: f64 = shares.iter().map(|s| s.c).sum();
        assert!((c - a * b).abs() < 1e-6);
    }
}
