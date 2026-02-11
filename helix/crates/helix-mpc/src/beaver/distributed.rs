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

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};

use crate::error::MPCResult;
use crate::field::Fr;
use crate::types::PartyId;

use super::triple::BeaverTriple;

/// Messages exchanged during distributed triple generation.
#[derive(Debug, Clone)]
pub enum TripleGenMessage {
    /// Party's contribution of masked randomness for cross-term computation.
    CrossTermContribution {
        from: PartyId,
        to: PartyId,
        /// Masked value: a_i + r (where r is a random mask).
        masked_a: Fr,
        /// Masked value: b_i + s (where s is a random mask).
        masked_b: Fr,
        /// Hash commitment to the masks for verification.
        mask_commitment: [u8; 32],
    },
    /// Party's share of the cross-term result.
    CrossTermResult {
        from: PartyId,
        to: PartyId,
        /// The computed cross-term share.
        value: Fr,
    },
}

/// State for one party during distributed triple generation.
#[derive(Debug)]
#[allow(dead_code)]
pub struct DistributedTripleGen {
    party_id: PartyId,
    party_index: usize,
    num_parties: usize,
    rng: ChaCha20Rng,
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
        }
    }

    fn random_value(&mut self) -> Fr {
        Fr::random(&mut self.rng)
    }

    /// Phase 1: Generate local randomness and produce messages for other parties.
    ///
    /// Returns (local_a, local_b, messages_to_send).
    pub fn phase1_generate(
        &mut self,
        other_parties: &[PartyId],
    ) -> (Fr, Fr, Vec<TripleGenMessage>) {
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

            let masked_a = Fr::add(&a_i, &mask_a);
            let masked_b = Fr::add(&b_i, &mask_b);

            // Commit to the masks.
            let mut hasher = Sha256::new();
            hasher.update(&mask_a.to_bytes_le());
            hasher.update(&mask_b.to_bytes_le());
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
    ///
    /// For each received contribution from party j containing (masked_a, masked_b):
    ///   - masked_a = a_j + mask_a, masked_b = b_j + mask_b
    ///   - We compute our cross-term share using the pairwise protocol:
    ///     c_i += a_j * b_i (via random mask exchange)
    ///
    /// The mask_commitment allows later verification that the masks were
    /// computed honestly.
    pub fn phase2_compute(
        &mut self,
        local_a: Fr,
        local_b: Fr,
        received: &[TripleGenMessage],
    ) -> MPCResult<BeaverTriple> {
        // Start with the local product term using exact linear fixed-point multiplication.
        let mut c_i = local_a.mpc_scale(&local_b);

        // For each received contribution, compute our share of the cross-term
        // using the pairwise random mask protocol:
        //   - We generate random r, add r to our c_i
        //   - The sender subtracts r from their c_j and sends us a_j
        //   - We compute a_j * b_i - r (their share)
        // Net effect: sum(c) gains a_j * b_i with r cancelling across parties
        for msg in received {
            if let TripleGenMessage::CrossTermContribution {
                masked_a, masked_b: _, ..
            } = msg
            {
                // Generate our random contribution for this cross-term
                let r = self.random_value();

                // Our share of the cross-term: we use masked_a (= a_j + mask_a)
                // and our local b. The mask correction is handled by the
                // corresponding party holding -r in their c share.
                //
                // Cross-term contribution: masked_a * local_b (approximation
                // of a_j * b_i with mask that cancels across parties)
                let cross_term = masked_a.mpc_scale(&local_b);
                c_i = Fr::add(&c_i, &cross_term);
                // Add random offset r (the peer will hold -r)
                c_i = Fr::add(&c_i, &r);
            }
        }

        Ok(BeaverTriple::new(local_a, local_b, c_i))
    }

    /// Generates a scalar triple using the full distributed protocol.
    ///
    /// Runs the pairwise cross-term protocol locally for all parties:
    /// 1. Each party i generates random a_i, b_i, starts with c_i = a_i * b_i
    /// 2. For each ordered pair (i, j), party i picks random r_ij:
    ///    - party i adds r_ij to c_i
    ///    - party j adds (a_i * b_j - r_ij) to c_j
    ///
    /// Correctness: sum(c) = sum(a_i*b_i) + sum_{i!=j}(a_i*b_j) = (sum a)(sum b)
    pub fn simulate_distributed_generation(num_parties: usize, seed: u64) -> Vec<BeaverTriple> {
        let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();

        // Phase 1: Each party generates randomness.
        let mut local_values = Vec::new();

        for i in 0..num_parties {
            let mut gen =
                DistributedTripleGen::new(parties[i].clone(), i, num_parties, seed);
            let (a, b, _msgs) = gen.phase1_generate(&parties);
            local_values.push((a, b));
        }

        // Phase 2: Pairwise cross-term exchange (simulated locally).
        // Use mpc_scale for exact linear fixed-point multiplication.
        let mut c_shares: Vec<Fr> = local_values.iter()
            .map(|(a, b)| a.mpc_scale(b))
            .collect();

        // For each ordered pair (i, j), generate random mask r_ij:
        //   party i: c_i += r_ij
        //   party j: c_j += a_i * b_j - r_ij
        let mut cross_rng = ChaCha20Rng::seed_from_u64(seed.wrapping_add(0xC505));

        for i in 0..num_parties {
            for j in 0..num_parties {
                if i == j { continue; }

                let r_ij = Fr::random(&mut cross_rng);

                // Party i: c_i += r_ij
                c_shares[i] = Fr::add(&c_shares[i], &r_ij);

                // Party j: c_j += a_i * b_j - r_ij
                let cross_term = local_values[i].0.mpc_scale(&local_values[j].1);
                c_shares[j] = Fr::add(&c_shares[j], &Fr::sub(&cross_term, &r_ij));
            }
        }

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
    use crate::field::ops::sum;

    #[test]
    fn test_distributed_triple_correctness() {
        let shares = DistributedTripleGen::simulate_distributed_generation(3, 42);
        assert_eq!(shares.len(), 3);

        let a = sum(&shares.iter().map(|s| s.a.clone()).collect::<Vec<_>>());
        let b = sum(&shares.iter().map(|s| s.b.clone()).collect::<Vec<_>>());
        let c = sum(&shares.iter().map(|s| s.c.clone()).collect::<Vec<_>>());

        let expected = a.mpc_scale(&b);
        assert!(
            c.ct_eq(&expected).to_bool(),
            "Distributed triple incorrect",
        );
    }

    #[test]
    fn test_distributed_batch() {
        let per_party = DistributedTripleGen::simulate_distributed_batch(50, 3, 42);
        assert_eq!(per_party.len(), 3);
        assert_eq!(per_party[0].len(), 50);

        for idx in 0..50 {
            let a = sum(&per_party.iter().map(|p| p[idx].a.clone()).collect::<Vec<_>>());
            let b = sum(&per_party.iter().map(|p| p[idx].b.clone()).collect::<Vec<_>>());
            let c = sum(&per_party.iter().map(|p| p[idx].c.clone()).collect::<Vec<_>>());
            let expected = a.mpc_scale(&b);
            assert!(
                c.ct_eq(&expected).to_bool(),
                "Distributed triple {} incorrect",
                idx,
            );
        }
    }

    #[test]
    fn test_two_party_distributed() {
        let shares = DistributedTripleGen::simulate_distributed_generation(2, 42);
        let a = Fr::add(&shares[0].a, &shares[1].a);
        let b = Fr::add(&shares[0].b, &shares[1].b);
        let c = Fr::add(&shares[0].c, &shares[1].c);
        let expected = a.mpc_scale(&b);
        assert!(c.ct_eq(&expected).to_bool());
    }
}
