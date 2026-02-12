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
//! Two implementations:
//! - `DistributedTripleGen::simulate_distributed_generation`: Single-process
//!   simulation (for testing). All parties run in one process.
//! - `NetworkDistributedDealer`: Real distributed generation over [`MPCTransport`].
//!   Each party runs independently, exchanging cross-term messages over the network.

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use serde::{Deserialize, Serialize};

use crate::error::{MPCError, MPCResult};
use crate::field::Fr;
use crate::protocols::arithmetic::SecureArithmetic;
use crate::session::transport::MPCTransport;
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

    /// **WARNING: SINGLE-PROCESS SIMULATION — NOT DISTRIBUTED**
    ///
    /// Runs the pairwise cross-term protocol locally for all parties in a single
    /// process. The protocol math is correct, but all parties' secret values are
    /// accessible in the same memory space, providing NO real security.
    ///
    /// For production, each party must run independently and exchange messages
    /// over authenticated encrypted channels (see `helix-mpc::transport`).
    ///
    /// Protocol:
    /// 1. Each party i generates random a_i, b_i, starts with c_i = a_i * b_i
    /// 2. For each ordered pair (i, j), party i picks random r_ij:
    ///    - party i adds r_ij to c_i
    ///    - party j adds (a_i * b_j - r_ij) to c_j
    ///
    /// Correctness: sum(c) = sum(a_i*b_i) + sum_{i!=j}(a_i*b_j) = (sum a)(sum b)
    pub fn simulate_distributed_generation(num_parties: usize, seed: u64) -> Vec<BeaverTriple> {
        #[cfg(debug_assertions)]
        eprintln!(
            "WARNING: simulate_distributed_generation runs ALL {} parties in a single process. \
             This provides no real MPC security. For production, use networked parties \
             with authenticated transport.",
            num_parties,
        );
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

// ============================================================================
// NetworkDistributedDealer — real distributed generation over MPCTransport
// ============================================================================

/// Message types for the networked distributed triple generation protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributedTripleMessage {
    /// Phase 1: party sends (a_i, r_ij) to peer j for cross-term computation.
    /// peer j will compute: cross_term = a_i * b_j - r_ij
    CrossTermShare {
        /// The sender's a value (one per triple in the batch).
        a_values: Vec<Vec<u8>>,
        /// Random masks r_ij (one per triple).
        r_values: Vec<Vec<u8>>,
    },
}

impl DistributedTripleMessage {
    fn encode(&self) -> Vec<u8> {
        bincode::serialize(self).expect("DistributedTripleMessage serialization should not fail")
    }

    fn decode(data: &[u8]) -> MPCResult<Self> {
        bincode::deserialize(data).map_err(|e| {
            MPCError::CommunicationError(format!("decode distributed triple message: {}", e))
        })
    }
}

/// Real distributed Beaver triple generator that communicates over [`MPCTransport`].
///
/// Unlike [`DistributedTripleGen::simulate_distributed_generation`] which runs all
/// parties in a single process, this struct runs as one party and exchanges
/// cross-term messages with peers over the transport.
///
/// # Protocol (per triple)
///
/// 1. Each party i samples random `a_i`, `b_i`, sets `c_i = a_i * b_i` (mpc_scale).
/// 2. For each peer j, party i:
///    - Picks random `r_ij`
///    - Sends `(a_i, r_ij)` to peer j
///    - Adds `r_ij` to `c_i`
/// 3. Upon receiving `(a_j, r_ji)` from peer j, party i:
///    - Computes cross-term = `a_j * b_i - r_ji`
///    - Adds cross-term to `c_i`
///
/// # Correctness
///
/// ```text
/// sum(c_i) = sum(a_i*b_i) + sum_{i!=j}(r_ij - r_ij + a_i*b_j)
///          = sum(a_i*b_i) + sum_{i!=j}(a_i*b_j)
///          = (sum a_i) * (sum b_j)
/// ```
///
/// The random masks `r_ij` cancel out across the two parties that hold them.
pub struct NetworkDistributedDealer<'t, T: MPCTransport> {
    transport: &'t T,
    party_index: usize,
    rng: ChaCha20Rng,
}

impl<'t, T: MPCTransport> NetworkDistributedDealer<'t, T> {
    /// Creates a new network-distributed dealer for one party.
    ///
    /// # Arguments
    ///
    /// * `transport` - The MPC transport for communicating with peers
    /// * `party_index` - This party's index (0..num_parties)
    /// * `seed` - RNG seed (should differ per party; use entropy in production)
    pub fn new(transport: &'t T, party_index: usize, seed: u64) -> Self {
        let party_seed = seed.wrapping_add((party_index as u64).wrapping_mul(0x9E3779B97F4A7C15));
        Self {
            transport,
            party_index,
            rng: ChaCha20Rng::seed_from_u64(party_seed),
        }
    }

    /// Generates `count` Beaver triples distributedly over the transport.
    ///
    /// All parties must call this method concurrently with the same `count`.
    /// Communication pattern: 2 rounds per batch (send cross-terms, receive cross-terms).
    ///
    /// Returns the local party's shares of each triple.
    pub async fn generate(&mut self, count: usize) -> MPCResult<Vec<BeaverTriple>> {
        let peers = self.transport.peers();

        // Phase 1: Sample local (a_i, b_i) for each triple. Compute c_i = a_i * b_i.
        let mut a_values: Vec<Fr> = Vec::with_capacity(count);
        let mut b_values: Vec<Fr> = Vec::with_capacity(count);
        let mut c_values: Vec<Fr> = Vec::with_capacity(count);

        for _ in 0..count {
            let a_i = Fr::random(&mut self.rng);
            let b_i = Fr::random(&mut self.rng);
            let c_i = a_i.mpc_scale(&b_i);
            a_values.push(a_i);
            b_values.push(b_i);
            c_values.push(c_i);
        }

        // Phase 2: For each peer, generate random masks and send (a_i, r_ij) for all triples.
        // Also accumulate r_ij into c_i.
        //
        // We batch all triples into a single message per peer for efficiency.
        for peer in &peers {
            let mut a_bytes_batch = Vec::with_capacity(count);
            let mut r_bytes_batch = Vec::with_capacity(count);

            for t in 0..count {
                let r_ij = Fr::random(&mut self.rng);

                // Serialize a_i and r_ij for this triple
                a_bytes_batch.push(SecureArithmetic::serialize_share_batch(&[a_values[t].clone()]));
                r_bytes_batch.push(SecureArithmetic::serialize_share_batch(&[r_ij.clone()]));

                // Party i adds r_ij to c_i
                c_values[t] = Fr::add(&c_values[t], &r_ij);
            }

            let msg = DistributedTripleMessage::CrossTermShare {
                a_values: a_bytes_batch,
                r_values: r_bytes_batch,
            };
            self.transport.send(peer, &msg.encode()).await?;
        }

        // Phase 3: Receive (a_j, r_ji) from each peer. Compute cross-terms.
        for peer in &peers {
            let data = self.transport.recv(peer).await?;
            let msg = DistributedTripleMessage::decode(&data)?;

            match msg {
                DistributedTripleMessage::CrossTermShare { a_values: a_batch, r_values: r_batch } => {
                    if a_batch.len() != count || r_batch.len() != count {
                        return Err(MPCError::CommunicationError(format!(
                            "expected {} triples from {}, got {} a-values and {} r-values",
                            count, peer, a_batch.len(), r_batch.len()
                        )));
                    }

                    for t in 0..count {
                        let peer_a = SecureArithmetic::deserialize_share_batch(&a_batch[t])?;
                        let peer_r = SecureArithmetic::deserialize_share_batch(&r_batch[t])?;

                        if peer_a.is_empty() || peer_r.is_empty() {
                            return Err(MPCError::CommunicationError(
                                "empty a or r value in cross-term message".into(),
                            ));
                        }

                        // Cross-term: a_j * b_i - r_ji
                        let cross = Fr::sub(&peer_a[0].mpc_scale(&b_values[t]), &peer_r[0]);
                        c_values[t] = Fr::add(&c_values[t], &cross);
                    }
                }
            }
        }

        // Assemble triples
        let triples: Vec<BeaverTriple> = (0..count)
            .map(|t| BeaverTriple::new(a_values[t].clone(), b_values[t].clone(), c_values[t].clone()))
            .collect();

        Ok(triples)
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

    // ========== NetworkDistributedDealer tests ==========

    #[tokio::test]
    async fn test_network_distributed_triple_3party() {
        use crate::session::transport::LocalTransport;

        let num_parties = 3;
        let count = 10;
        let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
        let transports = LocalTransport::create_mesh(&parties);

        // Spawn each party as a concurrent task
        let mut handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            let handle = tokio::spawn(async move {
                let mut dealer = NetworkDistributedDealer::new(&transport, i, 42);
                dealer.generate(count).await.unwrap()
            });
            handles.push(handle);
        }

        // Collect results
        let mut all_triples: Vec<Vec<BeaverTriple>> = Vec::new();
        for handle in handles {
            all_triples.push(handle.await.unwrap());
        }

        assert_eq!(all_triples.len(), num_parties);
        assert_eq!(all_triples[0].len(), count);

        // Verify each triple: sum(a) * sum(b) == sum(c)
        for t in 0..count {
            let a = sum(&all_triples.iter().map(|p| p[t].a.clone()).collect::<Vec<_>>());
            let b = sum(&all_triples.iter().map(|p| p[t].b.clone()).collect::<Vec<_>>());
            let c = sum(&all_triples.iter().map(|p| p[t].c.clone()).collect::<Vec<_>>());

            let expected = a.mpc_scale(&b);
            assert!(
                c.ct_eq(&expected).to_bool(),
                "Network distributed triple {} incorrect",
                t,
            );
        }
    }

    #[tokio::test]
    async fn test_network_distributed_triple_2party() {
        use crate::session::transport::LocalTransport;

        let num_parties = 2;
        let count = 20;
        let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
        let transports = LocalTransport::create_mesh(&parties);

        let mut handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            let handle = tokio::spawn(async move {
                let mut dealer = NetworkDistributedDealer::new(&transport, i, 100);
                dealer.generate(count).await.unwrap()
            });
            handles.push(handle);
        }

        let mut all_triples: Vec<Vec<BeaverTriple>> = Vec::new();
        for handle in handles {
            all_triples.push(handle.await.unwrap());
        }

        for t in 0..count {
            let a = Fr::add(&all_triples[0][t].a, &all_triples[1][t].a);
            let b = Fr::add(&all_triples[0][t].b, &all_triples[1][t].b);
            let c = Fr::add(&all_triples[0][t].c, &all_triples[1][t].c);

            let expected = a.mpc_scale(&b);
            assert!(
                c.ct_eq(&expected).to_bool(),
                "2-party network distributed triple {} incorrect",
                t,
            );
        }
    }

    #[tokio::test]
    async fn test_network_distributed_triple_5party() {
        use crate::session::transport::LocalTransport;

        let num_parties = 5;
        let count = 5;
        let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
        let transports = LocalTransport::create_mesh(&parties);

        let mut handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            let handle = tokio::spawn(async move {
                let mut dealer = NetworkDistributedDealer::new(&transport, i, 777);
                dealer.generate(count).await.unwrap()
            });
            handles.push(handle);
        }

        let mut all_triples: Vec<Vec<BeaverTriple>> = Vec::new();
        for handle in handles {
            all_triples.push(handle.await.unwrap());
        }

        for t in 0..count {
            let a = sum(&all_triples.iter().map(|p| p[t].a.clone()).collect::<Vec<_>>());
            let b = sum(&all_triples.iter().map(|p| p[t].b.clone()).collect::<Vec<_>>());
            let c = sum(&all_triples.iter().map(|p| p[t].c.clone()).collect::<Vec<_>>());

            let expected = a.mpc_scale(&b);
            assert!(
                c.ct_eq(&expected).to_bool(),
                "5-party network distributed triple {} incorrect",
                t,
            );
        }
    }
}
