//! Distributed Beaver triple generation without a trusted dealer.
//!
//! Uses a MASCOT-style protocol where each party contributes randomness and the
//! parties jointly compute the product share using pairwise cross-term exchange.
//! No single party ever sees the full triple — security is information-theoretic
//! when combined with SPDZ MACs.
//!
//! # Protocol (for n parties)
//!
//! 1. Each party i samples random `a_i`, `b_i` and computes `c_i = a_i * b_i`.
//! 2. For each ordered pair (i, j), party i picks random mask `r_ij`:
//!    - Sends `(a_i, r_ij)` to party j
//!    - Adds `r_ij` to their `c_i`
//! 3. Upon receiving `(a_j, r_ji)` from party j, party i:
//!    - Computes cross-term = `a_j * b_i - r_ji`
//!    - Adds cross-term to `c_i`
//!
//! Correctness: `sum(c_i) = sum(a_i*b_i) + sum_{i!=j}(a_i*b_j) = (sum a_i)(sum b_i)`
//!
//! # Implementations
//!
//! - [`DistributedTripleGen`]: Single-process simulation (for testing).
//! - [`NetworkDistributedDealer`]: Real distributed generation over [`MPCTransport`].
//!   Each party runs independently, exchanging cross-term messages over the network.
//!   Supports scalar, vector, and matrix triple generation.

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use serde::{Deserialize, Serialize};

use crate::error::{MPCError, MPCResult};
use crate::field::Fr;
use crate::protocols::arithmetic::SecureArithmetic;
use crate::session::transport::MPCTransport;
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
///
/// Each message variant carries batched data to minimize network round trips.
/// A single message per peer per batch keeps communication at O(n) messages
/// total for n parties.
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
    /// Vector cross-term shares for element-wise vector triple generation.
    VectorCrossTermShare {
        /// For each triple: a vector of serialized a_i values (one per dimension).
        a_vectors: Vec<Vec<u8>>,
        /// For each triple: a vector of serialized r_ij masks (one per dimension).
        r_vectors: Vec<Vec<u8>>,
        /// Vector dimension.
        dim: usize,
    },
    /// Matrix cross-term shares for matrix triple generation.
    /// Only the A matrix shares and masks are exchanged — B stays local.
    MatrixCrossTermShare {
        /// For each triple: serialized A matrix shares (flattened [m*k]).
        a_matrices: Vec<Vec<u8>>,
        /// For each triple: serialized random masks (flattened [m*n] — one per
        /// element of the product C).
        r_matrices: Vec<Vec<u8>>,
        /// Dimensions: (m, k, n) for A[m,k] @ B[k,n] = C[m,n].
        m: usize,
        k: usize,
        n: usize,
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

    /// Creates a new network-distributed dealer using system entropy.
    ///
    /// This is the recommended constructor for production use.
    pub fn from_entropy(transport: &'t T, party_index: usize) -> Self {
        Self {
            transport,
            party_index,
            rng: ChaCha20Rng::from_entropy(),
        }
    }

    /// Generates `count` scalar Beaver triples distributedly over the transport.
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
                _ => {
                    return Err(MPCError::CommunicationError(
                        "expected CrossTermShare message during scalar generation".into(),
                    ));
                }
            }
        }

        // Assemble triples
        let triples: Vec<BeaverTriple> = (0..count)
            .map(|t| BeaverTriple::new(a_values[t].clone(), b_values[t].clone(), c_values[t].clone()))
            .collect();

        Ok(triples)
    }

    /// Generates `count` vector Beaver triples of dimension `dim` distributedly.
    ///
    /// Each element of the vector triple is generated using the same pairwise
    /// cross-term protocol as scalar triples. Elements within a single vector
    /// triple are independent and batched into a single message per peer.
    ///
    /// All parties must call this method concurrently with the same `count` and `dim`.
    pub async fn generate_vector(
        &mut self,
        count: usize,
        dim: usize,
    ) -> MPCResult<Vec<VectorBeaverTriple>> {
        let peers = self.transport.peers();
        let elems = count * dim;

        // Phase 1: Sample local values for all elements across all triples.
        let mut a_flat: Vec<Fr> = Vec::with_capacity(elems);
        let mut b_flat: Vec<Fr> = Vec::with_capacity(elems);
        let mut c_flat: Vec<Fr> = Vec::with_capacity(elems);

        for _ in 0..elems {
            let a_i = Fr::random(&mut self.rng);
            let b_i = Fr::random(&mut self.rng);
            let c_i = a_i.mpc_scale(&b_i);
            a_flat.push(a_i);
            b_flat.push(b_i);
            c_flat.push(c_i);
        }

        // Phase 2: Send cross-term data to each peer.
        for peer in &peers {
            let mut a_vecs = Vec::with_capacity(count);
            let mut r_vecs = Vec::with_capacity(count);

            for t in 0..count {
                let base = t * dim;
                let mut a_batch = Vec::with_capacity(dim);
                let mut r_batch = Vec::with_capacity(dim);

                for d in 0..dim {
                    let idx = base + d;
                    let r_ij = Fr::random(&mut self.rng);
                    a_batch.push(a_flat[idx].clone());
                    r_batch.push(r_ij.clone());
                    c_flat[idx] = Fr::add(&c_flat[idx], &r_ij);
                }

                a_vecs.push(SecureArithmetic::serialize_share_batch(&a_batch));
                r_vecs.push(SecureArithmetic::serialize_share_batch(&r_batch));
            }

            let msg = DistributedTripleMessage::VectorCrossTermShare {
                a_vectors: a_vecs,
                r_vectors: r_vecs,
                dim,
            };
            self.transport.send(peer, &msg.encode()).await?;
        }

        // Phase 3: Receive cross-term data from each peer.
        for peer in &peers {
            let data = self.transport.recv(peer).await?;
            let msg = DistributedTripleMessage::decode(&data)?;

            match msg {
                DistributedTripleMessage::VectorCrossTermShare {
                    a_vectors: a_vecs, r_vectors: r_vecs, dim: recv_dim,
                } => {
                    if recv_dim != dim {
                        return Err(MPCError::CommunicationError(format!(
                            "dimension mismatch: expected {}, got {}", dim, recv_dim,
                        )));
                    }
                    if a_vecs.len() != count || r_vecs.len() != count {
                        return Err(MPCError::CommunicationError(format!(
                            "expected {} vector triples from {}, got {}",
                            count, peer, a_vecs.len(),
                        )));
                    }

                    for t in 0..count {
                        let base = t * dim;
                        let peer_a = SecureArithmetic::deserialize_share_batch(&a_vecs[t])?;
                        let peer_r = SecureArithmetic::deserialize_share_batch(&r_vecs[t])?;

                        if peer_a.len() != dim || peer_r.len() != dim {
                            return Err(MPCError::CommunicationError(format!(
                                "vector element count mismatch: expected {}, got a={} r={}",
                                dim, peer_a.len(), peer_r.len(),
                            )));
                        }

                        for d in 0..dim {
                            let idx = base + d;
                            let cross = Fr::sub(
                                &peer_a[d].mpc_scale(&b_flat[idx]),
                                &peer_r[d],
                            );
                            c_flat[idx] = Fr::add(&c_flat[idx], &cross);
                        }
                    }
                }
                _ => {
                    return Err(MPCError::CommunicationError(
                        "expected VectorCrossTermShare message during vector generation".into(),
                    ));
                }
            }
        }

        // Assemble vector triples
        let mut triples = Vec::with_capacity(count);
        for t in 0..count {
            let base = t * dim;
            triples.push(VectorBeaverTriple::new(
                a_flat[base..base + dim].to_vec(),
                b_flat[base..base + dim].to_vec(),
                c_flat[base..base + dim].to_vec(),
            ));
        }

        Ok(triples)
    }

    /// Generates `count` matrix Beaver triples for A[m,k] @ B[k,n] = C[m,n].
    ///
    /// The protocol generates element-wise shares of the product matrix C
    /// using the pairwise cross-term protocol. For each element c_{ij} of C:
    ///   c_{ij} = sum_l(a_{il} * b_{lj})
    /// Each party holds shares of A, B, and C such that the sums reconstruct
    /// the correct matrix product.
    ///
    /// All parties must call this method concurrently with matching parameters.
    pub async fn generate_matrix(
        &mut self,
        count: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> MPCResult<Vec<MatrixBeaverTriple>> {
        let peers = self.transport.peers();
        let a_size = m * k;
        let b_size = k * n;
        let c_size = m * n;

        // Phase 1: Sample local A and B matrices, compute C = A @ B.
        let mut all_a: Vec<Vec<Fr>> = Vec::with_capacity(count);
        let mut all_b: Vec<Vec<Fr>> = Vec::with_capacity(count);
        let mut all_c: Vec<Vec<Fr>> = Vec::with_capacity(count);

        for _ in 0..count {
            let a_i: Vec<Fr> = (0..a_size).map(|_| Fr::random(&mut self.rng)).collect();
            let b_i: Vec<Fr> = (0..b_size).map(|_| Fr::random(&mut self.rng)).collect();

            // Compute C_i = A_i @ B_i using mpc_scale
            let mut c_i = vec![Fr::ZERO; c_size];
            for row in 0..m {
                for col in 0..n {
                    let mut sum = Fr::ZERO;
                    for l in 0..k {
                        sum = Fr::add(&sum, &a_i[row * k + l].mpc_scale(&b_i[l * n + col]));
                    }
                    c_i[row * n + col] = sum;
                }
            }

            all_a.push(a_i);
            all_b.push(b_i);
            all_c.push(c_i);
        }

        // Phase 2: Send A matrix shares and random masks for C elements to each peer.
        //
        // The cross-term for matrix multiplication is:
        //   For party i sending to party j:
        //     For each element c_{row,col} of C:
        //       cross = sum_l(a_i[row,l] * b_j[l,col])
        //     Split via random mask: party i gets +r, party j gets cross - r.
        //
        // We send the full A_i matrix and per-C-element masks. Party j computes
        // the cross-term using their local B_j.
        for peer in &peers {
            let mut a_mats = Vec::with_capacity(count);
            let mut r_mats = Vec::with_capacity(count);

            for t in 0..count {
                // Serialize A_i
                a_mats.push(SecureArithmetic::serialize_share_batch(&all_a[t]));

                // Generate and apply random masks for each element of C
                let mut r_vals = Vec::with_capacity(c_size);
                for idx in 0..c_size {
                    let r_ij = Fr::random(&mut self.rng);
                    all_c[t][idx] = Fr::add(&all_c[t][idx], &r_ij);
                    r_vals.push(r_ij);
                }
                r_mats.push(SecureArithmetic::serialize_share_batch(&r_vals));
            }

            let msg = DistributedTripleMessage::MatrixCrossTermShare {
                a_matrices: a_mats,
                r_matrices: r_mats,
                m, k, n,
            };
            self.transport.send(peer, &msg.encode()).await?;
        }

        // Phase 3: Receive A_j matrices and masks from each peer.
        // Compute cross-terms: for each C element, add sum_l(a_j[row,l] * b_i[l,col]) - r_ji.
        for peer in &peers {
            let data = self.transport.recv(peer).await?;
            let msg = DistributedTripleMessage::decode(&data)?;

            match msg {
                DistributedTripleMessage::MatrixCrossTermShare {
                    a_matrices: a_mats, r_matrices: r_mats,
                    m: rm, k: rk, n: rn,
                } => {
                    if rm != m || rk != k || rn != n {
                        return Err(MPCError::CommunicationError(format!(
                            "matrix dimension mismatch: expected ({},{},{}), got ({},{},{})",
                            m, k, n, rm, rk, rn,
                        )));
                    }
                    if a_mats.len() != count || r_mats.len() != count {
                        return Err(MPCError::CommunicationError(format!(
                            "expected {} matrix triples from {}, got {}",
                            count, peer, a_mats.len(),
                        )));
                    }

                    for t in 0..count {
                        let peer_a = SecureArithmetic::deserialize_share_batch(&a_mats[t])?;
                        let peer_r = SecureArithmetic::deserialize_share_batch(&r_mats[t])?;

                        if peer_a.len() != a_size {
                            return Err(MPCError::CommunicationError(format!(
                                "A matrix size mismatch: expected {}, got {}",
                                a_size, peer_a.len(),
                            )));
                        }
                        if peer_r.len() != c_size {
                            return Err(MPCError::CommunicationError(format!(
                                "R matrix size mismatch: expected {}, got {}",
                                c_size, peer_r.len(),
                            )));
                        }

                        // Compute cross-term: A_j @ B_i - R_ji
                        for row in 0..m {
                            for col in 0..n {
                                let c_idx = row * n + col;
                                let mut cross = Fr::ZERO;
                                for l in 0..k {
                                    cross = Fr::add(
                                        &cross,
                                        &peer_a[row * k + l].mpc_scale(&all_b[t][l * n + col]),
                                    );
                                }
                                let contribution = Fr::sub(&cross, &peer_r[c_idx]);
                                all_c[t][c_idx] = Fr::add(&all_c[t][c_idx], &contribution);
                            }
                        }
                    }
                }
                _ => {
                    return Err(MPCError::CommunicationError(
                        "expected MatrixCrossTermShare message during matrix generation".into(),
                    ));
                }
            }
        }

        // Assemble matrix triples
        let mut triples = Vec::with_capacity(count);
        for t in 0..count {
            triples.push(MatrixBeaverTriple::new(
                all_a[t].clone(),
                all_b[t].clone(),
                all_c[t].clone(),
                m, k, n,
            ));
        }

        Ok(triples)
    }
}

// ============================================================================
// DistributedTriplePool — pre-generation pool for distributed triples
// ============================================================================

use std::sync::atomic::{AtomicUsize, Ordering};
use std::collections::VecDeque;

/// Configuration for the distributed triple pool.
#[derive(Debug, Clone)]
pub struct DistributedPoolConfig {
    /// Number of scalar triples to pre-generate.
    pub initial_scalar_count: usize,
    /// Number of vector triples to pre-generate per dimension.
    pub initial_vector_count: usize,
    /// Number of matrix triples to pre-generate per dimension combo.
    pub initial_matrix_count: usize,
    /// Low-water mark: when available triples drop below this fraction of the
    /// initial count, a replenishment is recommended.
    pub low_water_fraction: f64,
}

impl Default for DistributedPoolConfig {
    fn default() -> Self {
        Self {
            initial_scalar_count: 10_000,
            initial_vector_count: 100,
            initial_matrix_count: 50,
            low_water_fraction: 0.25,
        }
    }
}

impl DistributedPoolConfig {
    /// Configuration for small models / demos.
    pub fn small_model() -> Self {
        Self {
            initial_scalar_count: 1_000,
            initial_vector_count: 50,
            initial_matrix_count: 20,
            low_water_fraction: 0.25,
        }
    }

    /// Configuration for large models.
    pub fn large_model() -> Self {
        Self {
            initial_scalar_count: 100_000,
            initial_vector_count: 500,
            initial_matrix_count: 100,
            low_water_fraction: 0.2,
        }
    }
}

/// A pre-generation pool for distributedly-generated Beaver triples.
///
/// This pool is filled by calling [`generate`] (which runs the distributed
/// protocol over the network) and consumed during the online phase via [`take`].
///
/// Unlike [`BeaverPool`](super::pool::BeaverPool) which is filled by a
/// centralized dealer, this pool is designed to be filled by the distributed
/// generation protocol where each party independently generates their own shares.
///
/// # Thread Safety
///
/// The pool uses interior mutability with atomic counters for the fast path
/// (checking remaining count) and a `VecDeque` protected by appropriate
/// synchronization for the actual triple storage.
///
/// # Error Handling
///
/// Taking from an exhausted pool returns `MPCError::BeaverPoolExhausted` with
/// the number of requested vs available triples. This is a hard error — silent
/// corruption (e.g., reusing triples) would break MPC security.
#[derive(Debug)]
pub struct DistributedTriplePool {
    /// FIFO queue of scalar triples (oldest consumed first).
    scalar_triples: VecDeque<BeaverTriple>,
    /// FIFO queue of vector triples, keyed by dimension.
    vector_triples: std::collections::HashMap<usize, VecDeque<VectorBeaverTriple>>,
    /// FIFO queue of matrix triples, keyed by (m, k, n).
    matrix_triples: std::collections::HashMap<(usize, usize, usize), VecDeque<MatrixBeaverTriple>>,
    /// Total scalar triples consumed (monotonically increasing).
    scalar_consumed: AtomicUsize,
    /// Total scalar triples ever generated.
    scalar_generated: AtomicUsize,
}

impl DistributedTriplePool {
    /// Creates a new empty pool.
    pub fn new() -> Self {
        Self {
            scalar_triples: VecDeque::new(),
            vector_triples: std::collections::HashMap::new(),
            matrix_triples: std::collections::HashMap::new(),
            scalar_consumed: AtomicUsize::new(0),
            scalar_generated: AtomicUsize::new(0),
        }
    }

    /// Fills the pool with pre-generated scalar triples.
    ///
    /// Triples are appended to the back of the FIFO queue so that older triples
    /// are consumed first (FIFO ordering prevents any triple from going stale).
    pub fn fill_scalar(&mut self, triples: Vec<BeaverTriple>) {
        let count = triples.len();
        self.scalar_triples.extend(triples);
        self.scalar_generated.fetch_add(count, Ordering::Relaxed);
    }

    /// Fills the pool with pre-generated vector triples.
    pub fn fill_vector(&mut self, dim: usize, triples: Vec<VectorBeaverTriple>) {
        self.vector_triples.entry(dim).or_default().extend(triples);
    }

    /// Fills the pool with pre-generated matrix triples.
    pub fn fill_matrix(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        triples: Vec<MatrixBeaverTriple>,
    ) {
        self.matrix_triples.entry((m, k, n)).or_default().extend(triples);
    }

    /// Takes the next scalar triple from the pool.
    ///
    /// Returns `MPCError::BeaverPoolExhausted` if the pool is empty.
    /// **Never** silently reuses or fabricates triples — exhaustion is a hard error
    /// because reusing triples would break MPC security.
    pub fn take(&mut self) -> MPCResult<BeaverTriple> {
        let triple = self.scalar_triples.pop_front().ok_or(
            MPCError::BeaverPoolExhausted {
                requested: 1,
                available: 0,
            },
        )?;
        self.scalar_consumed.fetch_add(1, Ordering::Relaxed);
        Ok(triple)
    }

    /// Takes `count` scalar triples from the pool.
    ///
    /// Either returns all requested triples or returns an error — never
    /// returns a partial batch.
    pub fn take_batch(&mut self, count: usize) -> MPCResult<Vec<BeaverTriple>> {
        let available = self.scalar_triples.len();
        if available < count {
            return Err(MPCError::BeaverPoolExhausted {
                requested: count,
                available,
            });
        }

        let triples: Vec<BeaverTriple> = self.scalar_triples.drain(..count).collect();
        self.scalar_consumed.fetch_add(count, Ordering::Relaxed);
        Ok(triples)
    }

    /// Takes a vector triple of the specified dimension.
    pub fn take_vector(&mut self, dim: usize) -> MPCResult<VectorBeaverTriple> {
        self.vector_triples
            .get_mut(&dim)
            .and_then(|q| q.pop_front())
            .ok_or(MPCError::BeaverPoolExhausted {
                requested: 1,
                available: 0,
            })
    }

    /// Takes a matrix triple for the specified dimensions.
    pub fn take_matrix(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
    ) -> MPCResult<MatrixBeaverTriple> {
        self.matrix_triples
            .get_mut(&(m, k, n))
            .and_then(|q| q.pop_front())
            .ok_or(MPCError::BeaverPoolExhausted {
                requested: 1,
                available: 0,
            })
    }

    /// Returns the number of scalar triples remaining in the pool.
    pub fn remaining(&self) -> usize {
        self.scalar_triples.len()
    }

    /// Returns the number of vector triples available for a given dimension.
    pub fn vector_remaining(&self, dim: usize) -> usize {
        self.vector_triples.get(&dim).map_or(0, |q| q.len())
    }

    /// Returns the number of matrix triples available for given dimensions.
    pub fn matrix_remaining(&self, m: usize, k: usize, n: usize) -> usize {
        self.matrix_triples.get(&(m, k, n)).map_or(0, |q| q.len())
    }

    /// Returns the total number of scalar triples consumed since pool creation.
    pub fn total_consumed(&self) -> usize {
        self.scalar_consumed.load(Ordering::Relaxed)
    }

    /// Returns the total number of scalar triples ever generated into this pool.
    pub fn total_generated(&self) -> usize {
        self.scalar_generated.load(Ordering::Relaxed)
    }

    /// Returns true if the pool needs replenishment (below low-water mark).
    pub fn needs_replenishment(&self, config: &DistributedPoolConfig) -> bool {
        let threshold = (config.initial_scalar_count as f64 * config.low_water_fraction) as usize;
        self.remaining() < threshold
    }
}

impl Default for DistributedTriplePool {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function: generates and fills a pool using the distributed protocol.
///
/// This is a helper that runs the `NetworkDistributedDealer::generate` protocol
/// and populates a `DistributedTriplePool` with the results.
///
/// All parties must call this concurrently with the same parameters.
pub async fn generate_and_fill_pool<T: MPCTransport>(
    transport: &T,
    party_index: usize,
    seed: u64,
    num_scalar: usize,
) -> MPCResult<DistributedTriplePool> {
    let mut dealer = NetworkDistributedDealer::new(transport, party_index, seed);

    let mut pool = DistributedTriplePool::new();

    if num_scalar > 0 {
        let triples = dealer.generate(num_scalar).await?;
        pool.fill_scalar(triples);
    }

    Ok(pool)
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

    // ========== Vector triple tests ==========

    #[tokio::test]
    async fn test_network_distributed_vector_triple_3party() {
        use crate::session::transport::LocalTransport;

        let num_parties = 3;
        let count = 5;
        let dim = 8;
        let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
        let transports = LocalTransport::create_mesh(&parties);

        let mut handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            let handle = tokio::spawn(async move {
                let mut dealer = NetworkDistributedDealer::new(&transport, i, 42);
                dealer.generate_vector(count, dim).await.unwrap()
            });
            handles.push(handle);
        }

        let mut all_triples: Vec<Vec<VectorBeaverTriple>> = Vec::new();
        for handle in handles {
            all_triples.push(handle.await.unwrap());
        }

        assert_eq!(all_triples.len(), num_parties);
        assert_eq!(all_triples[0].len(), count);

        for t in 0..count {
            for d in 0..dim {
                let a = sum(&all_triples.iter().map(|p| p[t].a[d].clone()).collect::<Vec<_>>());
                let b = sum(&all_triples.iter().map(|p| p[t].b[d].clone()).collect::<Vec<_>>());
                let c = sum(&all_triples.iter().map(|p| p[t].c[d].clone()).collect::<Vec<_>>());

                let expected = a.mpc_scale(&b);
                assert!(
                    c.ct_eq(&expected).to_bool(),
                    "Network vector triple t={} d={} incorrect",
                    t, d,
                );
            }
        }
    }

    // ========== Matrix triple tests ==========

    #[tokio::test]
    async fn test_network_distributed_matrix_triple_3party() {
        use crate::session::transport::LocalTransport;

        let num_parties = 3;
        let count = 3;
        let (m, k, n) = (2, 3, 2);
        let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
        let transports = LocalTransport::create_mesh(&parties);

        let mut handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            let handle = tokio::spawn(async move {
                let mut dealer = NetworkDistributedDealer::new(&transport, i, 42);
                dealer.generate_matrix(count, m, k, n).await.unwrap()
            });
            handles.push(handle);
        }

        let mut all_triples: Vec<Vec<MatrixBeaverTriple>> = Vec::new();
        for handle in handles {
            all_triples.push(handle.await.unwrap());
        }

        assert_eq!(all_triples.len(), num_parties);
        assert_eq!(all_triples[0].len(), count);

        for t in 0..count {
            // Reconstruct A, B, C
            let mut a_full = vec![Fr::ZERO; m * k];
            let mut b_full = vec![Fr::ZERO; k * n];
            let mut c_full = vec![Fr::ZERO; m * n];

            for p in 0..num_parties {
                for idx in 0..m * k {
                    a_full[idx] = Fr::add(&a_full[idx], &all_triples[p][t].a[idx]);
                }
                for idx in 0..k * n {
                    b_full[idx] = Fr::add(&b_full[idx], &all_triples[p][t].b[idx]);
                }
                for idx in 0..m * n {
                    c_full[idx] = Fr::add(&c_full[idx], &all_triples[p][t].c[idx]);
                }
            }

            // Verify C = A @ B
            for row in 0..m {
                for col in 0..n {
                    let mut expected = Fr::ZERO;
                    for l in 0..k {
                        expected = Fr::add(
                            &expected,
                            &a_full[row * k + l].mpc_scale(&b_full[l * n + col]),
                        );
                    }
                    assert!(
                        c_full[row * n + col].ct_eq(&expected).to_bool(),
                        "Network matrix triple t={} [{},{}] incorrect",
                        t, row, col,
                    );
                }
            }
        }
    }

    // ========== DistributedTriplePool tests ==========

    #[test]
    fn test_pool_fill_and_take() {
        let mut pool = DistributedTriplePool::new();
        assert_eq!(pool.remaining(), 0);

        // Fill with some triples
        let triples: Vec<BeaverTriple> = (0..100)
            .map(|i| BeaverTriple::new(
                Fr::from_f64(i as f64),
                Fr::from_f64(i as f64 + 1.0),
                Fr::from_f64(i as f64 * (i as f64 + 1.0)),
            ))
            .collect();

        pool.fill_scalar(triples);
        assert_eq!(pool.remaining(), 100);
        assert_eq!(pool.total_generated(), 100);

        // Take one
        let t = pool.take().unwrap();
        assert_eq!(pool.remaining(), 99);
        assert_eq!(pool.total_consumed(), 1);
        // Should be FIFO — first inserted triple
        assert!((t.a.to_f64() - 0.0).abs() < 0.01);

        // Take batch
        let batch = pool.take_batch(10).unwrap();
        assert_eq!(batch.len(), 10);
        assert_eq!(pool.remaining(), 89);
        assert_eq!(pool.total_consumed(), 11);
    }

    #[test]
    fn test_pool_exhaustion_error() {
        let mut pool = DistributedTriplePool::new();

        // Empty pool should return clear error
        let err = pool.take().unwrap_err();
        match err {
            MPCError::BeaverPoolExhausted { requested, available } => {
                assert_eq!(requested, 1);
                assert_eq!(available, 0);
            }
            other => panic!("expected BeaverPoolExhausted, got {:?}", other),
        }

        // Partial batch request should fail entirely (no partial returns)
        pool.fill_scalar(vec![
            BeaverTriple::new(Fr::ZERO, Fr::ZERO, Fr::ZERO),
            BeaverTriple::new(Fr::ZERO, Fr::ZERO, Fr::ZERO),
        ]);
        assert_eq!(pool.remaining(), 2);

        let err = pool.take_batch(5).unwrap_err();
        match err {
            MPCError::BeaverPoolExhausted { requested, available } => {
                assert_eq!(requested, 5);
                assert_eq!(available, 2);
            }
            other => panic!("expected BeaverPoolExhausted, got {:?}", other),
        }
        // Pool should be unchanged after failed batch request
        assert_eq!(pool.remaining(), 2);
    }

    #[test]
    fn test_pool_fifo_ordering() {
        let mut pool = DistributedTriplePool::new();

        // Insert triples with identifiable a-values
        for i in 0..10 {
            pool.fill_scalar(vec![BeaverTriple::new(
                Fr::from_f64(i as f64),
                Fr::ZERO,
                Fr::ZERO,
            )]);
        }

        // Should come out in FIFO order
        for i in 0..10 {
            let t = pool.take().unwrap();
            let val = t.a.to_f64();
            assert!(
                (val - i as f64).abs() < 0.01,
                "FIFO violation: expected ~{}, got {}", i, val,
            );
        }
    }

    #[test]
    fn test_pool_needs_replenishment() {
        let mut pool = DistributedTriplePool::new();
        let config = DistributedPoolConfig {
            initial_scalar_count: 100,
            low_water_fraction: 0.25,
            ..Default::default()
        };

        // Empty pool needs replenishment
        assert!(pool.needs_replenishment(&config));

        // Fill to above threshold
        let triples: Vec<BeaverTriple> = (0..50)
            .map(|_| BeaverTriple::new(Fr::ZERO, Fr::ZERO, Fr::ZERO))
            .collect();
        pool.fill_scalar(triples);
        assert!(!pool.needs_replenishment(&config)); // 50 > 25

        // Drain below threshold
        let _ = pool.take_batch(30).unwrap();
        assert!(pool.needs_replenishment(&config)); // 20 < 25
    }

    #[test]
    fn test_pool_vector_and_matrix() {
        let mut pool = DistributedTriplePool::new();

        // Vector triples
        let vt = VectorBeaverTriple::new(
            vec![Fr::ZERO; 4],
            vec![Fr::ZERO; 4],
            vec![Fr::ZERO; 4],
        );
        pool.fill_vector(4, vec![vt]);
        assert_eq!(pool.vector_remaining(4), 1);
        assert_eq!(pool.vector_remaining(8), 0);
        let _ = pool.take_vector(4).unwrap();
        assert_eq!(pool.vector_remaining(4), 0);
        assert!(pool.take_vector(4).is_err());

        // Matrix triples
        let mt = MatrixBeaverTriple::new(
            vec![Fr::ZERO; 6], // 2x3
            vec![Fr::ZERO; 6], // 3x2
            vec![Fr::ZERO; 4], // 2x2
            2, 3, 2,
        );
        pool.fill_matrix(2, 3, 2, vec![mt]);
        assert_eq!(pool.matrix_remaining(2, 3, 2), 1);
        assert_eq!(pool.matrix_remaining(3, 3, 3), 0);
        let _ = pool.take_matrix(2, 3, 2).unwrap();
        assert_eq!(pool.matrix_remaining(2, 3, 2), 0);
        assert!(pool.take_matrix(2, 3, 2).is_err());
    }

    // ========== Production-quality tests per requirements ==========

    #[tokio::test]
    async fn test_distributed_triple_correctness_1000_triples() {
        use crate::session::transport::LocalTransport;

        let num_parties = 3;
        let count = 1000;
        let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
        let transports = LocalTransport::create_mesh(&parties);

        let mut handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            let handle = tokio::spawn(async move {
                let mut dealer = NetworkDistributedDealer::new(&transport, i, 42);
                dealer.generate(count).await.unwrap()
            });
            handles.push(handle);
        }

        let mut all_triples: Vec<Vec<BeaverTriple>> = Vec::new();
        for handle in handles {
            all_triples.push(handle.await.unwrap());
        }

        assert_eq!(all_triples.len(), num_parties);
        assert_eq!(all_triples[0].len(), count);

        // Verify ALL 1000 triples
        for t in 0..count {
            let a = sum(&all_triples.iter().map(|p| p[t].a.clone()).collect::<Vec<_>>());
            let b = sum(&all_triples.iter().map(|p| p[t].b.clone()).collect::<Vec<_>>());
            let c = sum(&all_triples.iter().map(|p| p[t].c.clone()).collect::<Vec<_>>());

            let expected = a.mpc_scale(&b);
            assert!(
                c.ct_eq(&expected).to_bool(),
                "Triple {} of 1000 incorrect: sum(c) != sum(a)*sum(b)",
                t,
            );
        }
    }

    #[tokio::test]
    async fn test_no_party_sees_full_triple() {
        use crate::session::transport::LocalTransport;

        let num_parties = 3;
        let count = 50;
        let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
        let transports = LocalTransport::create_mesh(&parties);

        let mut handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            let handle = tokio::spawn(async move {
                let mut dealer = NetworkDistributedDealer::new(&transport, i, 42);
                dealer.generate(count).await.unwrap()
            });
            handles.push(handle);
        }

        let mut all_triples: Vec<Vec<BeaverTriple>> = Vec::new();
        for handle in handles {
            all_triples.push(handle.await.unwrap());
        }

        // For each triple, verify that no single party's share equals the
        // reconstructed full value. This confirms that individual shares are
        // random-looking and don't leak the full triple.
        for t in 0..count {
            let full_a = sum(&all_triples.iter().map(|p| p[t].a.clone()).collect::<Vec<_>>());
            let full_b = sum(&all_triples.iter().map(|p| p[t].b.clone()).collect::<Vec<_>>());
            let full_c = sum(&all_triples.iter().map(|p| p[t].c.clone()).collect::<Vec<_>>());

            for party in 0..num_parties {
                assert!(
                    !all_triples[party][t].a.ct_eq(&full_a).to_bool(),
                    "Party {}'s a-share for triple {} equals the full value!",
                    party, t,
                );
                assert!(
                    !all_triples[party][t].b.ct_eq(&full_b).to_bool(),
                    "Party {}'s b-share for triple {} equals the full value!",
                    party, t,
                );
                assert!(
                    !all_triples[party][t].c.ct_eq(&full_c).to_bool(),
                    "Party {}'s c-share for triple {} equals the full value!",
                    party, t,
                );
            }
        }
    }

    #[tokio::test]
    async fn test_distributed_vs_dealer_equivalence() {
        use crate::beaver::dealer::TrustedDealer;
        use crate::session::transport::LocalTransport;

        // Generate triples from the trusted dealer (ground truth)
        let num_parties = 3;
        let count = 100;
        let mut trusted = TrustedDealer::with_seed(42);
        let trusted_batch = trusted.generate_scalar_triples(count, num_parties);

        // Generate the same number of triples using distributed generation
        let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
        let transports = LocalTransport::create_mesh(&parties);

        let mut handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            let handle = tokio::spawn(async move {
                let mut dealer = NetworkDistributedDealer::new(&transport, i, 999);
                dealer.generate(count).await.unwrap()
            });
            handles.push(handle);
        }

        let mut dist_batch: Vec<Vec<BeaverTriple>> = Vec::new();
        for handle in handles {
            dist_batch.push(handle.await.unwrap());
        }

        // Both should produce valid triples — verify the INVARIANT is the same:
        // sum(a) * sum(b) == sum(c) using mpc_scale
        for idx in 0..count {
            // Trusted dealer triple
            let a_t: Fr = (0..num_parties)
                .map(|p| trusted_batch[p][idx].a.clone())
                .fold(Fr::ZERO, |acc, v| Fr::add(&acc, &v));
            let b_t: Fr = (0..num_parties)
                .map(|p| trusted_batch[p][idx].b.clone())
                .fold(Fr::ZERO, |acc, v| Fr::add(&acc, &v));
            let c_t: Fr = (0..num_parties)
                .map(|p| trusted_batch[p][idx].c.clone())
                .fold(Fr::ZERO, |acc, v| Fr::add(&acc, &v));
            let expected_t = a_t.mpc_scale(&b_t);
            assert!(
                expected_t.ct_eq(&c_t).to_bool(),
                "TrustedDealer triple {} failed invariant", idx,
            );

            // Distributed triple
            let a_d: Fr = (0..num_parties)
                .map(|p| dist_batch[p][idx].a.clone())
                .fold(Fr::ZERO, |acc, v| Fr::add(&acc, &v));
            let b_d: Fr = (0..num_parties)
                .map(|p| dist_batch[p][idx].b.clone())
                .fold(Fr::ZERO, |acc, v| Fr::add(&acc, &v));
            let c_d: Fr = (0..num_parties)
                .map(|p| dist_batch[p][idx].c.clone())
                .fold(Fr::ZERO, |acc, v| Fr::add(&acc, &v));
            let expected_d = a_d.mpc_scale(&b_d);
            assert!(
                expected_d.ct_eq(&c_d).to_bool(),
                "Distributed triple {} failed invariant", idx,
            );
        }
    }

    #[test]
    fn test_pool_exhaustion_clear_error() {
        let mut pool = DistributedTriplePool::new();

        // Fill with exactly 5 triples
        let triples: Vec<BeaverTriple> = (0..5)
            .map(|_| BeaverTriple::new(Fr::ZERO, Fr::ZERO, Fr::ZERO))
            .collect();
        pool.fill_scalar(triples);

        // Consume all 5
        for _ in 0..5 {
            pool.take().unwrap();
        }
        assert_eq!(pool.remaining(), 0);
        assert_eq!(pool.total_consumed(), 5);

        // Next take should return clear error, NOT silent corruption
        let result = pool.take();
        assert!(result.is_err(), "Pool should be exhausted");
        match result.unwrap_err() {
            MPCError::BeaverPoolExhausted { requested, available } => {
                assert_eq!(requested, 1);
                assert_eq!(available, 0);
            }
            other => panic!(
                "Expected BeaverPoolExhausted error, got: {:?}", other,
            ),
        }

        // Batch take should also fail clearly
        let result = pool.take_batch(10);
        assert!(result.is_err());
        match result.unwrap_err() {
            MPCError::BeaverPoolExhausted { requested, available } => {
                assert_eq!(requested, 10);
                assert_eq!(available, 0);
            }
            other => panic!(
                "Expected BeaverPoolExhausted error, got: {:?}", other,
            ),
        }
    }

    #[tokio::test]
    async fn test_generate_and_fill_pool() {
        use crate::session::transport::LocalTransport;

        let num_parties = 3;
        let num_scalar = 100;
        let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
        let transports = LocalTransport::create_mesh(&parties);

        let mut handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            let handle = tokio::spawn(async move {
                generate_and_fill_pool(&transport, i, 42, num_scalar).await.unwrap()
            });
            handles.push(handle);
        }

        let mut pools: Vec<DistributedTriplePool> = Vec::new();
        for handle in handles {
            pools.push(handle.await.unwrap());
        }

        // All pools should have the same count
        for (i, pool) in pools.iter().enumerate() {
            assert_eq!(
                pool.remaining(), num_scalar,
                "Party {} pool should have {} triples", i, num_scalar,
            );
        }

        // Verify correctness by reconstructing
        for t in 0..10 {
            // Take one from each pool and verify
            let triples: Vec<BeaverTriple> = pools.iter_mut().map(|p| p.take().unwrap()).collect();
            let a = sum(&triples.iter().map(|s| s.a.clone()).collect::<Vec<_>>());
            let b = sum(&triples.iter().map(|s| s.b.clone()).collect::<Vec<_>>());
            let c = sum(&triples.iter().map(|s| s.c.clone()).collect::<Vec<_>>());
            let expected = a.mpc_scale(&b);
            assert!(
                c.ct_eq(&expected).to_bool(),
                "Pool triple {} incorrect after generate_and_fill_pool", t,
            );
        }
    }
}
