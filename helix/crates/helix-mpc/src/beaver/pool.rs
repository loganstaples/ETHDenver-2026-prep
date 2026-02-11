//! Beaver triple pool management.
//!
//! Pre-generated triples are stored in a pool and consumed during the online
//! phase of computation. The pool handles batching, replenishment, and
//! tracking of consumption rates.

use std::collections::HashMap;

use crate::error::{MPCError, MPCResult};
use super::dealer::TrustedDealer;
use super::triple::{BeaverTriple, MatrixBeaverTriple, VectorBeaverTriple};

/// Key for matrix triple pool entries (identified by dimensions).
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct MatrixDims {
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

/// Pool of pre-generated Beaver triples for one party.
#[derive(Debug)]
#[allow(dead_code)]
pub struct BeaverPool {
    /// Scalar triples available for consumption.
    scalar_triples: Vec<BeaverTriple>,
    /// Vector triples, keyed by dimension.
    vector_triples: HashMap<usize, Vec<VectorBeaverTriple>>,
    /// Matrix triples, keyed by dimensions.
    matrix_triples: HashMap<MatrixDims, Vec<MatrixBeaverTriple>>,
    /// Total scalar triples consumed.
    scalar_consumed: usize,
    /// Total matrix triples consumed, by dimensions.
    matrix_consumed: HashMap<MatrixDims, usize>,
    /// Batch size for auto-replenishment.
    batch_size: usize,
    /// Party index (for requesting from dealer).
    party_index: usize,
    /// Total number of parties.
    num_parties: usize,
}

impl BeaverPool {
    /// Creates a new empty pool.
    pub fn new(party_index: usize, num_parties: usize, batch_size: usize) -> Self {
        Self {
            scalar_triples: Vec::new(),
            vector_triples: HashMap::new(),
            matrix_triples: HashMap::new(),
            scalar_consumed: 0,
            matrix_consumed: HashMap::new(),
            batch_size,
            party_index,
            num_parties,
        }
    }

    /// Pre-populates the pool with scalar triples from a dealer.
    pub fn fill_scalar(&mut self, triples: Vec<BeaverTriple>) {
        self.scalar_triples.extend(triples);
    }

    /// Pre-populates the pool with vector triples.
    pub fn fill_vector(&mut self, dim: usize, triples: Vec<VectorBeaverTriple>) {
        self.vector_triples
            .entry(dim)
            .or_default()
            .extend(triples);
    }

    /// Pre-populates the pool with matrix triples.
    pub fn fill_matrix(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        triples: Vec<MatrixBeaverTriple>,
    ) {
        let key = MatrixDims { m, k, n };
        self.matrix_triples.entry(key).or_default().extend(triples);
    }

    /// Consumes a scalar triple from the pool.
    pub fn take_scalar(&mut self) -> MPCResult<BeaverTriple> {
        let triple = self.scalar_triples.pop().ok_or(MPCError::BeaverPoolExhausted {
            requested: 1,
            available: 0,
        })?;
        self.scalar_consumed += 1;
        Ok(triple)
    }

    /// Consumes multiple scalar triples.
    pub fn take_scalars(&mut self, count: usize) -> MPCResult<Vec<BeaverTriple>> {
        if self.scalar_triples.len() < count {
            return Err(MPCError::BeaverPoolExhausted {
                requested: count,
                available: self.scalar_triples.len(),
            });
        }

        let start = self.scalar_triples.len() - count;
        let result = self.scalar_triples.split_off(start);
        self.scalar_consumed += count;
        Ok(result)
    }

    /// Consumes a vector triple of the specified dimension.
    pub fn take_vector(&mut self, dim: usize) -> MPCResult<VectorBeaverTriple> {
        self.vector_triples
            .get_mut(&dim)
            .and_then(|v| v.pop())
            .ok_or(MPCError::BeaverPoolExhausted {
                requested: 1,
                available: 0,
            })
    }

    /// Consumes a matrix triple for the specified dimensions.
    pub fn take_matrix(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
    ) -> MPCResult<MatrixBeaverTriple> {
        let key = MatrixDims { m, k, n };
        let triple = self
            .matrix_triples
            .get_mut(&key)
            .and_then(|v| v.pop())
            .ok_or(MPCError::BeaverPoolExhausted {
                requested: 1,
                available: 0,
            })?;

        *self.matrix_consumed.entry(key).or_insert(0) += 1;
        Ok(triple)
    }

    /// Returns the number of scalar triples available.
    pub fn scalar_available(&self) -> usize {
        self.scalar_triples.len()
    }

    /// Returns the number of vector triples available for a dimension.
    pub fn vector_available(&self, dim: usize) -> usize {
        self.vector_triples.get(&dim).map_or(0, |v| v.len())
    }

    /// Returns the number of matrix triples available for given dimensions.
    pub fn matrix_available(&self, m: usize, k: usize, n: usize) -> usize {
        let key = MatrixDims { m, k, n };
        self.matrix_triples.get(&key).map_or(0, |v| v.len())
    }

    /// Returns consumption statistics.
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            scalar_available: self.scalar_triples.len(),
            scalar_consumed: self.scalar_consumed,
            vector_dims: self.vector_triples.keys().cloned().collect(),
            matrix_dims: self.matrix_triples.keys().cloned().collect(),
        }
    }

    /// Splits the pool into `count` sub-pools, distributing scalar triples
    /// roughly evenly. Used for parallel head computation where each head
    /// needs its own independent pool.
    ///
    /// The original pool is drained of scalar triples.
    /// Vector and matrix triples remain in the original pool (not split).
    pub fn split(&mut self, count: usize) -> Vec<BeaverPool> {
        let total = self.scalar_triples.len();
        let per_pool = total / count;
        let remainder = total % count;

        let mut sub_pools = Vec::with_capacity(count);
        for i in 0..count {
            let take = if i < remainder { per_pool + 1 } else { per_pool };
            let start = self.scalar_triples.len() - take;
            let triples = self.scalar_triples.split_off(start);

            let mut pool = BeaverPool::new(self.party_index, self.num_parties, self.batch_size);
            pool.fill_scalar(triples);
            sub_pools.push(pool);
        }

        sub_pools
    }

    /// Merges sub-pools back into this pool, recovering all remaining scalar
    /// triples. Consumption counters are accumulated.
    pub fn merge(&mut self, sub_pools: Vec<BeaverPool>) {
        for sub in sub_pools {
            self.scalar_triples.extend(sub.scalar_triples);
            self.scalar_consumed += sub.scalar_consumed;
        }
    }

    /// Fills the pool from a trusted dealer for a specific training step.
    /// Pre-generates triples needed for one forward/backward pass.
    pub fn fill_for_training_step(
        &mut self,
        dealer: &mut TrustedDealer,
        hidden_dim: usize,
        num_layers: usize,
    ) {
        // Each layer needs matrix triples for:
        // - 4 attention projections: [hidden, hidden] @ [hidden, seq]
        // - 2 MLP projections: [hidden, 4*hidden] and [4*hidden, hidden]
        // Plus scalar triples for activations.

        let mlp_dim = hidden_dim * 4;

        // Scalar triples for element-wise operations (activations, normalization).
        let scalars_needed = num_layers * hidden_dim * 4;
        let scalar_batches = dealer.generate_scalar_triples(scalars_needed, self.num_parties);
        self.fill_scalar(scalar_batches[self.party_index].clone());

        // Matrix triples for attention and MLP.
        for _ in 0..num_layers {
            // Attention: Q, K, V, O projections [hidden, hidden].
            for _ in 0..4 {
                let triples = dealer.generate_matrix_triple(
                    hidden_dim,
                    hidden_dim,
                    hidden_dim,
                    self.num_parties,
                );
                self.matrix_triples
                    .entry(MatrixDims {
                        m: hidden_dim,
                        k: hidden_dim,
                        n: hidden_dim,
                    })
                    .or_default()
                    .push(triples[self.party_index].clone());
            }

            // MLP up: [hidden, 4*hidden].
            let up_triples = dealer.generate_matrix_triple(
                hidden_dim,
                hidden_dim,
                mlp_dim,
                self.num_parties,
            );
            self.matrix_triples
                .entry(MatrixDims {
                    m: hidden_dim,
                    k: hidden_dim,
                    n: mlp_dim,
                })
                .or_default()
                .push(up_triples[self.party_index].clone());

            // MLP down: [4*hidden, hidden].
            let down_triples = dealer.generate_matrix_triple(
                hidden_dim,
                mlp_dim,
                hidden_dim,
                self.num_parties,
            );
            self.matrix_triples
                .entry(MatrixDims {
                    m: hidden_dim,
                    k: mlp_dim,
                    n: hidden_dim,
                })
                .or_default()
                .push(down_triples[self.party_index].clone());
        }
    }
}

/// Statistics about the Beaver triple pool.
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub scalar_available: usize,
    pub scalar_consumed: usize,
    pub vector_dims: Vec<usize>,
    pub matrix_dims: Vec<MatrixDims>,
}

impl std::fmt::Display for PoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BeaverPool: {} scalar available ({} consumed), {} vector dims, {} matrix dims",
            self.scalar_available,
            self.scalar_consumed,
            self.vector_dims.len(),
            self.matrix_dims.len(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_basic() {
        let mut dealer = TrustedDealer::with_seed(42);
        let per_party = dealer.generate_scalar_triples(100, 3);

        let mut pool = BeaverPool::new(0, 3, 64);
        pool.fill_scalar(per_party[0].clone());

        assert_eq!(pool.scalar_available(), 100);

        let triple = pool.take_scalar().unwrap();
        assert_eq!(pool.scalar_available(), 99);

        let batch = pool.take_scalars(10).unwrap();
        assert_eq!(batch.len(), 10);
        assert_eq!(pool.scalar_available(), 89);
    }

    #[test]
    fn test_pool_exhaustion() {
        let mut pool = BeaverPool::new(0, 3, 64);

        assert!(pool.take_scalar().is_err());
        assert!(pool.take_scalars(5).is_err());
    }

    #[test]
    fn test_pool_matrix() {
        let mut dealer = TrustedDealer::with_seed(42);
        let triples = dealer.generate_matrix_triple(4, 3, 2, 3);

        let mut pool = BeaverPool::new(0, 3, 64);
        pool.fill_matrix(4, 3, 2, vec![triples[0].clone()]);

        assert_eq!(pool.matrix_available(4, 3, 2), 1);
        assert_eq!(pool.matrix_available(4, 4, 4), 0);

        let t = pool.take_matrix(4, 3, 2).unwrap();
        assert_eq!(t.m, 4);
        assert_eq!(pool.matrix_available(4, 3, 2), 0);
    }

    #[test]
    fn test_pool_split_merge() {
        let mut dealer = TrustedDealer::with_seed(42);
        let per_party = dealer.generate_scalar_triples(100, 3);

        let mut pool = BeaverPool::new(0, 3, 64);
        pool.fill_scalar(per_party[0].clone());
        assert_eq!(pool.scalar_available(), 100);

        // Split into 4 sub-pools.
        let mut subs = pool.split(4);
        assert_eq!(subs.len(), 4);
        assert_eq!(pool.scalar_available(), 0);

        // 100 / 4 = 25 each.
        let total_in_subs: usize = subs.iter().map(|s| s.scalar_available()).sum();
        assert_eq!(total_in_subs, 100);

        // Consume some from each sub-pool.
        for sub in &mut subs {
            let _ = sub.take_scalar().unwrap();
        }

        let remaining: usize = subs.iter().map(|s| s.scalar_available()).sum();
        assert_eq!(remaining, 96);

        // Merge back.
        pool.merge(subs);
        assert_eq!(pool.scalar_available(), 96);
        // Consumed counter should reflect the 4 consumed.
        assert!(pool.stats().scalar_consumed >= 4);
    }

    #[test]
    fn test_pool_split_uneven() {
        let mut dealer = TrustedDealer::with_seed(42);
        let per_party = dealer.generate_scalar_triples(7, 3);

        let mut pool = BeaverPool::new(0, 3, 64);
        pool.fill_scalar(per_party[0].clone());

        // 7 triples split into 3 = [3, 2, 2]
        let subs = pool.split(3);
        let total: usize = subs.iter().map(|s| s.scalar_available()).sum();
        assert_eq!(total, 7);
    }

    #[test]
    fn test_pool_fill_for_training() {
        let mut dealer = TrustedDealer::with_seed(42);
        let mut pool = BeaverPool::new(0, 3, 64);

        pool.fill_for_training_step(&mut dealer, 8, 2);

        // Should have scalar triples.
        assert!(pool.scalar_available() > 0);

        // Should have matrix triples for attention dims.
        assert!(pool.matrix_available(8, 8, 8) > 0);

        let stats = pool.stats();
        assert!(stats.scalar_available > 0);
    }
}
