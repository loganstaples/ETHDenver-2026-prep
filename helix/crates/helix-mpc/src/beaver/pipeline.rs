//! Efficient Beaver triple pre-generation pipeline.
//!
//! This module implements a high-performance pipeline for generating Beaver triples
//! in the background, ensuring the online phase never blocks waiting for preprocessing.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
//! │   Demand        │───▶│   Generation     │───▶│   Verification  │
//! │   Predictor     │    │   Workers        │    │   & Commit      │
//! └─────────────────┘    └──────────────────┘    └─────────────────┘
//!         │                      │                       │
//!         │                      │                       ▼
//!         │                      │               ┌─────────────────┐
//!         │                      └──────────────▶│   Triple Pool   │
//!         │                                      └─────────────────┘
//!         │                                              ▲
//!         └──────────────────────────────────────────────┘
//!                         (feedback loop)
//! ```
//!
//! # Features
//!
//! - **Demand Prediction**: Estimates triple requirements based on model architecture
//! - **Parallel Generation**: Multiple worker threads generate triples concurrently
//! - **Background Replenishment**: Automatically maintains pool above threshold
//! - **Batch Optimization**: Groups triple generation for efficiency
//! - **Memory Management**: Bounds memory usage with configurable limits

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, Receiver, Sender, TryRecvError};
use parking_lot::{Mutex, RwLock};

use crate::beaver::dealer::TrustedDealer;
use crate::beaver::distributed::DistributedTripleGen;
use crate::beaver::pool::{BeaverPool, MatrixDims};
use crate::beaver::triple::{BeaverTriple, MatrixBeaverTriple, VectorBeaverTriple};
use crate::error::{MPCError, MPCResult};

/// Source of Beaver triple generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TripleSource {
    /// Centralized generation using a trusted dealer (demo/testing).
    TrustedDealer,
    /// Distributed generation using the pairwise cross-term protocol in Fr.
    /// No trusted party required — all arithmetic in BN254 scalar field.
    Distributed,
}

/// Configuration for the Beaver triple pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of worker threads for triple generation.
    pub num_workers: usize,
    /// Minimum scalar triples to maintain in pool.
    pub min_scalar_triples: usize,
    /// Maximum scalar triples to keep in pool.
    pub max_scalar_triples: usize,
    /// Batch size for scalar triple generation.
    pub scalar_batch_size: usize,
    /// Minimum matrix triples per dimension combo.
    pub min_matrix_triples: usize,
    /// Maximum matrix triples per dimension combo.
    pub max_matrix_triples: usize,
    /// Batch size for matrix triple generation.
    pub matrix_batch_size: usize,
    /// How often to check pool levels (ms).
    pub replenishment_interval_ms: u64,
    /// Enable demand prediction.
    pub enable_prediction: bool,
    /// Lookahead steps for demand prediction.
    pub prediction_lookahead: usize,
    /// Source of triple generation.
    pub triple_source: TripleSource,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            num_workers: 4,
            min_scalar_triples: 10_000,
            max_scalar_triples: 100_000,
            scalar_batch_size: 1_000,
            min_matrix_triples: 50,
            max_matrix_triples: 500,
            matrix_batch_size: 10,
            replenishment_interval_ms: 100,
            enable_prediction: true,
            prediction_lookahead: 5,
            triple_source: TripleSource::TrustedDealer,
        }
    }
}

impl PipelineConfig {
    /// Configuration optimized for small models (demo).
    pub fn small_model() -> Self {
        Self {
            num_workers: 2,
            min_scalar_triples: 1_000,
            max_scalar_triples: 10_000,
            scalar_batch_size: 500,
            min_matrix_triples: 20,
            max_matrix_triples: 100,
            matrix_batch_size: 5,
            replenishment_interval_ms: 50,
            enable_prediction: true,
            prediction_lookahead: 3,
            triple_source: TripleSource::TrustedDealer,
        }
    }

    /// Configuration for large models.
    pub fn large_model() -> Self {
        Self {
            num_workers: 8,
            min_scalar_triples: 100_000,
            max_scalar_triples: 1_000_000,
            scalar_batch_size: 10_000,
            min_matrix_triples: 100,
            max_matrix_triples: 1_000,
            matrix_batch_size: 20,
            replenishment_interval_ms: 200,
            enable_prediction: true,
            prediction_lookahead: 10,
            triple_source: TripleSource::TrustedDealer,
        }
    }

    /// Configuration with distributed (trustless) triple generation.
    pub fn distributed(num_workers: usize) -> Self {
        Self {
            triple_source: TripleSource::Distributed,
            num_workers,
            ..Self::default()
        }
    }
}

/// Types of triple generation requests.
#[derive(Debug, Clone)]
pub enum GenerationRequest {
    /// Generate scalar triples.
    Scalar {
        count: usize,
        priority: Priority,
    },
    /// Generate vector triples.
    Vector {
        dim: usize,
        count: usize,
        priority: Priority,
    },
    /// Generate matrix triples.
    Matrix {
        m: usize,
        k: usize,
        n: usize,
        count: usize,
        priority: Priority,
    },
    /// Shutdown the worker.
    Shutdown,
}

/// Priority for generation requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Generated triples ready for pool.
#[derive(Debug)]
pub enum GeneratedTriples {
    Scalar(Vec<Vec<BeaverTriple>>),
    Vector(usize, Vec<Vec<VectorBeaverTriple>>),
    Matrix(MatrixDims, Vec<Vec<MatrixBeaverTriple>>),
}

/// Demand prediction for Beaver triples.
#[derive(Debug)]
pub struct DemandPredictor {
    /// Historical consumption rates.
    scalar_consumption: RwLock<Vec<(Instant, usize)>>,
    matrix_consumption: RwLock<HashMap<MatrixDims, Vec<(Instant, usize)>>>,
    /// Model architecture for prediction.
    hidden_dim: AtomicUsize,
    num_layers: AtomicUsize,
    /// Prediction window.
    window_duration: Duration,
}

impl DemandPredictor {
    pub fn new(window_duration: Duration) -> Self {
        Self {
            scalar_consumption: RwLock::new(Vec::new()),
            matrix_consumption: RwLock::new(HashMap::new()),
            hidden_dim: AtomicUsize::new(0),
            num_layers: AtomicUsize::new(0),
            window_duration,
        }
    }

    /// Records model architecture for prediction.
    pub fn set_model_architecture(&self, hidden_dim: usize, num_layers: usize) {
        self.hidden_dim.store(hidden_dim, Ordering::SeqCst);
        self.num_layers.store(num_layers, Ordering::SeqCst);
    }

    /// Records scalar triple consumption.
    pub fn record_scalar_consumption(&self, count: usize) {
        let mut history = self.scalar_consumption.write();
        history.push((Instant::now(), count));
        self.prune_history(&mut history);
    }

    /// Records matrix triple consumption.
    pub fn record_matrix_consumption(&self, dims: MatrixDims, count: usize) {
        let mut history = self.matrix_consumption.write();
        let entries = history.entry(dims).or_default();
        entries.push((Instant::now(), count));
        self.prune_matrix_history(entries);
    }

    fn prune_history(&self, history: &mut Vec<(Instant, usize)>) {
        let cutoff = Instant::now() - self.window_duration;
        history.retain(|(t, _)| *t > cutoff);
    }

    fn prune_matrix_history(&self, history: &mut Vec<(Instant, usize)>) {
        let cutoff = Instant::now() - self.window_duration;
        history.retain(|(t, _)| *t > cutoff);
    }

    /// Predicts scalar triple demand for the next N steps.
    pub fn predict_scalar_demand(&self, lookahead_steps: usize) -> usize {
        let history = self.scalar_consumption.read();

        if history.is_empty() {
            // Use model architecture for initial prediction
            let hidden_dim = self.hidden_dim.load(Ordering::SeqCst);
            let num_layers = self.num_layers.load(Ordering::SeqCst);

            if hidden_dim > 0 && num_layers > 0 {
                // Estimate based on typical transformer operations
                let per_step = num_layers * hidden_dim * 4;
                return per_step * lookahead_steps;
            }
            return 1000 * lookahead_steps;
        }

        // Calculate rate from history
        let total: usize = history.iter().map(|(_, c)| c).sum();
        let avg_per_event = total / history.len().max(1);

        avg_per_event * lookahead_steps
    }

    /// Predicts matrix triple demand.
    pub fn predict_matrix_demand(&self, dims: &MatrixDims, lookahead_steps: usize) -> usize {
        let history = self.matrix_consumption.read();

        if let Some(entries) = history.get(dims) {
            if !entries.is_empty() {
                let total: usize = entries.iter().map(|(_, c)| c).sum();
                let avg = total / entries.len().max(1);
                return avg * lookahead_steps;
            }
        }

        // Default estimate
        lookahead_steps
    }

    /// Gets all active matrix dimensions from history.
    pub fn active_matrix_dims(&self) -> Vec<MatrixDims> {
        self.matrix_consumption.read().keys().cloned().collect()
    }
}

/// Pipeline statistics.
#[derive(Debug, Clone)]
pub struct PipelineStats {
    /// Total triples generated.
    pub triples_generated: u64,
    /// Total generation time.
    pub generation_time_ms: u64,
    /// Current queue depth.
    pub queue_depth: usize,
    /// Active workers.
    pub active_workers: usize,
    /// Triples per second.
    pub throughput: f64,
}

/// Beaver triple pre-generation pipeline.
#[allow(dead_code)]
pub struct BeaverPipeline {
    /// Configuration.
    config: PipelineConfig,
    /// Number of parties.
    num_parties: usize,
    /// Pools for each party.
    pools: Arc<Vec<RwLock<BeaverPool>>>,
    /// Demand predictor.
    predictor: Arc<DemandPredictor>,
    /// Request channel sender.
    request_tx: Sender<GenerationRequest>,
    /// Result channel receiver.
    result_rx: Receiver<GeneratedTriples>,
    /// Worker handles.
    workers: Mutex<Vec<JoinHandle<()>>>,
    /// Replenishment thread handle.
    replenisher: Mutex<Option<JoinHandle<()>>>,
    /// Shutdown flag.
    shutdown: Arc<AtomicBool>,
    /// Statistics.
    stats: Arc<PipelineStatsInternal>,
}

struct PipelineStatsInternal {
    triples_generated: AtomicU64,
    generation_time_ms: AtomicU64,
    active_workers: AtomicUsize,
}

impl BeaverPipeline {
    /// Creates a new Beaver triple pipeline.
    pub fn new(num_parties: usize, config: PipelineConfig) -> Self {
        let (request_tx, request_rx): (Sender<GenerationRequest>, Receiver<GenerationRequest>) = bounded(1000);
        let (result_tx, result_rx): (Sender<GeneratedTriples>, Receiver<GeneratedTriples>) = bounded(1000);

        // Create pools for each party
        let pools: Vec<RwLock<BeaverPool>> = (0..num_parties)
            .map(|i| RwLock::new(BeaverPool::new(i, num_parties, config.scalar_batch_size)))
            .collect();

        let pipeline = Self {
            config: config.clone(),
            num_parties,
            pools: Arc::new(pools),
            predictor: Arc::new(DemandPredictor::new(Duration::from_secs(60))),
            request_tx,
            result_rx,
            workers: Mutex::new(Vec::new()),
            replenisher: Mutex::new(None),
            shutdown: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(PipelineStatsInternal {
                triples_generated: AtomicU64::new(0),
                generation_time_ms: AtomicU64::new(0),
                active_workers: AtomicUsize::new(0),
            }),
        };

        // Spawn workers
        let mut workers = pipeline.workers.lock();
        for worker_id in 0..config.num_workers {
            let rx = request_rx.clone();
            let tx = result_tx.clone();
            let num_parties = num_parties;
            let stats = pipeline.stats.clone();
            let shutdown = pipeline.shutdown.clone();
            let triple_source = config.triple_source;

            let handle = thread::spawn(move || {
                Self::worker_loop(worker_id, rx, tx, num_parties, stats, shutdown, triple_source);
            });
            workers.push(handle);
        }
        drop(workers);

        pipeline
    }

    /// Starts the background replenishment thread.
    pub fn start_replenishment(&self) {
        let pools = self.pools.clone();
        let request_tx = self.request_tx.clone();
        let config = self.config.clone();
        let predictor = self.predictor.clone();
        let shutdown = self.shutdown.clone();
        let result_rx = self.result_rx.clone();

        let handle = thread::spawn(move || {
            Self::replenishment_loop(pools, request_tx, result_rx, config, predictor, shutdown);
        });

        *self.replenisher.lock() = Some(handle);
    }

    /// Worker thread loop.
    fn worker_loop(
        _worker_id: usize,
        request_rx: Receiver<GenerationRequest>,
        result_tx: Sender<GeneratedTriples>,
        num_parties: usize,
        stats: Arc<PipelineStatsInternal>,
        shutdown: Arc<AtomicBool>,
        triple_source: TripleSource,
    ) {
        let mut dealer = TrustedDealer::new();
        let mut distributed_seed: u64 = 0;
        stats.active_workers.fetch_add(1, Ordering::SeqCst);

        loop {
            if shutdown.load(Ordering::SeqCst) {
                break;
            }

            match request_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(request) => {
                    let start = Instant::now();

                    match request {
                        GenerationRequest::Scalar { count, .. } => {
                            let triples = match triple_source {
                                TripleSource::TrustedDealer => {
                                    dealer.generate_scalar_triples(count, num_parties)
                                }
                                TripleSource::Distributed => {
                                    distributed_seed = distributed_seed.wrapping_add(1);
                                    DistributedTripleGen::simulate_distributed_batch(
                                        count,
                                        num_parties,
                                        distributed_seed,
                                    )
                                }
                            };
                            stats.triples_generated.fetch_add(count as u64, Ordering::SeqCst);
                            let _ = result_tx.send(GeneratedTriples::Scalar(triples));
                        }
                        GenerationRequest::Vector { dim, count, .. } => {
                            let mut all_triples = Vec::with_capacity(num_parties);
                            for _ in 0..num_parties {
                                all_triples.push(Vec::with_capacity(count));
                            }

                            for _ in 0..count {
                                let triples = dealer.generate_vector_triple(dim, num_parties);
                                for (i, t) in triples.into_iter().enumerate() {
                                    all_triples[i].push(t);
                                }
                            }

                            stats.triples_generated.fetch_add(count as u64 * dim as u64, Ordering::SeqCst);
                            let _ = result_tx.send(GeneratedTriples::Vector(dim, all_triples));
                        }
                        GenerationRequest::Matrix { m, k, n, count, .. } => {
                            let mut all_triples = Vec::with_capacity(num_parties);
                            for _ in 0..num_parties {
                                all_triples.push(Vec::with_capacity(count));
                            }

                            for _ in 0..count {
                                let triples = dealer.generate_matrix_triple(m, k, n, num_parties);
                                for (i, t) in triples.into_iter().enumerate() {
                                    all_triples[i].push(t);
                                }
                            }

                            let dims = MatrixDims { m, k, n };
                            stats.triples_generated.fetch_add(count as u64, Ordering::SeqCst);
                            let _ = result_tx.send(GeneratedTriples::Matrix(dims, all_triples));
                        }
                        GenerationRequest::Shutdown => {
                            break;
                        }
                    }

                    let elapsed = start.elapsed().as_millis() as u64;
                    stats.generation_time_ms.fetch_add(elapsed, Ordering::SeqCst);
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
            }
        }

        stats.active_workers.fetch_sub(1, Ordering::SeqCst);
    }

    /// Replenishment thread loop.
    fn replenishment_loop(
        pools: Arc<Vec<RwLock<BeaverPool>>>,
        request_tx: Sender<GenerationRequest>,
        result_rx: Receiver<GeneratedTriples>,
        config: PipelineConfig,
        predictor: Arc<DemandPredictor>,
        shutdown: Arc<AtomicBool>,
    ) {
        let interval = Duration::from_millis(config.replenishment_interval_ms);

        loop {
            if shutdown.load(Ordering::SeqCst) {
                break;
            }

            // Process any completed generations
            loop {
                match result_rx.try_recv() {
                    Ok(generated) => {
                        Self::add_to_pools(&pools, generated);
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => return,
                }
            }

            // Check scalar triple levels
            let scalar_available: usize = pools
                .iter()
                .map(|p| p.read().scalar_available())
                .min()
                .unwrap_or(0);

            let predicted_demand = if config.enable_prediction {
                predictor.predict_scalar_demand(config.prediction_lookahead)
            } else {
                0
            };

            let target = config.min_scalar_triples.max(predicted_demand);
            if scalar_available < target && scalar_available < config.max_scalar_triples {
                let to_generate = (target - scalar_available).min(config.scalar_batch_size);
                let priority = if scalar_available < config.min_scalar_triples / 4 {
                    Priority::Critical
                } else if scalar_available < config.min_scalar_triples / 2 {
                    Priority::High
                } else {
                    Priority::Normal
                };

                let _ = request_tx.send(GenerationRequest::Scalar {
                    count: to_generate,
                    priority,
                });
            }

            // Check matrix triple levels for known dimensions
            for dims in predictor.active_matrix_dims() {
                let matrix_available: usize = pools
                    .iter()
                    .map(|p| p.read().matrix_available(dims.m, dims.k, dims.n))
                    .min()
                    .unwrap_or(0);

                let predicted = if config.enable_prediction {
                    predictor.predict_matrix_demand(&dims, config.prediction_lookahead)
                } else {
                    0
                };

                let target = config.min_matrix_triples.max(predicted);
                if matrix_available < target && matrix_available < config.max_matrix_triples {
                    let to_generate = (target - matrix_available).min(config.matrix_batch_size);
                    let _ = request_tx.send(GenerationRequest::Matrix {
                        m: dims.m,
                        k: dims.k,
                        n: dims.n,
                        count: to_generate,
                        priority: Priority::Normal,
                    });
                }
            }

            thread::sleep(interval);
        }
    }

    /// Adds generated triples to pools.
    fn add_to_pools(pools: &Arc<Vec<RwLock<BeaverPool>>>, generated: GeneratedTriples) {
        match generated {
            GeneratedTriples::Scalar(per_party) => {
                for (i, triples) in per_party.into_iter().enumerate() {
                    if i < pools.len() {
                        pools[i].write().fill_scalar(triples);
                    }
                }
            }
            GeneratedTriples::Vector(dim, per_party) => {
                for (i, triples) in per_party.into_iter().enumerate() {
                    if i < pools.len() {
                        pools[i].write().fill_vector(dim, triples);
                    }
                }
            }
            GeneratedTriples::Matrix(dims, per_party) => {
                for (i, triples) in per_party.into_iter().enumerate() {
                    if i < pools.len() {
                        pools[i].write().fill_matrix(dims.m, dims.k, dims.n, triples);
                    }
                }
            }
        }
    }

    /// Gets a reference to a party's pool.
    pub fn pool(&self, party_index: usize) -> Option<&RwLock<BeaverPool>> {
        self.pools.get(party_index)
    }

    /// Requests immediate generation of specific triples.
    pub fn request_generation(&self, request: GenerationRequest) -> MPCResult<()> {
        self.request_tx
            .send(request)
            .map_err(|_| MPCError::CommunicationError("Pipeline channel closed".into()))
    }

    /// Pre-generates triples for a training step.
    pub fn prepare_for_step(&self, hidden_dim: usize, num_layers: usize) -> MPCResult<()> {
        self.predictor.set_model_architecture(hidden_dim, num_layers);

        // Request scalar triples for activations
        let scalar_needed = num_layers * hidden_dim * 4;
        self.request_generation(GenerationRequest::Scalar {
            count: scalar_needed,
            priority: Priority::High,
        })?;

        // Request matrix triples for attention and MLP
        let mlp_dim = hidden_dim * 4;

        for _ in 0..num_layers {
            // Attention projections
            self.request_generation(GenerationRequest::Matrix {
                m: hidden_dim,
                k: hidden_dim,
                n: hidden_dim,
                count: 4,
                priority: Priority::High,
            })?;

            // MLP projections
            self.request_generation(GenerationRequest::Matrix {
                m: hidden_dim,
                k: hidden_dim,
                n: mlp_dim,
                count: 1,
                priority: Priority::High,
            })?;

            self.request_generation(GenerationRequest::Matrix {
                m: hidden_dim,
                k: mlp_dim,
                n: hidden_dim,
                count: 1,
                priority: Priority::High,
            })?;
        }

        Ok(())
    }

    /// Gets pipeline statistics.
    pub fn stats(&self) -> PipelineStats {
        let generated = self.stats.triples_generated.load(Ordering::SeqCst);
        let time_ms = self.stats.generation_time_ms.load(Ordering::SeqCst);
        let active = self.stats.active_workers.load(Ordering::SeqCst);

        let throughput = if time_ms > 0 {
            (generated as f64 / time_ms as f64) * 1000.0
        } else {
            0.0
        };

        PipelineStats {
            triples_generated: generated,
            generation_time_ms: time_ms,
            queue_depth: self.request_tx.len(),
            active_workers: active,
            throughput,
        }
    }

    /// Records scalar triple consumption for prediction.
    pub fn record_scalar_consumption(&self, count: usize) {
        self.predictor.record_scalar_consumption(count);
    }

    /// Records matrix triple consumption for prediction.
    pub fn record_matrix_consumption(&self, m: usize, k: usize, n: usize, count: usize) {
        self.predictor.record_matrix_consumption(MatrixDims { m, k, n }, count);
    }

    /// Blocks until at least `count` scalar triples are available in each
    /// party's pool, or `timeout` expires.
    ///
    /// If the pool is below the threshold, sends a Critical-priority generation
    /// request and polls until the triples arrive or the deadline passes.
    pub fn ensure_available(
        &self,
        count: usize,
        timeout: Duration,
    ) -> MPCResult<()> {
        let deadline = Instant::now() + timeout;
        let poll_interval = Duration::from_millis(10);

        loop {
            // Check minimum across all parties.
            let min_available: usize = self
                .pools
                .iter()
                .map(|p| p.read().scalar_available())
                .min()
                .unwrap_or(0);

            if min_available >= count {
                return Ok(());
            }

            if Instant::now() >= deadline {
                return Err(MPCError::BeaverPoolExhausted {
                    requested: count,
                    available: min_available,
                });
            }

            // Request urgent generation for the deficit.
            let deficit = count.saturating_sub(min_available);
            let _ = self.request_generation(GenerationRequest::Scalar {
                count: deficit,
                priority: Priority::Critical,
            });

            // Process any pending results.
            while let Ok(generated) = self.result_rx.try_recv() {
                Self::add_to_pools(&self.pools, generated);
            }

            thread::sleep(poll_interval);
        }
    }

    /// Shuts down the pipeline.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);

        // Send shutdown to all workers
        for _ in 0..self.config.num_workers {
            let _ = self.request_tx.send(GenerationRequest::Shutdown);
        }

        // Wait for workers
        let mut workers = self.workers.lock();
        for handle in workers.drain(..) {
            let _ = handle.join();
        }

        // Wait for replenisher
        if let Some(handle) = self.replenisher.lock().take() {
            let _ = handle.join();
        }
    }
}

impl Drop for BeaverPipeline {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Builder for configuring and creating a Beaver pipeline.
pub struct PipelineBuilder {
    num_parties: usize,
    config: PipelineConfig,
    initial_scalar: usize,
    matrix_dims: Vec<(usize, usize, usize, usize)>,
}

impl PipelineBuilder {
    pub fn new(num_parties: usize) -> Self {
        Self {
            num_parties,
            config: PipelineConfig::default(),
            initial_scalar: 0,
            matrix_dims: Vec::new(),
        }
    }

    pub fn config(mut self, config: PipelineConfig) -> Self {
        self.config = config;
        self
    }

    pub fn initial_scalar_triples(mut self, count: usize) -> Self {
        self.initial_scalar = count;
        self
    }

    pub fn add_matrix_dimension(mut self, m: usize, k: usize, n: usize, count: usize) -> Self {
        self.matrix_dims.push((m, k, n, count));
        self
    }

    pub fn build(self) -> MPCResult<BeaverPipeline> {
        let pipeline = BeaverPipeline::new(self.num_parties, self.config);

        // Pre-generate initial triples
        if self.initial_scalar > 0 {
            pipeline.request_generation(GenerationRequest::Scalar {
                count: self.initial_scalar,
                priority: Priority::High,
            })?;
        }

        for (m, k, n, count) in self.matrix_dims {
            pipeline.request_generation(GenerationRequest::Matrix {
                m,
                k,
                n,
                count,
                priority: Priority::High,
            })?;
        }

        pipeline.start_replenishment();

        Ok(pipeline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let pipeline = BeaverPipeline::new(3, PipelineConfig::small_model());
        assert_eq!(pipeline.pools.len(), 3);
        pipeline.shutdown();
    }

    #[test]
    fn test_pipeline_builder() {
        let pipeline = PipelineBuilder::new(3)
            .config(PipelineConfig::small_model())
            .initial_scalar_triples(100)
            .add_matrix_dimension(4, 4, 4, 5)
            .build()
            .unwrap();

        // Wait for generation
        std::thread::sleep(Duration::from_millis(200));

        let stats = pipeline.stats();
        assert!(stats.triples_generated > 0);

        pipeline.shutdown();
    }

    #[test]
    fn test_demand_predictor() {
        let predictor = DemandPredictor::new(Duration::from_secs(60));

        predictor.set_model_architecture(64, 2);
        let initial = predictor.predict_scalar_demand(1);
        assert!(initial > 0);

        predictor.record_scalar_consumption(100);
        predictor.record_scalar_consumption(120);

        let predicted = predictor.predict_scalar_demand(5);
        assert!(predicted > 0);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }

    #[test]
    fn test_pipeline_stats() {
        let pipeline = BeaverPipeline::new(3, PipelineConfig::small_model());

        pipeline
            .request_generation(GenerationRequest::Scalar {
                count: 100,
                priority: Priority::Normal,
            })
            .unwrap();

        std::thread::sleep(Duration::from_millis(200));

        let stats = pipeline.stats();
        assert!(stats.active_workers > 0);

        pipeline.shutdown();
    }

    #[test]
    fn test_prepare_for_step() {
        let pipeline = BeaverPipeline::new(3, PipelineConfig::small_model());
        pipeline.start_replenishment();

        pipeline.prepare_for_step(8, 1).unwrap();

        std::thread::sleep(Duration::from_millis(300));

        let stats = pipeline.stats();
        assert!(stats.triples_generated > 0);

        pipeline.shutdown();
    }

    #[test]
    fn test_matrix_dims_tracking() {
        let predictor = DemandPredictor::new(Duration::from_secs(60));

        predictor.record_matrix_consumption(MatrixDims { m: 8, k: 8, n: 8 }, 10);
        predictor.record_matrix_consumption(MatrixDims { m: 8, k: 8, n: 32 }, 5);

        let dims = predictor.active_matrix_dims();
        assert_eq!(dims.len(), 2);
    }
}
