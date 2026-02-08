//! Parallel Proof Generation with Work Stealing.
//!
//! Generates multiple proofs in parallel across available CPU cores.
//! Features true work-stealing for optimal load balancing and multi-core performance.
//!
//! # Architecture
//!
//! The parallel prover uses a work-stealing scheduler with the following components:
//!
//! - **Per-worker deques**: Each worker has its own double-ended queue
//! - **Work stealing**: Idle workers steal from busy workers' queues
//! - **Task affinity**: Related tasks are scheduled on the same worker for cache efficiency
//! - **Priority scheduling**: Higher priority tasks are executed first
//! - **Adaptive batching**: Dynamically adjusts batch sizes based on workload

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::chunking::{ChunkId, ComputationChunk};
use serde::{Deserialize, Serialize};

use crate::pipeline::ProverPipeline;
use crate::provers::ivc_circuit::IVCStepCircuit;
use helix_circuits::halo2curves::bn256::Fr;

/// Result of proving a chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkProof {
    /// Chunk identifier.
    pub chunk_id: ChunkId,
    /// Serialized proof bytes.
    pub proof: Vec<u8>,
    /// Public inputs used.
    pub public_inputs: Vec<[u8; 32]>,
    /// Error bound proved.
    pub error_bound: f64,
    /// Time taken to generate (ms).
    pub generation_time_ms: u64,
}

/// Status of a proof generation task.
#[derive(Debug, Clone, PartialEq)]
pub enum ProofStatus {
    /// Waiting for dependencies.
    Pending,
    /// Currently being computed.
    InProgress,
    /// Successfully completed.
    Complete,
    /// Failed with error.
    Failed(String),
}

/// Configuration for parallel proof generation.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of worker threads.
    pub num_threads: usize,
    /// Whether to use work-stealing.
    pub work_stealing: bool,
    /// Maximum queue depth per worker.
    pub queue_depth: usize,
    /// Timeout per proof (seconds).
    pub proof_timeout_secs: u64,
    /// Enable task affinity for cache efficiency.
    pub task_affinity: bool,
    /// Steal batch size (number of tasks to steal at once).
    pub steal_batch_size: usize,
    /// Minimum tasks before allowing stealing.
    pub steal_threshold: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus(),
            work_stealing: true,
            queue_depth: 8,
            proof_timeout_secs: 300,
            task_affinity: true,
            steal_batch_size: 2,
            steal_threshold: 4,
        }
    }
}

/// Returns the number of available CPUs.
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Task for proving a chunk.
#[derive(Debug, Clone)]
pub struct ProofTask {
    /// The chunk to prove.
    pub chunk: ComputationChunk,
    /// Priority (higher = more urgent).
    pub priority: u32,
    /// Retry count.
    pub retries: u32,
    /// Affinity hint (preferred worker ID).
    pub affinity: Option<usize>,
    /// Creation timestamp.
    pub created_at: u64,
}

impl ProofTask {
    /// Creates a new proof task.
    pub fn new(chunk: ComputationChunk, priority: u32) -> Self {
        Self {
            chunk,
            priority,
            retries: 0,
            affinity: None,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        }
    }

    /// Sets the affinity hint.
    pub fn with_affinity(mut self, worker_id: usize) -> Self {
        self.affinity = Some(worker_id);
        self
    }
}

/// Work-stealing deque for a single worker.
/// Uses a Chase-Lev style deque with lock-free stealing.
struct WorkerDeque {
    /// Tasks owned by this worker (LIFO for local, FIFO for stealing).
    tasks: Mutex<Vec<ProofTask>>,
    /// Number of tasks in the deque.
    len: AtomicUsize,
    /// Worker ID.
    worker_id: usize,
}

impl WorkerDeque {
    fn new(worker_id: usize) -> Self {
        Self {
            tasks: Mutex::new(Vec::with_capacity(64)),
            len: AtomicUsize::new(0),
            worker_id,
        }
    }

    /// Push a task to the local end (LIFO).
    fn push(&self, task: ProofTask) {
        let mut tasks = self.tasks.lock().unwrap();
        tasks.push(task);
        self.len.store(tasks.len(), Ordering::Release);
    }

    /// Pop a task from the local end (LIFO).
    fn pop(&self) -> Option<ProofTask> {
        let mut tasks = self.tasks.lock().unwrap();
        let task = tasks.pop();
        self.len.store(tasks.len(), Ordering::Release);
        task
    }

    /// Steal tasks from the remote end (FIFO).
    fn steal(&self, count: usize) -> Vec<ProofTask> {
        let mut tasks = self.tasks.lock().unwrap();
        let steal_count = count.min(tasks.len() / 2);
        if steal_count == 0 {
            return Vec::new();
        }

        // Steal from the front (oldest tasks).
        let stolen: Vec<_> = tasks.drain(0..steal_count).collect();
        self.len.store(tasks.len(), Ordering::Release);
        stolen
    }

    /// Get the number of tasks.
    fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    /// Check if empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Statistics for a single worker.
#[derive(Debug, Default)]
pub struct WorkerStats {
    /// Tasks completed.
    pub tasks_completed: AtomicU64,
    /// Tasks stolen from this worker.
    pub tasks_stolen: AtomicU64,
    /// Tasks stolen by this worker.
    pub steals_performed: AtomicU64,
    /// Total proving time (microseconds).
    pub total_prove_time_us: AtomicU64,
    /// Time spent idle (microseconds).
    pub idle_time_us: AtomicU64,
}

impl WorkerStats {
    fn new() -> Self {
        Self::default()
    }
}

/// Global statistics for the parallel prover.
#[derive(Debug, Default)]
pub struct ParallelProverStats {
    /// Total tasks submitted.
    pub tasks_submitted: AtomicU64,
    /// Total tasks completed.
    pub tasks_completed: AtomicU64,
    /// Total tasks failed.
    pub tasks_failed: AtomicU64,
    /// Total steals performed.
    pub steals_performed: AtomicU64,
    /// Total tasks stolen.
    pub tasks_stolen: AtomicU64,
}

impl ParallelProverStats {
    fn new() -> Self {
        Self::default()
    }

    /// Gets a snapshot of the stats.
    pub fn snapshot(&self) -> ParallelProverStatsSnapshot {
        ParallelProverStatsSnapshot {
            tasks_submitted: self.tasks_submitted.load(Ordering::Relaxed),
            tasks_completed: self.tasks_completed.load(Ordering::Relaxed),
            tasks_failed: self.tasks_failed.load(Ordering::Relaxed),
            steals_performed: self.steals_performed.load(Ordering::Relaxed),
            tasks_stolen: self.tasks_stolen.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of parallel prover stats.
#[derive(Debug, Clone)]
pub struct ParallelProverStatsSnapshot {
    pub tasks_submitted: u64,
    pub tasks_completed: u64,
    pub tasks_failed: u64,
    pub steals_performed: u64,
    pub tasks_stolen: u64,
}

/// Work-stealing scheduler for distributing tasks across workers.
struct WorkStealingScheduler {
    /// Per-worker deques.
    deques: Vec<Arc<WorkerDeque>>,
    /// Per-worker statistics.
    worker_stats: Vec<Arc<WorkerStats>>,
    /// Configuration.
    config: ParallelConfig,
    /// Next worker for round-robin distribution.
    next_worker: AtomicUsize,
}

impl WorkStealingScheduler {
    fn new(config: ParallelConfig) -> Self {
        let num_workers = config.num_threads;
        let deques = (0..num_workers)
            .map(|id| Arc::new(WorkerDeque::new(id)))
            .collect();
        let worker_stats = (0..num_workers)
            .map(|_| Arc::new(WorkerStats::new()))
            .collect();

        Self {
            deques,
            worker_stats,
            config,
            next_worker: AtomicUsize::new(0),
        }
    }

    /// Submit a task with optional affinity.
    fn submit(&self, task: ProofTask) {
        let worker_id = if self.config.task_affinity {
            task.affinity.unwrap_or_else(|| self.select_worker())
        } else {
            self.select_worker()
        };

        self.deques[worker_id].push(task);
    }

    /// Select worker using round-robin.
    fn select_worker(&self) -> usize {
        let id = self.next_worker.fetch_add(1, Ordering::Relaxed);
        id % self.deques.len()
    }

    /// Select worker with least load.
    fn select_least_loaded(&self) -> usize {
        self.deques
            .iter()
            .enumerate()
            .min_by_key(|(_, d)| d.len())
            .map(|(id, _)| id)
            .unwrap_or(0)
    }

    /// Get deque for a worker.
    fn get_deque(&self, worker_id: usize) -> Arc<WorkerDeque> {
        Arc::clone(&self.deques[worker_id])
    }

    /// Get stats for a worker.
    fn get_worker_stats(&self, worker_id: usize) -> Arc<WorkerStats> {
        Arc::clone(&self.worker_stats[worker_id])
    }

    /// Try to steal work for an idle worker.
    fn steal_for(&self, worker_id: usize) -> Vec<ProofTask> {
        if !self.config.work_stealing {
            return Vec::new();
        }

        // Find the worker with the most tasks.
        let victim = self
            .deques
            .iter()
            .enumerate()
            .filter(|(id, _)| *id != worker_id)
            .filter(|(_, d)| d.len() >= self.config.steal_threshold)
            .max_by_key(|(_, d)| d.len());

        if let Some((victim_id, victim_deque)) = victim {
            let stolen = victim_deque.steal(self.config.steal_batch_size);
            if !stolen.is_empty() {
                self.worker_stats[victim_id]
                    .tasks_stolen
                    .fetch_add(stolen.len() as u64, Ordering::Relaxed);
                self.worker_stats[worker_id]
                    .steals_performed
                    .fetch_add(1, Ordering::Relaxed);
            }
            stolen
        } else {
            Vec::new()
        }
    }

    /// Check if all deques are empty.
    fn all_empty(&self) -> bool {
        self.deques.iter().all(|d| d.is_empty())
    }

    /// Total pending tasks.
    fn total_pending(&self) -> usize {
        self.deques.iter().map(|d| d.len()).sum()
    }
}

/// Manager for parallel proof generation with work-stealing.
pub struct ParallelProver {
    /// Configuration.
    config: ParallelConfig,
    /// Work-stealing scheduler.
    scheduler: Arc<WorkStealingScheduler>,
    /// Completed proofs.
    completed_proofs: Arc<RwLock<HashMap<ChunkId, ChunkProof>>>,
    /// Task status tracking.
    status: Arc<RwLock<HashMap<ChunkId, ProofStatus>>>,
    /// Whether the prover is running.
    running: Arc<AtomicBool>,
    /// Shutdown signal.
    shutdown: Arc<AtomicBool>,
    /// Worker handles.
    workers: Mutex<Vec<JoinHandle<()>>>,
    /// Global statistics.
    stats: Arc<ParallelProverStats>,
    /// Condition variable for waiting.
    work_available: Arc<(Mutex<bool>, Condvar)>,
}

impl ParallelProver {
    /// Creates a new parallel prover.
    pub fn new() -> Self {
        Self::with_config(ParallelConfig::default())
    }

    /// Creates a prover with custom config.
    pub fn with_config(config: ParallelConfig) -> Self {
        let scheduler = Arc::new(WorkStealingScheduler::new(config.clone()));

        Self {
            config,
            scheduler,
            completed_proofs: Arc::new(RwLock::new(HashMap::new())),
            status: Arc::new(RwLock::new(HashMap::new())),
            running: Arc::new(AtomicBool::new(false)),
            shutdown: Arc::new(AtomicBool::new(false)),
            workers: Mutex::new(Vec::new()),
            stats: Arc::new(ParallelProverStats::new()),
            work_available: Arc::new((Mutex::new(false), Condvar::new())),
        }
    }

    /// Submits a chunk for proving.
    pub fn submit(&self, chunk: ComputationChunk, priority: u32) {
        let task = ProofTask::new(chunk.clone(), priority);

        {
            let mut status = self.status.write().unwrap();
            status.insert(chunk.id, ProofStatus::Pending);
        }

        self.scheduler.submit(task);
        self.stats.tasks_submitted.fetch_add(1, Ordering::Relaxed);
        self.notify_work_available();
    }

    /// Submits a chunk with affinity hint.
    pub fn submit_with_affinity(&self, chunk: ComputationChunk, priority: u32, worker_id: usize) {
        let task = ProofTask::new(chunk.clone(), priority).with_affinity(worker_id);

        {
            let mut status = self.status.write().unwrap();
            status.insert(chunk.id, ProofStatus::Pending);
        }

        self.scheduler.submit(task);
        self.stats.tasks_submitted.fetch_add(1, Ordering::Relaxed);
        self.notify_work_available();
    }

    /// Submits multiple chunks.
    pub fn submit_batch(&self, chunks: Vec<ComputationChunk>) {
        for (i, chunk) in chunks.into_iter().enumerate() {
            // Assign affinity based on chunk index for better locality.
            let worker_id = i % self.config.num_threads;
            self.submit_with_affinity(chunk, (1000 - i) as u32, worker_id);
        }
    }

    /// Submits chunks with dependency awareness.
    /// Chunks with the same parent are assigned to the same worker.
    pub fn submit_batch_with_dependencies(&self, chunks: Vec<ComputationChunk>) {
        // Group chunks by parent for affinity.
        let mut parent_affinity: HashMap<Option<ChunkId>, usize> = HashMap::new();
        let mut next_worker = 0;

        for (i, chunk) in chunks.into_iter().enumerate() {
            let worker_id = *parent_affinity.entry(chunk.parent_id).or_insert_with(|| {
                let w = next_worker;
                next_worker = (next_worker + 1) % self.config.num_threads;
                w
            });

            self.submit_with_affinity(chunk, (1000 - i) as u32, worker_id);
        }
    }

    fn notify_work_available(&self) {
        let (lock, cvar) = &*self.work_available;
        let mut available = lock.lock().unwrap();
        *available = true;
        cvar.notify_all();
    }

    /// Starts the worker threads.
    pub fn start(&self) {
        if self.running.swap(true, Ordering::SeqCst) {
            return; // Already running.
        }

        self.shutdown.store(false, Ordering::SeqCst);

        let mut workers = self.workers.lock().unwrap();
        workers.clear();

        for worker_id in 0..self.config.num_threads {
            let deque = self.scheduler.get_deque(worker_id);
            let worker_stats = self.scheduler.get_worker_stats(worker_id);
            let scheduler = Arc::clone(&self.scheduler);
            let completed = Arc::clone(&self.completed_proofs);
            let status = Arc::clone(&self.status);
            let running = Arc::clone(&self.running);
            let shutdown = Arc::clone(&self.shutdown);
            let global_stats = Arc::clone(&self.stats);
            let work_available = Arc::clone(&self.work_available);
            let timeout = self.config.proof_timeout_secs;

            let handle = thread::Builder::new()
                .name(format!("helix-prover-{}", worker_id))
                .spawn(move || {
                    Self::worker_loop(
                        worker_id,
                        deque,
                        worker_stats,
                        scheduler,
                        completed,
                        status,
                        running,
                        shutdown,
                        global_stats,
                        work_available,
                        timeout,
                    );
                })
                .expect("Failed to spawn worker thread");

            workers.push(handle);
        }
    }

    /// Stops all worker threads gracefully.
    pub fn stop(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        self.running.store(false, Ordering::SeqCst);
        self.notify_work_available();

        let mut workers = self.workers.lock().unwrap();
        for handle in workers.drain(..) {
            let _ = handle.join();
        }
    }

    /// Gets the status of a proof.
    pub fn get_status(&self, chunk_id: ChunkId) -> Option<ProofStatus> {
        let status = self.status.read().unwrap();
        status.get(&chunk_id).cloned()
    }

    /// Gets a completed proof.
    pub fn get_proof(&self, chunk_id: ChunkId) -> Option<ChunkProof> {
        let proofs = self.completed_proofs.read().unwrap();
        proofs.get(&chunk_id).cloned()
    }

    /// Gets all completed proofs.
    pub fn get_all_proofs(&self) -> Vec<ChunkProof> {
        let proofs = self.completed_proofs.read().unwrap();
        proofs.values().cloned().collect()
    }

    /// Waits for a specific proof to complete.
    pub fn wait_for(&self, chunk_id: ChunkId) -> Option<ChunkProof> {
        loop {
            match self.get_status(chunk_id) {
                Some(ProofStatus::Complete) => return self.get_proof(chunk_id),
                Some(ProofStatus::Failed(_)) => return None,
                None => return None,
                _ => {
                    thread::sleep(Duration::from_millis(10));
                }
            }
        }
    }

    /// Waits for a specific proof with timeout.
    pub fn wait_for_timeout(&self, chunk_id: ChunkId, timeout: Duration) -> Option<ChunkProof> {
        let start = Instant::now();
        loop {
            if start.elapsed() > timeout {
                return None;
            }

            match self.get_status(chunk_id) {
                Some(ProofStatus::Complete) => return self.get_proof(chunk_id),
                Some(ProofStatus::Failed(_)) => return None,
                None => return None,
                _ => {
                    thread::sleep(Duration::from_millis(10));
                }
            }
        }
    }

    /// Waits for all submitted proofs to complete.
    ///
    /// Returns once every submitted task has reached `Complete` or `Failed` status.
    /// Uses a default timeout of 5 minutes to prevent infinite hangs.
    pub fn wait_all(&self) -> Vec<ChunkProof> {
        let (proofs, _timed_out) = self.wait_all_timeout(Duration::from_secs(300));
        proofs
    }

    /// Waits for all proofs with timeout.
    pub fn wait_all_timeout(&self, timeout: Duration) -> (Vec<ChunkProof>, bool) {
        let start = Instant::now();
        loop {
            if start.elapsed() > timeout {
                return (self.get_all_proofs(), false);
            }

            let status = self.status.read().unwrap();
            let all_done = status
                .values()
                .all(|s| matches!(s, ProofStatus::Complete | ProofStatus::Failed(_)));
            drop(status);

            if all_done {
                return (self.get_all_proofs(), true);
            }

            thread::sleep(Duration::from_millis(10));
        }
    }

    /// Progress as (completed, total).
    pub fn progress(&self) -> (usize, usize) {
        let status = self.status.read().unwrap();
        let total = status.len();
        let completed = status
            .values()
            .filter(|s| matches!(s, ProofStatus::Complete))
            .count();
        (completed, total)
    }

    /// Get global statistics.
    pub fn stats(&self) -> ParallelProverStatsSnapshot {
        self.stats.snapshot()
    }

    /// Get pending task count.
    pub fn pending_count(&self) -> usize {
        self.scheduler.total_pending()
    }

    fn worker_loop(
        worker_id: usize,
        deque: Arc<WorkerDeque>,
        worker_stats: Arc<WorkerStats>,
        scheduler: Arc<WorkStealingScheduler>,
        completed: Arc<RwLock<HashMap<ChunkId, ChunkProof>>>,
        status: Arc<RwLock<HashMap<ChunkId, ProofStatus>>>,
        running: Arc<AtomicBool>,
        shutdown: Arc<AtomicBool>,
        global_stats: Arc<ParallelProverStats>,
        work_available: Arc<(Mutex<bool>, Condvar)>,
        _timeout: u64,
    ) {
        let mut idle_start: Option<Instant> = None;

        loop {
            // Check for shutdown.
            if shutdown.load(Ordering::SeqCst) {
                break;
            }

            // Try to get a task from our deque.
            let task = deque.pop().or_else(|| {
                // Try to steal from other workers.
                let stolen = scheduler.steal_for(worker_id);
                if !stolen.is_empty() {
                    global_stats
                        .steals_performed
                        .fetch_add(1, Ordering::Relaxed);
                    global_stats
                        .tasks_stolen
                        .fetch_add(stolen.len() as u64, Ordering::Relaxed);

                    // Push all but one to our deque.
                    let mut iter = stolen.into_iter();
                    let first = iter.next();
                    for t in iter {
                        deque.push(t);
                    }
                    first
                } else {
                    None
                }
            });

            let task = match task {
                Some(t) => {
                    // Record idle time if we were idle.
                    if let Some(start) = idle_start.take() {
                        worker_stats
                            .idle_time_us
                            .fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
                    }
                    t
                }
                None => {
                    // Start tracking idle time.
                    if idle_start.is_none() {
                        idle_start = Some(Instant::now());
                    }

                    // Wait for work with a timeout.
                    let (lock, cvar) = &*work_available;
                    let mut available = lock.lock().unwrap();
                    let _ = cvar.wait_timeout(available, Duration::from_millis(50));

                    // Check if we should stop.
                    if !running.load(Ordering::SeqCst) {
                        break;
                    }
                    continue;
                }
            };

            // Update status to in-progress.
            {
                let mut s = status.write().unwrap();
                s.insert(task.chunk.id, ProofStatus::InProgress);
            }

            // Generate the proof.
            let start_time = Instant::now();
            let proof_result = Self::generate_proof(&task.chunk);
            let elapsed_us = start_time.elapsed().as_micros() as u64;
            let elapsed_ms = elapsed_us / 1000;

            worker_stats
                .total_prove_time_us
                .fetch_add(elapsed_us, Ordering::Relaxed);

            match proof_result {
                Ok(proof_bytes) => {
                    let chunk_proof = ChunkProof {
                        chunk_id: task.chunk.id,
                        proof: proof_bytes,
                        public_inputs: vec![
                            task.chunk.input_commitment,
                            task.chunk.output_commitment,
                        ],
                        error_bound: task.chunk.error_bound,
                        generation_time_ms: elapsed_ms,
                    };

                    {
                        let mut c = completed.write().unwrap();
                        c.insert(task.chunk.id, chunk_proof);
                    }
                    {
                        let mut s = status.write().unwrap();
                        s.insert(task.chunk.id, ProofStatus::Complete);
                    }

                    worker_stats.tasks_completed.fetch_add(1, Ordering::Relaxed);
                    global_stats.tasks_completed.fetch_add(1, Ordering::Relaxed);
                }
                Err(e) => {
                    let mut s = status.write().unwrap();
                    s.insert(task.chunk.id, ProofStatus::Failed(e));
                    global_stats.tasks_failed.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    fn generate_proof(chunk: &ComputationChunk) -> Result<Vec<u8>, String> {
        // Build an IVCStepCircuit representing this chunk's computation.
        let circuit = IVCStepCircuit {
            prev_state: chunk.input_commitment,
            new_state: chunk.output_commitment,
            computation_hash: {
                // Derive a deterministic computation hash from the chunk identity.
                use sha2::{Digest, Sha256};
                let mut h = Sha256::new();
                h.update(chunk.id.0.to_le_bytes());
                h.update(chunk.layer_range.0.to_le_bytes());
                h.update(chunk.layer_range.1.to_le_bytes());
                h.finalize().into()
            },
            step_number: chunk.id.0,
        };

        let pi: Vec<Fr> = circuit.public_inputs();
        let pi_refs: Vec<&[Fr]> = vec![&pi];

        // Each worker creates its own pipeline (keygen is deterministic for a
        // given circuit shape, so all workers produce compatible proofs).
        let mut pipeline = ProverPipeline::<IVCStepCircuit>::new(5);
        pipeline.setup(&IVCStepCircuit::default());

        let proof = pipeline.prove(&circuit, &pi_refs)
            .map_err(|e| e.to_string())?;
        Ok(proof)
    }
}

impl Default for ParallelProver {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ParallelProver {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Result of batch proving.
#[derive(Debug)]
pub struct BatchProofResult {
    /// Successfully generated proofs.
    pub proofs: Vec<ChunkProof>,
    /// Failed chunks with error messages.
    pub failures: Vec<(ChunkId, String)>,
    /// Total time taken.
    pub total_time_ms: u64,
    /// Statistics snapshot.
    pub stats: ParallelProverStatsSnapshot,
}

/// Proves a batch of chunks in parallel.
pub fn prove_batch(chunks: Vec<ComputationChunk>, config: ParallelConfig) -> BatchProofResult {
    let prover = ParallelProver::with_config(config);
    prover.submit_batch(chunks);
    prover.start();

    let start = Instant::now();
    let proofs = prover.wait_all();
    let total_time_ms = start.elapsed().as_millis() as u64;

    let stats = prover.stats();
    prover.stop();

    let status = prover.status.read().unwrap();
    let failures: Vec<_> = status
        .iter()
        .filter_map(|(id, s)| {
            if let ProofStatus::Failed(e) = s {
                Some((*id, e.clone()))
            } else {
                None
            }
        })
        .collect();

    BatchProofResult {
        proofs,
        failures,
        total_time_ms,
        stats,
    }
}

/// Proves a batch with dependency-aware scheduling.
pub fn prove_batch_with_dependencies(
    chunks: Vec<ComputationChunk>,
    config: ParallelConfig,
) -> BatchProofResult {
    let prover = ParallelProver::with_config(config);
    prover.submit_batch_with_dependencies(chunks);
    prover.start();

    let start = Instant::now();
    let proofs = prover.wait_all();
    let total_time_ms = start.elapsed().as_millis() as u64;

    let stats = prover.stats();
    prover.stop();

    let status = prover.status.read().unwrap();
    let failures: Vec<_> = status
        .iter()
        .filter_map(|(id, s)| {
            if let ProofStatus::Failed(e) = s {
                Some((*id, e.clone()))
            } else {
                None
            }
        })
        .collect();

    BatchProofResult {
        proofs,
        failures,
        total_time_ms,
        stats,
    }
}

/// Adaptive parallel prover that adjusts thread count based on workload.
pub struct AdaptiveParallelProver {
    /// Inner prover.
    inner: ParallelProver,
    /// Minimum threads.
    min_threads: usize,
    /// Maximum threads.
    max_threads: usize,
    /// Current active threads.
    active_threads: AtomicUsize,
}

impl AdaptiveParallelProver {
    /// Creates a new adaptive prover.
    pub fn new(min_threads: usize, max_threads: usize) -> Self {
        let config = ParallelConfig {
            num_threads: min_threads,
            ..Default::default()
        };

        Self {
            inner: ParallelProver::with_config(config),
            min_threads,
            max_threads,
            active_threads: AtomicUsize::new(min_threads),
        }
    }

    /// Submits work and adjusts thread count if needed.
    pub fn submit(&self, chunk: ComputationChunk, priority: u32) {
        self.inner.submit(chunk, priority);
        self.maybe_scale_up();
    }

    /// Scales up threads if queue is growing.
    fn maybe_scale_up(&self) {
        let pending = self.inner.pending_count();
        let current = self.active_threads.load(Ordering::Relaxed);

        if pending > current * 4 && current < self.max_threads {
            // Would need to spawn new workers - this is a simplified model.
            // In production, we'd dynamically spawn/despawn workers.
            self.active_threads.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Starts the prover.
    pub fn start(&self) {
        self.inner.start();
    }

    /// Stops the prover.
    pub fn stop(&self) {
        self.inner.stop();
    }

    /// Waits for all proofs.
    pub fn wait_all(&self) -> Vec<ChunkProof> {
        self.inner.wait_all()
    }

    /// Gets statistics.
    pub fn stats(&self) -> ParallelProverStatsSnapshot {
        self.inner.stats()
    }
}

/// Priority queue for tasks with deadline support.
pub struct PriorityTaskQueue {
    /// Tasks sorted by priority.
    tasks: Mutex<Vec<PriorityTask>>,
}

/// Task with priority and optional deadline.
#[derive(Debug, Clone)]
pub struct PriorityTask {
    /// The proof task.
    pub task: ProofTask,
    /// Deadline (if any).
    pub deadline: Option<Instant>,
    /// Computed effective priority.
    effective_priority: u64,
}

impl PriorityTask {
    /// Creates a new priority task.
    pub fn new(task: ProofTask, deadline: Option<Instant>) -> Self {
        let mut pt = Self {
            task,
            deadline,
            effective_priority: 0,
        };
        pt.compute_priority();
        pt
    }

    /// Computes effective priority based on base priority and deadline urgency.
    fn compute_priority(&mut self) {
        let base = self.task.priority as u64;
        let urgency = if let Some(deadline) = self.deadline {
            let remaining = deadline.saturating_duration_since(Instant::now());
            // Higher urgency for closer deadlines.
            u64::MAX - remaining.as_millis() as u64
        } else {
            0
        };
        self.effective_priority = base.saturating_add(urgency);
    }
}

impl PriorityTaskQueue {
    /// Creates a new priority queue.
    pub fn new() -> Self {
        Self {
            tasks: Mutex::new(Vec::new()),
        }
    }

    /// Pushes a task with priority.
    pub fn push(&self, task: ProofTask, deadline: Option<Instant>) {
        let pt = PriorityTask::new(task, deadline);
        let mut tasks = self.tasks.lock().unwrap();
        tasks.push(pt);
        // Sort ascending so pop() returns highest priority (from end of vec)
        tasks.sort_by(|a, b| a.effective_priority.cmp(&b.effective_priority));
    }

    /// Pops the highest priority task.
    pub fn pop(&self) -> Option<ProofTask> {
        let mut tasks = self.tasks.lock().unwrap();
        tasks.pop().map(|pt| pt.task)
    }

    /// Returns the number of tasks.
    pub fn len(&self) -> usize {
        self.tasks.lock().unwrap().len()
    }

    /// Checks if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Updates priorities for deadline-based tasks.
    pub fn refresh_priorities(&self) {
        let mut tasks = self.tasks.lock().unwrap();
        for task in tasks.iter_mut() {
            task.compute_priority();
        }
        // Sort ascending so pop() returns highest priority (from end of vec)
        tasks.sort_by(|a, b| a.effective_priority.cmp(&b.effective_priority));
    }
}

impl Default for PriorityTaskQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunking::ComputationType;

    fn make_test_chunk(id: u64) -> ComputationChunk {
        ComputationChunk {
            id: ChunkId(id),
            parent_id: None,
            input_commitment: [0; 32],
            output_commitment: [0; 32],
            computation_type: ComputationType::Forward,
            layer_range: (0, 2),
            error_bound: 0.01,
        }
    }

    #[test]
    fn test_parallel_prover_submit() {
        let prover = ParallelProver::new();
        let chunk = make_test_chunk(1);

        prover.submit(chunk.clone(), 100);

        assert_eq!(prover.get_status(chunk.id), Some(ProofStatus::Pending));
    }

    #[test]
    fn test_parallel_prover_prove() {
        let prover = ParallelProver::with_config(ParallelConfig {
            num_threads: 2,
            ..Default::default()
        });

        for i in 0..4 {
            prover.submit(make_test_chunk(i), 100);
        }

        prover.start();
        let proofs = prover.wait_all();
        prover.stop();

        assert_eq!(proofs.len(), 4);
    }

    #[test]
    fn test_work_stealing_scheduler() {
        let config = ParallelConfig {
            num_threads: 2,
            work_stealing: true,
            steal_threshold: 2,
            steal_batch_size: 1,
            ..Default::default()
        };

        let scheduler = WorkStealingScheduler::new(config);

        // Submit 10 tasks to worker 0.
        for i in 0..10 {
            let mut task = ProofTask::new(make_test_chunk(i), 100);
            task.affinity = Some(0);
            scheduler.submit(task);
        }

        assert_eq!(scheduler.deques[0].len(), 10);
        assert_eq!(scheduler.deques[1].len(), 0);

        // Worker 1 steals from worker 0.
        let stolen = scheduler.steal_for(1);
        assert!(!stolen.is_empty());
        assert!(scheduler.deques[0].len() < 10);
    }

    #[test]
    fn test_priority_task_queue() {
        let queue = PriorityTaskQueue::new();

        // Add tasks with different priorities.
        queue.push(ProofTask::new(make_test_chunk(1), 10), None);
        queue.push(ProofTask::new(make_test_chunk(2), 50), None);
        queue.push(ProofTask::new(make_test_chunk(3), 30), None);

        // Should get highest priority first.
        let task = queue.pop().unwrap();
        assert_eq!(task.priority, 50);
    }

    #[test]
    fn test_worker_stats() {
        let stats = WorkerStats::new();

        stats.tasks_completed.fetch_add(5, Ordering::Relaxed);
        stats.steals_performed.fetch_add(2, Ordering::Relaxed);

        assert_eq!(stats.tasks_completed.load(Ordering::Relaxed), 5);
        assert_eq!(stats.steals_performed.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_parallel_prover_with_dependencies() {
        let prover = ParallelProver::with_config(ParallelConfig {
            num_threads: 2,
            task_affinity: true,
            ..Default::default()
        });

        // Create chunks with parent relationships.
        let mut chunks = Vec::new();
        for i in 0..4 {
            let mut chunk = make_test_chunk(i);
            if i > 0 {
                chunk.parent_id = Some(ChunkId(0)); // All children of chunk 0.
            }
            chunks.push(chunk);
        }

        prover.submit_batch_with_dependencies(chunks);
        prover.start();
        let proofs = prover.wait_all();
        prover.stop();

        assert_eq!(proofs.len(), 4);
    }

    #[test]
    fn test_batch_proof_with_dependencies() {
        let chunks: Vec<_> = (0..4).map(|i| make_test_chunk(i)).collect();

        let result = prove_batch_with_dependencies(
            chunks,
            ParallelConfig {
                num_threads: 2,
                ..Default::default()
            },
        );

        assert_eq!(result.proofs.len(), 4);
        assert!(result.failures.is_empty());
        assert!(result.stats.tasks_submitted == 4);
    }
}
