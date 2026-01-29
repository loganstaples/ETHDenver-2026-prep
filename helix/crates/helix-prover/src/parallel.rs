//! Parallel Proof Generation.
//!
//! Generates multiple proofs in parallel across available CPU cores.
//! Supports thread pools and work-stealing for optimal performance.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;

use super::chunking::{ChunkId, ComputationChunk};
use serde::{Deserialize, Serialize};

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
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus(),
            work_stealing: true,
            queue_depth: 8,
            proof_timeout_secs: 300,
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
}

/// Manager for parallel proof generation.
pub struct ParallelProver {
    /// Configuration.
    config: ParallelConfig,
    /// Pending tasks.
    pending_tasks: Arc<Mutex<Vec<ProofTask>>>,
    /// Completed proofs.
    completed_proofs: Arc<Mutex<HashMap<ChunkId, ChunkProof>>>,
    /// Task status tracking.
    status: Arc<Mutex<HashMap<ChunkId, ProofStatus>>>,
    /// Whether the prover is running.
    running: Arc<Mutex<bool>>,
}

impl ParallelProver {
    /// Creates a new parallel prover.
    pub fn new() -> Self {
        Self::with_config(ParallelConfig::default())
    }

    /// Creates a prover with custom config.
    pub fn with_config(config: ParallelConfig) -> Self {
        Self {
            config,
            pending_tasks: Arc::new(Mutex::new(Vec::new())),
            completed_proofs: Arc::new(Mutex::new(HashMap::new())),
            status: Arc::new(Mutex::new(HashMap::new())),
            running: Arc::new(Mutex::new(false)),
        }
    }

    /// Submits a chunk for proving.
    pub fn submit(&self, chunk: ComputationChunk, priority: u32) {
        let task = ProofTask {
            chunk: chunk.clone(),
            priority,
            retries: 0,
        };

        {
            let mut status = self.status.lock().unwrap();
            status.insert(chunk.id, ProofStatus::Pending);
        }

        {
            let mut tasks = self.pending_tasks.lock().unwrap();
            tasks.push(task);
            // Sort by priority (descending)
            tasks.sort_by(|a, b| b.priority.cmp(&a.priority));
        }
    }

    /// Submits multiple chunks.
    pub fn submit_batch(&self, chunks: Vec<ComputationChunk>) {
        for (i, chunk) in chunks.into_iter().enumerate() {
            self.submit(chunk, (1000 - i) as u32);
        }
    }

    /// Starts the worker threads.
    pub fn start(&self) {
        let mut running = self.running.lock().unwrap();
        if *running {
            return;
        }
        *running = true;
        drop(running);

        for worker_id in 0..self.config.num_threads {
            let pending = Arc::clone(&self.pending_tasks);
            let completed = Arc::clone(&self.completed_proofs);
            let status = Arc::clone(&self.status);
            let running = Arc::clone(&self.running);

            thread::spawn(move || {
                Self::worker_loop(worker_id, pending, completed, status, running);
            });
        }
    }

    /// Stops all worker threads.
    pub fn stop(&self) {
        let mut running = self.running.lock().unwrap();
        *running = false;
    }

    /// Gets the status of a proof.
    pub fn get_status(&self, chunk_id: ChunkId) -> Option<ProofStatus> {
        let status = self.status.lock().unwrap();
        status.get(&chunk_id).cloned()
    }

    /// Gets a completed proof.
    pub fn get_proof(&self, chunk_id: ChunkId) -> Option<ChunkProof> {
        let proofs = self.completed_proofs.lock().unwrap();
        proofs.get(&chunk_id).cloned()
    }

    /// Gets all completed proofs.
    pub fn get_all_proofs(&self) -> Vec<ChunkProof> {
        let proofs = self.completed_proofs.lock().unwrap();
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
                    thread::sleep(std::time::Duration::from_millis(100));
                }
            }
        }
    }

    /// Waits for all submitted proofs to complete.
    pub fn wait_all(&self) -> Vec<ChunkProof> {
        loop {
            let status = self.status.lock().unwrap();
            let all_done = status.values().all(|s| {
                matches!(s, ProofStatus::Complete | ProofStatus::Failed(_))
            });
            drop(status);

            if all_done {
                return self.get_all_proofs();
            }

            thread::sleep(std::time::Duration::from_millis(100));
        }
    }

    /// Progress as (completed, total).
    pub fn progress(&self) -> (usize, usize) {
        let status = self.status.lock().unwrap();
        let total = status.len();
        let completed = status.values()
            .filter(|s| matches!(s, ProofStatus::Complete))
            .count();
        (completed, total)
    }

    fn worker_loop(
        worker_id: usize,
        pending: Arc<Mutex<Vec<ProofTask>>>,
        completed: Arc<Mutex<HashMap<ChunkId, ChunkProof>>>,
        status: Arc<Mutex<HashMap<ChunkId, ProofStatus>>>,
        running: Arc<Mutex<bool>>,
    ) {
        loop {
            // Check if still running
            {
                let r = running.lock().unwrap();
                if !*r {
                    break;
                }
            }

            // Try to get a task
            let task = {
                let mut tasks = pending.lock().unwrap();
                tasks.pop()
            };

            let task = match task {
                Some(t) => t,
                None => {
                    thread::sleep(std::time::Duration::from_millis(50));
                    continue;
                }
            };

            // Update status to in-progress
            {
                let mut s = status.lock().unwrap();
                s.insert(task.chunk.id, ProofStatus::InProgress);
            }

            // Generate the proof (placeholder for actual proving)
            let start_time = std::time::Instant::now();
            let proof_result = Self::generate_proof(&task.chunk);
            let elapsed_ms = start_time.elapsed().as_millis() as u64;

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
                        let mut c = completed.lock().unwrap();
                        c.insert(task.chunk.id, chunk_proof);
                    }
                    {
                        let mut s = status.lock().unwrap();
                        s.insert(task.chunk.id, ProofStatus::Complete);
                    }
                }
                Err(e) => {
                    let mut s = status.lock().unwrap();
                    s.insert(task.chunk.id, ProofStatus::Failed(e));
                }
            }
        }
    }

    fn generate_proof(chunk: &ComputationChunk) -> Result<Vec<u8>, String> {
        // Placeholder - actual implementation would call the circuit prover
        // For now, generate a mock proof
        let mock_proof = format!(
            "HELIX_PROOF:{:?}:layers({}-{})",
            chunk.computation_type,
            chunk.layer_range.0,
            chunk.layer_range.1
        );
        Ok(mock_proof.into_bytes())
    }
}

impl Default for ParallelProver {
    fn default() -> Self {
        Self::new()
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
}

/// Proves a batch of chunks in parallel.
pub fn prove_batch(chunks: Vec<ComputationChunk>, config: ParallelConfig) -> BatchProofResult {
    let prover = ParallelProver::with_config(config);
    prover.submit_batch(chunks);
    prover.start();
    
    let start = std::time::Instant::now();
    let proofs = prover.wait_all();
    let total_time_ms = start.elapsed().as_millis() as u64;
    
    prover.stop();

    let status = prover.status.lock().unwrap();
    let failures: Vec<_> = status.iter()
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
}
