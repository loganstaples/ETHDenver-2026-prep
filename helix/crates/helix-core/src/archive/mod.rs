//! Proof Archive for HELIX.
//!
//! Provides persistent storage and retrieval of ZK proofs
//! generated during training rounds.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Proof metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofMetadata {
    /// Proof ID.
    pub id: String,
    /// Proof type.
    pub proof_type: ProofType,
    /// Training round.
    pub round_id: u64,
    /// Timestamp.
    pub timestamp: u64,
    /// Proof size in bytes.
    pub size: usize,
    /// Model hash.
    pub model_hash: String,
    /// Error bound.
    pub error_bound: f64,
    /// Verification time in ms.
    pub verification_time_ms: u64,
    /// On-chain transaction hash (if submitted).
    pub tx_hash: Option<String>,
}

/// Proof types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProofType {
    Training,
    Aggregation,
    Gradient,
    Computation,
}

/// Archived proof.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchivedProof {
    /// Metadata.
    pub metadata: ProofMetadata,
    /// Proof data.
    pub data: Vec<u8>,
    /// Public inputs.
    pub public_inputs: Vec<u8>,
    /// Verification key hash.
    pub vk_hash: String,
}

/// Proof archive storage.
pub struct ProofArchive {
    proofs: HashMap<String, ArchivedProof>,
    index_by_round: HashMap<u64, Vec<String>>,
    index_by_type: HashMap<ProofType, Vec<String>>,
}

impl ProofArchive {
    /// Creates a new empty archive.
    pub fn new() -> Self {
        Self {
            proofs: HashMap::new(),
            index_by_round: HashMap::new(),
            index_by_type: HashMap::new(),
        }
    }

    /// Archives a proof.
    pub fn archive(&mut self, proof: ArchivedProof) {
        let id = proof.metadata.id.clone();
        let round = proof.metadata.round_id;
        let proof_type = proof.metadata.proof_type;

        // Add to main storage
        self.proofs.insert(id.clone(), proof);

        // Update indices
        self.index_by_round
            .entry(round)
            .or_default()
            .push(id.clone());

        self.index_by_type
            .entry(proof_type)
            .or_default()
            .push(id);
    }

    /// Retrieves a proof by ID.
    pub fn get(&self, id: &str) -> Option<&ArchivedProof> {
        self.proofs.get(id)
    }

    /// Gets all proofs for a round.
    pub fn get_by_round(&self, round_id: u64) -> Vec<&ArchivedProof> {
        self.index_by_round
            .get(&round_id)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.proofs.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Gets all proofs of a type.
    pub fn get_by_type(&self, proof_type: ProofType) -> Vec<&ArchivedProof> {
        self.index_by_type
            .get(&proof_type)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.proofs.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Gets total proof count.
    pub fn count(&self) -> usize {
        self.proofs.len()
    }

    /// Gets total archive size in bytes.
    pub fn total_size(&self) -> usize {
        self.proofs.values().map(|p| p.metadata.size).sum()
    }

    /// Gets archive statistics.
    pub fn stats(&self) -> ArchiveStats {
        let total_count = self.count();
        let total_size = self.total_size();
        
        let mut by_type = HashMap::new();
        for (proof_type, ids) in &self.index_by_type {
            by_type.insert(*proof_type, ids.len());
        }

        let avg_verification_time = if total_count > 0 {
            self.proofs.values()
                .map(|p| p.metadata.verification_time_ms)
                .sum::<u64>() / total_count as u64
        } else {
            0
        };

        ArchiveStats {
            total_count,
            total_size,
            by_type,
            avg_verification_time_ms: avg_verification_time,
        }
    }

    /// Exports archive to JSON.
    pub fn export_json(&self) -> Result<String, String> {
        let proofs: Vec<_> = self.proofs.values().collect();
        serde_json::to_string_pretty(&proofs)
            .map_err(|e| format!("Serialization error: {}", e))
    }

    /// Saves the archive to a JSON file.
    pub fn save_to_file(&self, path: &Path) -> std::io::Result<()> {
        let proofs: Vec<&ArchivedProof> = self.proofs.values().collect();
        let json = serde_json::to_string_pretty(&proofs)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    /// Loads the archive from a JSON file, rebuilding indices.
    pub fn load_from_file(path: &Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let proofs: Vec<ArchivedProof> = serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let mut archive = Self::new();
        for proof in proofs {
            archive.archive(proof);
        }
        Ok(archive)
    }

    /// Prunes old proofs, keeping only the most recent N per round.
    pub fn prune(&mut self, keep_per_round: usize) {
        for (_, ids) in self.index_by_round.iter_mut() {
            while ids.len() > keep_per_round {
                if let Some(id) = ids.pop() {
                    self.proofs.remove(&id);
                }
            }
        }
    }
}

impl Default for ProofArchive {
    fn default() -> Self {
        Self::new()
    }
}

/// Archive statistics.
#[derive(Debug, Clone)]
pub struct ArchiveStats {
    /// Total proof count.
    pub total_count: usize,
    /// Total size in bytes.
    pub total_size: usize,
    /// Count by type.
    pub by_type: HashMap<ProofType, usize>,
    /// Average verification time.
    pub avg_verification_time_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_proof(id: &str, round: u64, proof_type: ProofType) -> ArchivedProof {
        ArchivedProof {
            metadata: ProofMetadata {
                id: id.to_string(),
                proof_type,
                round_id: round,
                timestamp: 1234567890,
                size: 1024,
                model_hash: "0xabc123".to_string(),
                error_bound: 0.0001,
                verification_time_ms: 100,
                tx_hash: None,
            },
            data: vec![1, 2, 3, 4],
            public_inputs: vec![5, 6, 7, 8],
            vk_hash: "0xvk123".to_string(),
        }
    }

    #[test]
    fn test_archive_and_retrieve() {
        let mut archive = ProofArchive::new();
        
        let proof = create_test_proof("proof_1", 1, ProofType::Training);
        archive.archive(proof);
        
        assert!(archive.get("proof_1").is_some());
        assert_eq!(archive.count(), 1);
    }

    #[test]
    fn test_get_by_round() {
        let mut archive = ProofArchive::new();
        
        archive.archive(create_test_proof("p1", 1, ProofType::Training));
        archive.archive(create_test_proof("p2", 1, ProofType::Aggregation));
        archive.archive(create_test_proof("p3", 2, ProofType::Training));
        
        let round1_proofs = archive.get_by_round(1);
        assert_eq!(round1_proofs.len(), 2);
        
        let round2_proofs = archive.get_by_round(2);
        assert_eq!(round2_proofs.len(), 1);
    }

    #[test]
    fn test_get_by_type() {
        let mut archive = ProofArchive::new();
        
        archive.archive(create_test_proof("p1", 1, ProofType::Training));
        archive.archive(create_test_proof("p2", 2, ProofType::Training));
        archive.archive(create_test_proof("p3", 1, ProofType::Aggregation));
        
        let training_proofs = archive.get_by_type(ProofType::Training);
        assert_eq!(training_proofs.len(), 2);
    }

    #[test]
    fn test_stats() {
        let mut archive = ProofArchive::new();

        archive.archive(create_test_proof("p1", 1, ProofType::Training));
        archive.archive(create_test_proof("p2", 1, ProofType::Aggregation));

        let stats = archive.stats();
        assert_eq!(stats.total_count, 2);
        assert_eq!(stats.total_size, 2048);
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let mut archive = ProofArchive::new();
        archive.archive(create_test_proof("p1", 1, ProofType::Training));
        archive.archive(create_test_proof("p2", 1, ProofType::Aggregation));
        archive.archive(create_test_proof("p3", 2, ProofType::Gradient));

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("archive.json");

        archive.save_to_file(&path).unwrap();
        let loaded = ProofArchive::load_from_file(&path).unwrap();

        assert_eq!(loaded.count(), 3);
        assert!(loaded.get("p1").is_some());
        assert!(loaded.get("p2").is_some());
        assert!(loaded.get("p3").is_some());
        assert_eq!(loaded.get_by_round(1).len(), 2);
        assert_eq!(loaded.get_by_type(ProofType::Gradient).len(), 1);
    }
}
