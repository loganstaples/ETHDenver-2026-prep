//! Model Registry Module.
//!
//! Provides a centralized registry for managing ML models,
//! their versions, and metadata.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;

/// Unique model identifier.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelId(pub String);

impl ModelId {
    /// Creates a new model ID.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Generates a random model ID.
    pub fn random() -> Self {
        Self(format!("model_{}", uuid_simple()))
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Model version.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ModelVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl ModelVersion {
    /// Creates a new version.
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self { major, minor, patch }
    }

    /// Parses a version string (e.g., "1.2.3").
    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return None;
        }
        Some(Self {
            major: parts[0].parse().ok()?,
            minor: parts[1].parse().ok()?,
            patch: parts[2].parse().ok()?,
        })
    }

    /// Increments the patch version.
    pub fn bump_patch(&mut self) {
        self.patch += 1;
    }

    /// Increments the minor version.
    pub fn bump_minor(&mut self) {
        self.minor += 1;
        self.patch = 0;
    }

    /// Increments the major version.
    pub fn bump_major(&mut self) {
        self.major += 1;
        self.minor = 0;
        self.patch = 0;
    }
}

impl std::fmt::Display for ModelVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Model status.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ModelStatus {
    /// Model is being trained.
    Training,
    /// Model is ready for use.
    Ready,
    /// Model is deprecated.
    Deprecated,
    /// Model is archived.
    Archived,
}

/// Model entry in the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Model ID.
    pub id: ModelId,
    /// Model name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Current version.
    pub version: ModelVersion,
    /// Model status.
    pub status: ModelStatus,
    /// Architecture type.
    pub architecture: String,
    /// Storage location (IPFS CID, URL, etc).
    pub storage_uri: String,
    /// Model hash.
    pub hash: [u8; 32],
    /// Total parameters.
    pub num_params: usize,
    /// Accuracy metrics.
    pub metrics: HashMap<String, f64>,
    /// Training round (for federated learning).
    pub training_round: u64,
    /// Error bound.
    pub error_bound: f64,
    /// Creation timestamp.
    pub created_at: u64,
    /// Last updated timestamp.
    pub updated_at: u64,
    /// Owner/creator.
    pub owner: String,
    /// Tags.
    pub tags: Vec<String>,
}

impl ModelEntry {
    /// Creates a new model entry.
    pub fn new(
        name: String,
        description: String,
        architecture: String,
        storage_uri: String,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            id: ModelId::random(),
            name,
            description,
            version: ModelVersion::new(0, 1, 0),
            status: ModelStatus::Training,
            architecture,
            storage_uri,
            hash: [0; 32],
            num_params: 0,
            metrics: HashMap::new(),
            training_round: 0,
            error_bound: 0.0,
            created_at: now,
            updated_at: now,
            owner: String::new(),
            tags: Vec::new(),
        }
    }

    /// Sets the model status to ready.
    pub fn mark_ready(&mut self) {
        self.status = ModelStatus::Ready;
        self.touch();
    }

    /// Deprecates the model.
    pub fn deprecate(&mut self) {
        self.status = ModelStatus::Deprecated;
        self.touch();
    }

    /// Updates the timestamp.
    fn touch(&mut self) {
        self.updated_at = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }

    /// Adds a metric.
    pub fn add_metric(&mut self, name: String, value: f64) {
        self.metrics.insert(name, value);
        self.touch();
    }

    /// Adds a tag.
    pub fn add_tag(&mut self, tag: String) {
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }
}

/// Version history entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionHistoryEntry {
    /// Version.
    pub version: ModelVersion,
    /// Storage URI.
    pub storage_uri: String,
    /// Hash.
    pub hash: [u8; 32],
    /// Timestamp.
    pub timestamp: u64,
    /// Change notes.
    pub notes: String,
}

/// Model registry for managing models.
pub struct ModelRegistry {
    /// Registered models by ID.
    models: HashMap<ModelId, ModelEntry>,
    /// Model name to ID mapping.
    name_index: HashMap<String, ModelId>,
    /// Version history.
    history: HashMap<ModelId, Vec<VersionHistoryEntry>>,
    /// Tag index.
    tag_index: HashMap<String, Vec<ModelId>>,
}

impl ModelRegistry {
    /// Creates a new empty registry.
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            name_index: HashMap::new(),
            history: HashMap::new(),
            tag_index: HashMap::new(),
        }
    }

    /// Registers a new model.
    pub fn register(&mut self, entry: ModelEntry) -> ModelId {
        let id = entry.id.clone();
        let name = entry.name.clone();

        // Update tag index
        for tag in &entry.tags {
            self.tag_index.entry(tag.clone()).or_default().push(id.clone());
        }

        self.name_index.insert(name, id.clone());
        self.models.insert(id.clone(), entry);
        
        id
    }

    /// Gets a model by ID.
    pub fn get(&self, id: &ModelId) -> Option<&ModelEntry> {
        self.models.get(id)
    }

    /// Gets a model by name.
    pub fn get_by_name(&self, name: &str) -> Option<&ModelEntry> {
        let id = self.name_index.get(name)?;
        self.models.get(id)
    }

    /// Updates a model.
    pub fn update(&mut self, id: &ModelId, update_fn: impl FnOnce(&mut ModelEntry)) -> bool {
        if let Some(entry) = self.models.get_mut(id) {
            update_fn(entry);
            entry.touch();
            true
        } else {
            false
        }
    }

    /// Creates a new version of a model.
    pub fn create_version(
        &mut self,
        id: &ModelId,
        storage_uri: String,
        hash: [u8; 32],
        notes: String,
    ) -> Option<ModelVersion> {
        let entry = self.models.get_mut(id)?;
        
        // Save current version to history
        let history_entry = VersionHistoryEntry {
            version: entry.version.clone(),
            storage_uri: entry.storage_uri.clone(),
            hash: entry.hash,
            timestamp: entry.updated_at,
            notes: notes.clone(),
        };
        self.history.entry(id.clone()).or_default().push(history_entry);

        // Update to new version
        entry.version.bump_patch();
        entry.storage_uri = storage_uri;
        entry.hash = hash;
        entry.touch();

        Some(entry.version.clone())
    }

    /// Gets version history for a model.
    pub fn get_history(&self, id: &ModelId) -> Vec<&VersionHistoryEntry> {
        self.history.get(id).map(|v| v.iter().collect()).unwrap_or_default()
    }

    /// Lists all models.
    pub fn list(&self) -> Vec<&ModelEntry> {
        self.models.values().collect()
    }

    /// Lists models by status.
    pub fn list_by_status(&self, status: ModelStatus) -> Vec<&ModelEntry> {
        self.models.values().filter(|m| m.status == status).collect()
    }

    /// Lists models by tag.
    pub fn list_by_tag(&self, tag: &str) -> Vec<&ModelEntry> {
        self.tag_index
            .get(tag)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.models.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Searches models by name.
    pub fn search(&self, query: &str) -> Vec<&ModelEntry> {
        let query = query.to_lowercase();
        self.models
            .values()
            .filter(|m| {
                m.name.to_lowercase().contains(&query)
                    || m.description.to_lowercase().contains(&query)
            })
            .collect()
    }

    /// Deletes a model.
    pub fn delete(&mut self, id: &ModelId) -> bool {
        if let Some(entry) = self.models.remove(id) {
            self.name_index.remove(&entry.name);
            self.history.remove(id);
            
            for tag in &entry.tags {
                if let Some(ids) = self.tag_index.get_mut(tag) {
                    ids.retain(|i| i != id);
                }
            }
            
            true
        } else {
            false
        }
    }

    /// Returns the number of registered models.
    pub fn count(&self) -> usize {
        self.models.len()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// Helper function
fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{:x}", now)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_version() {
        let mut version = ModelVersion::new(1, 2, 3);
        assert_eq!(version.to_string(), "1.2.3");

        version.bump_patch();
        assert_eq!(version.to_string(), "1.2.4");

        version.bump_minor();
        assert_eq!(version.to_string(), "1.3.0");

        version.bump_major();
        assert_eq!(version.to_string(), "2.0.0");
    }

    #[test]
    fn test_model_registry() {
        let mut registry = ModelRegistry::new();

        let entry = ModelEntry::new(
            "gpt_mini".to_string(),
            "A small GPT model".to_string(),
            "transformer".to_string(),
            "ipfs://Qm...".to_string(),
        );

        let id = registry.register(entry);

        assert_eq!(registry.count(), 1);

        let model = registry.get(&id).unwrap();
        assert_eq!(model.name, "gpt_mini");
    }

    #[test]
    fn test_version_history() {
        let mut registry = ModelRegistry::new();

        let entry = ModelEntry::new(
            "test_model".to_string(),
            "Test".to_string(),
            "mlp".to_string(),
            "v1".to_string(),
        );

        let id = registry.register(entry);

        registry.create_version(&id, "v2".to_string(), [1; 32], "Updated".to_string());
        registry.create_version(&id, "v3".to_string(), [2; 32], "Fixed bug".to_string());

        let history = registry.get_history(&id);
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_search_models() {
        let mut registry = ModelRegistry::new();

        registry.register(ModelEntry::new(
            "gpt_large".to_string(),
            "Large language model".to_string(),
            "transformer".to_string(),
            "uri1".to_string(),
        ));

        registry.register(ModelEntry::new(
            "bert_base".to_string(),
            "BERT base model".to_string(),
            "transformer".to_string(),
            "uri2".to_string(),
        ));

        let results = registry.search("gpt");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "gpt_large");
    }
}
