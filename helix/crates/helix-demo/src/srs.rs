//! Shared SRS (Structured Reference String) management.
//!
//! Ensures all workers in the demo use the same trusted setup parameters.
//! The SRS is cached to disk so it only needs to be generated once.

use std::path::PathBuf;

use anyhow::{Context, Result};
use helix_circuits::params::setup::{
    ParameterProfile, init_global_cache, global_cache,
};

/// Default K parameter for the demo (2^14 = 16K rows).
/// This matches V2ProverConfig::default().k and is sufficient for demo MLP models.
pub const DEMO_K: u32 = 14;

/// Returns the default cache directory for SRS files.
pub fn srs_cache_dir() -> PathBuf {
    let base = std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
        .join(".cache");
    base.join("helix").join("srs")
}

/// Ensures the shared SRS is generated and cached.
///
/// Returns the path to the SRS cache directory. All workers will use the same
/// Halo2 ParamsKZG generated from this SRS specification.
///
/// The actual KZG parameters are generated on-the-fly by the prover pipeline
/// using `ParamsKZG::setup(k, OsRng)`, but this function ensures the SRS
/// metadata is cached and validated for consistency across workers.
pub fn ensure_shared_srs() -> Result<PathBuf> {
    let cache_dir = srs_cache_dir();
    std::fs::create_dir_all(&cache_dir)
        .with_context(|| format!("Failed to create SRS cache dir: {:?}", cache_dir))?;

    // Initialize global SRS cache with persistent storage
    let _ = init_global_cache(&cache_dir);

    // Generate/cache SRS for the demo K value
    let cache = global_cache();
    let srs = cache
        .get_or_generate(DEMO_K)
        .map_err(|e| anyhow::anyhow!("SRS generation failed: {}", e))?;

    srs.validate()
        .map_err(|e| anyhow::anyhow!("SRS validation failed: {}", e))?;

    // Also write a manifest file for external tools
    let manifest = SrsManifest {
        k: DEMO_K,
        profile: "Small".to_string(),
        num_g1_elements: 1 << DEMO_K,
        num_g2_elements: 2,
        content_hash: hex::encode(srs.metadata().content_hash),
    };

    let manifest_path = cache_dir.join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&manifest_path, manifest_json)?;

    Ok(cache_dir)
}

/// SRS manifest for external tooling.
#[derive(serde::Serialize, serde::Deserialize)]
struct SrsManifest {
    k: u32,
    profile: String,
    num_g1_elements: usize,
    num_g2_elements: usize,
    content_hash: String,
}

/// Pre-warms the SRS cache for common profiles.
#[allow(dead_code)]
pub fn prewarm_cache() -> Result<()> {
    let cache = global_cache();
    cache
        .prewarm(&[ParameterProfile::Small])
        .map_err(|e| anyhow::anyhow!("SRS prewarm failed: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensure_shared_srs() {
        let path = ensure_shared_srs().expect("SRS generation should succeed");
        assert!(path.exists());
        assert!(path.join("manifest.json").exists());
    }

    #[test]
    fn test_srs_manifest_roundtrip() {
        let path = ensure_shared_srs().unwrap();
        let manifest_path = path.join("manifest.json");
        let content = std::fs::read_to_string(&manifest_path).unwrap();
        let manifest: SrsManifest = serde_json::from_str(&content).unwrap();
        assert_eq!(manifest.k, DEMO_K);
        assert_eq!(manifest.num_g1_elements, 1 << DEMO_K);
    }
}
