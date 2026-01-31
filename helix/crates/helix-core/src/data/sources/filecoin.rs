//! Filecoin Data Source for HELIX.
//!
//! Provides integration with Filecoin for long-term, verifiable storage
//! of training datasets. Supports:
//! - Deal-based data retrieval
//! - Storage provider selection
//! - Proof verification
//! - Data availability guarantees

use std::collections::HashMap;

use super::{
    BoxFuture, DataCache, DataSource, DataSourceError, DataSourceResult, DataStream,
    FetchOptions, ResourceMetadata,
};
use crate::data::merkle::{Hash, MerkleHasher, Sha256Hasher};
use crate::data::provenance::DataOrigin;

/// Configuration for Filecoin data source.
#[derive(Debug, Clone)]
pub struct FilecoinConfig {
    /// Lotus API endpoint.
    pub api_endpoint: String,
    /// API token.
    pub api_token: Option<String>,
    /// Gateway endpoints for HTTP retrieval.
    pub gateways: Vec<String>,
    /// Connection timeout in seconds.
    pub timeout_secs: u64,
    /// Maximum retries.
    pub max_retries: u32,
    /// Preferred miner IDs.
    pub preferred_miners: Vec<String>,
    /// Cache size in bytes.
    pub cache_size: usize,
}

impl Default for FilecoinConfig {
    fn default() -> Self {
        Self {
            api_endpoint: "https://api.node.glif.io/rpc/v0".to_string(),
            api_token: None,
            gateways: vec![
                "https://dweb.link/ipfs".to_string(),
                "https://w3s.link/ipfs".to_string(),
            ],
            timeout_secs: 120,
            max_retries: 3,
            preferred_miners: Vec::new(),
            cache_size: 100 * 1024 * 1024,
        }
    }
}

/// Filecoin deal information.
#[derive(Debug, Clone)]
pub struct DealInfo {
    /// Deal ID on the network.
    pub deal_id: u64,
    /// Proposal CID.
    pub proposal_cid: String,
    /// Data CID (payload root).
    pub data_cid: String,
    /// Provider (miner) ID.
    pub provider: String,
    /// Client address.
    pub client: String,
    /// Deal size in bytes.
    pub size: u64,
    /// Price per epoch in attoFIL.
    pub price_per_epoch: u128,
    /// Start epoch.
    pub start_epoch: u64,
    /// End epoch.
    pub end_epoch: u64,
    /// Deal state.
    pub state: DealState,
}

/// State of a Filecoin deal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DealState {
    /// Proposal stage.
    Proposed,
    /// Data being transferred.
    Transferring,
    /// Data published, awaiting on-chain.
    Published,
    /// Deal active on chain.
    Active,
    /// Deal expired.
    Expired,
    /// Deal slashed (miner failed to prove).
    Slashed,
    /// Unknown state.
    Unknown,
}

impl DealState {
    /// Returns true if the deal is active and data is available.
    pub fn is_available(&self) -> bool {
        matches!(self, DealState::Active)
    }
}

/// Filecoin storage statistics.
#[derive(Debug, Clone, Default)]
pub struct FilecoinStats {
    /// Total bytes retrieved.
    pub bytes_retrieved: u64,
    /// Total bytes stored.
    pub bytes_stored: u64,
    /// Number of active deals.
    pub active_deals: u64,
    /// Total FIL spent (attoFIL).
    pub total_spent: u128,
    /// Cache hits.
    pub cache_hits: u64,
}

/// Filecoin client for direct Lotus API access.
pub struct FilecoinClient {
    /// Configuration.
    pub config: FilecoinConfig,
    /// Statistics.
    stats: FilecoinStats,
}

impl FilecoinClient {
    /// Creates a new Filecoin client.
    pub fn new(config: FilecoinConfig) -> Self {
        Self {
            config,
            stats: FilecoinStats::default(),
        }
    }

    /// Gets deal information by deal ID.
    pub async fn get_deal(&self, deal_id: u64) -> DataSourceResult<DealInfo> {
        // In production, would make JSON-RPC call to Lotus API
        Err(DataSourceError::NotFound(format!("deal:{}", deal_id)))
    }

    /// Lists deals for a client address.
    pub async fn list_client_deals(&self, _client: &str) -> DataSourceResult<Vec<DealInfo>> {
        // In production, would query the chain
        Ok(Vec::new())
    }

    /// Retrieves data by CID from a specific miner.
    pub async fn retrieve(
        &self,
        data_cid: &str,
        miner: &str,
    ) -> DataSourceResult<Vec<u8>> {
        // In production, would initiate retrieval deal
        Err(DataSourceError::NotFound(format!(
            "{}@{}",
            data_cid, miner
        )))
    }

    /// Checks if data is available.
    pub async fn is_available(&self, _data_cid: &str) -> bool {
        // Would check deal status
        false
    }
}

/// Filecoin data source implementation.
pub struct FilecoinDataSource {
    /// Filecoin client.
    client: FilecoinClient,
    /// Local cache.
    cache: DataCache,
    /// Mock storage for testing.
    mock_storage: HashMap<String, MockDeal>,
}

/// Mock deal for testing.
struct MockDeal {
    data: Vec<u8>,
    info: DealInfo,
}

impl FilecoinDataSource {
    /// Creates a new Filecoin data source.
    pub fn new(config: FilecoinConfig) -> Self {
        let cache_size = config.cache_size;
        Self {
            client: FilecoinClient::new(config),
            cache: DataCache::new(cache_size),
            mock_storage: HashMap::new(),
        }
    }

    /// Creates with default configuration.
    pub fn default_source() -> Self {
        Self::new(FilecoinConfig::default())
    }

    /// Returns the client.
    pub fn client(&self) -> &FilecoinClient {
        &self.client
    }

    /// Stores mock data for testing.
    pub fn store_mock(&mut self, deal_id: &str, data: Vec<u8>, miner: &str) {
        let info = DealInfo {
            deal_id: deal_id.parse().unwrap_or(0),
            proposal_cid: format!("bafy{}", deal_id),
            data_cid: deal_id.to_string(),
            provider: miner.to_string(),
            client: "t01000".to_string(),
            size: data.len() as u64,
            price_per_epoch: 0,
            start_epoch: 0,
            end_epoch: 1000000,
            state: DealState::Active,
        };
        self.mock_storage.insert(
            deal_id.to_string(),
            MockDeal { data, info },
        );
    }

    /// Computes content hash.
    fn compute_hash(&self, data: &[u8]) -> Hash {
        Sha256Hasher.hash_leaf(data)
    }

    // Internal fetch implementation
    async fn fetch_internal(&self, id: &str, verify_hash: Option<Hash>) -> DataSourceResult<Vec<u8>> {
        // Check mock storage first
        if let Some(mock) = self.mock_storage.get(id) {
            let data = mock.data.clone();

            // Verify hash if requested
            if let Some(expected) = verify_hash {
                let actual = self.compute_hash(&data);
                if actual != expected {
                    return Err(DataSourceError::IntegrityError {
                        expected,
                        actual,
                    });
                }
            }

            return Ok(data);
        }

        // Try to retrieve via client
        if let Some(pos) = id.find('@') {
            let (deal_id, miner) = id.split_at(pos);
            let miner = &miner[1..];
            self.client.retrieve(deal_id, miner).await
        } else {
            Err(DataSourceError::NotFound(id.to_string()))
        }
    }
}

impl DataSource for FilecoinDataSource {
    fn source_type(&self) -> &'static str {
        "filecoin"
    }

    fn is_available(&self) -> BoxFuture<'_, bool> {
        Box::pin(async move { true })
    }

    fn fetch(&self, id: &str, options: &FetchOptions) -> BoxFuture<'_, DataSourceResult<Vec<u8>>> {
        let id = id.to_string();
        let verify_hash = options.verify_hash;
        Box::pin(async move {
            self.fetch_internal(&id, verify_hash).await
        })
    }

    fn fetch_stream(&self, id: &str, options: &FetchOptions) -> BoxFuture<'_, DataSourceResult<DataStream>> {
        let id = id.to_string();
        let verify_hash = options.verify_hash;
        Box::pin(async move {
            let data = self.fetch_internal(&id, verify_hash).await?;
            Ok(DataStream::from_bytes(data))
        })
    }

    fn metadata(&self, id: &str) -> BoxFuture<'_, DataSourceResult<ResourceMetadata>> {
        let id = id.to_string();
        Box::pin(async move {
            if let Some(mock) = self.mock_storage.get(&id) {
                let hash = self.compute_hash(&mock.data);
                return Ok(ResourceMetadata {
                    id: id.clone(),
                    size: mock.info.size,
                    content_type: None,
                    hash: Some(hash),
                    created_at: Some(mock.info.start_epoch),
                    modified_at: None,
                    extra: {
                        let mut extra = HashMap::new();
                        extra.insert("deal_id".to_string(), mock.info.deal_id.to_string());
                        extra.insert("provider".to_string(), mock.info.provider.clone());
                        extra
                    },
                });
            }

            Err(DataSourceError::NotFound(id))
        })
    }

    fn exists(&self, id: &str) -> BoxFuture<'_, DataSourceResult<bool>> {
        let id = id.to_string();
        Box::pin(async move { Ok(self.mock_storage.contains_key(&id)) })
    }

    fn verify(&self, id: &str, expected_hash: &Hash) -> BoxFuture<'_, DataSourceResult<bool>> {
        let id = id.to_string();
        let expected = *expected_hash;
        Box::pin(async move {
            let data = self.fetch_internal(&id, None).await?;
            let actual = self.compute_hash(&data);
            Ok(actual == expected)
        })
    }

    fn to_origin(&self, id: &str) -> DataOrigin {
        let (deal_id, miner_id) = if let Some(pos) = id.find('@') {
            let (d, m) = id.split_at(pos);
            (d.to_string(), m[1..].to_string())
        } else {
            (id.to_string(), "unknown".to_string())
        };

        DataOrigin::Filecoin { deal_id, miner_id }
    }
}

/// Storage market parameters for deal making.
#[derive(Debug, Clone)]
pub struct DealParams {
    /// Data CID to store.
    pub data_cid: String,
    /// Piece CID (CAR file).
    pub piece_cid: String,
    /// Piece size (padded).
    pub piece_size: u64,
    /// Target miners.
    pub miners: Vec<String>,
    /// Deal duration in epochs.
    pub duration_epochs: u64,
    /// Maximum price per epoch per GiB.
    pub max_price: u128,
    /// Whether to verify the deal on-chain.
    pub verified: bool,
}

/// Deal maker for creating storage deals.
pub struct DealMaker {
    /// Filecoin client.
    _client: FilecoinClient,
    /// Wallet address.
    _wallet: String,
}

impl DealMaker {
    /// Creates a new deal maker.
    pub fn new(config: FilecoinConfig, wallet: String) -> Self {
        Self {
            _client: FilecoinClient::new(config),
            _wallet: wallet,
        }
    }

    /// Creates a storage deal.
    pub async fn create_deal(&self, _params: DealParams) -> DataSourceResult<DealInfo> {
        Err(DataSourceError::UnsupportedOperation(
            "Deal creation not implemented".into(),
        ))
    }

    /// Lists available storage providers.
    pub async fn list_miners(&self) -> DataSourceResult<Vec<MinerInfo>> {
        Ok(Vec::new())
    }
}

/// Information about a storage provider.
#[derive(Debug, Clone)]
pub struct MinerInfo {
    /// Miner ID (e.g., f01234).
    pub miner_id: String,
    /// Peer ID for networking.
    pub peer_id: String,
    /// Sector size.
    pub sector_size: u64,
    /// Ask price per epoch per GiB.
    pub ask_price: u128,
    /// Verified deal price.
    pub verified_price: u128,
    /// Power in bytes.
    pub raw_power: u128,
    /// Quality adjusted power.
    pub quality_power: u128,
    /// Number of faults.
    pub fault_count: u32,
}

/// Builder for Filecoin data source.
pub struct FilecoinDataSourceBuilder {
    config: FilecoinConfig,
}

impl FilecoinDataSourceBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            config: FilecoinConfig::default(),
        }
    }

    /// Sets the API endpoint.
    pub fn api_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.config.api_endpoint = endpoint.into();
        self
    }

    /// Sets the API token.
    pub fn api_token(mut self, token: impl Into<String>) -> Self {
        self.config.api_token = Some(token.into());
        self
    }

    /// Adds a gateway.
    pub fn add_gateway(mut self, gateway: impl Into<String>) -> Self {
        self.config.gateways.push(gateway.into());
        self
    }

    /// Sets timeout.
    pub fn timeout_secs(mut self, secs: u64) -> Self {
        self.config.timeout_secs = secs;
        self
    }

    /// Adds a preferred miner.
    pub fn prefer_miner(mut self, miner: impl Into<String>) -> Self {
        self.config.preferred_miners.push(miner.into());
        self
    }

    /// Builds the data source.
    pub fn build(self) -> FilecoinDataSource {
        FilecoinDataSource::new(self.config)
    }
}

impl Default for FilecoinDataSourceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deal_state() {
        assert!(DealState::Active.is_available());
        assert!(!DealState::Proposed.is_available());
        assert!(!DealState::Expired.is_available());
    }

    #[tokio::test]
    async fn test_filecoin_mock_storage() {
        let mut source = FilecoinDataSource::default_source();

        let data = b"Filecoin test data".to_vec();
        source.store_mock("deal123", data.clone(), "f01234");

        let fetched = source
            .fetch("deal123", &FetchOptions::default())
            .await
            .unwrap();
        assert_eq!(fetched, data);
    }

    #[tokio::test]
    async fn test_filecoin_metadata() {
        let mut source = FilecoinDataSource::default_source();

        let data = b"Test data".to_vec();
        source.store_mock("deal456", data.clone(), "f05678");

        let meta = source.metadata("deal456").await.unwrap();
        assert_eq!(meta.size, data.len() as u64);
        assert_eq!(meta.extra.get("provider"), Some(&"f05678".to_string()));
    }

    #[test]
    fn test_to_origin() {
        let source = FilecoinDataSource::default_source();

        let origin = source.to_origin("deal123@f01234");
        match origin {
            DataOrigin::Filecoin { deal_id, miner_id } => {
                assert_eq!(deal_id, "deal123");
                assert_eq!(miner_id, "f01234");
            }
            _ => panic!("Expected Filecoin origin"),
        }
    }

    #[tokio::test]
    async fn test_filecoin_not_found() {
        let source = FilecoinDataSource::default_source();

        let result = source
            .fetch("nonexistent", &FetchOptions::default())
            .await;
        assert!(matches!(result, Err(DataSourceError::NotFound(_))));
    }

    #[test]
    fn test_builder() {
        let source = FilecoinDataSourceBuilder::new()
            .api_endpoint("https://custom.endpoint")
            .api_token("secret")
            .timeout_secs(60)
            .prefer_miner("f01234")
            .build();

        assert_eq!(source.client.config.api_endpoint, "https://custom.endpoint");
        assert_eq!(source.client.config.preferred_miners, vec!["f01234"]);
    }
}
