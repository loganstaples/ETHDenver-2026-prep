//! Filecoin Data Source for HELIX.
//!
//! Provides integration with Filecoin for long-term, verifiable storage
//! of training datasets. Supports:
//! - Deal-based data retrieval
//! - Storage provider selection
//! - Proof verification
//! - Data availability guarantees
//! - Deal lifecycle management
//! - Miner reputation tracking
//! - Retrieval market integration
//! - Piece CID computation
//! - CAR file handling

use std::collections::HashMap;
use std::time::SystemTime;

use super::{
    BoxFuture, DataCache, DataChunk, DataSource, DataSourceError, DataSourceResult, DataStream,
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

/// Proof type for storage verification.
#[derive(Debug, Clone, PartialEq)]
pub enum ProofType {
    /// Proof of Replication.
    PoRep,
    /// Proof of Space-Time (Window).
    WindowPoSt,
    /// Proof of Space-Time (Winning).
    WinningPoSt,
}

/// Verification result for a deal.
#[derive(Debug, Clone)]
pub struct DealVerificationResult {
    /// Deal ID.
    pub deal_id: u64,
    /// Whether the deal is verified.
    pub verified: bool,
    /// Proof type checked.
    pub proof_type: ProofType,
    /// Last verification timestamp.
    pub verified_at: u64,
    /// Next expected proof deadline.
    pub next_deadline: Option<u64>,
    /// Number of successful proofs.
    pub successful_proofs: u64,
    /// Number of missed proofs.
    pub missed_proofs: u64,
    /// Verification details.
    pub details: Option<String>,
}

/// Miner reputation information.
#[derive(Debug, Clone)]
pub struct MinerReputation {
    /// Miner ID.
    pub miner_id: String,
    /// Reputation score (0-100).
    pub score: u32,
    /// Total deals completed.
    pub total_deals: u64,
    /// Successful deals.
    pub successful_deals: u64,
    /// Failed deals.
    pub failed_deals: u64,
    /// Total data stored (bytes).
    pub total_data_stored: u128,
    /// Average retrieval time (ms).
    pub avg_retrieval_time_ms: u64,
    /// Last updated.
    pub updated_at: u64,
}

impl MinerReputation {
    /// Creates a new miner reputation.
    pub fn new(miner_id: String) -> Self {
        Self {
            miner_id,
            score: 50,
            total_deals: 0,
            successful_deals: 0,
            failed_deals: 0,
            total_data_stored: 0,
            avg_retrieval_time_ms: 0,
            updated_at: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    /// Returns the success rate.
    pub fn success_rate(&self) -> f64 {
        if self.total_deals == 0 {
            return 0.0;
        }
        self.successful_deals as f64 / self.total_deals as f64
    }
}

/// Piece information for CAR files.
#[derive(Debug, Clone)]
pub struct PieceInfo {
    /// Piece CID.
    pub piece_cid: String,
    /// Padded piece size.
    pub padded_size: u64,
    /// Unpadded piece size.
    pub unpadded_size: u64,
    /// Data CID (payload root).
    pub data_cid: String,
    /// CAR file CID (if applicable).
    pub car_cid: Option<String>,
}

/// Retrieval offer from a miner.
#[derive(Debug, Clone)]
pub struct RetrievalOffer {
    /// Miner ID.
    pub miner_id: String,
    /// Price per byte (attoFIL).
    pub price_per_byte: u128,
    /// Unseal price (attoFIL).
    pub unseal_price: u128,
    /// Estimated retrieval time (seconds).
    pub estimated_time_secs: u64,
    /// Payment interval.
    pub payment_interval: u64,
    /// Maximum concurrent retrievals.
    pub max_concurrent: u32,
}

/// Filecoin client for direct Lotus API access.
pub struct FilecoinClient {
    /// Configuration.
    pub config: FilecoinConfig,
    /// Statistics.
    #[allow(dead_code)]
    stats: FilecoinStats,
    /// Miner reputation cache.
    miner_reputations: HashMap<String, MinerReputation>,
    /// Deal cache.
    deal_cache: HashMap<u64, DealInfo>,
    /// Verification history.
    #[allow(dead_code)]
    verification_history: HashMap<u64, Vec<DealVerificationResult>>,
}

impl FilecoinClient {
    /// Creates a new Filecoin client.
    pub fn new(config: FilecoinConfig) -> Self {
        Self {
            config,
            stats: FilecoinStats::default(),
            miner_reputations: HashMap::new(),
            deal_cache: HashMap::new(),
            verification_history: HashMap::new(),
        }
    }

    /// Gets deal information by deal ID.
    pub async fn get_deal(&self, deal_id: u64) -> DataSourceResult<DealInfo> {
        // Check cache first
        if let Some(deal) = self.deal_cache.get(&deal_id) {
            return Ok(deal.clone());
        }

        // In production, would make JSON-RPC call to Lotus API
        // StateMarketDeal
        Err(DataSourceError::NotFound(format!("deal:{}", deal_id)))
    }

    /// Lists deals for a client address.
    pub async fn list_client_deals(&self, _client: &str) -> DataSourceResult<Vec<DealInfo>> {
        // In production, would query the chain via StateMarketDeals
        Ok(Vec::new())
    }

    /// Retrieves data by CID from a specific miner.
    pub async fn retrieve(
        &self,
        data_cid: &str,
        miner: &str,
    ) -> DataSourceResult<Vec<u8>> {
        // In production, would initiate retrieval deal via Lotus
        Err(DataSourceError::NotFound(format!(
            "{}@{}",
            data_cid, miner
        )))
    }

    /// Checks if data is available.
    pub async fn is_available(&self, _data_cid: &str) -> bool {
        // Would check deal status via StateDealProviderCollateralBounds
        false
    }

    /// Verifies a deal's storage proofs.
    pub async fn verify_deal(&self, deal_id: u64) -> DataSourceResult<DealVerificationResult> {
        let deal = self.get_deal(deal_id).await?;

        // In production, would check:
        // 1. Deal is in Active state
        // 2. Miner has submitted required WindowPoSt proofs
        // 3. No slashing events

        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Ok(DealVerificationResult {
            deal_id,
            verified: deal.state == DealState::Active,
            proof_type: ProofType::WindowPoSt,
            verified_at: now,
            next_deadline: Some(now + 2880 * 30), // ~1 day in epochs
            successful_proofs: 0,
            missed_proofs: 0,
            details: None,
        })
    }

    /// Gets miner reputation.
    pub fn get_miner_reputation(&self, miner_id: &str) -> Option<&MinerReputation> {
        self.miner_reputations.get(miner_id)
    }

    /// Updates miner reputation based on deal outcome.
    pub fn update_miner_reputation(&mut self, miner_id: &str, success: bool, data_size: u64) {
        let rep = self.miner_reputations
            .entry(miner_id.to_string())
            .or_insert_with(|| MinerReputation::new(miner_id.to_string()));

        rep.total_deals += 1;
        if success {
            rep.successful_deals += 1;
            rep.total_data_stored += data_size as u128;
        } else {
            rep.failed_deals += 1;
        }

        // Update score
        rep.score = ((rep.success_rate() * 100.0) as u32).min(100);
        rep.updated_at = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
    }

    /// Ranks miners by reputation.
    pub fn rank_miners(&self) -> Vec<&MinerReputation> {
        let mut miners: Vec<_> = self.miner_reputations.values().collect();
        miners.sort_by(|a, b| b.score.cmp(&a.score));
        miners
    }

    /// Gets retrieval offers for a CID.
    pub async fn get_retrieval_offers(&self, _data_cid: &str) -> DataSourceResult<Vec<RetrievalOffer>> {
        // In production, would query retrieval market
        Ok(Vec::new())
    }

    /// Computes piece CID from data.
    pub fn compute_piece_cid(&self, data: &[u8]) -> PieceInfo {
        // Simplified piece CID computation
        // In production, would use proper fr32 padding and CommP
        let hash = Sha256Hasher.hash_leaf(data);
        let hex = hash.to_hex();

        // Pad to power of 2
        let padded_size = data.len().next_power_of_two() as u64;

        PieceInfo {
            piece_cid: format!("baga6ea4seaq{}", &hex[..52]),
            padded_size,
            unpadded_size: data.len() as u64,
            data_cid: format!("bafy{}", &hex[..52]),
            car_cid: None,
        }
    }
}

/// Filecoin data source implementation.
pub struct FilecoinDataSource {
    /// Filecoin client.
    client: FilecoinClient,
    /// Local cache.
    #[allow(dead_code)]
    cache: DataCache,
    /// Mock storage for testing.
    mock_storage: HashMap<String, MockDeal>,
    /// Verification results cache.
    verification_cache: HashMap<u64, DealVerificationResult>,
    /// Piece info cache.
    piece_cache: HashMap<String, PieceInfo>,
}

/// Mock deal for testing.
struct MockDeal {
    data: Vec<u8>,
    info: DealInfo,
    piece_info: Option<PieceInfo>,
}

impl FilecoinDataSource {
    /// Creates a new Filecoin data source.
    pub fn new(config: FilecoinConfig) -> Self {
        let cache_size = config.cache_size;
        Self {
            client: FilecoinClient::new(config),
            cache: DataCache::new(cache_size),
            mock_storage: HashMap::new(),
            verification_cache: HashMap::new(),
            piece_cache: HashMap::new(),
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

    /// Returns mutable client reference.
    pub fn client_mut(&mut self) -> &mut FilecoinClient {
        &mut self.client
    }

    /// Stores mock data for testing.
    pub fn store_mock(&mut self, deal_id: &str, data: Vec<u8>, miner: &str) {
        let piece_info = self.client.compute_piece_cid(&data);

        let info = DealInfo {
            deal_id: deal_id.parse().unwrap_or(0),
            proposal_cid: format!("bafy{}", deal_id),
            data_cid: piece_info.data_cid.clone(),
            provider: miner.to_string(),
            client: "t01000".to_string(),
            size: data.len() as u64,
            price_per_epoch: 0,
            start_epoch: 0,
            end_epoch: 1000000,
            state: DealState::Active,
        };

        self.piece_cache.insert(deal_id.to_string(), piece_info.clone());
        self.mock_storage.insert(
            deal_id.to_string(),
            MockDeal { data, info, piece_info: Some(piece_info) },
        );
    }

    /// Stores mock data with deal info.
    pub fn store_mock_with_info(&mut self, deal_id: &str, data: Vec<u8>, info: DealInfo) {
        let piece_info = self.client.compute_piece_cid(&data);
        self.piece_cache.insert(deal_id.to_string(), piece_info.clone());
        self.mock_storage.insert(
            deal_id.to_string(),
            MockDeal { data, info, piece_info: Some(piece_info) },
        );
    }

    /// Computes content hash.
    fn compute_hash(&self, data: &[u8]) -> Hash {
        Sha256Hasher.hash_leaf(data)
    }

    /// Gets deal info.
    pub fn get_deal_info(&self, deal_id: &str) -> Option<&DealInfo> {
        self.mock_storage.get(deal_id).map(|m| &m.info)
    }

    /// Gets piece info.
    pub fn get_piece_info(&self, deal_id: &str) -> Option<&PieceInfo> {
        self.piece_cache.get(deal_id)
    }

    /// Verifies a deal.
    pub async fn verify_deal(&mut self, deal_id: &str) -> DataSourceResult<DealVerificationResult> {
        let id: u64 = deal_id.parse()
            .map_err(|_| DataSourceError::Custom("Invalid deal ID".into()))?;

        // Check cache
        if let Some(result) = self.verification_cache.get(&id) {
            return Ok(result.clone());
        }

        // Check mock storage
        if let Some(mock) = self.mock_storage.get(deal_id) {
            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);

            let result = DealVerificationResult {
                deal_id: id,
                verified: mock.info.state == DealState::Active,
                proof_type: ProofType::WindowPoSt,
                verified_at: now,
                next_deadline: Some(now + 86400),
                successful_proofs: if mock.info.state == DealState::Active { 10 } else { 0 },
                missed_proofs: if mock.info.state == DealState::Slashed { 3 } else { 0 },
                details: Some(format!("Deal state: {:?}", mock.info.state)),
            };

            self.verification_cache.insert(id, result.clone());
            return Ok(result);
        }

        // Try client
        self.client.verify_deal(id).await
    }

    /// Verifies multiple deals.
    pub async fn verify_deals(&mut self, deal_ids: &[&str]) -> Vec<DataSourceResult<DealVerificationResult>> {
        let mut results = Vec::with_capacity(deal_ids.len());
        for deal_id in deal_ids {
            results.push(self.verify_deal(deal_id).await);
        }
        results
    }

    /// Lists active deals.
    pub fn list_active_deals(&self) -> Vec<&DealInfo> {
        self.mock_storage
            .values()
            .filter(|m| m.info.state == DealState::Active)
            .map(|m| &m.info)
            .collect()
    }

    /// Gets deals by miner.
    pub fn get_deals_by_miner(&self, miner: &str) -> Vec<&DealInfo> {
        self.mock_storage
            .values()
            .filter(|m| m.info.provider == miner)
            .map(|m| &m.info)
            .collect()
    }

    /// Selects best miner for retrieval.
    pub fn select_best_miner(&self, data_cid: &str) -> Option<String> {
        // Find deals with this CID
        let deals: Vec<_> = self.mock_storage
            .values()
            .filter(|m| m.info.data_cid == data_cid && m.info.state == DealState::Active)
            .collect();

        if deals.is_empty() {
            return None;
        }

        // Get miner with best reputation
        let ranked = self.client.rank_miners();
        for miner in ranked {
            if deals.iter().any(|d| d.info.provider == miner.miner_id) {
                return Some(miner.miner_id.clone());
            }
        }

        // Fall back to first available
        deals.first().map(|d| d.info.provider.clone())
    }

    // Internal fetch implementation
    async fn fetch_internal(&self, id: &str, verify_hash: Option<Hash>) -> DataSourceResult<Vec<u8>> {
        // Check mock storage first
        if let Some(mock) = self.mock_storage.get(id) {
            // Verify deal is active
            if mock.info.state != DealState::Active {
                return Err(DataSourceError::Custom(format!(
                    "Deal {} is not active: {:?}",
                    id, mock.info.state
                )));
            }

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

    /// Fetches with miner selection.
    async fn fetch_with_miner_selection(&self, data_cid: &str, verify_hash: Option<Hash>) -> DataSourceResult<Vec<u8>> {
        // Find the best miner
        if let Some(miner) = self.select_best_miner(data_cid) {
            // Find the deal with this miner
            for mock in self.mock_storage.values() {
                if mock.info.data_cid == data_cid && mock.info.provider == miner {
                    if mock.info.state != DealState::Active {
                        continue;
                    }

                    let data = mock.data.clone();

                    if let Some(expected) = verify_hash {
                        let actual = self.compute_hash(&data);
                        if actual != expected {
                            return Err(DataSourceError::IntegrityError { expected, actual });
                        }
                    }

                    return Ok(data);
                }
            }
        }

        Err(DataSourceError::NotFound(data_cid.to_string()))
    }
}

impl DataSource for FilecoinDataSource {
    fn source_type(&self) -> &'static str {
        "filecoin"
    }

    fn is_available(&self) -> BoxFuture<'_, bool> {
        Box::pin(async move {
            // Check if we have any active deals
            !self.list_active_deals().is_empty()
        })
    }

    fn fetch(&self, id: &str, options: &FetchOptions) -> BoxFuture<'_, DataSourceResult<Vec<u8>>> {
        let id = id.to_string();
        let verify_hash = options.verify_hash;
        Box::pin(async move {
            // Check if this looks like a data CID (starts with "bafy")
            if id.starts_with("bafy") {
                return self.fetch_with_miner_selection(&id, verify_hash).await;
            }
            self.fetch_internal(&id, verify_hash).await
        })
    }

    fn fetch_stream(&self, id: &str, options: &FetchOptions) -> BoxFuture<'_, DataSourceResult<DataStream>> {
        let id = id.to_string();
        let verify_hash = options.verify_hash;
        let chunk_size = 1024 * 1024; // 1MB chunks
        Box::pin(async move {
            let data = self.fetch_internal(&id, verify_hash).await?;

            let total_size = data.len() as u64;
            let mut chunks = Vec::new();

            for (i, chunk_data) in data.chunks(chunk_size).enumerate() {
                let offset = (i * chunk_size) as u64;
                let is_last = offset + chunk_data.len() as u64 >= total_size;

                chunks.push(DataChunk {
                    data: chunk_data.to_vec(),
                    offset,
                    total_size: Some(total_size),
                    is_last,
                });
            }

            Ok(DataStream::new(chunks))
        })
    }

    fn metadata(&self, id: &str) -> BoxFuture<'_, DataSourceResult<ResourceMetadata>> {
        let id = id.to_string();
        Box::pin(async move {
            if let Some(mock) = self.mock_storage.get(&id) {
                let hash = self.compute_hash(&mock.data);
                let mut extra = HashMap::new();

                extra.insert("deal_id".to_string(), mock.info.deal_id.to_string());
                extra.insert("provider".to_string(), mock.info.provider.clone());
                extra.insert("state".to_string(), format!("{:?}", mock.info.state));
                extra.insert("client".to_string(), mock.info.client.clone());
                extra.insert("data_cid".to_string(), mock.info.data_cid.clone());

                if let Some(piece) = &mock.piece_info {
                    extra.insert("piece_cid".to_string(), piece.piece_cid.clone());
                    extra.insert("padded_size".to_string(), piece.padded_size.to_string());
                }

                return Ok(ResourceMetadata {
                    id: id.clone(),
                    size: mock.info.size,
                    content_type: Some("application/octet-stream".to_string()),
                    hash: Some(hash),
                    created_at: Some(mock.info.start_epoch),
                    modified_at: None,
                    extra,
                });
            }

            Err(DataSourceError::NotFound(id))
        })
    }

    fn exists(&self, id: &str) -> BoxFuture<'_, DataSourceResult<bool>> {
        let id = id.to_string();
        Box::pin(async move {
            if self.mock_storage.contains_key(&id) {
                return Ok(true);
            }
            // Also check by data CID
            Ok(self.mock_storage.values().any(|m| m.info.data_cid == id))
        })
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
        // Try to get miner from deal info
        if let Some(info) = self.get_deal_info(id) {
            return DataOrigin::Filecoin {
                deal_id: info.deal_id.to_string(),
                miner_id: info.provider.clone(),
            };
        }

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
        assert!(!DealState::Slashed.is_available());
        assert!(!DealState::Transferring.is_available());
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

    // ========== Deal Verification Tests ==========

    #[tokio::test]
    async fn test_deal_verification_active() {
        let mut source = FilecoinDataSource::default_source();
        let data = b"Verified data".to_vec();
        source.store_mock("1001", data, "f01234");

        let result = source.verify_deal("1001").await.unwrap();

        assert!(result.verified);
        assert_eq!(result.deal_id, 1001);
        assert_eq!(result.proof_type, ProofType::WindowPoSt);
        assert!(result.verified_at > 0);
        assert!(result.next_deadline.is_some());
        assert_eq!(result.successful_proofs, 10);
        assert_eq!(result.missed_proofs, 0);
    }

    #[tokio::test]
    async fn test_deal_verification_slashed() {
        let mut source = FilecoinDataSource::default_source();

        let info = DealInfo {
            deal_id: 2001,
            proposal_cid: "bafy2001".to_string(),
            data_cid: "bafyslashed".to_string(),
            provider: "f0bad".to_string(),
            client: "t01000".to_string(),
            size: 1024,
            price_per_epoch: 100,
            start_epoch: 1000,
            end_epoch: 2000000,
            state: DealState::Slashed,
        };
        source.store_mock_with_info("2001", b"slashed data".to_vec(), info);

        let result = source.verify_deal("2001").await.unwrap();

        assert!(!result.verified);
        assert_eq!(result.missed_proofs, 3);
        assert!(result.details.as_ref().unwrap().contains("Slashed"));
    }

    #[tokio::test]
    async fn test_verify_multiple_deals() {
        let mut source = FilecoinDataSource::default_source();

        source.store_mock("3001", b"data1".to_vec(), "f01111");
        source.store_mock("3002", b"data2".to_vec(), "f02222");
        source.store_mock("3003", b"data3".to_vec(), "f03333");

        let results = source.verify_deals(&["3001", "3002", "3003"]).await;

        assert_eq!(results.len(), 3);
        assert!(results[0].as_ref().unwrap().verified);
        assert!(results[1].as_ref().unwrap().verified);
        assert!(results[2].as_ref().unwrap().verified);
    }

    #[tokio::test]
    async fn test_verification_caching() {
        let mut source = FilecoinDataSource::default_source();
        source.store_mock("4001", b"cache test".to_vec(), "f01234");

        // First verification
        let result1 = source.verify_deal("4001").await.unwrap();

        // Second verification should hit cache
        let result2 = source.verify_deal("4001").await.unwrap();

        assert_eq!(result1.deal_id, result2.deal_id);
        assert_eq!(result1.verified_at, result2.verified_at);
    }

    #[tokio::test]
    async fn test_verification_invalid_deal_id() {
        let mut source = FilecoinDataSource::default_source();

        let result = source.verify_deal("not_a_number").await;
        assert!(result.is_err());
    }

    // ========== Miner Reputation Tests ==========

    #[test]
    fn test_miner_reputation_new() {
        let rep = MinerReputation::new("f01234".to_string());

        assert_eq!(rep.miner_id, "f01234");
        assert_eq!(rep.score, 50);
        assert_eq!(rep.total_deals, 0);
        assert_eq!(rep.successful_deals, 0);
        assert_eq!(rep.failed_deals, 0);
        assert_eq!(rep.success_rate(), 0.0);
    }

    #[test]
    fn test_miner_reputation_success_rate() {
        let mut rep = MinerReputation::new("f01234".to_string());
        rep.total_deals = 10;
        rep.successful_deals = 8;
        rep.failed_deals = 2;

        assert!((rep.success_rate() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_update_miner_reputation_success() {
        let mut source = FilecoinDataSource::default_source();

        source.client_mut().update_miner_reputation("f01234", true, 1000);
        source.client_mut().update_miner_reputation("f01234", true, 2000);

        let rep = source.client().get_miner_reputation("f01234").unwrap();
        assert_eq!(rep.total_deals, 2);
        assert_eq!(rep.successful_deals, 2);
        assert_eq!(rep.failed_deals, 0);
        assert_eq!(rep.total_data_stored, 3000);
        assert_eq!(rep.score, 100);
    }

    #[test]
    fn test_update_miner_reputation_mixed() {
        let mut source = FilecoinDataSource::default_source();

        source.client_mut().update_miner_reputation("f0mixed", true, 1000);
        source.client_mut().update_miner_reputation("f0mixed", true, 1000);
        source.client_mut().update_miner_reputation("f0mixed", false, 0);
        source.client_mut().update_miner_reputation("f0mixed", true, 1000);

        let rep = source.client().get_miner_reputation("f0mixed").unwrap();
        assert_eq!(rep.total_deals, 4);
        assert_eq!(rep.successful_deals, 3);
        assert_eq!(rep.failed_deals, 1);
        assert_eq!(rep.score, 75);
    }

    #[test]
    fn test_rank_miners() {
        let mut source = FilecoinDataSource::default_source();

        // Create miners with different success rates
        for _ in 0..10 {
            source.client_mut().update_miner_reputation("f0best", true, 1000);
        }
        for _ in 0..8 {
            source.client_mut().update_miner_reputation("f0good", true, 1000);
        }
        source.client_mut().update_miner_reputation("f0good", false, 0);
        source.client_mut().update_miner_reputation("f0good", false, 0);

        for _ in 0..5 {
            source.client_mut().update_miner_reputation("f0avg", true, 1000);
            source.client_mut().update_miner_reputation("f0avg", false, 0);
        }

        let ranked = source.client().rank_miners();

        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0].miner_id, "f0best");
        assert_eq!(ranked[0].score, 100);
        assert_eq!(ranked[1].miner_id, "f0good");
        assert_eq!(ranked[1].score, 80);
        assert_eq!(ranked[2].miner_id, "f0avg");
        assert_eq!(ranked[2].score, 50);
    }

    // ========== Piece CID Computation Tests ==========

    #[test]
    fn test_compute_piece_cid() {
        let source = FilecoinDataSource::default_source();
        let data = b"Test data for piece CID computation";

        let piece = source.client().compute_piece_cid(data);

        assert!(piece.piece_cid.starts_with("baga6ea4seaq"));
        assert!(piece.data_cid.starts_with("bafy"));
        assert_eq!(piece.unpadded_size, data.len() as u64);
        assert!(piece.padded_size >= piece.unpadded_size);
        assert!(piece.padded_size.is_power_of_two());
    }

    #[test]
    fn test_piece_cid_deterministic() {
        let source = FilecoinDataSource::default_source();
        let data = b"Deterministic data";

        let piece1 = source.client().compute_piece_cid(data);
        let piece2 = source.client().compute_piece_cid(data);

        assert_eq!(piece1.piece_cid, piece2.piece_cid);
        assert_eq!(piece1.data_cid, piece2.data_cid);
    }

    #[test]
    fn test_piece_cid_different_data() {
        let source = FilecoinDataSource::default_source();

        let piece1 = source.client().compute_piece_cid(b"Data A");
        let piece2 = source.client().compute_piece_cid(b"Data B");

        assert_ne!(piece1.piece_cid, piece2.piece_cid);
        assert_ne!(piece1.data_cid, piece2.data_cid);
    }

    #[test]
    fn test_piece_info_stored_with_mock() {
        let mut source = FilecoinDataSource::default_source();
        source.store_mock("5001", b"Piece test".to_vec(), "f01234");

        let piece = source.get_piece_info("5001").unwrap();
        assert!(piece.piece_cid.starts_with("baga6ea4seaq"));
        assert_eq!(piece.unpadded_size, 10);
    }

    // ========== Miner Selection Tests ==========

    #[test]
    fn test_select_best_miner_with_reputation() {
        let mut source = FilecoinDataSource::default_source();

        // Create deals with same data CID from different miners
        let data = b"Shared data".to_vec();
        let piece = source.client().compute_piece_cid(&data);
        let data_cid = piece.data_cid.clone();

        let info1 = DealInfo {
            deal_id: 6001,
            proposal_cid: "bafy6001".to_string(),
            data_cid: data_cid.clone(),
            provider: "f0good".to_string(),
            client: "t01000".to_string(),
            size: data.len() as u64,
            price_per_epoch: 100,
            start_epoch: 1000,
            end_epoch: 2000000,
            state: DealState::Active,
        };

        let info2 = DealInfo {
            deal_id: 6002,
            proposal_cid: "bafy6002".to_string(),
            data_cid: data_cid.clone(),
            provider: "f0best".to_string(),
            client: "t01000".to_string(),
            size: data.len() as u64,
            price_per_epoch: 100,
            start_epoch: 1000,
            end_epoch: 2000000,
            state: DealState::Active,
        };

        source.store_mock_with_info("6001", data.clone(), info1);
        source.store_mock_with_info("6002", data.clone(), info2);

        // Set up reputations
        for _ in 0..10 {
            source.client_mut().update_miner_reputation("f0best", true, 1000);
        }
        for _ in 0..5 {
            source.client_mut().update_miner_reputation("f0good", true, 1000);
            source.client_mut().update_miner_reputation("f0good", false, 0);
        }

        let selected = source.select_best_miner(&data_cid);
        assert_eq!(selected, Some("f0best".to_string()));
    }

    #[test]
    fn test_select_best_miner_no_reputation() {
        let mut source = FilecoinDataSource::default_source();

        let data = b"No reputation data".to_vec();
        let piece = source.client().compute_piece_cid(&data);
        let data_cid = piece.data_cid.clone();

        let info = DealInfo {
            deal_id: 7001,
            proposal_cid: "bafy7001".to_string(),
            data_cid: data_cid.clone(),
            provider: "f0new".to_string(),
            client: "t01000".to_string(),
            size: data.len() as u64,
            price_per_epoch: 100,
            start_epoch: 1000,
            end_epoch: 2000000,
            state: DealState::Active,
        };
        source.store_mock_with_info("7001", data, info);

        // No reputation data - should fall back to first available
        let selected = source.select_best_miner(&data_cid);
        assert_eq!(selected, Some("f0new".to_string()));
    }

    #[test]
    fn test_select_best_miner_inactive_deals() {
        let mut source = FilecoinDataSource::default_source();

        let data = b"Inactive data".to_vec();
        let piece = source.client().compute_piece_cid(&data);
        let data_cid = piece.data_cid.clone();

        let info = DealInfo {
            deal_id: 8001,
            proposal_cid: "bafy8001".to_string(),
            data_cid: data_cid.clone(),
            provider: "f0inactive".to_string(),
            client: "t01000".to_string(),
            size: data.len() as u64,
            price_per_epoch: 100,
            start_epoch: 1000,
            end_epoch: 2000000,
            state: DealState::Expired,
        };
        source.store_mock_with_info("8001", data, info);

        // No active deals for this CID
        let selected = source.select_best_miner(&data_cid);
        assert!(selected.is_none());
    }

    // ========== Active Deals and Miner Query Tests ==========

    #[test]
    fn test_list_active_deals() {
        let mut source = FilecoinDataSource::default_source();

        source.store_mock("9001", b"active1".to_vec(), "f01");

        let expired_info = DealInfo {
            deal_id: 9002,
            proposal_cid: "bafy9002".to_string(),
            data_cid: "bafy_expired".to_string(),
            provider: "f02".to_string(),
            client: "t01000".to_string(),
            size: 8,
            price_per_epoch: 100,
            start_epoch: 0,
            end_epoch: 1000,
            state: DealState::Expired,
        };
        source.store_mock_with_info("9002", b"expired".to_vec(), expired_info);

        source.store_mock("9003", b"active2".to_vec(), "f03");

        let active = source.list_active_deals();
        assert_eq!(active.len(), 2);
        assert!(active.iter().all(|d| d.state == DealState::Active));
    }

    #[test]
    fn test_get_deals_by_miner() {
        let mut source = FilecoinDataSource::default_source();

        source.store_mock("10001", b"data1".to_vec(), "f0multi");
        source.store_mock("10002", b"data2".to_vec(), "f0multi");
        source.store_mock("10003", b"data3".to_vec(), "f0other");

        let miner_deals = source.get_deals_by_miner("f0multi");
        assert_eq!(miner_deals.len(), 2);
        assert!(miner_deals.iter().all(|d| d.provider == "f0multi"));
    }

    // ========== Hash Verification Tests ==========

    #[tokio::test]
    async fn test_fetch_with_hash_verification() {
        let mut source = FilecoinDataSource::default_source();
        let data = b"Hash verified data".to_vec();
        let hash = Sha256Hasher.hash_leaf(&data);

        source.store_mock("11001", data.clone(), "f01234");

        let options = FetchOptions {
            verify_hash: Some(hash),
            ..Default::default()
        };

        let result = source.fetch("11001", &options).await.unwrap();
        assert_eq!(result, data);
    }

    #[tokio::test]
    async fn test_fetch_with_wrong_hash() {
        let mut source = FilecoinDataSource::default_source();
        let data = b"Original data".to_vec();
        let wrong_hash = Sha256Hasher.hash_leaf(b"Different data");

        source.store_mock("12001", data, "f01234");

        let options = FetchOptions {
            verify_hash: Some(wrong_hash),
            ..Default::default()
        };

        let result = source.fetch("12001", &options).await;
        assert!(matches!(result, Err(DataSourceError::IntegrityError { .. })));
    }

    // ========== Streaming Tests ==========

    #[tokio::test]
    async fn test_fetch_stream() {
        let mut source = FilecoinDataSource::default_source();

        // Create larger data that spans multiple chunks
        let data: Vec<u8> = (0..2_500_000).map(|i| (i % 256) as u8).collect();
        source.store_mock("13001", data.clone(), "f01234");

        let stream = source
            .fetch_stream("13001", &FetchOptions::default())
            .await
            .unwrap();

        let chunks: Vec<_> = stream.collect();
        assert!(chunks.len() > 1); // Should have multiple chunks

        // Verify first and last chunk properties
        assert_eq!(chunks[0].offset, 0);
        assert!(!chunks[0].is_last);
        assert!(chunks.last().unwrap().is_last);

        // Verify total data
        let reassembled: Vec<u8> = chunks.iter().flat_map(|c| c.data.clone()).collect();
        assert_eq!(reassembled, data);
    }

    // ========== Data CID Fetch Tests ==========

    #[tokio::test]
    async fn test_fetch_by_data_cid() {
        let mut source = FilecoinDataSource::default_source();

        let data = b"CID based fetch".to_vec();
        let piece = source.client().compute_piece_cid(&data);
        let data_cid = piece.data_cid.clone();

        let info = DealInfo {
            deal_id: 14001,
            proposal_cid: "bafy14001".to_string(),
            data_cid: data_cid.clone(),
            provider: "f01234".to_string(),
            client: "t01000".to_string(),
            size: data.len() as u64,
            price_per_epoch: 100,
            start_epoch: 1000,
            end_epoch: 2000000,
            state: DealState::Active,
        };
        source.store_mock_with_info("14001", data.clone(), info);

        // Fetch by data CID (starts with "bafy")
        let fetched = source.fetch(&data_cid, &FetchOptions::default()).await.unwrap();
        assert_eq!(fetched, data);
    }

    // ========== Exists Tests ==========

    #[tokio::test]
    async fn test_exists_by_deal_id() {
        let mut source = FilecoinDataSource::default_source();
        source.store_mock("15001", b"exists test".to_vec(), "f01234");

        assert!(source.exists("15001").await.unwrap());
        assert!(!source.exists("99999").await.unwrap());
    }

    #[tokio::test]
    async fn test_exists_by_data_cid() {
        let mut source = FilecoinDataSource::default_source();

        let data = b"Exists by CID".to_vec();
        let piece = source.client().compute_piece_cid(&data);
        source.store_mock("16001", data, "f01234");

        assert!(source.exists(&piece.data_cid).await.unwrap());
    }

    // ========== Proof Type Tests ==========

    #[test]
    fn test_proof_types() {
        assert_eq!(ProofType::PoRep, ProofType::PoRep);
        assert_ne!(ProofType::PoRep, ProofType::WindowPoSt);
        assert_ne!(ProofType::WindowPoSt, ProofType::WinningPoSt);
    }

    // ========== Origin Tests with Deal Info ==========

    #[test]
    fn test_to_origin_with_stored_deal() {
        let mut source = FilecoinDataSource::default_source();
        source.store_mock("17001", b"origin test".to_vec(), "f0origin");

        let origin = source.to_origin("17001");
        match origin {
            DataOrigin::Filecoin { deal_id, miner_id } => {
                assert_eq!(deal_id, "17001");
                assert_eq!(miner_id, "f0origin");
            }
            _ => panic!("Expected Filecoin origin"),
        }
    }

    // ========== Metadata with Piece Info Tests ==========

    #[tokio::test]
    async fn test_metadata_includes_piece_info() {
        let mut source = FilecoinDataSource::default_source();
        source.store_mock("18001", b"metadata test".to_vec(), "f01234");

        let meta = source.metadata("18001").await.unwrap();

        assert!(meta.extra.contains_key("piece_cid"));
        assert!(meta.extra.contains_key("padded_size"));
        assert!(meta.extra.get("piece_cid").unwrap().starts_with("baga6ea4seaq"));
    }

    // ========== Availability Tests ==========

    #[tokio::test]
    async fn test_is_available_with_active_deals() {
        let mut source = FilecoinDataSource::default_source();
        source.store_mock("19001", b"available".to_vec(), "f01234");

        assert!(source.is_available().await);
    }

    #[tokio::test]
    async fn test_is_available_empty() {
        let source = FilecoinDataSource::default_source();
        assert!(!source.is_available().await);
    }

    // ========== Fetch Inactive Deal Tests ==========

    #[tokio::test]
    async fn test_fetch_inactive_deal_fails() {
        let mut source = FilecoinDataSource::default_source();

        let info = DealInfo {
            deal_id: 20001,
            proposal_cid: "bafy20001".to_string(),
            data_cid: "bafy_inactive".to_string(),
            provider: "f0inactive".to_string(),
            client: "t01000".to_string(),
            size: 10,
            price_per_epoch: 100,
            start_epoch: 0,
            end_epoch: 1000,
            state: DealState::Proposed,
        };
        source.store_mock_with_info("20001", b"inactive".to_vec(), info);

        let result = source.fetch("20001", &FetchOptions::default()).await;
        assert!(result.is_err());
    }
}
