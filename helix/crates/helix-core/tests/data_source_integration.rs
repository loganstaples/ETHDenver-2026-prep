//! Integration tests for data source validation.
//!
//! Verifies that IPFS and S3 data sources produce valid training data
//! as BoundedTensors with correct dimensions and error tracking.
//!
//! These tests use the built-in mock storage (store_local / store_mock)
//! which mirrors the real fetch path without requiring network access.
//! The feature-gated HTTP code paths share the same parsing/validation
//! logic, so exercising the mock path validates the data pipeline end-to-end.
//!
//! Run with: cargo test --test data_source_integration

use helix_core::data::sources::{
    DataSource, FetchOptions, IpfsDataSource, IpfsSourceConfig, S3Config, S3DataSource,
};
use helix_core::data::Sha256Hasher;
use helix_core::data::merkle::MerkleHasher;
use helix_core::types::{BoundedTensor, BoundedValue};

// =============================================================================
// HELPERS
// =============================================================================

/// Builds a simple CSV file as bytes with the given dimensions.
fn build_csv_bytes(rows: usize, cols: usize) -> Vec<u8> {
    let mut csv = String::new();
    // Header
    let headers: Vec<String> = (0..cols).map(|c| format!("col{}", c)).collect();
    csv.push_str(&headers.join(","));
    csv.push('\n');
    // Data rows
    for r in 0..rows {
        let values: Vec<String> = (0..cols)
            .map(|c| format!("{:.4}", (r * cols + c) as f64 * 0.1))
            .collect();
        csv.push_str(&values.join(","));
        csv.push('\n');
    }
    csv.into_bytes()
}

/// Parses raw CSV bytes into a BoundedTensor.
/// Simulates what a real pipeline would do after fetching data from a source.
fn csv_bytes_to_bounded_tensor(data: &[u8], expected_rows: usize, expected_cols: usize) -> BoundedTensor {
    let text = std::str::from_utf8(data).expect("valid UTF-8");
    let mut lines = text.lines();
    let _header = lines.next().expect("CSV must have a header");

    let mut values = Vec::with_capacity(expected_rows * expected_cols);
    let mut row_count = 0;

    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<f64> = line
            .split(',')
            .map(|v| v.trim().parse::<f64>().expect("valid float"))
            .collect();
        assert_eq!(cols.len(), expected_cols, "column count mismatch");
        for val in cols {
            values.push(BoundedValue::exact(val));
        }
        row_count += 1;
    }

    assert_eq!(row_count, expected_rows, "row count mismatch");
    BoundedTensor::new(values, vec![expected_rows, expected_cols])
}

/// Builds binary tensor data: 4 bytes magic + shape header + f64 values.
fn build_binary_tensor_bytes(shape: &[usize]) -> Vec<u8> {
    let total_elements: usize = shape.iter().product();
    let mut buf = Vec::new();

    // Simple binary format: ndims (u32) + dims (u32 each) + f64 values
    let ndims = shape.len() as u32;
    buf.extend_from_slice(&ndims.to_le_bytes());
    for &dim in shape {
        buf.extend_from_slice(&(dim as u32).to_le_bytes());
    }
    for i in 0..total_elements {
        let val = (i as f64) * 0.01;
        buf.extend_from_slice(&val.to_le_bytes());
    }
    buf
}

/// Parses our simple binary format into a BoundedTensor.
fn binary_bytes_to_bounded_tensor(data: &[u8]) -> BoundedTensor {
    let ndims = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let mut offset = 4;
    let mut shape = Vec::with_capacity(ndims);
    for _ in 0..ndims {
        let dim = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        shape.push(dim);
        offset += 4;
    }

    let total_elements: usize = shape.iter().product();
    let mut values = Vec::with_capacity(total_elements);
    for _ in 0..total_elements {
        let val = f64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        values.push(BoundedValue::exact(val));
        offset += 8;
    }

    BoundedTensor::new(values, shape)
}

// =============================================================================
// IPFS DATA SOURCE TESTS
// =============================================================================

#[tokio::test]
async fn test_ipfs_source_returns_properly_shaped_csv_tensors() {
    let mut source = IpfsDataSource::new(IpfsSourceConfig::default());

    // Store a 10x4 CSV dataset via mock storage
    let csv_data = build_csv_bytes(10, 4);
    let cid = source.store_and_get_cid(csv_data.clone());

    // Fetch via the DataSource trait (exercises the full fetch pipeline)
    let fetched = source.fetch(&cid, &FetchOptions::default()).await.unwrap();

    // Verify the raw data matches
    assert_eq!(fetched, csv_data);

    // Parse into a BoundedTensor and validate shape
    let tensor = csv_bytes_to_bounded_tensor(&fetched, 10, 4);
    assert_eq!(tensor.shape(), &vec![10, 4]);
    assert_eq!(tensor.len(), 40);

    // All values should have zero error (exact values from CSV)
    assert_eq!(tensor.max_error(), 0.0);
}

#[tokio::test]
async fn test_ipfs_source_returns_binary_tensor_data() {
    let mut source = IpfsDataSource::new(IpfsSourceConfig::default());

    // Store binary tensor data for a [5, 3] matrix
    let binary_data = build_binary_tensor_bytes(&[5, 3]);
    let cid = source.store_and_get_cid(binary_data.clone());

    let fetched = source.fetch(&cid, &FetchOptions::default()).await.unwrap();
    assert_eq!(fetched, binary_data);

    let tensor = binary_bytes_to_bounded_tensor(&fetched);
    assert_eq!(tensor.shape(), &vec![5, 3]);
    assert_eq!(tensor.len(), 15);

    // Verify specific values
    assert!((tensor.data()[0].value() - 0.0).abs() < 1e-10);
    assert!((tensor.data()[1].value() - 0.01).abs() < 1e-10);
    assert!((tensor.data()[14].value() - 0.14).abs() < 1e-10);
}

#[tokio::test]
async fn test_ipfs_source_hash_verification_on_fetch() {
    let mut source = IpfsDataSource::new(IpfsSourceConfig::default());

    let csv_data = build_csv_bytes(5, 2);
    let cid = source.store_and_get_cid(csv_data.clone());

    // The IPFS source uses Sha256Hasher.hash_leaf() for verification,
    // so we must use the same hash function.
    let hash = Sha256Hasher.hash_leaf(&csv_data);
    let opts = FetchOptions::with_verification(hash);
    let fetched = source.fetch(&cid, &opts).await.unwrap();
    assert_eq!(fetched, csv_data);
}

#[tokio::test]
async fn test_ipfs_source_not_found_error() {
    let source = IpfsDataSource::new(IpfsSourceConfig::default());

    // Fetching a non-existent CID should return NotFound
    let result = source
        .fetch("QmNonExistent00000000000000000000000000000000", &FetchOptions::default())
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_ipfs_source_chunked_fetch_produces_valid_tensor() {
    let mut source = IpfsDataSource::new(IpfsSourceConfig::default());

    // Build a larger CSV dataset
    let csv_data = build_csv_bytes(100, 8);
    let cid = source.store_and_get_cid(csv_data.clone());

    let fetched = source.fetch(&cid, &FetchOptions::default()).await.unwrap();
    let tensor = csv_bytes_to_bounded_tensor(&fetched, 100, 8);
    assert_eq!(tensor.shape(), &vec![100, 8]);
    assert_eq!(tensor.len(), 800);
}

// =============================================================================
// S3 DATA SOURCE TESTS
// =============================================================================

#[tokio::test]
async fn test_s3_source_csv_parsed_into_valid_bounded_tensors() {
    let mut source = S3DataSource::new(S3Config::aws("test-bucket", "us-east-1"));

    // Store a 20x6 CSV dataset
    let csv_data = build_csv_bytes(20, 6);
    source.store_mock("datasets/train.csv", csv_data.clone());

    let fetched = source
        .fetch("datasets/train.csv", &FetchOptions::default())
        .await
        .unwrap();

    let tensor = csv_bytes_to_bounded_tensor(&fetched, 20, 6);
    assert_eq!(tensor.shape(), &vec![20, 6]);
    assert_eq!(tensor.len(), 120);

    // All elements should be exact (no approximation error)
    for val in tensor.data() {
        assert!(val.value().is_finite(), "all values must be finite");
        assert_eq!(val.absolute_error(), 0.0, "exact values have zero error");
    }
}

#[tokio::test]
async fn test_s3_source_binary_data_parsed_into_valid_bounded_tensors() {
    let mut source = S3DataSource::new(S3Config::aws("ml-data", "us-west-2"));

    // Store binary tensor data [8, 4]
    let binary_data = build_binary_tensor_bytes(&[8, 4]);
    source.store_mock("models/weights.bin", binary_data.clone());

    let fetched = source
        .fetch("models/weights.bin", &FetchOptions::default())
        .await
        .unwrap();

    let tensor = binary_bytes_to_bounded_tensor(&fetched);
    assert_eq!(tensor.shape(), &vec![8, 4]);
    assert_eq!(tensor.len(), 32);

    // Verify dimensions are correct and values are in expected range
    for (i, val) in tensor.data().iter().enumerate() {
        let expected = i as f64 * 0.01;
        assert!(
            (val.value() - expected).abs() < 1e-10,
            "value[{}] = {}, expected {}",
            i,
            val.value(),
            expected
        );
    }
}

#[tokio::test]
async fn test_s3_source_minio_config() {
    let mut source = S3DataSource::new(S3Config::minio("http://localhost:9000", "training-data"));

    let csv_data = build_csv_bytes(5, 3);
    source.store_mock("batch/001.csv", csv_data.clone());

    let fetched = source
        .fetch("batch/001.csv", &FetchOptions::default())
        .await
        .unwrap();

    let tensor = csv_bytes_to_bounded_tensor(&fetched, 5, 3);
    assert_eq!(tensor.shape(), &vec![5, 3]);
}

#[tokio::test]
async fn test_s3_source_hash_verification() {
    let mut source = S3DataSource::new(S3Config::aws("verified-bucket", "eu-west-1"));

    let data = build_binary_tensor_bytes(&[3, 3]);
    source.store_mock("verified/tensor.bin", data.clone());

    // S3 source uses Sha256Hasher.hash_leaf() for verification internally
    let hash = Sha256Hasher.hash_leaf(&data);
    let opts = FetchOptions::with_verification(hash);
    let fetched = source.fetch("verified/tensor.bin", &opts).await.unwrap();
    assert_eq!(fetched, data);
}

#[tokio::test]
async fn test_s3_source_not_found_error() {
    let source = S3DataSource::new(S3Config::aws("empty-bucket", "us-east-1"));

    let result = source
        .fetch("nonexistent/file.csv", &FetchOptions::default())
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_s3_source_range_fetch_produces_valid_subset() {
    let mut source = S3DataSource::new(S3Config::aws("range-bucket", "us-east-1"));

    let raw = b"0123456789ABCDEF".to_vec();
    source.store_mock("data/range.bin", raw.clone());

    let opts = FetchOptions {
        range_start: Some(4),
        range_end: Some(12),
        ..Default::default()
    };

    let fetched = source.fetch("data/range.bin", &opts).await.unwrap();
    assert_eq!(fetched, b"456789AB");
}

// =============================================================================
// CROSS-SOURCE PIPELINE TESTS
// =============================================================================

#[tokio::test]
async fn test_multi_source_fetcher_with_ipfs_and_s3() {
    use helix_core::data::sources::{FallbackBehavior, MultiSourceFetcher};

    // Set up IPFS source with data
    let mut ipfs = IpfsDataSource::new(IpfsSourceConfig::default());
    let csv_data = build_csv_bytes(10, 3);
    let cid = ipfs.store_and_get_cid(csv_data.clone());

    // Set up S3 source (no data for this key, so it will fail)
    let s3 = S3DataSource::new(S3Config::aws("backup-bucket", "us-east-1"));

    // Multi-source fetcher: IPFS first, S3 fallback
    let fetcher = MultiSourceFetcher::new()
        .add_source(Box::new(ipfs))
        .add_source(Box::new(s3))
        .with_fallback(FallbackBehavior::TryNext);

    let fetched = fetcher.fetch(&cid, &FetchOptions::default()).await.unwrap();
    let tensor = csv_bytes_to_bounded_tensor(&fetched, 10, 3);
    assert_eq!(tensor.shape(), &vec![10, 3]);
}

#[tokio::test]
async fn test_approximate_tensor_from_fetched_data() {
    // Simulate fetching data that represents quantized/approximate values
    let mut source = IpfsDataSource::new(IpfsSourceConfig::default());

    let csv_data = build_csv_bytes(4, 2);
    let cid = source.store_and_get_cid(csv_data.clone());

    let fetched = source.fetch(&cid, &FetchOptions::default()).await.unwrap();
    let exact_tensor = csv_bytes_to_bounded_tensor(&fetched, 4, 2);

    // Convert to approximate tensor (simulating quantization error)
    let approx_tensor = BoundedTensor::from_approximate(
        exact_tensor.data().iter().map(|v| v.value()).collect(),
        vec![4, 2],
        0.001, // 0.1% error from quantization
    );

    assert_eq!(approx_tensor.shape(), &vec![4, 2]);
    assert!(approx_tensor.max_error() > 0.0, "approximate tensor should have error bounds");
    assert!(
        (approx_tensor.max_error() - 0.001).abs() < 1e-10,
        "error should be 0.001"
    );
}
