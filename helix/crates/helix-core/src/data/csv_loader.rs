//! Production CSV and binary data loader.
//!
//! Provides real data loading for MNIST-scale datasets with proper batching,
//! normalization, and train/test splitting. Supports both CSV and compact
//! binary formats.
//!
//! # Binary format layout
//!
//! ```text
//! [magic: 4 bytes "HXDT"]
//! [version: u32 LE]
//! [num_samples: u64 LE]
//! [features_per_sample: u32 LE]
//! [labels_per_sample: u32 LE]
//! [dtype: u8 (0=f32, 1=f64, 2=u8)]
//! [reserved: 3 bytes]
//! [data: features then labels for each sample, packed]
//! [checksum: SHA256 of all preceding bytes]
//! ```

use sha2::{Digest, Sha256};

use crate::error::{DataError, HelixError, HelixResult};

use super::DataLoader;

/// Magic bytes for the binary format.
const BINARY_MAGIC: &[u8; 4] = b"HXDT";
/// Current binary format version.
const BINARY_VERSION: u32 = 1;

/// Normalization strategy for features.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Normalization {
    /// No normalization.
    None,
    /// Scale to [0, 1] by dividing by `divisor`.
    Scale { divisor: f64 },
    /// Z-score normalization: (x - mean) / std.
    ZScore,
    /// Min-max normalization to [0, 1].
    MinMax,
}

/// Configuration for CSV data loading.
#[derive(Debug, Clone)]
pub struct CsvLoaderConfig {
    /// Column indices to use as features.
    pub feature_columns: Vec<usize>,
    /// Column indices to use as labels.
    pub label_columns: Vec<usize>,
    /// Whether the CSV has a header row.
    pub has_header: bool,
    /// Batch size.
    pub batch_size: usize,
    /// Feature normalization strategy.
    pub normalization: Normalization,
    /// Fraction of data to use for training (rest is test). 0.0-1.0.
    pub train_fraction: f64,
    /// Random seed for shuffling (None = no shuffle).
    pub shuffle_seed: Option<u64>,
    /// CSV delimiter character.
    pub delimiter: u8,
    /// Maximum allowed file size in bytes. `None` disables the check.
    /// Defaults to `Some(1_073_741_824)` (1 GB).
    pub max_file_bytes: Option<usize>,
}

impl Default for CsvLoaderConfig {
    fn default() -> Self {
        Self {
            feature_columns: Vec::new(),
            label_columns: Vec::new(),
            has_header: true,
            batch_size: 32,
            normalization: Normalization::None,
            train_fraction: 0.8,
            shuffle_seed: None,
            delimiter: b',',
            max_file_bytes: Some(1_073_741_824),
        }
    }
}

/// Configuration for JSON data loading.
#[derive(Debug, Clone)]
pub struct JsonLoaderConfig {
    /// JSON keys to use as features.
    pub feature_keys: Vec<String>,
    /// JSON keys to use as labels.
    pub label_keys: Vec<String>,
    /// Batch size.
    pub batch_size: usize,
    /// Feature normalization strategy.
    pub normalization: Normalization,
    /// Fraction of data to use for training (rest is test). 0.0-1.0.
    pub train_fraction: f64,
    /// Random seed for shuffling (None = no shuffle).
    pub shuffle_seed: Option<u64>,
}

impl Default for JsonLoaderConfig {
    fn default() -> Self {
        Self {
            feature_keys: Vec::new(),
            label_keys: Vec::new(),
            batch_size: 32,
            normalization: Normalization::None,
            train_fraction: 0.8,
            shuffle_seed: None,
        }
    }
}

/// Configuration for LibSVM format data loading.
#[derive(Debug, Clone)]
pub struct LibSvmLoaderConfig {
    /// Total number of features (sparse indices are expanded to dense).
    pub num_features: usize,
    /// Batch size.
    pub batch_size: usize,
    /// Feature normalization strategy.
    pub normalization: Normalization,
    /// Fraction of data to use for training (rest is test). 0.0-1.0.
    pub train_fraction: f64,
    /// Random seed for shuffling (None = no shuffle).
    pub shuffle_seed: Option<u64>,
    /// Whether feature indices start at 0. LibSVM is conventionally 1-indexed.
    pub zero_indexed: bool,
}

impl Default for LibSvmLoaderConfig {
    fn default() -> Self {
        Self {
            num_features: 0,
            batch_size: 32,
            normalization: Normalization::None,
            train_fraction: 0.8,
            shuffle_seed: None,
            zero_indexed: false,
        }
    }
}

/// A production CSV data loader that supports MNIST-scale datasets.
///
/// Loads data into memory, applies normalization, and serves batches
/// efficiently. Supports train/test splitting with deterministic shuffling.
#[derive(Debug, Clone)]
pub struct CsvDataLoader {
    /// Batched training data: (features, labels) per batch.
    train_batches: Vec<(Vec<f64>, Vec<f64>)>,
    /// Batched test data: (features, labels) per batch.
    test_batches: Vec<(Vec<f64>, Vec<f64>)>,
    /// Features per sample.
    pub features_per_sample: usize,
    /// Labels per sample.
    pub labels_per_sample: usize,
    /// Total samples loaded (before splitting).
    pub total_samples: usize,
}

impl CsvDataLoader {
    /// Loads data from a CSV file.
    pub fn from_csv(path: &str, config: &CsvLoaderConfig) -> HelixResult<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| HelixError::Data(DataError::SourceError(format!(
                "failed to read CSV file '{}': {}", path, e
            ))))?;

        Self::from_csv_string(&content, config)
    }

    /// Loads data from a CSV string (useful for testing).
    pub fn from_csv_string(content: &str, config: &CsvLoaderConfig) -> HelixResult<Self> {
        // Validate: feature and label columns must not both be empty
        if config.feature_columns.is_empty() && config.label_columns.is_empty() {
            return Err(HelixError::Data(DataError::InvalidFormat(
                "both feature_columns and label_columns are empty; at least one must be specified".to_string()
            )));
        }

        // Max file size check
        if let Some(max_bytes) = config.max_file_bytes {
            if content.len() > max_bytes {
                return Err(HelixError::Data(DataError::DataTooLarge {
                    size: content.len(),
                    max: max_bytes,
                }));
            }
        }

        let delimiter = config.delimiter as char;
        let mut lines: Vec<&str> = content.lines().collect();

        if config.has_header && !lines.is_empty() {
            // Validate column names for path traversal and injection attacks
            let header_line = lines[0].trim();
            if !header_line.is_empty() {
                for col_name in header_line.split(delimiter) {
                    let col_name = col_name.trim();
                    if !col_name.is_empty() {
                        super::security::validate_column_name(col_name).map_err(|msg| {
                            HelixError::Data(DataError::InvalidFormat(
                                format!("unsafe column name: {}", msg)
                            ))
                        })?;
                    }
                }
            }
            lines.remove(0);
        }

        // Parse all rows
        let mut all_features: Vec<Vec<f64>> = Vec::with_capacity(lines.len());
        let mut all_labels: Vec<Vec<f64>> = Vec::with_capacity(lines.len());
        let mut expected_col_count: Option<usize> = None;

        for (line_num, line) in lines.iter().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let values: Vec<&str> = line.split(delimiter).collect();

            // Column count validation: all rows must have the same number of columns
            match expected_col_count {
                None => {
                    expected_col_count = Some(values.len());
                }
                Some(expected) => {
                    if values.len() != expected {
                        return Err(HelixError::Data(DataError::InvalidFormat(
                            format!(
                                "line {}: expected {} columns but found {}",
                                line_num + 1, expected, values.len()
                            )
                        )));
                    }
                }
            }

            let features: Vec<f64> = config.feature_columns.iter().map(|&col| {
                let val = values.get(col)
                    .and_then(|s| s.trim().parse::<f64>().ok())
                    .ok_or_else(|| HelixError::Data(DataError::InvalidFormat(
                        format!("line {}: could not parse column {} as f64", line_num + 1, col)
                    )))?;
                // NaN/Inf rejection for features
                if val.is_nan() {
                    return Err(HelixError::Data(DataError::InvalidFormat(
                        format!("line {}: column {} contains NaN", line_num + 1, col)
                    )));
                }
                if val.is_infinite() {
                    return Err(HelixError::Data(DataError::InvalidFormat(
                        format!("line {}: column {} contains Inf", line_num + 1, col)
                    )));
                }
                Ok(val)
            }).collect::<HelixResult<Vec<f64>>>()?;

            let labels: Vec<f64> = config.label_columns.iter().map(|&col| {
                values.get(col)
                    .and_then(|s| s.trim().parse::<f64>().ok())
                    .ok_or_else(|| HelixError::Data(DataError::InvalidFormat(
                        format!("line {}: could not parse label column {} as f64", line_num + 1, col)
                    )))
            }).collect::<HelixResult<Vec<f64>>>()?;

            all_features.push(features);
            all_labels.push(labels);
        }

        if all_features.is_empty() {
            return Err(HelixError::Data(DataError::InvalidFormat(
                "CSV contains no data rows".to_string()
            )));
        }

        let features_per_sample = all_features[0].len();
        let labels_per_sample = all_labels[0].len();
        let total_samples = all_features.len();

        // Apply shuffle if configured
        let mut indices: Vec<usize> = (0..total_samples).collect();
        if let Some(seed) = config.shuffle_seed {
            // Simple deterministic Fisher-Yates using a seeded LCG
            let mut rng_state = seed;
            for i in (1..indices.len()).rev() {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let j = (rng_state >> 33) as usize % (i + 1);
                indices.swap(i, j);
            }
        }

        // Apply normalization to features
        let mut flat_features: Vec<f64> = indices.iter()
            .flat_map(|&i| all_features[i].iter().copied())
            .collect();

        match config.normalization {
            Normalization::None => {}
            Normalization::Scale { divisor } => {
                for v in flat_features.iter_mut() {
                    *v /= divisor;
                }
            }
            Normalization::ZScore => {
                // Per-feature normalization
                for col in 0..features_per_sample {
                    let vals: Vec<f64> = (0..total_samples)
                        .map(|row| flat_features[row * features_per_sample + col])
                        .collect();
                    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                    let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
                    let std = variance.sqrt().max(1e-8);
                    for row in 0..total_samples {
                        flat_features[row * features_per_sample + col] = (flat_features[row * features_per_sample + col] - mean) / std;
                    }
                }
            }
            Normalization::MinMax => {
                for col in 0..features_per_sample {
                    let vals: Vec<f64> = (0..total_samples)
                        .map(|row| flat_features[row * features_per_sample + col])
                        .collect();
                    let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let range = (max - min).max(1e-8);
                    for row in 0..total_samples {
                        flat_features[row * features_per_sample + col] = (flat_features[row * features_per_sample + col] - min) / range;
                    }
                }
            }
        }

        let flat_labels: Vec<f64> = indices.iter()
            .flat_map(|&i| all_labels[i].iter().copied())
            .collect();

        // Split into train/test
        let train_count = ((total_samples as f64) * config.train_fraction).round() as usize;
        let train_count = train_count.max(1).min(total_samples);

        let train_features = &flat_features[..train_count * features_per_sample];
        let train_labels = &flat_labels[..train_count * labels_per_sample];
        let test_features = &flat_features[train_count * features_per_sample..];
        let test_labels = &flat_labels[train_count * labels_per_sample..];

        let train_batches = Self::batch_data(
            train_features, train_labels,
            features_per_sample, labels_per_sample,
            config.batch_size,
        );
        let test_batches = Self::batch_data(
            test_features, test_labels,
            features_per_sample, labels_per_sample,
            config.batch_size,
        );

        Ok(Self {
            train_batches,
            test_batches,
            features_per_sample,
            labels_per_sample,
            total_samples,
        })
    }

    /// Loads from a JSON string containing an array of objects.
    ///
    /// Expected format:
    /// ```json
    /// [
    ///   {"feature1": 1.0, "feature2": 2.0, "label": 0},
    ///   {"feature1": 3.0, "feature2": 4.0, "label": 1}
    /// ]
    /// ```
    pub fn from_json_string(content: &str, config: &JsonLoaderConfig) -> HelixResult<Self> {
        if config.feature_keys.is_empty() && config.label_keys.is_empty() {
            return Err(HelixError::Data(DataError::InvalidFormat(
                "both feature_keys and label_keys are empty; at least one must be specified".to_string()
            )));
        }

        let parsed: serde_json::Value = serde_json::from_str(content)
            .map_err(|e| HelixError::Data(DataError::InvalidFormat(
                format!("failed to parse JSON: {}", e)
            )))?;

        let array = parsed.as_array().ok_or_else(|| {
            HelixError::Data(DataError::InvalidFormat(
                "JSON root must be an array".to_string()
            ))
        })?;

        if array.is_empty() {
            return Err(HelixError::Data(DataError::InvalidFormat(
                "JSON array is empty".to_string()
            )));
        }

        let features_per_sample = config.feature_keys.len();
        let labels_per_sample = config.label_keys.len();

        let mut flat_features: Vec<f64> = Vec::with_capacity(array.len() * features_per_sample);
        let mut flat_labels: Vec<f64> = Vec::with_capacity(array.len() * labels_per_sample);

        for (idx, item) in array.iter().enumerate() {
            let obj = item.as_object().ok_or_else(|| {
                HelixError::Data(DataError::InvalidFormat(
                    format!("JSON element at index {} is not an object", idx)
                ))
            })?;

            for key in &config.feature_keys {
                let val = obj.get(key.as_str()).ok_or_else(|| {
                    HelixError::Data(DataError::InvalidFormat(
                        format!("element {}: missing feature key '{}'", idx, key)
                    ))
                })?;
                let num = val.as_f64().ok_or_else(|| {
                    HelixError::Data(DataError::InvalidFormat(
                        format!("element {}: feature key '{}' is not a number", idx, key)
                    ))
                })?;
                if num.is_nan() {
                    return Err(HelixError::Data(DataError::InvalidFormat(
                        format!("element {}: feature key '{}' contains NaN", idx, key)
                    )));
                }
                if num.is_infinite() {
                    return Err(HelixError::Data(DataError::InvalidFormat(
                        format!("element {}: feature key '{}' contains Inf", idx, key)
                    )));
                }
                flat_features.push(num);
            }

            for key in &config.label_keys {
                let val = obj.get(key.as_str()).ok_or_else(|| {
                    HelixError::Data(DataError::InvalidFormat(
                        format!("element {}: missing label key '{}'", idx, key)
                    ))
                })?;
                let num = val.as_f64().ok_or_else(|| {
                    HelixError::Data(DataError::InvalidFormat(
                        format!("element {}: label key '{}' is not a number", idx, key)
                    ))
                })?;
                flat_labels.push(num);
            }
        }

        // Apply normalization
        let total_samples = array.len();
        match config.normalization {
            Normalization::None => {}
            Normalization::Scale { divisor } => {
                for v in flat_features.iter_mut() {
                    *v /= divisor;
                }
            }
            Normalization::ZScore => {
                for col in 0..features_per_sample {
                    let vals: Vec<f64> = (0..total_samples)
                        .map(|row| flat_features[row * features_per_sample + col])
                        .collect();
                    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                    let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
                    let std = variance.sqrt().max(1e-8);
                    for row in 0..total_samples {
                        flat_features[row * features_per_sample + col] =
                            (flat_features[row * features_per_sample + col] - mean) / std;
                    }
                }
            }
            Normalization::MinMax => {
                for col in 0..features_per_sample {
                    let vals: Vec<f64> = (0..total_samples)
                        .map(|row| flat_features[row * features_per_sample + col])
                        .collect();
                    let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let range = (max - min).max(1e-8);
                    for row in 0..total_samples {
                        flat_features[row * features_per_sample + col] =
                            (flat_features[row * features_per_sample + col] - min) / range;
                    }
                }
            }
        }

        // Apply shuffle if configured
        if let Some(seed) = config.shuffle_seed {
            let mut indices: Vec<usize> = (0..total_samples).collect();
            let mut rng_state = seed;
            for i in (1..indices.len()).rev() {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let j = (rng_state >> 33) as usize % (i + 1);
                indices.swap(i, j);
            }
            let shuffled_features: Vec<f64> = indices.iter()
                .flat_map(|&i| flat_features[i * features_per_sample..(i + 1) * features_per_sample].iter().copied())
                .collect();
            let shuffled_labels: Vec<f64> = indices.iter()
                .flat_map(|&i| flat_labels[i * labels_per_sample..(i + 1) * labels_per_sample].iter().copied())
                .collect();
            return Self::from_flat(
                &shuffled_features, &shuffled_labels,
                features_per_sample, labels_per_sample,
                config.batch_size, config.train_fraction,
            );
        }

        Self::from_flat(
            &flat_features, &flat_labels,
            features_per_sample, labels_per_sample,
            config.batch_size, config.train_fraction,
        )
    }

    /// Loads from LibSVM format string.
    ///
    /// LibSVM format: `label index1:value1 index2:value2 ...`
    /// Example: `1 1:0.5 3:0.8 5:1.0`
    ///
    /// Sparse features are expanded into a dense vector of length `num_features`
    /// with zeros for missing indices.
    pub fn from_libsvm_string(content: &str, config: &LibSvmLoaderConfig) -> HelixResult<Self> {
        if config.num_features == 0 {
            return Err(HelixError::Data(DataError::InvalidFormat(
                "num_features must be greater than 0".to_string()
            )));
        }

        let lines: Vec<&str> = content.lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect();

        if lines.is_empty() {
            return Err(HelixError::Data(DataError::InvalidFormat(
                "LibSVM input contains no data lines".to_string()
            )));
        }

        let num_features = config.num_features;
        let total_samples = lines.len();
        let mut flat_features: Vec<f64> = vec![0.0; total_samples * num_features];
        let mut flat_labels: Vec<f64> = Vec::with_capacity(total_samples);

        for (line_num, line) in lines.iter().enumerate() {
            let mut parts = line.split_whitespace();

            let label_str = parts.next().ok_or_else(|| {
                HelixError::Data(DataError::InvalidFormat(
                    format!("line {}: missing label", line_num + 1)
                ))
            })?;
            let label: f64 = label_str.parse().map_err(|_| {
                HelixError::Data(DataError::InvalidFormat(
                    format!("line {}: could not parse label '{}' as f64", line_num + 1, label_str)
                ))
            })?;
            flat_labels.push(label);

            for token in parts {
                let colon_pos = token.find(':').ok_or_else(|| {
                    HelixError::Data(DataError::InvalidFormat(
                        format!("line {}: expected 'index:value' format, got '{}'", line_num + 1, token)
                    ))
                })?;
                let idx_str = &token[..colon_pos];
                let val_str = &token[colon_pos + 1..];

                let raw_idx: usize = idx_str.parse().map_err(|_| {
                    HelixError::Data(DataError::InvalidFormat(
                        format!("line {}: could not parse index '{}' as integer", line_num + 1, idx_str)
                    ))
                })?;

                let feature_idx = if config.zero_indexed { raw_idx } else { raw_idx.checked_sub(1).ok_or_else(|| {
                    HelixError::Data(DataError::InvalidFormat(
                        format!("line {}: 1-indexed feature index cannot be 0", line_num + 1)
                    ))
                })? };

                if feature_idx >= num_features {
                    return Err(HelixError::Data(DataError::InvalidFormat(
                        format!(
                            "line {}: feature index {} out of range (num_features={})",
                            line_num + 1, feature_idx, num_features
                        )
                    )));
                }

                let val: f64 = val_str.parse().map_err(|_| {
                    HelixError::Data(DataError::InvalidFormat(
                        format!("line {}: could not parse value '{}' as f64", line_num + 1, val_str)
                    ))
                })?;

                if val.is_nan() {
                    return Err(HelixError::Data(DataError::InvalidFormat(
                        format!("line {}: feature index {} contains NaN", line_num + 1, feature_idx)
                    )));
                }
                if val.is_infinite() {
                    return Err(HelixError::Data(DataError::InvalidFormat(
                        format!("line {}: feature index {} contains Inf", line_num + 1, feature_idx)
                    )));
                }

                flat_features[line_num * num_features + feature_idx] = val;
            }
        }

        // Apply normalization
        match config.normalization {
            Normalization::None => {}
            Normalization::Scale { divisor } => {
                for v in flat_features.iter_mut() {
                    *v /= divisor;
                }
            }
            Normalization::ZScore => {
                for col in 0..num_features {
                    let vals: Vec<f64> = (0..total_samples)
                        .map(|row| flat_features[row * num_features + col])
                        .collect();
                    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                    let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
                    let std = variance.sqrt().max(1e-8);
                    for row in 0..total_samples {
                        flat_features[row * num_features + col] =
                            (flat_features[row * num_features + col] - mean) / std;
                    }
                }
            }
            Normalization::MinMax => {
                for col in 0..num_features {
                    let vals: Vec<f64> = (0..total_samples)
                        .map(|row| flat_features[row * num_features + col])
                        .collect();
                    let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let range = (max - min).max(1e-8);
                    for row in 0..total_samples {
                        flat_features[row * num_features + col] =
                            (flat_features[row * num_features + col] - min) / range;
                    }
                }
            }
        }

        // Apply shuffle if configured
        if let Some(seed) = config.shuffle_seed {
            let mut indices: Vec<usize> = (0..total_samples).collect();
            let mut rng_state = seed;
            for i in (1..indices.len()).rev() {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let j = (rng_state >> 33) as usize % (i + 1);
                indices.swap(i, j);
            }
            let shuffled_features: Vec<f64> = indices.iter()
                .flat_map(|&i| flat_features[i * num_features..(i + 1) * num_features].iter().copied())
                .collect();
            let shuffled_labels: Vec<f64> = indices.iter()
                .map(|&i| flat_labels[i])
                .collect();
            return Self::from_flat(
                &shuffled_features, &shuffled_labels,
                num_features, 1,
                config.batch_size, config.train_fraction,
            );
        }

        Self::from_flat(
            &flat_features, &flat_labels,
            num_features, 1,
            config.batch_size, config.train_fraction,
        )
    }

    /// Creates a loader from flat feature/label arrays.
    pub fn from_flat(
        features: &[f64],
        labels: &[f64],
        features_per_sample: usize,
        labels_per_sample: usize,
        batch_size: usize,
        train_fraction: f64,
    ) -> HelixResult<Self> {
        let total_samples = features.len() / features_per_sample;
        if total_samples == 0 {
            return Err(HelixError::Data(DataError::InvalidFormat(
                "no samples in input data".to_string()
            )));
        }

        let train_count = ((total_samples as f64) * train_fraction).round() as usize;
        let train_count = train_count.max(1).min(total_samples);

        let train_features = &features[..train_count * features_per_sample];
        let train_labels = &labels[..train_count * labels_per_sample];
        let test_features = &features[train_count * features_per_sample..];
        let test_labels = &labels[train_count * labels_per_sample..];

        let train_batches = Self::batch_data(
            train_features, train_labels,
            features_per_sample, labels_per_sample,
            batch_size,
        );
        let test_batches = Self::batch_data(
            test_features, test_labels,
            features_per_sample, labels_per_sample,
            batch_size,
        );

        Ok(Self {
            train_batches,
            test_batches,
            features_per_sample,
            labels_per_sample,
            total_samples,
        })
    }

    /// Loads from the HELIX binary format.
    pub fn from_binary(path: &str, batch_size: usize, train_fraction: f64) -> HelixResult<Self> {
        let data = std::fs::read(path)
            .map_err(|e| HelixError::Data(DataError::SourceError(format!(
                "failed to read binary file '{}': {}", path, e
            ))))?;

        Self::from_binary_bytes(&data, batch_size, train_fraction)
    }

    /// Loads from binary bytes.
    pub fn from_binary_bytes(data: &[u8], batch_size: usize, train_fraction: f64) -> HelixResult<Self> {
        // Minimum: 4 magic + 4 version + 8 num_samples + 4 features + 4 labels + 1 dtype + 3 reserved + 32 checksum = 60
        if data.len() < 60 {
            return Err(HelixError::Data(DataError::InvalidFormat(
                "binary data too short".to_string()
            )));
        }

        // Verify magic
        if &data[0..4] != BINARY_MAGIC {
            return Err(HelixError::Data(DataError::InvalidFormat(
                "invalid binary magic".to_string()
            )));
        }

        // Verify version
        let version = u32::from_le_bytes(
            data[4..8].try_into().map_err(|_| HelixError::Data(DataError::InvalidFormat(
                "failed to read version bytes".to_string()
            )))?
        );
        if version != BINARY_VERSION {
            return Err(HelixError::Data(DataError::InvalidFormat(
                format!("unsupported binary version: {}", version)
            )));
        }

        // Verify checksum
        let payload = &data[..data.len() - 32];
        let stored_checksum = &data[data.len() - 32..];
        let mut hasher = Sha256::new();
        hasher.update(payload);
        let computed: [u8; 32] = hasher.finalize().into();
        if computed[..] != stored_checksum[..] {
            return Err(HelixError::Data(DataError::IntegrityError {
                expected: hex_encode(&computed),
                actual: hex_encode(stored_checksum),
            }));
        }

        let num_samples = u64::from_le_bytes(
            data[8..16].try_into().map_err(|_| HelixError::Data(DataError::InvalidFormat(
                "failed to read num_samples bytes".to_string()
            )))?
        ) as usize;
        let features_per_sample = u32::from_le_bytes(
            data[16..20].try_into().map_err(|_| HelixError::Data(DataError::InvalidFormat(
                "failed to read features_per_sample bytes".to_string()
            )))?
        ) as usize;
        let labels_per_sample = u32::from_le_bytes(
            data[20..24].try_into().map_err(|_| HelixError::Data(DataError::InvalidFormat(
                "failed to read labels_per_sample bytes".to_string()
            )))?
        ) as usize;
        let dtype = data[24];

        let header_size = 28; // 4 + 4 + 8 + 4 + 4 + 1 + 3
        // Read all values
        let mut features = Vec::with_capacity(num_samples * features_per_sample);
        let mut labels = Vec::with_capacity(num_samples * labels_per_sample);
        let mut offset = header_size;

        for _ in 0..num_samples {
            for _ in 0..features_per_sample {
                let v = read_value(&data, &mut offset, dtype)?;
                features.push(v);
            }
            for _ in 0..labels_per_sample {
                let v = read_value(&data, &mut offset, dtype)?;
                labels.push(v);
            }
        }

        Self::from_flat(
            &features, &labels,
            features_per_sample, labels_per_sample,
            batch_size, train_fraction,
        )
    }

    /// Writes data to HELIX binary format.
    pub fn write_binary(
        path: &str,
        features: &[f64],
        labels: &[f64],
        features_per_sample: usize,
        labels_per_sample: usize,
    ) -> HelixResult<()> {
        let num_samples = features.len() / features_per_sample;
        let mut buf = Vec::new();

        // Header
        buf.extend_from_slice(BINARY_MAGIC);
        buf.extend_from_slice(&BINARY_VERSION.to_le_bytes());
        buf.extend_from_slice(&(num_samples as u64).to_le_bytes());
        buf.extend_from_slice(&(features_per_sample as u32).to_le_bytes());
        buf.extend_from_slice(&(labels_per_sample as u32).to_le_bytes());
        buf.push(1); // dtype: f64
        buf.extend_from_slice(&[0u8; 3]); // reserved

        // Data: interleaved features then labels per sample
        for i in 0..num_samples {
            let f_start = i * features_per_sample;
            for j in 0..features_per_sample {
                buf.extend_from_slice(&features[f_start + j].to_le_bytes());
            }
            let l_start = i * labels_per_sample;
            for j in 0..labels_per_sample {
                buf.extend_from_slice(&labels[l_start + j].to_le_bytes());
            }
        }

        // Checksum
        let mut hasher = Sha256::new();
        hasher.update(&buf);
        let checksum: [u8; 32] = hasher.finalize().into();
        buf.extend_from_slice(&checksum);

        std::fs::write(path, &buf)
            .map_err(|e| HelixError::Data(DataError::SourceError(format!(
                "failed to write binary file '{}': {}", path, e
            ))))?;

        Ok(())
    }

    /// Returns the test data loader (separate from the train loader).
    pub fn test_loader(&self) -> TestDataLoader<'_> {
        TestDataLoader {
            batches: &self.test_batches,
        }
    }

    /// Returns the number of test batches.
    pub fn num_test_batches(&self) -> u64 {
        self.test_batches.len() as u64
    }

    /// Loads a test batch by ID.
    pub fn load_test_batch(&self, batch_id: u64) -> HelixResult<(Vec<f64>, Vec<f64>)> {
        let idx = batch_id as usize;
        self.test_batches.get(idx).cloned().ok_or_else(|| {
            HelixError::Data(DataError::BatchNotFound(idx))
        })
    }

    /// Batches flat data into (features_batch, labels_batch) pairs.
    fn batch_data(
        features: &[f64],
        labels: &[f64],
        features_per_sample: usize,
        labels_per_sample: usize,
        batch_size: usize,
    ) -> Vec<(Vec<f64>, Vec<f64>)> {
        let num_samples = if features_per_sample > 0 {
            features.len() / features_per_sample
        } else {
            0
        };
        let mut batches = Vec::new();

        for start in (0..num_samples).step_by(batch_size) {
            let end = (start + batch_size).min(num_samples);
            let f_start = start * features_per_sample;
            let f_end = end * features_per_sample;
            let l_start = start * labels_per_sample;
            let l_end = end * labels_per_sample;

            if f_end <= features.len() && l_end <= labels.len() {
                batches.push((
                    features[f_start..f_end].to_vec(),
                    labels[l_start..l_end].to_vec(),
                ));
            }
        }

        batches
    }
}

impl DataLoader for CsvDataLoader {
    fn load_batch(&self, batch_id: u64) -> HelixResult<(Vec<f64>, Vec<f64>)> {
        let idx = batch_id as usize;
        self.train_batches.get(idx).cloned().ok_or_else(|| {
            HelixError::Data(DataError::BatchNotFound(idx))
        })
    }

    fn num_batches(&self) -> u64 {
        self.train_batches.len() as u64
    }
}

/// Borrowed test data loader.
pub struct TestDataLoader<'a> {
    batches: &'a [(Vec<f64>, Vec<f64>)],
}

impl<'a> TestDataLoader<'a> {
    /// Loads a test batch.
    pub fn load_batch(&self, batch_id: u64) -> HelixResult<(Vec<f64>, Vec<f64>)> {
        let idx = batch_id as usize;
        self.batches.get(idx).cloned().ok_or_else(|| {
            HelixError::Data(DataError::BatchNotFound(idx))
        })
    }

    /// Number of test batches.
    pub fn num_batches(&self) -> u64 {
        self.batches.len() as u64
    }
}

/// Reads a single value from binary data based on dtype.
fn read_value(data: &[u8], offset: &mut usize, dtype: u8) -> HelixResult<f64> {
    match dtype {
        0 => {
            // f32
            if *offset + 4 > data.len() {
                return Err(HelixError::Data(DataError::InvalidFormat("truncated f32".to_string())));
            }
            let bytes: [u8; 4] = data[*offset..*offset + 4].try_into()
                .map_err(|_| HelixError::Data(DataError::InvalidFormat("invalid f32 bytes".to_string())))?;
            *offset += 4;
            Ok(f32::from_le_bytes(bytes) as f64)
        }
        1 => {
            // f64
            if *offset + 8 > data.len() {
                return Err(HelixError::Data(DataError::InvalidFormat("truncated f64".to_string())));
            }
            let bytes: [u8; 8] = data[*offset..*offset + 8].try_into()
                .map_err(|_| HelixError::Data(DataError::InvalidFormat("invalid f64 bytes".to_string())))?;
            *offset += 8;
            Ok(f64::from_le_bytes(bytes))
        }
        2 => {
            // u8
            if *offset + 1 > data.len() {
                return Err(HelixError::Data(DataError::InvalidFormat("truncated u8".to_string())));
            }
            let v = data[*offset] as f64;
            *offset += 1;
            Ok(v)
        }
        _ => Err(HelixError::Data(DataError::InvalidFormat(
            format!("unknown dtype: {}", dtype)
        ))),
    }
}

/// Encodes bytes as lowercase hex.
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_loader_basic() {
        let csv = "f1,f2,label\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,0\n7.0,8.0,1\n9.0,10.0,0\n";
        let config = CsvLoaderConfig {
            feature_columns: vec![0, 1],
            label_columns: vec![2],
            has_header: true,
            batch_size: 2,
            normalization: Normalization::None,
            train_fraction: 0.8,
            ..Default::default()
        };

        let loader = CsvDataLoader::from_csv_string(csv, &config).unwrap();
        assert_eq!(loader.total_samples, 5);
        assert_eq!(loader.features_per_sample, 2);
        assert_eq!(loader.labels_per_sample, 1);
        // 4 train samples -> 2 batches, 1 test sample -> 1 batch
        assert_eq!(loader.num_batches(), 2);
        assert_eq!(loader.num_test_batches(), 1);
    }

    #[test]
    fn test_csv_loader_with_normalization() {
        let csv = "f1,label\n0.0,0\n100.0,1\n200.0,0\n300.0,1\n400.0,0\n";
        let config = CsvLoaderConfig {
            feature_columns: vec![0],
            label_columns: vec![1],
            has_header: true,
            batch_size: 100,
            normalization: Normalization::Scale { divisor: 400.0 },
            train_fraction: 1.0,
            ..Default::default()
        };

        let loader = CsvDataLoader::from_csv_string(csv, &config).unwrap();
        let (features, _) = loader.load_batch(0).unwrap();
        // All values should be in [0, 1]
        for f in &features {
            assert!(*f >= 0.0 && *f <= 1.0, "feature {} not in [0, 1]", f);
        }
    }

    #[test]
    fn test_csv_loader_zscore_normalization() {
        let csv = "f1,label\n10.0,0\n20.0,1\n30.0,0\n40.0,1\n50.0,0\n";
        let config = CsvLoaderConfig {
            feature_columns: vec![0],
            label_columns: vec![1],
            has_header: true,
            batch_size: 100,
            normalization: Normalization::ZScore,
            train_fraction: 1.0,
            ..Default::default()
        };

        let loader = CsvDataLoader::from_csv_string(csv, &config).unwrap();
        let (features, _) = loader.load_batch(0).unwrap();
        // Z-scored data should have mean ~0
        let mean: f64 = features.iter().sum::<f64>() / features.len() as f64;
        assert!(mean.abs() < 1e-10, "z-score mean should be ~0, got {}", mean);
    }

    #[test]
    fn test_csv_loader_minmax_normalization() {
        let csv = "f1,label\n10.0,0\n20.0,1\n30.0,0\n40.0,1\n50.0,0\n";
        let config = CsvLoaderConfig {
            feature_columns: vec![0],
            label_columns: vec![1],
            has_header: true,
            batch_size: 100,
            normalization: Normalization::MinMax,
            train_fraction: 1.0,
            ..Default::default()
        };

        let loader = CsvDataLoader::from_csv_string(csv, &config).unwrap();
        let (features, _) = loader.load_batch(0).unwrap();
        let min = features.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = features.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!((min - 0.0).abs() < 1e-10);
        assert!((max - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_csv_loader_shuffle() {
        let csv = "f1,label\n1.0,0\n2.0,1\n3.0,0\n4.0,1\n5.0,0\n6.0,1\n7.0,0\n8.0,1\n9.0,0\n10.0,1\n";
        let config = CsvLoaderConfig {
            feature_columns: vec![0],
            label_columns: vec![1],
            has_header: true,
            batch_size: 100,
            normalization: Normalization::None,
            train_fraction: 1.0,
            shuffle_seed: Some(42),
            ..Default::default()
        };

        let loader1 = CsvDataLoader::from_csv_string(csv, &config).unwrap();
        let loader2 = CsvDataLoader::from_csv_string(csv, &config).unwrap();

        let (f1, _) = loader1.load_batch(0).unwrap();
        let (f2, _) = loader2.load_batch(0).unwrap();
        // Same seed = same shuffle
        assert_eq!(f1, f2);

        // Verify it's actually shuffled (not in original order)
        let original: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        assert_ne!(f1, original, "shuffled data should differ from original order");
    }

    #[test]
    fn test_csv_loader_train_test_split() {
        // 10 samples, 80% train = 8 train, 2 test
        let csv = "f1,label\n1.0,0\n2.0,1\n3.0,0\n4.0,1\n5.0,0\n6.0,1\n7.0,0\n8.0,1\n9.0,0\n10.0,1\n";
        let config = CsvLoaderConfig {
            feature_columns: vec![0],
            label_columns: vec![1],
            has_header: true,
            batch_size: 4,
            normalization: Normalization::None,
            train_fraction: 0.8,
            ..Default::default()
        };

        let loader = CsvDataLoader::from_csv_string(csv, &config).unwrap();
        assert_eq!(loader.num_batches(), 2); // 8 samples / 4 = 2 batches
        assert_eq!(loader.num_test_batches(), 1); // 2 samples -> 1 batch (partial)
    }

    #[test]
    fn test_csv_loader_empty_data() {
        let csv = "f1,label\n";
        let config = CsvLoaderConfig {
            feature_columns: vec![0],
            label_columns: vec![1],
            has_header: true,
            ..Default::default()
        };

        let result = CsvDataLoader::from_csv_string(csv, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_csv_loader_custom_delimiter() {
        let csv = "f1\tf2\tlabel\n1.0\t2.0\t0\n3.0\t4.0\t1\n";
        let config = CsvLoaderConfig {
            feature_columns: vec![0, 1],
            label_columns: vec![2],
            has_header: true,
            batch_size: 2,
            train_fraction: 1.0,
            delimiter: b'\t',
            ..Default::default()
        };

        let loader = CsvDataLoader::from_csv_string(csv, &config).unwrap();
        assert_eq!(loader.total_samples, 2);
        let (features, _) = loader.load_batch(0).unwrap();
        assert_eq!(features, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_binary_roundtrip() {
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let labels = vec![0.0, 1.0, 0.0];
        let features_per_sample = 2;
        let labels_per_sample = 1;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.hxdt");
        let path_str = path.to_str().unwrap();

        CsvDataLoader::write_binary(
            path_str, &features, &labels,
            features_per_sample, labels_per_sample,
        ).unwrap();

        let loader = CsvDataLoader::from_binary(path_str, 2, 1.0).unwrap();
        assert_eq!(loader.total_samples, 3);
        assert_eq!(loader.features_per_sample, 2);
        assert_eq!(loader.labels_per_sample, 1);

        let (f, l) = loader.load_batch(0).unwrap();
        assert_eq!(f, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(l, vec![0.0, 1.0]);
    }

    #[test]
    fn test_binary_checksum_validation() {
        let features = vec![1.0, 2.0];
        let labels = vec![0.0];

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.hxdt");
        let path_str = path.to_str().unwrap();

        CsvDataLoader::write_binary(path_str, &features, &labels, 2, 1).unwrap();

        // Corrupt the data
        let mut data = std::fs::read(path_str).unwrap();
        data[30] ^= 0xFF; // Flip a byte in the data section
        std::fs::write(path_str, &data).unwrap();

        let result = CsvDataLoader::from_binary(path_str, 1, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_binary_invalid_magic() {
        let data = vec![0u8; 60]; // All zeros, wrong magic
        let result = CsvDataLoader::from_binary_bytes(&data, 1, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_flat() {
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let labels = vec![0.0, 1.0, 0.0, 1.0];

        let loader = CsvDataLoader::from_flat(
            &features, &labels, 2, 1, 2, 0.75,
        ).unwrap();

        assert_eq!(loader.total_samples, 4);
        // 75% of 4 = 3 train samples -> 1 full batch + 1 partial
        assert!(loader.num_batches() >= 1);
    }

    #[test]
    fn test_dataloader_trait() {
        let csv = "f1,f2,label\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,0\n7.0,8.0,1\n";
        let config = CsvLoaderConfig {
            feature_columns: vec![0, 1],
            label_columns: vec![2],
            has_header: true,
            batch_size: 2,
            train_fraction: 1.0,
            ..Default::default()
        };

        let loader = CsvDataLoader::from_csv_string(csv, &config).unwrap();

        // Use as trait object
        let dyn_loader: &dyn DataLoader = &loader;
        assert_eq!(dyn_loader.num_batches(), 2);

        let (features, labels) = dyn_loader.load_batch(0).unwrap();
        assert_eq!(features.len(), 4); // 2 samples * 2 features
        assert_eq!(labels.len(), 2); // 2 samples * 1 label
    }

    // ========================================================================
    // CSV hardening tests
    // ========================================================================

    #[test]
    fn test_csv_loader_column_count_validation() {
        let csv = "f1,f2,label\n1.0,2.0,0\n3.0,4.0\n5.0,6.0,0\n";
        let config = CsvLoaderConfig {
            feature_columns: vec![0, 1],
            label_columns: vec![2],
            has_header: true,
            ..Default::default()
        };

        let result = CsvDataLoader::from_csv_string(csv, &config);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("expected 3 columns but found 2"), "got: {}", err_msg);
    }

    #[test]
    fn test_csv_loader_nan_rejection() {
        let csv = "f1,label\n1.0,0\nNaN,1\n3.0,0\n";
        let config = CsvLoaderConfig {
            feature_columns: vec![0],
            label_columns: vec![1],
            has_header: true,
            ..Default::default()
        };

        let result = CsvDataLoader::from_csv_string(csv, &config);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("NaN"), "got: {}", err_msg);
    }

    #[test]
    fn test_csv_loader_inf_rejection() {
        let csv = "f1,label\n1.0,0\nInf,1\n3.0,0\n";
        let config = CsvLoaderConfig {
            feature_columns: vec![0],
            label_columns: vec![1],
            has_header: true,
            ..Default::default()
        };

        let result = CsvDataLoader::from_csv_string(csv, &config);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("Inf"), "got: {}", err_msg);
    }

    #[test]
    fn test_csv_loader_max_file_size() {
        let csv = "f1,label\n1.0,0\n2.0,1\n";
        let config = CsvLoaderConfig {
            feature_columns: vec![0],
            label_columns: vec![1],
            has_header: true,
            max_file_bytes: Some(10), // very small limit
            ..Default::default()
        };

        let result = CsvDataLoader::from_csv_string(csv, &config);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("too large"), "got: {}", err_msg);
    }

    #[test]
    fn test_csv_loader_empty_columns_config() {
        let csv = "f1,label\n1.0,0\n";
        let config = CsvLoaderConfig {
            feature_columns: vec![],
            label_columns: vec![],
            has_header: true,
            ..Default::default()
        };

        let result = CsvDataLoader::from_csv_string(csv, &config);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("empty"), "got: {}", err_msg);
    }

    // ========================================================================
    // JSON loader tests
    // ========================================================================

    #[test]
    fn test_json_loader_basic() {
        let json = r#"[
            {"x": 1.0, "y": 2.0, "label": 0.0},
            {"x": 3.0, "y": 4.0, "label": 1.0},
            {"x": 5.0, "y": 6.0, "label": 0.0},
            {"x": 7.0, "y": 8.0, "label": 1.0},
            {"x": 9.0, "y": 10.0, "label": 0.0}
        ]"#;
        let config = JsonLoaderConfig {
            feature_keys: vec!["x".to_string(), "y".to_string()],
            label_keys: vec!["label".to_string()],
            batch_size: 100,
            train_fraction: 1.0,
            ..Default::default()
        };

        let loader = CsvDataLoader::from_json_string(json, &config).unwrap();
        assert_eq!(loader.total_samples, 5);
        assert_eq!(loader.features_per_sample, 2);
        assert_eq!(loader.labels_per_sample, 1);

        let (features, labels) = loader.load_batch(0).unwrap();
        assert_eq!(features.len(), 10); // 5 samples * 2 features
        assert_eq!(labels.len(), 5);
        assert_eq!(features[0], 1.0);
        assert_eq!(features[1], 2.0);
        assert_eq!(labels[0], 0.0);
    }

    #[test]
    fn test_json_loader_with_normalization() {
        let json = r#"[
            {"x": 10.0, "label": 0.0},
            {"x": 20.0, "label": 1.0},
            {"x": 30.0, "label": 0.0},
            {"x": 40.0, "label": 1.0},
            {"x": 50.0, "label": 0.0}
        ]"#;
        let config = JsonLoaderConfig {
            feature_keys: vec!["x".to_string()],
            label_keys: vec!["label".to_string()],
            batch_size: 100,
            normalization: Normalization::ZScore,
            train_fraction: 1.0,
            ..Default::default()
        };

        let loader = CsvDataLoader::from_json_string(json, &config).unwrap();
        let (features, _) = loader.load_batch(0).unwrap();
        let mean: f64 = features.iter().sum::<f64>() / features.len() as f64;
        assert!(mean.abs() < 1e-10, "z-score mean should be ~0, got {}", mean);
    }

    // ========================================================================
    // LibSVM loader tests
    // ========================================================================

    #[test]
    fn test_libsvm_loader_basic() {
        // 1-indexed features, 3 features total
        let libsvm = "0 1:1.0 2:2.0 3:3.0\n1 1:4.0 2:5.0 3:6.0\n0 1:7.0 2:8.0 3:9.0\n";
        let config = LibSvmLoaderConfig {
            num_features: 3,
            batch_size: 100,
            train_fraction: 1.0,
            ..Default::default()
        };

        let loader = CsvDataLoader::from_libsvm_string(libsvm, &config).unwrap();
        assert_eq!(loader.total_samples, 3);
        assert_eq!(loader.features_per_sample, 3);
        assert_eq!(loader.labels_per_sample, 1);

        let (features, labels) = loader.load_batch(0).unwrap();
        assert_eq!(features, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_eq!(labels, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_libsvm_loader_sparse() {
        // Sparse: only some features specified, rest should be 0.0
        let libsvm = "1 1:0.5 3:0.8 5:1.0\n0 2:0.3 4:0.7\n";
        let config = LibSvmLoaderConfig {
            num_features: 5,
            batch_size: 100,
            train_fraction: 1.0,
            ..Default::default()
        };

        let loader = CsvDataLoader::from_libsvm_string(libsvm, &config).unwrap();
        assert_eq!(loader.total_samples, 2);

        let (features, labels) = loader.load_batch(0).unwrap();
        // Sample 1: [0.5, 0.0, 0.8, 0.0, 1.0]
        assert_eq!(features[0], 0.5);
        assert_eq!(features[1], 0.0); // missing -> 0
        assert_eq!(features[2], 0.8);
        assert_eq!(features[3], 0.0); // missing -> 0
        assert_eq!(features[4], 1.0);
        // Sample 2: [0.0, 0.3, 0.0, 0.7, 0.0]
        assert_eq!(features[5], 0.0); // missing -> 0
        assert_eq!(features[6], 0.3);
        assert_eq!(features[7], 0.0); // missing -> 0
        assert_eq!(features[8], 0.7);
        assert_eq!(features[9], 0.0); // missing -> 0

        assert_eq!(labels, vec![1.0, 0.0]);
    }

    #[test]
    fn test_libsvm_loader_zero_indexed() {
        // Zero-indexed features
        let libsvm = "1 0:1.0 1:2.0 2:3.0\n0 0:4.0 1:5.0 2:6.0\n";
        let config = LibSvmLoaderConfig {
            num_features: 3,
            batch_size: 100,
            train_fraction: 1.0,
            zero_indexed: true,
            ..Default::default()
        };

        let loader = CsvDataLoader::from_libsvm_string(libsvm, &config).unwrap();
        assert_eq!(loader.total_samples, 2);

        let (features, labels) = loader.load_batch(0).unwrap();
        assert_eq!(features, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(labels, vec![1.0, 0.0]);
    }

    // ========================================================================
    // Column name sanitization tests
    // ========================================================================

    #[test]
    fn test_csv_loader_safe_column_names() {
        let csv = "feature_1,feature_2,label\n1.0,2.0,0\n3.0,4.0,1\n";
        let config = CsvLoaderConfig {
            feature_columns: vec![0, 1],
            label_columns: vec![2],
            has_header: true,
            batch_size: 100,
            train_fraction: 1.0,
            ..Default::default()
        };

        let loader = CsvDataLoader::from_csv_string(csv, &config).unwrap();
        assert_eq!(loader.total_samples, 2);
    }

    #[test]
    fn test_csv_loader_rejects_path_traversal_column_name() {
        let csv = "../etc/passwd,label\n1.0,0\n2.0,1\n";
        let config = CsvLoaderConfig {
            feature_columns: vec![0],
            label_columns: vec![1],
            has_header: true,
            ..Default::default()
        };

        let result = CsvDataLoader::from_csv_string(csv, &config);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("unsafe column name"), "got: {}", err_msg);
    }

    #[test]
    fn test_csv_loader_rejects_slash_in_column_name() {
        let csv = "path/to/file,label\n1.0,0\n2.0,1\n";
        let config = CsvLoaderConfig {
            feature_columns: vec![0],
            label_columns: vec![1],
            has_header: true,
            ..Default::default()
        };

        let result = CsvDataLoader::from_csv_string(csv, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_csv_loader_rejects_null_byte_in_column_name() {
        let csv = "col\0name,label\n1.0,0\n2.0,1\n";
        let config = CsvLoaderConfig {
            feature_columns: vec![0],
            label_columns: vec![1],
            has_header: true,
            ..Default::default()
        };

        let result = CsvDataLoader::from_csv_string(csv, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_csv_loader_rejects_control_char_in_column_name() {
        let csv = "col\x01name,label\n1.0,0\n2.0,1\n";
        let config = CsvLoaderConfig {
            feature_columns: vec![0],
            label_columns: vec![1],
            has_header: true,
            ..Default::default()
        };

        let result = CsvDataLoader::from_csv_string(csv, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_csv_loader_allows_no_header() {
        // When has_header is false, no column name validation occurs
        let csv = "1.0,0\n2.0,1\n";
        let config = CsvLoaderConfig {
            feature_columns: vec![0],
            label_columns: vec![1],
            has_header: false,
            batch_size: 100,
            train_fraction: 1.0,
            ..Default::default()
        };

        let loader = CsvDataLoader::from_csv_string(csv, &config).unwrap();
        assert_eq!(loader.total_samples, 2);
    }
}
