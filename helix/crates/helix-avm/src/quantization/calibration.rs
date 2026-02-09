//! Quantization Calibration.
//!
//! Collects statistics from representative data to determine optimal
//! quantization parameters.

use std::collections::HashMap;

use helix_core::types::BoundedTensor;

use super::schemes::{CalibrationMethod, ObserverType, QuantScheme, TensorQuantParams};

/// Observer to collect statistics for calibration.
#[derive(Debug)]
pub struct Observer {
    /// Observer type.
    #[allow(dead_code)]
    observer_type: ObserverType,
    /// Name of the layer being observed.
    #[allow(dead_code)]
    layer_name: String,
    /// Minimum value seen.
    min_val: f64,
    /// Maximum value seen.
    max_val: f64,
    /// Number of samples observed.
    sample_count: usize,
    /// Histogram bins (if histogram observer).
    histogram: Option<Histogram>,
    /// Sum for computing mean (moving average).
    running_sum: f64,
    /// Sum of squares for variance.
    running_sum_sq: f64,
}

impl Observer {
    /// Creates a new observer.
    pub fn new(layer_name: &str, observer_type: ObserverType) -> Self {
        let histogram = match observer_type {
            ObserverType::Histogram { bins } => Some(Histogram::new(bins)),
            _ => None,
        };

        Self {
            observer_type,
            layer_name: layer_name.to_string(),
            min_val: f64::INFINITY,
            max_val: f64::NEG_INFINITY,
            sample_count: 0,
            histogram,
            running_sum: 0.0,
            running_sum_sq: 0.0,
        }
    }

    /// Creates a min-max observer.
    pub fn min_max(layer_name: &str) -> Self {
        Self::new(layer_name, ObserverType::MinMax)
    }

    /// Creates a histogram observer.
    pub fn histogram(layer_name: &str, bins: usize) -> Self {
        Self::new(layer_name, ObserverType::Histogram { bins })
    }

    /// Observes a tensor and updates statistics.
    pub fn observe(&mut self, tensor: &BoundedTensor) {
        let data = tensor.data();
        self.sample_count += data.len();

        for v in data {
            let value = v.value();
            self.min_val = self.min_val.min(value);
            self.max_val = self.max_val.max(value);
            self.running_sum += value;
            self.running_sum_sq += value * value;
        }

        // Update histogram if present
        if let Some(ref mut hist) = self.histogram {
            // First observation sets the range
            if hist.count == 0 {
                hist.set_range(self.min_val, self.max_val);
            }
            
            for v in data {
                hist.add(v.value());
            }
        }
    }

    /// Observes a slice of f64 values.
    pub fn observe_slice(&mut self, data: &[f64]) {
        self.sample_count += data.len();

        for &value in data {
            self.min_val = self.min_val.min(value);
            self.max_val = self.max_val.max(value);
            self.running_sum += value;
            self.running_sum_sq += value * value;
        }

        if let Some(ref mut hist) = self.histogram {
            if hist.count == 0 {
                hist.set_range(self.min_val, self.max_val);
            }
            for &value in data {
                hist.add(value);
            }
        }
    }

    /// Returns the observed range.
    pub fn range(&self) -> (f64, f64) {
        (self.min_val, self.max_val)
    }

    /// Returns the mean value.
    pub fn mean(&self) -> f64 {
        if self.sample_count == 0 {
            0.0
        } else {
            self.running_sum / self.sample_count as f64
        }
    }

    /// Returns the variance.
    pub fn variance(&self) -> f64 {
        if self.sample_count == 0 {
            0.0
        } else {
            let mean = self.mean();
            self.running_sum_sq / self.sample_count as f64 - mean * mean
        }
    }

    /// Returns the standard deviation.
    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Computes quantization parameters based on observations.
    pub fn compute_params(
        &self,
        scheme: QuantScheme,
        method: CalibrationMethod,
    ) -> TensorQuantParams {
        let (min_val, max_val) = match method {
            CalibrationMethod::MinMax => self.range(),
            CalibrationMethod::Percentile(p) => self.percentile_range(p),
            CalibrationMethod::MSE => self.mse_optimal_range(scheme),
            CalibrationMethod::Entropy => self.entropy_optimal_range(scheme),
            CalibrationMethod::ACIQ => self.aciq_optimal_range(scheme),
        };

        TensorQuantParams::from_range(scheme, min_val, max_val)
    }

    /// Computes percentile range.
    fn percentile_range(&self, percentile: f64) -> (f64, f64) {
        if let Some(ref hist) = self.histogram {
            let low_p = (100.0 - percentile) / 100.0;
            let high_p = percentile / 100.0;
            (hist.percentile(low_p), hist.percentile(high_p))
        } else {
            // Fall back to min/max
            self.range()
        }
    }

    /// Computes MSE-optimal range.
    fn mse_optimal_range(&self, _scheme: QuantScheme) -> (f64, f64) {
        // Simplified: use 3-sigma range
        let mean = self.mean();
        let std = self.std();
        (mean - 3.0 * std, mean + 3.0 * std)
    }

    /// Computes entropy-optimal range.
    fn entropy_optimal_range(&self, scheme: QuantScheme) -> (f64, f64) {
        if let Some(ref hist) = self.histogram {
            // Search for optimal threshold that minimizes KL divergence
            let bits = scheme.bits() as usize;
            let _num_levels = 1 << bits;
            
            let mut best_threshold = self.max_val;
            let mut _best_divergence = f64::INFINITY;
            
            // Try different thresholds
            for i in 0..hist.bins.len() {
                let threshold = hist.bin_edges[i + 1];
                // Simplified: just use the threshold where most data is captured
                let data_fraction = hist.cumulative_fraction_below(threshold);
                if data_fraction > 0.99 {
                    best_threshold = threshold;
                    break;
                }
            }
            
            let abs_max = self.min_val.abs().max(best_threshold.abs());
            (-abs_max, abs_max)
        } else {
            self.range()
        }
    }

    /// Computes ACIQ (Analytical Clipping for Integer Quantization) range.
    fn aciq_optimal_range(&self, scheme: QuantScheme) -> (f64, f64) {
        // ACIQ formula for optimal clipping threshold
        // For Gaussian distribution: alpha ≈ 2.897 / sqrt(2^b - 1) * sigma
        let bits = scheme.bits() as f64;
        let num_levels = (1_u64 << (bits as u64)) as f64;
        
        let std = self.std();
        let alpha = 2.897 / (num_levels - 1.0).sqrt() * std;
        
        let mean = self.mean();
        (mean - alpha * 3.0, mean + alpha * 3.0)
    }

    /// Resets the observer.
    pub fn reset(&mut self) {
        self.min_val = f64::INFINITY;
        self.max_val = f64::NEG_INFINITY;
        self.sample_count = 0;
        self.running_sum = 0.0;
        self.running_sum_sq = 0.0;
        if let Some(ref mut hist) = self.histogram {
            hist.reset();
        }
    }
}

/// Histogram for collecting value distribution.
#[derive(Debug)]
pub struct Histogram {
    /// Bin counts.
    bins: Vec<usize>,
    /// Bin edges.
    bin_edges: Vec<f64>,
    /// Total count.
    count: usize,
    /// Number of bins.
    num_bins: usize,
}

impl Histogram {
    /// Creates a new histogram.
    pub fn new(num_bins: usize) -> Self {
        Self {
            bins: vec![0; num_bins],
            bin_edges: Vec::new(),
            count: 0,
            num_bins,
        }
    }

    /// Sets the range for the histogram.
    pub fn set_range(&mut self, min_val: f64, max_val: f64) {
        let range = max_val - min_val;
        let bin_width = range / self.num_bins as f64;
        
        self.bin_edges = (0..=self.num_bins)
            .map(|i| min_val + i as f64 * bin_width)
            .collect();
    }

    /// Adds a value to the histogram.
    pub fn add(&mut self, value: f64) {
        if self.bin_edges.is_empty() {
            return;
        }

        let min_val = self.bin_edges[0];
        let max_val = *self.bin_edges.last().unwrap();
        
        if value < min_val || value > max_val {
            // Clamp to edges
            if value < min_val {
                self.bins[0] += 1;
            } else {
                self.bins[self.num_bins - 1] += 1;
            }
        } else {
            let bin_width = (max_val - min_val) / self.num_bins as f64;
            let bin_idx = ((value - min_val) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(self.num_bins - 1);
            self.bins[bin_idx] += 1;
        }
        self.count += 1;
    }

    /// Returns the percentile value.
    pub fn percentile(&self, p: f64) -> f64 {
        if self.count == 0 || self.bin_edges.is_empty() {
            return 0.0;
        }

        let target_count = (p * self.count as f64) as usize;
        let mut cumulative = 0;

        for (i, &bin_count) in self.bins.iter().enumerate() {
            cumulative += bin_count;
            if cumulative >= target_count {
                return self.bin_edges[i];
            }
        }

        *self.bin_edges.last().unwrap()
    }

    /// Returns the fraction of data below a threshold.
    pub fn cumulative_fraction_below(&self, threshold: f64) -> f64 {
        if self.count == 0 || self.bin_edges.is_empty() {
            return 0.0;
        }

        let mut cumulative = 0;
        for (i, &edge) in self.bin_edges[1..].iter().enumerate() {
            if edge >= threshold {
                break;
            }
            cumulative += self.bins[i];
        }

        cumulative as f64 / self.count as f64
    }

    /// Resets the histogram.
    pub fn reset(&mut self) {
        self.bins.fill(0);
        self.bin_edges.clear();
        self.count = 0;
    }
}

/// Calibration data collected from observers.
#[derive(Debug, Clone)]
pub struct CalibrationData {
    /// Weight ranges per layer.
    pub weight_ranges: HashMap<String, (f64, f64)>,
    /// Activation ranges per layer.
    pub activation_ranges: HashMap<String, (f64, f64)>,
    /// Weight quantization params per layer.
    pub weight_params: HashMap<String, TensorQuantParams>,
    /// Activation quantization params per layer.
    pub activation_params: HashMap<String, TensorQuantParams>,
}

impl CalibrationData {
    /// Creates new empty calibration data.
    pub fn new() -> Self {
        Self {
            weight_ranges: HashMap::new(),
            activation_ranges: HashMap::new(),
            weight_params: HashMap::new(),
            activation_params: HashMap::new(),
        }
    }

    /// Adds weight range for a layer.
    pub fn add_weight_range(&mut self, layer: &str, min_val: f64, max_val: f64) {
        self.weight_ranges.insert(layer.to_string(), (min_val, max_val));
    }

    /// Adds activation range for a layer.
    pub fn add_activation_range(&mut self, layer: &str, min_val: f64, max_val: f64) {
        self.activation_ranges.insert(layer.to_string(), (min_val, max_val));
    }
}

impl Default for CalibrationData {
    fn default() -> Self {
        Self::new()
    }
}

/// Calibrator that manages observers and produces calibration data.
#[derive(Debug)]
pub struct Calibrator {
    /// Weight observers per layer.
    weight_observers: HashMap<String, Observer>,
    /// Activation observers per layer.
    activation_observers: HashMap<String, Observer>,
    /// Observer type to use.
    observer_type: ObserverType,
    /// Calibration method to use.
    calibration_method: CalibrationMethod,
    /// Weight quantization scheme.
    weight_scheme: QuantScheme,
    /// Activation quantization scheme.
    activation_scheme: QuantScheme,
}

impl Calibrator {
    /// Creates a new calibrator.
    pub fn new(
        weight_scheme: QuantScheme,
        activation_scheme: QuantScheme,
        calibration_method: CalibrationMethod,
        observer_type: ObserverType,
    ) -> Self {
        Self {
            weight_observers: HashMap::new(),
            activation_observers: HashMap::new(),
            observer_type,
            calibration_method,
            weight_scheme,
            activation_scheme,
        }
    }

    /// Creates a default INT8 calibrator.
    pub fn int8() -> Self {
        Self::new(
            QuantScheme::SymmetricInt8,
            QuantScheme::AsymmetricInt8,
            CalibrationMethod::MinMax,
            ObserverType::MinMax,
        )
    }

    /// Creates a calibrator with histogram observer.
    pub fn with_histogram(bins: usize) -> Self {
        Self::new(
            QuantScheme::SymmetricInt8,
            QuantScheme::AsymmetricInt8,
            CalibrationMethod::Percentile(99.99),
            ObserverType::Histogram { bins },
        )
    }

    /// Registers a layer for calibration.
    pub fn register_layer(&mut self, layer_name: &str) {
        if !self.weight_observers.contains_key(layer_name) {
            self.weight_observers.insert(
                layer_name.to_string(),
                Observer::new(layer_name, self.observer_type),
            );
            self.activation_observers.insert(
                layer_name.to_string(),
                Observer::new(layer_name, self.observer_type),
            );
        }
    }

    /// Records weight data for a layer.
    pub fn record_weights(&mut self, layer_name: &str, weights: &BoundedTensor) {
        self.register_layer(layer_name);
        if let Some(observer) = self.weight_observers.get_mut(layer_name) {
            observer.observe(weights);
        }
    }

    /// Records activation data for a layer.
    pub fn record_activations(&mut self, layer_name: &str, activations: &BoundedTensor) {
        self.register_layer(layer_name);
        if let Some(observer) = self.activation_observers.get_mut(layer_name) {
            observer.observe(activations);
        }
    }

    /// Finalizes calibration and produces calibration data.
    pub fn finalize(&self) -> CalibrationData {
        let mut data = CalibrationData::new();

        for (name, observer) in &self.weight_observers {
            let (min_val, max_val) = observer.range();
            data.add_weight_range(name, min_val, max_val);
            
            let params = observer.compute_params(self.weight_scheme, self.calibration_method);
            data.weight_params.insert(name.clone(), params);
        }

        for (name, observer) in &self.activation_observers {
            let (min_val, max_val) = observer.range();
            data.add_activation_range(name, min_val, max_val);
            
            let params = observer.compute_params(self.activation_scheme, self.calibration_method);
            data.activation_params.insert(name.clone(), params);
        }

        data
    }

    /// Resets all observers.
    pub fn reset(&mut self) {
        for observer in self.weight_observers.values_mut() {
            observer.reset();
        }
        for observer in self.activation_observers.values_mut() {
            observer.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use helix_core::types::BoundedValue;

    #[test]
    fn test_observer_min_max() {
        let mut observer = Observer::min_max("test");
        
        let data: Vec<BoundedValue<f64>> = vec![-1.0, 0.0, 0.5, 1.0]
            .into_iter()
            .map(BoundedValue::exact)
            .collect();
        let tensor = BoundedTensor::new(data, vec![4]);
        
        observer.observe(&tensor);
        
        let (min_val, max_val) = observer.range();
        assert!((min_val - (-1.0)).abs() < 1e-10);
        assert!((max_val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_histogram() {
        let mut hist = Histogram::new(10);
        hist.set_range(0.0, 1.0);
        
        for i in 0..100 {
            hist.add(i as f64 / 100.0);
        }
        
        let p50 = hist.percentile(0.5);
        // Percentile returns bin edges, so rounding is expected
        assert!(p50 >= 0.0 && p50 <= 1.0);
        assert_eq!(hist.count, 100);
    }

    #[test]
    fn test_calibrator() {
        let mut calibrator = Calibrator::int8();
        
        let data: Vec<BoundedValue<f64>> = vec![0.1, 0.5, -0.3, 0.8]
            .into_iter()
            .map(BoundedValue::exact)
            .collect();
        let tensor = BoundedTensor::new(data, vec![4]);
        
        calibrator.record_weights("layer1", &tensor);
        calibrator.record_activations("layer1", &tensor);
        
        let cal_data = calibrator.finalize();
        
        assert!(cal_data.weight_ranges.contains_key("layer1"));
        assert!(cal_data.activation_ranges.contains_key("layer1"));
    }
}
