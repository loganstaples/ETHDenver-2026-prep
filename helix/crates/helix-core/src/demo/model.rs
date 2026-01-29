//! Demo Model Training for HELIX.
//!
//! Provides a simple demo model that can be used for
//! showcasing the distributed training pipeline.

use serde::{Deserialize, Serialize};

/// Demo model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoModelConfig {
    /// Model name.
    pub name: String,
    /// Input dimension.
    pub input_dim: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Output dimension.
    pub output_dim: usize,
    /// Number of layers.
    pub num_layers: usize,
    /// Activation function.
    pub activation: String,
    /// Dropout rate.
    pub dropout: f64,
}

impl Default for DemoModelConfig {
    fn default() -> Self {
        Self {
            name: "helix-demo-mlp".to_string(),
            input_dim: 784,  // MNIST-like
            hidden_dim: 256,
            output_dim: 10,
            num_layers: 3,
            activation: "relu".to_string(),
            dropout: 0.1,
        }
    }
}

/// Simple demo model for showcasing HELIX.
#[derive(Debug, Clone)]
pub struct DemoModel {
    config: DemoModelConfig,
    weights: Vec<Vec<f64>>,
    biases: Vec<Vec<f64>>,
}

impl DemoModel {
    /// Creates a new demo model.
    pub fn new(config: DemoModelConfig) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        // Initialize weights with Xavier initialization
        let mut prev_dim = config.input_dim;
        for i in 0..config.num_layers {
            let next_dim = if i == config.num_layers - 1 {
                config.output_dim
            } else {
                config.hidden_dim
            };

            let scale = (2.0 / (prev_dim + next_dim) as f64).sqrt();
            let layer_weights: Vec<f64> = (0..prev_dim * next_dim)
                .map(|j| ((j * 7 + 13) % 100) as f64 / 100.0 * scale - scale / 2.0)
                .collect();
            
            weights.push(layer_weights);
            biases.push(vec![0.0; next_dim]);
            
            prev_dim = next_dim;
        }

        Self { config, weights, biases }
    }

    /// Forward pass.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut x = input.to_vec();

        for (layer_idx, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let out_dim = b.len();
            let in_dim = x.len();
            
            let mut output = vec![0.0; out_dim];
            
            // Matrix multiply
            for i in 0..out_dim {
                let mut sum = b[i];
                for j in 0..in_dim {
                    sum += x[j] * w[j * out_dim + i];
                }
                output[i] = sum;
            }

            // Apply activation (except last layer)
            if layer_idx < self.config.num_layers - 1 {
                for val in &mut output {
                    *val = self.apply_activation(*val);
                }
            }

            x = output;
        }

        // Softmax for classification
        let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = x.iter().map(|v| (v - max_val).exp()).sum();
        x.iter().map(|v| (v - max_val).exp() / exp_sum).collect()
    }

    fn apply_activation(&self, x: f64) -> f64 {
        match self.config.activation.as_str() {
            "relu" => x.max(0.0),
            "sigmoid" => 1.0 / (1.0 + (-x).exp()),
            "tanh" => x.tanh(),
            _ => x,
        }
    }

    /// Computes loss and gradients.
    pub fn train_step(&mut self, inputs: &[Vec<f64>], targets: &[usize], lr: f64) -> f64 {
        let mut total_loss = 0.0;
        
        for (input, &target) in inputs.iter().zip(targets.iter()) {
            let output = self.forward(input);
            
            // Cross-entropy loss
            let loss = -output[target].ln();
            total_loss += loss;
            
            // Simplified gradient update (demo purposes)
            let mut grad = output.clone();
            grad[target] -= 1.0;
            
            // Update last layer weights
            if let Some(last_weights) = self.weights.last_mut() {
                for i in 0..last_weights.len() {
                    last_weights[i] -= lr * grad[i % self.config.output_dim] * 0.01;
                }
            }
        }

        total_loss / inputs.len() as f64
    }

    /// Gets model parameters.
    pub fn get_parameters(&self) -> Vec<f64> {
        let mut params = Vec::new();
        for w in &self.weights {
            params.extend(w);
        }
        for b in &self.biases {
            params.extend(b);
        }
        params
    }

    /// Gets parameter count.
    pub fn num_parameters(&self) -> usize {
        self.weights.iter().map(|w| w.len()).sum::<usize>() +
        self.biases.iter().map(|b| b.len()).sum::<usize>()
    }

    /// Applies gradient update.
    pub fn apply_gradients(&mut self, gradients: &[f64], lr: f64) {
        let mut idx = 0;
        
        for w in &mut self.weights {
            for weight in w.iter_mut() {
                if idx < gradients.len() {
                    *weight -= lr * gradients[idx];
                    idx += 1;
                }
            }
        }
        
        for b in &mut self.biases {
            for bias in b.iter_mut() {
                if idx < gradients.len() {
                    *bias -= lr * gradients[idx];
                    idx += 1;
                }
            }
        }
    }

    /// Serializes model to bytes.
    pub fn serialize(&self) -> Vec<u8> {
        let params = self.get_parameters();
        let bytes: Vec<u8> = params.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        bytes
    }
}

/// Training loop for demo model.
pub struct DemoTrainer {
    model: DemoModel,
    learning_rate: f64,
    epochs: usize,
    batch_size: usize,
}

impl DemoTrainer {
    /// Creates a new trainer.
    pub fn new(config: DemoModelConfig, lr: f64, epochs: usize, batch_size: usize) -> Self {
        Self {
            model: DemoModel::new(config),
            learning_rate: lr,
            epochs,
            batch_size,
        }
    }

    /// Trains on synthetic data.
    pub fn train_synthetic(&mut self) -> TrainingResult {
        let mut history = Vec::new();
        
        // Generate synthetic data
        let (inputs, targets) = self.generate_synthetic_data(1000);
        
        for epoch in 0..self.epochs {
            let mut epoch_loss = 0.0;
            let num_batches = (inputs.len() + self.batch_size - 1) / self.batch_size;
            
            for batch_idx in 0..num_batches {
                let start = batch_idx * self.batch_size;
                let end = (start + self.batch_size).min(inputs.len());
                
                let batch_inputs: Vec<_> = inputs[start..end].to_vec();
                let batch_targets: Vec<_> = targets[start..end].to_vec();
                
                let loss = self.model.train_step(&batch_inputs, &batch_targets, self.learning_rate);
                epoch_loss += loss;
            }
            
            let avg_loss = epoch_loss / num_batches as f64;
            history.push(avg_loss);
            
            // Compute accuracy
            let accuracy = self.compute_accuracy(&inputs, &targets);
            
            if epoch % 10 == 0 || epoch == self.epochs - 1 {
                println!("Epoch {}: loss={:.4}, accuracy={:.2}%", epoch, avg_loss, accuracy * 100.0);
            }
        }

        let final_accuracy = self.compute_accuracy(&inputs, &targets);
        
        TrainingResult {
            final_loss: *history.last().unwrap_or(&1.0),
            final_accuracy,
            epochs_completed: self.epochs,
            loss_history: history,
        }
    }

    fn generate_synthetic_data(&self, n: usize) -> (Vec<Vec<f64>>, Vec<usize>) {
        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        
        for i in 0..n {
            // Generate random input
            let input: Vec<f64> = (0..self.model.config.input_dim)
                .map(|j| ((i * 31 + j * 17) % 100) as f64 / 100.0)
                .collect();
            
            // Target based on sum of features
            let target = (input.iter().sum::<f64>() * 10.0) as usize % self.model.config.output_dim;
            
            inputs.push(input);
            targets.push(target);
        }
        
        (inputs, targets)
    }

    fn compute_accuracy(&self, inputs: &[Vec<f64>], targets: &[usize]) -> f64 {
        let mut correct = 0;
        
        for (input, &target) in inputs.iter().zip(targets.iter()) {
            let output = self.model.forward(input);
            let predicted = output.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            
            if predicted == target {
                correct += 1;
            }
        }
        
        correct as f64 / inputs.len() as f64
    }

    /// Gets the trained model.
    pub fn get_model(&self) -> &DemoModel {
        &self.model
    }
}

/// Training result.
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Final loss.
    pub final_loss: f64,
    /// Final accuracy.
    pub final_accuracy: f64,
    /// Epochs completed.
    pub epochs_completed: usize,
    /// Loss history.
    pub loss_history: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_model_creation() {
        let config = DemoModelConfig::default();
        let model = DemoModel::new(config);
        
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_forward_pass() {
        let config = DemoModelConfig {
            input_dim: 16,
            hidden_dim: 8,
            output_dim: 4,
            num_layers: 2,
            ..Default::default()
        };
        let model = DemoModel::new(config);
        
        let input = vec![0.5; 16];
        let output = model.forward(&input);
        
        assert_eq!(output.len(), 4);
        // Softmax output should sum to 1
        let sum: f64 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_training() {
        let config = DemoModelConfig {
            input_dim: 16,
            hidden_dim: 8,
            output_dim: 4,
            num_layers: 2,
            ..Default::default()
        };
        
        let mut trainer = DemoTrainer::new(config, 0.01, 10, 32);
        let result = trainer.train_synthetic();
        
        assert_eq!(result.epochs_completed, 10);
        assert!(result.final_accuracy > 0.0);
    }
}
