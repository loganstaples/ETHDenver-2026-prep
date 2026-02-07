use super::state_circuit::{StateTransitionCircuit, StateCircuitConfig};
use crate::pipeline::ProverPipeline;
use helix_circuits::halo2curves::bn256::Fr;

pub struct StateProver {
    pub pipeline: ProverPipeline<StateTransitionCircuit>,
}

impl StateProver {
    pub fn new(k: u32) -> Self {
        let mut pipeline = ProverPipeline::new(k);
        let empty_circuit = StateTransitionCircuit::default();
        pipeline.setup(&empty_circuit);
        Self { pipeline }
    }

    pub fn prove(
        &self,
        old_w: Fr,
        old_err: Fr,
        grad: Fr,
        grad_err: Fr,
        lr: Fr,
        new_w: Fr,
        new_err: Fr,
    ) -> Vec<u8> {
        let circuit = StateTransitionCircuit {
            old_w: Some(old_w),
            old_err: Some(old_err),
            grad: Some(grad),
            grad_err: Some(grad_err),
            lr: Some(lr),
            new_w: Some(new_w),
            new_err: Some(new_err),
        };

        // Public inputs?
        // Usually commitment to (old, new, grad).
        // For this demo stage, we haven't exposed public inputs in the circuit yet (Instance Column).
        // The current StateTransitionCircuit doesn't implement instance columns in configure/synthesize.
        // So public_inputs is empty.
        self.pipeline.prove(&circuit, &[])
            .unwrap_or_else(|e| {
                tracing::error!("State transition proof generation failed: {e}");
                Vec::new()
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_prover_init() {
        // Just verify we can setup the pipeline without crash
        // K=9 should be enough
        let _prover = StateProver::new(9);
    }
}
