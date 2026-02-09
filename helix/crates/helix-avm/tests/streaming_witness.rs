//! Integration tests for the streaming witness collector and deserialization.

use helix_avm::vm::trace::TraceStep;
use helix_avm::vm::Instruction;
use helix_avm::witness::collector::{StreamingConfig, StreamingWitnessCollector};
use helix_avm::witness::serializer;
use helix_core::types::{BoundedTensor, BoundedValue, ErrorMargin};

/// Helper: build a TraceStep with the given output values and cumulative error.
fn make_step(output_vals: &[f64], cumulative_error: f64) -> TraceStep {
    let output = if output_vals.is_empty() {
        None
    } else {
        let data: Vec<BoundedValue<f64>> =
            output_vals.iter().map(|&v| BoundedValue::exact(v)).collect();
        Some(BoundedTensor::new(data, vec![output_vals.len()]))
    };

    TraceStep {
        pc: 0,
        instruction: Instruction::halt(),
        inputs: vec![],
        output,
        step_error: ErrorMargin::ZERO,
        cumulative_error,
    }
}

#[test]
fn test_streaming_witness_single_chunk() {
    let config = StreamingConfig {
        chunk_size: 100,
        ..StreamingConfig::default()
    };
    let mut collector = StreamingWitnessCollector::new(config);

    // Push 3 steps — all fit in one chunk (chunk_size=100).
    for i in 0..3 {
        let step = make_step(&[i as f64, (i + 1) as f64], 0.01 * (i + 1) as f64);
        collector.push_step(&step);
    }

    let summary = collector.finish();

    assert_eq!(summary.total_operations, 3);
    assert_eq!(summary.total_chunks, 1);
    assert_eq!(summary.chunks.len(), 1);
    assert_eq!(summary.chunks[0].num_operations, 3);
}

#[test]
fn test_streaming_auto_flush() {
    let config = StreamingConfig {
        chunk_size: 2, // flush every 2 ops
        ..StreamingConfig::default()
    };
    let mut collector = StreamingWitnessCollector::new(config);

    // Push 5 steps: should produce 2 auto-flushed chunks (2 ops each)
    // + 1 final chunk (1 op) on finish.
    for i in 0..5 {
        let step = make_step(&[i as f64], 0.1 * (i + 1) as f64);
        collector.push_step(&step);
    }

    let summary = collector.finish();

    assert_eq!(summary.total_operations, 5);
    // 2 auto-flushed + 1 from finish
    assert_eq!(summary.total_chunks, 3);
    assert_eq!(summary.chunks[0].num_operations, 2);
    assert_eq!(summary.chunks[1].num_operations, 2);
    assert_eq!(summary.chunks[2].num_operations, 1);
    // Chunk indices should be sequential
    for (idx, hdr) in summary.chunks.iter().enumerate() {
        assert_eq!(hdr.chunk_index, idx);
    }
}

#[test]
fn test_streaming_boundary_hash_chaining() {
    let config = StreamingConfig {
        chunk_size: 1, // flush after every single op
        include_intermediates: true,
        compute_boundary_hash: true,
    };
    let mut collector = StreamingWitnessCollector::new(config);

    // Push 3 distinct steps
    for i in 0..3 {
        let step = make_step(&[(i * 10) as f64], 0.0);
        collector.push_step(&step);
    }

    let summary = collector.finish();

    assert_eq!(summary.total_chunks, 3);

    // First chunk's input_hash is all zeros (no prior chunk).
    assert_eq!(summary.chunks[0].input_hash, [0u8; 32]);

    // Chain: output_hash[N] == input_hash[N+1]
    for window in summary.chunks.windows(2) {
        assert_eq!(
            window[0].output_hash, window[1].input_hash,
            "output_hash of chunk {} must equal input_hash of chunk {}",
            window[0].chunk_index, window[1].chunk_index
        );
    }

    // Each chunk has different data, so boundary hashes should differ.
    let h0 = summary.chunks[0].output_hash;
    let h1 = summary.chunks[1].output_hash;
    let h2 = summary.chunks[2].output_hash;
    assert_ne!(h0, h1, "Distinct chunks should have distinct boundary hashes");
    assert_ne!(h1, h2, "Distinct chunks should have distinct boundary hashes");
    assert_ne!(h0, h2, "Distinct chunks should have distinct boundary hashes");
}

#[test]
fn test_deserialize_roundtrip() {
    // Single tensor
    let data = vec![
        BoundedValue::exact(1.0),
        BoundedValue::exact(2.0),
        BoundedValue::exact(3.0),
        BoundedValue::exact(4.0),
    ];
    let tensor = BoundedTensor::new(data, vec![2, 2]);
    let bytes = serializer::serialize_tensor(&tensor);
    let (recovered, consumed) = serializer::deserialize_tensor(&bytes).unwrap();
    assert_eq!(consumed, bytes.len());
    assert_eq!(recovered.shape(), tensor.shape());
    for (a, b) in recovered.data().iter().zip(tensor.data().iter()) {
        assert!((a.value() - b.value()).abs() < 1e-15);
    }

    // Multiple tensors
    let t1 = BoundedTensor::zeros(vec![3]);
    let data2 = vec![BoundedValue::exact(5.0), BoundedValue::exact(6.0)];
    let t2 = BoundedTensor::new(data2, vec![2]);
    let multi_bytes = serializer::serialize_tensors(&[t1.clone(), t2.clone()]);
    let recovered_list = serializer::deserialize_tensors(&multi_bytes).unwrap();
    assert_eq!(recovered_list.len(), 2);
    assert_eq!(recovered_list[0].shape(), t1.shape());
    assert_eq!(recovered_list[1].shape(), t2.shape());
    assert!((recovered_list[1].data()[0].value() - 5.0).abs() < 1e-15);
    assert!((recovered_list[1].data()[1].value() - 6.0).abs() < 1e-15);
}
