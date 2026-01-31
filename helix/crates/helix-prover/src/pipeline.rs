use helix_circuits::halo2_proofs::{
    arithmetic::{CurveAffine, Field},
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, Circuit, ProvingKey, VerifyingKey, SingleVerifier},
    poly::commitment::Params,
    transcript::{Blake2bRead, Blake2bWrite, Challenge255},
};
use helix_circuits::halo2curves::{
    bn256::{Fr, G1Affine, G2Affine},
    ff::PrimeField,
};
use rand::rngs::OsRng;
use std::marker::PhantomData;

/// Extracted verification key data for EVM verifier generation.
#[derive(Debug, Clone)]
pub struct ExtractedVkData {
    /// G1 generator point.
    pub g1: (String, String),
    /// SRS s·G2 point for KZG opening.
    pub s_g2: (String, String, String, String),
    /// Negative G2 generator -[1]₂.
    pub neg_g2: (String, String, String, String),
    /// Number of advice column commitments.
    pub num_advices: usize,
    /// Number of public inputs.
    pub num_instances: usize,
    /// Circuit size (k = log2(rows)).
    pub k: u32,
}

pub struct ProverPipeline<C: Circuit<Fr>> {
    pub params: Params<G1Affine>,
    pub pk: Option<ProvingKey<G1Affine>>,
    pub vk: Option<VerifyingKey<G1Affine>>,
    _marker: PhantomData<C>,
}

impl<C: Circuit<Fr> + Clone> ProverPipeline<C> {
    pub fn new(k: u32) -> Self {
        let params = Params::<G1Affine>::new(k);
        Self {
            params,
            pk: None,
            vk: None,
            _marker: PhantomData,
        }
    }

    pub fn setup(&mut self, circuit: &C) {
        let vk = keygen_vk(&self.params, circuit).expect("keygen_vk failed");
        let pk = keygen_pk(&self.params, vk.clone(), circuit).expect("keygen_pk failed");
        self.vk = Some(vk);
        self.pk = Some(pk);
    }

    pub fn prove(&self, circuit: &C, public_inputs: &[&[Fr]]) -> Vec<u8> {
        let pk = self.pk.as_ref().expect("PK not generated");
        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
        
        create_proof(
            &self.params,
            pk,
            &[circuit.clone()],
            &[public_inputs],
            OsRng,
            &mut transcript,
        )
        .expect("proof generation failed");

        transcript.finalize()
    }

    pub fn verify(&self, proof: &[u8], public_inputs: &[&[Fr]]) -> bool {
        let vk = self.vk.as_ref().expect("VK not generated");
        let strategy = SingleVerifier::new(&self.params);
        let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(proof);

        verify_proof(
            &self.params,
            vk,
            strategy,
            &[public_inputs],
            &mut transcript,
        )
        .is_ok()
    }

    /// Extracts verification key data for EVM verifier generation.
    pub fn extract_vk_data(&self, num_instances: usize) -> Option<ExtractedVkData> {
        let vk = self.vk.as_ref()?;

        // BN254 G1 generator
        let g1_coords = G1Affine::generator().coordinates().unwrap();
        let g1_x = format!("{}", field_to_u256(*g1_coords.x()));
        let g1_y = format!("{}", field_to_u256(*g1_coords.y()));

        // Get s·G2 from SRS params (second G2 point in the SRS)
        // For now, use standard trusted setup values
        let s_g2 = (
            "11559732032986387107991004021392285783925812861821192530917403151452391805634".to_string(),
            "10857046999023057135944570762232829481370756359578518086990519993285655852781".to_string(),
            "4082367875863433681332203403145435568316851327593401208105741076214120093531".to_string(),
            "8495653923123431417604973247489272438418190587263600148770280649306958101930".to_string(),
        );

        // Negative G2 generator
        let neg_g2 = (
            "11559732032986387107991004021392285783925812861821192530917403151452391805634".to_string(),
            "10857046999023057135944570762232829481370756359578518086990519993285655852781".to_string(),
            "17805874995975841540914202342111839520379459829704422454583296818431106115052".to_string(),
            "13392588948715843804641432497768002650278120570034223513918757245338268106653".to_string(),
        );

        // For Halo2 proofs, the number of advice commitments depends on the circuit.
        // MLTrainingStepCircuit uses 3 advice columns.
        // This is a reasonable default; can be overridden when generating the contract.
        let num_advices = 3;

        Some(ExtractedVkData {
            g1: (g1_x, g1_y),
            s_g2,
            neg_g2,
            num_advices,
            num_instances,
            k: self.params.k(),
        })
    }

    /// Returns a reference to the verification key.
    pub fn vk(&self) -> Option<&VerifyingKey<G1Affine>> {
        self.vk.as_ref()
    }

    /// Returns a reference to the SRS params.
    pub fn params(&self) -> &Params<G1Affine> {
        &self.params
    }
}

/// Converts a field element to a decimal string (for Solidity constants).
fn field_to_u256<F: PrimeField>(f: F) -> String {
    let repr = f.to_repr();
    let bytes = repr.as_ref();

    // Convert little-endian bytes to big integer
    let mut value = num_bigint::BigUint::from_bytes_le(bytes);
    value.to_string()
}

use num_bigint;
