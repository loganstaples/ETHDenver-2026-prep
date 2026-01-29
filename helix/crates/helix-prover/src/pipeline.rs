use helix_circuits::halo2_proofs::{
    arithmetic::{CurveAffine, Field},
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, Circuit, ProvingKey, VerifyingKey, SingleVerifier},
    poly::commitment::Params,
    transcript::{Blake2bRead, Blake2bWrite, Challenge255},
};
use helix_circuits::halo2curves::{
    bn256::{Fr, G1Affine},
};
use rand::rngs::OsRng;
use std::marker::PhantomData;

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
}
