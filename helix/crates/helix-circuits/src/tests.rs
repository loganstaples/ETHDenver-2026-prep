use crate::gadgets::range::{RangeChip, RangeConfig};
use crate::gadgets::arithmetic::{ArithmeticChip, ArithmeticConfig};
use crate::approximate::bounded_add::{BoundedAddChip, BoundedAddConfig};
use crate::approximate::bounded_mul::{BoundedMulChip, BoundedMulConfig};
use crate::approximate::bounded_matmul::{BoundedMatMulChip, BoundedMatMulConfig};
use crate::approximate::activation::{ReLUChip, ReLUConfig};

use halo2curves::bn256::Fr;
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    dev::MockProver,
    plonk::{Circuit, ConstraintSystem, Error},
};

#[derive(Clone)]
struct TestCircuitConfig {
    add: BoundedAddConfig<Fr, 100>, 
    mul: BoundedMulConfig<Fr, 100>,
    matmul: BoundedMatMulConfig<Fr, 100>,
    relu: ReLUConfig<Fr, 100>,
}

#[derive(Default)]
struct TestCircuit {
    // Inputs (Option allows testing missing values behavior if needed, generally Some)
    a_val: Option<u64>,
    a_err: Option<u64>,
    b_val: Option<u64>,
    b_err: Option<u64>,
    
    // Expected Output override (if None, computes correct one)
    c_val: Option<u64>,
    c_err: Option<u64>,

    // Mode switch (0=add, 1=mul, 2=matmul, 3=relu)
    mode: usize,
}

impl Circuit<Fr> for TestCircuit {
    type Config = TestCircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        let a = meta.advice_column();
        let b = meta.advice_column();
        let c = meta.advice_column();
        let table = meta.lookup_table_column();
        
        let arith_config = ArithmeticChip::configure(meta, a, b, c);
        // Range check is on column 'c', which is the output of arithmetic usually.
        // But gadgets might copy values there.
        let range_config = RangeConfig::configure(meta, table, c);
        let s_relu = meta.selector();

        TestCircuitConfig {
            add: BoundedAddConfig {
                arithmetic: arith_config.clone(),
                range: range_config.clone(),
            },
            mul: BoundedMulConfig {
                arithmetic: arith_config.clone(),
                range: range_config.clone(),
            },
            matmul: BoundedMatMulConfig {
                arithmetic: arith_config.clone(),
                range: range_config.clone(),
            },
            relu: ReLUConfig {
                range: range_config,
                arithmetic: arith_config,
                s_relu,
            }
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        let add_chip = BoundedAddChip::new(config.add.clone());
        let mul_chip = BoundedMulChip::new(config.mul.clone());
        let matmul_chip = BoundedMatMulChip::new(config.matmul.clone());
        let relu_chip = ReLUChip::new(config.relu.clone());
        
        // Load range table once
        add_chip.range_chip.load(&mut layouter)?;

        let val_a = Value::known(Fr::from(self.a_val.unwrap_or(0)));
        let err_a = Value::known(Fr::from(self.a_err.unwrap_or(0)));
        let val_b = Value::known(Fr::from(self.b_val.unwrap_or(0)));
        let err_b = Value::known(Fr::from(self.b_err.unwrap_or(0)));

        if self.mode == 0 {
            // Add
            // Use expected values if provided, else compute correct ones
            let val_c = self.c_val.map(|v| Value::known(Fr::from(v)))
                .unwrap_or_else(|| val_a + val_b);
            let err_c = self.c_err.map(|v| Value::known(Fr::from(v)))
                .unwrap_or_else(|| err_a + err_b);

            add_chip.assign(layouter, val_a, err_a, val_b, err_b, val_c, err_c)?;
        } else if self.mode == 1 {
            // Mul
            let val_c = self.c_val.map(|v| Value::known(Fr::from(v)))
                .unwrap_or_else(|| val_a * val_b);
                
            let err_c = self.c_err.map(|v| Value::known(Fr::from(v)))
                .unwrap_or_else(|| {
                    let term1 = val_a * err_b;
                    let term2 = val_b * err_a;
                    let term3 = err_a * err_b;
                    term1 + term2 + term3
                });

            mul_chip.assign(layouter, val_a, err_a, val_b, err_b, val_c, err_c)?;
        } else if self.mode == 2 {
            // MatMul (1x1 for simplicity)
            let row_a_v = vec![val_a];
            let row_a_e = vec![err_a];
            let col_b_v = vec![val_b];
            let col_b_e = vec![err_b];
            
            let res_val = self.c_val.map(|v| Value::known(Fr::from(v)))
                .unwrap_or_else(|| val_a * val_b);

            let err_c = self.c_err.map(|v| Value::known(Fr::from(v)))
                .unwrap_or_else(|| {
                    let term1 = val_a * err_b;
                    let term2 = val_b * err_a;
                    let term3 = err_a * err_b;
                    term1 + term2 + term3
                });
            
            matmul_chip.assign_dot_product(layouter, &row_a_v, &row_a_e, &col_b_v, &col_b_e, res_val, err_c)?;
        } else if self.mode == 3 {
            // ReLU
            // If checking failure, inputs might be manipulated.
            // For valid ReLU: if a > 0, output=a. If a < 0 (large field element), output=0.
            let out_val = self.c_val.map(|v| Value::known(Fr::from(v)))
                .unwrap_or(val_a); // Default to identity (positive case)
            
            let out_err = self.c_err.map(|v| Value::known(Fr::from(v)))
                .unwrap_or(err_a);

            relu_chip.assign(layouter, val_a, err_a, out_val, out_err)?;
        }

        Ok(())
    }
}

// --- Positive Tests ---

#[test]
fn test_bounded_add_valid() {
    let circuit = TestCircuit { 
        a_val: Some(10), a_err: Some(1),
        b_val: Some(20), b_err: Some(2),
        c_val: None, c_err: None,
        mode: 0 
    };
    let prover = MockProver::run(8, &circuit, vec![]).unwrap();
    assert_eq!(prover.verify(), Ok(()));
}

#[test]
fn test_bounded_mul_valid() {
    let circuit = TestCircuit { 
        a_val: Some(10), a_err: Some(1),
        b_val: Some(20), b_err: Some(2),
        c_val: None, c_err: None,
        mode: 1 
    };
    // Expected err: 10*2 + 20*1 + 1*2 = 20 + 20 + 2 = 42
    // 42 < 100 (range)
    let prover = MockProver::run(8, &circuit, vec![]).unwrap();
    assert_eq!(prover.verify(), Ok(()));
}

#[test]
fn test_bounded_matmul_valid() {
    let circuit = TestCircuit { 
        a_val: Some(2), a_err: Some(1),
        b_val: Some(3), b_err: Some(1),
        c_val: None, c_err: None,
        mode: 2 
    };
    // err = 2*1 + 3*1 + 1*1 = 6.
    let prover = MockProver::run(8, &circuit, vec![]).unwrap();
    assert_eq!(prover.verify(), Ok(()));
}

#[test]
fn test_relu_valid() {
    let circuit = TestCircuit {
        a_val: Some(10), a_err: Some(5),
        b_val: None, b_err: None,
        c_val: Some(10), c_err: Some(5), // Identity for positive
        mode: 3
    };
    let prover = MockProver::run(8, &circuit, vec![]).unwrap();
    assert_eq!(prover.verify(), Ok(()));
}

// --- Negative Tests (Must Fail) ---

#[test]
fn test_bounded_add_invalid_value() {
    let circuit = TestCircuit { 
        a_val: Some(10), a_err: Some(1),
        b_val: Some(20), b_err: Some(2),
        // 10 + 20 != 31
        c_val: Some(31), c_err: None, 
        mode: 0 
    };
    let prover = MockProver::run(8, &circuit, vec![]).unwrap();
    // verification should fail implies Ok(()) is NOT equal to prover.verify()
    assert!(prover.verify().is_err());
}

#[test]
fn test_bounded_add_invalid_error() {
    let circuit = TestCircuit { 
        a_val: Some(10), a_err: Some(1),
        b_val: Some(20), b_err: Some(2),
        // 1 + 2 != 5
        c_val: None, c_err: Some(5),
        mode: 0 
    };
    let prover = MockProver::run(8, &circuit, vec![]).unwrap();
    assert!(prover.verify().is_err());
}

#[test]
fn test_bounded_add_error_out_of_range() {
    let circuit = TestCircuit { 
        a_val: Some(10), a_err: Some(60),
        b_val: Some(20), b_err: Some(50), 
        // 60 + 50 = 110. Range is 100. Should fail range check.
        c_val: None, c_err: None,
        mode: 0 
    };
    let prover = MockProver::run(8, &circuit, vec![]).unwrap();
    assert!(prover.verify().is_err());
}

#[test]
fn test_bounded_mul_invalid_value() {
    let circuit = TestCircuit { 
        a_val: Some(10), a_err: Some(1),
        b_val: Some(20), b_err: Some(2),
        c_val: Some(201), c_err: None, // 10*20 != 201
        mode: 1 
    };
    let prover = MockProver::run(8, &circuit, vec![]).unwrap();
    assert!(prover.verify().is_err());
}

#[test]
fn test_bounded_mul_invalid_error() {
    let circuit = TestCircuit { 
        a_val: Some(10), a_err: Some(1),
        b_val: Some(20), b_err: Some(2),
        c_val: None, c_err: Some(10), // Expected 42
        mode: 1 
    };
    let prover = MockProver::run(8, &circuit, vec![]).unwrap();
    assert!(prover.verify().is_err());
}

#[test]
fn test_state_transition() {
    // Placeholder to confirm compilation of new modules
    assert_eq!(1, 1);
}

