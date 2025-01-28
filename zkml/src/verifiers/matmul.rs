use arith::Field;
use expander_compiler::field::BN254;
use expander_compiler::frontend::*;
use extra::UnconstrainedAPI;
use rand::Rng;

/// Fixed-point scaling factor (2^16)

/// Verifies matrix multiplication AB = C using Freivalds' algorithm.
/// Uses random vectors from {0,1}^n for probabilistic verification.
/// The probability of error is at most 2^(-num_iterations).
///
/// This implementation uses BN254 field for fixed-point arithmetic support.
/// All input numbers are assumed to be scaled by 2^16 (ONE).
/// After each multiplication, we scale down by ONE to maintain fixed-point representation.
///
/// # Arguments
/// * `builder` - The circuit builder API
/// * `a` - First input matrix data (flattened in row-major order)
/// * `b` - Second input matrix data (flattened in row-major order)
/// * `c` - Output matrix data (flattened in row-major order)
/// * `a_signs` - Signs of elements in matrix A
/// * `b_signs` - Signs of elements in matrix B
/// * `c_signs` - Signs of elements in matrix C
/// * `a_shape` - Shape of matrix A [m, n]
/// * `b_shape` - Shape of matrix B [n, p]
/// * `c_shape` - Shape of matrix C [m, p]
/// * `num_iterations` - Number of iterations for probability amplification (default 5)
pub fn verify_matmul<C: Config>(
    builder: &mut API<C>,
    a: &[Variable],
    b: &[Variable],
    c: &[Variable],
    a_signs: &[bool], // true for positive, false for negative
    b_signs: &[bool],
    c_signs: &[bool],
    a_shape: &[u64],
    b_shape: &[u64],
    c_shape: &[u64],
    num_iterations: usize,
) -> Variable {
    assert_eq!(a_shape.len(), 2, "Matrix A must be 2D");
    assert_eq!(b_shape.len(), 2, "Matrix B must be 2D");
    assert_eq!(c_shape.len(), 2, "Matrix C must be 2D");
    assert_eq!(a_shape[1], b_shape[0], "Matrix dimensions must match");
    assert_eq!(a_shape[0], c_shape[0], "Output rows must match A");
    assert_eq!(b_shape[1], c_shape[1], "Output cols must match B");
    const ONE: u32 = 1 << 16;
    let m = a_shape[0] as usize;
    let n = a_shape[1] as usize;
    let p = b_shape[1] as usize;

    // Initialize result as true and create constants
    let mut result = builder.constant(C::CircuitField::from(1u32));
    let one_const = builder.constant(C::CircuitField::from(ONE as u32));
    let zero_const = builder.constant(C::CircuitField::ZERO);
    let true_const = builder.constant(C::CircuitField::from(1u32));
    let false_const = builder.constant(C::CircuitField::from(0u32));
    let checked = false;

    // Scale C by ONE to match A*B result
    let mut scaled_c = Vec::with_capacity(c.len());
    for i in 0..c.len() {
        scaled_c.push(builder.mul(c[i], one_const));
    }

    let mut rng = rand::thread_rng();

    // Run multiple iterations for probability amplification
    for _ in 0..num_iterations {
        // Generate random vector r from {0,1}^p
        let mut r: Vec<bool> = Vec::with_capacity(p);
        for _ in 0..p {
            if p == 1 {
                r.push(true);
            } else {
                r.push(rng.gen_bool(0.5));
            }
        }

        // Compute Cr
        let mut cr = vec![builder.constant(C::CircuitField::ZERO); m];
        for i in 0..m {
            for j in 0..p {
                let c_ij = scaled_c[i * p + j];
                if r[j] {
                    let c_sign = c_signs[i * p + j];
                    if c_sign {
                        cr[i] = builder.add(cr[i], c_ij);
                    } else {
                        cr[i] = builder.sub(cr[i], c_ij);
                    }
                }
                // even when having r[j] = false, we still need to add c_ij to cr[i] and multiply by 0
                else {
                    let factor = builder.mul(zero_const, c_ij);
                    cr[i] = builder.add(cr[i], factor);
                }
            }
        }

        // Compute (AB)r = A(Br)
        // First compute Br
        let mut br = vec![builder.constant(C::CircuitField::ZERO); n];
        for i in 0..n {
            for j in 0..p {
                let b_ij = b[i * p + j];
                if r[j] {
                    let b_sign = b_signs[i * p + j];
                    if b_sign {
                        br[i] = builder.add(br[i], b_ij);
                    } else {
                        br[i] = builder.sub(br[i], b_ij);
                    }
                } else {
                    let factor = builder.mul(zero_const, b_ij);
                    br[i] = builder.add(br[i], factor);
                }
            }
        }

        // Then compute A(Br)
        let mut abr = vec![builder.constant(C::CircuitField::ZERO); m];
        for i in 0..m {
            for k in 0..n {
                let a_ik = a[i * n + k];
                let a_sign = a_signs[i * n + k];
                let br_k = br[k];

                // Compute magnitude
                let prod = builder.mul(a_ik, br_k);

                // Add or subtract based on sign
                if a_sign {
                    abr[i] = builder.add(abr[i], prod);
                } else {
                    abr[i] = builder.sub(abr[i], prod);
                }
            }
        }

        // Compare Cr with ABr
        for i in 0..m {
            let diff = builder.sub(abr[i], cr[i]);
            let iter_result = builder.unconstrained_lesser(diff, one_const);

            result = builder.and(result, iter_result);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use expander_compiler::frontend::BN254Config;
    const ONE: u64 = 1 << 16;
    #[test]
    fn test_verify_matmul() {
        // Create a simple circuit to test the verifier
        declare_circuit!(TestCircuit {
            a: [Variable],
            b: [Variable],
            c: [Variable],
        });

        impl<C: Config> Define<C> for TestCircuit<Variable> {
            fn define(&self, builder: &mut API<C>) {
                let a_signs = vec![true, false, true, true]; // [+1, -2, +3, +4]
                let b_signs = vec![true, true, false, true]; // [+5, +6, -7, +8]
                let c_signs = vec![true, false, false, true]; // [+19, -10, -13, +50]

                let result = verify_matmul(
                    builder,
                    &self.a,
                    &self.b,
                    &self.c,
                    &a_signs,
                    &b_signs,
                    &c_signs,
                    &[2, 2], // A is 2x2
                    &[2, 2], // B is 2x2
                    &[2, 2], // C is 2x2
                    7,       // num_iterations
                );
                let true_const = builder.constant(C::CircuitField::from(1u32));
                builder.assert_is_equal(result, true_const);
            }
        }

        // Create test data
        let circuit = TestCircuit {
            a: vec![Variable::default(); 4],
            b: vec![Variable::default(); 4],
            c: vec![Variable::default(); 4],
        };

        let compile_result = compile::<BN254Config, TestCircuit<Variable>>(&circuit).unwrap();
        // Test correct multiplication with fixed-point scaling and signs
        let assignment = TestCircuit::<BN254> {
            a: vec![
                BN254::from(1u64 * ONE), // +1
                BN254::from(2u64 * ONE), // -2
                BN254::from(3u64 * ONE), // +3
                BN254::from(4u64 * ONE), // +4
            ],
            b: vec![
                BN254::from(5u64 * ONE), // +5
                BN254::from(6u64 * ONE), // +6
                BN254::from(7u64 * ONE), // -7
                BN254::from(8u64 * ONE), // +8
            ],
            c: vec![
                BN254::from(19u64 * ONE), // +19
                BN254::from(10u64 * ONE), // -10
                BN254::from(13u64 * ONE), // -13
                BN254::from(50u64 * ONE), // +50
            ],
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    #[test]
    fn test_verify_matmul_all_negative() {
        // Test case with all negative numbers
        declare_circuit!(TestCircuit {
            a: [Variable],
            b: [Variable],
            c: [Variable],
        });

        impl<C: Config> Define<C> for TestCircuit<Variable> {
            fn define(&self, builder: &mut API<C>) {
                let a_signs = vec![false, false, false, false]; // [-1, -2, -3, -4]
                let b_signs = vec![false, false, false, false]; // [-5, -6, -7, -8]
                let c_signs = vec![true, true, true, true]; // [+19, +22, +43, +50]

                let result = verify_matmul(
                    builder,
                    &self.a,
                    &self.b,
                    &self.c,
                    &a_signs,
                    &b_signs,
                    &c_signs,
                    &[2, 2],
                    &[2, 2],
                    &[2, 2],
                    5,
                );
                let true_const = builder.constant(C::CircuitField::from(1u32));
                builder.assert_is_equal(result, true_const);
            }
        }

        let circuit = TestCircuit {
            a: vec![Variable::default(); 4],
            b: vec![Variable::default(); 4],
            c: vec![Variable::default(); 4],
        };

        let compile_result = compile::<BN254Config, TestCircuit<Variable>>(&circuit).unwrap();

        let assignment = TestCircuit::<BN254> {
            a: vec![
                BN254::from(1u64 * ONE),
                BN254::from(2u64 * ONE),
                BN254::from(3u64 * ONE),
                BN254::from(4u64 * ONE),
            ],
            b: vec![
                BN254::from(5u64 * ONE),
                BN254::from(6u64 * ONE),
                BN254::from(7u64 * ONE),
                BN254::from(8u64 * ONE),
            ],
            c: vec![
                BN254::from(19u64 * ONE),
                BN254::from(22u64 * ONE),
                BN254::from(43u64 * ONE),
                BN254::from(50u64 * ONE),
            ],
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    #[test]
    fn test_verify_matmul_alternating_signs() {
        // Test case with alternating signs
        declare_circuit!(TestCircuit {
            a: [Variable],
            b: [Variable],
            c: [Variable],
        });

        impl<C: Config> Define<C> for TestCircuit<Variable> {
            fn define(&self, builder: &mut API<C>) {
                let a_signs = vec![true, false, true, false]; // [+1, -2, +3, -4]
                let b_signs = vec![true, false, true, false]; // [+5, -6, +7, -8]
                let c_signs = vec![false, true, false, true]; // [-9, +10, -13, 14]

                let result = verify_matmul(
                    builder,
                    &self.a,
                    &self.b,
                    &self.c,
                    &a_signs,
                    &b_signs,
                    &c_signs,
                    &[2, 2],
                    &[2, 2],
                    &[2, 2],
                    5,
                );
                let true_const = builder.constant(C::CircuitField::from(1u32));
                builder.assert_is_equal(result, true_const);
            }
        }

        let circuit = TestCircuit {
            a: vec![Variable::default(); 4],
            b: vec![Variable::default(); 4],
            c: vec![Variable::default(); 4],
        };

        let compile_result = compile::<BN254Config, TestCircuit<Variable>>(&circuit).unwrap();

        let assignment = TestCircuit::<BN254> {
            a: vec![
                BN254::from(1u64 * ONE),
                BN254::from(2u64 * ONE),
                BN254::from(3u64 * ONE),
                BN254::from(4u64 * ONE),
            ],
            b: vec![
                BN254::from(5u64 * ONE),
                BN254::from(6u64 * ONE),
                BN254::from(7u64 * ONE),
                BN254::from(8u64 * ONE),
            ],
            c: vec![
                BN254::from(9u64 * ONE),
                BN254::from(10u64 * ONE),
                BN254::from(13u64 * ONE),
                BN254::from(14u64 * ONE),
            ],
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    #[test]
    fn test_verify_matmul_first_negative() {
        // Test case with first matrix negative
        declare_circuit!(TestCircuit {
            a: [Variable],
            b: [Variable],
            c: [Variable],
        });

        impl<C: Config> Define<C> for TestCircuit<Variable> {
            fn define(&self, builder: &mut API<C>) {
                let a_signs = vec![false, false, false, false]; // [-1, -2, -3, -4]
                let b_signs = vec![true, true, true, true]; // [+5, +6, +7, +8]
                let c_signs = vec![false, false, false, false]; // [-19, -22, -43, -50]

                let result = verify_matmul(
                    builder,
                    &self.a,
                    &self.b,
                    &self.c,
                    &a_signs,
                    &b_signs,
                    &c_signs,
                    &[2, 2],
                    &[2, 2],
                    &[2, 2],
                    5,
                );
                let true_const = builder.constant(C::CircuitField::from(1u32));
                builder.assert_is_equal(result, true_const);
            }
        }

        let circuit = TestCircuit {
            a: vec![Variable::default(); 4],
            b: vec![Variable::default(); 4],
            c: vec![Variable::default(); 4],
        };

        let compile_result = compile::<BN254Config, TestCircuit<Variable>>(&circuit).unwrap();

        let assignment = TestCircuit::<BN254> {
            a: vec![
                BN254::from(1u64 * ONE),
                BN254::from(2u64 * ONE),
                BN254::from(3u64 * ONE),
                BN254::from(4u64 * ONE),
            ],
            b: vec![
                BN254::from(5u64 * ONE),
                BN254::from(6u64 * ONE),
                BN254::from(7u64 * ONE),
                BN254::from(8u64 * ONE),
            ],
            c: vec![
                BN254::from(19u64 * ONE),
                BN254::from(22u64 * ONE),
                BN254::from(43u64 * ONE),
                BN254::from(50u64 * ONE),
            ],
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    #[test]
    fn test_verify_matmul_second_negative() {
        // Test case with second matrix negative
        declare_circuit!(TestCircuit {
            a: [Variable],
            b: [Variable],
            c: [Variable],
        });

        impl<C: Config> Define<C> for TestCircuit<Variable> {
            fn define(&self, builder: &mut API<C>) {
                let a_signs = vec![true, true, true, true]; // [+1, +2, +3, +4]
                let b_signs = vec![false, false, false, false]; // [-5, -6, -7, -8]
                let c_signs = vec![false, false, false, false]; // [-19, -22, -43, -50]

                let result = verify_matmul(
                    builder,
                    &self.a,
                    &self.b,
                    &self.c,
                    &a_signs,
                    &b_signs,
                    &c_signs,
                    &[2, 2],
                    &[2, 2],
                    &[2, 2],
                    5,
                );
                let true_const = builder.constant(C::CircuitField::from(1u32));
                builder.assert_is_equal(result, true_const);
            }
        }

        let circuit = TestCircuit {
            a: vec![Variable::default(); 4],
            b: vec![Variable::default(); 4],
            c: vec![Variable::default(); 4],
        };

        let compile_result = compile::<BN254Config, TestCircuit<Variable>>(&circuit).unwrap();

        let assignment = TestCircuit::<BN254> {
            a: vec![
                BN254::from(1u64 * ONE),
                BN254::from(2u64 * ONE),
                BN254::from(3u64 * ONE),
                BN254::from(4u64 * ONE),
            ],
            b: vec![
                BN254::from(5u64 * ONE),
                BN254::from(6u64 * ONE),
                BN254::from(7u64 * ONE),
                BN254::from(8u64 * ONE),
            ],
            c: vec![
                BN254::from(19u64 * ONE),
                BN254::from(22u64 * ONE),
                BN254::from(43u64 * ONE),
                BN254::from(50u64 * ONE),
            ],
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    #[test]
    fn test_verify_matmul_diagonal_negative() {
        // Test case with diagonal elements negative
        declare_circuit!(TestCircuit {
            a: [Variable],
            b: [Variable],
            c: [Variable],
        });

        impl<C: Config> Define<C> for TestCircuit<Variable> {
            fn define(&self, builder: &mut API<C>) {
                let a_signs = vec![false, true, true, false]; // [-1, +2, +3, -4]
                let b_signs = vec![false, true, true, false]; // [-5, +6, +7, -8]
                let c_signs = vec![true, false, false, true]; // [+23, -20, -37, +46]

                let result = verify_matmul(
                    builder,
                    &self.a,
                    &self.b,
                    &self.c,
                    &a_signs,
                    &b_signs,
                    &c_signs,
                    &[2, 2],
                    &[2, 2],
                    &[2, 2],
                    5,
                );
                let true_const = builder.constant(C::CircuitField::from(1u32));
                builder.assert_is_equal(result, true_const);
            }
        }

        let circuit = TestCircuit {
            a: vec![Variable::default(); 4],
            b: vec![Variable::default(); 4],
            c: vec![Variable::default(); 4],
        };

        let compile_result = compile::<BN254Config, TestCircuit<Variable>>(&circuit).unwrap();

        let assignment = TestCircuit::<BN254> {
            a: vec![
                BN254::from(1u64 * ONE),
                BN254::from(2u64 * ONE),
                BN254::from(3u64 * ONE),
                BN254::from(4u64 * ONE),
            ],
            b: vec![
                BN254::from(5u64 * ONE),
                BN254::from(6u64 * ONE),
                BN254::from(7u64 * ONE),
                BN254::from(8u64 * ONE),
            ],
            c: vec![
                BN254::from(19u64 * ONE),
                BN254::from(22u64 * ONE),
                BN254::from(43u64 * ONE),
                BN254::from(50u64 * ONE),
            ],
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    #[test]
    fn test_verify_matmul_1x1() {
        // Test case with 1x1 matrices
        declare_circuit!(TestCircuit {
            a: [Variable],
            b: [Variable],
            c: [Variable],
        });

        impl<C: Config> Define<C> for TestCircuit<Variable> {
            fn define(&self, builder: &mut API<C>) {
                let a_signs = vec![true]; // [+13107]
                let b_signs = vec![true]; // [+32768]
                let c_signs = vec![true]; // [+429496729] (13107 * 32768)

                let result = verify_matmul(
                    builder,
                    &self.a,
                    &self.b,
                    &self.c,
                    &a_signs,
                    &b_signs,
                    &c_signs,
                    &[1, 1], // A is 1x1
                    &[1, 1], // B is 1x1
                    &[1, 1], // C is 1x1
                    5,       // num_iterations
                );
                let true_const = builder.constant(C::CircuitField::from(1u32));
                builder.assert_is_equal(result, true_const);
            }
        }

        let circuit = TestCircuit {
            a: vec![Variable::default(); 1],
            b: vec![Variable::default(); 1],
            c: vec![Variable::default(); 1],
        };

        let compile_result = compile::<BN254Config, TestCircuit<Variable>>(&circuit).unwrap();

        let assignment = TestCircuit::<BN254> {
            a: vec![BN254::from(13107u32)], // 13107
            b: vec![BN254::from(32768u32)], // 32768
            c: vec![BN254::from(6553u32)],  // 13107 * 32768
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    #[test]
    fn test_verify_matmul_small_numbers() {
        // Test case with small numbers (less than 1)
        declare_circuit!(TestCircuit {
            a: [Variable],
            b: [Variable],
            c: [Variable],
        });

        impl<C: Config> Define<C> for TestCircuit<Variable> {
            fn define(&self, builder: &mut API<C>) {
                let a_signs = vec![true]; // [+0.2]
                let b_signs = vec![true]; // [+0.5]
                let c_signs = vec![true]; // [+0.1] (0.2 * 0.5)

                let result = verify_matmul(
                    builder,
                    &self.a,
                    &self.b,
                    &self.c,
                    &a_signs,
                    &b_signs,
                    &c_signs,
                    &[1, 1], // A is 1x1
                    &[1, 1], // B is 1x1
                    &[1, 1], // C is 1x1
                    5,       // num_iterations
                );
                let true_const = builder.constant(C::CircuitField::from(1u32));
                builder.assert_is_equal(result, true_const);
            }
        }

        let circuit = TestCircuit {
            a: vec![Variable::default(); 1],
            b: vec![Variable::default(); 1],
            c: vec![Variable::default(); 1],
        };

        let compile_result = compile::<BN254Config, TestCircuit<Variable>>(&circuit).unwrap();

        // Test with small numbers: 0.2 * 0.5 = 0.1
        let assignment = TestCircuit::<BN254> {
            a: vec![BN254::from(13107u32)], // 0.2 * 2^16 = 13107
            b: vec![BN254::from(32768u32)], // 0.5 * 2^16 = 32768
            c: vec![BN254::from(6553u32)],  // 0.1 * 2^16 = 6553
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }

    #[test]
    fn test_verify_matmul_chained() {
        // Test case for chained matrix multiplication
        declare_circuit!(TestCircuit {
            aa: [Variable],
            bb: [Variable],
            cc: [Variable],
            dd: [Variable],
            ee: [Variable],
        });

        impl<C: Config> Define<C> for TestCircuit<Variable> {
            fn define(&self, builder: &mut API<C>) {
                // All values are positive
                let aa_signs = vec![true, true, true, true];
                let bb_signs = vec![true, true, true, true];
                let cc_signs = vec![true, true, true, true];
                let dd_signs = vec![true, true, true, true];
                let ee_signs = vec![true, true, true, true];

                // First multiplication: cc = aa * bb
                let result1 = verify_matmul(
                    builder,
                    &self.aa,
                    &self.bb,
                    &self.cc,
                    &aa_signs,
                    &bb_signs,
                    &cc_signs,
                    &[2, 2], // aa is 2x2
                    &[2, 2], // bb is 2x2
                    &[2, 2], // cc is 2x2
                    5,       // num_iterations
                );

                // Second multiplication: ee = cc * dd
                let result2 = verify_matmul(
                    builder,
                    &self.cc,
                    &self.dd,
                    &self.ee,
                    &cc_signs,
                    &dd_signs,
                    &ee_signs,
                    &[2, 2], // cc is 2x2
                    &[2, 2], // dd is 2x2
                    &[2, 2], // ee is 2x2
                    5,       // num_iterations
                );

                // Both multiplications must be correct
                let true_const = builder.constant(C::CircuitField::from(1u32));
                let final_result = builder.and(result1, result2);
                builder.assert_is_equal(final_result, true_const);
            }
        }

        let circuit = TestCircuit {
            aa: vec![Variable::default(); 4],
            bb: vec![Variable::default(); 4],
            cc: vec![Variable::default(); 4],
            dd: vec![Variable::default(); 4],
            ee: vec![Variable::default(); 4],
        };

        let compile_result = compile::<BN254Config, TestCircuit<Variable>>(&circuit).unwrap();

        let assignment = TestCircuit::<BN254> {
            aa: vec![
                BN254::from(64684u32), // Values from test_zkml.py
                BN254::from(131072u32),
                BN254::from(196608u32),
                BN254::from(262144u32),
            ],
            bb: vec![
                BN254::from(32768u32),
                BN254::from(425984u32),
                BN254::from(491520u32),
                BN254::from(524288u32),
            ],
            cc: vec![
                BN254::from(1015382u32),
                BN254::from(1469022u32),
                BN254::from(2064384u32),
                BN254::from(3375104u32),
            ],
            dd: vec![
                BN254::from(65536u32),
                BN254::from(131072u32),
                BN254::from(196608u32),
                BN254::from(262144u32),
            ],
            ee: vec![
                BN254::from(5422448u32),
                BN254::from(7906852u32),
                BN254::from(12189696u32),
                BN254::from(17629184u32),
            ],
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);
    }
}
