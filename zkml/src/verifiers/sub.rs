use arith::Field;
use expander_compiler::frontend::*;

/// Verifies that the element-wise subtraction of two tensors equals the output tensor.
///
/// # Arguments
/// * `builder` - The circuit builder API
/// * `input1` - First input tensor data (flattened)
/// * `input2` - Second input tensor data (flattened)
/// * `output` - Output tensor data (flattened)
/// * `a_signs` - Signs of elements in first tensor (true for positive, false for negative)
/// * `b_signs` - Signs of elements in second tensor
/// * `c_signs` - Signs of elements in output tensor
/// * `shape` - Shape of all tensors (must be same for all)
///
/// # Returns
/// * `bool` - Whether the verification constraints were satisfied
pub fn verify_tensor_sub<C: Config>(
    builder: &mut API<C>,
    a: &[Variable],
    b: &[Variable],
    c: &[Variable],
    a_signs: &[bool],
    b_signs: &[bool],
    c_signs: &[bool],
    shape: &[u64],
) -> Variable {
    let size: usize = shape.iter().product::<u64>() as usize;
    assert_eq!(a.len(), size, "Tensor A size must match shape");
    assert_eq!(b.len(), size, "Tensor B size must match shape");
    assert_eq!(c.len(), size, "Tensor C size must match shape");
    assert_eq!(a_signs.len(), size, "A signs length must match shape");
    assert_eq!(b_signs.len(), size, "B signs length must match shape");
    assert_eq!(c_signs.len(), size, "C signs length must match shape");

    // Initialize result as true
    let mut result = builder.constant(C::CircuitField::from(1));

    // For each element
    for i in 0..size {
        let a_val = if a_signs[i] { a[i] } else { builder.neg(a[i]) };
        let b_val = if b_signs[i] { b[i] } else { builder.neg(b[i]) };
        let diff = builder.sub(a_val, b_val);
        let c_val = if c_signs[i] { c[i] } else { builder.neg(c[i]) };
        let diff2 = builder.sub(c_val, diff);
        let zero = builder.constant(C::CircuitField::ZERO);
        let eq = builder.is_zero(diff2);
        result = builder.and(result, eq);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use expander_compiler::field::BN254;
    use expander_compiler::frontend::BN254Config;

    const ONE: u32 = 1 << 16;

    #[test]
    fn test_verify_tensor_sub() {
        // Create a simple circuit to test the verifier
        declare_circuit!(TestCircuit {
            input1: [Variable],
            input2: [Variable],
            output: [Variable],
        });

        impl<C: Config> Define<C> for TestCircuit<Variable> {
            fn define(&self, builder: &mut API<C>) {
                let a_signs = vec![true, false, true, false]; // [+1, -2, +3, -4]
                let b_signs = vec![true, true, false, false]; // [+5, +6, -7, -8]
                let c_signs = vec![false, false, true, true]; // [-4, -8, +10, +4]

                let result = verify_tensor_sub(
                    builder,
                    &self.input1,
                    &self.input2,
                    &self.output,
                    &a_signs,
                    &b_signs,
                    &c_signs,
                    &[2, 2],
                );
                let true_const = builder.constant(C::CircuitField::from(1u32));
                builder.assert_is_equal(result, true_const);
            }
        }

        // Create test data
        let circuit = TestCircuit {
            input1: vec![Variable::default(); 4],
            input2: vec![Variable::default(); 4],
            output: vec![Variable::default(); 4],
        };

        let compile_result = compile::<BN254Config, TestCircuit<Variable>>(&circuit).unwrap();

        // Test correct subtraction with signs
        let assignment = TestCircuit::<BN254> {
            input1: vec![
                BN254::from(1u32 * ONE), // +1
                BN254::from(2u32 * ONE), // -2
                BN254::from(3u32 * ONE), // +3
                BN254::from(4u32 * ONE), // -4
            ],
            input2: vec![
                BN254::from(5u32 * ONE), // +5
                BN254::from(6u32 * ONE), // +6
                BN254::from(7u32 * ONE), // -7
                BN254::from(8u32 * ONE), // -8
            ],
            output: vec![
                BN254::from(4u32 * ONE),  // -4
                BN254::from(8u32 * ONE),  // -8
                BN254::from(10u32 * ONE), // +10
                BN254::from(4u32 * ONE),  // +4
            ],
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);

        // Test incorrect subtraction
        let wrong_assignment = TestCircuit::<BN254> {
            input1: vec![
                BN254::from(1u32 * ONE), // +1
                BN254::from(2u32 * ONE), // -2
                BN254::from(3u32 * ONE), // +3
                BN254::from(4u32 * ONE), // -4
            ],
            input2: vec![
                BN254::from(5u32 * ONE), // +5
                BN254::from(6u32 * ONE), // +6
                BN254::from(7u32 * ONE), // -7
                BN254::from(8u32 * ONE), // -8
            ],
            output: vec![
                BN254::from(4u32 * ONE),  // -4
                BN254::from(8u32 * ONE),  // -8
                BN254::from(10u32 * ONE), // +10
                BN254::from(5u32 * ONE),  // Wrong value
            ],
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&wrong_assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![false]);
    }

    #[test]
    fn test_verify_tensor_sub_all_positive() {
        declare_circuit!(TestCircuit {
            input1: [Variable],
            input2: [Variable],
            output: [Variable],
        });

        impl<C: Config> Define<C> for TestCircuit<Variable> {
            fn define(&self, builder: &mut API<C>) {
                let a_signs = vec![true, true, true, true];
                let b_signs = vec![true, true, true, true];
                let c_signs = vec![false, false, false, false]; // Results are negative since b > a

                let result = verify_tensor_sub(
                    builder,
                    &self.input1,
                    &self.input2,
                    &self.output,
                    &a_signs,
                    &b_signs,
                    &c_signs,
                    &[2, 2],
                );
                let true_const = builder.constant(C::CircuitField::from(1u32));
                builder.assert_is_equal(result, true_const);
            }
        }

        let circuit = TestCircuit {
            input1: vec![Variable::default(); 4],
            input2: vec![Variable::default(); 4],
            output: vec![Variable::default(); 4],
        };

        let compile_result = compile::<BN254Config, TestCircuit<Variable>>(&circuit).unwrap();

        let assignment = TestCircuit::<BN254> {
            input1: vec![
                BN254::from(1u32 * ONE),
                BN254::from(2u32 * ONE),
                BN254::from(3u32 * ONE),
                BN254::from(4u32 * ONE),
            ],
            input2: vec![
                BN254::from(2u32 * ONE),
                BN254::from(4u32 * ONE),
                BN254::from(6u32 * ONE),
                BN254::from(8u32 * ONE),
            ],
            output: vec![
                BN254::from(1u32 * ONE), // -1 (1-2)
                BN254::from(2u32 * ONE), // -2 (2-4)
                BN254::from(3u32 * ONE), // -3 (3-6)
                BN254::from(4u32 * ONE), // -4 (4-8)
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
    fn test_verify_tensor_sub_all_negative() {
        declare_circuit!(TestCircuit {
            input1: [Variable],
            input2: [Variable],
            output: [Variable],
        });

        impl<C: Config> Define<C> for TestCircuit<Variable> {
            fn define(&self, builder: &mut API<C>) {
                let a_signs = vec![false, false, false, false];
                let b_signs = vec![false, false, false, false];
                let c_signs = vec![false, false, false, false];

                let result = verify_tensor_sub(
                    builder,
                    &self.input1,
                    &self.input2,
                    &self.output,
                    &a_signs,
                    &b_signs,
                    &c_signs,
                    &[2, 2],
                );
                let true_const = builder.constant(C::CircuitField::from(1u32));
                builder.assert_is_equal(result, true_const);
            }
        }

        let circuit = TestCircuit {
            input1: vec![Variable::default(); 4],
            input2: vec![Variable::default(); 4],
            output: vec![Variable::default(); 4],
        };

        let compile_result = compile::<BN254Config, TestCircuit<Variable>>(&circuit).unwrap();

        let assignment = TestCircuit::<BN254> {
            input1: vec![
                BN254::from(5u32 * ONE), // -5
                BN254::from(6u32 * ONE), // -6
                BN254::from(7u32 * ONE), // -7
                BN254::from(8u32 * ONE), // -8
            ],
            input2: vec![
                BN254::from(2u32 * ONE), // -2
                BN254::from(3u32 * ONE), // -3
                BN254::from(4u32 * ONE), // -4
                BN254::from(5u32 * ONE), // -5
            ],
            output: vec![
                BN254::from(3u32 * ONE), // -3 (-5-(-2))
                BN254::from(3u32 * ONE), // -3 (-6-(-3))
                BN254::from(3u32 * ONE), // -3 (-7-(-4))
                BN254::from(3u32 * ONE), // -3 (-8-(-5))
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
    fn test_verify_tensor_sub_zero_result() {
        declare_circuit!(TestCircuit {
            input1: [Variable],
            input2: [Variable],
            output: [Variable],
        });

        impl<C: Config> Define<C> for TestCircuit<Variable> {
            fn define(&self, builder: &mut API<C>) {
                let a_signs = vec![true, true, false, false];
                let b_signs = vec![true, true, false, false];
                let c_signs = vec![true, true, true, true]; // Zero is considered positive

                let result = verify_tensor_sub(
                    builder,
                    &self.input1,
                    &self.input2,
                    &self.output,
                    &a_signs,
                    &b_signs,
                    &c_signs,
                    &[2, 2],
                );
                let true_const = builder.constant(C::CircuitField::from(1u32));
                builder.assert_is_equal(result, true_const);
            }
        }

        let circuit = TestCircuit {
            input1: vec![Variable::default(); 4],
            input2: vec![Variable::default(); 4],
            output: vec![Variable::default(); 4],
        };

        let compile_result = compile::<BN254Config, TestCircuit<Variable>>(&circuit).unwrap();

        let assignment = TestCircuit::<BN254> {
            input1: vec![
                BN254::from(5u32 * ONE), // +5
                BN254::from(3u32 * ONE), // +3
                BN254::from(5u32 * ONE), // -5
                BN254::from(3u32 * ONE), // -3
            ],
            input2: vec![
                BN254::from(5u32 * ONE), // +5
                BN254::from(3u32 * ONE), // +3
                BN254::from(5u32 * ONE), // -5
                BN254::from(3u32 * ONE), // -3
            ],
            output: vec![
                BN254::from(0u32 * ONE),
                BN254::from(0u32 * ONE),
                BN254::from(0u32 * ONE),
                BN254::from(0u32 * ONE),
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
    fn test_verify_tensor_sub_large_numbers() {
        declare_circuit!(TestCircuit {
            input1: [Variable],
            input2: [Variable],
            output: [Variable],
        });

        impl<C: Config> Define<C> for TestCircuit<Variable> {
            fn define(&self, builder: &mut API<C>) {
                let a_signs = vec![true, true, true, true];
                let b_signs = vec![true, true, true, true];
                let c_signs = vec![false, false, false, false];

                let result = verify_tensor_sub(
                    builder,
                    &self.input1,
                    &self.input2,
                    &self.output,
                    &a_signs,
                    &b_signs,
                    &c_signs,
                    &[2, 2],
                );
                let true_const = builder.constant(C::CircuitField::from(1u32));
                builder.assert_is_equal(result, true_const);
            }
        }

        let circuit = TestCircuit {
            input1: vec![Variable::default(); 4],
            input2: vec![Variable::default(); 4],
            output: vec![Variable::default(); 4],
        };

        let compile_result = compile::<BN254Config, TestCircuit<Variable>>(&circuit).unwrap();

        let assignment = TestCircuit::<BN254> {
            input1: vec![
                BN254::from(1000u32 * ONE),
                BN254::from(2000u32 * ONE),
                BN254::from(3000u32 * ONE),
                BN254::from(4000u32 * ONE),
            ],
            input2: vec![
                BN254::from(2000u32 * ONE),
                BN254::from(3000u32 * ONE),
                BN254::from(4000u32 * ONE),
                BN254::from(5000u32 * ONE),
            ],
            output: vec![
                BN254::from(1000u32 * ONE), // -1000
                BN254::from(1000u32 * ONE), // -1000
                BN254::from(1000u32 * ONE), // -1000
                BN254::from(1000u32 * ONE), // -1000
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
    fn test_verify_tensor_sub_mixed_signs() {
        declare_circuit!(TestCircuit {
            input1: [Variable],
            input2: [Variable],
            output: [Variable],
        });

        impl<C: Config> Define<C> for TestCircuit<Variable> {
            fn define(&self, builder: &mut API<C>) {
                let a_signs = vec![true, false, true, false];
                let b_signs = vec![false, true, false, true];
                let c_signs = vec![true, false, true, false];

                let result = verify_tensor_sub(
                    builder,
                    &self.input1,
                    &self.input2,
                    &self.output,
                    &a_signs,
                    &b_signs,
                    &c_signs,
                    &[2, 2],
                );
                let true_const = builder.constant(C::CircuitField::from(1u32));
                builder.assert_is_equal(result, true_const);
            }
        }

        let circuit = TestCircuit {
            input1: vec![Variable::default(); 4],
            input2: vec![Variable::default(); 4],
            output: vec![Variable::default(); 4],
        };

        let compile_result = compile::<BN254Config, TestCircuit<Variable>>(&circuit).unwrap();

        let assignment = TestCircuit::<BN254> {
            input1: vec![
                BN254::from(5u32 * ONE), // +5
                BN254::from(6u32 * ONE), // -6
                BN254::from(7u32 * ONE), // +7
                BN254::from(8u32 * ONE), // -8
            ],
            input2: vec![
                BN254::from(2u32 * ONE), // -2
                BN254::from(3u32 * ONE), // +3
                BN254::from(4u32 * ONE), // -4
                BN254::from(5u32 * ONE), // +5
            ],
            output: vec![
                BN254::from(7u32 * ONE),  // +7 (5-(-2))
                BN254::from(9u32 * ONE),  // -9 (-6-3)
                BN254::from(11u32 * ONE), // +11 (7-(-4))
                BN254::from(13u32 * ONE), // -13 (-8-5)
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
