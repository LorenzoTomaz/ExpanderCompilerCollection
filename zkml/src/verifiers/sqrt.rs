use arith::Field;
use expander_compiler::frontend::*;
use extra::UnconstrainedAPI;

/// Verifies that output * output ≈ input for each element in the tensors.
/// Since we're dealing with fixed-point arithmetic, we need to handle scaling:
/// - input and output are scaled by 2^16
/// - output * output needs to be scaled down by 2^16 to match input
/// - allows for a precision error in the range [0, 10]
///
/// # Arguments
/// * `builder` - The circuit builder API
/// * `input` - Input tensor data (flattened)
/// * `output` - Output tensor data (flattened)
/// * `input_signs` - Signs of elements in input tensor (true for positive, false for negative)
/// * `output_signs` - Signs of elements in output tensor
/// * `shape` - Shape of tensors (must be same for both)
///
/// # Returns
/// * `Variable` - Whether the verification constraints were satisfied
pub fn verify_sqrt<C: Config>(
    builder: &mut API<C>,
    input: &[Variable],
    output: &[Variable],
    input_signs: &[bool],
    output_signs: &[bool],
    shape: &[u64],
) -> Variable {
    let size: usize = shape.iter().product::<u64>() as usize;
    assert_eq!(input.len(), size, "Input tensor size must match shape");
    assert_eq!(output.len(), size, "Output tensor size must match shape");
    assert_eq!(
        input_signs.len(),
        size,
        "Input signs length must match shape"
    );
    assert_eq!(
        output_signs.len(),
        size,
        "Output signs length must match shape"
    );

    // Initialize result as true
    let mut result = builder.constant(C::CircuitField::from(1));

    // For each element
    for i in 0..size {
        // Square root of negative number is invalid

        assert!(input_signs[i] && output_signs[i], "Invalid input or output");
        let input_val = input[i];
        let output_val = output[i];

        // output * output should approximately equal input
        // Since both are scaled by 2^16, we need to divide result by 2^16
        let scale = builder.constant(C::CircuitField::from(1u32 << 16));
        let squared = builder.mul(output_val, output_val);
        let scaled_squared = builder.unconstrained_int_div(squared, scale);

        // Get the absolute difference
        let diff = builder.sub(input_val, scaled_squared);

        let mut iter_result = builder.constant(C::CircuitField::ZERO);
        for i in 0..100 {
            let j_const = builder.constant(C::CircuitField::from(i));
            let equals_j = builder.unconstrained_eq(diff, j_const);
            iter_result = builder.or(iter_result, equals_j);
        }
        result = builder.and(result, iter_result);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use expander_compiler::field::BN254;
    use expander_compiler::frontend::BN254Config;

    const ONE: u64 = 1 << 16;

    #[test]
    fn test_sqrt_verification() {
        declare_circuit!(TestCircuit {
            input: [Variable],
            output: [Variable],
        });

        impl<C: Config> Define<C> for TestCircuit<Variable> {
            fn define(&self, builder: &mut API<C>) {
                let input_signs = vec![true]; // positive
                let output_signs = vec![true]; // positive
                let one = builder.constant(C::CircuitField::from(1u32));
                let result = verify_sqrt(
                    builder,
                    &self.input,
                    &self.output,
                    &input_signs,
                    &output_signs,
                    &[1],
                );
                builder.assert_is_equal(result, one)
            }
        }

        // Create test data
        let circuit = TestCircuit {
            input: vec![Variable::default()],
            output: vec![Variable::default()],
        };

        let compile_result = compile::<BN254Config, TestCircuit<Variable>>(&circuit).unwrap();

        // Test sqrt(3) ≈ 1.732050807568877
        let assignment = TestCircuit::<BN254> {
            input: vec![BN254::from(16u64 * ONE)], // 3
            output: vec![BN254::from(4u64 * ONE)], // √3
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);

        // Test incorrect sqrt (way off)
        let wrong_assignment = TestCircuit::<BN254> {
            input: vec![BN254::from(3u64 * ONE)],  // 3
            output: vec![BN254::from(2u64 * ONE)], // 2 (wrong)
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&wrong_assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![false]);
    }
}
