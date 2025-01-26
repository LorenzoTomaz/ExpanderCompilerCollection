use arith::Field;
use expander_compiler::frontend::*;

/// Verifies that the element-wise addition of two tensors equals the output tensor.
///
/// # Arguments
/// * `builder` - The circuit builder API
/// * `input1` - First input tensor data (flattened)
/// * `input2` - Second input tensor data (flattened)
/// * `output` - Output tensor data (flattened)
/// * `shape` - Shape of all tensors (must be same for all)
///
/// # Returns
/// * `bool` - Whether the verification constraints were satisfied
pub fn verify_tensor_add<C: Config>(
    builder: &mut API<C>,
    input1: &[Variable],
    input2: &[Variable],
    output: &[Variable],
    shape: &[u64],
) {
    // Verify all tensors have same number of elements
    let n_elements = input1.len();
    assert_eq!(
        input2.len(),
        n_elements,
        "Input tensors must have same length"
    );
    assert_eq!(
        output.len(),
        n_elements,
        "Output tensor must have same length as inputs"
    );
    assert_eq!(
        n_elements,
        shape.iter().product::<u64>() as usize,
        "Shape product must match tensor lengths"
    );

    // For each element, verify input1[i] + input2[i] = output[i]
    // Which is equivalent to input1[i] + input2[i] - output[i] = 0
    let zero = builder.constant(0);
    for i in 0..n_elements {
        let sum = builder.add(input1[i], input2[i]);
        let diff = builder.sub(sum, output[i]);
        builder.assert_is_equal(diff, zero);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use expander_compiler::field::M31;
    use expander_compiler::frontend::M31Config;

    #[test]
    fn test_verify_tensor_add() {
        // Create a simple circuit to test the verifier
        declare_circuit!(TestCircuit {
            input1: [Variable],
            input2: [Variable],
            output: [Variable],
        });

        impl<C: Config> Define<C> for TestCircuit<Variable> {
            fn define(&self, builder: &mut API<C>) {
                verify_tensor_add(builder, &self.input1, &self.input2, &self.output, &[2, 2]);
            }
        }

        // Create test data
        let circuit = TestCircuit {
            input1: vec![Variable::default(); 4],
            input2: vec![Variable::default(); 4],
            output: vec![Variable::default(); 4],
        };

        let compile_result = compile::<M31Config, TestCircuit<Variable>>(&circuit).unwrap();

        // Test correct addition
        let assignment = TestCircuit::<M31> {
            input1: vec![M31::from(1), M31::from(2), M31::from(3), M31::from(4)],
            input2: vec![M31::from(5), M31::from(6), M31::from(7), M31::from(8)],
            output: vec![M31::from(6), M31::from(8), M31::from(10), M31::from(12)],
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);

        // Test incorrect addition
        let wrong_assignment = TestCircuit::<M31> {
            input1: vec![M31::from(1), M31::from(2), M31::from(3), M31::from(4)],
            input2: vec![M31::from(5), M31::from(6), M31::from(7), M31::from(8)],
            output: vec![M31::from(6), M31::from(8), M31::from(10), M31::from(13)], // Last element is wrong
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&wrong_assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![false]);
    }
}
