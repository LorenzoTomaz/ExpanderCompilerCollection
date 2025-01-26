use arith::Field;
use expander_compiler::frontend::*;

/// Verifies that the element-wise subtraction of two tensors equals the output tensor.
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
pub fn verify_tensor_sub<C: Config>(
    builder: &mut API<C>,
    a: &[Variable],
    b: &[Variable],
    c: &[Variable],
    shape: &[u64],
) -> Variable {
    let size: usize = shape.iter().product::<u64>() as usize;
    assert_eq!(a.len(), size, "Tensor A size must match shape");
    assert_eq!(b.len(), size, "Tensor B size must match shape");
    assert_eq!(c.len(), size, "Tensor C size must match shape");

    // Initialize result as true
    let mut result = builder.constant(C::CircuitField::from(1));

    // For each element
    for i in 0..size {
        let diff = builder.sub(a[i], b[i]);
        let diff2 = builder.sub(c[i], diff);
        let zero = builder.constant(C::CircuitField::ZERO);
        let eq = builder.is_zero(diff2);
        result = builder.and(result, eq);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use expander_compiler::field::M31;
    use expander_compiler::frontend::M31Config;

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
                let result =
                    verify_tensor_sub(builder, &self.input1, &self.input2, &self.output, &[2, 2]);
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

        let compile_result = compile::<M31Config, TestCircuit<Variable>>(&circuit).unwrap();

        // Test correct subtraction
        let assignment = TestCircuit::<M31> {
            input1: vec![M31::from(10), M31::from(8), M31::from(6), M31::from(4)],
            input2: vec![M31::from(3), M31::from(2), M31::from(1), M31::from(2)],
            output: vec![M31::from(7), M31::from(6), M31::from(5), M31::from(2)],
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);

        // Test incorrect subtraction
        let wrong_assignment = TestCircuit::<M31> {
            input1: vec![M31::from(10), M31::from(8), M31::from(6), M31::from(4)],
            input2: vec![M31::from(3), M31::from(2), M31::from(1), M31::from(2)],
            output: vec![M31::from(7), M31::from(6), M31::from(5), M31::from(3)], // Last element is wrong
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&wrong_assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![false]);
    }
}
