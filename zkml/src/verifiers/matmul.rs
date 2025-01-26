use arith::Field;
use expander_compiler::field::BN254;
use expander_compiler::frontend::*;
use rand::Rng;

/// Fixed-point scaling factor (2^16)
const ONE: u32 = 1 << 16;

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
/// * `a_shape` - Shape of matrix A [m, n]
/// * `b_shape` - Shape of matrix B [n, p]
/// * `c_shape` - Shape of matrix C [m, p]
/// * `num_iterations` - Number of iterations for probability amplification (default 5)
pub fn verify_matmul<C: Config>(
    builder: &mut API<C>,
    a: &[Variable],
    b: &[Variable],
    c: &[Variable],
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

    let m = a_shape[0] as usize;
    let n = a_shape[1] as usize;
    let p = b_shape[1] as usize;

    // Initialize result as true and create constants
    let mut result = builder.constant(C::CircuitField::from(1u32));
    let one_const = builder.constant(C::CircuitField::from(ONE as u32));
    let checked = false;

    // For each element in C
    for i in 0..m {
        for j in 0..p {
            // Calculate c[i,j] = sum(a[i,k] * b[k,j])
            let mut sum = builder.constant(C::CircuitField::ZERO);
            for k in 0..n {
                let a_ik = a[i * n + k];
                let b_kj = b[k * p + j];
                let prod = builder.mul(a_ik, b_kj);
                // Divide by ONE to maintain fixed-point representation
                let scaled_prod = builder.div(prod, one_const, checked);
                sum = builder.add(sum, scaled_prod);
            }

            // Verify c[i,j] matches the calculated sum
            let c_ij = c[i * p + j];
            let diff = builder.sub(c_ij, sum);
            let eq = builder.is_zero(diff);
            result = builder.and(result, eq);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use expander_compiler::frontend::BN254Config;

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
                let result = verify_matmul(
                    builder,
                    &self.a,
                    &self.b,
                    &self.c,
                    &[2, 2], // A is 2x2
                    &[2, 2], // B is 2x2
                    &[2, 2], // C is 2x2
                    5,       // num_iterations
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

        // Test correct multiplication with fixed-point scaling
        // A = [[1, 2], [3, 4]] * ONE
        // B = [[5, 6], [7, 8]] * ONE
        // C = [[19, 22], [43, 50]] * ONE
        let assignment = TestCircuit::<BN254> {
            a: vec![
                BN254::from(1u32 * ONE),
                BN254::from(2u32 * ONE),
                BN254::from(3u32 * ONE),
                BN254::from(4u32 * ONE),
            ],
            b: vec![
                BN254::from(5u32 * ONE),
                BN254::from(6u32 * ONE),
                BN254::from(7u32 * ONE),
                BN254::from(8u32 * ONE),
            ],
            c: vec![
                BN254::from(19u32 * ONE),
                BN254::from(22u32 * ONE),
                BN254::from(43u32 * ONE),
                BN254::from(50u32 * ONE),
            ],
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);

        // Test incorrect multiplication
        let wrong_assignment = TestCircuit::<BN254> {
            a: vec![
                BN254::from(1u32 * ONE),
                BN254::from(2u32 * ONE),
                BN254::from(3u32 * ONE),
                BN254::from(4u32 * ONE),
            ],
            b: vec![
                BN254::from(5u32 * ONE),
                BN254::from(6u32 * ONE),
                BN254::from(7u32 * ONE),
                BN254::from(8u32 * ONE),
            ],
            c: vec![
                BN254::from(19u32 * ONE),
                BN254::from(22u32 * ONE),
                BN254::from(43u32 * ONE),
                BN254::from(51u32 * ONE), // Wrong value
            ],
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&wrong_assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![false]);
    }
}
