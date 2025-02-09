use arith::Field;
use expander_compiler::frontend::*;
use extra::UnconstrainedAPI;

/// Helper struct to represent a complex number in the circuit
struct ComplexVar {
    real: Variable,
    imag: Variable,
    real_sign: Variable,
    imag_sign: Variable,
}

impl ComplexVar {
    fn new(real: Variable, imag: Variable, real_sign: Variable, imag_sign: Variable) -> Self {
        Self {
            real,
            imag,
            real_sign,
            imag_sign,
        }
    }

    /// Extract complex number from flattened arrays at given index
    fn from_arrays<C: Config>(
        builder: &mut API<C>,
        values: &[Variable],
        signs: &[bool],
        idx: usize,
    ) -> Self {
        let one = builder.constant(C::CircuitField::from(1u32));
        let zero = builder.constant(C::CircuitField::ZERO);
        Self {
            real: values[idx * 2],
            imag: values[idx * 2 + 1],
            real_sign: if signs[idx * 2] { one } else { zero },
            imag_sign: if signs[idx * 2 + 1] { one } else { zero },
        }
    }

    /// Extract complex number from flattened arrays with Variable signs at given index
    fn from_arrays_convert<C: Config>(
        builder: &mut API<C>,
        values: &[Variable],
        signs: &[Variable],
        idx: usize,
    ) -> Self {
        Self {
            real: values[idx * 2],
            imag: values[idx * 2 + 1],
            real_sign: signs[idx * 2],
            imag_sign: signs[idx * 2 + 1],
        }
    }
}

/// Helper functions for complex arithmetic in the circuit
fn complex_add<C: Config>(builder: &mut API<C>, a: &ComplexVar, b: &ComplexVar) -> ComplexVar {
    let one = builder.constant(C::CircuitField::from(1u32));
    let zero = builder.constant(C::CircuitField::ZERO);

    // Add real parts
    let real = {
        let real_signs_eq = builder.unconstrained_eq(a.real_sign, b.real_sign);
        let imag_signs_eq = builder.unconstrained_eq(a.imag_sign, b.imag_sign);
        let signs_same = builder.unconstrained_eq(real_signs_eq, imag_signs_eq);

        // Same signs path: just add and keep sign
        let sum = builder.add(a.real, b.real);
        let same_signs_result = (sum, a.real_sign);

        // Different signs path: subtract and check if result > 2^31
        let max_val = builder.constant(C::CircuitField::from(1u32 << 31));
        let diff = builder.sub(a.real, b.real);

        // Check if result > 2^31
        let is_large = builder.unconstrained_lesser(max_val, diff);
        let should_negate = builder.unconstrained_eq(is_large, one);
        let one_minus_should = builder.unconstrained_not_eq(should_negate, one);

        // Compute both possibilities and select based on should_negate
        let neg_result = builder.neg(diff);
        let pos_term = builder.mul(diff, one_minus_should);
        let neg_term = builder.mul(neg_result, should_negate);
        let result = builder.add(pos_term, neg_term);

        // Sign is determined by whether we negated
        let sign = builder.unconstrained_eq(should_negate, zero);
        let diff_signs_result = (result, sign);

        // Select between same signs and different signs paths
        let one_minus_signs_same = builder.sub(one, signs_same);
        let diff_term = builder.mul(diff_signs_result.0, one_minus_signs_same);
        let final_mag = builder.mul(same_signs_result.0, signs_same);
        let final_mag = builder.add(final_mag, diff_term);

        let sign_term = builder.mul(diff_signs_result.1, one_minus_signs_same);
        let final_sign = builder.mul(same_signs_result.1, signs_same);
        let final_sign = builder.add(final_sign, sign_term);

        (final_mag, final_sign)
    };

    // Add imaginary parts (same logic as real parts)
    let imag = {
        let real_signs_eq = builder.unconstrained_eq(a.real_sign, b.real_sign);
        let imag_signs_eq = builder.unconstrained_eq(a.imag_sign, b.imag_sign);
        let signs_same = builder.unconstrained_eq(real_signs_eq, imag_signs_eq);

        // Same signs path: just add and keep sign
        let sum = builder.add(a.imag, b.imag);
        let same_signs_result = (sum, a.imag_sign);

        // Different signs path: subtract and check if result > 2^31
        let max_val = builder.constant(C::CircuitField::from(1u32 << 31));
        let diff = builder.sub(a.imag, b.imag);

        // Check if result > 2^31
        let is_large = builder.unconstrained_lesser(max_val, diff);
        let should_negate = builder.unconstrained_eq(is_large, one);
        let one_minus_should = builder.unconstrained_not_eq(should_negate, one);

        // Compute both possibilities and select based on should_negate
        let neg_result = builder.neg(diff);
        let pos_term = builder.mul(diff, one_minus_should);
        let neg_term = builder.mul(neg_result, should_negate);
        let result = builder.add(pos_term, neg_term);

        // Sign is determined by whether we negated
        let sign = builder.unconstrained_eq(should_negate, zero);
        let diff_signs_result = (result, sign);

        // Select between same signs and different signs paths
        let one_minus_signs_same = builder.sub(one, signs_same);
        let diff_term = builder.mul(diff_signs_result.0, one_minus_signs_same);
        let final_mag = builder.mul(same_signs_result.0, signs_same);
        let final_mag = builder.add(final_mag, diff_term);

        let sign_term = builder.mul(diff_signs_result.1, one_minus_signs_same);
        let final_sign = builder.mul(same_signs_result.1, signs_same);
        let final_sign = builder.add(final_sign, sign_term);

        (final_mag, final_sign)
    };

    ComplexVar::new(real.0, imag.0, real.1, imag.1)
}

fn complex_sub<C: Config>(builder: &mut API<C>, a: &ComplexVar, b: &ComplexVar) -> ComplexVar {
    // Negate b's signs by subtracting from 1
    let one = builder.constant(C::CircuitField::from(1u32));
    let neg_b = ComplexVar::new(
        b.real,
        b.imag,
        builder.sub(one, b.real_sign),
        builder.sub(one, b.imag_sign),
    );
    complex_add(builder, a, &neg_b)
}

fn complex_mul<C: Config>(
    builder: &mut API<C>,
    a: &ComplexVar,
    b: &ComplexVar,
    one: Variable,
) -> ComplexVar {
    // (a + bi)(c + di) = (ac - bd) + (ad + bc)i

    // Compute products
    let ac = builder.mul(a.real, b.real);
    let bd = builder.mul(a.imag, b.imag);
    let ad = builder.mul(a.real, b.imag);
    let bc = builder.mul(a.imag, b.real);

    // Scale products
    let ac_scaled = builder.unconstrained_int_div(ac, one);
    let bd_scaled = builder.unconstrained_int_div(bd, one);
    let ad_scaled = builder.unconstrained_int_div(ad, one);
    let bc_scaled = builder.unconstrained_int_div(bc, one);

    let one_const = builder.constant(C::CircuitField::from(1u32));

    // Real part: ac - bd
    let real = {
        let real_signs_eq = builder.unconstrained_eq(a.real_sign, b.real_sign);
        let imag_signs_eq = builder.unconstrained_eq(a.imag_sign, b.imag_sign);
        let signs_same = builder.unconstrained_eq(real_signs_eq, imag_signs_eq);

        // Signs are same, subtract magnitudes
        let diff = builder.sub(ac_scaled, bd_scaled);
        let sum = builder.add(ac_scaled, bd_scaled);

        // Use arithmetic to select: signs_same * diff + (1-signs_same) * sum
        let one = builder.constant(C::CircuitField::from(1u32));
        let not_signs_same = builder.sub(one, signs_same);
        let diff_term = builder.mul(signs_same, diff);
        let sum_term = builder.mul(not_signs_same, sum);
        let mag = builder.add(diff_term, sum_term);

        // Compute sign: XOR of input signs
        let real_sign = builder.unconstrained_eq(real_signs_eq, one_const);

        (mag, real_sign)
    };

    // Imaginary part: ad + bc
    let imag = {
        let real_signs_eq = builder.unconstrained_eq(a.real_sign, b.imag_sign);
        let imag_signs_eq = builder.unconstrained_eq(a.imag_sign, b.real_sign);
        let signs_same = builder.unconstrained_eq(real_signs_eq, imag_signs_eq);

        // Signs are same, add magnitudes
        let sum = builder.add(ad_scaled, bc_scaled);
        let diff = builder.sub(ad_scaled, bc_scaled);

        // Use arithmetic to select: signs_same * sum + (1-signs_same) * diff
        let one = builder.constant(C::CircuitField::from(1u32));
        let not_signs_same = builder.sub(one, signs_same);
        let sum_term = builder.mul(signs_same, sum);
        let diff_term = builder.mul(not_signs_same, diff);
        let mag = builder.add(sum_term, diff_term);

        // Compute sign: XOR of input signs
        let imag_sign = builder.unconstrained_eq(real_signs_eq, one_const);

        (mag, imag_sign)
    };

    ComplexVar::new(real.0, imag.0, real.1, imag.1)
}

/// Helper function to convert bool signs to Variable signs
fn convert_bool_to_signs<C: Config>(builder: &mut API<C>, signs: &[bool]) -> Vec<Variable> {
    let one = builder.constant(C::CircuitField::from(1u32));
    let zero = builder.constant(C::CircuitField::ZERO);
    signs
        .iter()
        .map(|&sign| if sign { one } else { zero })
        .collect()
}

/// Verifies that the claimed FFT output matches the input polynomial using the Cooley-Tukey algorithm.
/// Uses a recursive divide-and-conquer approach with random linear combinations for efficiency.
///
/// # Arguments
/// * `builder` - The circuit builder API
/// * `input` - Input polynomial coefficients (flattened complex numbers: [real0, imag0, real1, imag1, ...])
/// * `output` - Claimed FFT output values (flattened complex numbers: [real0, imag0, real1, imag1, ...])
/// * `input_signs` - Signs for input values [real0_sign, imag0_sign, real1_sign, imag1_sign, ...]
/// * `output_signs` - Signs for output values [real0_sign, imag0_sign, real1_sign, imag1_sign, ...]
/// * `roots` - Pre-computed roots of unity for each recursive step (flattened complex numbers)
/// * `root_signs` - Signs for the roots of unity
/// * `shape` - Shape of input/output tensors (must be power of 2)
/// * `level` - Current recursion level (starts at 0)
pub fn verify_fft<C: Config>(
    builder: &mut API<C>,
    input: &[Variable],
    output: &[Variable],
    input_signs: &[bool],
    output_signs: &[bool],
    roots: &[Variable],
    root_signs: &[bool],
    shape: &[u64],
    level: usize,
) -> Variable {
    let input_signs_var = convert_bool_to_signs(builder, input_signs);
    let output_signs_var = convert_bool_to_signs(builder, output_signs);
    let root_signs_var = convert_bool_to_signs(builder, root_signs);

    verify_fft_internal(
        builder,
        input,
        output,
        &input_signs_var,
        &output_signs_var,
        roots,
        &root_signs_var,
        shape,
        level,
    )
}

/// Internal implementation that works with Variable signs
fn verify_fft_internal<C: Config>(
    builder: &mut API<C>,
    input: &[Variable],
    output: &[Variable],
    input_signs: &[Variable],
    output_signs: &[Variable],
    roots: &[Variable],
    root_signs: &[Variable],
    shape: &[u64],
    level: usize,
) -> Variable {
    let size: usize = shape.iter().product::<u64>() as usize;
    assert_eq!(
        input.len(),
        size * 2,
        "Input tensor size must match shape * 2 (complex numbers)"
    );
    assert_eq!(
        output.len(),
        size * 2,
        "Output tensor size must match shape * 2 (complex numbers)"
    );
    assert_eq!(
        input_signs.len(),
        size * 2,
        "Input signs length must match shape * 2"
    );
    assert_eq!(
        output_signs.len(),
        size * 2,
        "Output signs length must match shape * 2"
    );
    assert!(size.is_power_of_two(), "Size must be a power of 2");

    // Constants
    let one = builder.constant(C::CircuitField::from(1u32 << 16)); // 2^16 for fixed-point
    let half = builder.constant(C::CircuitField::from(1u32 << 15)); // 0.5 * 2^16

    // Base case: size = 1
    if size == 1 {
        let input_c = ComplexVar::from_arrays_convert(builder, input, input_signs, 0);
        let output_c = ComplexVar::from_arrays_convert(builder, output, output_signs, 0);

        // Compare difference with threshold
        let diff = complex_sub(builder, &output_c, &input_c);
        let threshold = builder.constant(C::CircuitField::from(256)); // 0.00390625 * 2^16

        let real_check = builder.unconstrained_lesser(diff.real, threshold);
        let imag_check = builder.unconstrained_lesser(diff.imag, threshold);

        return builder.and(real_check, imag_check);
    }

    let half_size = size / 2;

    // Split input into even and odd parts
    let mut even = Vec::with_capacity(half_size * 2);
    let mut odd = Vec::with_capacity(half_size * 2);
    let mut even_signs = Vec::with_capacity(half_size * 2);
    let mut odd_signs = Vec::with_capacity(half_size * 2);

    for i in 0..half_size {
        // Copy real and imaginary parts
        even.extend_from_slice(&input[4 * i..4 * i + 2]);
        odd.extend_from_slice(&input[4 * i + 2..4 * i + 4]);
        even_signs.extend_from_slice(&input_signs[4 * i..4 * i + 2]);
        odd_signs.extend_from_slice(&input_signs[4 * i + 2..4 * i + 4]);
    }

    // Get random value for linear combination
    let r = ComplexVar::from_arrays_convert(builder, roots, root_signs, level);

    // Compute linear combination of even and odd parts
    let mut rlc_even_odd = Vec::with_capacity(half_size * 2);
    let mut rlc_signs = Vec::with_capacity(half_size * 2);

    for i in 0..half_size {
        let even_c = ComplexVar::from_arrays_convert(builder, &even, &even_signs, i);
        let odd_c = ComplexVar::from_arrays_convert(builder, &odd, &odd_signs, i);

        // r * odd[i]
        let r_odd = complex_mul(builder, &r, &odd_c, one);

        // even[i] + r * odd[i]
        let combined = complex_add(builder, &even_c, &r_odd);

        rlc_even_odd.extend_from_slice(&[combined.real, combined.imag]);
        let zero = builder.constant(C::CircuitField::ZERO);
        let one = builder.constant(C::CircuitField::from(1u32));
        let real_sign = builder.unconstrained_lesser(zero, combined.real_sign);
        let imag_sign = builder.unconstrained_lesser(zero, combined.imag_sign);
        rlc_signs.extend_from_slice(&[real_sign, imag_sign]);
    }

    // Extract odd component from claimed FFT
    let mut odd_fft = Vec::with_capacity(half_size * 2);
    let mut odd_fft_signs = Vec::with_capacity(half_size * 2);

    for i in 0..half_size {
        let out_1 = ComplexVar::from_arrays_convert(builder, output, output_signs, i);
        let out_2 = ComplexVar::from_arrays_convert(builder, output, output_signs, i + half_size);

        let diff = complex_sub(builder, &out_1, &out_2);
        odd_fft.extend_from_slice(&[diff.real, diff.imag]);
        let zero = builder.constant(C::CircuitField::ZERO);
        let one = builder.constant(C::CircuitField::from(1u32));
        let real_sign = builder.unconstrained_lesser(zero, diff.real_sign);
        let imag_sign = builder.unconstrained_lesser(zero, diff.imag_sign);
        odd_fft_signs.extend_from_slice(&[real_sign, imag_sign]);
    }

    // Apply twiddle factors and scaling
    let mut scaled_odd_fft = Vec::with_capacity(half_size * 2);
    let mut scaled_odd_signs = Vec::with_capacity(half_size * 2);

    for i in 0..half_size {
        let odd_c = ComplexVar::from_arrays_convert(builder, &odd_fft, &odd_fft_signs, i);
        let root_signs_var = root_signs;
        let twiddle =
            ComplexVar::from_arrays_convert(builder, roots, &root_signs_var, level + 1 + i);

        // First multiply by twiddle
        let twiddled = complex_mul(builder, &odd_c, &twiddle, one);

        // Then scale by HALF
        let half_c = ComplexVar::new(
            half,
            builder.constant(C::CircuitField::ZERO),
            builder.constant(C::CircuitField::from(1u32)),
            builder.constant(C::CircuitField::from(1u32)),
        );
        let scaled = complex_mul(builder, &twiddled, &half_c, one);

        scaled_odd_fft.extend_from_slice(&[scaled.real, scaled.imag]);
        let zero = builder.constant(C::CircuitField::ZERO);
        let one = builder.constant(C::CircuitField::from(1u32));
        let real_sign = builder.unconstrained_lesser(zero, scaled.real_sign);
        let imag_sign = builder.unconstrained_lesser(zero, scaled.imag_sign);
        scaled_odd_signs.extend_from_slice(&[real_sign, imag_sign]);
    }

    // Extract and scale even component
    let mut even_fft = Vec::with_capacity(half_size * 2);
    let mut even_fft_signs = Vec::with_capacity(half_size * 2);

    for i in 0..half_size {
        let out_1 = ComplexVar::from_arrays_convert(builder, output, output_signs, i);
        let out_2 = ComplexVar::from_arrays_convert(builder, output, output_signs, i + half_size);

        let sum = complex_add(builder, &out_1, &out_2);

        // Scale by HALF
        let half_c = ComplexVar::new(
            half,
            builder.constant(C::CircuitField::ZERO),
            builder.constant(C::CircuitField::from(1u32)),
            builder.constant(C::CircuitField::from(1u32)),
        );
        let scaled = complex_mul(builder, &sum, &half_c, one);

        even_fft.extend_from_slice(&[scaled.real, scaled.imag]);
        let zero = builder.constant(C::CircuitField::ZERO);
        let one = builder.constant(C::CircuitField::from(1u32));
        let real_sign = builder.unconstrained_lesser(zero, scaled.real_sign);
        let imag_sign = builder.unconstrained_lesser(zero, scaled.imag_sign);
        even_fft_signs.extend_from_slice(&[real_sign, imag_sign]);
    }

    // Combine components with linear combination
    let mut combined = Vec::with_capacity(half_size * 2);
    let mut combined_signs = Vec::with_capacity(half_size * 2);

    for i in 0..half_size {
        let even_c = ComplexVar::from_arrays_convert(builder, &even_fft, &even_fft_signs, i);
        let odd_c = ComplexVar::from_arrays_convert(builder, &scaled_odd_fft, &scaled_odd_signs, i);

        // r * odd_fft[i]
        let r_odd = complex_mul(builder, &r, &odd_c, one);

        // even_fft[i] + r * odd_fft[i]
        let result = complex_add(builder, &even_c, &r_odd);

        combined.extend_from_slice(&[result.real, result.imag]);
        let zero = builder.constant(C::CircuitField::ZERO);
        let one = builder.constant(C::CircuitField::from(1u32));
        let real_sign = builder.unconstrained_lesser(zero, result.real_sign);
        let imag_sign = builder.unconstrained_lesser(zero, result.imag_sign);
        combined_signs.extend_from_slice(&[real_sign, imag_sign]);
    }

    // Keep signs as Variables for recursive verification
    verify_fft_internal(
        builder,
        &rlc_even_odd,
        &combined,
        &rlc_signs,
        &combined_signs,
        roots,
        root_signs,
        &[half_size as u64],
        level + half_size + 1,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use expander_compiler::field::BN254;
    use expander_compiler::frontend::BN254Config;

    const ONE: u64 = 1 << 16;

    #[test]
    fn test_verify_fft_size_2() {
        declare_circuit!(TestCircuit {
            input: [Variable],
            output: [Variable],
            roots: [Variable],
        });

        impl<C: Config> Define<C> for TestCircuit<Variable> {
            fn define(&self, builder: &mut API<C>) {
                let input_signs = vec![true, true, true, true]; // Signs for real,imag parts
                let output_signs = vec![true, true, true, true];
                let root_signs = vec![true, true, true, true, true, true]; // Signs for complex roots

                let result = verify_fft(
                    builder,
                    &self.input,
                    &self.output,
                    &input_signs,
                    &output_signs,
                    &self.roots,
                    &root_signs,
                    &[2],
                    0,
                );

                let true_const = builder.constant(C::CircuitField::from(1u32));
                builder.assert_is_equal(result, true_const);
            }
        }

        let circuit = TestCircuit {
            input: vec![Variable::default(); 4], // 2 complex numbers
            output: vec![Variable::default(); 4],
            roots: vec![Variable::default(); 6], // 3 complex roots
        };

        let compile_result = compile::<BN254Config, TestCircuit<Variable>>(&circuit).unwrap();

        // Test with input [1+0i, 1+0i] -> FFT -> [2+0i, 0+0i]
        let assignment = TestCircuit::<BN254> {
            input: vec![
                BN254::from(1u64 * ONE),
                BN254::from(0u64), // 1+0i
                BN254::from(1u64 * ONE),
                BN254::from(0u64), // 1+0i
            ],
            output: vec![
                BN254::from(2u64 * ONE),
                BN254::from(0u64), // 2+0i
                BN254::from(0u64),
                BN254::from(0u64), // 0+0i
            ],
            roots: vec![
                BN254::from(123u64),
                BN254::from(0u64), // Random value for linear combination
                BN254::from(u64::MAX),
                BN254::from(0u64), // -1+0i as root of unity
                BN254::from(1u64),
                BN254::from(0u64), // 1+0i for recursion
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
    fn test_verify_fft_size_4() {
        declare_circuit!(TestCircuit {
            input: [Variable],
            output: [Variable],
            roots: [Variable],
        });

        impl<C: Config> Define<C> for TestCircuit<Variable> {
            fn define(&self, builder: &mut API<C>) {
                let input_signs = vec![true; 8]; // Signs for 4 complex numbers
                let output_signs = vec![true; 8];
                let root_signs = vec![true; 12]; // Signs for 6 complex roots

                let result = verify_fft(
                    builder,
                    &self.input,
                    &self.output,
                    &input_signs,
                    &output_signs,
                    &self.roots,
                    &root_signs,
                    &[4],
                    0,
                );

                let true_const = builder.constant(C::CircuitField::from(1u32));
                builder.assert_is_equal(result, true_const);
            }
        }

        let circuit = TestCircuit {
            input: vec![Variable::default(); 8], // 4 complex numbers
            output: vec![Variable::default(); 8],
            roots: vec![Variable::default(); 12], // 6 complex roots
        };

        let compile_result = compile::<BN254Config, TestCircuit<Variable>>(&circuit).unwrap();

        // Test with input [1+0i, 1+0i, 1+0i, 1+0i] -> FFT -> [4+0i, 0+0i, 0+0i, 0+0i]
        let assignment = TestCircuit::<BN254> {
            input: vec![
                BN254::from(1u64 * ONE),
                BN254::from(0u64),
                BN254::from(1u64 * ONE),
                BN254::from(0u64),
                BN254::from(1u64 * ONE),
                BN254::from(0u64),
                BN254::from(1u64 * ONE),
                BN254::from(0u64),
            ],
            output: vec![
                BN254::from(4u64 * ONE),
                BN254::from(0u64),
                BN254::from(0u64),
                BN254::from(0u64),
                BN254::from(0u64),
                BN254::from(0u64),
                BN254::from(0u64),
                BN254::from(0u64),
            ],
            roots: vec![
                BN254::from(123u64),
                BN254::from(0u64), // Random for first linear combination
                BN254::from((1u64 << 32) - 1),
                BN254::from(0u64), // First primitive 4th root
                BN254::from((1u64 << 16) - 1),
                BN254::from(0u64), // Second primitive 4th root
                BN254::from(456u64),
                BN254::from(0u64), // Random for second linear combination
                BN254::from(u64::MAX),
                BN254::from(0u64), // -1 as root of unity
                BN254::from(1u64),
                BN254::from(0u64), // Additional root
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
