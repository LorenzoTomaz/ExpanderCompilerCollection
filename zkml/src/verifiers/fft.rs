use arith::{BN254Fr, Field};
use expander_compiler::field::BN254;
use expander_compiler::frontend::extra::*;
use expander_compiler::frontend::internal::DumpLoadTwoVariables;
use expander_compiler::frontend::*;

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
    fn from_arrays_convert<C: Config, Builder: RootAPI<C>>(
        builder: &mut Builder,
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
fn complex_add<C: Config, Builder: RootAPI<C>>(
    builder: &mut Builder,
    a: &ComplexVar,
    b: &ComplexVar,
) -> ComplexVar {
    let one = builder.constant(C::CircuitField::from(1u32));
    let zero = builder.constant(C::CircuitField::ZERO);

    // Add real parts
    let real = {
        let signs_same = builder.unconstrained_eq(a.real_sign, b.real_sign);
        builder.display("signs_same", signs_same);
        // Same signs path: just add and keep sign
        let sum = builder.add(a.real, b.real);
        builder.display("sum", sum);
        let same_signs_result = (sum, a.real_sign);

        // Different signs path: subtract and check if result > 2^31
        let max_val = builder.constant(C::CircuitField::from(1u32 << 31));
        let diff = builder.sub(a.real, b.real);
        builder.display("diff", diff);
        let is_large = builder.unconstrained_lesser(max_val, diff);
        builder.display("is_large", is_large);
        let should_negate = builder.unconstrained_eq(is_large, one);
        builder.display("should_negate", should_negate);
        let one_minus_should = builder.unconstrained_not_eq(should_negate, one);
        builder.display("one_minus_should", one_minus_should);
        // Compute magnitude
        let neg_result = builder.neg(diff);
        builder.display("neg_result", neg_result);
        let pos_term = builder.mul(diff, one_minus_should);
        builder.display("pos_term", pos_term);
        let neg_term = builder.mul(neg_result, should_negate);
        builder.display("neg_term", neg_term);
        let result = builder.add(pos_term, neg_term);
        builder.display("result", result);
        // Sign should be b's sign when |a| < |b| and a's sign when |a| > |b|
        let sign = builder.unconstrained_eq(is_large, b.real_sign);
        builder.display("sign", sign);
        let diff_signs_result = (result, sign);
        // Select between same signs and different signs paths
        let one_minus_signs_same = builder.sub(one, signs_same);
        builder.display("one_minus_signs_same", one_minus_signs_same);
        let diff_term = builder.mul(diff_signs_result.0, one_minus_signs_same);
        builder.display("diff_term", diff_term);
        let final_mag = builder.mul(same_signs_result.0, signs_same);
        builder.display("final_mag", final_mag);
        let final_mag = builder.add(final_mag, diff_term);
        builder.display("final_mag", final_mag);
        let sign_term = builder.mul(diff_signs_result.1, one_minus_signs_same);
        builder.display("sign_term", sign_term);
        let final_sign = builder.mul(same_signs_result.1, signs_same);
        builder.display("final_sign", final_sign);
        let final_sign = builder.add(final_sign, sign_term);
        builder.display("final_sign", final_sign);
        (final_mag, final_sign)
    };

    // Add imaginary parts (same logic as real parts)
    let imag = {
        let signs_same = builder.unconstrained_eq(a.imag_sign, b.imag_sign);
        builder.display("signs_same_imag", signs_same);
        // Same signs path: just add and keep sign
        let sum = builder.add(a.imag, b.imag);
        builder.display("sum_imag", sum);
        let same_signs_result = (sum, a.imag_sign);

        // Different signs path: subtract and check if result > 2^31
        let max_val = builder.constant(C::CircuitField::from(1u32 << 31));
        let diff = builder.sub(a.imag, b.imag);
        builder.display("diff_imag", diff);
        let is_large = builder.unconstrained_lesser(max_val, diff);
        builder.display("is_large_imag", is_large);
        let should_negate = builder.unconstrained_eq(is_large, one);
        builder.display("should_negate_imag", should_negate);
        let one_minus_should = builder.unconstrained_not_eq(should_negate, one);
        builder.display("one_minus_should_imag", one_minus_should);
        // Compute magnitude
        let neg_result = builder.neg(diff);
        builder.display("neg_result_imag", neg_result);
        let pos_term = builder.mul(diff, one_minus_should);
        builder.display("pos_term_imag", pos_term);
        let neg_term = builder.mul(neg_result, should_negate);
        builder.display("neg_term_imag", neg_term);
        let result = builder.add(pos_term, neg_term);
        builder.display("result_imag", result);
        // Sign should be b's sign when |a| < |b| and a's sign when |a| > |b|
        let sign = builder.unconstrained_eq(is_large, b.imag_sign);
        builder.display("sign_imag", sign);
        let diff_signs_result = (result, sign);
        // Select between same signs and different signs paths
        let one_minus_signs_same = builder.sub(one, signs_same);
        builder.display("one_minus_signs_same_imag", one_minus_signs_same);
        let diff_term = builder.mul(diff_signs_result.0, one_minus_signs_same);
        builder.display("diff_term_imag", diff_term);
        let final_mag = builder.mul(same_signs_result.0, signs_same);
        builder.display("final_mag_imag", final_mag);
        let final_mag = builder.add(final_mag, diff_term);
        builder.display("final_mag_imag", final_mag);
        let sign_term = builder.mul(diff_signs_result.1, one_minus_signs_same);
        builder.display("sign_term_imag", sign_term);
        let final_sign = builder.mul(same_signs_result.1, signs_same);
        builder.display("final_sign_imag", final_sign);
        let final_sign = builder.add(final_sign, sign_term);
        builder.display("final_sign_imag", final_sign);
        (final_mag, final_sign)
    };

    ComplexVar::new(real.0, imag.0, real.1, imag.1)
}

fn complex_sub<C: Config, Builder: RootAPI<C>>(
    builder: &mut Builder,
    a: &ComplexVar,
    b: &ComplexVar,
) -> ComplexVar {
    let one = builder.constant(C::CircuitField::from(1u32));
    let zero = builder.constant(C::CircuitField::ZERO);

    // Helper closure for subtracting individual components
    let mut sub_components =
        |a_mag: Variable, a_sign: Variable, b_mag: Variable, b_sign: Variable| {
            // Check if signs are same
            let signs_same = builder.unconstrained_eq(a_sign, b_sign);

            // For same signs:
            // Compare magnitudes
            let a_geq_b = builder.unconstrained_greater_eq(a_mag, b_mag);

            // If a >= b: result = a - b, keep a's sign
            let same_signs_diff = builder.sub(a_mag, b_mag);
            let same_signs_result = (same_signs_diff, a_sign);

            // If a < b: result = b - a, flip a's sign
            let flipped_diff = builder.sub(b_mag, a_mag);
            let flipped_result = (flipped_diff, builder.unconstrained_not_eq(one, a_sign));

            // Select between a >= b and a < b paths
            let same_signs_term = builder.mul(same_signs_result.0, a_geq_b);
            let one_minus_a_geq_b = builder.sub(one, a_geq_b);
            let flipped_term = builder.mul(flipped_result.0, one_minus_a_geq_b);
            let final_mag_same = builder.add(same_signs_term, flipped_term);

            let sign_term_same = builder.mul(same_signs_result.1, a_geq_b);
            let one_minus_a_geq_b = builder.sub(one, a_geq_b);
            let sign_flipped_term = builder.mul(flipped_result.1, one_minus_a_geq_b);
            let final_sign_same = builder.add(sign_term_same, sign_flipped_term);

            // For different signs:
            // Just add magnitudes and keep a's sign
            let diff_signs_result = (builder.add(a_mag, b_mag), a_sign);

            // Select between same signs and different signs paths
            let one_minus_signs_same = builder.sub(one, signs_same);
            let diff_term = builder.mul(diff_signs_result.0, one_minus_signs_same);
            let final_mag = builder.mul(final_mag_same, signs_same);
            let final_mag = builder.add(final_mag, diff_term);

            let sign_term = builder.mul(diff_signs_result.1, one_minus_signs_same);
            let final_sign = builder.mul(final_sign_same, signs_same);
            let final_sign = builder.add(final_sign, sign_term);

            (final_mag, final_sign)
        };

    // Subtract real and imaginary parts separately
    let real = sub_components(a.real, a.real_sign, b.real, b.real_sign);
    let imag = sub_components(a.imag, a.imag_sign, b.imag, b.imag_sign);

    ComplexVar::new(real.0, imag.0, real.1, imag.1)
}

fn complex_mul<C: Config, Builder: RootAPI<C>>(
    builder: &mut Builder,
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
        // Create ComplexVar for ac term
        builder.display("a.real_sign", a.real_sign);
        builder.display("b.real_sign", b.real_sign);
        builder.display("a.imag_sign", a.imag_sign);
        builder.display("b.imag_sign", b.imag_sign);
        builder.display("ac_scaled", ac_scaled);
        builder.display("bd_scaled", bd_scaled);
        let ac_sign = builder.unconstrained_not_eq(a.real_sign, b.real_sign);
        builder.display("ac_sign", ac_sign);
        let ac_term = ComplexVar {
            real: ac_scaled,
            imag: builder.constant(C::CircuitField::ZERO),
            real_sign: ac_sign,
            imag_sign: builder.constant(C::CircuitField::ZERO),
        };

        // + + -> -
        // + - -> +
        // - + -> +
        // - - -> -
        // Create ComplexVar for -bd term (note the negation via sign flip)
        let zero = builder.constant(C::CircuitField::ZERO);
        let mul_bd_sign = builder.unconstrained_not_eq(a.imag_sign, b.imag_sign); // flip sign because of addition
        builder.display("bd_xor_sign", mul_bd_sign);
        let bd_sign = builder.unconstrained_not_eq(mul_bd_sign, one_const);
        builder.display("bd_sign", bd_sign);

        let bd_term = ComplexVar {
            real: bd_scaled,
            imag: zero,
            real_sign: bd_sign, // Negate bd by flipping sign
            imag_sign: zero,
        };

        // Use complex_add to compute ac + (-bd)
        let result = complex_add(builder, &ac_term, &bd_term);
        builder.display("result.real", result.real);
        builder.display("result.real_sign", result.real_sign);
        (result.real, result.real_sign)
    };

    // Imaginary part: ad + bc
    let imag = {
        builder.display("a.real_sign", a.real_sign);
        builder.display("b.imag_sign", b.imag_sign);
        builder.display("a.imag_sign", a.imag_sign);
        builder.display("b.real_sign", b.real_sign);
        builder.display("ad_scaled", ad_scaled);
        builder.display("bc_scaled", bc_scaled);

        // Create ComplexVar for ad term
        let ad_sign = builder.unconstrained_not_eq(a.real_sign, b.imag_sign);
        builder.display("ad_sign", ad_sign);
        let ad_term = ComplexVar {
            real: ad_scaled,
            imag: builder.constant(C::CircuitField::ZERO),
            real_sign: ad_sign,
            imag_sign: builder.constant(C::CircuitField::ZERO),
        };

        // Create ComplexVar for bc term
        let bc_sign = builder.unconstrained_not_eq(a.imag_sign, b.real_sign);
        builder.display("bc_sign", bc_sign);
        let bc_term = ComplexVar {
            real: bc_scaled,
            imag: builder.constant(C::CircuitField::ZERO),
            real_sign: bc_sign,
            imag_sign: builder.constant(C::CircuitField::ZERO),
        };

        // Use complex_add to compute ad + bc
        let result = complex_add(builder, &ad_term, &bc_term);
        builder.display("result.real", result.real);
        builder.display("result.real_sign", result.real_sign);
        (result.real, result.real_sign)
    };

    ComplexVar::new(real.0, imag.0, real.1, imag.1)
}

/// Helper function to convert bool signs to Variable signs
fn convert_bool_to_signs<C: Config, Builder: RootAPI<C>>(
    builder: &mut Builder,
    signs: &[bool],
) -> Vec<Variable> {
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
pub fn verify_fft<C: Config, Builder: RootAPI<C>>(
    builder: &mut Builder,
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
fn verify_fft_internal<C: Config, Builder: RootAPI<C>>(
    builder: &mut Builder,
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
    // assert_eq!(
    //     input.len(),
    //     size * 2,
    //     "Input tensor size must match shape * 2 (complex numbers)"
    // );
    // assert_eq!(
    //     output.len(),
    //     size * 2,
    //     "Output tensor size must match shape * 2 (complex numbers)"
    // );
    // assert_eq!(
    //     input_signs.len(),
    //     size * 2,
    //     "Input signs length must match shape * 2"
    // );
    // assert_eq!(
    //     output_signs.len(),
    //     size * 2,
    //     "Output signs length must match shape * 2"
    // );
    // assert!(size.is_power_of_two(), "Size must be a power of 2");

    // Constants
    let one = builder.constant(C::CircuitField::from(1u32 << 16)); // 2^16 for fixed-point
    let half = builder.constant(C::CircuitField::from(1u32 << 15)); // 0.5 * 2^16

    // Base case: size = 1
    if size == 1 {
        let input_c = ComplexVar::from_arrays_convert(builder, input, input_signs, 0);
        let output_c = ComplexVar::from_arrays_convert(builder, output, output_signs, 0);
        // signs
        builder.display("input_c.real", input_c.real);
        builder.display("input_c.imag", input_c.imag);
        builder.display("output_c.real", output_c.real);
        builder.display("output_c.imag", output_c.imag);
        builder.display("input_c.real_sign", input_c.real_sign);
        builder.display("input_c.imag_sign", input_c.imag_sign);
        builder.display("output_c.real_sign", output_c.real_sign);
        builder.display("output_c.imag_sign", output_c.imag_sign);
        // Compare difference with threshold
        let diff = complex_sub(builder, &output_c, &input_c);
        let threshold = builder.constant(C::CircuitField::from(256)); // 0.00390625 * 2^16

        let real_check = builder.unconstrained_lesser(diff.real, threshold);
        let imag_check = builder.unconstrained_lesser(diff.imag, threshold);
        let result = builder.and(real_check, imag_check);
        builder.display("result", result);
        builder.display("diff.real", diff.real);
        builder.display("diff.imag", diff.imag);
        builder.display("threshold", threshold);
        builder.display("real_check", real_check);
        builder.display("imag_check", imag_check);
        return result;
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

    const ONE: u32 = 1 << 16;

    declare_circuit!(TestCircuit {
        input: [Variable],
        output: [Variable],
        roots: [Variable],
    });

    // impl DumpLoadTwoVariables<BN254Fr> for TestCircuit<BN254Fr> {
    //     fn dump_into(&self, vars: &mut Vec<BN254Fr>, public_vars: &mut Vec<BN254Fr>) {
    //         vars.extend_from_slice(&self.input);
    //         public_vars.extend_from_slice(&self.output);
    //     }

    //     fn load_from(&mut self, vars: &mut &[BN254Fr], public_vars: &mut &[BN254Fr]) {
    //         self.input.copy_from_slice(&vars[..2]);
    //         *vars = &vars[2..];
    //         self.output.copy_from_slice(&public_vars[..2]);
    //         *public_vars = &public_vars[2..];
    //     }

    //     fn num_vars(&self) -> (usize, usize) {
    //         (2, 2)
    //     }
    // }

    #[test]
    fn test_verify_fft_size_1() {
        declare_circuit!(TestCircuit {
            input: [Variable],
            output: [Variable],
            roots: [Variable],
        });

        impl<C: Config> GenericDefine<C> for TestCircuit<Variable> {
            fn define<Builder: RootAPI<C>>(&self, builder: &mut Builder) {
                let input_signs = vec![true, true]; // Signs for real,imag parts
                let output_signs = vec![true, true];
                let root_signs = vec![true, true]; // Signs for complex root

                let result = verify_fft(
                    builder,
                    &self.input,
                    &self.output,
                    &input_signs,
                    &output_signs,
                    &self.roots,
                    &root_signs,
                    &[1],
                    0,
                );

                let true_const = builder.constant(C::CircuitField::from(1u32));
                builder.assert_is_equal(result, true_const);
            }
        }

        let circuit = TestCircuit {
            input: vec![Variable::default(); 2],
            output: vec![Variable::default(); 2],
            roots: vec![Variable::default(); 2],
        };

        let compile_result = compile_generic::<BN254Config, TestCircuit<Variable>>(
            &circuit,
            CompileOptions::default(),
        )
        .unwrap();

        // Test with input = 2^16, output = 2^16, root = 2^16
        let assignment = TestCircuit::<BN254> {
            input: vec![
                BN254::from(ONE),  // Real part = 2^16
                BN254::from(0u32), // Imaginary part = 0
            ],
            output: vec![
                BN254::from(ONE),  // Real part = 2^16
                BN254::from(0u32), // Imaginary part = 0
            ],
            roots: vec![
                BN254::from(ONE),  // Real part = 2^16
                BN254::from(0u32), // Imaginary part = 0
            ],
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        debug_eval::<BN254Config, _, _, _>(&circuit, &assignment, EmptyHintCaller);
        assert_eq!(output, vec![true]);
    }

    #[test]
    fn test_complex_add_positive() {
        declare_circuit!(TestCircuit {
            input: [Variable],
            output: [Variable],
        });

        impl<C: Config> GenericDefine<C> for TestCircuit<Variable> {
            fn define<Builder: RootAPI<C>>(&self, builder: &mut Builder) {
                let a = ComplexVar::new(
                    self.input[0],
                    self.input[1],
                    builder.constant(C::CircuitField::from(0u32)),
                    builder.constant(C::CircuitField::from(0u32)),
                ); // 1 + 0i
                let b = ComplexVar::new(
                    self.input[2],
                    self.input[3],
                    builder.constant(C::CircuitField::from(0u32)),
                    builder.constant(C::CircuitField::from(0u32)),
                );
                let result = complex_add(builder, &a, &b);
                let zero = builder.constant(C::CircuitField::ZERO);
                builder.assert_is_equal(result.real, self.output[0]);
                builder.assert_is_equal(result.imag, self.output[1]);
                builder.assert_is_equal(result.real_sign, zero);
                builder.assert_is_equal(result.imag_sign, zero);
            }
        }

        let circuit = TestCircuit {
            input: vec![Variable::default(); 4],
            output: vec![Variable::default(); 2],
        };

        let compile_result = compile_generic::<BN254Config, TestCircuit<Variable>>(
            &circuit,
            CompileOptions::default(),
        )
        .unwrap();

        let assignment = TestCircuit::<BN254Fr> {
            input: vec![
                BN254Fr::from(ONE),     // Real part = 2^16
                BN254Fr::from(0u32),    // Imaginary part = 0
                BN254Fr::from(2 * ONE), // Real part = 2^16
                BN254Fr::from(ONE),     // Imaginary part = 0
            ],
            output: vec![
                BN254Fr::from(3 * ONE), // Real part = 2^16
                BN254Fr::from(ONE),     // Imaginary part = 0
            ],
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        compile_result.layered_circuit.run(&witness);
        debug_eval::<BN254Config, _, _, _>(&circuit, &assignment, EmptyHintCaller);
    }

    #[test]
    fn test_complex_add_positive_negative_a_less_than_b() {
        declare_circuit!(TestCircuit {
            input: [Variable],
            output: [Variable],
        });

        impl<C: Config> GenericDefine<C> for TestCircuit<Variable> {
            fn define<Builder: RootAPI<C>>(&self, builder: &mut Builder) {
                let a = ComplexVar::new(
                    self.input[0],
                    self.input[1],
                    builder.constant(C::CircuitField::from(0u32)),
                    builder.constant(C::CircuitField::from(0u32)),
                ); // 1 + 0i
                let b = ComplexVar::new(
                    self.input[2],
                    self.input[3],
                    builder.constant(C::CircuitField::from(1u32)),
                    builder.constant(C::CircuitField::from(1u32)),
                ); // -2 + 0i
                let result = complex_add(builder, &a, &b);
                let one = builder.constant(C::CircuitField::from(1u32));

                builder.display("result.real", result.real);
                builder.display("result.imag", result.imag);
                builder.display("result.real_sign", result.real_sign);
                builder.display("result.imag_sign", result.imag_sign);
                builder.display("expected_real", self.output[0]);
                builder.display("expected_imag", self.output[1]);
                builder.display("expected_real_sign", one);
                builder.display("expected_imag_sign", one);
                builder.assert_is_equal(result.real, self.output[0]);
                builder.assert_is_equal(result.imag, self.output[1]);
                builder.assert_is_equal(result.real_sign, one);
                builder.assert_is_equal(result.imag_sign, one);
            }
        }

        let circuit = TestCircuit {
            input: vec![Variable::default(); 4],
            output: vec![Variable::default(); 2],
        };

        let compile_result = compile_generic::<BN254Config, TestCircuit<Variable>>(
            &circuit,
            CompileOptions::default(),
        )
        .unwrap();

        let assignment = TestCircuit::<BN254Fr> {
            input: vec![
                BN254Fr::from(ONE),     // Real part = 1
                BN254Fr::from(ONE),     // Imaginary part = 1
                BN254Fr::from(2 * ONE), // Real part = -2
                BN254Fr::from(2 * ONE), // Imaginary part = 2
            ],
            output: vec![
                BN254Fr::from(ONE), // Real part = 1 (1-(-2))
                BN254Fr::from(ONE), // Imaginary part = 1 (1-(-2))
            ],
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        compile_result.layered_circuit.run(&witness);
        debug_eval::<BN254Config, _, _, _>(&circuit, &assignment, EmptyHintCaller);
    }
    #[test]
    fn test_complex_add_negative_positive_a_less_than_b() {
        declare_circuit!(TestCircuit {
            input: [Variable],
            output: [Variable],
        });

        impl<C: Config> GenericDefine<C> for TestCircuit<Variable> {
            fn define<Builder: RootAPI<C>>(&self, builder: &mut Builder) {
                let a = ComplexVar::new(
                    self.input[0],
                    self.input[1],
                    builder.constant(C::CircuitField::from(1u32)),
                    builder.constant(C::CircuitField::from(1u32)),
                ); // 1 + 0i
                let b = ComplexVar::new(
                    self.input[2],
                    self.input[3],
                    builder.constant(C::CircuitField::from(0u32)),
                    builder.constant(C::CircuitField::from(0u32)),
                ); // -2 + 0i
                let result = complex_add(builder, &a, &b);
                let one = builder.constant(C::CircuitField::from(1u32));
                let zero = builder.constant(C::CircuitField::ZERO);
                builder.display("result.real", result.real);
                builder.display("result.imag", result.imag);
                builder.display("result.real_sign", result.real_sign);
                builder.display("result.imag_sign", result.imag_sign);
                builder.display("expected_real", self.output[0]);
                builder.display("expected_imag", self.output[1]);
                builder.display("expected_real_sign", zero);
                builder.display("expected_imag_sign", zero);
                builder.assert_is_equal(result.real, self.output[0]);
                builder.assert_is_equal(result.imag, self.output[1]);
                builder.assert_is_equal(result.real_sign, zero);
                builder.assert_is_equal(result.imag_sign, zero);
            }
        }

        let circuit = TestCircuit {
            input: vec![Variable::default(); 4],
            output: vec![Variable::default(); 2],
        };

        let compile_result = compile_generic::<BN254Config, TestCircuit<Variable>>(
            &circuit,
            CompileOptions::default(),
        )
        .unwrap();

        let assignment = TestCircuit::<BN254Fr> {
            input: vec![
                BN254Fr::from(2 * ONE), // Real part = -2
                BN254Fr::from(2 * ONE), // Imaginary part = 1
                BN254Fr::from(6 * ONE), // Real part = 6
                BN254Fr::from(6 * ONE), // Imaginary part = 2
            ],
            output: vec![
                BN254Fr::from(4 * ONE), // Real part = 4 (-2+6)
                BN254Fr::from(4 * ONE), // Imaginary part = 2 (1+1)
            ],
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        compile_result.layered_circuit.run(&witness);
        debug_eval::<BN254Config, _, _, _>(&circuit, &assignment, EmptyHintCaller);
    }
    #[test]
    fn test_complex_add_positive_negative_a_greater_than_b() {
        declare_circuit!(TestCircuit {
            input: [Variable],
            output: [Variable],
        });

        impl<C: Config> GenericDefine<C> for TestCircuit<Variable> {
            fn define<Builder: RootAPI<C>>(&self, builder: &mut Builder) {
                let a = ComplexVar::new(
                    self.input[0],
                    self.input[1],
                    builder.constant(C::CircuitField::from(0u32)),
                    builder.constant(C::CircuitField::from(0u32)),
                ); // 1 + 0i
                let b = ComplexVar::new(
                    self.input[2],
                    self.input[3],
                    builder.constant(C::CircuitField::from(1u32)),
                    builder.constant(C::CircuitField::from(1u32)),
                ); // -2 + 0i
                let result = complex_add(builder, &a, &b);
                let zero = builder.constant(C::CircuitField::ZERO);
                let one = builder.constant(C::CircuitField::from(1u32));

                builder.display("result.real", result.real);
                builder.display("result.imag", result.imag);
                builder.display("result.real_sign", result.real_sign);
                builder.display("result.imag_sign", result.imag_sign);
                builder.display("expected_real", self.output[0]);
                builder.display("expected_imag", self.output[1]);
                builder.display("expected_real_sign", one);
                builder.display("expected_imag_sign", zero);
                builder.assert_is_equal(result.real, self.output[0]);
                builder.assert_is_equal(result.imag, self.output[1]);
                builder.assert_is_equal(result.real_sign, zero);
                builder.assert_is_equal(result.imag_sign, zero);
            }
        }

        let circuit = TestCircuit {
            input: vec![Variable::default(); 4],
            output: vec![Variable::default(); 2],
        };

        let compile_result = compile_generic::<BN254Config, TestCircuit<Variable>>(
            &circuit,
            CompileOptions::default(),
        )
        .unwrap();

        let assignment = TestCircuit::<BN254Fr> {
            input: vec![
                BN254Fr::from(2 * ONE), // Real part = 2
                BN254Fr::from(3 * ONE), // Imaginary part = 3
                BN254Fr::from(ONE),     // Real part = -1
                BN254Fr::from(2 * ONE), // Imaginary part = 2
            ],
            output: vec![
                BN254Fr::from(ONE), // Real part = 1 (2-(-1))
                BN254Fr::from(ONE), // Imaginary part = 1 (3-(-2))
            ],
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        compile_result.layered_circuit.run(&witness);
        debug_eval::<BN254Config, _, _, _>(&circuit, &assignment, EmptyHintCaller);
    }

    #[test]
    fn test_complex_add_negative() {
        declare_circuit!(TestCircuit {
            input: [Variable],
            output: [Variable],
        });

        impl<C: Config> GenericDefine<C> for TestCircuit<Variable> {
            fn define<Builder: RootAPI<C>>(&self, builder: &mut Builder) {
                let a = ComplexVar::new(
                    self.input[0],
                    self.input[1],
                    builder.constant(C::CircuitField::from(1u32)),
                    builder.constant(C::CircuitField::from(0u32)),
                ); // -1 + 0i
                let b = ComplexVar::new(
                    self.input[2],
                    self.input[3],
                    builder.constant(C::CircuitField::from(1u32)),
                    builder.constant(C::CircuitField::from(0u32)),
                ); // -1 + 0i
                let result = complex_add(builder, &a, &b);
                let zero = builder.constant(C::CircuitField::ZERO);
                let one = builder.constant(C::CircuitField::from(1u32));
                builder.assert_is_equal(result.real, self.output[0]);
                builder.assert_is_equal(result.imag, self.output[1]);
                builder.assert_is_equal(result.real_sign, one);
                builder.assert_is_equal(result.imag_sign, zero);
            }
        }

        let circuit = TestCircuit {
            input: vec![Variable::default(); 4],
            output: vec![Variable::default(); 2],
        };

        let compile_result = compile_generic::<BN254Config, TestCircuit<Variable>>(
            &circuit,
            CompileOptions::default(),
        )
        .unwrap();

        let assignment = TestCircuit::<BN254Fr> {
            input: vec![
                BN254Fr::from(ONE),  // Real part = 2^16
                BN254Fr::from(0u32), // Imaginary part = 0
                BN254Fr::from(ONE),  // Real part = 2^16
                BN254Fr::from(0u32), // Imaginary part = 0
            ],
            output: vec![
                BN254Fr::from(2 * ONE), // Real part = 2^16
                BN254Fr::from(0u32),    // Imaginary part = 0
            ],
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        compile_result.layered_circuit.run(&witness);
        debug_eval::<BN254Config, _, _, _>(&circuit, &assignment, EmptyHintCaller);
    }

    #[test]
    fn test_complex_sub_positive() {
        declare_circuit!(TestCircuit {
            input: [Variable],
            output: [Variable],
        });

        impl<C: Config> GenericDefine<C> for TestCircuit<Variable> {
            fn define<Builder: RootAPI<C>>(&self, builder: &mut Builder) {
                let a = ComplexVar::new(
                    self.input[0],
                    self.input[1],
                    builder.constant(C::CircuitField::from(0u32)),
                    builder.constant(C::CircuitField::from(0u32)),
                ); // 2 + 3i
                let b = ComplexVar::new(
                    self.input[2],
                    self.input[3],
                    builder.constant(C::CircuitField::from(1u32)), // inverse sign because we are using the add ops instead of sub ops
                    builder.constant(C::CircuitField::from(1u32)),
                ); // 1 + 2i
                let result = complex_add(builder, &a, &b);
                let zero = builder.constant(C::CircuitField::ZERO);

                builder.display("result.real", result.real);
                builder.display("result.imag", result.imag);
                builder.display("result.real_sign", result.real_sign);
                builder.display("result.imag_sign", result.imag_sign);
                builder.display("expected_real", self.output[0]);
                builder.display("expected_imag", self.output[1]);
                builder.assert_is_equal(result.real, self.output[0]);
                builder.assert_is_equal(result.imag, self.output[1]);
                builder.assert_is_equal(result.real_sign, zero);
                builder.assert_is_equal(result.imag_sign, zero);
            }
        }

        let circuit = TestCircuit {
            input: vec![Variable::default(); 4],
            output: vec![Variable::default(); 2],
        };

        let compile_result = compile_generic::<BN254Config, TestCircuit<Variable>>(
            &circuit,
            CompileOptions::default(),
        )
        .unwrap();

        let assignment = TestCircuit::<BN254Fr> {
            input: vec![
                BN254Fr::from(2 * ONE), // Real part = 2
                BN254Fr::from(3 * ONE), // Imaginary part = 3
                BN254Fr::from(ONE),     // Real part = 1
                BN254Fr::from(2 * ONE), // Imaginary part = 2
            ],
            output: vec![
                BN254Fr::from(ONE), // Real part = 1 (2-1)
                BN254Fr::from(ONE), // Imaginary part = 1 (3-2)
            ],
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        compile_result.layered_circuit.run(&witness);
        debug_eval::<BN254Config, _, _, _>(&circuit, &assignment, EmptyHintCaller);
    }

    #[test]
    fn test_complex_sub_positive_negative_a_less_than_b() {
        declare_circuit!(TestCircuit {
            input: [Variable],
            output: [Variable],
        });

        impl<C: Config> GenericDefine<C> for TestCircuit<Variable> {
            fn define<Builder: RootAPI<C>>(&self, builder: &mut Builder) {
                let a = ComplexVar::new(
                    self.input[0],
                    self.input[1],
                    builder.constant(C::CircuitField::from(0u32)),
                    builder.constant(C::CircuitField::from(0u32)),
                ); // 1 + i
                let b = ComplexVar::new(
                    self.input[2],
                    self.input[3],
                    builder.constant(C::CircuitField::from(0u32)), // inverse sign because we are using the add ops instead of sub ops
                    builder.constant(C::CircuitField::from(0u32)),
                ); // -2 - 2i
                let result = complex_add(builder, &a, &b);
                let zero = builder.constant(C::CircuitField::ZERO);

                builder.display("result.real", result.real);
                builder.display("result.imag", result.imag);
                builder.display("result.real_sign", result.real_sign);
                builder.display("result.imag_sign", result.imag_sign);
                builder.display("expected_real", self.output[0]);
                builder.display("expected_imag", self.output[1]);
                builder.assert_is_equal(result.real, self.output[0]);
                builder.assert_is_equal(result.imag, self.output[1]);
                builder.assert_is_equal(result.real_sign, zero);
                builder.assert_is_equal(result.imag_sign, zero);
            }
        }

        let circuit = TestCircuit {
            input: vec![Variable::default(); 4],
            output: vec![Variable::default(); 2],
        };

        let compile_result = compile_generic::<BN254Config, TestCircuit<Variable>>(
            &circuit,
            CompileOptions::default(),
        )
        .unwrap();

        let assignment = TestCircuit::<BN254Fr> {
            input: vec![
                BN254Fr::from(ONE),     // Real part = 1
                BN254Fr::from(ONE),     // Imaginary part = 1
                BN254Fr::from(2 * ONE), // Real part = -2
                BN254Fr::from(2 * ONE), // Imaginary part = -2
            ],
            output: vec![
                BN254Fr::from(3 * ONE), // Real part = 3 (1-(-2))
                BN254Fr::from(3 * ONE), // Imaginary part = 3 (1-(-2))
            ],
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        compile_result.layered_circuit.run(&witness);
        debug_eval::<BN254Config, _, _, _>(&circuit, &assignment, EmptyHintCaller);
    }

    #[test]
    fn test_complex_sub_positive_negative_a_greater_than_b() {
        declare_circuit!(TestCircuit {
            input: [Variable],
            output: [Variable],
        });

        impl<C: Config> GenericDefine<C> for TestCircuit<Variable> {
            fn define<Builder: RootAPI<C>>(&self, builder: &mut Builder) {
                let a = ComplexVar::new(
                    self.input[0],
                    self.input[1],
                    builder.constant(C::CircuitField::from(0u32)),
                    builder.constant(C::CircuitField::from(0u32)),
                ); // 2 + 3i
                let b = ComplexVar::new(
                    self.input[2],
                    self.input[3],
                    builder.constant(C::CircuitField::from(0u32)), // inverse sign because we are using the add ops instead of sub ops
                    builder.constant(C::CircuitField::from(0u32)),
                ); // -1 - 2i
                let result = complex_add(builder, &a, &b);
                let zero = builder.constant(C::CircuitField::ZERO);

                builder.display("result.real", result.real);
                builder.display("result.imag", result.imag);
                builder.display("result.real_sign", result.real_sign);
                builder.display("result.imag_sign", result.imag_sign);
                builder.display("expected_real", self.output[0]);
                builder.display("expected_imag", self.output[1]);
                builder.assert_is_equal(result.real, self.output[0]);
                builder.assert_is_equal(result.imag, self.output[1]);
                builder.assert_is_equal(result.real_sign, zero);
                builder.assert_is_equal(result.imag_sign, zero);
            }
        }

        let circuit = TestCircuit {
            input: vec![Variable::default(); 4],
            output: vec![Variable::default(); 2],
        };

        let compile_result = compile_generic::<BN254Config, TestCircuit<Variable>>(
            &circuit,
            CompileOptions::default(),
        )
        .unwrap();

        let assignment = TestCircuit::<BN254Fr> {
            input: vec![
                BN254Fr::from(2 * ONE), // Real part = 2
                BN254Fr::from(3 * ONE), // Imaginary part = 3
                BN254Fr::from(ONE),     // Real part = -1
                BN254Fr::from(2 * ONE), // Imaginary part = -2
            ],
            output: vec![
                BN254Fr::from(3 * ONE), // Real part = 3 (2-(-1))
                BN254Fr::from(5 * ONE), // Imaginary part = 5 (3-(-2))
            ],
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        compile_result.layered_circuit.run(&witness);
        debug_eval::<BN254Config, _, _, _>(&circuit, &assignment, EmptyHintCaller);
    }

    #[test]
    fn test_complex_mul_positive() {
        declare_circuit!(TestCircuit {
            input: [Variable],
            output: [Variable],
        });

        impl<C: Config> GenericDefine<C> for TestCircuit<Variable> {
            fn define<Builder: RootAPI<C>>(&self, builder: &mut Builder) {
                let a = ComplexVar::new(
                    self.input[0],
                    self.input[1],
                    builder.constant(C::CircuitField::from(0u32)),
                    builder.constant(C::CircuitField::from(0u32)),
                ); // 2 + 3i
                let b = ComplexVar::new(
                    self.input[2],
                    self.input[3],
                    builder.constant(C::CircuitField::from(0u32)),
                    builder.constant(C::CircuitField::from(0u32)),
                ); // 1 + 2i
                let one_scaled = builder.constant(C::CircuitField::from(ONE));
                let result = complex_mul(builder, &a, &b, one_scaled);
                let zero = builder.constant(C::CircuitField::ZERO);
                let one = builder.constant(C::CircuitField::from(1u32));
                builder.display("result.real", result.real);
                builder.display("result.imag", result.imag);
                builder.display("result.real_sign", result.real_sign);
                builder.display("result.imag_sign", result.imag_sign);
                builder.display("expected_real", self.output[0]);
                builder.display("expected_imag", self.output[1]);
                builder.assert_is_equal(result.real, self.output[0]);
                builder.assert_is_equal(result.imag, self.output[1]);
                builder.assert_is_equal(result.real_sign, one);
                builder.assert_is_equal(result.imag_sign, zero);
            }
        }

        let circuit = TestCircuit {
            input: vec![Variable::default(); 4],
            output: vec![Variable::default(); 2],
        };

        let compile_result = compile_generic::<BN254Config, TestCircuit<Variable>>(
            &circuit,
            CompileOptions::default(),
        )
        .unwrap();

        // (2 + 3i)(1 + 2i) = (2 - 6) + (3 + 2)i = -4 + 5i
        let assignment = TestCircuit::<BN254Fr> {
            input: vec![
                BN254Fr::from(2 * ONE), // Real part = 2
                BN254Fr::from(3 * ONE), // Imaginary part = 3
                BN254Fr::from(ONE),     // Real part = 1
                BN254Fr::from(2 * ONE), // Imaginary part = 2
            ],
            output: vec![
                BN254Fr::from(4 * ONE), // Real part = -4 (2*1 - 3*2)
                BN254Fr::from(7 * ONE), // Imaginary part = 7 (2*2 + 3*1)
            ],
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        compile_result.layered_circuit.run(&witness);
        debug_eval::<BN254Config, _, _, _>(&circuit, &assignment, EmptyHintCaller);
    }

    #[test]
    fn test_complex_mul_mixed_signs() {
        declare_circuit!(TestCircuit {
            input: [Variable],
            output: [Variable],
        });

        impl<C: Config> GenericDefine<C> for TestCircuit<Variable> {
            fn define<Builder: RootAPI<C>>(&self, builder: &mut Builder) {
                let a = ComplexVar::new(
                    self.input[0],
                    self.input[1],
                    builder.constant(C::CircuitField::from(0u32)),
                    builder.constant(C::CircuitField::from(1u32)),
                ); // 2 - 3i
                let b = ComplexVar::new(
                    self.input[2],
                    self.input[3],
                    builder.constant(C::CircuitField::from(1u32)),
                    builder.constant(C::CircuitField::from(0u32)),
                ); // -1 + 2i
                let one = builder.constant(C::CircuitField::from(ONE));
                let result = complex_mul(builder, &a, &b, one);
                let zero = builder.constant(C::CircuitField::ZERO);
                let one_const = builder.constant(C::CircuitField::from(1u32));

                builder.display("result.real", result.real);
                builder.display("result.imag", result.imag);
                builder.display("result.real_sign", result.real_sign);
                builder.display("result.imag_sign", result.imag_sign);
                builder.display("expected_real", self.output[0]);
                builder.display("expected_imag", self.output[1]);
                builder.assert_is_equal(result.real, self.output[0]);
                builder.assert_is_equal(result.imag, self.output[1]);
                builder.assert_is_equal(result.real_sign, zero);
                builder.assert_is_equal(result.imag_sign, zero);
            }
        }

        let circuit = TestCircuit {
            input: vec![Variable::default(); 4],
            output: vec![Variable::default(); 2],
        };

        let compile_result = compile_generic::<BN254Config, TestCircuit<Variable>>(
            &circuit,
            CompileOptions::default(),
        )
        .unwrap();

        // (2 - 3i)(-1 + 2i) = (-2 + 3i)(1 - 2i) = (-2 + 6) + (3 + 4)i = 4 + 7i
        let assignment = TestCircuit::<BN254Fr> {
            input: vec![
                BN254Fr::from(2 * ONE), // Real part = 2
                BN254Fr::from(3 * ONE), // Imaginary part = -3
                BN254Fr::from(ONE),     // Real part = -1
                BN254Fr::from(2 * ONE), // Imaginary part = 2
            ],
            output: vec![
                BN254Fr::from(4 * ONE), // Real part = 4 (2*(-1) - (-3)*2)
                BN254Fr::from(7 * ONE), // Imaginary part = 7 (2*2 + (-3)*(-1))
            ],
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        compile_result.layered_circuit.run(&witness);
        debug_eval::<BN254Config, _, _, _>(&circuit, &assignment, EmptyHintCaller);
    }

    #[test]
    fn test_complex_mul_pure_imaginary() {
        declare_circuit!(TestCircuit {
            input: [Variable],
            output: [Variable],
        });

        impl<C: Config> GenericDefine<C> for TestCircuit<Variable> {
            fn define<Builder: RootAPI<C>>(&self, builder: &mut Builder) {
                let a = ComplexVar::new(
                    self.input[0],
                    self.input[1],
                    builder.constant(C::CircuitField::from(0u32)),
                    builder.constant(C::CircuitField::from(0u32)),
                ); // 0 + 2i
                let b = ComplexVar::new(
                    self.input[2],
                    self.input[3],
                    builder.constant(C::CircuitField::from(0u32)),
                    builder.constant(C::CircuitField::from(0u32)),
                ); // 0 + 3i
                let one = builder.constant(C::CircuitField::from(ONE));
                let result = complex_mul(builder, &a, &b, one);
                let zero = builder.constant(C::CircuitField::ZERO);
                let one_const = builder.constant(C::CircuitField::from(1u32));

                builder.display("result.real", result.real);
                builder.display("result.imag", result.imag);
                builder.display("result.real_sign", result.real_sign);
                builder.display("result.imag_sign", result.imag_sign);
                builder.display("expected_real", self.output[0]);
                builder.display("expected_imag", self.output[1]);
                builder.assert_is_equal(result.real, self.output[0]);
                builder.assert_is_equal(result.imag, self.output[1]);
                builder.assert_is_equal(result.real_sign, one_const);
                builder.assert_is_equal(result.imag_sign, zero);
            }
        }

        let circuit = TestCircuit {
            input: vec![Variable::default(); 4],
            output: vec![Variable::default(); 2],
        };

        let compile_result = compile_generic::<BN254Config, TestCircuit<Variable>>(
            &circuit,
            CompileOptions::default(),
        )
        .unwrap();

        // (2i)(3i) = -6
        let assignment = TestCircuit::<BN254Fr> {
            input: vec![
                BN254Fr::from(0u32),    // Real part = 0
                BN254Fr::from(2 * ONE), // Imaginary part = 2
                BN254Fr::from(0u32),    // Real part = 0
                BN254Fr::from(3 * ONE), // Imaginary part = 3
            ],
            output: vec![
                BN254Fr::from(6 * ONE), // Real part = -6 (0*0 - 2*3)
                BN254Fr::from(0u32),    // Imaginary part = 0 (0*3 + 2*0)
            ],
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        compile_result.layered_circuit.run(&witness);
        debug_eval::<BN254Config, _, _, _>(&circuit, &assignment, EmptyHintCaller);
    }

    #[test]
    fn test_complex_mul_zero() {
        declare_circuit!(TestCircuit {
            input: [Variable],
            output: [Variable],
        });

        impl<C: Config> GenericDefine<C> for TestCircuit<Variable> {
            fn define<Builder: RootAPI<C>>(&self, builder: &mut Builder) {
                let a = ComplexVar::new(
                    self.input[0],
                    self.input[1],
                    builder.constant(C::CircuitField::from(0u32)),
                    builder.constant(C::CircuitField::from(0u32)),
                ); // 2 + 3i
                let b = ComplexVar::new(
                    self.input[2],
                    self.input[3],
                    builder.constant(C::CircuitField::from(0u32)),
                    builder.constant(C::CircuitField::from(0u32)),
                ); // 0 + 0i
                let one = builder.constant(C::CircuitField::from(ONE));
                let result = complex_mul(builder, &a, &b, one);
                let zero = builder.constant(C::CircuitField::ZERO);

                builder.display("result.real", result.real);
                builder.display("result.imag", result.imag);
                builder.display("result.real_sign", result.real_sign);
                builder.display("result.imag_sign", result.imag_sign);
                builder.display("expected_real", self.output[0]);
                builder.display("expected_imag", self.output[1]);
                builder.assert_is_equal(result.real, self.output[0]);
                builder.assert_is_equal(result.imag, self.output[1]);
                builder.assert_is_equal(result.real_sign, zero);
                builder.assert_is_equal(result.imag_sign, zero);
            }
        }

        let circuit = TestCircuit {
            input: vec![Variable::default(); 4],
            output: vec![Variable::default(); 2],
        };

        let compile_result = compile_generic::<BN254Config, TestCircuit<Variable>>(
            &circuit,
            CompileOptions::default(),
        )
        .unwrap();

        // (2 + 3i)(0 + 0i) = 0
        let assignment = TestCircuit::<BN254Fr> {
            input: vec![
                BN254Fr::from(2 * ONE), // Real part = 2
                BN254Fr::from(3 * ONE), // Imaginary part = 3
                BN254Fr::from(0u32),    // Real part = 0
                BN254Fr::from(0u32),    // Imaginary part = 0
            ],
            output: vec![
                BN254Fr::from(0u32), // Real part = 0
                BN254Fr::from(0u32), // Imaginary part = 0
            ],
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        compile_result.layered_circuit.run(&witness);
        debug_eval::<BN254Config, _, _, _>(&circuit, &assignment, EmptyHintCaller);
    }
}
