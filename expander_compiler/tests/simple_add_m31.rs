use expander_compiler::frontend::*;

declare_circuit!(Circuit {
    sum: PublicVariable,
    x: [Variable; 2],
});

impl Define<BN254Config> for Circuit<Variable> {
    fn define(&self, builder: &mut API<BN254Config>) {
        let sum = builder.add(self.x[0], self.x[1]);
        let y = builder.constant(123);
        let sum = builder.add(sum, y);
        builder.assert_is_equal(sum, self.sum);
    }
}

#[test]
fn test_circuit_eval_simple() {
    let compile_result = compile(&Circuit::default()).unwrap();
    let assignment = Circuit::<BN254> {
        sum: BN254::from(126u64),
        x: [BN254::from(1u64), BN254::from(2u64)],
    };
    let witness = compile_result
        .witness_solver
        .solve_witness(&assignment)
        .unwrap();
    let output = compile_result.layered_circuit.run(&witness);
    assert_eq!(output, vec![true]);

    let assignment = Circuit::<BN254> {
        sum: BN254::from(127u64),
        x: [BN254::from(1u64), BN254::from(2u64)],
    };
    let witness = compile_result
        .witness_solver
        .solve_witness(&assignment)
        .unwrap();
    let output = compile_result.layered_circuit.run(&witness);
    assert_eq!(output, vec![false]);
}
