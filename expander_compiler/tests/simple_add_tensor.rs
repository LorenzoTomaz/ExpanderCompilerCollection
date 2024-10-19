use expander_compiler::frontend::*;
use expander_compiler::tensor::Tensor;
declare_circuit!(Circuit {
    sum: PublicVariable,
    x: [Variable; 2],
});

impl Define<BN254Config> for Circuit<Variable> {
    fn define(&self, builder: &mut API<BN254Config>) {
        let tensor0 = Tensor::<Variable>::new(Some(&[self.x[0]]), &[1]).unwrap();
        let tensor1 = Tensor::<Variable>::new(Some(&[self.x[1]]), &[1]).unwrap();
        let res = tensor0
            .add_constraint::<BN254Config>(tensor1, builder)
            .unwrap();
        let tensor2 = Tensor::<Variable>::new(Some(&[builder.constant(123)]), &[1]).unwrap();
        let res1 = res.add_constraint::<BN254Config>(tensor2, builder).unwrap();
        let sum = res1[0];
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
