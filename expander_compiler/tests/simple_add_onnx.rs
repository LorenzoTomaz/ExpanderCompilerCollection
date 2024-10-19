use expander_compiler::frontend::*;
use expander_compiler::tensor::Tensor;
use smallvec::{smallvec, SmallVec};
use std::collections::HashMap;
use tract_core::model::{Outlet, OutletId, TypedFact};
use tract_onnx::prelude::*;
fn parse_onnx_model(file_path: &str) -> TractResult<TypedModel> {
    let model = tract_onnx::onnx()
        .model_for_path(file_path)?
        .into_optimized()?;
    Ok(model)
}

fn extract_onnx_operations(
    model: &TypedModel,
) -> Vec<(String, Vec<usize>, SmallVec<[Outlet<TypedFact>; 4]>)> {
    let mut operations = vec![];
    for node in model.nodes().iter() {
        // Extract the operation type, inputs, and outputs
        let op_type = format!("{:?}", node.op.name());

        // Extract input indices (from the inputs of the node)
        let inputs: Vec<usize> = node.inputs.iter().map(|input| input.node).collect();

        // Extract output indices (from the outputs of the node)
        let outputs: SmallVec<[Outlet<TypedFact>; 4]> =
            node.outputs.iter().map(|output| output.clone()).collect();

        // Push the extracted data into the operations vector
        operations.push((op_type, inputs, outputs));
    }
    operations
}

declare_circuit!(Circuit {
    sum: [PublicVariable; 1],
    x: [Variable; 2],
});

impl Define<BN254Config> for Circuit<Variable> {
    fn define(&self, builder: &mut API<BN254Config>) {
        let mut tensors: HashMap<usize, Tensor<Variable>> = HashMap::new();

        // We have exactly two inputs, so let's create two tensors for them
        let tensor_input_0 = Tensor::<Variable>::new(Some(&[self.x[0]]), &[1]).unwrap();
        let tensor_input_1 = Tensor::<Variable>::new(Some(&[self.x[1]]), &[1]).unwrap();
        tensors.insert(0, tensor_input_0);
        tensors.insert(1, tensor_input_1);

        // Dynamically iterate over ONNX operations and map them to circuit operations
        let model = parse_onnx_model("./tests/addition.onnx").unwrap();
        let operations = extract_onnx_operations(&model);

        for (op_type, inputs, _) in operations {
            match op_type.as_str() {
                "\"AddUnicast\"" => {
                    let lhs = tensors.get(&inputs[0]).unwrap().clone();
                    let rhs = tensors.get(&inputs[1]).unwrap().clone();
                    let sum = lhs.add_constraint(rhs, builder).unwrap();
                    tensors.insert(2, sum);
                }
                "\"Source\"" => {
                    print!("TypedSource");
                }
                // Handle other operations like Mul, Div, etc.
                _ => unimplemented!("Operation {} is not supported", op_type),
            }
        }

        //Assert that the final output (result) is equal to the public variable
        let final_tensor = tensors.get(&2).unwrap().clone();
        builder.assert_is_equal(final_tensor[0], self.sum[0]);
    }
}

#[test]
fn test_circuit_eval_simple() {
    let compile_result = compile(&Circuit::default()).unwrap();
    let assignment = Circuit::<BN254> {
        sum: [BN254::from(3u64)],
        x: [BN254::from(1u64), BN254::from(2u64)],
    };
    let witness = compile_result
        .witness_solver
        .solve_witness(&assignment)
        .unwrap();
    let output = compile_result.layered_circuit.run(&witness);
    assert_eq!(output, vec![true]);

    let assignment = Circuit::<BN254> {
        sum: [BN254::from(127u64)],
        x: [BN254::from(1u64), BN254::from(2u64)],
    };
    let witness = compile_result
        .witness_solver
        .solve_witness(&assignment)
        .unwrap();
    let output = compile_result.layered_circuit.run(&witness);
    assert_eq!(output, vec![false]);
}
