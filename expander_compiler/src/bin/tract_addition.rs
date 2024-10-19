use tract_onnx::prelude::*;

fn main() -> TractResult<()> {
    // Load the ONNX model from the file
    let model = tract_onnx::onnx()
        .model_for_path("/Users/lorenzotomaz/projects/client/50b/0k/2_addition.onnx")? // Specify the ONNX model file path
        .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1)))?
        .with_input_fact(1, InferenceFact::dt_shape(f32::datum_type(), tvec!(1)))?
        .into_optimized()?; // Optimize the model

    // Extract nodes from the computational graph
    let nodes = model.nodes();

    println!("Model nodes:");

    // Loop through the nodes and print some information about each node
    for (i, node) in nodes.iter().enumerate() {
        println!("Node {}: {:?}", i, node.op);
        println!("  Name: {}", node.name);
        println!("  Inputs: {:?}", node.inputs);
        println!("  Outputs: {:?}", node.outputs);
    }

    Ok(())
}
