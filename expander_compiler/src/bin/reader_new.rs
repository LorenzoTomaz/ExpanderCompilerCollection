use std::collections::HashMap;
use tract_hir::infer::ShapeFactoid;
use tract_onnx::prelude::*;
type TractResult = (Graph<TypedFact, Box<dyn TypedOp>>, SymbolValues);
// Define a structure for the Node in the graph
#[derive(Debug)]
struct CircuitNode {
    name: String,
    inputs: Vec<(OutletId, String)>, // Store both index and input information
    outputs: Vec<(Outlet<InferenceFact>, String)>, // Store both index and output information
}

// Define the structure of the computational Graph
#[derive(Debug)]
struct CircuitGraph {
    nodes: HashMap<usize, CircuitNode>,
}

impl CircuitGraph {
    fn new() -> Self {
        CircuitGraph {
            nodes: HashMap::new(),
        }
    }

    fn add_node(&mut self, index: usize, node: CircuitNode) {
        self.nodes.insert(index, node);
    }

    fn display_graph(&self) {
        println!("Graph:");
        let mut sorted_nodes: Vec<_> = self.nodes.iter().collect(); // Collect nodes into a vector
        sorted_nodes.sort_by_key(|(&index, _)| index); // Sort by the index

        for (index, node) in sorted_nodes {
            println!("Node {}: {}", index, node.name);
            println!("  Inputs: {:?}", node.inputs);
            println!("  Outputs: {:?}", node.outputs);
        }
    }
}

fn load_onnx_using_tract(reader: &mut dyn std::io::Read) -> Result<TractResult, TractError> {
    use std::collections::HashMap;
    use tract_onnx::tract_hir::internal::GenericFactoid;

    // Load the ONNX model from the reader
    let mut model = tract_onnx::onnx().model_for_read(reader)?;

    // Create a placeholder HashMap for variables
    let variables: HashMap<String, usize> = HashMap::new(); // Empty since run_args is removed

    // Iterate through the inputs and set input facts
    for (i, id) in model.clone().inputs.iter().enumerate() {
        let input = model.node_mut(id.node);
        let mut fact: InferenceFact = input.outputs[0].fact.clone();

        for (i, x) in fact.clone().shape.dims().enumerate() {
            if matches!(x, GenericFactoid::Any) {
                // Placeholder value since run_args is removed
                let batch_size = variables.get("batch_size").unwrap_or(&1); // Use default batch_size
                fact.shape
                    .set_dim(i, tract_onnx::prelude::TDim::Val(*batch_size as i64));
            }
        }

        // Set the input fact back to the model
        model.set_input_fact(i, fact)?;
    }

    // Set default output facts for each output
    for (i, _) in model.clone().outputs.iter().enumerate() {
        model.set_output_fact(i, InferenceFact::default())?;
    }

    // Since `run_args` and variables are removed, the symbol values will be empty
    let symbol_values = SymbolValues::default();

    // Convert the model into a typed model and declutter
    let typed_model = model
        .into_typed()?
        .concretize_dims(&symbol_values)?
        .into_decluttered()?;

    Ok((typed_model, symbol_values))
}

// Function to format the TypedFact
fn format_fact(fact: &InferenceFact) -> String {
    let data_type = fact.datum_type;
    let shape = format_shape(&fact.shape);
    format!("Type: {:?}, Shape: {:?}", data_type, shape)
}

// Function to format the ShapeFact by iterating over dimensions
fn format_shape(shape: &ShapeFactoid) -> String {
    let dims: Vec<String> = shape
        .dims()
        .map(|dim| format!("{}", dim)) // Format each dimension
        .collect();
    format!("[{}]", dims.join(", "))
}

// New function to handle Outlet<TypedFact>
fn format_outlet(outlet: &Outlet<InferenceFact>) -> String {
    let fact_info = format_fact(&outlet.fact); // Reuse the format_fact function
    format!("Outlet - {}", fact_info)
}

fn main() -> () {
    // Load and optimize the ONNX model
    load_onnx_using_tract(&mut std::fs::File::open("./tests/iris_model.onnx").unwrap()).unwrap();
    let model = tract_onnx::onnx()
        .model_for_path("./tests/iris_model.onnx")
        .unwrap();

    // Initialize the graph to store nodes
    let mut graph = CircuitGraph::new();

    // Extract the nodes from the model
    let nodes = model.nodes();

    println!("Model nodes:");

    // Iterate over each node and capture its inputs, outputs, and types
    for (i, node) in nodes.iter().enumerate() {
        // Get the name of the node
        let node_name = node.op.name();

        // Format the inputs and also store the input index
        let node_inputs: Vec<(OutletId, String)> = node
            .inputs
            .iter()
            .map(|input| (*input, format_fact(model.outlet_fact(*input).unwrap()))) // Store index and fact
            .collect();

        // Format the outputs by using their index (OutletId) and extracting the outlet fact
        let node_outputs: Vec<(Outlet<InferenceFact>, String)> = node
            .outputs
            .iter()
            .map(|output_id| (output_id.clone(), format_outlet(output_id))) // Store index and outlet fact
            .collect();

        // Create the Node struct and add it to the graph
        let graph_node = CircuitNode {
            name: node_name.to_string(),
            inputs: node_inputs,
            outputs: node_outputs,
        };

        // Add the node to the graph
        graph.add_node(i, graph_node);
    }

    // Display the full graph with input/output types and shapes
    graph.display_graph();
}
