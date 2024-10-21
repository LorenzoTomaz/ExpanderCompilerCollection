use std::collections::HashMap;
use tract_onnx::prelude::*;
type TractResult = (Graph<TypedFact, Box<dyn TypedOp>>, SymbolValues);
use expander_compiler::expander_runner::executor;
use expander_compiler::fieldutils;
use expander_compiler::frontend::internal::Serde;
use expander_compiler::frontend::*;
use expander_compiler::tensor::{Tensor as FPTensor, TensorError};
use expander_compiler::Scale;
use serde::{Deserialize, Serialize};
use thiserror::Error;
#[cfg(not(target_arch = "wasm32"))]
use tract_onnx::tract_hir::internal::DimLike;

fn get_shape(shape: &ShapeFact) -> Vec<usize> {
    let dims: Vec<usize> = shape
        .iter()
        .map(|dim| dim.to_usize().unwrap()) // Format each dimension
        .collect();
    dims
}

/// An enum representing the operations that can be expressed as arithmetic (non lookup) operations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PolyOp {
    Add,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Input {
    Source,
}

pub fn add_c(
    inputs: &[FPTensor<Variable>],
    builder: &mut API<BN254Config>,
) -> Result<FPTensor<Variable>, TensorError> {
    let sum = inputs[0].clone();
    let rhs = inputs[1].clone();
    let result = sum.add_constraint::<BN254Config>(rhs, builder).unwrap();
    Ok(result)
}

//implement Add using the layout method
impl PolyOp {
    pub fn layout(
        &self,
    ) -> fn(&[FPTensor<Variable>], &mut API<BN254Config>) -> Result<FPTensor<Variable>, TensorError>
    {
        match self {
            PolyOp::Add => add_c,
        }
    }
}

pub struct TensorNode {
    pub op: String,
    pub input_ids: Vec<usize>,
    pub output: FPTensor<Variable>,
}

fn process_graph(
    graph: CircuitGraph,
    builder: &mut API<BN254Config>,
    circuit_input: &Vec<Vec<Variable>>,
) -> FPTensor<Variable> {
    // a hashmap to store the nodes
    let mut nodes_ops: HashMap<usize, TensorNode> = HashMap::new();
    // sort graph.nodes by index
    let mut sorted_nodes: Vec<_> = graph.nodes.iter().collect();
    sorted_nodes.sort_by_key(|(&index, _)| index);
    for (idx, node) in sorted_nodes.iter() {
        print!("Node: {:?}", **idx);
        match node.name.as_ref() {
            "Add" => {
                for x in node.inputs.iter() {
                    let a = x.node;
                    print!("Input: {:?}", a);
                    nodes_ops.get(&a).unwrap();
                }
                let inputs: Vec<FPTensor<Variable>> = node
                    .inputs
                    .iter()
                    .map(|id| {
                        print!("Input: {:?}", id.node);
                        let tensor = nodes_ops.get(&id.node).unwrap();
                        tensor.output.clone()
                    })
                    .collect();
                let op = PolyOp::Add;
                let layout = op.layout();
                let result = layout(&inputs, builder).unwrap();
                let tensor_node = TensorNode {
                    op: node.name.clone(),
                    input_ids: node.inputs.iter().map(|id| id.node).collect(),
                    output: result,
                };
                nodes_ops.insert(**idx, tensor_node);
            }
            "Source" => {
                let input = Input::Source;
                //let layout = input.layout();
                let const_value = &circuit_input[**idx]
                    .iter()
                    .map(|value| builder.constant(value))
                    .collect::<Vec<_>>();
                let constant_tensor =
                    FPTensor::<Variable>::new(Some(const_value), &node.shape).unwrap();

                let tensor_node = TensorNode {
                    op: node.name.clone(),
                    input_ids: [].to_vec(),
                    output: constant_tensor,
                };
                nodes_ops.insert(**idx, tensor_node);
            }
            _ => {
                // Handle other cases or add a default behavior
            }
        }
    }
    // get last idx of node_ops
    let last_idx = sorted_nodes.last().unwrap().0;
    let last_node = nodes_ops.get(&last_idx).unwrap();
    last_node.output.clone()
}

declare_circuit!(Circuit {
    sum: PublicVariable,
    x: [[Variable; 1]; 2],
});

impl Define<BN254Config> for Circuit<Variable> {
    fn define(&self, builder: &mut API<BN254Config>) {
        let model =
            load_onnx_using_tract(&mut std::fs::File::open("./tests/2_addition.onnx").unwrap())
                .unwrap()
                .0;
        // Initialize the graph to store nodes
        let mut graph = CircuitGraph::new();
        // Extract the nodes from the model
        let nodes = model.nodes.clone();

        println!("Model nodes:");
        //sort nodes by index
        // Iterate over each node and capture its inputs, outputs, and types
        for (i, node) in nodes.iter().enumerate() {
            // Get the name of the node
            let node_name = node.op.name();

            // Format the inputs and also store the input index
            let node_inputs: Vec<OutletId> = node
                .inputs
                .iter()
                .map(|input| *input) // Store index and fact
                .collect();

            // Format the outputs by using their index (OutletId) and extracting the outlet fact
            let node_outputs: Vec<Outlet<TypedFact>> = node
                .outputs
                .iter()
                .map(|output_id| output_id.clone()) // Store index and outlet fact
                .collect();

            let node_shape = get_shape(&node.outputs[0].fact.shape);
            // Create the Node struct and add it to the graph
            let graph_node = CircuitNode {
                name: node_name.to_string(),
                inputs: node_inputs,
                outputs: node_outputs,
                shape: node_shape,
            };
            // Add the node to the graph
            graph.add_node(i, graph_node);
        }
        graph.display_graph();
        let inputs: Vec<Vec<Variable>> = self.x.iter().map(|x| x.to_vec()).collect();
        let output_node = process_graph(graph, builder, &inputs);
        let first_obj = output_node.clone();
        let f = first_obj[0].clone();
        builder.assert_is_equal(f, self.sum);
    }
}

pub enum SupportedOp {
    /// A linear operation.
    Linear(PolyOp),
    Input(Input),
}

/// circuit related errors.
#[derive(Debug, Error)]
pub enum GraphError {
    /// The wrong inputs were passed to a lookup node
    #[error("invalid inputs for a lookup node")]
    InvalidLookupInputs,
    /// Shape mismatch in circuit construction
    #[error("invalid dimensions used for node {0} ({1})")]
    InvalidDims(usize, String),
    /// Wrong method was called to configure an op
    #[error("wrong method was called to configure node {0} ({1})")]
    WrongMethod(usize, String),
    /// A requested node is missing in the graph
    #[error("a requested node is missing in the graph: {0}")]
    MissingNode(usize),
    /// The wrong method was called on an operation
    #[error("an unsupported method was called on node {0} ({1})")]
    OpMismatch(usize, String),
    #[error("unsupported data type: {0}")]
    UnsupportedDataType(usize, String),
    #[error("tensor error: {0}")]
    TensorError(#[from] TensorError),
}

/// Converts a scale (log base 2) to a fixed point multiplier.
pub fn scale_to_multiplier(scale: Scale) -> f64 {
    f64::powf(2., scale as f64)
}

pub fn quantize_float(
    elem: &f64,
    shift: f64,
    scale: Scale,
) -> Result<fieldutils::IntegerRep, TensorError> {
    let mult = scale_to_multiplier(scale);
    let max_value = ((fieldutils::IntegerRep::MAX as f64 - shift) / mult).round(); // the maximum value that can be represented w/o sig bit truncation

    if *elem > max_value {
        return Err(TensorError::SigBitTruncationError);
    }

    // we parallelize the quantization process as it seems to be quite slow at times
    let scaled = (mult * *elem + shift).round() as fieldutils::IntegerRep;

    Ok(scaled)
}

/// Converts a tensor to a [ValTensor] with a given scale.
pub fn quantize_tensor(
    const_value: FPTensor<f32>,
    scale: crate::Scale,
) -> Result<FPTensor<fieldutils::IntegerRep>, TensorError> {
    let value: FPTensor<fieldutils::IntegerRep> = const_value.par_enum_map(|_, x| {
        Ok::<_, TensorError>(
            quantize_float(&(x).into(), 0.0, scale).unwrap() as fieldutils::IntegerRep
        )
    })?;

    //value.set_scale(scale);
    Ok(value)
}

#[cfg(not(target_arch = "wasm32"))]
use tract_onnx::prelude::SymbolValues;
#[cfg(not(target_arch = "wasm32"))]
/// Extracts the raw values from a tensor.
pub fn extract_tensor_value(
    input: Arc<tract_onnx::prelude::Tensor>,
) -> Result<FPTensor<f32>, GraphError> {
    let dt = input.datum_type();
    let dims = input.shape().to_vec();

    let mut const_value: FPTensor<f32>;
    if dims.is_empty() && input.len() == 0 {
        const_value = FPTensor::<f32>::new(None, &dims).unwrap();
        return Ok(const_value);
    }

    match dt {
        // DatumType::F16 => {
        //     let vec = input.as_slice::<tract_onnx::prelude::f16>()?.to_vec();
        //     let cast: Vec<f32> = vec.par_iter().map(|x| (*x).into()).collect();
        //     const_value = FPTensor::<f32>::new(Some(&cast), &dims)?;
        // }
        DatumType::F32 => {
            let vec = input.as_slice::<f32>().unwrap().to_vec();
            const_value = FPTensor::<f32>::new(Some(&vec), &dims).unwrap();
        }
        // DatumType::F64 => {
        //     let vec = input.as_slice::<f64>()?.to_vec();
        //     let cast: Vec<f32> = vec.par_iter().map(|x| *x as f32).collect();
        //     const_value = FPTensor::<f32>::new(Some(&cast), &dims)?;
        // }
        // DatumType::I64 => {
        //     // Generally a shape or hyperparam
        //     let vec = input.as_slice::<i64>()?.to_vec();
        //     let cast: Vec<f32> = vec.par_iter().map(|x| *x as f32).collect();
        //     const_value = FPTensor::<f32>::new(Some(&cast), &dims)?;
        // }
        // DatumType::I32 => {
        //     // Generally a shape or hyperparam
        //     let vec = input.as_slice::<i32>()?.to_vec();
        //     let cast: Vec<f32> = vec.par_iter().map(|x| *x as f32).collect();
        //     const_value = FPTensor::<f32>::new(Some(&cast), &dims)?;
        // }
        // DatumType::I16 => {
        //     // Generally a shape or hyperparam
        //     let vec = input.as_slice::<i16>()?.to_vec();
        //     let cast: Vec<f32> = vec.par_iter().map(|x| *x as f32).collect();
        //     const_value = FPTensor::<f32>::new(Some(&cast), &dims)?;
        // }
        // DatumType::I8 => {
        //     // Generally a shape or hyperparam
        //     let vec = input.as_slice::<i8>()?.to_vec();
        //     let cast: Vec<f32> = vec.par_iter().map(|x| *x as f32).collect();
        //     const_value = FPTensor::<f32>::new(Some(&cast), &dims)?;
        // }
        // DatumType::U8 => {
        //     // Generally a shape or hyperparam
        //     let vec = input.as_slice::<u8>()?.to_vec();
        //     let cast: Vec<f32> = vec.par_iter().map(|x| *x as f32).collect();
        //     const_value = FPTensor::<f32>::new(Some(&cast), &dims)?;
        // }
        // DatumType::U16 => {
        //     // Generally a shape or hyperparam
        //     let vec = input.as_slice::<u16>()?.to_vec();
        //     let cast: Vec<f32> = vec.par_iter().map(|x| *x as f32).collect();
        //     const_value = FPTensor::<f32>::new(Some(&cast), &dims)?;
        // }
        // DatumType::U32 => {
        //     // Generally a shape or hyperparam
        //     let vec = input.as_slice::<u32>()?.to_vec();
        //     let cast: Vec<f32> = vec.par_iter().map(|x| *x as f32).collect();
        //     const_value = FPTensor::<f32>::new(Some(&cast), &dims)?;
        // }
        // DatumType::U64 => {
        //     // Generally a shape or hyperparam
        //     let vec = input.as_slice::<u64>()?.to_vec();
        //     let cast: Vec<f32> = vec.par_iter().map(|x| *x as f32).collect();
        //     const_value = FPTensor::<f32>::new(Some(&cast), &dims)?;
        // }
        // DatumType::Bool => {
        //     // Generally a shape or hyperparam
        //     let vec = input.as_slice::<bool>()?.to_vec();
        //     let cast: Vec<f32> = vec.par_iter().map(|x| *x as usize as f32).collect();
        //     const_value = FPTensor::<f32>::new(Some(&cast), &dims)?;
        // }
        // DatumType::TDim => {
        //     // Generally a shape or hyperparam
        //     let vec = input.as_slice::<tract_onnx::prelude::TDim>()?.to_vec();

        //     let cast: Result<Vec<f32>, GraphError> = vec
        //         .par_iter()
        //         .map(|x| match x.to_i64() {
        //             Ok(v) => Ok(v as f32),
        //             Err(_) => match x.to_i64() {
        //                 Ok(v) => Ok(v as f32),
        //                 Err(_) => Err(GraphError::UnsupportedDataType(0, "TDim".to_string())),
        //             },
        //         })
        //         .collect();

        //     const_value = FPTensor::<f32>::new(Some(&cast?), &dims)?;
        // }
        _ => return Err(GraphError::UnsupportedDataType(0, format!("{:?}", dt))),
    }
    const_value.reshape(&dims).unwrap();

    Ok(const_value)
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

// Define a structure for the Node in the graph
#[derive(Debug)]
struct CircuitNode {
    name: String,
    inputs: Vec<OutletId>,           // Store both index and input information
    outputs: Vec<Outlet<TypedFact>>, // Store both index and output information
    shape: Vec<usize>,
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
            println!(
                "  Inputs: {:?}",
                node.inputs.iter().map(|id| (id.node)).collect::<Vec<_>>()
            );
            println!("  Outputs: {:?}", node.outputs[0].fact.konst);
            // check if node.outputs[0].0.fact.konst is not None
            println!("  Tensor shape: {:?}", node.shape);
            let konst = node.outputs[0].fact.konst.clone();
            if konst.is_some() {
                print!(
                    "Tensor {:?}",
                    quantize_tensor(
                        extract_tensor_value(node.outputs[0].fact.konst.clone().unwrap()).unwrap(),
                        10
                    )
                    .unwrap()
                );
            }
        }
    }
}

fn main() -> () {
    // count time
    let mut start = std::time::Instant::now();
    let compile_result = compile(&Circuit::default()).unwrap();
    println!("Time to compile: {:?}", start.elapsed());
    let CompileResult {
        witness_solver,
        layered_circuit,
    } = compile_result;
    let assignment = Circuit::<BN254> {
        sum: BN254::from(4u64),
        x: [[BN254::from(1u64)], [BN254::from(2u64)]],
    };
    let witness = witness_solver.solve_witness(&assignment).unwrap();
    let output = layered_circuit.run(&witness);
    assert_eq!(output, vec![true]);
    let file = std::fs::File::create("circuit.txt").unwrap();
    let writer = std::io::BufWriter::new(file);
    layered_circuit.serialize_into(writer).unwrap();

    let file = std::fs::File::create("witness.txt").unwrap();
    let writer = std::io::BufWriter::new(file);
    witness.serialize_into(writer).unwrap();
    println!("dumped to files");
    start = std::time::Instant::now();
    executor();
    println!("Time to execute: {:?}", start.elapsed());
}
