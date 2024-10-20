use std::collections::HashMap;
use tract_onnx::prelude::*;
type TractResult = (Graph<TypedFact, Box<dyn TypedOp>>, SymbolValues);
use expander_compiler::fieldutils::{self, IntegerRep};
use expander_compiler::tensor::{Tensor as FPTensor, TensorError, TensorType};
use expander_compiler::Scale;
use halo2curves::bn256::Fr;
use halo2curves::ff::PrimeField;
use std::convert::Infallible;
use thiserror::Error;
#[cfg(not(target_arch = "wasm32"))]
use tract_onnx::tract_hir::{
    internal::DimLike,
    ops::array::{Pad, PadMode, TypedConcat},
    ops::cnn::PoolSpec,
    ops::konst::Const,
    ops::nn::DataFormat,
    tract_core::ops::cast::Cast,
    tract_core::ops::cnn::{conv::KernelFormat, MaxPool, SumPool},
};

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
    let mut value: FPTensor<fieldutils::IntegerRep> = const_value.par_enum_map(|_, x| {
        Ok::<_, TensorError>(
            quantize_float(&(x).into(), 0.0, scale).unwrap() as fieldutils::IntegerRep
        )
    })?;

    //value.set_scale(scale);
    Ok(value)
}

#[cfg(not(target_arch = "wasm32"))]
fn load_op<C: tract_onnx::prelude::Op + Clone>(
    op: &dyn tract_onnx::prelude::Op,
    idx: usize,
    name: String,
) -> Result<C, GraphError> {
    // Extract the slope layer hyperparams
    let op: &C = match op.downcast_ref::<C>() {
        Some(b) => b,
        None => {
            return Err(GraphError::OpMismatch(idx, name));
        }
    };

    Ok(op.clone())
}

#[cfg(not(target_arch = "wasm32"))]
use tract_onnx::prelude::SymbolValues;
#[cfg(not(target_arch = "wasm32"))]
/// Extracts the raw values from a tensor.
pub fn extract_tensor_value(
    input: Arc<tract_onnx::prelude::Tensor>,
) -> Result<FPTensor<f32>, GraphError> {
    use maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

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

fn new_op_from_onnx(
    node: &Node<TypedFact, Box<dyn TypedOp>>,
    idx: usize,
) -> Result<FPTensor<fieldutils::IntegerRep>, GraphError> {
    match node.op().name().as_ref() {
        "Const" => {
            let op: Const = load_op::<Const>(node.op(), idx, node.op().name().to_string())?;
            let dt = op.0.datum_type();
            // Raw values are always f32
            let raw_value = extract_tensor_value(op.0)?;
            // If bool or a tensor dimension then don't scale
            let mut constant_scale = match dt {
                DatumType::Bool
                | DatumType::TDim
                | DatumType::I64
                | DatumType::I32
                | DatumType::I16
                | DatumType::I8
                | DatumType::U8
                | DatumType::U16
                | DatumType::U32
                | DatumType::U64 => 0,
                DatumType::F16 | DatumType::F32 | DatumType::F64 => 10,
                _ => {
                    return Err(GraphError::UnsupportedDataType(idx, format!("{:?}", dt)));
                }
            };
            let quantized_value = quantize_tensor(raw_value.clone(), constant_scale)?;
            // Quantize the raw value
            // let quantized_value = quantize_tensor(raw_value.clone(), constant_scale)?;
            Ok(quantized_value)
        }
        "Source" => {
            let quantized_value = FPTensor::<IntegerRep>::new(None, &[1])?;
            Ok(quantized_value)
        }
        "EinSum" => {
            let quantized_value = FPTensor::<IntegerRep>::new(None, &[1])?;
            Ok(quantized_value)
        }
        "Add" => {
            let quantized_value = FPTensor::<IntegerRep>::new(None, &[1])?;
            Ok(quantized_value)
        }
        "Max" => {
            let quantized_value = FPTensor::<IntegerRep>::new(None, &[1])?;
            Ok(quantized_value)
        }
        _ => Err(GraphError::OpMismatch(idx, node.op().name().to_string())),
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

// Define a structure for the Node in the graph
#[derive(Debug)]
struct CircuitNode {
    name: String,
    inputs: Vec<(OutletId, String)>, // Store both index and input information
    outputs: Vec<(Outlet<TypedFact>, String)>, // Store both index and output information
    node_datum: FPTensor<fieldutils::IntegerRep>,
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
                node.inputs
                    .iter()
                    .map(|(id, info)| (id.node))
                    .collect::<Vec<_>>()
            );
            println!("  Outputs: {:?}", node.outputs[0].0.fact.konst);
            // check if node.outputs[0].0.fact.konst is not None
            let konst = node.outputs[0].0.fact.konst.clone();
            if konst.is_some() {
                print!(
                    "Tensor {:?}",
                    quantize_tensor(
                        extract_tensor_value(node.outputs[0].0.fact.konst.clone().unwrap())
                            .unwrap(),
                        10
                    )
                    .unwrap()
                );
            }
            //new_op_from_onnx(node.outputs[0].0, *index);
        }
    }
}

// Function to format the TypedFact
fn format_fact(fact: &TypedFact) -> String {
    let data_type = fact.datum_type;
    let shape = format_shape(&fact.shape);
    format!("Type: {:?}, Shape: {:?}", data_type, shape)
}

// Function to format the ShapeFact by iterating over dimensions
fn format_shape(shape: &ShapeFact) -> String {
    let dims: Vec<String> = shape
        .iter()
        .map(|dim| format!("{}", dim)) // Format each dimension
        .collect();
    format!("[{}]", dims.join(", "))
}

// New function to handle Outlet<TypedFact>
fn format_outlet(outlet: &Outlet<TypedFact>) -> String {
    let fact_info = format_fact(&outlet.fact); // Reuse the format_fact function
    format!("Outlet - {}", fact_info)
}

fn main() -> () {
    // // Load and optimize the ONNX model
    // let model = tract_onnx::onnx()
    //     .model_for_path("./tests/iris_model.onnx")
    //     .unwrap();
    let model = load_onnx_using_tract(&mut std::fs::File::open("./tests/addition.onnx").unwrap())
        .unwrap()
        .0;
    // Initialize the graph to store nodes
    let mut graph = CircuitGraph::new();
    let a = model.nodes.clone();
    // Extract the nodes from the model
    let nodes = model.nodes.clone();

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
        let node_outputs: Vec<(Outlet<TypedFact>, String)> = node
            .outputs
            .iter()
            .map(|output_id| (output_id.clone(), format_outlet(output_id))) // Store index and outlet fact
            .collect();
        let node_datum = new_op_from_onnx(node, i).unwrap();
        // Create the Node struct and add it to the graph
        let graph_node = CircuitNode {
            name: node_name.to_string(),
            inputs: node_inputs,
            outputs: node_outputs,
            node_datum: node_datum,
        };

        // Add the node to the graph
        graph.add_node(i, graph_node);
    }

    // Display the full graph with input/output types and shapes
    graph.display_graph();
}
