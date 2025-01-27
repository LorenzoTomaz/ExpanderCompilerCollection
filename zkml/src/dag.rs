use arith::Field;
use arith::FieldSerde;
use expander_compiler::circuit::layered::witness::Witness;
use expander_compiler::circuit::layered::Circuit;
use expander_compiler::field::BN254;
use expander_compiler::frontend::internal::DumpLoadTwoVariables;
use expander_compiler::frontend::Variable;
use expander_compiler::frontend::*;
use expander_compiler::frontend::{compile, BN254Config};
use expander_config::{self, BN254ConfigKeccak};
use gkr::{self, Prover, Verifier};
use internal::Serde;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use std::fs::File;
use std::io::BufWriter;

/// Trait for assigning values to circuit variables
pub trait Assignment<F: Field, C> {
    fn assign(&self, circuit: &C) -> Vec<(Variable, F)>;
}

/// Represents a tensor node in the computation graph
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TensorNode {
    /// Unique identifier for the tensor
    pub uuid: String,
    /// Shape of the tensor
    pub shape: Vec<u64>,
    /// Operation that produced this tensor
    pub op_name: String,
    /// UUIDs of parent tensors
    pub parents: Vec<String>,
    /// Optional parameters for the operation
    pub parameters: Option<serde_json::Value>,
    /// Signs of the tensor elements (true for positive, false for negative)
    #[serde(default)]
    pub signs: Vec<bool>,
}

/// Represents the entire computation graph
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ComputationGraph {
    /// All nodes in the graph, keyed by their UUID
    pub nodes: HashMap<String, TensorNode>,
    /// The final output node's UUID
    pub output_node: String,
}

/// Circuit that verifies a computation graph
#[derive(Clone)]
pub struct DagCircuit {
    /// The computation graph to verify
    pub graph: ComputationGraph,
    /// The flattened tensor data for each node
    tensor_data: HashMap<String, Vec<Variable>>,
}

impl DagCircuit {
    pub fn new(graph: ComputationGraph) -> Self {
        let tensor_data = HashMap::new();
        Self { graph, tensor_data }
    }

    /// Initialize tensor data for a node
    pub fn init_tensor(&mut self, uuid: &str) -> Vec<Variable> {
        let node = self.graph.nodes.get(uuid).unwrap();
        let size = node.shape.iter().product::<u64>() as usize;
        let data = vec![Variable::default(); size];
        self.tensor_data.insert(uuid.to_string(), data.clone());
        data
    }

    /// Get or initialize tensor data for a node
    pub fn get_tensor(&mut self, uuid: &str) -> Vec<Variable> {
        if let Some(data) = self.tensor_data.get(uuid) {
            data.clone()
        } else {
            self.init_tensor(uuid)
        }
    }

    /// Update signs for a node
    pub fn update_signs(&mut self, uuid: &str, signs: Vec<bool>) {
        if let Some(node) = self.graph.nodes.get_mut(uuid) {
            node.signs = signs;
        }
    }
}

impl<C: Config> Define<C> for DagCircuit {
    fn define(&self, builder: &mut API<C>) {
        let mut processed = HashMap::new();
        let mut to_process = vec![self.graph.output_node.clone()];

        // Initialize result as true
        let mut result = builder.constant(C::CircuitField::from(1u32));

        // Process nodes in reverse topological order
        while let Some(uuid) = to_process.pop() {
            if processed.contains_key(&uuid) {
                continue;
            }

            let node = self.graph.nodes.get(&uuid).unwrap();

            // Check if all parents are processed
            let mut all_parents_processed = true;
            for parent in &node.parents {
                if !processed.contains_key(parent) {
                    all_parents_processed = false;
                    to_process.push(uuid.clone()); // Add current node back
                    to_process.push(parent.clone()); // Add unprocessed parent
                }
            }

            // Skip if parents aren't processed yet
            if !all_parents_processed {
                continue;
            }

            // Get tensor data
            let output_data = self.tensor_data.get(&uuid).unwrap();

            // Verify operation and combine result
            let verifier_result = match node.op_name.as_str() {
                "input" => {
                    // Input nodes don't need verification
                    builder.constant(C::CircuitField::from(1u32))
                }
                "add" => {
                    assert_eq!(node.parents.len(), 2, "Add requires 2 inputs");
                    let a_data = self.tensor_data.get(&node.parents[0]).unwrap();
                    let b_data = self.tensor_data.get(&node.parents[1]).unwrap();
                    let a_node = self.graph.nodes.get(&node.parents[0]).unwrap();
                    let b_node = self.graph.nodes.get(&node.parents[1]).unwrap();

                    crate::verifiers::verify_tensor_add(
                        builder,
                        a_data,
                        b_data,
                        output_data,
                        &a_node.signs,
                        &b_node.signs,
                        &node.signs,
                        &node.shape,
                    )
                }
                "sub" => {
                    assert_eq!(node.parents.len(), 2, "Sub requires 2 inputs");
                    let a_data = self.tensor_data.get(&node.parents[0]).unwrap();
                    let b_data = self.tensor_data.get(&node.parents[1]).unwrap();
                    let a_node = self.graph.nodes.get(&node.parents[0]).unwrap();
                    let b_node = self.graph.nodes.get(&node.parents[1]).unwrap();

                    crate::verifiers::verify_tensor_sub(
                        builder,
                        a_data,
                        b_data,
                        output_data,
                        &a_node.signs,
                        &b_node.signs,
                        &node.signs,
                        &node.shape,
                    )
                }
                "matmul" => {
                    assert_eq!(node.parents.len(), 2, "Matmul requires 2 inputs");
                    let a_data = self.tensor_data.get(&node.parents[0]).unwrap();
                    let b_data = self.tensor_data.get(&node.parents[1]).unwrap();
                    let a_node = self.graph.nodes.get(&node.parents[0]).unwrap();
                    let b_node = self.graph.nodes.get(&node.parents[1]).unwrap();

                    crate::verifiers::verify_matmul(
                        builder,
                        a_data,
                        b_data,
                        output_data,
                        &a_node.signs,
                        &b_node.signs,
                        &node.signs,
                        &a_node.shape,
                        &b_node.shape,
                        &node.shape,
                        5, // Default number of iterations
                    )
                }
                "sqrt" => {
                    assert_eq!(node.parents.len(), 1, "Sqrt requires 1 input");
                    let input_data = self.tensor_data.get(&node.parents[0]).unwrap();
                    let input_node = self.graph.nodes.get(&node.parents[0]).unwrap();

                    crate::verifiers::verify_sqrt(
                        builder,
                        input_data,
                        output_data,
                        &input_node.signs,
                        &node.signs,
                        &node.shape,
                    )
                }
                // Add more operations as they are implemented
                _ => panic!("Unsupported operation: {}", node.op_name),
            };

            // Combine with previous results
            result = builder.and(result, verifier_result);
            processed.insert(uuid.clone(), true);
        }

        // Assert final result
        let true_const = builder.constant(C::CircuitField::from(1u32));
        builder.assert_is_equal(result, true_const)
    }
}

impl DumpLoadTwoVariables<Variable> for DagCircuit {
    fn dump_into(&self, vars1: &mut Vec<Variable>, vars2: &mut Vec<Variable>) {
        // Process nodes in deterministic order
        let mut processed = HashSet::new();
        let mut to_process = vec![];

        // Get sorted input nodes first
        let mut input_nodes: Vec<_> = self
            .graph
            .nodes
            .iter()
            .filter(|(_, node)| node.op_name == "input")
            .collect();
        input_nodes.sort_by_key(|(uuid, _)| *uuid);
        for (uuid, _) in input_nodes {
            to_process.push(uuid.clone());
        }

        // Then add output node
        to_process.push(self.graph.output_node.clone());

        while let Some(uuid) = to_process.pop() {
            if processed.contains(&uuid) {
                continue;
            }

            let node = self.graph.nodes.get(&uuid).unwrap();

            // Add unprocessed parents to the queue in sorted order
            let mut parents: Vec<_> = node
                .parents
                .iter()
                .filter(|p| !processed.contains(*p))
                .collect();
            parents.sort(); // Sort parents for deterministic order
            if !parents.is_empty() {
                to_process.push(uuid.clone());
                for parent in parents.iter().rev() {
                    // Reverse to maintain order with stack
                    to_process.push((*parent).clone());
                }
                continue;
            }

            // Process this node's variables
            if let Some(tensor_data) = self.tensor_data.get(&uuid) {
                for var in tensor_data {
                    vars1.push(*var);
                    vars2.push(*var);
                }
            }

            processed.insert(uuid);
        }
    }

    fn load_from(&mut self, vars1: &mut &[Variable], vars2: &mut &[Variable]) {
        // Process nodes in same deterministic order as dump_into
        let mut processed = HashSet::new();
        let mut to_process = vec![];

        // Get sorted input nodes first
        let mut input_nodes: Vec<_> = self
            .graph
            .nodes
            .iter()
            .filter(|(_, node)| node.op_name == "input")
            .collect();
        input_nodes.sort_by_key(|(uuid, _)| *uuid);
        for (uuid, _) in input_nodes {
            to_process.push(uuid.clone());
        }

        // Then add output node
        to_process.push(self.graph.output_node.clone());

        while let Some(uuid) = to_process.pop() {
            if processed.contains(&uuid) {
                continue;
            }

            let node = self.graph.nodes.get(&uuid).unwrap();

            // Add unprocessed parents to the queue in sorted order
            let mut parents: Vec<_> = node
                .parents
                .iter()
                .filter(|p| !processed.contains(*p))
                .collect();
            parents.sort(); // Sort parents for deterministic order
            if !parents.is_empty() {
                to_process.push(uuid.clone());
                for parent in parents.iter().rev() {
                    // Reverse to maintain order with stack
                    to_process.push((*parent).clone());
                }
                continue;
            }

            // Process this node's variables
            if let Some(tensor_data) = self.tensor_data.get_mut(&uuid) {
                for var in tensor_data {
                    *var = vars1[0];
                    *vars1 = &vars1[1..];
                    *vars2 = &vars2[1..];
                }
            }

            processed.insert(uuid);
        }
    }

    fn num_vars(&self) -> (usize, usize) {
        let n = self.tensor_data.values().map(|v| v.len()).sum();
        (n, n)
    }
}

/// Generate witness files for a DAG circuit
pub fn generate_witness(
    circuit: &DagCircuit,
    assignment: &DagAssignment<BN254>,
    circuit_path: &str,
    witness_path: &str,
    witness_solver_path: &str,
    proof_path: &str,
) -> bool {
    let compile_result = compile::<BN254Config, DagCircuit>(circuit).unwrap();
    let witness = compile_result
        .witness_solver
        .solve_witness(assignment)
        .unwrap();
    let output = compile_result.layered_circuit.run(&witness);
    assert_eq!(output, vec![true]);
    // Generate witness files
    let file = File::create(circuit_path).unwrap();
    let writer = BufWriter::new(file);
    compile_result
        .layered_circuit
        .serialize_into(writer)
        .unwrap();

    let file = File::create(witness_path).unwrap();
    let writer = BufWriter::new(file);
    witness.serialize_into(writer).unwrap();

    let file = File::create(witness_solver_path).unwrap();
    let writer = BufWriter::new(file);
    compile_result
        .witness_solver
        .serialize_into(writer)
        .unwrap();

    println!("Witness files generated successfully");

    // Generate and verify proof
    generate_and_verify_proof(&compile_result.layered_circuit, &witness, proof_path)
}

/// Generate proof from existing witness files
pub fn generate_proof_from_files(
    circuit_path: &str,
    witness_path: &str,
    witness_solver_path: &str,
    proof_path: &str,
) -> bool {
    // Load circuit and witness from files
    let circuit_file = File::open(circuit_path).unwrap();
    let witness_file = File::open(witness_path).unwrap();
    let layered_circuit = Circuit::<BN254Config>::deserialize_from(circuit_file).unwrap();
    let witness = Witness::<BN254Config>::deserialize_from(witness_file).unwrap();

    // Generate and verify proof
    generate_and_verify_proof(&layered_circuit, &witness, proof_path)
}

/// Generate and verify proof for a circuit
pub fn generate_and_verify_proof(
    layered_circuit: &Circuit<BN254Config>,
    witness: &Witness<BN254Config>,
    proof_path: &str,
) -> bool {
    // Generate expander proof
    type GKRConfig = expander_config::BN254ConfigKeccak;
    let mut expander_circuit = layered_circuit.export_to_expander::<GKRConfig>().flatten();

    let config = expander_config::Config::<GKRConfig>::new(
        expander_config::GKRScheme::Vanilla,
        expander_config::MPIConfig::new(),
    );

    let (simd_input, simd_public_input) =
        witness.to_simd::<<GKRConfig as expander_config::GKRConfig>::SimdCircuitField>();
    expander_circuit.layers[0].input_vals = simd_input;
    expander_circuit.public_input = simd_public_input.clone();

    // Prove
    expander_circuit.evaluate();
    let mut prover = gkr::Prover::new(&config);
    prover.prepare_mem(&expander_circuit);
    let (claimed_v, proof) = prover.prove(&mut expander_circuit);

    // Save proof to file
    let file = File::create(proof_path).unwrap();
    let writer = BufWriter::new(file);
    proof.serialize_into(writer).unwrap();

    // Verify
    let verifier = gkr::Verifier::new(&config);
    let result = verifier.verify(
        &mut expander_circuit,
        &simd_public_input,
        &claimed_v,
        &proof,
    );

    println!("Expander proof generated and verified successfully");
    assert_eq!(result, true);
    result
}

/// Assignment for the DAG circuit
#[derive(Debug)]
pub struct DagAssignment<F: Field> {
    /// Values for each tensor in the graph
    pub tensor_values: HashMap<String, Vec<F>>,
}

impl<F: Field> Assignment<F, DagCircuit> for DagAssignment<F> {
    fn assign(&self, circuit: &DagCircuit) -> Vec<(Variable, F)> {
        let mut assignment = Vec::new();

        // Assign values to all tensors in the circuit
        for (uuid, tensor_data) in &circuit.tensor_data {
            let values = self.tensor_values.get(uuid).expect("Missing tensor values");
            assert_eq!(tensor_data.len(), values.len(), "Tensor size mismatch");

            for (var, value) in tensor_data.iter().zip(values.iter()) {
                assignment.push((*var, *value));
            }
        }

        assignment
    }
}

impl<F: Field> DumpLoadTwoVariables<F> for DagAssignment<F> {
    fn dump_into(&self, vars1: &mut Vec<F>, vars2: &mut Vec<F>) {
        // Process nodes in same deterministic order as circuit
        let mut nodes: Vec<_> = self.tensor_values.iter().collect();
        nodes.sort_by_key(|(uuid, _)| *uuid);
        for (_, values) in nodes {
            for value in values {
                vars1.push(*value);
                vars2.push(*value);
            }
        }
    }

    fn load_from(&mut self, vars1: &mut &[F], vars2: &mut &[F]) {
        // Process nodes in same deterministic order as dump_into
        let mut nodes: Vec<_> = self.tensor_values.iter_mut().collect();
        nodes.sort_by_key(|(uuid, _)| *uuid);
        for (_, values) in nodes {
            for value in values {
                *value = vars1[0];
                *vars1 = &vars1[1..];
                *vars2 = &vars2[1..];
            }
        }
    }

    fn num_vars(&self) -> (usize, usize) {
        let n = self.tensor_values.values().map(|v| v.len()).sum();
        (n, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use expander_compiler::field::BN254;
    use expander_compiler::frontend::{compile, BN254Config};
    use internal::Serde;
    use std::fs::File;
    use std::io::BufWriter;

    const ONE: u32 = 1 << 16;

    #[test]
    fn test_matmul() {
        // Create a simple DAG for matrix multiplication
        let mut nodes = HashMap::new();

        // Input matrices A and B
        nodes.insert(
            "a".to_string(),
            TensorNode {
                uuid: "a".to_string(),
                shape: vec![2, 2],
                op_name: "input".to_string(),
                parents: vec![],
                parameters: None,
                signs: vec![true, false, true, true], // [+1, -2, +3, +4]
            },
        );
        nodes.insert(
            "b".to_string(),
            TensorNode {
                uuid: "b".to_string(),
                shape: vec![2, 2],
                op_name: "input".to_string(),
                parents: vec![],
                parameters: None,
                signs: vec![true, true, false, true], // [+5, +6, -7, +8]
            },
        );

        // Output matrix C = A * B
        nodes.insert(
            "c".to_string(),
            TensorNode {
                uuid: "c".to_string(),
                shape: vec![2, 2],
                op_name: "matmul".to_string(),
                parents: vec!["a".to_string(), "b".to_string()],
                parameters: None,
                signs: vec![true, false, false, true], // [+19, -10, -13, +50]
            },
        );

        let graph = ComputationGraph {
            nodes,
            output_node: "c".to_string(),
        };

        let mut circuit = DagCircuit::new(graph);

        // Initialize input tensors with variables
        circuit.init_tensor("a");
        circuit.init_tensor("b");
        circuit.init_tensor("c");

        // Test correct multiplication with signs
        let mut tensor_values = HashMap::new();
        tensor_values.insert(
            "a".to_string(),
            vec![
                BN254::from(1u32 * ONE), // +1
                BN254::from(2u32 * ONE), // -2
                BN254::from(3u32 * ONE), // +3
                BN254::from(4u32 * ONE), // +4
            ],
        );
        tensor_values.insert(
            "b".to_string(),
            vec![
                BN254::from(5u32 * ONE), // +5
                BN254::from(6u32 * ONE), // +6
                BN254::from(7u32 * ONE), // -7
                BN254::from(8u32 * ONE), // +8
            ],
        );
        tensor_values.insert(
            "c".to_string(),
            vec![
                BN254::from(19u32 * ONE), // -19
                BN254::from(10u32 * ONE), // +22
                BN254::from(13u32 * ONE), // +43
                BN254::from(50u32 * ONE), // +50
            ],
        );

        let assignment = DagAssignment { tensor_values };

        assert!(generate_witness(
            &circuit,
            &assignment,
            "circuit_dag_bn254.txt",
            "witness_dag_bn254.txt",
            "witness_dag_bn254_solver.txt",
            "proof_dag_bn254.txt"
        ));
    }

    #[test]
    fn test_matmul_dag_from_json() {
        // Load the test graph from JSON
        let json_str = fs::read_to_string("tests/assets/matmul_test.json")
            .expect("Failed to read test JSON file");
        let mut graph: ComputationGraph =
            serde_json::from_str(&json_str).expect("Failed to parse JSON into ComputationGraph");

        // Initialize signs for each node
        graph.nodes.get_mut("a").unwrap().signs = vec![true, false, true, true]; // [+1, -2, +3, +4]
        graph.nodes.get_mut("b").unwrap().signs = vec![true, true, false, true]; // [+5, +6, -7, +8]
        graph.nodes.get_mut("c").unwrap().signs = vec![true, false, false, true]; // [+19, -10, -13, +50]

        let mut circuit = DagCircuit::new(graph);

        // Initialize input tensors with variables
        circuit.init_tensor("a");
        circuit.init_tensor("b");
        circuit.init_tensor("c");

        let compile_result = compile::<BN254Config, DagCircuit>(&circuit).unwrap();

        // Test correct multiplication with signs
        let mut tensor_values = HashMap::new();
        tensor_values.insert(
            "a".to_string(),
            vec![
                BN254::from(1u32 * ONE), // +1
                BN254::from(2u32 * ONE), // -2
                BN254::from(3u32 * ONE), // +3
                BN254::from(4u32 * ONE), // +4
            ],
        );
        tensor_values.insert(
            "b".to_string(),
            vec![
                BN254::from(5u32 * ONE), // +5
                BN254::from(6u32 * ONE), // +6
                BN254::from(7u32 * ONE), // -7
                BN254::from(8u32 * ONE), // +8
            ],
        );
        tensor_values.insert(
            "c".to_string(),
            vec![
                BN254::from(19u32 * ONE), // -19 (+1*+5 + (-2)*+6)
                BN254::from(10u32 * ONE), // +22 (+1*+6 + (-2)*(-7))
                BN254::from(13u32 * ONE), // +43 (+3*+5 + +4*+6)
                BN254::from(50u32 * ONE), // +50 (+3*+6 + +4*+8)
            ],
        );

        let assignment = DagAssignment { tensor_values };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);

        // Test incorrect multiplication
        let mut wrong_tensor_values = HashMap::new();
        wrong_tensor_values.insert(
            "a".to_string(),
            vec![
                BN254::from(1u32 * ONE), // +1
                BN254::from(2u32 * ONE), // -2
                BN254::from(3u32 * ONE), // +3
                BN254::from(4u32 * ONE), // +4
            ],
        );
        wrong_tensor_values.insert(
            "b".to_string(),
            vec![
                BN254::from(5u32 * ONE), // +5
                BN254::from(6u32 * ONE), // +6
                BN254::from(7u32 * ONE), // -7
                BN254::from(8u32 * ONE), // +8
            ],
        );
        wrong_tensor_values.insert(
            "c".to_string(),
            vec![
                BN254::from(19u32 * ONE), // -19
                BN254::from(22u32 * ONE), // +22
                BN254::from(43u32 * ONE), // +43
                BN254::from(51u32 * ONE), // Wrong value
            ],
        );

        let wrong_assignment = DagAssignment {
            tensor_values: wrong_tensor_values,
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&wrong_assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![false]);
    }

    #[test]
    fn test_sqrt() {
        // Create a simple DAG for square root
        let mut nodes = HashMap::new();

        // Input value
        nodes.insert(
            "0".to_string(),
            TensorNode {
                uuid: "0".to_string(),
                shape: vec![1], // Changed from [1,1] to [1]
                op_name: "input".to_string(),
                parents: vec![],
                parameters: None,
                signs: vec![true], // positive input
            },
        );

        // Output sqrt
        nodes.insert(
            "1".to_string(),
            TensorNode {
                uuid: "1".to_string(),
                shape: vec![1], // Changed from [1,1] to [1]
                op_name: "sqrt".to_string(),
                parents: vec!["0".to_string()],
                parameters: None,
                signs: vec![true], // positive output
            },
        );

        let graph = ComputationGraph {
            nodes,
            output_node: "1".to_string(),
        };

        let mut circuit = DagCircuit::new(graph);

        // Initialize tensors with variables
        circuit.init_tensor("0");
        circuit.init_tensor("1");

        // Test sqrt(16) = 4
        let mut tensor_values = HashMap::new();
        tensor_values.insert(
            "0".to_string(),
            vec![BN254::from(16u32 * ONE)], // 16
        );
        tensor_values.insert(
            "1".to_string(),
            vec![BN254::from(4u32 * ONE)], // 4
        );

        let assignment = DagAssignment { tensor_values };

        // Generate witness and verify
        let compile_result = compile::<BN254Config, DagCircuit>(&circuit).unwrap();
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);

        // Test with proof generation
        assert!(generate_witness(
            &circuit,
            &assignment,
            "circuit_sqrt_bn254.txt",
            "witness_sqrt_bn254.txt",
            "witness_sqrt_bn254_solver.txt",
            "proof_sqrt_bn254.txt"
        ));

        // Clean up test files
        for file in [
            "circuit_sqrt_bn254.txt",
            "witness_sqrt_bn254.txt",
            "witness_sqrt_bn254_solver.txt",
            "proof_sqrt_bn254.txt",
        ] {
            if std::path::Path::new(file).exists() {
                std::fs::remove_file(file).unwrap();
            }
        }
    }

    #[test]
    fn test_chained_matmul_dag() {
        // Load the test graph from JSON
        let json_str = r#"{"nodes": {"0": {"uuid": "0", "shape": [2, 2], "op_name": "input", "parents": [], "parameters": null}, "1": {"uuid": "1", "shape": [2, 2], "op_name": "input", "parents": [], "parameters": null}, "2": {"uuid": "2", "shape": [2, 2], "op_name": "input", "parents": [], "parameters": null}, "3": {"uuid": "3", "shape": [2, 2], "op_name": "matmul", "parents": ["0", "1"], "parameters": null}, "4": {"uuid": "4", "shape": [2, 2], "op_name": "matmul", "parents": ["3", "2"], "parameters": null}}, "output_node": "4"}"#;
        let mut graph: ComputationGraph =
            serde_json::from_str(json_str).expect("Failed to parse JSON into ComputationGraph");

        // Initialize signs for each node
        graph.nodes.get_mut("0").unwrap().signs = vec![true, true, true, true]; // aa values
        graph.nodes.get_mut("1").unwrap().signs = vec![true, true, true, true]; // bb values
        graph.nodes.get_mut("2").unwrap().signs = vec![true, true, true, true]; // dd values
        graph.nodes.get_mut("3").unwrap().signs = vec![true, true, true, true]; // cc values
        graph.nodes.get_mut("4").unwrap().signs = vec![true, true, true, true]; // ee values

        let mut circuit = DagCircuit::new(graph);

        // Initialize tensors with variables
        circuit.init_tensor("0");
        circuit.init_tensor("1");
        circuit.init_tensor("2");
        circuit.init_tensor("3");
        circuit.init_tensor("4");

        let compile_result = compile::<BN254Config, DagCircuit>(&circuit).unwrap();

        // Test correct multiplication with fixed-point scaling
        let mut tensor_values = HashMap::new();
        tensor_values.insert(
            "0".to_string(),
            vec![
                BN254::from(64684u32), // aa values
                BN254::from(131072u32),
                BN254::from(196608u32),
                BN254::from(262144u32),
            ],
        );
        tensor_values.insert(
            "1".to_string(),
            vec![
                BN254::from(32768u32), // bb values
                BN254::from(425984u32),
                BN254::from(491520u32),
                BN254::from(524288u32),
            ],
        );
        tensor_values.insert(
            "2".to_string(),
            vec![
                BN254::from(65536u32), // dd values
                BN254::from(131072u32),
                BN254::from(196608u32),
                BN254::from(262144u32),
            ],
        );
        tensor_values.insert(
            "3".to_string(),
            vec![
                BN254::from(1015382u32), // cc values
                BN254::from(1469022u32),
                BN254::from(2064384u32),
                BN254::from(3375104u32),
            ],
        );
        tensor_values.insert(
            "4".to_string(),
            vec![
                BN254::from(5422448u32), // ee values
                BN254::from(7906852u32),
                BN254::from(12189696u32),
                BN254::from(17629184u32),
            ],
        );

        let assignment = DagAssignment { tensor_values };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![true]);

        // Test with proof generation
        assert!(generate_witness(
            &circuit,
            &assignment,
            "circuit_chained_matmul_bn254.txt",
            "witness_chained_matmul_bn254.txt",
            "witness_chained_matmul_bn254_solver.txt",
            "proof_chained_matmul_bn254.txt"
        ));

        // Clean up test files
        for file in [
            "circuit_chained_matmul_bn254.txt",
            "witness_chained_matmul_bn254.txt",
            "witness_chained_matmul_bn254_solver.txt",
            "proof_chained_matmul_bn254.txt",
        ] {
            if std::path::Path::new(file).exists() {
                std::fs::remove_file(file).unwrap();
            }
        }
    }
}
