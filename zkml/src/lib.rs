use pyo3::prelude::*;
use pyo3::types::PyDict;

pub mod dag;
pub mod verifiers;

// Re-export commonly used verifiers
pub use verifiers::{verify_matmul, verify_tensor_add, verify_tensor_sub};

// Re-export DAG types
pub use dag::{
    generate_proof_from_files, generate_witness, ComputationGraph, DagAssignment, DagCircuit,
    TensorNode,
};

use expander_compiler::frontend::compile;

#[pyclass]
struct PyDagCircuit {
    inner: DagCircuit,
}

#[pymethods]
impl PyDagCircuit {
    #[new]
    fn new(graph_json: &str) -> PyResult<Self> {
        let graph: ComputationGraph = serde_json::from_str(graph_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: DagCircuit::new(graph),
        })
    }

    fn init_tensor(&mut self, uuid: &str) -> PyResult<()> {
        self.inner.init_tensor(uuid);
        Ok(())
    }

    fn generate_witness(
        &mut self,
        tensor_values: &PyDict,
        tensor_signs: &PyDict,
        circuit_path: &str,
        witness_path: &str,
        witness_solver_path: &str,
        proof_path: &str,
        should_scale: Option<bool>,
    ) -> PyResult<bool> {
        use expander_compiler::field::BN254;

        let mut values = std::collections::HashMap::new();
        for (key, value) in tensor_values.iter() {
            let key = key.extract::<String>()?;
            let value = value.extract::<Vec<u64>>()?;
            let field_values: Vec<BN254> = value
                .into_iter()
                .map(|x| {
                    if should_scale.unwrap_or(false) {
                        BN254::from(x * (1 << 16))
                    } else {
                        BN254::from(x)
                    }
                })
                .collect();
            values.insert(key.clone(), field_values);
        }

        // Update signs in the graph
        for (key, signs) in tensor_signs.iter() {
            let key = key.extract::<String>()?;
            let signs = signs.extract::<Vec<bool>>()?;
            self.inner.update_signs(&key, signs);
        }

        let assignment = DagAssignment {
            tensor_values: values,
        };
        let result = std::panic::catch_unwind(|| {
            generate_witness(
                &self.inner,
                &assignment,
                circuit_path,
                witness_path,
                witness_solver_path,
                proof_path,
            )
        });

        // Return false if panic occurred, true if verification succeeded
        match result {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    #[staticmethod]
    fn generate_proof(
        circuit_path: &str,
        witness_path: &str,
        witness_solver_path: &str,
        proof_path: &str,
    ) -> PyResult<bool> {
        let result = std::panic::catch_unwind(|| {
            generate_proof_from_files(circuit_path, witness_path, witness_solver_path, proof_path)
        });

        match result {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
}

#[pymodule]
fn zkml(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyDagCircuit>()?;
    Ok(())
}
