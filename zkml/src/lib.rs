pub mod dag;
pub mod verifiers;

// Re-export commonly used verifiers
pub use verifiers::verify_matmul;
pub use verifiers::verify_tensor_add;
pub use verifiers::verify_tensor_sub;

// Re-export DAG types
pub use dag::{ComputationGraph, DagAssignment, DagCircuit, TensorNode};
