# ZKML Python Package

This package provides Python bindings for the ZKML library, which implements zero-knowledge proofs for machine learning operations.

## Installation

```bash
pip install setuptools-rust
pip install .
```

## Usage

```python
import json
from zkml import PyDagCircuit

# Create a computation graph
graph = {
    "nodes": {
        "a": {
            "uuid": "a",
            "shape": [2, 2],
            "op_name": "input",
            "parents": [],
            "parameters": None
        },
        "b": {
            "uuid": "b",
            "shape": [2, 2],
            "op_name": "input",
            "parents": [],
            "parameters": None
        },
        "c": {
            "uuid": "c",
            "shape": [2, 2],
            "op_name": "matmul",
            "parents": ["a", "b"],
            "parameters": None
        }
    },
    "output_node": "c"
}

# Create circuit
circuit = PyDagCircuit(json.dumps(graph))

# Initialize tensors
circuit.init_tensor("a")
circuit.init_tensor("b")
circuit.init_tensor("c")

# Verify computation
tensor_values = {
    "a": [1, 2, 3, 4],
    "b": [5, 6, 7, 8],
    "c": [19, 22, 43, 50]
}

result = circuit.verify(tensor_values)
print(f"Verification result: {result}") 