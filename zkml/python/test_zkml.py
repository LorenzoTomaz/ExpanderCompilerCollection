import json
import os
from zkml import PyDagCircuit

ONE = 1 << 16  # 2^16 for fixed-point arithmetic


def test_invalid_matmul():
    # Create the same graph as above
    graph = {
        "nodes": {
            "a": {
                "uuid": "a",
                "shape": [2, 2],
                "op_name": "input",
                "parents": [],
                "parameters": None,
            },
            "b": {
                "uuid": "b",
                "shape": [2, 2],
                "op_name": "input",
                "parents": [],
                "parameters": None,
            },
            "c": {
                "uuid": "c",
                "shape": [2, 2],
                "op_name": "matmul",
                "parents": ["a", "b"],
                "parameters": None,
            },
        },
        "output_node": "c",
    }

    circuit = PyDagCircuit(json.dumps(graph))

    # Initialize tensors
    circuit.init_tensor("a")
    circuit.init_tensor("b")
    circuit.init_tensor("c")

    # Test with incorrect values (scaled by 2^16)
    tensor_values = {
        "a": [1 * ONE, 2 * ONE, 3 * ONE, 4 * ONE],
        "b": [5 * ONE, 6 * ONE, 7 * ONE, 8 * ONE],
        "c": [0 * ONE, 0 * ONE, 0 * ONE, 0 * ONE],  # Incorrect result
    }

    # Generate witness files
    circuit_path = "test_circuit_invalid.txt"
    witness_path = "test_witness_invalid.txt"
    witness_solver_path = "test_witness_solver_invalid.txt"
    proof_path = "test_proof_invalid.txt"

    try:
        # Generate witness - this should succeed
        result = circuit.generate_witness(
            tensor_values,
            circuit_path,
            witness_path,
            witness_solver_path,
            proof_path,
        )
        assert (
            result is True
        ), "Witness generation should succeed even with invalid values"

        # Generate proof from existing files - this should fail
        result = PyDagCircuit.generate_proof(
            circuit_path,
            witness_path,
            witness_solver_path,
            proof_path,
        )
        assert result is False, "Proof verification should fail with invalid values"

    finally:
        # Cleanup test files
        for file in [circuit_path, witness_path, witness_solver_path, proof_path]:
            if os.path.exists(file):
                os.remove(file)
