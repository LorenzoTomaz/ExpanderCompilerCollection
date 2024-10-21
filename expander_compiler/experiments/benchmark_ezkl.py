# flake8: noqa
import ezkl
import os
import json
import torch
import subprocess
from pprint import pprint
from pprint import pformat
from typing import Any

from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexers import PythonLexer


def pprint_color(obj: Any) -> None:
    """Pretty-print in color."""
    print(highlight(pformat(obj), PythonLexer(), Terminal256Formatter()), end="")


# Define a simple model that adds two tensors
class AddModel(torch.nn.Module):
    def forward(self, x, y):
        z = x + y
        return x + z


# Create an instance of the model
circuit = AddModel()

# Create two example input tensors with shape [1]
x = torch.tensor([1.0], dtype=torch.float32)
y = torch.tensor([2.0], dtype=torch.float32)

model_path = os.path.join("network.onnx")
compiled_model_path = os.path.join("network.compiled")
pk_path = os.path.join("test.pk")
vk_path = os.path.join("test.vk")
settings_path = os.path.join("settings.json")
srs_path = os.path.join("kzg.srs")
witness_path = os.path.join("witness.json")
data_path = os.path.join("input.json")

# Switch the model to evaluation mode
circuit.eval()

# # Export the model to ONNX format
torch.onnx.export(
    circuit,  # model being run
    (x, y),  # model inputs (tuple for multiple inputs)
    model_path,  # where to save the model (file or file-like object)
    export_params=True,
    opset_version=10,  # the ONNX version to export the model to
    do_constant_folding=True,
    input_names=["input1", "input2"],  # the model's input names
    output_names=["output"],  # the model's output names
)

# Serialize the input data into JSON
data_array1 = ((x).detach().numpy()).reshape([-1]).tolist()
data_array2 = ((y).detach().numpy()).reshape([-1]).tolist()

data = dict(input_data=[data_array1, data_array2])

# Save the input data to a JSON file
json.dump(data, open(data_path, "w"))

# Create and set PyRunArgs
py_run_args = ezkl.PyRunArgs()
py_run_args.input_visibility = "public"
py_run_args.output_visibility = "public"
py_run_args.param_visibility = "private"  # private by default
py_run_args.scale_rebase_multiplier = 10


async def main():
    # Generate the settings
    ezkl.gen_settings(model_path, settings_path, py_run_args=py_run_args)

    # Compile the circuit
    ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
    # Fetch the SRS (Structured Reference String)
    await ezkl.get_srs(settings_path)

    # Generate the witness file
    await ezkl.gen_witness(data_path, compiled_model_path, witness_path)
    assert os.path.isfile(witness_path)

    # Setup the proving key (pk) and verification key (vk)
    ezkl.setup(compiled_model_path, vk_path, pk_path)
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)
    assert os.path.isfile(settings_path)

    # Generate the proof
    proof_path = os.path.join("test.pf")
    import time

    s = time.time()
    ezkl.prove(
        witness_path,
        compiled_model_path,
        pk_path,
        proof_path,
        "single",  # Proof type
    )
    end = time.time() - s

    subprocess.run(["cargo", "run", "--bin", "reader"])
    pprint_color(f"Ezkl proof generated in {end} seconds")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
