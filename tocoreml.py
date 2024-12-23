import torch
import coremltools as ct
import numpy as np  # Make sure to import numpy
from config import INPUT_SIZE

# Load the traced model
traced_model = torch.jit.load("traced.pt")

# Convert to CoreML with 8-bit quantization
model_input = ct.TensorType(shape=(1, 3, INPUT_SIZE, INPUT_SIZE))
coreml_model = ct.convert(
    traced_model,
    inputs=[model_input],
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.iOS16,
)

# Save the original model first
coreml_model.save("facehairseg128.mlpackage")
