import torch
import coremltools as ct
import numpy as np  # Make sure to import numpy

# Load the traced model
traced_model = torch.jit.load("traced.pt")

# Convert to CoreML
model_input = ct.TensorType(shape=(1, 3, 224, 224))
coreml_model = ct.convert(
    traced_model,
    inputs=[model_input],
    compute_units=ct.ComputeUnit.ALL
)

# Save as .mlpackage instead of .mlmodel
coreml_model.save("facehairseg.mlpackage")
