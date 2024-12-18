import torch
import coremltools as ct
import numpy as np  # Make sure to import numpy
from coremltools.models.neural_network import quantization_utils

# Load the traced model
traced_model = torch.jit.load("traced.pt")

# Convert to CoreML with 8-bit quantization
model_input = ct.TensorType(shape=(1, 3, 224, 224))
coreml_model = ct.convert(
    traced_model,
    inputs=[model_input],
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.iOS16,
)

# Save the original model first
coreml_model.save("facehairseg.mlpackage")

# Load the saved model with weights
model = ct.models.MLModel(
    "facehairseg.mlpackage", 
    weights_dir="facehairseg.mlpackage/Data/com.apple.CoreML/weights")

# Apply 8-bit quantization
quantized_model = quantization_utils.quantize_weights(
    full_precision_model=model,
    nbits=8,
    quantization_mode='linear')

# Save the quantized model
quantized_model.save("facehairseg_quantized.mlpackage")
