import torch
from nets.MobileNetV2_unet import MobileNetV2_unet

def load_model():
    model = MobileNetV2_unet(None).to(torch.device("cpu"))
    state_dict = torch.load("checkpoints/model.pt", map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# Prepare model for quantization
torch.backends.quantized.engine = 'qnnpack'
model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
torch.quantization.prepare(model, inplace=True)

# Calibrate the model (you should use real data here)
example_input = torch.randn(1, 3, 224, 224)
model(example_input)  # Run inference with example data for calibration

# Convert to quantized model
torch.quantization.convert(model, inplace=True)

# Export the quantized model
traced_model = torch.jit.trace(model, example_input)
traced_model.save("traced_quantized.pt")