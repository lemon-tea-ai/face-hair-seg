import torch
from nets.MobileNetV2_unet import MobileNetV2_unet
from config import INPUT_SIZE

def load_model():
    model = MobileNetV2_unet(None).to(torch.device("cpu"))
    state_dict = torch.load("checkpoints/model.pt", map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

example_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

traced_model = torch.jit.trace(model, example_input)
traced_model.save("traced.pt")