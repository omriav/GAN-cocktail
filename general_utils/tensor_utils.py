import torch

def convert_tensor_prediction_to_numpy(tensor_prediction):
    numpy_img = (tensor_prediction.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()

    return numpy_img