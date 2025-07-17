
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
import os

def load_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.eval()
    return model

def generate_gradcam(model, input_tensor, original_path):
    model.eval()
    feature_maps = []
    gradients = []

    def forward_hook(module, input, output):
        feature_maps.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_fw = model.layer4.register_forward_hook(forward_hook)
    handle_bw = model.layer4.register_backward_hook(backward_hook)

    output = model(input_tensor)
    class_idx = torch.argmax(output, dim=1).item()
    model.zero_grad()
    class_score = output[0, class_idx]
    class_score.backward()

    grads_val = gradients[0].cpu().data.numpy()[0]
    fmap = feature_maps[0].cpu().data.numpy()[0]
    weights = np.mean(grads_val, axis=(1, 2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * fmap[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= np.min(cam)
    cam /= np.max(cam)
    cam = np.uint8(255 * cam)
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    original_img = Image.open(original_path).resize((224, 224)).convert('RGB')
    original_np = np.array(original_img)
    overlayed = cv2.addWeighted(original_np, 0.5, cam, 0.5, 0)

    gradcam_path = os.path.join('static', 'uploads', 'gradcam_' + os.path.basename(original_path))
    cv2.imwrite(gradcam_path, overlayed[:, :, ::-1])

    handle_fw.remove()
    handle_bw.remove()

    return gradcam_path
