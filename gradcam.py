# gradcam.py

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer_name: str):
        """
        model: a resnet50(num_classes=1) on CPU, in eval() mode.
        target_layer_name: e.g. 'layer2.3.conv3'
        """
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # Register hooks
        for name, module in self.model.named_modules():
            if name == target_layer_name:
                module.register_forward_hook(self._save_activation)
                module.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        # output: Tensor [B, C, H', W']
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple; take the first element
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, index=None):
        """
        input_tensor: 1×3×224×224 (CPU)
        index: which class logit to backpropagate (0 for single-output)
        Returns heatmap (H×W) in [0,1].
        """

        # Ensure everything is on CPU
        assert input_tensor.device.type == 'cpu'
        assert next(self.model.parameters()).device.type == 'cpu'

        # Forward pass
        logits = self.model(input_tensor)        # shape [1,1]
        if index is None:
            index = 0
        score = logits[0, index]                # scalar tensor

        self.model.zero_grad()
        score.backward()

        # gradients: [1, C, H', W']
        grads = self.gradients[0]                # [C, H', W']
        fmap = self.activations[0]               # [C, H', W']

        # Global average pool over H'×W'
        weights = grads.mean(dim=(1, 2))         # [C]

        # Weighted sum of fmap channels
        gcam = torch.zeros(fmap.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            gcam += w * fmap[i]

        # ReLU
        gcam = F.relu(gcam)

        # Normalize to [0,1]
        gcam -= gcam.min()
        if gcam.max() != 0:
            gcam /= gcam.max()

        # Upsample to input resolution
        gcam_np = gcam.cpu().numpy()
        gcam_resized = cv2.resize(
            gcam_np, (input_tensor.shape[3], input_tensor.shape[2])
        )
        return gcam_resized


def load_image(path, load_size=256, crop_size=224):
    """
    Returns:
      - img_tensor: 1×3×224×224 float32 (CPU)
      - orig_bgr:    H×W×3 uint8 BGR (for overlay)
    """
    img = Image.open(path).convert('RGB')
    orig = np.array(img)
    orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)

    transform = transforms.Compose([
        transforms.Resize(load_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img_tensor = transform(img).unsqueeze(0)  # 1×3×224×224
    return img_tensor, orig_bgr


def overlay_heatmap(orig_bgr, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    orig_bgr: H×W×3 uint8
    heatmap:  H×W float32 in [0,1]
    Returns blended BGR image (uint8).
    """
    hmap = np.uint8(255 * heatmap)
    hmap = cv2.applyColorMap(hmap, colormap)  # H×W×3 BGR
    blended = cv2.addWeighted(orig_bgr, 1 - alpha, hmap, alpha, 0)
    return blended
