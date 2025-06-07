import os
import pandas as pd
import torch
import cv2
import matplotlib.pyplot as plt

from networks.resnet import resnet50
from gradcam import GradCAM, load_image, overlay_heatmap

def main():
    # 1) Path to your per-image CSV
    csv_path = 'results/custom_per_image.csv'
    df = pd.read_csv(csv_path)

    # 2) Load NPR model onto CPU, stripping "module." prefixes
    model = resnet50(num_classes=1)
    checkpoint = torch.load('NPR.pth', map_location='cpu')
    raw_state = checkpoint.get('model', checkpoint)
    state_dict = {k.replace('module.', ''): v for k, v in raw_state.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 3) Initialize Grad-CAM on "layer2.3.conv3"
    gcam = GradCAM(model, target_layer_name='layer2.3.conv3')

    # 4) Create output directory
    output_dir = 'heatmaps_all'
    os.makedirs(output_dir, exist_ok=True)

    # 5) Loop over every row in CSV
    for idx, row in df.iterrows():
        img_path = row['image_path']
        # Load and preprocess
        x, orig_bgr = load_image(img_path, load_size=256, crop_size=224)
        # Compute heatmap (224×224)
        heatmap = gcam(x)
        # Resize original to match heatmap size
        H, W = heatmap.shape
        orig_resized = cv2.resize(orig_bgr, (W, H))
        # Overlay heatmap
        blended = overlay_heatmap(orig_resized, heatmap, alpha=0.5)
        # Save overlay
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_dir, f"heatmap_{idx}_{filename}")
        cv2.imwrite(save_path, blended)

    print(f"Saved all heatmaps to '{output_dir}/'")

    # 6) Display first 5 overlays as examples
    example_files = sorted(os.listdir(output_dir))[:5]
    if len(example_files) == 0:
        print("No heatmaps to display.")
        return

    plt.figure(figsize=(15, 6))
    for i, fname in enumerate(example_files):
        img = cv2.imread(os.path.join(output_dir, fname))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 5, i+1)
        plt.imshow(img_rgb)
        plt.title(fname)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
