import torch
import numpy as np
import glob
import os
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, accuracy_score
from options.test_options import TestOptions
from data import create_dataloader

def validate(model, opt):
    """
    Runs inference over opt.dataroot (one “subfolder” at a time).  
    We now gather all image files under opt.dataroot (recursively), so nested folders are fine.
    Returns:
      - acc: overall accuracy
      - ap: average precision
      - r_acc: accuracy on real images (true_label == 0)
      - f_acc: accuracy on fake images (true_label == 1)
      - details: list of (image_path, true_label, score, pred_label)
    """

    # 1. Build a recursive list of all image files under opt.dataroot
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    # Use '**/*.*' with recursive=True to capture nested folders
    all_candidates = glob.glob(os.path.join(opt.dataroot, '**', '*.*'), recursive=True)
    all_paths = sorted([p for p in all_candidates if os.path.splitext(p)[1].lower() in exts])

    # print(f"[DEBUG] Found {len(all_paths)} image files under '{opt.dataroot}':")
    # for i, p in enumerate(all_paths[:10]):
    #     print(f"  {i+1:3d}) {p}")
    # if len(all_paths) > 10:
    #     print(f"  ... (and {len(all_paths)-10} more)")

    path_idx = 0

    data_loader = create_dataloader(opt)

    y_true_all = []
    y_pred_all = []
    details = []

    model.eval()
    with torch.no_grad():
        for batch_num, batch in enumerate(data_loader):
            # The loader still returns (img, label), so unpack accordingly:
            img, label = batch
            B = img.shape[0]  # batch size

            # Forward pass on CPU
            logits = model(img).flatten()
            scores = torch.sigmoid(logits).cpu().numpy()   # probabilities [0,1]
            truths = label.cpu().numpy().astype(int)       # 0 or 1
            preds  = (scores > 0.5).astype(int)

            # Grab the next B paths (if available). Otherwise, pad with empty strings
            batch_paths = all_paths[path_idx : path_idx + B]
            if len(batch_paths) < B:
                # Fewer paths than batch size: something’s off with ordering/structure
                print(f"[WARNING] Only found {len(batch_paths)} paths for batch {batch_num} (expected {B}).")
                # Pad the remainder with empty strings, so indexing never fails
                batch_paths += [''] * (B - len(batch_paths))

            path_idx += B

            for i in range(B):
                image_path = batch_paths[i]
                true_label = truths[i]
                score      = float(scores[i])
                pred_label = int(preds[i])
                details.append((image_path, true_label, score, pred_label))

            y_true_all.extend(truths.tolist())
            y_pred_all.extend(scores.tolist())

    # Compute per-class accuracies
    y_true_arr = np.array(y_true_all)
    y_pred_arr = np.array(y_pred_all)

    if np.any(y_true_arr == 0):
        r_acc = accuracy_score(y_true_arr[y_true_arr == 0],
                               (y_pred_arr[y_true_arr == 0] > 0.5))
    else:
        r_acc = float('nan')

    if np.any(y_true_arr == 1):
        f_acc = accuracy_score(y_true_arr[y_true_arr == 1],
                               (y_pred_arr[y_true_arr == 1] > 0.5))
    else:
        f_acc = float('nan')

    acc = accuracy_score(y_true_arr, (y_pred_arr > 0.5))
    ap  = average_precision_score(y_true_arr, y_pred_arr)

    return acc, ap, r_acc, f_acc,details,0


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    model = resnet50(num_classes=1)
    checkpoint = torch.load(opt.model_path, map_location='cpu')

    raw_state = checkpoint.get('model', checkpoint)
    state_dict = {}
    for k, v in raw_state.items():
        new_key = k.replace('module.', '')
        state_dict[new_key] = v
    model.load_state_dict(state_dict, strict=True)

    # We’re forcing CPU mode, so do NOT call model.cuda()
    model.eval()

    acc, avg_precision, r_acc, f_acc, details = validate(model, opt)

    print(f"\nOverall accuracy:        {acc:.4f}")
    print(f"Average precision (AP):  {avg_precision:.4f}")
    print(f"Accuracy on real (0):    {r_acc:.4f}")
    print(f"Accuracy on fake (1):    {f_acc:.4f}")

    print("\nSample of per-image results:")
    for i, (img_path, true_lbl, score, pred_lbl) in enumerate(details[:10]):
        print(f" {i:2d}) Path: {img_path} | True={true_lbl} | Score={score:.3f} | Pred={pred_lbl}")

