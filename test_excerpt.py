# test.py (excerpt)

import os, time, csv, numpy as np, torch
from validate import validate
from util import printSet
from networks.resnet import resnet50
from options.test_options import TestOptions
import sys
import time
import os
import csv
import torch
from util import Logger, printSet
from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
import networks.resnet as resnet
import numpy as np
import random
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # no cuda
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
seed_torch(100)
DetectionTests = {
                # 'ForenSynths': { 'dataroot'   : '/opt/data/private/DeepfakeDetection/ForenSynths/',
                #                  'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
                #                  'no_crop'    : True,
                #                },

          #  'GANGen-Detection': { 'dataroot'   : '/opt/data/private/DeepfakeDetection/GANGen-Detection/',
          #                        'no_resize'  : True,
          #                        'no_crop'    : True,
          #                      },
            # 'GANGen-Detection': { 'dataroot'   : './data/datasets/GANGen-Detection/',
            #                         'no_resize'  : True,
            #                         'no_crop'    : True,
            #                    },
            'custom': { 'dataroot'   : './data/datasets/custom/',
                                'no_resize'  : False,
                                'no_crop'    : True,
                            },
        #  'DiffusionForensics': { 'dataroot'   : '/opt/data/private/DeepfakeDetection/DiffusionForensics/',
        #                          'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
        #                          'no_crop'    : True,
        #                        },

        # 'UniversalFakeDetect': { 'dataroot'   : '/opt/data/private/DeepfakeDetection/UniversalFakeDetect/',
        #                          'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
        #                          'no_crop'    : True,
        #                        },

                 }


opt = TestOptions().parse(print_options=False)
# FORCE CPU mode:
opt.gpu_ids = []   # this prevents torch.cuda.set_device(...) from running

print(f'Model_path {opt.model_path}')

# # get model
# model = resnet50(num_classes=1)
# model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=True)
# # can't do cuda on my machine
# # model.cuda()
# model.eval()

model = resnet50(num_classes=1)

# Load the full checkpoint (it contains keys "model", "optimizer", "total_steps", etc.)
checkpoint = torch.load(opt.model_path, map_location='cpu')

# Extract the actual state_dict
if 'model' in checkpoint:
    raw_state = checkpoint['model']
else:
    raw_state = checkpoint

# Strip any "module." prefixes (added by DataParallel) so keys match your resnet50 definition
state_dict = {}
for k, v in raw_state.items():
    new_key = k.replace('module.', '')  # remove leading "module."
    state_dict[new_key] = v

# Now load into the ResNet exactly
model.load_state_dict(state_dict, strict=True)

# (We’re in CPU‐only mode, so do NOT call model.cuda())
model.eval()


for testSet in DetectionTests.keys():
    dataroot = DetectionTests[testSet]['dataroot']
    printSet(testSet)

    accs = [];aps = []
    print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    for v_id, val in enumerate(os.listdir(dataroot)):
        opt.dataroot = '{}/{}'.format(dataroot, val)
        opt.classes  = '' #os.listdir(opt.dataroot) if multiclass[v_id] else ['']
        opt.no_resize = DetectionTests[testSet]['no_resize']
        opt.no_crop   = DetectionTests[testSet]['no_crop']
        acc, ap, _, _, _, _ = validate(model, opt)
        accs.append(acc);aps.append(ap)
        print("({} {:12}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
    print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 


for testSet in DetectionTests.keys():
    dataroot = DetectionTests[testSet]['dataroot']
    printSet(testSet)

    # Create a CSV file for this split
    results_folder = './results'
    os.makedirs(results_folder, exist_ok=True)
    result_csv = os.path.join(results_folder, f'{testSet}_per_image.csv')
    with open(result_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_path', 'true_label', 'score', 'pred_label'])

        accs = []
        aps = []
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

        # Enumerate each sub-folder under dataroot
        for v_id, val in enumerate(sorted(os.listdir(dataroot))):
            opt.dataroot = os.path.join(dataroot, val)
            opt.no_resize = DetectionTests[testSet]['no_resize']
            opt.no_crop   = DetectionTests[testSet]['no_crop']

            acc, ap, r_acc, f_acc, details, _ = validate(model, opt)
            accs.append(acc)
            aps.append(ap)

            print(f"({v_id:2d} {val:12s}) acc: {acc*100:.1f}; ap: {ap*100:.1f}")

            # Write per-image lines
            for (img_path, true_lbl, score, pred_lbl) in details:
                writer.writerow([img_path, true_lbl, f"{score:.4f}", pred_lbl])

        mean_acc = np.mean(accs)
        mean_ap = np.mean(aps)
        print(f"({v_id+1:2d} {'Mean':10s}) acc: {mean_acc*100:.1f}; ap: {mean_ap*100:.1f}")
        print('*' * 25)

    print(f"\nSaved per-image results for {testSet} → {result_csv}")
