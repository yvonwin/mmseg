from glob import glob
import numpy as np
import cv2
import os
from sklearn.model_selection import StratifiedKFold


"""
python mmsegmentation/myconfigs/data_prepare_multi.py
"""

Fold = 5
all_mask_files = glob("/home/zhaodao/HuBMAP/input/mmseg_multi_data/masks/*")
masks = []
num_mask = np.zeros((6,Fold))

for i in range(len(all_mask_files)):
    mask = cv2.imread(all_mask_files[i])
    masks.append(mask.max())

split = list(StratifiedKFold(n_splits=Fold, shuffle=True, random_state=2022).split(all_mask_files, masks))
for fold, (train_idx, valid_idx) in enumerate(split):
    for i in valid_idx:
        num_mask[masks[i]]+=1
    with open(f"./mmseg_multi_data/splits/fold_{fold}.txt", "w") as f:
        for idx in train_idx:
            f.write(os.path.basename(all_mask_files[idx])[:-4] + "\n")
    with open(f"./mmseg_multi_data/splits/valid_{fold}.txt", "w") as f:
        for idx in valid_idx:
            f.write(os.path.basename(all_mask_files[idx])[:-4] + "\n")
print(num_mask)