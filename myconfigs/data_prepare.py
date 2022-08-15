# 参考 https://www.kaggle.com/code/w3579628328/mmsegmentation-trainning

from glob import glob
import numpy as np
import cv2
import os
from sklearn.model_selection import StratifiedKFold
Fold = 5
all_mask_files = glob("/home/zhaodao/HuBMAP/input/mmseg_data/labels/*")
masks = []
num_wo_mask = np.zeros(Fold)
num_w_mask = np.zeros(Fold)
for i in range(len(all_mask_files)):
    mask = cv2.imread(all_mask_files[i])
    if mask.sum()==0:
        masks.append(0)
    else:
        masks.append(1)
split = list(StratifiedKFold(n_splits=Fold, shuffle=True, random_state=2022).split(all_mask_files, masks))
for fold, (train_idx, valid_idx) in enumerate(split):
    for i in valid_idx:
        if masks[i]==0:
            num_wo_mask[fold]+=1
        else:
            num_w_mask[fold]+=1
    with open(f"/home/zhaodao/HuBMAP/input/mmseg_data/splits/fold_{fold}.txt", "w") as f:
        for idx in train_idx:
            # print(idx)
            f.write(os.path.basename(all_mask_files[idx])[:-4] + "\n")
    with open(f"/home/zhaodao/HuBMAP/input/mmseg_data/splits/valid_{fold}.txt", "w") as f:
        for idx in valid_idx:
            f.write(os.path.basename(all_mask_files[idx])[:-4] + "\n")
print(num_wo_mask)
print(num_w_mask)