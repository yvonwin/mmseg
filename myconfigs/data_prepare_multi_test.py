from glob import glob
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

"""
python mmsegmentation/myconfigs/data_prepare_multi.py
"""

Fold = 5

LABELS = './mmseg_multi_data/train.csv'
all_mask_files = glob("/home/zhaodao/HuBMAP/input/mmseg_multi_data/masks/*")
masks = []
num_mask = np.zeros((6,Fold))

df = pd.read_csv(LABELS)

for i in range(len(all_mask_files)):
    mask = cv2.imread(all_mask_files[i])
    masks.append(mask.max())

# split = list(StratifiedKFold(n_splits=Fold, shuffle=True, random_state=2022).split(all_mask_files, masks))
skf = KFold(n_splits=Fold,shuffle=True,random_state=42)
print(skf)

df.loc[:, 'fold'] = -1
for fold, (t_idx, v_idx) in enumerate(skf.split(X=df['id'], y=df['organ'])):
    df.iloc[v_idx, -1] = fold
    for i in v_idx:
        num_mask[masks[i]]+=1
    with open(f"./mmseg_multi_data/splits/fold_{fold}.txt", "w") as f:
        for idx in t_idx:
            f.write(os.path.basename(all_mask_files[idx])[:-4] + "\n")
    with open(f"./mmseg_multi_data/splits/valid_{fold}.txt", "w") as f:
        for idx in v_idx:
            f.write(os.path.basename(all_mask_files[idx])[:-4] + "\n")
print(num_mask)

# for fold, (train_idx, valid_idx) in enumerate(split):
#     for i in valid_idx:
#         num_mask[masks[i]]+=1
#     with open(f"./mmseg_multi_data/splits/fold_{fold}.txt", "w") as f:
#         for idx in train_idx:
#             f.write(os.path.basename(all_mask_files[idx])[:-4] + "\n")
#     with open(f"./mmseg_multi_data/splits/valid_{fold}.txt", "w") as f:
#         for idx in valid_idx:
#             f.write(os.path.basename(all_mask_files[idx])[:-4] + "\n")
# print(num_mask)