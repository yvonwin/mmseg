import cv2
import numpy as np
import pandas as pd
import os
from glob import glob
# from tqdm.notebook import tqdm
from tqdm import tqdm
import sys
import gc
# sys.path.append('./mmseg')
# sys.path.append('./mmsegmentation')
from mmseg.apis import init_segmentor, inference_segmentor
from mmcv.utils import config
# import tifffile


import pandas as pd
# load model
# configs = [
#     '/content/drive/MyDrive/humap-weights/baseline/config.py',
# ]
# ckpts = [
#     '/content/drive/MyDrive/humap-weights/baseline/latest.pth',
# ]

configs = [
    '/home/zhaodao/HuBMAP/mmseg-mit-b2/segb2config_transform_test.py',
]
ckpts = [
    '/home/zhaodao/HuBMAP/mmseg-mit-b2/latest.pth',
    # '/home/zhaodao/HuBMAP/knet-swinl/iter_100000.pth',
    # '/content/drive/MyDrive/humap-weights/mmseg-mit-b2-new/iter_44000.pth',
]



DATA = '/home/zhaodao/HuBMAP/input/hubmap-organ-segmentation/test_images'
df_sample = pd.read_csv('/home/zhaodao/HuBMAP/input/hubmap-organ-segmentation/sample_submission.csv').set_index('id')

models = []
for idx,(cfg, ckpt) in enumerate(zip(configs, ckpts)):
    cfg = config.Config.fromfile(cfg)
    model = init_segmentor(cfg, ckpt, device='cuda:0')
    models.append(model)

def rle_encode_less_memory(img):
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# '''
# To ensemble models, you need to do some modifications with
# "../input/mmsegm/mmsegmentation-master/mmseg/models/segmentors/encoder_decoder.py"
# '''
names,preds = [],[]
imgs, pd_mks = [],[]
debug = len(df_sample)<2
for idx,row in tqdm(df_sample.iterrows(),total=len(df_sample)):
    img  = cv2.imread(os.path.join(DATA,str(idx)+'.tiff'))
    # img = cv2.imread('/home/zhaodao/HuBMAP/input/hubmap-organ-segmentation/train_images/1184.tiff')
    im_ = img.astype(np.float32)/img.max()
    # im_ = img

    print(im_.shape)
    pred = inference_segmentor(models[0], im_)[0]
    pred = (pred>0).astype(np.uint8)
    rle = rle_encode_less_memory(pred)
    print('rle is:',rle)
    names.append(str(idx))
    preds.append(rle)
    if debug:
        imgs.append(img)
        pd_mks.append(pred)
    del img, pred, rle, idx, row
    gc.collect()

if debug:
    import matplotlib.pyplot as plt
    for img, mask in zip(imgs, pd_mks):
        print(mask.shape)
        plt.figure(figsize=(12, 7))
        plt.subplot(1, 3, 1); plt.imshow(img); plt.axis('OFF'); plt.title('image')
        plt.subplot(1, 3, 2); plt.imshow(mask*255); plt.axis('OFF'); plt.title('mask')
        plt.subplot(1, 3, 3); plt.imshow(img); plt.imshow(mask*255, alpha=0.4); plt.axis('OFF'); plt.title('overlay')
        plt.tight_layout()
        plt.show()
