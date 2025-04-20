import os

import cv2
import numpy as np
import torch
from skimage import io, transform
from tqdm import tqdm

from inference import MedSAM
from segment_anything import sam_model_registry

device='cuda'
sam_checkpoint = './work_dir/MedSAM-ViT-B-20250416-1422/medsam_model_latest.pth'
SAM = sam_model_registry["vit_b"](num_class=4, checkpoint=None)
medsam_model = MedSAM(
        SAM=SAM,
        sam_checkpoint=sam_checkpoint,
        device=device
    ).to(device)
medsam_model.eval()
img_dir = '../BRATS2017-Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/MRI'
im_list = os.listdir(img_dir)
with torch.no_grad():
    for img_name in tqdm(im_list[30:]):
        img_path = os.path.join(img_dir, img_name)
        tub_num = img_name.split('_')[0]
        medsam_model.get_txt_encoder(img_path, tub_num=tub_num)
        wb_text = medsam_model.wb_text
        # 使用文件的 write 方法将字符串保存到 txt 文件
        txt_name = f'../BRATS2017-Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/txt_q/{img_name.replace(".jpg",".txt")}'
        with open(txt_name, "w", encoding="utf-8") as f:
            f.write(wb_text)
