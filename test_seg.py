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
lab_dir = '../BRATS2017-Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/MRI_seg'
im_list = os.listdir(img_dir)
image_size = 1024  #1024太大了显存受不了
with torch.no_grad():
    for img_name in tqdm(im_list):
        img_path = os.path.join(img_dir, img_name)
        lab_path = os.path.join(lab_dir, img_name)

        img_np = io.imread(img_path)
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np
        H, W, _ = img_3c.shape
        img_3c = transform.resize(
            img_3c, (image_size // 4, image_size // 4), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_3c = (img_3c - img_3c.min()) / np.clip(
            img_3c.max() - img_3c.min(), a_min=1e-8, a_max=None
        )
        img_3c = (
            torch.tensor(img_3c).float().permute(2, 0, 1).unsqueeze(0).to(device)
        )

        img_256 = torch.Tensor(img_3c).to(device)
        pre = medsam_model.SAM.cat_model(img_256)
        pre_seg = torch.argmax(pre, dim=1).cpu().numpy()[0]

        tub_num = img_name.split('_')[0]
        medsam_model.get_txt_encoder(img_path, tub_num=tub_num)
        wb_text = medsam_model.medsam_model
        # 使用文件的 write 方法将字符串保存到 txt 文件
        txt_name = f'../BRATS2017-Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/txt_q/{img_name.replace(".jpg")}'
        with open(txt_name, "w", encoding="utf-8") as f:
            f.write(wb_text)
        cv2.imwrite(f'../BRATS2017-Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/MRI_pre/{img_name}',pre_seg*(255/3))
