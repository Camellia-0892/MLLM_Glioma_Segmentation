import os
import cv2
import numpy as np
from medpy.metric.binary import hd, hd95
from tqdm import tqdm

# 设置路径
mask_dir = "D:\Bioproject\BRATS_TCGA_GBM_and_Segmentations\MRI_err"        # ground truth mask
pred_dir = "D:\Bioproject\BRATS_TCGA_GBM_and_Segmentations\MRI_pre" # predicted segmentation

output_file = 'D:\Bioproject\BRATS_TCGA_GBM_and_Segmentations\hd_results2.txt'

# 获取文件名列表
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.jpg')])
pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.jpg')])

# mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.jpg')])[:50]
# pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.jpg')])[:50]


dice_list = []
hd_list = []
hd95_list = []

with open(output_file, 'w') as f_out:
    f_out.write("Image\tDice\tHD\tHD95\n")
    for filename in tqdm(mask_files):
        mask_path = os.path.join(mask_dir, filename)
        pred_path = os.path.join(pred_dir, filename)

        if not os.path.exists(pred_path):
            print(f"Missing prediction: {filename}")
            continue

        # 读取图像为灰度图
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        # resize mask 到 pred 的尺寸
        if mask.shape != pred.shape:
            mask = cv2.resize(mask, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)

        # if pred.shape != mask.shape:
        #     pred = cv2.resize(pred, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 二值化
        mask_bin = (mask > 0).astype(np.bool_)
        pred_bin = (pred > 0).astype(np.bool_)

        dice_score = hd_value = hd95_value = float('nan')

        if np.sum(mask_bin) > 0 or np.sum(pred_bin) > 0:
            try:
                intersection = np.logical_and(mask_bin, pred_bin).sum()
                dice_score = 2 * intersection / (mask_bin.sum() + pred_bin.sum() + 1e-8)

                if np.sum(mask_bin) > 0 and np.sum(pred_bin) > 0:
                    hd_value = hd(pred_bin, mask_bin)
                    hd95_value = hd95(pred_bin, mask_bin)

            except Exception as e:
                print(f"Error on {filename}: {e}")

        f_out.write(f"{filename}\t{dice_score:.4f}\t{hd_value:.2f}\t{hd95_value:.2f}\n")

        # 收集非 nan 结果用于平均计算
        if not np.isnan(dice_score):
            dice_list.append(dice_score)
        if not np.isnan(hd_value):
            hd_list.append(hd_value)
        if not np.isnan(hd95_value):
            hd95_list.append(hd95_value)

    # 平均值写入
    f_out.write("\n")
    f_out.write(f"Mean\t{np.mean(dice_list):.4f}\t{np.mean(hd_list):.2f}\t{np.mean(hd95_list):.2f}\n")