import os
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

# 设置路径
gt_mask_dir = "D:\Bioproject\BRATS_TCGA_GBM_and_Segmentations\MRI_seg"        # ground truth mask
pred_seg_dir = "D:\Bioproject\BRATS_TCGA_GBM_and_Segmentations\MRI_pre" # predicted segmentation
output_dir = "D:\Bioproject\BRATS_TCGA_GBM_and_Segmentations\MRI_err_tri2"

os.makedirs(output_dir, exist_ok=True)

distance_threshold = 3


# 距离阈值（像素）
for filename in os.listdir(gt_mask_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        gt_path = os.path.join(gt_mask_dir, filename)
        pred_path = os.path.join(pred_seg_dir, filename)

        if not os.path.exists(pred_path):
            print(f"Warning: {pred_path} not found, skipping.")
            continue

        # 读取灰度图
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        if gt.shape != pred.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        # 非零视为“有”
        gt_mask = (gt != 0).astype(np.uint8)
        pred_mask = (pred != 0).astype(np.uint8)

        # # 可选：开运算去除小噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gt_mask = cv2.morphologyEx(gt_mask, cv2.MORPH_OPEN, kernel)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)

        # 可视化图像：用预测图生成彩色底图
        vis = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)

        # 正确预测区域（保持不变）
        correct_mask = (gt_mask == 1) & (pred_mask == 1)

        # 漏检：GT有，预测无 → 红色
        false_negative = (gt_mask == 1) & (pred_mask == 0)
        vis[false_negative] = [0, 0, 255]

        # 误检（预测有，GT无）：全部先记录
        false_positive = (gt_mask == 0) & (pred_mask == 1)

        # 计算 GT 区域的距离图
        gt_inv = (gt_mask == 0).astype(np.uint8)
        dist_to_gt = distance_transform_edt(gt_inv)

        # 在误检区域中，离 GT 很近的 → 红色
        red_fp_mask = false_positive & (dist_to_gt <= distance_threshold)

        # 其余误检（远离 GT） → 黄色
        yellow_fp_mask = false_positive & (~red_fp_mask)

        # 上色（注意：红和黄互斥）
        vis[yellow_fp_mask] = [0, 255, 255]  # 黄色：远离 GT 的误检
        vis[red_fp_mask] = [0, 0, 255]       # 红色：靠近 GT 的误检

        # 保存结果
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, vis)
        print(f"Processed: {filename}")

print("✅ 全部图像处理完成，结果保存在：", output_dir)