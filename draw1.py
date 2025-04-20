import os
import cv2
import numpy as np

# 设置路径
gt_mask_dir = "D:\Bioproject\BRATS_TCGA_GBM_and_Segmentations\MRI_seg"        # ground truth mask
pred_seg_dir = "D:\Bioproject\BRATS_TCGA_GBM_and_Segmentations\MRI_pre" # predicted segmentation
output_dir = "D:\Bioproject\BRATS_TCGA_GBM_and_Segmentations\MRI_err_tri2"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(gt_mask_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        gt_path = os.path.join(gt_mask_dir, filename)
        pred_path = os.path.join(pred_seg_dir, filename)

        if not os.path.exists(pred_path):
            print(f"Warning: {pred_path} not found, skipping.")
            continue

        # 加载灰度图
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        # 尺寸对齐：GT resize 到预测图大小
        if gt.shape != pred.shape:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 非零视为“有”
        gt_mask = (gt > 10).astype(np.uint8)
        pred_mask = (pred > 10).astype(np.uint8)

        # # 可选：开运算去除小噪声
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # gt_mask = cv2.morphologyEx(gt_mask, cv2.MORPH_OPEN, kernel)
        # pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)

        # 转换为彩色图
        vis = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)

        # 正确：不变
        correct_mask = (gt_mask == 1) & (pred_mask == 1)

        # 误检（False Positive）：预测有，GT 无 → 黄色
        false_positive = (gt_mask == 0) & (pred_mask == 1)
        vis[false_positive] = [0, 255, 255]  # 黄色

        # 漏检（False Negative）：GT 有，预测无 → 红色
        false_negative = (gt_mask == 1) & (pred_mask == 0)
        vis[false_negative] = [0, 0, 255]  # 红色

        # 保存图像
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, vis)
        print(f"Processed: {filename}")

print("✅ 所有图像处理完成，输出保存在：", output_dir)