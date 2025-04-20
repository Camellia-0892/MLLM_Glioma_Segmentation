import numpy as np
import matplotlib.pyplot as plt
import os
from torch import nn
join = os.path.join
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
from segment_anything.modeling import CMEncoder
import dashscope
import pandas as pd
from dashscope import TextEmbedding


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


class MedSAM(nn.Module):
    def __init__(
            self,
            SAM,
            sam_checkpoint,
            device='cuda'
    ):
        super().__init__()
        self.image_encoder = SAM.image_encoder
        self.mask_decoder = SAM.mask_decoder
        self.prompt_encoder = SAM.prompt_encoder
        self.SAM = SAM
        self.CMEncoder = CMEncoder(in_channel=256, norm_dim=256, mlp_dim=256, mlp_hid_dim=512).to(device)  # 融合模块

        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        self.device = device
        checkpoint = torch.load(sam_checkpoint, map_location=device)
        self.load_state_dict(checkpoint["model"], strict=False)
        self.SAM.cat_model.load_state_dict(checkpoint["seg_c"])
        # 设置阿里云API Key（需到阿里云控制台获取）
        self.api_key = 'sk-59078b9cd8a0405eb6e7078877b93b0c'


        self.title = "ID,Date,VOLUME_ET,VOLUME_NET,VOLUME_ED,VOLUME_TC,VOLUME_WT,VOLUME_BRAIN,VOLUME_ET_OVER_NET,VOLUME_ET_OVER_ED,VOLUME_NET_OVER_ED,VOLUME_ET_over_TC,VOLUME_NET_over_TC,VOLUME_ED_over_TC,VOLUME_ET_OVER_WT,VOLUME_NET_OVER_WT,VOLUME_ED_OVER_WT,VOLUME_TC_OVER_WT,VOLUME_ET_OVER_BRAIN,VOLUME_NET_OVER_BRAIN,VOLUME_ED_over_BRAIN,VOLUME_TC_over_BRAIN,VOLUME_WT_OVER_BRAIN,DIST_Vent_TC,DIST_Vent_ED,INTENSITY_Mean_ET_T1Gd,INTENSITY_STD_ET_T1Gd,INTENSITY_Mean_ET_T1,INTENSITY_STD_ET_T1,INTENSITY_Mean_ET_T2,INTENSITY_STD_ET_T2,INTENSITY_Mean_ET_FLAIR,INTENSITY_STD_ET_FLAIR,INTENSITY_Mean_NET_T1Gd,INTENSITY_STD_NET_T1Gd,INTENSITY_Mean_NET_T1,INTENSITY_STD_NET_T1,INTENSITY_Mean_NET_T2,INTENSITY_STD_NET_T2,INTENSITY_Mean_NET_FLAIR,INTENSITY_STD_NET_FLAIR,INTENSITY_Mean_ED_T1Gd,INTENSITY_STD_ED_T1Gd,INTENSITY_Mean_ED_T1,INTENSITY_STD_ED_T1,INTENSITY_Mean_ED_T2,INTENSITY_STD_ED_T2,INTENSITY_Mean_ED_FLAIR,INTENSITY_STD_ED_FLAIR,HISTO_ET_T1Gd_Bin1,HISTO_ET_T1Gd_Bin2,HISTO_ET_T1Gd_Bin3,HISTO_ET_T1Gd_Bin4,HISTO_ET_T1Gd_Bin5,HISTO_ET_T1Gd_Bin6,HISTO_ET_T1Gd_Bin7,HISTO_ET_T1Gd_Bin8,HISTO_ET_T1Gd_Bin9,HISTO_ET_T1Gd_Bin10,HISTO_ED_T1Gd_Bin1,HISTO_ED_T1Gd_Bin2,HISTO_ED_T1Gd_Bin3,HISTO_ED_T1Gd_Bin4,HISTO_ED_T1Gd_Bin5,HISTO_ED_T1Gd_Bin6,HISTO_ED_T1Gd_Bin7,HISTO_ED_T1Gd_Bin8,HISTO_ED_T1Gd_Bin9,HISTO_ED_T1Gd_Bin10,HISTO_NET_T1Gd_Bin1,HISTO_NET_T1Gd_Bin2,HISTO_NET_T1Gd_Bin3,HISTO_NET_T1Gd_Bin4,HISTO_NET_T1Gd_Bin5,HISTO_NET_T1Gd_Bin6,HISTO_NET_T1Gd_Bin7,HISTO_NET_T1Gd_Bin8,HISTO_NET_T1Gd_Bin9,HISTO_NET_T1Gd_Bin10,HISTO_ET_T1_Bin1,HISTO_ET_T1_Bin2,HISTO_ET_T1_Bin3,HISTO_ET_T1_Bin4,HISTO_ET_T1_Bin5,HISTO_ET_T1_Bin6,HISTO_ET_T1_Bin7,HISTO_ET_T1_Bin8,HISTO_ET_T1_Bin9,HISTO_ET_T1_Bin10,HISTO_ED_T1_Bin1,HISTO_ED_T1_Bin2,HISTO_ED_T1_Bin3,HISTO_ED_T1_Bin4,HISTO_ED_T1_Bin5,HISTO_ED_T1_Bin6,HISTO_ED_T1_Bin7,HISTO_ED_T1_Bin8,HISTO_ED_T1_Bin9,HISTO_ED_T1_Bin10,HISTO_NET_T1_Bin1,HISTO_NET_T1_Bin2,HISTO_NET_T1_Bin3,HISTO_NET_T1_Bin4,HISTO_NET_T1_Bin5,HISTO_NET_T1_Bin6,HISTO_NET_T1_Bin7,HISTO_NET_T1_Bin8,HISTO_NET_T1_Bin9,HISTO_NET_T1_Bin10,HISTO_ET_T2_Bin1,HISTO_ET_T2_Bin2,HISTO_ET_T2_Bin3,HISTO_ET_T2_Bin4,HISTO_ET_T2_Bin5,HISTO_ET_T2_Bin6,HISTO_ET_T2_Bin7,HISTO_ET_T2_Bin8,HISTO_ET_T2_Bin9,HISTO_ET_T2_Bin10,HISTO_ED_T2_Bin1,HISTO_ED_T2_Bin2,HISTO_ED_T2_Bin3,HISTO_ED_T2_Bin4,HISTO_ED_T2_Bin5,HISTO_ED_T2_Bin6,HISTO_ED_T2_Bin7,HISTO_ED_T2_Bin8,HISTO_ED_T2_Bin9,HISTO_ED_T2_Bin10,HISTO_NET_T2_Bin1,HISTO_NET_T2_Bin2,HISTO_NET_T2_Bin3,HISTO_NET_T2_Bin4,HISTO_NET_T2_Bin5,HISTO_NET_T2_Bin6,HISTO_NET_T2_Bin7,HISTO_NET_T2_Bin8,HISTO_NET_T2_Bin9,HISTO_NET_T2_Bin10,HISTO_ET_FLAIR_Bin1,HISTO_ET_FLAIR_Bin2,HISTO_ET_FLAIR_Bin3,HISTO_ET_FLAIR_Bin4,HISTO_ET_FLAIR_Bin5,HISTO_ET_FLAIR_Bin6,HISTO_ET_FLAIR_Bin7,HISTO_ET_FLAIR_Bin8,HISTO_ET_FLAIR_Bin9,HISTO_ET_FLAIR_Bin10,HISTO_ED_FLAIR_Bin1,HISTO_ED_FLAIR_Bin2,HISTO_ED_FLAIR_Bin3,HISTO_ED_FLAIR_Bin4,HISTO_ED_FLAIR_Bin5,HISTO_ED_FLAIR_Bin6,HISTO_ED_FLAIR_Bin7,HISTO_ED_FLAIR_Bin8,HISTO_ED_FLAIR_Bin9,HISTO_ED_FLAIR_Bin10,HISTO_NET_FLAIR_Bin1,HISTO_NET_FLAIR_Bin2,HISTO_NET_FLAIR_Bin3,HISTO_NET_FLAIR_Bin4,HISTO_NET_FLAIR_Bin5,HISTO_NET_FLAIR_Bin6,HISTO_NET_FLAIR_Bin7,HISTO_NET_FLAIR_Bin8,HISTO_NET_FLAIR_Bin9,HISTO_NET_FLAIR_Bin10,SPATIAL_Frontal,SPATIAL_Temporal,SPATIAL_Parietal,SPATIAL_Basal_G,SPATIAL_Insula,SPATIAL_CC_Fornix,SPATIAL_Occipital,SPATIAL_Cere,SPATIAL_Brain_stem,ECCENTRICITY_ET,ECCENTRICITY_ED,ECCENTRICITY_NET,SOLIDITY_ET,SOLIDITY_ED,SOLIDITY_NET,TEXTURE_GLOBAL_ET_T1Gd_Variance,TEXTURE_GLOBAL_ET_T1Gd_Skewness,TEXTURE_GLOBAL_ET_T1Gd_Kurtosis,TEXTURE_GLOBAL_ET_T1_Variance,TEXTURE_GLOBAL_ET_T1_Skewness,TEXTURE_GLOBAL_ET_T1_Kurtosis,TEXTURE_GLOBAL_ET_T2_Variance,TEXTURE_GLOBAL_ET_T2_Skewness,TEXTURE_GLOBAL_ET_T2_Kurtosis,TEXTURE_GLOBAL_ET_FLAIR_Variance,TEXTURE_GLOBAL_ET_FLAIR_Skewness,TEXTURE_GLOBAL_ET_FLAIR_Kurtosis,TEXTURE_GLOBAL_ED_T1Gd_Variance,TEXTURE_GLOBAL_ED_T1Gd_Skewness,TEXTURE_GLOBAL_ED_T1Gd_Kurtosis,TEXTURE_GLOBAL_ED_T1_Variance,TEXTURE_GLOBAL_ED_T1_Skewness,TEXTURE_GLOBAL_ED_T1_Kurtosis,TEXTURE_GLOBAL_ED_T2_Variance,TEXTURE_GLOBAL_ED_T2_Skewness,TEXTURE_GLOBAL_ED_T2_Kurtosis,TEXTURE_GLOBAL_ED_FLAIR_Variance,TEXTURE_GLOBAL_ED_FLAIR_Skewness,TEXTURE_GLOBAL_ED_FLAIR_Kurtosis,TEXTURE_GLOBAL_NET_T1Gd_Variance,TEXTURE_GLOBAL_NET_T1Gd_Skewness,TEXTURE_GLOBAL_NET_T1Gd_Kurtosis,TEXTURE_GLOBAL_NET_T1_Variance,TEXTURE_GLOBAL_NET_T1_Skewness,TEXTURE_GLOBAL_NET_T1_Kurtosis,TEXTURE_GLOBAL_NET_T2_Variance,TEXTURE_GLOBAL_NET_T2_Skewness,TEXTURE_GLOBAL_NET_T2_Kurtosis,TEXTURE_GLOBAL_NET_FLAIR_Variance,TEXTURE_GLOBAL_NET_FLAIR_Skewness,TEXTURE_GLOBAL_NET_FLAIR_Kurtosis,TEXTURE_GLCM_ET_T1Gd_Energy,TEXTURE_GLCM_ET_T1Gd_Contrast,TEXTURE_GLCM_ET_T1Gd_Entropy,TEXTURE_GLCM_ET_T1Gd_Homogeneity,TEXTURE_GLCM_ET_T1Gd_Correlation,TEXTURE_GLCM_ET_T1Gd_SumAverage,TEXTURE_GLCM_ET_T1Gd_Variance,TEXTURE_GLCM_ET_T1Gd_Dissimilarity,TEXTURE_GLCM_ET_T1Gd_AutoCorrelation,TEXTURE_GLCM_ET_T1_Energy,TEXTURE_GLCM_ET_T1_Contrast,TEXTURE_GLCM_ET_T1_Entropy,TEXTURE_GLCM_ET_T1_Homogeneity,TEXTURE_GLCM_ET_T1_Correlation,TEXTURE_GLCM_ET_T1_SumAverage,TEXTURE_GLCM_ET_T1_Variance,TEXTURE_GLCM_ET_T1_Dissimilarity,TEXTURE_GLCM_ET_T1_AutoCorrelation,TEXTURE_GLCM_ET_T2_Energy,TEXTURE_GLCM_ET_T2_Contrast,TEXTURE_GLCM_ET_T2_Entropy,TEXTURE_GLCM_ET_T2_Homogeneity,TEXTURE_GLCM_ET_T2_Correlation,TEXTURE_GLCM_ET_T2_SumAverage,TEXTURE_GLCM_ET_T2_Variance,TEXTURE_GLCM_ET_T2_Dissimilarity,TEXTURE_GLCM_ET_T2_AutoCorrelation,TEXTURE_GLCM_ET_FLAIR_Energy,TEXTURE_GLCM_ET_FLAIR_Contrast,TEXTURE_GLCM_ET_FLAIR_Entropy,TEXTURE_GLCM_ET_FLAIR_Homogeneity,TEXTURE_GLCM_ET_FLAIR_Correlation,TEXTURE_GLCM_ET_FLAIR_SumAverage,TEXTURE_GLCM_ET_FLAIR_Variance,TEXTURE_GLCM_ET_FLAIR_Dissimilarity,TEXTURE_GLCM_ET_FLAIR_AutoCorrelation,TEXTURE_GLCM_ED_T1Gd_Energy,TEXTURE_GLCM_ED_T1Gd_Contrast,TEXTURE_GLCM_ED_T1Gd_Entropy,TEXTURE_GLCM_ED_T1Gd_Homogeneity,TEXTURE_GLCM_ED_T1Gd_Correlation,TEXTURE_GLCM_ED_T1Gd_SumAverage,TEXTURE_GLCM_ED_T1Gd_Variance,TEXTURE_GLCM_ED_T1Gd_Dissimilarity,TEXTURE_GLCM_ED_T1Gd_AutoCorrelation,TEXTURE_GLCM_ED_T1_Energy,TEXTURE_GLCM_ED_T1_Contrast,TEXTURE_GLCM_ED_T1_Entropy,TEXTURE_GLCM_ED_T1_Homogeneity,TEXTURE_GLCM_ED_T1_Correlation,TEXTURE_GLCM_ED_T1_SumAverage,TEXTURE_GLCM_ED_T1_Variance,TEXTURE_GLCM_ED_T1_Dissimilarity,TEXTURE_GLCM_ED_T1_AutoCorrelation,TEXTURE_GLCM_ED_T2_Energy,TEXTURE_GLCM_ED_T2_Contrast,TEXTURE_GLCM_ED_T2_Entropy,TEXTURE_GLCM_ED_T2_Homogeneity,TEXTURE_GLCM_ED_T2_Correlation,TEXTURE_GLCM_ED_T2_SumAverage,TEXTURE_GLCM_ED_T2_Variance,TEXTURE_GLCM_ED_T2_Dissimilarity,TEXTURE_GLCM_ED_T2_AutoCorrelation,TEXTURE_GLCM_ED_FLAIR_Energy,TEXTURE_GLCM_ED_FLAIR_Contrast,TEXTURE_GLCM_ED_FLAIR_Entropy,TEXTURE_GLCM_ED_FLAIR_Homogeneity,TEXTURE_GLCM_ED_FLAIR_Correlation,TEXTURE_GLCM_ED_FLAIR_SumAverage,TEXTURE_GLCM_ED_FLAIR_Variance,TEXTURE_GLCM_ED_FLAIR_Dissimilarity,TEXTURE_GLCM_ED_FLAIR_AutoCorrelation,TEXTURE_GLCM_NET_T1Gd_Energy,TEXTURE_GLCM_NET_T1Gd_Contrast,TEXTURE_GLCM_NET_T1Gd_Entropy,TEXTURE_GLCM_NET_T1Gd_Homogeneity,TEXTURE_GLCM_NET_T1Gd_Correlation,TEXTURE_GLCM_NET_T1Gd_SumAverage,TEXTURE_GLCM_NET_T1Gd_Variance,TEXTURE_GLCM_NET_T1Gd_Dissimilarity,TEXTURE_GLCM_NET_T1Gd_AutoCorrelation,TEXTURE_GLCM_NET_T1_Energy,TEXTURE_GLCM_NET_T1_Contrast,TEXTURE_GLCM_NET_T1_Entropy,TEXTURE_GLCM_NET_T1_Homogeneity,TEXTURE_GLCM_NET_T1_Correlation,TEXTURE_GLCM_NET_T1_SumAverage,TEXTURE_GLCM_NET_T1_Variance,TEXTURE_GLCM_NET_T1_Dissimilarity,TEXTURE_GLCM_NET_T1_AutoCorrelation,TEXTURE_GLCM_NET_T2_Energy,TEXTURE_GLCM_NET_T2_Contrast,TEXTURE_GLCM_NET_T2_Entropy,TEXTURE_GLCM_NET_T2_Homogeneity,TEXTURE_GLCM_NET_T2_Correlation,TEXTURE_GLCM_NET_T2_SumAverage,TEXTURE_GLCM_NET_T2_Variance,TEXTURE_GLCM_NET_T2_Dissimilarity,TEXTURE_GLCM_NET_T2_AutoCorrelation,TEXTURE_GLCM_NET_FLAIR_Energy,TEXTURE_GLCM_NET_FLAIR_Contrast,TEXTURE_GLCM_NET_FLAIR_Entropy,TEXTURE_GLCM_NET_FLAIR_Homogeneity,TEXTURE_GLCM_NET_FLAIR_Correlation,TEXTURE_GLCM_NET_FLAIR_SumAverage,TEXTURE_GLCM_NET_FLAIR_Variance,TEXTURE_GLCM_NET_FLAIR_Dissimilarity,TEXTURE_GLCM_NET_FLAIR_AutoCorrelation,TEXTURE_GLRLM_ET_T1Gd_SRE,TEXTURE_GLRLM_ET_T1Gd_LRE,TEXTURE_GLRLM_ET_T1Gd_GLN,TEXTURE_GLRLM_ET_T1Gd_RLN,TEXTURE_GLRLM_ET_T1Gd_RP,TEXTURE_GLRLM_ET_T1Gd_LGRE,TEXTURE_GLRLM_ET_T1Gd_HGRE,TEXTURE_GLRLM_ET_T1Gd_SRLGE,TEXTURE_GLRLM_ET_T1Gd_SRHGE,TEXTURE_GLRLM_ET_T1Gd_LRLGE,TEXTURE_GLRLM_ET_T1Gd_LRHGE,TEXTURE_GLRLM_ET_T1Gd_GLV,TEXTURE_GLRLM_ET_T1Gd_RLV,TEXTURE_GLRLM_ET_T1_SRE,TEXTURE_GLRLM_ET_T1_LRE,TEXTURE_GLRLM_ET_T1_GLN,TEXTURE_GLRLM_ET_T1_RLN,TEXTURE_GLRLM_ET_T1_RP,TEXTURE_GLRLM_ET_T1_LGRE,TEXTURE_GLRLM_ET_T1_HGRE,TEXTURE_GLRLM_ET_T1_SRLGE,TEXTURE_GLRLM_ET_T1_SRHGE,TEXTURE_GLRLM_ET_T1_LRLGE,TEXTURE_GLRLM_ET_T1_LRHGE,TEXTURE_GLRLM_ET_T1_GLV,TEXTURE_GLRLM_ET_T1_RLV,TEXTURE_GLRLM_ET_T2_SRE,TEXTURE_GLRLM_ET_T2_LRE,TEXTURE_GLRLM_ET_T2_GLN,TEXTURE_GLRLM_ET_T2_RLN,TEXTURE_GLRLM_ET_T2_RP,TEXTURE_GLRLM_ET_T2_LGRE,TEXTURE_GLRLM_ET_T2_HGRE,TEXTURE_GLRLM_ET_T2_SRLGE,TEXTURE_GLRLM_ET_T2_SRHGE,TEXTURE_GLRLM_ET_T2_LRLGE,TEXTURE_GLRLM_ET_T2_LRHGE,TEXTURE_GLRLM_ET_T2_GLV,TEXTURE_GLRLM_ET_T2_RLV,TEXTURE_GLRLM_ET_FLAIR_SRE,TEXTURE_GLRLM_ET_FLAIR_LRE,TEXTURE_GLRLM_ET_FLAIR_GLN,TEXTURE_GLRLM_ET_FLAIR_RLN,TEXTURE_GLRLM_ET_FLAIR_RP,TEXTURE_GLRLM_ET_FLAIR_LGRE,TEXTURE_GLRLM_ET_FLAIR_HGRE,TEXTURE_GLRLM_ET_FLAIR_SRLGE,TEXTURE_GLRLM_ET_FLAIR_SRHGE,TEXTURE_GLRLM_ET_FLAIR_LRLGE,TEXTURE_GLRLM_ET_FLAIR_LRHGE,TEXTURE_GLRLM_ET_FLAIR_GLV,TEXTURE_GLRLM_ET_FLAIR_RLV,TEXTURE_GLRLM_ED_T1Gd_SRE,TEXTURE_GLRLM_ED_T1Gd_LRE,TEXTURE_GLRLM_ED_T1Gd_GLN,TEXTURE_GLRLM_ED_T1Gd_RLN,TEXTURE_GLRLM_ED_T1Gd_RP,TEXTURE_GLRLM_ED_T1Gd_LGRE,TEXTURE_GLRLM_ED_T1Gd_HGRE,TEXTURE_GLRLM_ED_T1Gd_SRLGE,TEXTURE_GLRLM_ED_T1Gd_SRHGE,TEXTURE_GLRLM_ED_T1Gd_LRLGE,TEXTURE_GLRLM_ED_T1Gd_LRHGE,TEXTURE_GLRLM_ED_T1Gd_GLV,TEXTURE_GLRLM_ED_T1Gd_RLV,TEXTURE_GLRLM_ED_T1_SRE,TEXTURE_GLRLM_ED_T1_LRE,TEXTURE_GLRLM_ED_T1_GLN,TEXTURE_GLRLM_ED_T1_RLN,TEXTURE_GLRLM_ED_T1_RP,TEXTURE_GLRLM_ED_T1_LGRE,TEXTURE_GLRLM_ED_T1_HGRE,TEXTURE_GLRLM_ED_T1_SRLGE,TEXTURE_GLRLM_ED_T1_SRHGE,TEXTURE_GLRLM_ED_T1_LRLGE,TEXTURE_GLRLM_ED_T1_LRHGE,TEXTURE_GLRLM_ED_T1_GLV,TEXTURE_GLRLM_ED_T1_RLV,TEXTURE_GLRLM_ED_T2_SRE,TEXTURE_GLRLM_ED_T2_LRE,TEXTURE_GLRLM_ED_T2_GLN,TEXTURE_GLRLM_ED_T2_RLN,TEXTURE_GLRLM_ED_T2_RP,TEXTURE_GLRLM_ED_T2_LGRE,TEXTURE_GLRLM_ED_T2_HGRE,TEXTURE_GLRLM_ED_T2_SRLGE,TEXTURE_GLRLM_ED_T2_SRHGE,TEXTURE_GLRLM_ED_T2_LRLGE,TEXTURE_GLRLM_ED_T2_LRHGE,TEXTURE_GLRLM_ED_T2_GLV,TEXTURE_GLRLM_ED_T2_RLV,TEXTURE_GLRLM_ED_FLAIR_SRE,TEXTURE_GLRLM_ED_FLAIR_LRE,TEXTURE_GLRLM_ED_FLAIR_GLN,TEXTURE_GLRLM_ED_FLAIR_RLN,TEXTURE_GLRLM_ED_FLAIR_RP,TEXTURE_GLRLM_ED_FLAIR_LGRE,TEXTURE_GLRLM_ED_FLAIR_HGRE,TEXTURE_GLRLM_ED_FLAIR_SRLGE,TEXTURE_GLRLM_ED_FLAIR_SRHGE,TEXTURE_GLRLM_ED_FLAIR_LRLGE,TEXTURE_GLRLM_ED_FLAIR_LRHGE,TEXTURE_GLRLM_ED_FLAIR_GLV,TEXTURE_GLRLM_ED_FLAIR_RLV,TEXTURE_GLRLM_NET_T1Gd_SRE,TEXTURE_GLRLM_NET_T1Gd_LRE,TEXTURE_GLRLM_NET_T1Gd_GLN,TEXTURE_GLRLM_NET_T1Gd_RLN,TEXTURE_GLRLM_NET_T1Gd_RP,TEXTURE_GLRLM_NET_T1Gd_LGRE,TEXTURE_GLRLM_NET_T1Gd_HGRE,TEXTURE_GLRLM_NET_T1Gd_SRLGE,TEXTURE_GLRLM_NET_T1Gd_SRHGE,TEXTURE_GLRLM_NET_T1Gd_LRLGE,TEXTURE_GLRLM_NET_T1Gd_LRHGE,TEXTURE_GLRLM_NET_T1Gd_GLV,TEXTURE_GLRLM_NET_T1Gd_RLV,TEXTURE_GLRLM_NET_T1_SRE,TEXTURE_GLRLM_NET_T1_LRE,TEXTURE_GLRLM_NET_T1_GLN,TEXTURE_GLRLM_NET_T1_RLN,TEXTURE_GLRLM_NET_T1_RP,TEXTURE_GLRLM_NET_T1_LGRE,TEXTURE_GLRLM_NET_T1_HGRE,TEXTURE_GLRLM_NET_T1_SRLGE,TEXTURE_GLRLM_NET_T1_SRHGE,TEXTURE_GLRLM_NET_T1_LRLGE,TEXTURE_GLRLM_NET_T1_LRHGE,TEXTURE_GLRLM_NET_T1_GLV,TEXTURE_GLRLM_NET_T1_RLV,TEXTURE_GLRLM_NET_T2_SRE,TEXTURE_GLRLM_NET_T2_LRE,TEXTURE_GLRLM_NET_T2_GLN,TEXTURE_GLRLM_NET_T2_RLN,TEXTURE_GLRLM_NET_T2_RP,TEXTURE_GLRLM_NET_T2_LGRE,TEXTURE_GLRLM_NET_T2_HGRE,TEXTURE_GLRLM_NET_T2_SRLGE,TEXTURE_GLRLM_NET_T2_SRHGE,TEXTURE_GLRLM_NET_T2_LRLGE,TEXTURE_GLRLM_NET_T2_LRHGE,TEXTURE_GLRLM_NET_T2_GLV,TEXTURE_GLRLM_NET_T2_RLV,TEXTURE_GLRLM_NET_FLAIR_SRE,TEXTURE_GLRLM_NET_FLAIR_LRE,TEXTURE_GLRLM_NET_FLAIR_GLN,TEXTURE_GLRLM_NET_FLAIR_RLN,TEXTURE_GLRLM_NET_FLAIR_RP,TEXTURE_GLRLM_NET_FLAIR_LGRE,TEXTURE_GLRLM_NET_FLAIR_HGRE,TEXTURE_GLRLM_NET_FLAIR_SRLGE,TEXTURE_GLRLM_NET_FLAIR_SRHGE,TEXTURE_GLRLM_NET_FLAIR_LRLGE,TEXTURE_GLRLM_NET_FLAIR_LRHGE,TEXTURE_GLRLM_NET_FLAIR_GLV,TEXTURE_GLRLM_NET_FLAIR_RLV,TEXTURE_GLSZM_ET_T1Gd_SZE,TEXTURE_GLSZM_ET_T1Gd_LZE,TEXTURE_GLSZM_ET_T1Gd_GLN,TEXTURE_GLSZM_ET_T1Gd_ZSN,TEXTURE_GLSZM_ET_T1Gd_ZP,TEXTURE_GLSZM_ET_T1Gd_LGZE,TEXTURE_GLSZM_ET_T1Gd_HGZE,TEXTURE_GLSZM_ET_T1Gd_SZLGE,TEXTURE_GLSZM_ET_T1Gd_SZHGE,TEXTURE_GLSZM_ET_T1Gd_LZLGE,TEXTURE_GLSZM_ET_T1Gd_LZHGE,TEXTURE_GLSZM_ET_T1Gd_GLV,TEXTURE_GLSZM_ET_T1Gd_ZSV,TEXTURE_GLSZM_ET_T1_SZE,TEXTURE_GLSZM_ET_T1_LZE,TEXTURE_GLSZM_ET_T1_GLN,TEXTURE_GLSZM_ET_T1_ZSN,TEXTURE_GLSZM_ET_T1_ZP,TEXTURE_GLSZM_ET_T1_LGZE,TEXTURE_GLSZM_ET_T1_HGZE,TEXTURE_GLSZM_ET_T1_SZLGE,TEXTURE_GLSZM_ET_T1_SZHGE,TEXTURE_GLSZM_ET_T1_LZLGE,TEXTURE_GLSZM_ET_T1_LZHGE,TEXTURE_GLSZM_ET_T1_GLV,TEXTURE_GLSZM_ET_T1_ZSV,TEXTURE_GLSZM_ET_T2_SZE,TEXTURE_GLSZM_ET_T2_LZE,TEXTURE_GLSZM_ET_T2_GLN,TEXTURE_GLSZM_ET_T2_ZSN,TEXTURE_GLSZM_ET_T2_ZP,TEXTURE_GLSZM_ET_T2_LGZE,TEXTURE_GLSZM_ET_T2_HGZE,TEXTURE_GLSZM_ET_T2_SZLGE,TEXTURE_GLSZM_ET_T2_SZHGE,TEXTURE_GLSZM_ET_T2_LZLGE,TEXTURE_GLSZM_ET_T2_LZHGE,TEXTURE_GLSZM_ET_T2_GLV,TEXTURE_GLSZM_ET_T2_ZSV,TEXTURE_GLSZM_ET_FLAIR_SZE,TEXTURE_GLSZM_ET_FLAIR_LZE,TEXTURE_GLSZM_ET_FLAIR_GLN,TEXTURE_GLSZM_ET_FLAIR_ZSN,TEXTURE_GLSZM_ET_FLAIR_ZP,TEXTURE_GLSZM_ET_FLAIR_LGZE,TEXTURE_GLSZM_ET_FLAIR_HGZE,TEXTURE_GLSZM_ET_FLAIR_SZLGE,TEXTURE_GLSZM_ET_FLAIR_SZHGE,TEXTURE_GLSZM_ET_FLAIR_LZLGE,TEXTURE_GLSZM_ET_FLAIR_LZHGE,TEXTURE_GLSZM_ET_FLAIR_GLV,TEXTURE_GLSZM_ET_FLAIR_ZSV,TEXTURE_GLSZM_ED_T1Gd_SZE,TEXTURE_GLSZM_ED_T1Gd_LZE,TEXTURE_GLSZM_ED_T1Gd_GLN,TEXTURE_GLSZM_ED_T1Gd_ZSN,TEXTURE_GLSZM_ED_T1Gd_ZP,TEXTURE_GLSZM_ED_T1Gd_LGZE,TEXTURE_GLSZM_ED_T1Gd_HGZE,TEXTURE_GLSZM_ED_T1Gd_SZLGE,TEXTURE_GLSZM_ED_T1Gd_SZHGE,TEXTURE_GLSZM_ED_T1Gd_LZLGE,TEXTURE_GLSZM_ED_T1Gd_LZHGE,TEXTURE_GLSZM_ED_T1Gd_GLV,TEXTURE_GLSZM_ED_T1Gd_ZSV,TEXTURE_GLSZM_ED_T1_SZE,TEXTURE_GLSZM_ED_T1_LZE,TEXTURE_GLSZM_ED_T1_GLN,TEXTURE_GLSZM_ED_T1_ZSN,TEXTURE_GLSZM_ED_T1_ZP,TEXTURE_GLSZM_ED_T1_LGZE,TEXTURE_GLSZM_ED_T1_HGZE,TEXTURE_GLSZM_ED_T1_SZLGE,TEXTURE_GLSZM_ED_T1_SZHGE,TEXTURE_GLSZM_ED_T1_LZLGE,TEXTURE_GLSZM_ED_T1_LZHGE,TEXTURE_GLSZM_ED_T1_GLV,TEXTURE_GLSZM_ED_T1_ZSV,TEXTURE_GLSZM_ED_T2_SZE,TEXTURE_GLSZM_ED_T2_LZE,TEXTURE_GLSZM_ED_T2_GLN,TEXTURE_GLSZM_ED_T2_ZSN,TEXTURE_GLSZM_ED_T2_ZP,TEXTURE_GLSZM_ED_T2_LGZE,TEXTURE_GLSZM_ED_T2_HGZE,TEXTURE_GLSZM_ED_T2_SZLGE,TEXTURE_GLSZM_ED_T2_SZHGE,TEXTURE_GLSZM_ED_T2_LZLGE,TEXTURE_GLSZM_ED_T2_LZHGE,TEXTURE_GLSZM_ED_T2_GLV,TEXTURE_GLSZM_ED_T2_ZSV,TEXTURE_GLSZM_ED_FLAIR_SZE,TEXTURE_GLSZM_ED_FLAIR_LZE,TEXTURE_GLSZM_ED_FLAIR_GLN,TEXTURE_GLSZM_ED_FLAIR_ZSN,TEXTURE_GLSZM_ED_FLAIR_ZP,TEXTURE_GLSZM_ED_FLAIR_LGZE,TEXTURE_GLSZM_ED_FLAIR_HGZE,TEXTURE_GLSZM_ED_FLAIR_SZLGE,TEXTURE_GLSZM_ED_FLAIR_SZHGE,TEXTURE_GLSZM_ED_FLAIR_LZLGE,TEXTURE_GLSZM_ED_FLAIR_LZHGE,TEXTURE_GLSZM_ED_FLAIR_GLV,TEXTURE_GLSZM_ED_FLAIR_ZSV,TEXTURE_GLSZM_NET_T1Gd_SZE,TEXTURE_GLSZM_NET_T1Gd_LZE,TEXTURE_GLSZM_NET_T1Gd_GLN,TEXTURE_GLSZM_NET_T1Gd_ZSN,TEXTURE_GLSZM_NET_T1Gd_ZP,TEXTURE_GLSZM_NET_T1Gd_LGZE,TEXTURE_GLSZM_NET_T1Gd_HGZE,TEXTURE_GLSZM_NET_T1Gd_SZLGE,TEXTURE_GLSZM_NET_T1Gd_SZHGE,TEXTURE_GLSZM_NET_T1Gd_LZLGE,TEXTURE_GLSZM_NET_T1Gd_LZHGE,TEXTURE_GLSZM_NET_T1Gd_GLV,TEXTURE_GLSZM_NET_T1Gd_ZSV,TEXTURE_GLSZM_NET_T1_SZE,TEXTURE_GLSZM_NET_T1_LZE,TEXTURE_GLSZM_NET_T1_GLN,TEXTURE_GLSZM_NET_T1_ZSN,TEXTURE_GLSZM_NET_T1_ZP,TEXTURE_GLSZM_NET_T1_LGZE,TEXTURE_GLSZM_NET_T1_HGZE,TEXTURE_GLSZM_NET_T1_SZLGE,TEXTURE_GLSZM_NET_T1_SZHGE,TEXTURE_GLSZM_NET_T1_LZLGE,TEXTURE_GLSZM_NET_T1_LZHGE,TEXTURE_GLSZM_NET_T1_GLV,TEXTURE_GLSZM_NET_T1_ZSV,TEXTURE_GLSZM_NET_T2_SZE,TEXTURE_GLSZM_NET_T2_LZE,TEXTURE_GLSZM_NET_T2_GLN,TEXTURE_GLSZM_NET_T2_ZSN,TEXTURE_GLSZM_NET_T2_ZP,TEXTURE_GLSZM_NET_T2_LGZE,TEXTURE_GLSZM_NET_T2_HGZE,TEXTURE_GLSZM_NET_T2_SZLGE,TEXTURE_GLSZM_NET_T2_SZHGE,TEXTURE_GLSZM_NET_T2_LZLGE,TEXTURE_GLSZM_NET_T2_LZHGE,TEXTURE_GLSZM_NET_T2_GLV,TEXTURE_GLSZM_NET_T2_ZSV,TEXTURE_GLSZM_NET_FLAIR_SZE,TEXTURE_GLSZM_NET_FLAIR_LZE,TEXTURE_GLSZM_NET_FLAIR_GLN,TEXTURE_GLSZM_NET_FLAIR_ZSN,TEXTURE_GLSZM_NET_FLAIR_ZP,TEXTURE_GLSZM_NET_FLAIR_LGZE,TEXTURE_GLSZM_NET_FLAIR_HGZE,TEXTURE_GLSZM_NET_FLAIR_SZLGE,TEXTURE_GLSZM_NET_FLAIR_SZHGE,TEXTURE_GLSZM_NET_FLAIR_LZLGE,TEXTURE_GLSZM_NET_FLAIR_LZHGE,TEXTURE_GLSZM_NET_FLAIR_GLV,TEXTURE_GLSZM_NET_FLAIR_ZSV,TEXTURE_NGTDM_ET_T1Gd_Coarseness,TEXTURE_NGTDM_ET_T1Gd_Contrast,TEXTURE_NGTDM_ET_T1Gd_Busyness,TEXTURE_NGTDM_ET_T1Gd_Complexity,TEXTURE_NGTDM_ET_T1Gd_Strength,TEXTURE_NGTDM_ET_T1_Coarseness,TEXTURE_NGTDM_ET_T1_Contrast,TEXTURE_NGTDM_ET_T1_Busyness,TEXTURE_NGTDM_ET_T1_Complexity,TEXTURE_NGTDM_ET_T1_Strength,TEXTURE_NGTDM_ET_T2_Coarseness,TEXTURE_NGTDM_ET_T2_Contrast,TEXTURE_NGTDM_ET_T2_Busyness,TEXTURE_NGTDM_ET_T2_Complexity,TEXTURE_NGTDM_ET_T2_Strength,TEXTURE_NGTDM_ET_FLAIR_Coarseness,TEXTURE_NGTDM_ET_FLAIR_Contrast,TEXTURE_NGTDM_ET_FLAIR_Busyness,TEXTURE_NGTDM_ET_FLAIR_Complexity,TEXTURE_NGTDM_ET_FLAIR_Strength,TEXTURE_NGTDM_ED_T1Gd_Coarseness,TEXTURE_NGTDM_ED_T1Gd_Contrast,TEXTURE_NGTDM_ED_T1Gd_Busyness,TEXTURE_NGTDM_ED_T1Gd_Complexity,TEXTURE_NGTDM_ED_T1Gd_Strength,TEXTURE_NGTDM_ED_T1_Coarseness,TEXTURE_NGTDM_ED_T1_Contrast,TEXTURE_NGTDM_ED_T1_Busyness,TEXTURE_NGTDM_ED_T1_Complexity,TEXTURE_NGTDM_ED_T1_Strength,TEXTURE_NGTDM_ED_T2_Coarseness,TEXTURE_NGTDM_ED_T2_Contrast,TEXTURE_NGTDM_ED_T2_Busyness,TEXTURE_NGTDM_ED_T2_Complexity,TEXTURE_NGTDM_ED_T2_Strength,TEXTURE_NGTDM_ED_FLAIR_Coarseness,TEXTURE_NGTDM_ED_FLAIR_Contrast,TEXTURE_NGTDM_ED_FLAIR_Busyness,TEXTURE_NGTDM_ED_FLAIR_Complexity,TEXTURE_NGTDM_ED_FLAIR_Strength,TEXTURE_NGTDM_NET_T1Gd_Coarseness,TEXTURE_NGTDM_NET_T1Gd_Contrast,TEXTURE_NGTDM_NET_T1Gd_Busyness,TEXTURE_NGTDM_NET_T1Gd_Complexity,TEXTURE_NGTDM_NET_T1Gd_Strength,TEXTURE_NGTDM_NET_T1_Coarseness,TEXTURE_NGTDM_NET_T1_Contrast,TEXTURE_NGTDM_NET_T1_Busyness,TEXTURE_NGTDM_NET_T1_Complexity,TEXTURE_NGTDM_NET_T1_Strength,TEXTURE_NGTDM_NET_T2_Coarseness,TEXTURE_NGTDM_NET_T2_Contrast,TEXTURE_NGTDM_NET_T2_Busyness,TEXTURE_NGTDM_NET_T2_Complexity,TEXTURE_NGTDM_NET_T2_Strength,TEXTURE_NGTDM_NET_FLAIR_Coarseness,TEXTURE_NGTDM_NET_FLAIR_Contrast,TEXTURE_NGTDM_NET_FLAIR_Busyness,TEXTURE_NGTDM_NET_FLAIR_Complexity,TEXTURE_NGTDM_NET_FLAIR_Strength,TGM_p1,TGM_dw,TGM_Cog_X_1,TGM_Cog_Y_1,TGM_Cog_Z_1,TGM_T_1,TGM_Cog_X_2,TGM_Cog_Y_2,TGM_Cog_Z_2,TGM_T_2,TGM_Cog_X_3,TGM_Cog_Y_3,TGM_Cog_Z_3,TGM_T_3,TGM_Cog_X_4,TGM_Cog_Y_4,TGM_Cog_Z_4,TGM_T_4,TGM_Cog_X_5,TGM_Cog_Y_5,TGM_Cog_Z_5,TGM_T_5,TGM_Cog_X_6,TGM_Cog_Y_6,TGM_Cog_Z_6,TGM_T_6"
        self.context = ""
        self.tuble = pd.read_csv('../BRATS_TCGA_GBM_and_Segmentations/TCGA_GBM_radiomicFeatures.csv')

    def get_txt_encoder(self, image_path,tub_num):
        dashscope.api_key = self.api_key
        self.context = ""
        for i, row in enumerate(self.tuble['ID']):
            if row == tub_num:  # read the correspding line
                self.context = str(self.tuble.loc[i])
                break  # if find, break loop
        text = self.context
        messages = [
            {
                "role": "system",
                "content": [
                    {"text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {"image": image_path},
                    {"text": text + ".Based on the image information, please provide a description of the image segmentation word that is applicable to the Sam Vit model"}]
            }]
        response = dashscope.MultiModalConversation.call(
            api_key=self.api_key,
            model='qwen-vl-max-latest',
            messages=messages
        )

        self.wb_text = response.output.choices[0].message.content[0]["text"]
        self.vector = torch.Tensor(self.text_to_vector(self.wb_text)).to(self.device) #(1, 256, 64)
        self.sparse_embeddings = torch.zeros([1, 2, 256]).to(self.device)
        self.min_pool = torch.nn.MaxPool2d(kernel_size=4, stride=4).to(self.device)

    def text_to_vector(self,text):
        # 调用Qwen文本嵌入接口
        resp = TextEmbedding.call(
            model="text-embedding-v1",  # 专用文本向量模型
            input=text
        )

        # 获取原始向量（默认1024维）
        raw_vector = np.array(resp.output['embeddings'][0]['embedding'])

        # 目标维度为 1 × 256 × 64，共 16384 维
        target_dim = 1 * 256 * 64

        if len(raw_vector) < target_dim:
            padded_vector = np.pad(raw_vector, (0, target_dim - len(raw_vector)), mode='constant')
        else:
            padded_vector = raw_vector[:target_dim]

        # 重塑为目标形状
        return padded_vector.reshape((1, 256, 64))

    def forward(self, image, image_path, tub_num=None):
        self.get_txt_encoder(image_path, tub_num)  # 大语言模型得到编码向量
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)

        dense_embeddings = self.CMEncoder(image_embedding,self.vector).to(device) #交叉注意力融合
        [low_res_masks, _], _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=self.sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=True,
            se = self.SAM.cat_model(self.min_pool(image)),
        )

        return low_res_masks,self.wb_text

if __name__ == '__main__':
    device = 'cuda'#args.device

    sam_checkpoint = './work_dir/medsam_model_latest.pth'
    # s = torch.load(sam_checkpoint, map_location=device)
    # checkpoint = './work_dir/MedSAM-ViT-B-20250419-0027/medsam_model_latest.pth'
    # unet_c = torch.load(checkpoint, map_location=device)
    # s['seg_c'] = unet_c['model']
    # torch.save(s, sam_checkpoint)
    # print(s.keys())
    # print(unet_c.keys())
    # exit()
    SAM = sam_model_registry["vit_b"](num_class=4, checkpoint=None)
    medsam_model = MedSAM(
            SAM=SAM,
            sam_checkpoint=sam_checkpoint
        ).to(device)
    medsam_model.eval()

    # ''' 输入图片 '''
    # # data_path = '../BRATS_TCGA_GBM_and_Segmentations/MRI/TCGA-02-0009_1997.06.14__029.jpg'
    # data_path = '../BRATS_TCGA_GBM_and_Segmentations/MRI/TCGA-08-0385_2001.08.27__042.jpg'

    # img_np = io.imread(data_path)
    # if len(img_np.shape) == 2:
    #     img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    # else:
    #     img_3c = img_np
    # H, W, _ = img_3c.shape
    # image_size = 1024  #1024太大了显存受不了
    # # %% image preprocessing
    # img_1024 = transform.resize(
    #     img_3c, (image_size, image_size), order=3, preserve_range=True, anti_aliasing=True
    # ).astype(np.uint8)
    # img_1024 = (img_1024 - img_1024.min()) / np.clip(
    #     img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    # )  # normalize to [0, 1], (H, W, 3)
    # # convert the shape to (3, H, W)
    # img_1024_tensor = (
    #     torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    # )
    # img_256 = transform.resize(
    #     img_3c, (image_size//4, image_size//4), order=3, preserve_range=True, anti_aliasing=True
    # ).astype(np.uint8)
    # ''' 推理阶段 '''
    # with torch.no_grad():
    #     tub_num = data_path.split('/')[-1].split('_')[0]
    #     pre_seg,wb_text = medsam_model(img_1024_tensor,data_path,tub_num=tub_num)#,[[0,0,1024,1024]]) # (1, 256,64, 64)
    #     pre_seg = torch.argmax(pre_seg,dim=1).cpu().numpy()[0]
    #     # 绘制原图和分割图
    #     plt.figure(figsize=(12, 8))

    #     # 绘制原图
    #     plt.subplot(1, 2, 1)
    #     plt.title("Original Image")
    #     plt.imshow(img_256)
    #     plt.axis("off")

    #     # 绘制分割图叠加在原图上
    #     plt.subplot(1, 2, 2)
    #     plt.title("Segmentation on Original Image")
    #     plt.imshow(img_256)  # 首先显示原图
    #     plt.imshow(pre_seg, alpha=0.5, cmap='jet')  # 叠加分割图，使用透明度和颜色映射
    #     plt.axis("off")

    #     plt.tight_layout()
    #     plt.show()

    #     print(wb_text)


    import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage import io, transform

# 指定文件夹路径和输出路径
data_dir = '../BRATS_TCGA_GBM_and_Segmentations/MRI'
output_dir = './Inference_results'
os.makedirs(output_dir, exist_ok=True)

# 获取所有jpg图像
all_images = sorted([f for f in os.listdir(data_dir) if f.endswith('.jpg')])
step = 100  # 每隔100张处理一次

for idx in range(0, len(all_images), step):
    data_path = os.path.join(data_dir, all_images[idx])
    print(f"Processing: {data_path}")

    # === 图像预处理 ===
    img_np = io.imread(data_path)
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np
    H, W, _ = img_3c.shape
    image_size = 1024

    img_1024 = transform.resize(
        img_3c, (image_size, image_size), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), 1e-8, None)
    img_1024_tensor = (
        torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )

    img_256 = transform.resize(
        img_3c, (image_size//4, image_size//4), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)

    # === 模型推理 ===
    with torch.no_grad():
        tub_num = data_path.split('/')[-1].split('_')[0]
        pre_seg, wb_text = medsam_model(img_1024_tensor, data_path, tub_num=tub_num)
        pre_seg = torch.argmax(pre_seg, dim=1).cpu().numpy()[0]

    # === 可视化并保存结果 ===
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(img_256)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(img_256)
    axs[1].imshow(pre_seg, alpha=0.5, cmap='jet')
    axs[1].set_title("Segmentation Overlay")
    axs[1].axis("off")

    plt.tight_layout()
    
    # 输出文件名
    save_name = os.path.splitext(all_images[idx])[0] + '_result.jpg'
    plt.savefig(os.path.join(output_dir, save_name))
    plt.close()

    # 保存描述文本
    with open(os.path.join(output_dir, os.path.splitext(save_name)[0] + '_desc.txt'), 'w') as f:
        f.write(wb_text)

    print(f"Saved to {save_name}")
