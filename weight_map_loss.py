import torch
import torch.nn as nn
import cv2, os
from skimage.measure import label, regionprops
import scipy.ndimage as ndimage
import numpy as np

def get_obj_dis_weight(dis_map, w0=10, eps=1e-20):
    """
    获得前景（晶界）权重图,基于正态分布曲线在[-2.58*sigma, 2.58*sigma]处概率密度为99%
    因此每次求取最大值max_dis，反推sigma = max_dis / 2.58
    并根据Unet的原文计算Loss

    Obtain a foreground (grain boundary) weight map based on a normal distribution curve with a probability density of 99% at [-2.58*sigma, 2.58*sigma]
  So each time you get the maximum value max_dis, and then calculate sigma = max_dis / 2.58
  finally calculate Loss based on the original paper of U-Net
    """
    max_dis = np.amax(dis_map)
    std = max_dis / 2.58 + eps
    weight_matrix = w0 * np.exp(-1 * pow(dis_map, 2) / (2 * pow(std, 2)))
    return weight_matrix

def get_bck_dis_weight(dis_map, w0=10, eps=1e-20):
    """
    获得背景（晶粒内部）权重图   Obtain background (inside grain) weight map
    """
    max_dis = np.amax(dis_map)
    std = max_dis / 2.58 + eps
    weight_matrix = w0 * np.exp(-1 * pow((max_dis - dis_map), 2) / (2 * pow(std, 2)))
    return weight_matrix

def caculate_weight_map(maskAddress, saveAddress, type = "last"):
    """
    计算真值图对应的权重图  Calculate the weight map corresponding to the mask image
    :param maskAddress:  Address for mask image
    :param saveAddress:  save directory
    :return:  "adaptive_dis_weight" is the weight map for loss   "adaptive_bck_dis_weight_norm" is the weight map for last information
    """

    mask = cv2.imread(maskAddress, 0)
    labeled, label_num = label(mask, neighbors=4, background=255, return_num=True)
    image_props = regionprops(labeled, cache=False)
    dis_trf = ndimage.distance_transform_edt(255 - mask)
    adaptive_obj_dis_weight = np.zeros(mask.shape, dtype=np.float32)
    adaptive_obj_dis_weight = adaptive_obj_dis_weight + (mask / 255) * 30
    adaptive_bck_dis_weight = np.ones(mask.shape, dtype=np.float32)

    for num in range(1, label_num + 1):
        image_prop = image_props[num - 1]
        bool_dis = np.zeros(image_prop.image.shape)
        bool_dis[image_prop.image] = 1.0
        (min_row, min_col, max_row, max_col) = image_prop.bbox
        temp_dis = np.zeros(bool_dis.shape)
        temp_dis = dis_trf[min_row: max_row, min_col: max_col] * bool_dis

        adaptive_obj_dis_weight[min_row: max_row, min_col: max_col] = adaptive_obj_dis_weight[min_row: max_row, min_col: max_col] + get_obj_dis_weight(temp_dis) * bool_dis
        adaptive_bck_dis_weight[min_row: max_row, min_col: max_col] = adaptive_bck_dis_weight[min_row: max_row, min_col: max_col] + get_bck_dis_weight(temp_dis) * bool_dis

        # get weight map for loss
        adaptive_bck_dis_weight = adaptive_bck_dis_weight[:, :, np.newaxis]
        adaptive_obj_dis_weight = adaptive_obj_dis_weight[:, :, np.newaxis]
        adaptive_dis_weight = np.concatenate((adaptive_bck_dis_weight, adaptive_obj_dis_weight), axis=2)

        np.save(os.path.join(saveAddress, "weight_map_loss.npy"), adaptive_dis_weight)

        print("adaptive_obj_dis_weight range ", np.max(adaptive_obj_dis_weight), " ", np.min(adaptive_obj_dis_weight))
        print("adaptive_bck_dis_weight range ", np.max(adaptive_bck_dis_weight), " ", np.min(adaptive_bck_dis_weight))

        # get weight for last information
        bck_maxinum = np.max(adaptive_bck_dis_weight)
        bck_mininum = np.min(adaptive_bck_dis_weight)
        adaptive_bck_dis_weight_norm = (adaptive_bck_dis_weight - bck_mininum) / (bck_maxinum - bck_mininum)
        adaptive_bck_dis_weight_norm = (1 - adaptive_bck_dis_weight_norm) * (-7) + 1

        np.save(os.path.join(saveAddress, "weight_map_last.npy"), adaptive_bck_dis_weight_norm)

        return adaptive_dis_weight, adaptive_bck_dis_weight_norm


class WeightMapLoss(nn.Module):
    """
    calculate weights with weight maps in two channels
    """

    def __init__(self, _eps=1e-20, _dilate_cof=1):
        super(WeightMapLoss, self).__init__()
        self._eps = _eps
        # Dilate Coefficient of Mask
        self._dilate_cof = _dilate_cof
        # class balance weight, which is adjusted according to the dilate coefficient. The dilate coefficient can be 1, 3, 5, 7, 9 ....
        self._weight_cof = torch.Tensor([_dilate_cof, 20]).cuda()
    
    def _calculate_maps(self, mask, weight_maps, method):
        if -1 < method <= 6:  # WCE  LOSS
            weight_bck = torch.zeros_like(mask)
            weight_obj = torch.zeros_like(mask)
            if method == 1:  # class balance weighted loss
                weight_bck = (1 - mask) * self._weight_cof[0]
                weight_obj = mask * self._weight_cof[1]
            elif method == 2:  # 自适应膨胀 晶界加权（bck也加权） Adaptive weighted loss with dilated mask (bck is also weighted)
                weight_bck = (1 - mask) * weight_maps[:, 0, :, :]
                weight_obj = mask * weight_maps[:, 1, :, :]
            elif method == 3:  # 自适应膨胀 晶界加权（bck为1） Adaptive weighted loss with dilated mask (bck is set to 1)
                weight_bck = (1 - mask) * self._weight_cof[0]
                weight_obj = mask * weight_maps[:, 1, :, :]
            elif method == 4:  # 自适应对比晶界加权（bck也加权） Adaptive weighted loss described in our paper (bck is is also weighted)
                temp_weight_bck = weight_maps[:, 0, :, :]
                temp_weight_obj = weight_maps[:, 1, :, :]
                weight_obj[temp_weight_obj >= temp_weight_bck] = temp_weight_obj[temp_weight_obj >= temp_weight_bck]
                weight_obj = mask * weight_obj
                weight_bck[weight_obj <= temp_weight_bck] = temp_weight_bck[weight_obj <= temp_weight_bck]
            elif method == 5:  # 自适应对比晶界加权（bck为1）  Adaptive weighted loss described in our paper (bck is set to 1)
                temp_weight_bck = weight_maps[:, 0, :, :]
                temp_weight_obj = weight_maps[:, 1, :, :]
                weight_obj[temp_weight_obj >= temp_weight_bck] = temp_weight_obj[temp_weight_obj >= temp_weight_bck]
                weight_obj = mask * weight_obj
                weight_bck[weight_obj <= temp_weight_bck] = 1
            return weight_bck, weight_obj
        elif method >= 7:     # MSE  LOSS
            weight_map = torch.zeros_like(mask)
            # MSE LOSS
            if method == 7:   # class banlance weighted loss
                weight_map = mask * (self._weight_cof[1] - self._weight_cof[0])
                weight_map = weight_map + self._weight_cof[0]
            elif method == 8:   # Adaptive weighted loss described in our paper
                weight_map = weight_map + mask * 30
                weight_map = weight_map + (1 - mask) * weight_maps[:, 0, :, :]
            weight_map[weight_map < 1] = 1
            return weight_map
    
    def forward(self, input, target, weight_maps, method=0):
        """
        target: The target map, LongTensor, unique(target) = [0 1]
        weight_maps: The weights for two channels，weight_maps = [weight_bck_map, weight_obj_map]
        method：control the method of weight calculation
        """
        mask = target.float()
        if -1 < method <= 6:  # WCE
            weight_bck, weight_obj = self._calculate_maps(mask, weight_maps, method)
            logit = torch.softmax(input, dim=1)
            logit = logit.clamp(self._eps, 1. - self._eps)
            loss = -1 * weight_bck * torch.log(logit[:, 0, :, :]) - weight_obj * torch.log(logit[:, 1, :, :])
            weight_sum = weight_bck + weight_obj
            return loss.sum() / weight_sum.sum()
        elif method >= 7:  # MSE
            weight_map = self._calculate_maps(mask, weight_maps, method)
            loss = weight_map * torch.pow(input - mask, 2)
            return torch.mean(loss.sum(dim=(1,2)) / weight_map.sum(dim=(1,2)))
    
    def show_weight(self, target, weight_maps, method=0):
        """
        Return the weights
        target: The target map, LongTensor, unique(target) = [0 1]
        weight_maps: The weights for two channels，weight_maps = [weight_bck_map, weight_obj_map]
        method：control the method of weight calculation
        """
        mask = target.float()
        if -1 < method <= 6: # WCE
            weight_bck, weight_obj = self._calculate_maps(mask, weight_maps, method)
            return weight_bck, weight_obj
        elif method >= 7:  # MSE
            weight_map = self._calculate_maps(mask, weight_maps, method)
            return weight_map