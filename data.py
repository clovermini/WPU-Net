# DataSet loader class for WPU-Net
# Based on https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

from torch.utils.data.dataset import Dataset
import os
import torchvision.transforms as tr
import torch
import numpy as np
from tools import getImg, rand_crop, rand_rotation, rand_horizontalFlip, rand_verticalFlip


class IronDataset(Dataset):
    __file = []
    __im = []
    __mask = []
    __last = []
    __weight = []
    dataset_size = 0

    def __init__(self, dataset_folder, mask_address="manual_dilate_5/", last_information="last_map/", train=True, transform=None, crop = False, crop_size=(400, 400)):
        """
        DataSet for our data.
        :param dataset_folder: Address for train set and test set
        :param mask_address: Path for label image, it should be a dilate image of mask when the last information is set to be weight map
        :param last_information:  Path for last information in WPU-Net, it can be mask, mask-expansion or weight map of last slice as described in our paper
        :param train:  True if you load train set, False if you load test set
        :param transform:  The pytorch transform you used
        :param crop: Used when you want to randomly crop images
        :param crop_size: Used when randomly cropping images
        """

        self.__file = []
        self.__im = []
        self.__mask = []
        self.__last = []
        self.__weight = []
        self.transform = transform
        self.crop = crop
        self.crop_size = crop_size
        self.train = train
        self.last_information = last_information

        if train:
            folder = dataset_folder + "train/"
        else:
            folder = dataset_folder + "test/"

        org_folder = folder + "images_tif/"    # folder for original images
        mask_folder = folder + mask_address    # folder for labels
        last_folder = folder + last_information  # folder for last information
        weight_folder = folder + "adaptive_weight_map/"   # folder for weight map

        # Find the largest and smallest image id, training and testing needs to start from the second picture in WPU-Net
        max_file = 0
        min_file = 10000000
        for file in os.listdir(org_folder):
            if file.endswith(".png") or file.endswith(".tif"):
                pic_num = int(os.path.splitext(file)[0].split("_")[0])
                if pic_num > max_file:
                    max_file = pic_num
                if pic_num < min_file:
                    min_file = pic_num

        for file in os.listdir(org_folder):
            if file.endswith(".png") or file.endswith(".tif"):
                filename = os.path.splitext(file)[0]
                pic_num = int(os.path.splitext(file)[0].split("_")[0])
                if pic_num != min_file:
                    # 1. read file name
                    self.__file.append(filename)
                    # 2. read original image
                    self.__im.append(org_folder + file)
                    # 3. read  mask image
                    self.__mask.append(mask_folder + filename + ".png")
                    # 4. load last mask
                    self.__last.append(last_folder + filename + ".npy")
                    # load weight map
                    self.__weight.append(weight_folder + filename + ".npy")

        self.dataset_size = len(self.__file)

    def __getitem__(self, index):

        img = getImg(self.__im[index])  # Original
        mask = getImg(self.__mask[index])  # mask
        last = getImg(self.__last[index])  # last information
        weight = getImg(self.__weight[index])  # weight map

        # # 裁剪图像 保证img和mask随机裁剪在同一位置   Crop image, Ensuring that img and mask are randomly cropped in the same location
        # if self.crop:
        #     img, mask, last, weight = rand_crop(data=img, label=mask, last=last, weight=weight, height=self.crop_size[1], width=self.crop_size[0])
        #     img, mask, last, weight = rand_rotation(data=img, label=mask, last=last, weight=weight)
        #     img, mask, last, weight = rand_verticalFlip(data=img, label=mask, last=last, weight=weight)
        #     img, mask, last, weight = rand_horizontalFlip(data=img, label=mask, last=last, weight=weight)

        if self.transform is not None:
            img = self.transform(img)
            mask = tr.ToTensor()(mask)
            last = torch.Tensor(np.array(last)).unsqueeze(0)
            weight = np.ascontiguousarray(weight, dtype=np.float32)
            weight = torch.from_numpy(weight.transpose((2, 0, 1)))

        return img, mask, last, weight

    def __len__(self):
        return len(self.__im)