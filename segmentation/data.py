# DataSet loader class for WPU-Net
# Based on https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

from torch.utils.data.dataset import Dataset
import os, time
import torch
import numpy as np
from segmentation.tools import getImg, rand_crop, rand_rotation, rand_horizontalFlip, rand_verticalFlip, dilate_mask, get_expansion
from segmentation.weight_map_loss import caculate_weight_map

class IronDataset(Dataset):

    def __init__(self, dataset_folder, train=True, transform=None, crop = True, crop_size=(400, 400), dilate = 5):
        """
        DataSet for our data.
        :param dataset_folder: Address for train set and test set
        :param train:  True if you load train set, False if you load test set
        :param transform:  The pytorch transform you used
        :param crop: Used when you want to randomly crop images
        :param crop_size: Used when randomly cropping images
        :param dilate: Used when dilated boundary is used
        """

        self.__file = []
        self.__im = []
        self.__mask = []
        self.__last = []
        self.transform = transform
        self.crop = crop
        self.crop_size = crop_size
        self.train = train
        self.dilate = dilate

        if self.train:
            folder = dataset_folder + "/train/"
        else:
            folder = dataset_folder + "/val_crop/"

        org_folder = folder + "images/"    # folder for original images
        mask_folder = folder + "labels/"    # folder for labels

        # Find the largest and smallest image id, training and testing needs to start from the second picture in WPU-Net
        max_file = 0
        min_file = 10000000
        for file in os.listdir(org_folder):
            if file.endswith(".png") or file.endswith(".tif"):
                if train:
                    pic_num = int(os.path.splitext(file)[0])
                else:
                    pic_num = int(os.path.splitext(file)[0].split("_")[0])
                if pic_num > max_file:
                    max_file = pic_num
                if pic_num < min_file:
                    min_file = pic_num

        for file in os.listdir(org_folder):
            if file.endswith(".png") or file.endswith(".tif"):
                filename = os.path.splitext(file)[0]
                if train:
                    pic_num = int(os.path.splitext(file)[0])
                else:
                    pic_num = int(os.path.splitext(file)[0].split("_")[0])
                if pic_num != min_file:
                    # 1. read file name
                    self.__file.append(filename)
                    # 2. read original image
                    self.__im.append(org_folder + file)
                    # 3. read  mask image
                    self.__mask.append(mask_folder + filename + ".png")
                    
                    # 4. load last label mask or last result mask
                    if self.train:
                        file_last = str(int(filename) - 1).zfill(3)
                    else:
                        file_last = str(pic_num - 1).zfill(3) + '_' + filename.split('_')[1] + '_' + filename.split('_')[2]
                    
                    self.__last.append(mask_folder + file_last + ".png")
                    
        self.dataset_size = len(self.__file)

    def __getitem__(self, index):

        img = getImg(self.__im[index])  # Original
        mask = getImg(self.__mask[index])  # mask
        last = getImg(self.__last[index])  # last information

        # 裁剪图像 保证img和mask随机裁剪在同一位置   Crop image, Ensuring that img and mask are randomly cropped in the same location
        if self.train and self.crop:  # , weight 
            img, mask, last = rand_crop(data=img, label=mask, last=last, height=self.crop_size[1], width=self.crop_size[0])
            img, mask, last = rand_rotation(data=img, label=mask, last=last)
            img, mask, last = rand_verticalFlip(data=img, label=mask, last=last)
            img, mask, last = rand_horizontalFlip(data=img, label=mask, last=last)
        
        weight,_ = caculate_weight_map(np.array(mask))
        mask = dilate_mask(np.array(mask), iteration=int((self.dilate-1)/2))

        if self.transform is not None:
            img = self.transform(img)
            mask = torch.Tensor(np.array(mask)).unsqueeze(0)/255
            last = torch.Tensor(np.array(last)).unsqueeze(0)
            weight = np.ascontiguousarray(weight, dtype=np.float32)
            weight = torch.from_numpy(weight.transpose((2, 0, 1)))

        return img, mask, last, weight

    def __len__(self):
        return len(self.__im)