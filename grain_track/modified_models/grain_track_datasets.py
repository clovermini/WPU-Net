from torch.utils.data.dataset import Dataset
import numpy as np
import os
import PIL.Image
import random


class GrainTrackDatasets(Dataset):
    """
    The Dataset of grain track 建立晶粒追踪数据集
    """
    def __init__(self, dataDir, transform):
        """
        Function: initialize the dataset
        功能: 初始化数据集
        :param dataDir: 数据集目录，该目录下每个文件夹为一个类别
        :param transform: 数据集增强方式
        """
        self.dir_list = None
        self.file_list = []
        self.class_list = []
        self.transform = transform
        for root, dirs, files in os.walk(dataDir):
            if len(files) > 0:
                value = os.path.basename(root)
                for item in files:
                    self.file_list.append(os.path.join(root, item))
                    self.class_list.append(int(value))
        self.file_num = len(self.file_list)
        self.class_num = 2

    def __len__(self):
        """
        Function: return the length of the datasets
        功能：返回数据集长度
        :return: 数据集长度
        """
        return self.file_num

    def __getitem__(self, idx):
        image = np.load(self.file_list[idx])
        image_class = self.class_list[idx]
        if self.transform:
            image_pil = PIL.Image.fromarray(image)
            image = np.array(self.transform(image_pil))
        sample = {'image':image, 'label':image_class}
        return sample


class RandomChannelFlip(object):
    """ Flip the given PIL Image with 2 channels in channel direction randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            img_array = np.array(img)
            temp = img_array[:, :, 0].copy()
            img_array[:, :, 0] = img_array[:, :, 1]
            img_array[:, :, 1] = temp
            return PIL.Image.fromarray(img_array)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
