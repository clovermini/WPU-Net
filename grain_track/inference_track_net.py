import os
import numpy as np
from skimage.measure import label, regionprops
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from grain_track import utility
from grain_track.modified_models import densenet
from grain_track.modified_models import vgg
from grain_track.modified_models.grain_track_datasets import RandomChannelFlip


class CnnTracker:
    """
    CNN Tracker Class, perform grain track by using different CNN models
    CNN 追踪类，以不同神经网络执行晶粒追踪
    """

    def __init__(self, model=1, pre_train_address="", device='cuda:0', need_augment=False, max_num_tensor=30,
                 verbose=False):
        """
        :param model: specify the certain cnn model, 0 represents vgg13_bn, 1 represents densenet161
        :param pre_train_address: specify the address of pretrain model
        :param device: specify device to perform
        :param need_augment: whether or not to use test time augmentation
        :param max_num_tensor: the max number of data in one batch, it will be suited to gpu memory
        :param verbose: whether or not to print log
        """
        self._model_num = model
        self._pre_train_address = pre_train_address
        self._device = device
        self._need_augment = need_augment
        if self._need_augment:
            self._num_augment = 7
        else:
            self._num_augment = 1
        self._verbose = verbose
        self._num_classes = 2
        self._max_num_tensor = max_num_tensor
        # define transforms 定义测试时增强的增强方案
        self._oridinal_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
            transforms.ToTensor(),
        ])

        self._channel_flip_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
            RandomChannelFlip(1),
            transforms.ToTensor(),
        ])

        self._horizontal_flip_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
            transforms.RandomHorizontalFlip(1),
            transforms.ToTensor(),
        ])

        self._vertical_flip_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
            transforms.RandomVerticalFlip(1),
            transforms.ToTensor(),
        ])

        self._rotate90_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
            transforms.RandomRotation((90, 90), expand=True),
            transforms.ToTensor(),
        ])

        self._rotate180_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
            transforms.RandomRotation((180, 180), expand=True),
            transforms.ToTensor(),
        ])

        self._rotate270_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
            transforms.RandomRotation((270, 270), expand=True),
            transforms.ToTensor(),
        ])
        self._net_model = None

    def net_init(self):
        """
        initialize pre-train model of cnn
        初始化神经网络模型，加载预训练参数
        """
        if self._model_num == 0:  # vgg13_bn()
            self._net_model = vgg.vgg13_bn()
            self._net_model.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                                                       nn.ReLU(True),
                                                       nn.Dropout(),
                                                       nn.Linear(4096, 4096),
                                                       nn.ReLU(True),
                                                       nn.Dropout(),
                                                       nn.Linear(4096, self._num_classes))
            self._net_model.to(self._device)
            self._net_model.load_state_dict(
                {k.replace('module.', ''): v for k, v in torch.load(self._pre_train_address).items()})
        if self._model_num == 1:  # densenet161()
            self._net_model = densenet.densenet161()
            num_ftrs = self._net_model.classifier.in_features
            self._net_model.classifier = nn.Linear(num_ftrs, self._num_classes)
            self._net_model.to(self._device)
            self._net_model.load_state_dict(
                {k.replace('module.', ''): v for k, v in torch.load(self._pre_train_address).items()})
        self._net_model.eval()

    def run(self, input_arrays):
        """
        Perform Track for input_arrays
        对输入input_arrays执行推理，返回scores_list（每个元素为该input_array追踪正确的）的概率
        """
        input_num = len(input_arrays)
        input_tensors = self._data_transform(input_arrays, input_num)
        output_tensors = None
        scores_list = []
        with torch.no_grad():
            if self._need_augment == False or (
                    input_num * self._num_augment < self._max_num_tensor and self._need_augment):
                output_tensors = self._net_model(input_tensors).data
            else:
                output_tensors = torch.zeros([input_num * self._num_augment, 2]).to(self._device)
                implement_num = 0
                while implement_num < input_num * self._num_augment:
                    remain_num = input_num * self._num_augment - implement_num
                    if remain_num > self._max_num_tensor:
                        output_tensors[implement_num:implement_num + self._max_num_tensor, :] = self._net_model(
                            input_tensors[implement_num:implement_num + self._max_num_tensor, :, :, :]).data
                    else:
                        output_tensors[implement_num:input_num * self._num_augment, :] = self._net_model(
                            input_tensors[implement_num: input_num * self._num_augment, :, :, :]).data
                    implement_num += self._max_num_tensor

        output_scores = F.softmax(output_tensors, dim=1)
        for index in range(input_num):
            temp_score = \
                torch.mean(output_scores[index * self._num_augment: index * self._num_augment + self._num_augment, :],
                           dim=0)[0]
            scores_list.append(temp_score.item())
        return scores_list

    def _data_transform(self, input_arrays, input_num):
        """
        Transform input_arrays to input_tensors,considering test time augmentation
        将input_arrays转换为input_tensors, 考虑是否有测试时增强
        """
        input_tensors = None
        if self._need_augment:
            input_tensors = torch.zeros((input_num * self._num_augment, 2, 224, 224)).to(self._device)
            for index, array in enumerate(input_arrays):
                image_pil = PIL.Image.fromarray(array)
                input_tensors[index * self._num_augment + 0, :, :, :] = self._oridinal_transform(image_pil)
                input_tensors[index * self._num_augment + 1, :, :, :] = self._channel_flip_transform(image_pil)
                input_tensors[index * self._num_augment + 2, :, :, :] = self._horizontal_flip_transform(image_pil)
                input_tensors[index * self._num_augment + 3, :, :, :] = self._vertical_flip_transform(image_pil)
                input_tensors[index * self._num_augment + 4, :, :, :] = self._rotate90_transform(image_pil)
                input_tensors[index * self._num_augment + 5, :, :, :] = self._rotate180_transform(image_pil)
                input_tensors[index * self._num_augment + 6, :, :, :] = self._rotate270_transform(image_pil)
        else:
            input_tensors = torch.zeros((input_num, 2, 224, 224)).to(self._device)
            for index, array in enumerate(input_arrays):
                image_pil = PIL.Image.fromarray(array)
                input_tensors[index, :, :, :] = self._oridinal_transform(image_pil)
        return input_tensors

    def _print_screen_log(self, content):
        """
        Function: print content if self._verbose is True
        如果self._verbose为真，则打印信息
        :param content: str
        """
        if self._verbose:
            print(content)


class GrainTrack():
    """
    Class of grain track 晶粒追踪类
    """

    def __init__(self, input_dir, reverse_label=False, verbose=True):
        """
        :param input_dir: the dir of images
        :param reverse_label: whether or not to read label start from last slice
        :param verbose: whether or not to print log
        """
        self._verbose = verbose
        self._images_list = []
        for item in sorted(os.listdir(input_dir), reverse=reverse_label):
            if item == ".ipynb_checkpoints":
                continue
            self._images_list.append(os.path.join(input_dir, item))
        self._cnn_tracker = CnnTracker()

    def get_tracked_label_stack(self, method=0):
        """
        achieve tracked labek stack by different method
        :param method: 0 represents
        :return: tracked_label_stack
        """
        last_labeled, last_num = utility.get_label_from_boundary(self._images_list[0])

        # construct 3d array from shape of first image and number of image
        # 根据第一张图像的长宽和文件数构建3D校正标记矩阵
        label_stack = np.zeros((last_labeled.shape[0], last_labeled.shape[1], len(self._images_list)), dtype=np.int64)

        # assign last_labeled to first slice of label_stack
        # 将第一层的标记直接赋值矩阵的第一层
        label_stack[:, :, 0] = last_labeled

        # record the number of grains
        # 记录目前统计了多少晶粒
        label_num = last_num

        self._print_screen_log("Start tracking")

        # Tracking grian for every slice
        # 对 labelStack 的每一层进行标记追踪
        for index in range(1, len(self._images_list)):
            #             self._print_screen_log("Track label {}th layer".format(index + 1))
            this_labeled, this_num = utility.get_label_from_boundary(self._images_list[index])
            # return track_labeled of this slice according to last label by different method
            # 根据上层标注结果和选择的方法返回本层截面的追踪结果
            track_labeled = self._track_label(last_labeled, this_labeled, label_num, method=method)

            # Rectify last_labeled by track_labeled, if there are one slice grian, it will be justified
            # 由于有了本层追踪结果，即可判断上层中有无单层晶粒，有的话修正上层，并根据修正的结果重新追踪本层
            if index > 1:
                # is_find_slg(Single layer grain) whether or not have one slice grain in last slice 判断上层是否有单层晶粒
                is_find_slg, last_labeled = self._justify_last_labeled(label_stack[:, :, index - 2],
                                                                       label_stack[:, :, index - 1], track_labeled)
                if is_find_slg:
                    label_stack[:, :, index - 1] = last_labeled
                    # Retrack this slice by modified last_labeled, it will spend much more time
                    # 根据修正的上层标注重新追踪本层，会多耗一倍的时间
                    # track_labeled = self._track_label(last_labeled, this_labeled, label_num, method = method)
            label_stack[:, :, index] = track_labeled
            label_num = np.amax(label_stack[:, :, 0: index + 1])
            # Reassign last_labeled by track_labeled
            # 将本层的标记结果当成上一层的标记结果，迭代
            last_labeled = track_labeled

        self._print_screen_log("Tracking done")
        label_stack, label_num = label(label_stack, return_num=True)
        # Return tracked result and number
        # 返回追踪的结果和标记数目
        return label_stack, label_num

    def _track_label(self, last_labeled, this_labeled, label_num, method=1):
        """
        Track this_labeled from last_labeled by different method
        :param last_labeled, np.array
        :param this_labeled, np.array
        :param label_num, number of max grains
        :param method, 1 represents minimun centroid distance, 2 represents maximum overlap area, 3 represents CNN model.
        """
        if method == 1:  # 根据最小质心距离
            return self._track_by_min_centroid_dis(last_labeled, this_labeled, label_num)
        elif method == 2:  # 根据重合面积最大
            return self._track_by_max_overlap_area(last_labeled, this_labeled, label_num)
        elif method == 3:  # 根据CNN
            return self._track_by_cnn(last_labeled, this_labeled, label_num)

    def _track_by_cnn(self, last_labeled, this_labeled, label_num):
        """
        Function: track this_labeled by CNN
        功能：根据上层的标记，重新标记本层的晶粒
        :param last_labeled: 上层标记
        :param this_labeled: 本层初步标记
        :param label_num：目前晶粒个数，用于新增晶粒计数
        :return: 标记追踪结果
        """

        track_labeled = np.zeros(this_labeled.shape, dtype=np.int64)

        last_label_list = np.unique(last_labeled).tolist()
        last_regions = regionprops(last_labeled, cache=True)
        this_regions = regionprops(this_labeled, cache=True)

        is_this_justified = [False for _ in range(len(this_regions))]  # 判断本层的晶粒是否打上标签
        count_num = 0

        # 由于本方法对于上层的每个晶粒可能匹配本层多个晶粒，故需先记录输出结果，再取结果的最大值，再赋值
        # 为上层每个晶粒label记录匹配，每个晶粒可能记录多个匹配（字典）{this_label：similarity},对于多个匹配，取最大相似度的匹配
        match_list = [{} for _ in range(len(last_regions))]

        # ***********************************************晶粒判断**********************************************
        for this_index in range(0, len(this_regions)):
            target_this_label = this_regions[this_index].label
            overlap_last_labels = np.unique(last_labeled[this_labeled == target_this_label])
            input_arrays = []
            input_labels = []
            max_score = 0
            max_score_label = 0
            for overlap_last_label in overlap_last_labels:
                input_array = utility.crop_two_slice_data(last_labeled, overlap_last_label, this_labeled,
                                                          target_this_label)
                h, w, d = input_array.shape
                if h < 2 or w < 2 or d < 2:
                    count_num += 1
                    continue
                input_arrays.append(input_array)
                input_labels.append(overlap_last_label)
            if len(input_arrays) > 0:
                output_scores = self._cnn_tracker.run(input_arrays)
                max_score = max(output_scores)
                max_score_label = input_labels[output_scores.index(max_score)]
                if max_score > 0.5:
                    match_list[last_label_list.index(max_score_label)][str(target_this_label)] = max_score
        # 若上层某个晶粒同时和本层多个晶粒的重叠面积同为最大，则将上层的晶粒赋值给重叠面积最大的那个
        for index, match_dict in enumerate(match_list):
            if len(match_dict) == 0:
                continue
            match_this_label = int(sorted(match_dict.items(), key=lambda d: d[1], reverse=True)[0][0])
            match_last_label = last_label_list[index]
            track_labeled[this_labeled == match_this_label] = match_last_label
            is_this_justified[match_this_label - 1] = True

            # ***********************************************新增晶粒判断**********************************************
        # 仍有未标注的新增晶粒或噪声，则按照新增值处理
        for this_index in range(0, len(this_regions)):
            if is_this_justified[this_index]:
                continue
            target_this_label = this_regions[this_index].label
            label_num = label_num + 1
            track_labeled[this_labeled == target_this_label] = label_num
            is_this_justified[this_index] = True
        return track_labeled

    def set_cnn_tracker(self, model=0, pretrain_address="", device='cuda:2', need_augment=False, max_num_tensor=30):
        """
        Set cnn model
        :param model: specify the certain cnn model, 0 represents vgg13_bn, 1 represents densenet161
        :param pre_train_address: specify the address of pretrain model
        :param device: specify device to perform
        :param need_augment: whether or not to use test time augmentation
        :param max_num_tensor: the max number of data in one batch, it will be suited to gpu memory
        """
        self._cnn_tracker = CnnTracker(model, pretrain_address, device, need_augment, max_num_tensor,
                                       verbose=self._verbose)
        self._cnn_tracker.net_init()

    def _track_by_max_overlap_area(self, last_labeled, this_labeled, label_num):
        """
        Function： track grain by max overlap area, that is for each grain in this slice, assign label with max overlap area
        from last labels set
        功能：根据最大重贴面积追踪
        :param last_label: 上层标记
        :param this_label: 本层初步标记
        :param laebl_num：目前晶粒个数，用于新增晶粒计数
        :return: 标记追踪结果
        """

        track_labeled = np.zeros(this_labeled.shape, dtype=np.int64)

        last_label_list = np.unique(last_labeled).tolist()

        last_regions = regionprops(last_labeled, cache=True)
        this_regions = regionprops(this_labeled, cache=True)

        is_last_justified = [False for _ in range(len(last_regions))]  # 判断上层的晶粒是否打上标签
        is_this_justified = [False for _ in range(len(this_regions))]  # 判断本层的晶粒是否打上标签
        # ****************************************最大面积匹配方法**********************************************
        # 由于本方法对于上层的每个晶粒可能匹配本层多个晶粒，故需先记录面积，再取面积的最大值，再赋值
        # 为上层每个晶粒label记录匹配，每个晶粒可能记录多个匹配（字典）{this_label：overlapArea},对于多个匹配，取最大的覆盖面积
        match_list = [{} for _ in range(len(last_regions))]

        none_track_num = np.count_nonzero(np.array(is_this_justified))

        while (True):
            # 根据逐一计算本层的每个晶粒和上层各个晶粒的覆盖面积，并在matchList中记录最大覆盖面积和对应的下一层晶粒
            for this_index in range(0, len(this_regions)):
                if is_this_justified[this_index]:
                    continue
                target_this_label = this_regions[this_index].label
                # 记录该晶粒相对于上一层覆盖了多少晶粒标签
                overlap_last_labels = np.unique(last_labeled[this_labeled == target_this_label])
                if len(overlap_last_labels) > 0:
                    match_label, max_overlap_area = -1, 0
                    # 寻找上层所有相交的晶粒中，未被传播到下层且面积占优的晶粒
                    for overlap_last_label in overlap_last_labels:
                        if is_last_justified[last_label_list.index(overlap_last_label)]:
                            continue
                        overlap_area = np.count_nonzero(
                            last_labeled[this_labeled == target_this_label] == overlap_last_label)
                        if overlap_area > max_overlap_area:
                            max_overlap_area = overlap_area
                            match_label = overlap_last_label
                    if max_overlap_area > 0:
                        match_list[last_label_list.index(match_label)][str(target_this_label)] = max_overlap_area

            # 若上层某个晶粒同时和本层多个晶粒的重叠面积同为最大，则将上层的晶粒赋值给重叠面积最大的那个
            for index, match_dict in enumerate(match_list):
                if len(match_dict) == 0:
                    continue
                match_this_label = int(sorted(match_dict.items(), key=lambda d: d[1], reverse=True)[0][0])
                match_last_label = last_label_list[index]
                track_labeled[this_labeled == match_this_label] = match_last_label
                is_last_justified[index] = True
                is_this_justified[match_this_label - 1] = True

            # 判断本层尚未判断的晶粒和上一层循环是否相同，相同则退出
            if none_track_num == np.count_nonzero(np.array(is_this_justified)):
                break
            else:
                none_track_num = np.count_nonzero(np.array(is_this_justified))

        # ***********************************************新增晶粒判断**********************************************
        # 仍有未标注的新增晶粒或噪声，则按照新增值处理
        for this_index in range(0, len(this_regions)):
            if is_this_justified[this_index]:
                continue
            target_this_label = this_regions[this_index].label
            label_num = label_num + 1
            track_labeled[this_labeled == target_this_label] = label_num
            is_this_justified[this_index] = True
        return track_labeled

    def _track_by_min_centroid_dis(self, last_labeled, this_labeled, label_num):
        """
        Function： track grain by min centroid_dis, that is for each grain in this slice, assign label with min centroid dis
        from last labels set
        功能：根据最小质心距离的原则追踪
        :param last_labeled: 上层标记
        :param this_labeled: 本层初步标记
        :param laebl_num：
        :return:
        """
        track_labeled = np.zeros(this_labeled.shape, dtype=np.int64)

        last_label_list = np.unique(last_labeled).tolist()
        last_regions = regionprops(last_labeled, cache=True)
        this_regions = regionprops(this_labeled, cache=True)

        is_last_justified = [False for _ in range(len(last_regions) + 1)]  # 判断上层的晶粒是否打上标签，+1为了索引一致
        is_this_justified = [False for _ in range(len(this_regions) + 1)]  # 判断本层的晶粒是否打上标签，+1为了索引一致

        # ****************************************质心最小距离匹配方法**********************************************
        # 由于本方法对于上层的每个晶粒可能匹配本层多个晶粒，故需先记录距离，再取距离的最小值，再赋值
        # 为上层每个晶粒label记录匹配，每个晶粒可能记录多个匹配（字典）{this_label：distance},对于多个匹配，取最小距离的匹配
        match_list = [{} for _ in range(len(last_regions))]

        # 记录上一层每个晶粒的质心点坐标
        centroid_last = np.zeros((len(last_regions), 2), dtype=np.float32)
        for i in range(len(last_regions)):
            centroid_last[i, :] = last_regions[i].centroid

        # 根据质心逐一计算本层的每个晶粒和上层各个晶粒的距离，并在matchList中记录最小距离和对应的下一层晶粒
        for this_index in range(0, len(this_regions)):
            distances = np.sqrt(
                np.sum(np.power(centroid_last - np.array(this_regions[this_index].centroid), 2), axis=1))
            # 计算质心间最小距离
            min_dis = np.min(distances)
            min_dis_index = int(np.argmin(distances))
            # 判断质心最小的同时要求上下层晶粒相交
            overlap_last_labels = np.unique(last_labeled[this_labeled == this_index + 1])

            if last_label_list[min_dis_index] in overlap_last_labels:
                overlap_area = np.count_nonzero(
                    last_labeled[this_labeled == this_index + 1] == last_label_list[min_dis_index])
                # 放在一张图上计算并集
                union = np.count_nonzero(
                    (this_labeled == this_index + 1) | (last_labeled == last_label_list[min_dis_index]))
                iou = overlap_area / union
                if iou > 0.2:
                    match_list[min_dis_index][str(this_regions[this_index].label)] = min_dis

        # 若上层某个晶粒的质心同时离本层多个晶粒的质心同为距离最小，则取最近的赋值
        for index, match_dict in enumerate(match_list):
            if len(match_dict) == 0:
                # print("无匹配", matchIndex)
                continue
            match_this_Label = int(sorted(match_dict.items(), key=lambda d: d[1])[0][0])
            match_last_label = last_label_list[index]
            track_labeled[this_labeled == match_this_Label] = match_last_label
            is_last_justified[index + 1] = True
            is_this_justified[match_this_Label] = True

        # ***********************************************新增晶粒判断**********************************************
        # 仍有未标注的新增晶粒或噪声，则按照新增值处理
        for this_index in range(0, len(this_regions)):
            if is_this_justified[this_index + 1]:
                continue
            label_num = label_num + 1
            track_labeled[this_labeled == this_index + 1] = label_num

        return track_labeled

    def _find_overlap_domin_label(self, last_labeled, this_labeled, target_this_label, match_threshold=0,
                                  compare_metric=0):
        """
        return overlap domian label for this_label in last slice
        返回本层截面中某个晶粒在上层中重叠的主要晶粒编号
        """
        if compare_metric == 0:  # compare by max overlap area 根据重叠面积最大比较
            match_label, max_area = None, -1
            # 计算上层中面积重叠区域
            overlap_labels = np.unique(last_labeled[this_labeled == target_this_label])
            for overlap_label in overlap_labels:
                overlap_area = np.count_nonzero(last_labeled[this_labeled == target_this_label] == overlap_label)
                if overlap_area > max_area:
                    max_area = overlap_area
                    match_label = overlap_label
            if max_area > match_threshold:
                return match_label, max_area
            else:
                return None, 0
        elif compare_metric == 1:  # compare by max iou 根据IOU最大比较
            match_label, max_iou = None, -1
            # 计算上层中晶粒的IOU
            overlap_labels = np.unique(last_labeled[this_labeled == target_this_label])
            for overlap_label in overlap_labels:
                overlap_area = np.count_nonzero(last_labeled[this_labeled == target_this_label] == overlap_label)
                union_area = np.count_nonzero((this_labeled == target_this_label) | (last_labeled == overlap_label))
                iou = overlap_area / union_area
                if iou > max_iou:
                    max_iou = iou
                    match_label = overlap_label
            if max_iou > match_threshold:
                return match_label, max_iou
            else:
                return None, 0

    def _justify_last_labeled(self, last_labeled, this_labeled, next_labeled):
        """
        Function: justify last label according to whether or not have one slice grain
        功能：判断本层的各个晶粒是否为单层晶粒，是的话根据上层晶粒质心最近值矫正
        :param last_label: np.array 上层标记
        :param this_label: np.array 本层标记
        :param next_label: np.array 下层标记
        :return: (is_find_slg, justify_this_label) return whether or not find single slice grain and justify result
        返回是否发现单层晶粒以及矫正结果
        """
        # is_find_slg(Singlelayergrain) 判断上层是否有单层晶粒
        is_find_slg = False

        last_props = regionprops(last_labeled, cache=True)
        # 记录上层每个晶粒的质心点坐标
        last_centroid = np.zeros((len(last_props), 2), dtype=np.float32)
        for i in range(len(last_props)):
            last_centroid[i, :] = last_props[i].centroid

        this_props = regionprops(this_labeled, cache=True)
        justify_this_labeled = this_labeled.copy()
        this_num = len(this_props)

        for this_Index in range(this_num):
            target_label = this_props[this_Index].label
            # 首先判断本标记是否为单层标记，即这个晶粒只有一层
            is_slg = False
            if target_label not in last_labeled and target_label not in next_labeled:
                is_slg = True
                is_find_slg = True
                if is_slg:
                    match_label, max_match_ratio = None, -1

                    temp_match_label, temp_max_match_ratio = self._find_overlap_domin_label(last_labeled, this_labeled,
                                                                                            target_label)
                    if temp_match_label and temp_max_match_ratio > max_match_ratio:
                        match_label = temp_match_label
                        max_match_ratio = temp_max_match_ratio

                    temp_match_label, temp_max_match_ratio = self._find_overlap_domin_label(next_labeled, this_labeled,
                                                                                            target_label)
                    if temp_match_label and temp_max_match_ratio > max_match_ratio:
                        match_label = temp_match_label
                        max_match_ratio = temp_max_match_ratio

                    if match_label:
                        justify_this_labeled[this_labeled == target_label] = match_label

        # 返回是否发现单层晶粒以及矫正结果
        return is_find_slg, justify_this_labeled

    def _print_screen_log(self, content):
        """
        Function: print content if self._verbose is True
        如果self._verbose为真，则打印信息
        :param content: str
        """
        if self._verbose:
            print(content)
