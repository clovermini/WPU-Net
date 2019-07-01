import os
import cv2
import numpy as np
import random
from skimage.measure import label, regionprops
from skimage import morphology as sm
import uuid
import gala.evaluate as ev


def make_out_dir(path):
    """
    Function: Create the dir of folder
    功能：创建文件夹目录，先判断文件夹是否存在，不存在则创建
    :param path: path of dir文件夹目录
    :return:
    """
    try:
        os.makedirs(path)
    except OSError:
        pass


def get_label_stack_from_npy(file_address):
    """
    Function: read ndarry from file_address
    功能：从npy文件中读取三维矩阵并返回
    :param file_address: the address of npy 文件地址
    :return: (label_stack, label_num), return the 3d label stack and number of grain 返回的三维矩阵, 数目
    """
    label_stack = np.load(file_address)
    label_num = int(np.unique(label_stack).shape[0])
    return label_stack, label_num


def dilate_target(mask, iteration=1):
    """
    Function: dilate image
    功能：使用边长为2的正方形结构元素膨胀
    :param mask: np.ndarry
    :param iteration: iteration of operation
    :return: mask np.ndarry
    """
    for i in range(0, iteration):
        mask = sm.dilation(mask, sm.square(2))  # we use kernel size of 2 to dilate 用边长为2的正方形结构元素进行膨胀
    return mask


def get_label_from_boundary(image_address):
    '''

    功能：根据边缘图像生成标记结果，颜色区域扩张，用于处理晶界，获取晶粒，同时可以消除“毛刺”
    :param image_address: 待处理图像
    :return: label_nonEdge ：去除晶界和毛刺的标记图number ：标记的数目
    '''
    image = cv2.imread(image_address, 0)
    #  image = dilate_target(image)
    image[0, :] = 255
    image[-1, :] = 255
    image[:, 0] = 255
    image[:, -1] = 255
    # 获取label图像
    (labeled, number) = label(image, background=255, neighbors=4, return_num=True)
    props = regionprops(labeled)
    props_sorted = sorted(props, key=lambda props: props.area, reverse=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    label_nonEdge = np.zeros((image.shape[0], image.shape[1]), np.int)

    for prop in props_sorted:
        temp = np.zeros((image.shape[0], image.shape[1]))
        temp[labeled == prop.label] = 255
        temp = cv2.dilate(temp, kernel)
        label_nonEdge[temp == 255] = prop.label
        del temp

    # delete small background 去除某些情况下存在的背景像素点
    while 0 in label_nonEdge:
        locations = np.argwhere(label_nonEdge == 0)
        for item in locations:
            if item[0] != 0 and label_nonEdge[item[0] - 1, item[1]] != 0:
                label_nonEdge[item[0], item[1]] = label_nonEdge[item[0] - 1, item[1]]
            elif item[0] != image.shape[0] - 1 and label_nonEdge[item[0] + 1, item[1]] != 0:
                label_nonEdge[item[0], item[1]] = label_nonEdge[item[0] + 1, item[1]]
            elif item[1] != 0 and label_nonEdge[item[0], item[1] - 1] != 0:
                label_nonEdge[item[0], item[1]] = label_nonEdge[item[0], item[1] - 1]
            elif item[1] != image.shape[1] - 1 and label_nonEdge[item[0], item[1] + 1] != 0:
                label_nonEdge[item[0], item[1]] = label_nonEdge[item[0], item[1] + 1]

    # newEdges = getEdgesFromLabel(label_nonEdge)
    return label_nonEdge, number


def get_boundary_from_label(labeled, neighbors=4):
    """
    Get boundary map from labeled image
    Each pixel have its own label. If one pixel find different label in its neighborhood,
    the pixel should be boundary pixel and set to 255. Otherwise, it should be set to 0.
    从标注图像中获得边缘图像
    每个像素均有其自己的label,如果在一个像素的领域内找到与其不同的label,
    说明该像素是边缘像素，应该置为255，否则，应被置为0
    :param labeled: np.array
        The labeled image.标注图像
    :param neighbors: {4, 8}, int, optional
        Default 4 neighbors for pixel neighborhood.连通区域类型，默认为4邻接
    :return: edges, np.array
        The edge map for labeled.返回标注图对应的边缘图像
    """
    boundary = np.zeros(labeled.shape, dtype=np.uint8)
    for row in range(0, labeled.shape[0]):
        for col in range(0, labeled.shape[1]):
            value = labeled[row, col]
            is_boundary = False
            if value == 0:
                boundary[row, col] = 255
                continue
            if neighbors == 4:
                if row != 0 and value != labeled[row - 1, col]:
                    is_boundary = True
                elif col != 0 and value != labeled[row, col - 1]:
                    is_boundary = True
                elif row != labeled.shape[0] - 1 and value != labeled[row + 1, col]:
                    is_boundary = True
                elif col != labeled.shape[1] - 1 and value != labeled[row, col + 1]:
                    is_boundary = True
            elif neighbors == 8:
                if row != 0 and value != labeled[row - 1, col]:
                    is_boundary = True
                elif row != 0 and col != 0 and value != labeled[row - 1, col - 1]:
                    is_boundary = True
                elif row != 0 and col != labeled.shape[1] - 1 and value != labeled[row - 1, col + 1]:
                    is_boundary = True
                elif row != labeled.shape[0] - 1 and value != labeled[row + 1, col]:
                    is_boundary = True
                elif row != labeled.shape[0] - 1 and col != 0 and value != labeled[row + 1, col - 1]:
                    is_boundary = True
                elif row != labeled.shape[0] - 1 and col != labeled.shape[1] - 1 and value != labeled[row + 1, col + 1]:
                    is_boundary = True
                elif col != 0 and value != labeled[row, col - 1]:
                    is_boundary = True
                elif col != labeled.shape[1] - 1 and value != labeled[row, col + 1]:
                    is_boundary = True
            if is_boundary:
                boundary[row, col] = 255
    return boundary


def dyeing_and_section(out_dir, image_stack):
    '''
    功能：对标记的三维结构进行染色，给每个晶粒分配一个独特的颜色，然后将三维结构切片，每一层截面图片按照label染上相应的颜色，生成RGB图像，用于后续的可视化。首先生成与晶粒个数相等个数的颜色，然后进行颜色分配和切片。
    :param out_dir: 生成的系列rgb图像保存地址，最好为绝对地址
    :param image_stack: 待处理的三维结构，是一个三维标记数组，其中值相同的像素点为同一个晶粒 [w, h, depth]
    :return: "success"
    '''
    w, h, depth = image_stack.shape

    labels = np.unique(image_stack)  # [1, 2, 3, 4, 5]
    labels = list(map(str, labels))  # ['1', '2', '3', '4', '5']
    label_num = len(labels)

    # 生成不重复的颜色序列
    color_list = []
    while len(color_list) < label_num:
        r = random.randint(1, 255)
        g = random.randint(1, 255)
        b = random.randint(1, 255)
        rgb = str(r) + "," + str(g) + "," + str(b)
        color_list.append(rgb)
        color_list = list({}.fromkeys(color_list).keys())

    # 给每个晶粒分配一个独特的颜色
    # {'1': '113,148,191', '2': '171,242,3', '3': '198,243,70', '4': '226,150,130', '5': '6,186,66'}
    label_color_dict = dict(zip(labels, color_list))

    # 切片
    for item in range(0, depth):
        image_label = image_stack[:, :, item]
        image_label_num = np.unique(image_label)
        image_rgb = np.zeros((w, h, 3))
        for num in image_label_num:
            image_rgb[image_label == num] = label_color_dict[str(
                num)].split(",")  # ['113', '148', '191']
        image_rgb[image_label == 0] = [0, 0, 0]
        cv2.imwrite(
            os.path.join(
                out_dir, str(
                    item + 1).zfill(3) + ".png"), image_rgb)

    return "success"


def validate_label_stack_by_rvi(pred_stack, gt_stack):
    '''
    Get segment error， 根据信息论评估方法（RI，VI）评估追踪结果
    :param pred_stack: ndarry
        Predict result
    :param gt_stack: ndarry
        GroundTruth result
    :return: r_index, adjust_r_index, v_index, merger_error, split_error
    '''
    r_index = ev.rand_index(pred_stack, gt_stack)
    adjust_r_index = ev.adj_rand_index(pred_stack, gt_stack)
    merger_error, split_error = ev.split_vi(pred_stack, gt_stack)
    v_index = merger_error + split_error
    return r_index, adjust_r_index, v_index, merger_error, split_error


def crop_two_slice_data(last_image, last_label, this_image, this_label):
    """
    Crop two slice data from two images and two labels, 从相邻两层图像中截取训练和测试数据
    :param last_image:np.array
    :param last_label:int
    :param this_image:np.array
    :param this_label:int
    :return: np.array
    """
    h, w = last_image.shape
    temp = np.zeros((h, w, 2), dtype=np.uint8)
    temp[:, :, 0][last_image == last_label] = 255
    temp[:, :, 1][this_image == this_label] = 255
    temp_label = label(temp, background=0)
    props = regionprops(temp_label, coordinates="rc", cache=True)
    min_row, min_col, min_depth, max_row, max_col, max_depth = props[0].bbox
    return temp[min_row:max_row, min_col: max_col, min_depth: max_depth]


def produce_two_slice_data(label_stack, out_dir):
    """
    Produce two slice data (npy) from label_stack, the first slice contain one grain and the
    second slice contain the grain which is connect with above grain
    从label_stack中生成多个两层数据文件（npy），第一层是上一层某个晶粒的截面，第二层是与之在三维上连通的
    一个晶粒的截面，在out_dir下建立“0”,"1"两个文件夹，分别存储追踪成功
    和追踪错误的数据
    :param label_stack:3d array
    :param out_dir:str, the output of data
    :return:
    """
    positive_num, negative_num = 0, 0
    h, w, d = label_stack.shape
    # analysis every slice 遍历所有层
    for d_index in range(1, d):
        print("crop {}th slice".format(d_index))
        last_image = label_stack[:, :, d_index - 1]
        this_image = label_stack[:, :, d_index]
        this_labels = np.unique(this_image).tolist()
        # analysis every grain 遍历每层所有晶粒
        for this_label in this_labels:
            overlap_last_labels = np.unique(last_image[this_image == this_label])
            # 寻找每个晶粒与之重合的上层晶粒
            for overlap_last_label in overlap_last_labels:
                label_data = crop_two_slice_data(
                    last_image, overlap_last_label, this_image, this_label)
                h, w, d = label_data.shape
                if h < 2 or w < 2 or d < 2:
                    continue
                if overlap_last_label == this_label:
                    positive_num += 1
                    data_class = "0"
                else:
                    negative_num += 1
                    data_class = "1"
                temp_out_dir = os.path.join(out_dir, data_class)
                make_out_dir(temp_out_dir)
                file_name = str(uuid.uuid4())
                np.save(os.path.join(temp_out_dir, file_name) + ".npy", label_data)
    return positive_num, negative_num
