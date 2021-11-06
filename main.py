from skimage.measure import label
import cv2
import os, copy
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as tr
from segmentation.model import UNet, WPU_Net
from segmentation.tools import posprecess, stitch, proprecess_img, tailor, download_from_url
import time
from segmentation.evaluation import eval_RI_VI, eval_F_mapKaggle

def inference():
    '''
    predict results and evaluate on test samples for segmentation with wpu-net
    '''
    transform = tr.Compose([
        tr.ToTensor(),
        tr.Normalize(mean=[0.9336267],  # RGB
                     std=[0.1365774])
    ])


    cwd = os.getcwd()
    output_test_dir = os.path.join(cwd, "datasets", "segmentation", "net_test", "test")

    # generate overlapping cropped image for test
    output_test_crop_dir = os.path.join(cwd, "datasets", "segmentation", "net_test", "test_overlap_crop")
    if not os.path.exists(output_test_crop_dir):
        os.mkdir(output_test_crop_dir)
        os.mkdir(os.path.join(output_test_crop_dir, 'images'))
        os.mkdir(os.path.join(output_test_crop_dir, 'labels'))
    for item in sorted(os.listdir(os.path.join(output_test_dir, 'images'))):
        tailor(256, 256, os.path.join(output_test_dir, 'images', item), os.path.join(output_test_crop_dir, 'images'),region=32)
        tailor(256, 256, os.path.join(output_test_dir, 'labels', item), os.path.join(output_test_crop_dir, 'labels'),region=32)

    key_name = 'WPU_Net_model'

    if not os.path.exists(os.path.join(cwd, 'segmentation', 'parameter')):
        download_from_url(url='https://doc-0k-3k-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/c4ag025396gf90cp0rt7vcaci3ml4teo/1567137600000/03563112468744709654/*/1Gc2j-DrJhX0E4fnvRItf95o0BXWQa-wr?e=download',
                          filename='wpu_net_parameters.zip', save_dir=os.path.join(cwd, 'segmentation'))

    model_path = os.path.join(cwd, 'segmentation', 'parameter', key_name, 'best_model_state.pth')
    result_save_dir = os.path.join(cwd, 'segmentation', 'result', key_name)
    result_total_save_dir = os.path.join(cwd, 'segmentation', 'result_total', key_name)

    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)

    if not os.path.exists(result_total_save_dir):
        os.makedirs(result_total_save_dir)

    model = WPU_Net(num_channels=2, num_classes=2, multi_layer=True)

    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()

    # 先加载模型参数dict文件
    state_dict = torch.load(model_path)
    from collections import OrderedDict
    # 初始化一个空 dict
    new_state_dict = OrderedDict()
    # 修改 key，没有module字段则需要不上，如果有，则需要修改为 module.features
    for k, v in state_dict.items():
        if 'module' not in k:
            k = 'module.' + k
        else:
            k = k.replace('features.module.', 'module.features.')
        new_state_dict[k] = v
    # 加载修改后的新参数dict文件
    model.load_state_dict(new_state_dict)

    # model.load_state_dict(torch.load(model_path))
    model.eval()

    # inferece cropped images
    images = sorted(os.listdir(os.path.join(output_test_crop_dir, "images")))
    print(len(images))
    start_time = time.time()
    count = 0
    min_file = 149  # 1  117  149
    max_file = 296  # 116  148  296
    for item in images:
        if item.endswith(".png"):
            filename = item.split(".")[0]
            pic_num = item.split("_")[0]
            if int(pic_num) > min_file and int(pic_num) <= max_file:
                count += 1
                test_image = os.path.join(output_test_crop_dir, "images", filename + ".png")
                img = proprecess_img(test_image)
                # last mask
                last_name = str(int(pic_num) - 1).zfill(3) + '_' + filename.split('_')[1] + '_' + filename.split('_')[2]
                last_mask = cv2.imread(os.path.join(output_test_crop_dir, "labels", last_name + ".png"), 0)
                last_tensor = torch.Tensor(np.array(last_mask)).unsqueeze(0).unsqueeze(0)
                last_tensor[last_tensor == 255] = -6
                last_tensor[last_tensor == 0] = 1
                output = model(inputs=img, last=last_tensor)
                result_npy = posprecess(output, close=True)
                cv2.imwrite(os.path.join(result_save_dir, filename + ".png"), result_npy)

    end_time = time.time()
    average_time = (end_time - start_time) / count
    print("end ...", average_time)

    # stitch cropped images
    imgList = sorted(os.listdir(os.path.join(output_test_dir, "images")))
    print(len(imgList))
    n = 0
    for img in imgList:
        if img.endswith(".png"):
            name = img.split(".")[0]
            if int(name) > min_file and int(name) <= max_file:
                print('you are stitching picture ', name)
                stitch(256, 256, name, result_save_dir, os.path.join(result_total_save_dir, name + ".png"), 32)
                n += 1
    print("end...")

    # evaluate
    RI_save_dir = os.path.join(cwd, 'segmentation', 'evaluations', 'big_RI_VI')
    Map_save_dir = os.path.join(cwd, 'segmentation', 'evaluations', 'big_F_mAP')

    if not os.path.exists(RI_save_dir):
        os.makedirs(RI_save_dir)

    if not os.path.exists(Map_save_dir):
        os.makedirs(Map_save_dir)

    print(key_name + " model " + "#####" * 20)

    eval_RI_VI(os.path.join(cwd, 'segmentation', 'result_total', key_name),  os.path.join(RI_save_dir, key_name + ".txt"), gt_dir=os.path.join(cwd, 'datasets', 'segmentation', 'net_test', 'test', 'labels'))
    eval_F_mapKaggle(os.path.join(cwd, 'segmentation', 'result_total', key_name), os.path.join(Map_save_dir, key_name + ".txt"), gt_dir=os.path.join(cwd, 'datasets', 'segmentation', 'net_test', 'test', 'labels'))




if __name__ == "__main__":
    print('############## segmentation inference  ##############')
    inference()
    print('##############   inference completed   ##############')