from grain_track import utility
from grain_track.inference_track_net import *
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
    model.load_state_dict(torch.load(model_path))
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


def grain_track_for_gt():
    """
    We evaluate the performance of different tracking methods on gt slices
    """
    cwd = os.getcwd()
    parameter_address = os.path.join(cwd, "grain_track", "parameter")
    if not os.path.exists(parameter_address):
        download_from_url(url='https://doc-08-3k-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/77lk68r7uoskmfilgeq0f39rvqpco6pm/1567137600000/03563112468744709654/*/1dhwSwmxDKBwub9Wi4DPXpJotnHrNOyaL?e=download',
                          filename='grain_track_parameters.zip', save_dir=os.path.join(cwd, 'grain_track/'))
    cnn_device = "cuda:0"

    # Performance of tracking on real data set with different algorithms.
    print("For real data")
    data_address = os.path.join(cwd, "datasets", "grain_track", "net_test", "real")
    input_address_pred = os.path.join(data_address, "real_boundary")
    input_address_gt = os.path.join(data_address, "real_gt_label_stack.npy")
    label_stack_gt = np.load(input_address_gt)
    print("The number of grain in GT is {}".format(len(np.unique(label_stack_gt) - 1)))
    grain_track = GrainTrack(input_address_pred, reverse_label=False)

    # method = 1 min centroid dis
    print("Analyzing by min centroid dis")
    start_time = time.time()
    label_stack_pred, label_num_pred = grain_track.get_tracked_label_stack(method=1)
    end_time = time.time()
    print("The number of grain is {}".format(label_num_pred))
    r_index, adjust_r_index, v_index, merger_error, split_error = utility.validate_label_stack_by_rvi(label_stack_pred,
                                                                                                      label_stack_gt)
    print("The ri is {:.8f}, ari is {:.8f}, vi is {:.8f}, merger_error is {:.8f}, split_error is {:.8f}"
          .format(r_index, adjust_r_index, v_index, merger_error, split_error))
    print("The duriation of min centroid dis is {:.2f}'s".format(end_time - start_time))
    np.save(os.path.join(data_address, "real_gt_min_centroid_dis_label_stack.npy"), label_stack_pred)

    # method = 2 max overlap area
    print("Analyzing by max overlap area")
    start_time = time.time()
    label_stack_pred, label_num_pred = grain_track.get_tracked_label_stack(method=2)
    end_time = time.time()
    print("The number of grain is {}".format(label_num_pred))
    r_index, adjust_r_index, v_index, merger_error, split_error = utility.validate_label_stack_by_rvi(label_stack_pred,
                                                                                                      label_stack_gt)
    print("The ri is {:.8f}, ari is {:.8f}, vi is {:.8f}, merger_error is {:.8f}, split_error is {:.8f}"
          .format(r_index, adjust_r_index, v_index, merger_error, split_error))
    print("The duriation of max overlap area is {:.2f}'s".format(end_time - start_time))
    np.save(os.path.join(data_address, "real_gt_max_overlap_area_label_stack.npy"), label_stack_pred)

    # method = 3 cnn vgg13_bn
    print("Analyzing by vgg13_bn")
    start_time = time.time()
    grain_track.set_cnn_tracker(model=0, pretrain_address=os.path.join(parameter_address, "real_vgg13_bn.pkl"),
                                device=cnn_device, need_augment=False, max_num_tensor=30)
    label_stack_pred, label_num_pred = grain_track.get_tracked_label_stack(method=3)
    end_time = time.time()
    print("The number of grain is {}".format(label_num_pred))
    r_index, adjust_r_index, v_index, merger_error, split_error = utility.validate_label_stack_by_rvi(label_stack_pred,
                                                                                                      label_stack_gt)
    print("The ri is {:.8f}, ari is {:.8f}, vi is {:.8f}, merger_error is {:.8f}, split_error is {:.8f}"
          .format(r_index, adjust_r_index, v_index, merger_error, split_error))
    print("The duriation of vgg13_bn is {:.2f}'s".format(end_time - start_time))
    np.save(os.path.join(data_address, "real_gt_vgg13_bn_label_stack.npy"), label_stack_pred)

    # method = 3 cnn densenet161
    print("Analyzing by densenet161")
    start_time = time.time()
    grain_track.set_cnn_tracker(model=1, pretrain_address=os.path.join(parameter_address, "real_densenet161.pkl"),
                                device=cnn_device, need_augment=False, max_num_tensor=30)
    label_stack_pred, label_num_pred = grain_track.get_tracked_label_stack(method=3)
    end_time = time.time()
    print("The number of grain is {}".format(label_num_pred))
    r_index, adjust_r_index, v_index, merger_error, split_error = utility.validate_label_stack_by_rvi(label_stack_pred,
                                                                                                      label_stack_gt)
    print("The ri is {:.8f}, ari is {:.8f}, vi is {:.8f}, merger_error is {:.8f}, split_error is {:.8f}"
          .format(r_index, adjust_r_index, v_index, merger_error, split_error))
    print("The duriation of densenet161 is {:.2f}'s".format(end_time - start_time))
    np.save(os.path.join(data_address, "real_gt_densenet161_label_stack.npy"), label_stack_pred)

    # Performance of tracking on real data set with different algorithms.
    print("For simulated data")
    data_address = os.path.join(cwd, "datasets", "grain_track", "net_test", "simulated")
    input_address_pred = os.path.join(data_address, "simulated_boundary")
    input_address_gt = os.path.join(data_address, "simulated_gt_label_stack.npy")
    label_stack_gt = np.load(input_address_gt)
    print("The number of grain in GT is {}".format(len(np.unique(label_stack_gt) - 1)))
    grain_track = GrainTrack(input_address_pred, reverse_label=False)

    # method = 1 min centroid dis
    print("Analyzing by min centroid dis")
    start_time = time.time()
    label_stack_pred, label_num_pred = grain_track.get_tracked_label_stack(method=1)
    end_time = time.time()
    print("The number of grain is {}".format(label_num_pred))
    r_index, adjust_r_index, v_index, merger_error, split_error = utility.validate_label_stack_by_rvi(label_stack_pred,
                                                                                                      label_stack_gt)
    print("The ri is {:.8f}, ari is {:.8f}, vi is {:.8f}, merger_error is {:.8f}, split_error is {:.8f}"
          .format(r_index, adjust_r_index, v_index, merger_error, split_error))
    print("The duriation of min centroid dis is {:.2f}'s".format(end_time - start_time))
    np.save(os.path.join(data_address, "simulated_gt_min_centroid_dis_label_stack.npy"), label_stack_pred)

    # method = 2 max overlap area
    print("Analyzing by max overlap area")
    start_time = time.time()
    label_stack_pred, label_num_pred = grain_track.get_tracked_label_stack(method=2)
    end_time = time.time()
    print("The number of grain is {}".format(label_num_pred))
    r_index, adjust_r_index, v_index, merger_error, split_error = utility.validate_label_stack_by_rvi(label_stack_pred,
                                                                                                      label_stack_gt)
    print("The ri is {:.8f}, ari is {:.8f}, vi is {:.8f}, merger_error is {:.8f}, split_error is {:.8f}"
          .format(r_index, adjust_r_index, v_index, merger_error, split_error))
    print("The duriation of max overlap area is {:.2f}'s".format(end_time - start_time))
    np.save(os.path.join(data_address, "simulated_gt_max_overlap_area_label_stack.npy"), label_stack_pred)

    # method = 3 cnn vgg13_bn
    print("Analyzing by vgg13_bn")
    start_time = time.time()
    grain_track.set_cnn_tracker(model=0, pretrain_address=os.path.join(parameter_address, "simulated_vgg13_bn.pkl"),
                                device=cnn_device, need_augment=False, max_num_tensor=30)
    label_stack_pred, label_num_pred = grain_track.get_tracked_label_stack(method=3)
    end_time = time.time()
    print("The number of grain is {}".format(label_num_pred))
    r_index, adjust_r_index, v_index, merger_error, split_error = utility.validate_label_stack_by_rvi(label_stack_pred,
                                                                                                      label_stack_gt)
    print("The ri is {:.8f}, ari is {:.8f}, vi is {:.8f}, merger_error is {:.8f}, split_error is {:.8f}"
          .format(r_index, adjust_r_index, v_index, merger_error, split_error))
    print("The duriation of vgg13_bn is {:.2f}'s".format(end_time - start_time))
    np.save(os.path.join(data_address, "simulated_gt_vgg13_bn_label_stack.npy"), label_stack_pred)

    # method = 3 cnn densenet161
    print("Analyzing by densenet161")
    start_time = time.time()
    grain_track.set_cnn_tracker(model=1, pretrain_address=os.path.join(parameter_address, "simulated_densenet161.pkl"),
                                device=cnn_device, need_augment=False, max_num_tensor=30)
    label_stack_pred, label_num_pred = grain_track.get_tracked_label_stack(method=3)
    end_time = time.time()
    print("The number of grain is {}".format(label_num_pred))
    r_index, adjust_r_index, v_index, merger_error, split_error = utility.validate_label_stack_by_rvi(label_stack_pred,
                                                                                                      label_stack_gt)
    print("The ri is {:.8f}, ari is {:.8f}, vi is {:.8f}, merger_error is {:.8f}, split_error is {:.8f}"
          .format(r_index, adjust_r_index, v_index, merger_error, split_error))
    print("The duriation of densenet161 is {:.2f}'s".format(end_time - start_time))
    np.save(os.path.join(data_address, "simulated_gt_densenet161_label_stack.npy"), label_stack_pred)


def grain_track_for_real_pred():
    """
    We evaluate the performance of WPU-net using densenet161 on real test set(149 - 296). Because Unet-Bdelstm needs
    3 slices as input and output 1 slice segmentation result. Thus, for fair comparison, we only track and analyse 150 - 295 slices.
    """
    cwd = os.getcwd()

    # Prepapre gt label stack
    data_address = os.path.join(cwd, "datasets", "grain_track", "net_test", "real")
    input_address_gt = os.path.join(data_address, "real_gt_label_stack.npy")
    label_stack_gt = np.load(input_address_gt)[:, :, 1: -1]  # 150 - 295
    label_stack_gt, label_num_gt = label(label_stack_gt, return_num=True)
    print("The number of grain in GT is {} and the shape is {}".format(label_num_gt, label_stack_gt.shape))

    parameter_address = os.path.join(cwd, "grain_track", "parameter")
    if not os.path.exists(parameter_address):
        download_from_url(url='https://drive.google.com/file/d/1dhwSwmxDKBwub9Wi4DPXpJotnHrNOyaL/view?usp=sharing',
                          filename='grain_track_parameters.zip', save_dir=os.path.join(cwd, 'grain_track/'))

    pretrain_address = os.path.join(cwd, "grain_track", "parameter",
                                    "real_densenet161.pkl")  # we only test this tracking method
    cnn_device = "cuda:0"

    result_dir = os.path.join(cwd, "segmentation", "result_total")
    methods_list = os.listdir(result_dir)
    for method_item in methods_list:
        print("For " + method_item)
        method_address = os.path.join(result_dir, method_item)
        grain_track = GrainTrack(method_address, reverse_label=False)
        grain_track.set_cnn_tracker(model=1, pretrain_address=pretrain_address, device=cnn_device, need_augment=False,
                                    max_num_tensor=30)
        label_stack_pred, label_num_pred = grain_track.get_tracked_label_stack(method=3)
        np.save(os.path.join(data_address, "real_" + method_item + "_densenet161_label_stack.npy"), label_stack_pred)
        print("The number of grain is {} and the shape is {}".format(label_num_pred, label_stack_gt.shape))
        r_index, adjust_r_index, v_index, merger_error, split_error = utility.validate_label_stack_by_rvi(
            label_stack_pred, label_stack_gt)
        print("The ri is {:.8f}, ari is {:.8f}, vi is {:.8f}, merger_error is {:.8f}, split_error is {:.8f}"
              .format(r_index, adjust_r_index, v_index, merger_error, split_error))


if __name__ == "__main__":
    print('############## segmentation inference  ##############')
    inference()
    print('############## grain_track_for_gt  ##############')
    grain_track_for_gt()
    print('############## grain_track_for_real_pred  ##############')
    grain_track_for_real_pred()