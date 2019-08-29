import os
import cv2
import h5py
import numpy as np
from grain_track import utility
from segmentation.tools import tailor


cwd = os.getcwd()

# We generate train and val data from "pure_iron_grain_data_sets.hdf5"
# 我们从 "pure_iron_grain_data_sets.hdf5"生成训练和验证数据
file_h5 = h5py.File("pure_iron_grain_data_sets.hdf5", "r")

real_image = file_h5['real']['image']
real_label = file_h5["real"]["label"]
real_boundary = file_h5["real"]["boundary"]
simulated_label = file_h5["simulated"]["label"]
simulated_boundary = file_h5["simulated"]["boundary"]

print(real_image.shape, ' ', real_label.shape, ' ', real_boundary.shape)

# ************************************************* Segmentation ********************************************************
output_train_dir = os.path.join(cwd, "datasets", "segmentation", "net_train", "train")
output_train_crop_dir = os.path.join(cwd, "datasets", "segmentation", "net_train", "train_crop")
output_val_dir = os.path.join(cwd, "datasets", "segmentation", "net_train", "val")
output_val_crop_dir = os.path.join(cwd, "datasets", "segmentation", "net_train", "val_crop")
output_test_dir = os.path.join(cwd, "datasets", "segmentation", "net_test", "test")
output_test_crop_dir = os.path.join(cwd, "datasets", "segmentation", "net_test", "test_overlap_crop")

if not os.path.exists(output_train_dir):
    os.makedirs(output_train_dir)
    os.mkdir(os.path.join(output_train_dir, 'images'))
    os.mkdir(os.path.join(output_train_dir, 'labels'))

if not os.path.exists(output_train_crop_dir):
    os.makedirs(output_train_crop_dir)
    os.mkdir(os.path.join(output_train_crop_dir, 'images'))
    os.mkdir(os.path.join(output_train_crop_dir, 'labels'))

if not os.path.exists(output_val_dir):
    os.makedirs(output_val_dir)
    os.mkdir(os.path.join(output_val_dir, 'images'))
    os.mkdir(os.path.join(output_val_dir, 'labels'))

if not os.path.exists(output_val_crop_dir):
    os.makedirs(output_val_crop_dir)
    os.mkdir(os.path.join(output_val_crop_dir, 'images'))
    os.mkdir(os.path.join(output_val_crop_dir, 'labels'))

if not os.path.exists(output_test_dir):
    os.makedirs(output_test_dir)
    os.mkdir(os.path.join(output_test_dir, 'images'))
    os.mkdir(os.path.join(output_test_dir, 'labels'))

if not os.path.exists(output_test_crop_dir):
    os.makedirs(output_test_crop_dir)
    os.mkdir(os.path.join(output_test_crop_dir, 'images'))
    os.mkdir(os.path.join(output_test_crop_dir, 'labels'))

# save image 0:116 to train_set | image 116: 148 to val_set | image 148:296 to test_set

print('******************** train set **************************')
# train set
for item in range(0, 116):
    name = str(item+1).zfill(3) + '.png'
    image = real_image[:,:,item]
    label = real_boundary[:,:,item]
    print(name, ' ', image.shape, ' ', label.shape, ' ', np.unique(label))
    cv2.imwrite(os.path.join(output_train_dir, 'images', name), image)
    cv2.imwrite(os.path.join(output_train_dir, 'labels', name), label)

    # crop
    tailor(256, 256, os.path.join(output_train_dir, 'images', name), os.path.join(output_train_crop_dir, 'images'), region = 32)
    tailor(256, 256, os.path.join(output_train_dir, 'labels', name), os.path.join(output_train_crop_dir, 'labels'), region = 32)

print('******************** val set **************************')
# val set
for item in range(116, 148):
    name = str(item+1).zfill(3) + '.png'
    image = real_image[:,:,item]
    label = real_boundary[:,:,item]
    print(name, ' ', image.shape, ' ', label.shape, ' ', np.unique(label))
    cv2.imwrite(os.path.join(output_val_dir, 'images', name), image)
    cv2.imwrite(os.path.join(output_val_dir, 'labels', name), label)

    # crop
    tailor(256, 256, os.path.join(output_val_dir, 'images', name), os.path.join(output_val_crop_dir, 'images'), region = 0)
    tailor(256, 256, os.path.join(output_val_dir, 'labels', name), os.path.join(output_val_crop_dir, 'labels'), region = 0)

print('******************** test set **************************')
# test set
for item in range(148, 296):
    name = str(item+1).zfill(3) + '.png'
    image = real_image[:,:,item]
    label = real_boundary[:,:,item]
    print(name, ' ', image.shape, ' ', label.shape, ' ', np.unique(label))
    cv2.imwrite(os.path.join(output_test_dir, 'images', name), image)
    cv2.imwrite(os.path.join(output_test_dir, 'labels', name), label)

    # crop
    tailor(256, 256, os.path.join(output_test_dir, 'images', name), os.path.join(output_test_crop_dir, 'images'), region = 32)
    tailor(256, 256, os.path.join(output_test_dir, 'labels', name), os.path.join(output_test_crop_dir, 'labels'), region = 32)

# ************************************************* Grain Track ********************************************************
# For real data [1024, 1024, 296],
# train = data[:, :, 0:116], val = data[:,:,116:148], test = data[:, :, 148:]
# 对于真实数据，我们取前116层作为训练集，117-148（32层）作为验证集，149-296（148层）作为测试集
output_train_dir = os.path.join(cwd, "datasets", "grain_track", "net_train", "real", "train")
output_val_dir = os.path.join(cwd, "datasets", "grain_track", "net_train", "real", "val")
output_test_dir = os.path.join(cwd, "datasets", "grain_track", "net_test", "real")
output_test_boundary_dir = os.path.join(output_test_dir, "real_boundary")
utility.make_out_dir(output_train_dir)
utility.make_out_dir(output_val_dir)
utility.make_out_dir(output_test_dir)
utility.make_out_dir(output_test_boundary_dir)

real_train = real_label[:, :, 0: 116]
print("The shape of real_train is {}".format(real_train.shape))
positive_num, negative_num = utility.produce_two_slice_data(real_train, output_train_dir)
# positive_num = 31482, negative_num = 107435
print("For real train data, we have {} positive num and {} negative num.".format(positive_num, negative_num))

real_val = real_label[:, :, 116: 148]
print("The shape of real_val is {}".format(real_val.shape))
positive_num, negative_num = utility.produce_two_slice_data(real_val, output_val_dir)
# positive_num = 8901, negative_num = 29960
print("For real val data, we have {} positive num and {} negative num.".format(positive_num, negative_num))

real_test = real_label[:, :, 148:]
real_boundary = real_boundary[:, :, 148:]
print("The shape of real_test is {}".format(real_test.shape))
np.save(os.path.join(output_test_dir, "real_gt_label_stack.npy"), real_test)
print("For real test data, we save npy and its boundary")
h, w, d = real_boundary.shape
for index in range(d):
    cv2.imwrite(os.path.join(output_test_boundary_dir, str(148 + 1 + index).zfill(3) + ".png"), real_boundary[:,:, index])

# For simulated data [400, 400, 400],
# train = data[:, :, 0:240], val = data[:,:,240:320], test = data[:, :, 320:]
# 对于模拟数据，我们取前240层作为训练集，241-320（80层）作为验证集，321-400（80层）作为测试集
output_train_dir = os.path.join(cwd, "datasets", "grain_track", "net_train", "simulated", "train")
output_val_dir = os.path.join(cwd, "datasets", "grain_track", "net_train", "simulated", "val")
output_test_dir = os.path.join(cwd, "datasets", "grain_track", "net_test", "simulated")
output_test_boundary_dir = os.path.join(output_test_dir, "simulated_boundary")
utility.make_out_dir(output_train_dir)
utility.make_out_dir(output_val_dir)
utility.make_out_dir(output_test_dir)
utility.make_out_dir(output_test_boundary_dir)

simulated_train = simulated_label[:, :, 0: 240]
print("The shape of simulated_train is {}".format(simulated_train.shape))
positive_num, negative_num = utility.produce_two_slice_data(simulated_train, output_train_dir)
# positive_num = 42703, negative_num = 103785
print("For simulated train data, we have {} positive num and {} negative num.".format(positive_num, negative_num))

simulated_val = simulated_label[:, :, 240: 320]
print("The shape of simulated_val is {}".format(simulated_val.shape))
positive_num, negative_num = utility.produce_two_slice_data(simulated_val, output_val_dir)
# positive_num = 14616, negative_num = 35458
print("For simulated val data, we have {} positive num and {} negative num.".format(positive_num, negative_num))

simulated_test = simulated_label[:, :, 320:]
simulated_boundary = simulated_boundary[:, :, 320:]
print("The shape of simulated_test is {}".format(simulated_test.shape))
np.save(os.path.join(output_test_dir, "simulated_gt_label_stack.npy"), simulated_test)
print("For simulated test data, we save npy and its boundary")
h, w, d = simulated_boundary.shape
for index in range(d):
    cv2.imwrite(os.path.join(output_test_boundary_dir, str(320 + 1 + index).zfill(3) + ".png"), simulated_boundary[:,:, index])
