import os
import cv2
import h5py
import numpy as np
from segmentation.tools import tailor, download_from_url


cwd = os.getcwd()

# We generate train and val data from "pure_iron_grain_data_sets.hdf5"
# 我们从 "pure_iron_grain_data_sets.hdf5"生成训练和验证数据
data_dir = "pure_iron_grain_data_sets.hdf5"
if not os.path.exists(data_dir):
    download_from_url(url='https://drive.google.com/u/0/uc?export=download&confirm=CMbO&id=1iDHpLFRvyUA2Sdu8QNcO25NnGaogE8Pv', filename=data_dir, save_dir=cwd)
file_h5 = h5py.File(data_dir, "r")

real_image = file_h5['image']
real_label = file_h5["label"]
real_boundary = file_h5["boundary"]
# simulated_label = file_h5["simulated"]["label"]
# simulated_boundary = file_h5["simulated"]["boundary"]

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
