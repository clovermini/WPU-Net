import os
import time
import numpy as np
from grain_track import utility
from grain_track.inference_track_net import *

cwd = os.getcwd()
parameter_address = os.path.join(cwd, "parameter")
cnn_device = "cuda:0"

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
r_index, adjust_r_index, v_index, merger_error, split_error = utility.validate_label_stack_by_rvi(label_stack_pred, label_stack_gt)
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
r_index, adjust_r_index, v_index, merger_error, split_error = utility.validate_label_stack_by_rvi(label_stack_pred, label_stack_gt)
print("The ri is {:.8f}, ari is {:.8f}, vi is {:.8f}, merger_error is {:.8f}, split_error is {:.8f}"
      .format(r_index, adjust_r_index, v_index, merger_error, split_error))
print("The duriation of max overlap area is {:.2f}'s".format(end_time - start_time))
np.save(os.path.join(data_address, "real_gt_max_overlap_area_label_stack.npy"), label_stack_pred)

# method = 3 cnn vgg13_bn
print("Analyzing by vgg13_bn")
start_time = time.time()
grain_track.set_cnn_tracker(model=0, pretrain_address=os.path.join(parameter_address, "real_vgg13_bn.pkl"), device=cnn_device, need_augment=False, max_num_tensor=30)
label_stack_pred, label_num_pred = grain_track.get_tracked_label_stack(method=3)
end_time = time.time()
print("The number of grain is {}".format(label_num_pred))
r_index, adjust_r_index, v_index, merger_error, split_error = utility.validate_label_stack_by_rvi(label_stack_pred, label_stack_gt)
print("The ri is {:.8f}, ari is {:.8f}, vi is {:.8f}, merger_error is {:.8f}, split_error is {:.8f}"
      .format(r_index, adjust_r_index, v_index, merger_error, split_error))
print("The duriation of vgg13_bn is {:.2f}'s".format(end_time - start_time))
np.save(os.path.join(data_address, "real_gt_vgg13_bn_label_stack.npy"), label_stack_pred)

# method = 3 cnn densenet161
print("Analyzing by densenet161")
start_time = time.time()
grain_track.set_cnn_tracker(model=1, pretrain_address=os.path.join(parameter_address, "real_densenet161.pkl"), device=cnn_device, need_augment=False, max_num_tensor=30)
label_stack_pred, label_num_pred = grain_track.get_tracked_label_stack(method=3)
end_time = time.time()
print("The number of grain is {}".format(label_num_pred))
r_index, adjust_r_index, v_index, merger_error, split_error = utility.validate_label_stack_by_rvi(label_stack_pred, label_stack_gt)
print("The ri is {:.8f}, ari is {:.8f}, vi is {:.8f}, merger_error is {:.8f}, split_error is {:.8f}"
      .format(r_index, adjust_r_index, v_index, merger_error, split_error))
print("The duriation of densenet161 is {:.2f}'s".format(end_time - start_time))
np.save(os.path.join(data_address, "real_gt_densenet161_label_stack.npy"), label_stack_pred)


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
r_index, adjust_r_index, v_index, merger_error, split_error = utility.validate_label_stack_by_rvi(label_stack_pred, label_stack_gt)
print("The ri is {:.8f}, ari is {:.8f}, vi is {:.8f}, merger_error is {:.8f}, split_error is {:.8f}"
      .format(r_index, adjust_r_index, v_index, merger_error, split_error))
print("The duriation of min centroid dis is {:.2f}'s".format(end_time - start_time))
np.save(os.path.join(data_address, "simulated_gt_min_centroid_dis_label_stack.npy", label_stack_pred))

# method = 2 max overlap area
print("Analyzing by max overlap area")
start_time = time.time()
label_stack_pred, label_num_pred = grain_track.get_tracked_label_stack(method=2)
end_time = time.time()
print("The number of grain is {}".format(label_num_pred))
r_index, adjust_r_index, v_index, merger_error, split_error = utility.validate_label_stack_by_rvi(label_stack_pred, label_stack_gt)
print("The ri is {:.8f}, ari is {:.8f}, vi is {:.8f}, merger_error is {:.8f}, split_error is {:.8f}"
      .format(r_index, adjust_r_index, v_index, merger_error, split_error))
print("The duriation of max overlap area is {:.2f}'s".format(end_time - start_time))
np.save(os.path.join(data_address, "simulated_gt_max_overlap_area_label_stack.npy", label_stack_pred))

# method = 3 cnn vgg13_bn
print("Analyzing by vgg13_bn")
start_time = time.time()
grain_track.set_cnn_tracker(model=0, pretrain_address=os.path.join(parameter_address, "simulated_vgg13_bn.pkl"), device=cnn_device, need_augment=False, max_num_tensor=30)
label_stack_pred, label_num_pred = grain_track.get_tracked_label_stack(method=3)
end_time = time.time()
print("The number of grain is {}".format(label_num_pred))
r_index, adjust_r_index, v_index, merger_error, split_error = utility.validate_label_stack_by_rvi(label_stack_pred, label_stack_gt)
print("The ri is {:.8f}, ari is {:.8f}, vi is {:.8f}, merger_error is {:.8f}, split_error is {:.8f}"
      .format(r_index, adjust_r_index, v_index, merger_error, split_error))
print("The duriation of vgg13_bn is {:.2f}'s".format(end_time - start_time))
np.save(os.path.join(data_address, "simulated_gt_vgg13_bn_label_stack.npy", label_stack_pred))

# method = 3 cnn densenet161
print("Analyzing by densenet161")
start_time = time.time()
grain_track.set_cnn_tracker(model=1, pretrain_address=os.path.join(parameter_address, "simulated_densenet161.pkl"), device=cnn_device, need_augment=False, max_num_tensor=30)
label_stack_pred, label_num_pred = grain_track.get_tracked_label_stack(method=3)
end_time = time.time()
print("The number of grain is {}".format(label_num_pred))
r_index, adjust_r_index, v_index, merger_error, split_error = utility.validate_label_stack_by_rvi(label_stack_pred, label_stack_gt)
print("The ri is {:.8f}, ari is {:.8f}, vi is {:.8f}, merger_error is {:.8f}, split_error is {:.8f}"
      .format(r_index, adjust_r_index, v_index, merger_error, split_error))
print("The duriation of densenet161 is {:.2f}'s".format(end_time - start_time))
np.save(os.path.join(data_address, "simulated_gt_densenet161_label_stack.npy", label_stack_pred))