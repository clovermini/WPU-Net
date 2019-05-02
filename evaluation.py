import numpy as np
import gala.evaluate as ev
from skimage.measure import label
import time, os, cv2

def get_figure_of_merit(pred, mask, const_index=0.1):
    """
    Use figure of merit to evaluate the edge detection quality of proposed method
    :param pred: predictions [0,255], 0 - background, 255 - foreground
    :param mask: groud truth  [0,255], 0 - background, 255 - foreground
    :return: f_score
    """
    num_pred = np.count_nonzero(pred[pred == 255])
    num_mask = np.count_nonzero(mask[mask == 255])
    num_max = num_pred if num_pred > num_mask else num_mask

    temp = 0.0
    for index_x in range(0, pred.shape[0]):
        for index_y in range(0, pred.shape[1]):
            if pred[index_x, index_y] == 255:
                distance = get_dis_from_mask_point(mask, index_x, index_y)
                temp = temp + 1 / (1 + const_index * pow(distance, 2))
    f_score = (1.0 / num_max) * temp
    return f_score


def get_dis_from_mask_point(mask, index_x, index_y, neighbor_length=60):
    """
    计算检测到的边缘点与离它最近边缘点的距离
    Calculate the distance between the detected boundary point and the nearest groud truth boundary point
    """

    if mask[index_x, index_y] == 255:
        return 0
    region_start_row = 0
    region_start_col = 0
    region_end_row = mask.shape[0]
    region_end_col = mask.shape[1]
    if index_x - neighbor_length > 0:
        region_start_row = index_x - neighbor_length
    if index_x + neighbor_length < mask.shape[0]:
        region_end_row = index_x + neighbor_length
    if index_y - neighbor_length > 0:
        region_start_col = index_y - neighbor_length
    if index_y + neighbor_length < mask.shape[1]:
        region_end_col = index_y + neighbor_length
    # Get the corrdinate of mask in neighbor region
    # becuase the corrdinate will be chaneged after slice operation, we add it manually
    x, y = np.where(mask[region_start_row: region_end_row, region_start_col: region_end_col] == 255)

    if len(x) == 0:
        min_distance = 30
    else:
        min_distance = np.amin(
            np.linalg.norm(np.array([x + region_start_row, y + region_start_col]) - np.array([[index_x], [index_y]]),
                           axis=0))

    return min_distance


def get_map_2018kdasb_new(pred, mask, target_image=0):
    """
    Use map to evaluate the edge detection quality of proposed method, Only for binary images
    Our implementation is based on 2018 kaggle data science bowl:
    https://www.kaggle.com/c/data-science-bowl-2018/overview/evaluation
    :param pred: predictions [0,255], 0 - background, 255 - foreground
    :param mask: groud truth  [0,255], 0 - background, 255 - foreground
    :param target_image: 0 - pure iron images, 1 - 铝镧枝晶图像
    :return: map_score
    """
    thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    tp = np.zeros(10)
    if target_image == 0:
        # Eliminate the impact of poor performance on the border of image
        pred[0, :] = 255
        pred[:, 0] = 255
        pred[-1, :] = 255
        pred[:, -1] = 255
        mask[0, :] = 255
        mask[:, 0] = 255
        mask[-1, :] = 255
        mask[:, -1] = 255

    label_mask, num_mask = label(mask, neighbors=4, background=0, return_num=True)
    label_pred, num_pred = label(pred, neighbors=4, background=0, return_num=True)

    for i_pred in range(1, num_pred + 1):
        intersect_mask_labels = list(np.unique(label_mask[label_pred == i_pred]))    # Get all the labels that intersect with it
        if 0 in intersect_mask_labels:
            intersect_mask_labels.remove(0)

        if len(intersect_mask_labels) == 0:  # If a label of pred does not have a corresponding label intersect with it in mask, pass
            continue

        intersect_mask_label_area = np.zeros((len(intersect_mask_labels), 1))
        union_mask_label_area = np.zeros((len(intersect_mask_labels), 1))

        for index, i_mask in enumerate(intersect_mask_labels):
            intersect_mask_label_area[index, 0] = np.count_nonzero(label_pred[label_mask == i_mask] == i_pred)
            union_mask_label_area[index, 0] = np.count_nonzero((label_mask == i_mask) | (label_pred == i_pred))
        # Calculate iou for all labels that intersect with it, and then take the highest value as max_iou
        iou = intersect_mask_label_area / union_mask_label_area
        max_iou = np.max(iou, axis=0)
        # 根据最值将tp赋值  Calculate tp based on the thresholds and max_iou
        tp[thresholds < max_iou] = tp[thresholds < max_iou] + 1
    # 此处基于一个重要理论：对于一个预测的晶粒，真实的晶粒有且仅有一个晶粒与其iou>0.5
    # Calculate fp and fn based on a theory: for a predicted grain, the corresponding real grain has one and only one with iou > 0.5
    fp = num_pred - tp
    fn = num_mask - tp
    map_score = np.average(tp / (tp + fp + fn))
    return map_score


gt_dir = "./datasets/test/manual/"

def eval_RI_VI(results_dir, outAd):
    """
    :param results_dir: address for network outputs
    :param outAd: save directory for evaluation results
    :return:
    """

    results_path = sorted(os.listdir(results_dir))

    mRI = 0
    mad_RI = 0
    m_merger_error = 0
    m_split_error = 0
    mVI = 0
    count = 0
    out = open(outAd, "a")  # record the result
    out.write(
        "RI, mRI, adjust_RI, m_adjust_RI, merger_error, m_merger_error, split_error, m_split_error, VI, mVI" + '\n')
    out.close()
    for r in results_path:
        name = r.split(".")[0]
        gt_path = os.path.join(gt_dir, name + ".png")
        if os.path.exists(gt_path):
            count += 1

            gt = cv2.imread(gt_path, 0)

            result = cv2.imread(os.path.join(results_dir, r), 0)
            # Eliminate the impact of poor performance on the border of image
            result = result[10:-10, 10:-10]
            gt = gt[10:-10, 10:-10]

            result, num_result = label(result, background=255, neighbors=4, return_num=True)
            gt, num_gt = label(gt, background=255, neighbors=4, return_num=True)

            # false merges(缺失), false splits（划痕）
            merger_error, split_error = ev.split_vi(result, gt)
            VI = merger_error + split_error
            RI = ev.rand_index(result, gt)
            adjust_RI = ev.adj_rand_index(result, gt)

            m_merger_error += merger_error
            m_split_error += split_error
            mRI += RI
            mVI += VI
            mad_RI += adjust_RI

            out = open(outAd, "a")  # # record the result
            # "RI, mRI, adjust_RI, m_adjust_RI, merger_error, m_merger_error, split_error, m_split_error, VI, mVI"
            line = str(RI) + "," + str(mRI / count) + "," + str(adjust_RI) + "," + str(mad_RI / count) + "," + str(
                merger_error) + "," + str(m_merger_error / count) + "," + str(split_error) + "," + str(
                m_split_error / count) + "," + str(VI) + "," + str(mVI / count) + "\n"
            out.write(line)
            out.close()

    print("average RI : ", mRI / count, "average adRI : ", mad_RI / count, "average VI : ", mVI / count,
          "average merger_error : ", m_merger_error / count, "average split_error : ", m_split_error / count)


def eval_F_mapKaggle(results_dir, outAd):

    results_path = sorted(os.listdir(results_dir))
    F = 0
    mAP = 0
    count = 0

    out = open(outAd, "a")  # # record the result
    out.write("F,avF,mAP,avmAP" + '\n')
    out.close()
    for r in results_path:
        name = r.split(".")[0]
        gt_path = os.path.join(gt_dir, name + ".png")
        if os.path.exists(gt_path):
            count += 1

            gt = cv2.imread(gt_path, 0)
            result = cv2.imread(os.path.join(results_dir, r), 0)
            # Eliminate the impact of poor performance on the border of image
            result = result[10:-10, 10:-10]
            gt = gt[10:-10, 10:-10]

            F_test = get_figure_of_merit(result, gt)
            F += F_test

            mAP_test = get_map_2018kdasb_new(result, gt)
            mAP += mAP_test

            out = open(outAd, "a")  # # record the result
            line = str(F_test) + "," + str(F / count) + "," + str(mAP_test) + "," + str(mAP / count) + "\n"
            out.write(line)
            out.close()

    F = F / count
    mAP = mAP / count
    print("count ", count, " average F : ", F, "average mAP : ", mAP)


if __name__ == '__main__':
    RI_save_dir = "./evaluation/big_RI_VI/"
    Map_save_dir = "./evaluation/big_F_mAP/"

    print("WPU_Net" + "#####"*20)
    eval_RI_VI("./result_total/WPU_Net/", RI_save_dir+"WPU_Net.txt")
    eval_F_mapKaggle("./result_total/WPU_Net/", Map_save_dir+"WPU_Net.txt")
