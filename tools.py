import numpy as np
from PIL import Image
import random
import torch, cv2, os
import torchvision.transforms.functional as F
from skimage import morphology as sm
import torchvision.transforms as tr

transform = tr.Compose([
    tr.ToTensor(),
    tr.Normalize(mean = [ 0.94034111, 0.94034111, 0.94034111 ],    # RGB
                std = [ 0.12718913, 0.12718913, 0.12718913 ])
])

def dilate_mask(mask, iteration=1):
    for i in range(0, iteration):
        mask = sm.dilation(mask, sm.square(3))
    return mask

def getImg(img_path):
    """
    加载图片和npy文件，用于data.py
    Load images and npy files for data.py
    :param img_path:  Address for images or npy file
    :return: PIL image or numpy array
    """
    if img_path.endswith(".npy"):
        img = np.load(img_path)
    else:
        img = Image.open(img_path).convert('L')
    return img

def proprecess_img(image):
    image = Image.open(image)
    tensor = transform(image).resize_(1,1,image.size[1],image.size[0])
    return tensor

def posprecess(output, close=False):
    """
    Postprocessing network prediction results
    :param output: network prediction results
    :param close: close operation
    :return:
    """
    out = torch.sigmoid(output)
    result_npy = out.data.squeeze().cpu().numpy()

    result_npy_final = out.max(1)[1].data.squeeze().cpu().numpy()
    result_npy_final = np.array(result_npy_final).astype('uint8') * 255
    if close:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        result_npy_final = cv2.morphologyEx(result_npy_final, cv2.MORPH_CLOSE, kernel)
    result_npy_final = sm.skeletonize(result_npy_final / 255) * 255  # 骨架化
    return result_npy_final

# cropping
def tailor(sizeX, sizeY, oriAddress, saveAddress, region = 32):
    img_name = oriAddress.split("/")[-1].split(".")[0]
    print("file : ", img_name)
    image_last = Image.open(oriAddress)

    (w_image, h_image) = image_last.size

    row = int(h_image/sizeY)
    col = int(w_image/sizeX)

    # 裁剪成200*200的图片
    n = 0
    for i in range(row):
        for j in range(col):
            box_X = j*sizeX
            box_Y = i*sizeY
            box_W = sizeX
            box_H = sizeY
            if i == 0:
                box_Y = 0
                box_H = sizeY + region
            elif i == row-1:
                box_Y = i*sizeY - region
                box_H = sizeY + region
            else:
                box_Y = i*sizeY - region
                box_H = sizeY + 2*region
            if j == 0:
                box_X = 0
                box_W = sizeX + region
            elif j == col-1:
                box_X = j*sizeX - region
                box_W = sizeX + region
            else:
                box_X = j*sizeX - region
                box_W = sizeX + 2*region
            box = (box_X, box_Y, box_X + box_W, box_Y + box_H)
            roi_last = image_last.crop(box)
            n += 1
            destination_ori = os.path.join(saveAddress, img_name+"_"+str(i)+"_"+str(j)+'.png')
            roi_last.save(destination_ori)
    return "success"

# Get the maximum number of rows and columns for a cropped picture (saved with row-column names)
def getXAndY(name, pieceAddress):
    imgList = os.listdir(pieceAddress)
    tmpList = []
    tmpX = 0    # 行
    tmpY = 0    # 列
    for img in imgList:
        if img.endswith(".png") and img.startswith(name):
            tmpList.append(img.replace(name+'_', '').replace('.png', '').split('_'))
    for size in tmpList:
        if int(size[0]) > tmpX:
            tmpX = int(size[0])
        if int(size[1]) > tmpY:
            tmpY = int(size[1])
    return [tmpX, tmpY]

# stitching
def stitch(sizeX, sizeY, name, pieceAddress, saveRoad, region = 16):
    tmpX = getXAndY(name, pieceAddress)[0]
    tmpY = getXAndY(name, pieceAddress)[1]
    print(tmpX, " ", tmpY)
    resultWidth = sizeX*(tmpY+1)
    resultHeight = sizeY*(tmpX+1)
    result = Image.new("RGB", (resultWidth, resultHeight))
    for i in range(tmpX+1):
        for j in range(tmpY+1):
            fname = os.path.join(pieceAddress, name+"_"+str(i)+"_"+str(j)+'.png')
            piece = Image.open(fname)
            box_H = sizeY
            box_W = sizeX
            if i == 0:
                box_Y = 0
            else:
                box_Y = region
            if j == 0:
                box_X = 0
            else:
                box_X = region
            box = (box_X, box_Y, box_X + box_W, box_Y + box_H)
            roi_result = piece.crop(box)
            result.paste(roi_result, (j*sizeX, i*sizeY))
    result.save(saveRoad)
    return result

def rand_crop(data, label, height, width, last=None, weight=None):
    """
    Random crop for original image, corresponding label, and others
    :param data:  Original image
    :param label: Corresponding label
    :param height: Crop size
    :param width: Crop size
    :param last:  Last information for WPU-Net
    :param weight:  Weight map for WPU-Net
    :return:
    """
    # 随机选择裁剪区域   Randomly select the crop area
    random_h = random.randint(0, data.size[0] - height)
    random_w = random.randint(0, data.size[1] - width)

    box = (random_h, random_w, (random_h + height), (random_w + width))

    # Crop for PIL.Image object
    data = data.crop(box)
    label = label.crop(box)

    if last is not None:
        # Crop for numpy array
        last = last[random_w: random_w + width, random_h: random_h + height]
        weight = weight[random_w: random_w + width, random_h: random_h + height, :]
        return data, label, last, weight

    return data, label


def rand_rotation(data, label, last=None, weight=None):
    """
    Random rotation for original image, corresponding label, and others
    :param data:  Original image
    :param label: Corresponding label
    :param last: Last information for WPU-Net
    :param weight: Weight map for WPU-Net
    :return:
    """
    # 随机选择旋转角度  Randomly select the rotation angle
    angle = random.choice([0, 90, 180, 270])

    data = F.rotate(data, angle, expand=True)
    label = F.rotate(label, angle, expand=True)

    if last is not None:
        last = np.rot90(last, k=angle / 90)
        weight = np.rot90(weight, k=angle / 90)
        return data, label, last, weight

    return data, label


def rand_verticalFlip(data, label, last=None, weight=None):
    """
    Random vertical flip for original image, corresponding label, and others
    :param data:  Original image
    :param label: Corresponding label
    :param last: Last information for WPU-Net
    :param weight: Weight map for WPU-Net
    :return:
    """
    # 0.5的概率垂直翻转  Vertical flip with 0.5 probability
    if random.random() < 0.5:
        data = F.vflip(data)
        label = F.vflip(label)

        if last is not None:
            last = np.flip(last, 0)
            weight = np.flip(weight, 0)

    if last is not None:
        return data, label, last, weight

    return data, label

def rand_horizontalFlip(data, label, last=None, weight=None):
    """
    Random horizontal flip for original image, corresponding label, and others
    :param data:  Original image
    :param label: Corresponding label
    :param last: Last information for WPU-Net
    :param weight: Weight map for WPU-Net
    :return:
    """
    # 0.5的概率水平翻转  Horizontal flip with 0.5 probability
    if random.random() < 0.5:
        data = F.hflip(data)
        label = F.hflip(label)

        if last is not None:
            last = np.flip(last, 1)
            weight = np.flip(weight, 1)

    if last is not None:
        return data, label, last, weight

    return data, label