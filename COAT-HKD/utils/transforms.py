# This file is part of COAT, and is distributed under the
# OSI-approved BSD 3-Clause License. See top-level LICENSE file or
# https://github.com/Kitware/COAT/blob/master/LICENSE for details.

import random
import math
import torch
import numpy as np
from copy import deepcopy
from torchvision.transforms import functional as F
import cv2
import copy
from PIL import Image
from random import choice


class DataAugment:
    def __init__(self, debug=False, target=None):
        self.debug = debug
        self.tgt = target
        # print("Data augment...")

    def show(self, img, lines=None, color=(0, 255, 255), rec=None):
        if lines is not None:
            cv2.polylines(img, lines, 1, color=(0, 255, 0), thickness=2)
            # 使用过四条边中心的新矩形作为变换后拟合框，比完全包含四边形的矩形框要好，前者可以保证面积和变换后的框相同，避免拟合框过大
            ctrx = int((lines[0][0][0] + lines[0][2][0]) / 2)
            ctry = int((lines[0][0][1] + lines[0][2][1]) / 2)
            wid = abs(int(lines[0][0][0] + lines[0][1][0]) / 2 - int((lines[0][2][0] + lines[0][3][0]) / 2))
            hei = abs(int((lines[0][1][1] + lines[0][2][1]) / 2) - int((lines[0][0][1] + lines[0][3][1]) / 2))
            xmin, xmax = int(ctrx - wid / 2), int(ctrx + wid / 2)
            ymin, ymax = int(ctry - hei / 2), int(ctry + hei / 2)

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=2)
        if rec is not None:
            cv2.rectangle(img, (int(rec[0]), int(rec[1])), (int(rec[2]), int(rec[3])), color, thickness=2)
        cv2.imshow("outimg", img)
        cv2.waitKey()

    # 基础变换矩阵
    def basic_matrix(self, translation):
        return np.array(
            [[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])  # translation = center = [width/2, height/2]

    # 基础变换矩阵
    def adjust_transform_for_image(self, img, trans_matrix):
        transform_matrix = copy.deepcopy(trans_matrix)  # deep copy
        # print("trans_matrix is \n", trans_matrix)
        # 下面的操作看似高大上，其实只有平移操作的仿射矩阵第三列非零。那就说明了最初的平移矩阵第三列给的只是一个相对的平移比例，而非绝对的像素长度
        height, width, channels = img.shape
        transform_matrix[0:2, 2] *= [width, height]
        # print("transform_matrix is \n", transform_matrix)
        # 下面微调旋转相关的仿射矩阵
        center = np.array((0.5 * width, 0.5 * height))
        transform_matrix = np.linalg.multi_dot(
            [self.basic_matrix(center), transform_matrix, self.basic_matrix(-center)])
        # print("transform_matrix np.linalg.multi_dot is \n", transform_matrix, '\n')
        # ################# trans_matrix ################
        # 平移: 沿x轴平移aw个像素，沿y轴平移bh个像素
        #                      [1, 0, aw]
        #   transform_matrix = [0, 1, bh]
        #                      [0, 0,  1]
        #
        # 垂直反转：
        #                      [1,  0, 0]
        #   flip_matrix   =    [0, -1, h]
        #                      [0,  0, 1]
        #
        # 水平反转：
        #                      [-1, 0, w]
        #   flip_matrix   =    [ 0, 1, 0]
        #                      [ 0, 0, 1]
        #
        # 旋转a度(顺)：以(w/2, h/2)为中心，顺时针旋转a度
        #                      [cos(a), -sin(a), w/2-(w/2)cos(a)+(h/2)sin(a)]
        #   rotate_matrix  =   [sin(a),  cos(a), h/2-(h/2)cos(a)-(w/2)sin(a)]
        #                      [   0  ,     0  ,             1              ]
        #
        # 图片缩放：沿x轴放大a个像素，沿y轴放大b个像素
        #                      [ a, 0, w/2-(w/2)*a]
        #   scale_matrix  =    [ 0, b, h/2-(h/2)*b]
        #                      [ 0, 0,       1    ]
        #
        # 图片错切：
        #                      [ 1, a, -(h/2)*a]
        #   crop_matrix   =    [ b, 1, -(w/2)*b]
        #                      [ 0, 0,     1   ]
        return transform_matrix

    # 仿射变换
    def apply_transform(self, img, transform):
        # 传入的Image的图片是RGB的，cv2只能处理BGR
        img = np.array(img)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        output = cv2.warpAffine(img, transform[:2, :], dsize=(img.shape[1], img.shape[0]),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT, borderValue=0, )
        output = np.uint8(output)
        output = Image.fromarray(cv2.cvtColor(output,cv2.COLOR_BGR2RGB))
        # ################################ borderMode ###########################################
        # [1] cv2.BORDER_CONSTANT  填充边界时使用常数填充
        # [2] cv2.BORDER_REPLICATE： 使用边界最接近的像素填充，也就是用边缘像素填充
        # [3] cv2.BORDER_REFLECT， 反射法，是指对图像的像素两边进行复制
        # [4] cv2.BORDER_REFLECT101： 反射法，把边缘的像素作为轴，对称的复制
        # [5] cv2.BORDER_WRAP： 用另一边的像素进行填充
        # [6] cv2.BORDER_TRANSPARENT

        return output

    # 应用变换
    def apply(self, img, trans_matrix):
        tmp_matrix = self.adjust_transform_for_image(img, trans_matrix)
        out_img = self.apply_transform(img, tmp_matrix)
        newtgt = None
        out_box = []
        if self.tgt is not None:
            # lines是多边形的新平行四边形框，比较拟合拉伸后的真实形状，而newtgt是矩形框，有误差，应用中只能用矩形框
            for box in self.tgt:
                lines=self.gen_lines(tmp_matrix, box)
                ctrx = int((lines[0][0][0] + lines[0][2][0]) / 2)
                ctry = int((lines[0][0][1] + lines[0][2][1]) / 2)
                wid = abs(int(lines[0][0][0] + lines[0][1][0]) / 2 - int((lines[0][2][0] + lines[0][3][0]) / 2))
                hei = abs(int((lines[0][1][1] + lines[0][2][1]) / 2) - int((lines[0][0][1] + lines[0][3][1]) / 2))
                xmin, xmax = int(ctrx - wid / 2), int(ctrx + wid / 2)
                ymin, ymax = int(ctry - hei / 2), int(ctry + hei / 2)
                out_box.append([xmin,ymin,xmax,ymax])

        if self.debug:
            self.show(img=out_img, lines=out_box, rec=newtgt)

        return out_img, out_box

    # 生成范围矩阵
    def random_vector(self, min, max):
        min = np.array(min)
        max = np.array(max)
        # print(min.shape, max.shape)
        assert min.shape == max.shape
        assert len(min.shape) == 1
        return np.random.uniform(min, max)

    # 随机旋转(顺)
    def random_rotate(self, img, factor):
        # 除了以下方法，还可以考虑cv2.getRotationMatrix2D这个函数
        angle = np.random.randint(factor[0], factor[1])
        # print("angle : {}°".format(angle))
        angle = np.pi / 180.0 * angle
        rotate_matrix = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])  # 顺时针
        out_img, out_box = self.apply(img, rotate_matrix)
        return out_img, out_box

    # 随机缩放
    def random_scale(self, img, min_translation, max_translation):
        factor = self.random_vector(min_translation, max_translation)
        scale_matrix = np.array([[factor[0], 0, 0], [0, factor[1], 0], [0, 0, 1]])
        out_img, out_box = self.apply(img, scale_matrix)
        return out_img, out_box

    # 随机剪切，包括横向和众向剪切
    def random_shear(self, img, factor):
        angle = np.random.uniform(factor[0], factor[1])
        # print("fc:{}".format(angle))
        crop_matrix = np.array([[1, factor[0], 0], [factor[1], 1, 0], [0, 0, 1]])
        out_img, out_box = self.apply(img, crop_matrix)
        return out_img, out_box

    # 对原标注框的四个角都仿射变换一下
    def gen_lines(self, mat, box):
        leftup = [box[0], box[1], 1]
        leftdown = [box[0], box[3], 1]
        rightdown = [box[2], box[3], 1]
        rightup = [box[2], box[1], 1]

        lux = np.dot(mat[0, :], leftup)
        luy = np.dot(mat[1, :], leftup)
        ldx = np.dot(mat[0, :], leftdown)
        ldy = np.dot(mat[1, :], leftdown)
        rdx = np.dot(mat[0, :], rightdown)
        rdy = np.dot(mat[1, :], rightdown)
        rux = np.dot(mat[0, :], rightup)
        ruy = np.dot(mat[1, :], rightup)
        cord = [[[lux, luy], [ldx, ldy], [rdx, rdy], [rux, ruy]]]
        lines = np.array(cord, dtype=np.int32)
        return lines


def mixup_data(images, alpha=0.8):
    # images [bs,chw]
    if 0. < alpha < 1.:
        lam = random.uniform(alpha, 1)
    else:
        lam = 1.

    batch_size = len(images)
    min_x = 9999
    min_y = 9999
    for i in range(batch_size):
        min_x = min(min_x, images[i].shape[1])
        min_y = min(min_y, images[i].shape[2])

    shuffle_images = deepcopy(images)
    random.shuffle(shuffle_images)
    mixed_images = deepcopy(images)
    for i in range(batch_size):
        mixed_images[i][:, :min_x, :min_y] = lam * images[i][:, :min_x, :min_y] + (1 - lam) * shuffle_images[i][:,
                                                                                              :min_x, :min_y]

    return mixed_images


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

# I added it
class RandomRotate:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = np.array(image)
            bbox = target["boxes"]
            daPic = DataAugment(target=bbox)
            image, boxes = daPic.random_rotate(image, (-12, 12))  # 在±15°之间顺时针旋转
            target["boxes"] = torch.tensor(boxes)
        return image, target


class RandomScale:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = np.array(image)
            bbox = target["boxes"]
            daPic = DataAugment(target=bbox)
            image, boxes = daPic.random_scale(image, (1.2, 1.2), (1.3,1.3))
            target["boxes"] = torch.tensor(boxes)
        return image, target


class RandomShear:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = np.array(image)
            bbox = target["boxes"]
            daPic = DataAugment(target=bbox)
            image, boxes = daPic.random_shear(image, (0.1, 0.2))
            target["boxes"] = torch.tensor(boxes)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes=2, length=100):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img, target):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img, target


class RandomErasing(object):
    '''
    https://github.com/zhunzhong07/CamStyle/blob/master/reid/utils/data/transforms.py
    '''

    def __init__(self, EPSILON=0.5, mean=[0.485, 0.456, 0.406]):
        self.EPSILON = EPSILON
        self.mean = mean

    def __call__(self, img, target):
        if random.uniform(0, 1) > self.EPSILON:
            return img, target

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            # 下面四行实现的就是把w*h的图片，先缩小面积到w2*h2，再根据两种高宽比，将其变形成w1*h1的图片，变形后面积等于小图片面积
            target_area = random.uniform(0.02, 0.2) * area
            aspect_ratio = random.uniform(0.3, 3)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size()[2] and h <= img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]

                return img, target

        return img, target


class ToTensor:
    def __call__(self, image, target):
        # convert [0, 255] to [0, 1]
        image = F.to_tensor(image)
        return image, target


def build_transforms(cfg, is_train):
    transforms = []
    if is_train:
        # 这些基于原图的操作适合用于Image刚读取出来的np数组，而非归一化后的tensor，为了免去解析tensor再封装为tensor的麻烦，这里就在totensor之前就把图像级的处理干完
        if cfg.INPUT.IMAGE_ROTATE:
            transforms.append(RandomRotate())
        if cfg.INPUT.IMAGE_SCALE:
            transforms.append(RandomScale())
        if cfg.INPUT.IMAGE_SHEAR:
            transforms.append(RandomShear())

    transforms.append(ToTensor())
    if is_train:
        transforms.append(RandomHorizontalFlip())
        if cfg.INPUT.IMAGE_CUTOUT:
            transforms.append(Cutout())
        if cfg.INPUT.IMAGE_ERASE:
            transforms.append(RandomErasing())

    return Compose(transforms)
