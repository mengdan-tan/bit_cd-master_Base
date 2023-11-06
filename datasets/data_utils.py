# import random
# import numpy as np
#
# from PIL import Image
# from PIL import ImageFilter
#
# import torchvision.transforms.functional as TF
# from torchvision import transforms
# import torch
#
#
# def to_tensor_and_norm(imgs, labels):
#     # to tensor
#     imgs = [TF.to_tensor(img) for img in imgs]
#     labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
#               for img in labels]
#
#     imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#             for img in imgs]
#     return imgs, labels
#
#
# class CDDataAugmentation:
#
#     def __init__(
#             self,
#             img_size,
#             with_random_hflip=False,    # 随机水平翻转
#             with_random_vflip=False,    # 随机垂直翻转
#             with_random_rot=False,      # 随机旋转
#             with_random_crop=False,     # 随机裁剪
#             with_scale_random_crop=False,   # 缩放后再随机裁剪
#             with_random_blur=False,         # 随机模糊
#     ):
#         self.img_size = img_size
#         if self.img_size is None:
#             self.img_size_dynamic = True
#         else:
#             self.img_size_dynamic = False
#         self.with_random_hflip = with_random_hflip
#         self.with_random_vflip = with_random_vflip
#         self.with_random_rot = with_random_rot
#         self.with_random_crop = with_random_crop
#         self.with_scale_random_crop = with_scale_random_crop
#         self.with_random_blur = with_random_blur
#     def transform(self, imgs, labels, to_tensor=True):
#         """
#         : imgs:list(2),imgA imgB
#         :param imgs: [ndarray,]
#         :param labels: [ndarray,]
#         :return: [ndarray,],[ndarray,]
#         """
#         # resize image and covert to tensor
#         imgs = [TF.to_pil_image(img) for img in imgs]
#         if self.img_size is None:
#             self.img_size = None
#
#         if not self.img_size_dynamic:
#             if imgs[0].size != (self.img_size, self.img_size):
#                 imgs = [TF.resize(img, [self.img_size, self.img_size], interpolation=3)
#                         for img in imgs]
#         else:
#             self.img_size = imgs[0].size[0]
#
#         labels = [TF.to_pil_image(img) for img in labels]
#         if len(labels) != 0:
#             if labels[0].size != (self.img_size, self.img_size):
#                 labels = [TF.resize(img, [self.img_size, self.img_size], interpolation=0)
#                         for img in labels]
#
#         random_base = 0.5
#         if self.with_random_hflip and random.random() > 0.5:
#             imgs = [TF.hflip(img) for img in imgs]
#             labels = [TF.hflip(img) for img in labels]
#
#         if self.with_random_vflip and random.random() > 0.5:
#             imgs = [TF.vflip(img) for img in imgs]
#             labels = [TF.vflip(img) for img in labels]
#
#         if self.with_random_rot and random.random() > random_base:
#             angles = [90, 180, 270]
#             index = random.randint(0, 2)
#             angle = angles[index]
#             imgs = [TF.rotate(img, angle) for img in imgs]
#             labels = [TF.rotate(img, angle) for img in labels]
#
#         if self.with_random_crop and random.random() > 0:
#             i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
#                 get_params(img=imgs[0], scale=(0.8, 1.0), ratio=(1, 1))
#
#             imgs = [TF.resized_crop(img, i, j, h, w,
#                                     size=(self.img_size, self.img_size),
#                                     interpolation=Image.CUBIC)
#                     for img in imgs]
#
#             labels = [TF.resized_crop(img, i, j, h, w,
#                                       size=(self.img_size, self.img_size),
#                                       interpolation=Image.NEAREST)
#                       for img in labels]
#
#         if self.with_scale_random_crop:
#             # rescale
#             scale_range = [1, 1.2]
#             target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
#
#             imgs = [pil_rescale(img, target_scale, order=3) for img in imgs]
#             labels = [pil_rescale(img, target_scale, order=0) for img in labels]
#             # crop
#             imgsize = imgs[0].size  # h, w
#             box = get_random_crop_box(imgsize=imgsize, cropsize=self.img_size)
#             imgs = [pil_crop(img, box, cropsize=self.img_size, default_value=0)
#                     for img in imgs]
#             labels = [pil_crop(img, box, cropsize=self.img_size, default_value=255)
#                     for img in labels]
#
#         if self.with_random_blur and random.random() > 0:
#             radius = random.random()
#             imgs = [img.filter(ImageFilter.GaussianBlur(radius=radius))
#                     for img in imgs]
#
#         if to_tensor:
#             # to tensor
#             imgs = [TF.to_tensor(img) for img in imgs]
#             labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
#                       for img in labels]
#
#             imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
#                     for img in imgs]
#
#         return imgs, labels
#
#
# def pil_crop(image, box, cropsize, default_value):
#     assert isinstance(image, Image.Image)
#     img = np.array(image)
#
#     if len(img.shape) == 3:
#         cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value
#     else:
#         cont = np.ones((cropsize, cropsize), img.dtype)*default_value
#     cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
#
#     return Image.fromarray(cont)
#
#
# def get_random_crop_box(imgsize, cropsize):
#     h, w = imgsize
#     ch = min(cropsize, h)
#     cw = min(cropsize, w)
#
#     w_space = w - cropsize
#     h_space = h - cropsize
#
#     if w_space > 0:
#         cont_left = 0
#         img_left = random.randrange(w_space + 1)
#     else:
#         cont_left = random.randrange(-w_space + 1)
#         img_left = 0
#
#     if h_space > 0:
#         cont_top = 0
#         img_top = random.randrange(h_space + 1)
#     else:
#         cont_top = random.randrange(-h_space + 1)
#         img_top = 0
#
#     return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw
#
#
# def pil_rescale(img, scale, order):
#     assert isinstance(img, Image.Image)
#     height, width = img.size
#     target_size = (int(np.round(height*scale)), int(np.round(width*scale)))
#     return pil_resize(img, target_size, order)
#
#
# def pil_resize(img, size, order):
#     assert isinstance(img, Image.Image)
#     if size[0] == img.size[0] and size[1] == img.size[1]:
#         return img
#     if order == 3:
#         resample = Image.BICUBIC
#     elif order == 0:
#         resample = Image.NEAREST
#     return img.resize(size[::-1], resample)



############################################################
# changeFormer中的
############################################################

import random
import numpy as np

from PIL import Image
from PIL import ImageFilter

import torchvision.transforms.functional as TF
from torchvision import transforms
import torch


def to_tensor_and_norm(imgs, labels):
    # to tensor
    imgs = [TF.to_tensor(img) for img in imgs]
    labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
              for img in labels]

    imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            for img in imgs]
    return imgs, labels



##############################################################
import cv2

def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):  # yolov5的超参数
    img = np.array(img)
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    # hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))  # PIL的图像是RGB格式
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    # cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)  # no return needed # PIL的图像是RGB格式
    img = Image.fromarray(img)
    return img


# 直方图均衡化，clahe为True，则使用自适应直方图均衡化算法
def hist_equalize(img, clahe=True, bgr=False):  # 直方图均衡化
    # Equalize histogram on BGR image 'img' with img.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB

# random_num = random.random()
# if random_num < 0.2:  # 直方图均衡化
#     img1 = np.array(img1)
#     img2 = np.array(img2)
#     img1 = self.hist_equalize(img1)
#     img2 = self.hist_equalize(img2)
#     img1 = Image.fromarray(img1)
#     img2 = Image.fromarray(img2)
# 直方图匹配   这里没有考虑PIL是RGB，opencv是BGR的问题，因为这个hist_refs1的三通道基本是一样的


######################################################################################




class CDDataAugmentation:

    def __init__(
            self,
            img_size,
            with_random_hflip=False,
            with_random_vflip=False,
            with_random_rot=False,
            with_random_crop=False,
            with_scale_random_crop=False,
            with_random_blur=False,

            with_random_color_tf=False,
            with_exchange_order=False,
            with_augment_hsv=False,
            with_hist_equalize=False
    ):
        self.img_size = img_size
        if self.img_size is None:
            self.img_size_dynamic = True
        else:
            self.img_size_dynamic = False
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rot = with_random_rot
        self.with_random_crop = with_random_crop
        self.with_scale_random_crop = with_scale_random_crop
        self.with_random_blur = with_random_blur

### 注意图像增强各操作的顺序，以及to tensor的顺序等
        self.with_random_color_tf = with_random_color_tf
        self.with_exchange_order = with_exchange_order
        self.with_augment_hsv=with_augment_hsv
        self.with_hist_equalize=with_hist_equalize

    def transform(self, imgs, labels, to_tensor=True):
        """
        :param imgs: [ndarray,]
        :param labels: [ndarray,]
        :return: [ndarray,],[ndarray,]
        """
        # resize image and covert to tensor
        imgs = [TF.to_pil_image(img) for img in imgs]
        if self.img_size is None:
            self.img_size = None

        if not self.img_size_dynamic:
            if imgs[0].size != (self.img_size, self.img_size):
                imgs = [TF.resize(img, [self.img_size, self.img_size], interpolation=3)
                        for img in imgs]
        else:
            self.img_size = imgs[0].size[0]

        labels = [TF.to_pil_image(img) for img in labels]
        if len(labels) != 0:
            if labels[0].size != (self.img_size, self.img_size):
                labels = [TF.resize(img, [self.img_size, self.img_size], interpolation=0)
                          for img in labels]

        random_base = 0.5
        if self.with_random_hflip and random.random() > 0.5:
            imgs = [TF.hflip(img) for img in imgs]
            labels = [TF.hflip(img) for img in labels]

        if self.with_random_vflip and random.random() > 0.5:
            imgs = [TF.vflip(img) for img in imgs]
            labels = [TF.vflip(img) for img in labels]

        if self.with_random_rot and random.random() > random_base:
            angles = [90, 180, 270]
            index = random.randint(0, 2)
            angle = angles[index]
            imgs = [TF.rotate(img, angle) for img in imgs]
            labels = [TF.rotate(img, angle) for img in labels]

        if self.with_random_crop and random.random() > 0:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=imgs[0], scale=(0.8, 1.2), ratio=(1, 1))

            imgs = [TF.resized_crop(img, i, j, h, w,
                                    size=(self.img_size, self.img_size),
                                    interpolation=Image.CUBIC)
                    for img in imgs]

            labels = [TF.resized_crop(img, i, j, h, w,
                                      size=(self.img_size, self.img_size),
                                      interpolation=Image.NEAREST)
                      for img in labels]

        if self.with_scale_random_crop:
            # rescale
            scale_range = [1, 1.2]
            target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

            imgs = [pil_rescale(img, target_scale, order=3) for img in imgs]
            labels = [pil_rescale(img, target_scale, order=0) for img in labels]
            # crop
            imgsize = imgs[0].size  # h, w
            box = get_random_crop_box(imgsize=imgsize, cropsize=self.img_size)
            imgs = [pil_crop(img, box, cropsize=self.img_size, default_value=0)
                    for img in imgs]
            labels = [pil_crop(img, box, cropsize=self.img_size, default_value=255)
                      for img in labels]

##################################################################
        # 应该在裁剪后进行操作
        # 直方图均衡化
        random_num = random.random()
        if self.with_hist_equalize and random_num > 0.5:
            imgs = [hist_equalize(img) for img in imgs]

        # hsv色彩增强
        random_num = random.random()
        if self.with_augment_hsv and random_num > 0.5:
            imgs = [augment_hsv(img) for img in imgs]
##############################################################



        # 随机模糊，label不用变
        if self.with_random_blur and random.random() > 0:
            radius = random.random()
            imgs = [img.filter(ImageFilter.GaussianBlur(radius=radius))
                    for img in imgs]

        # label不用变
        if self.with_random_color_tf and random.random() > 0.5:
            color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
            imgs_tf = []
            for img in imgs:
                tf = transforms.ColorJitter(
                    color_jitter.brightness,    # 亮度
                    color_jitter.contrast,      # 对比度
                    color_jitter.saturation,    # 饱和度
                    color_jitter.hue)           # 色调
                imgs_tf.append(tf(img))
            imgs = imgs_tf


    # 一定的概率交换 A,B 图片顺序
        if self.with_exchange_order and random.random() >0.5:
            temp=imgs[0]
            imgs[0]=imgs[1]
            imgs[1]=temp

        if to_tensor:
            # to tensor
            imgs = [TF.to_tensor(img) for img in imgs]
            labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
                      for img in labels]

            # 归一化到[-1,1]的范围
            imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    for img in imgs]

        return imgs, labels






def pil_crop(image, box, cropsize, default_value):
    assert isinstance(image, Image.Image)
    img = np.array(image)

    if len(img.shape) == 3:
        cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype) * default_value
    else:
        cont = np.ones((cropsize, cropsize), img.dtype) * default_value
    cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]

    return Image.fromarray(cont)


def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize
    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top + ch, cont_left, cont_left + cw, img_top, img_top + ch, img_left, img_left + cw


def pil_rescale(img, scale, order):
    assert isinstance(img, Image.Image)
    height, width = img.size
    target_size = (int(np.round(height * scale)), int(np.round(width * scale)))
    return pil_resize(img, target_size, order)


def pil_resize(img, size, order):
    assert isinstance(img, Image.Image)
    if size[0] == img.size[0] and size[1] == img.size[1]:
        return img
    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST
    return img.resize(size[::-1], resample)




