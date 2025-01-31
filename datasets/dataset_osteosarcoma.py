import os
import random
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom
from torch.utils.data import Dataset
import imgaug as ia
import imgaug.augmenters as iaa  # 导入iaa


def mask_to_onehot(mask):
    """
    Converts a segmentation mask (H, W) to (H, W, K) where the last dim is a one
    hot encoding vector, K is the number of classes.
    """
    semantic_map = []
    mask = np.expand_dims(mask, -1)
    for colour in range(9):  # 假设9类
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.int32)
    return semantic_map


def augment_seg(img_aug, img, seg):
    """
    Applies augmentation to both image and segmentation map.
    """
    seg = mask_to_onehot(seg)
    aug_det = img_aug.to_deterministic()
    image_aug = aug_det.augment_image(img)

    segmap = ia.SegmentationMapOnImage(seg, nb_classes=np.max(seg) + 1, shape=img.shape)
    segmap_aug = aug_det.augment_segmentation_maps(segmap)
    segmap_aug = segmap_aug.get_arr_int()
    segmap_aug = np.argmax(segmap_aug, axis=-1).astype(np.float32)
    return image_aug, segmap_aug


def random_rot_flip(image, label):
    """
    Randomly rotate and flip the image and label.
    """
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    """
    Randomly rotate the image and label by a small angle.
    """
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=3, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class OsteosarcomaDataset(Dataset):
    def __init__(self, base_dir, list_dir, split, img_size, norm_x_transform=None, norm_y_transform=None):
        self.norm_x_transform = norm_x_transform
        self.norm_y_transform = norm_y_transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir
        self.img_size = img_size

        self.img_aug = iaa.SomeOf((0, 4), [
            iaa.Flipud(0.5, name="Flipud"),
            iaa.Fliplr(0.5, name="Fliplr"),
            iaa.AdditiveGaussianNoise(scale=0.005 * 255),
            iaa.GaussianBlur(sigma=(1.0)),
            iaa.LinearContrast((0.5, 1.5), per_channel=0.5),
            iaa.Affine(scale={"x": (0.5, 2), "y": (0.5, 2)}),
            iaa.Affine(rotate=(-40, 40)),
            iaa.Affine(shear=(-16, 16)),
            iaa.PiecewiseAffine(scale=(0.008, 0.03)),
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
        ], random_order=True)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # 获取图片和标签的路径
        image_path = os.path.join(self.data_dir, "images", self.sample_list[idx].strip() + ".png")
        label_path = os.path.join(self.data_dir, "labels", self.sample_list[idx].strip() + ".png")

        # 读取图片和标签
        image = np.array(Image.open(image_path).convert('L'))  # 灰度图
        label = np.array(Image.open(label_path))  # 标签

        if self.split == "train":
            # 数据增强
            image, label = augment_seg(self.img_aug, image, label)

        # 缩放到目标大小
        x, y = image.shape
        if x != self.img_size or y != self.img_size:
            image = zoom(image, (self.img_size / x, self.img_size / y), order=3)
            label = zoom(label, (self.img_size / x, self.img_size / y), order=0)

        # 转为 PyTorch 张量
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))

        # 返回样本
        sample = {'image': image, 'label': label.long(), 'case_name': self.sample_list[idx].strip()}
        return sample
