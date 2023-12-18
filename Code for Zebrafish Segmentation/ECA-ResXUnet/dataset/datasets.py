#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
from tools import cv_imread
from torch.utils.data import Dataset as BaseDataset


class CropToEven(albu.Crop):
    """Crop region from image.
    make the height and width are even numbers
    """
    def __init__(self, always_apply=True, p=1.0):
        super(CropToEven, self).__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        self.y_max, self.x_max = img.shape[:2]
        self.x_max = self.x_max if self.x_max % 2 == 0 else self.x_max - 1
        self.y_max = self.y_max if self.y_max % 2 == 0 else self.y_max - 1
        return super(CropToEven, self).apply(img, x_min=0, y_min=0, x_max=self.x_max, y_max=self.y_max)


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def make_images_masks_list_with_random_shuffle(images_dir, masks_dir):
    ids = os.listdir(masks_dir)
    # random.shuffle(ids)
    images_fps = [os.path.join(images_dir, image_id) for image_id in ids]
    for id, img in enumerate(images_fps):
        if not os.path.exists(img):
            portion = os.path.splitext(img)
            images_fps[id] = portion[0] + '.bmp'
            if not os.path.exists(images_fps[id]):
                print("can not find", img)
            else:
                print('rename as:', images_fps[id])
        else:
            pass

    masks_fps = [os.path.join(masks_dir, image_id) for image_id in ids]
    return images_fps, masks_fps


class ZebrafishDataset(BaseDataset):
    """Zibrafish Dataset. Read images, apply augmentation and preprocessing transformations."""
    def __init__(
            self,
            images_list,
            masks_list,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        # init make ids based on the masks
        self.images_fps = images_list
        self.masks_fps = masks_list

        # convert str names to class values on masks
        if classes:
            self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        else:
            self.class_values = None
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        image = cv_imread(self.images_fps[i])
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            print('can not open image file: ', self.images_fps[i])
            if self.masks_fps is None:
                return None
            else:
                return None, None

        if self.masks_fps is not None:
            mask = cv2.imread(self.masks_fps[i], 0)
            # extract certain classes from mask
            if self.class_values:
                masks = [(mask == v) for v in self.class_values]
            else: # just use threshold if don't indicate the classes to extract
                masks = [mask > 128]
            mask[mask > 0] = 1
            mask = np.stack(masks, axis=-1).astype('float')

            # apply augmentations
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']

            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']

            return image, mask
        else: # in case we just inference the images and don't have ground truth of the masks
            if self.augmentation:
                sample = self.augmentation(image=image)
                image = sample['image']

            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image)
                image = sample['image']

            return image

    def __len__(self):
        return len(self.images_fps)

# Augmentations
def get_training_augmentation(size):
    train_transform = [
        albu.Resize(size[0],size[1]),
        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.07, rotate_limit=10, shift_limit=0.1, interpolation=cv2.INTER_CUBIC, p=1, border_mode=0),
        albu.PadIfNeeded(min_height=size[0], min_width=size[1], always_apply=True, border_mode=0),
        albu.RandomCrop(height=size[0], width=size[1], always_apply=True),
        albu.HorizontalFlip(p=0.5),
        albu.IAAAdditiveGaussianNoise(p=0.1),
        albu.IAAPerspective(scale=(0.02, 0.05), p=0.5)]
    return albu.Compose(train_transform)


def get_validation_augmentation(size):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(size[0],size[1])

    ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)



