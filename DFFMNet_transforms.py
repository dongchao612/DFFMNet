import random
import torch
import matplotlib
import torchvision
import matplotlib.colors
import numpy as np

from skimage.transform import resize

from torchvision import transforms

image_w = 320
image_h = 240


class RandomHSV(object):
    def __init__(self, h_range, s_range, v_range):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, sample):
        img = sample['image']
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        return {'image': img_new, 'depth': sample['depth'], 'label': sample['label']}


class scaleNorm(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # Bi-linear
        image = resize(image, (image_h, image_w), order=1, mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = resize(depth, (image_h, image_w), order=0, mode='reflect', preserve_range=True)
        label = resize(label, (image_h, image_w), order=0, mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}


class RandomScale(object):
    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        target_scale = random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        target_height = int(round(target_scale * image.shape[0]))
        target_width = int(round(target_scale * image.shape[1]))
        # Bi-linear
        image = resize(image, (target_height, target_width),
                       order=1, mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = resize(depth, (target_height, target_width),
                       order=0, mode='reflect', preserve_range=True)
        label = resize(label, (target_height, target_width),
                       order=0, mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}


class RandomCrop(object):
    def __init__(self, th, tw):
        self.th = th
        self.tw = tw

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        h = image.shape[0]
        w = image.shape[1]
        i = random.randint(0, h - self.th)
        j = random.randint(0, w - self.tw)

        return {'image': image[i:i + image_h, j:j + image_w, :],
                'depth': depth[i:i + image_h, j:j + image_w],
                'label': label[i:i + image_h, j:j + image_w]}


class RandomFlip(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            depth = np.fliplr(depth).copy()
            label = np.fliplr(label).copy()

        return {'image': image, 'depth': depth, 'label': label}


# Transforms on torch.*Tensor
class Normalize(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = image / 255
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])(image)
        depth = transforms.Normalize(mean=[19050],
                                     std=[9650])(depth)
        sample['image'] = image
        sample['depth'] = depth

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # Generate different label scales
        # label2 = skimage.transform.resize(label, (label.shape[0] // 2, label.shape[1] // 2),
        #                                   order=0, mode='reflect', preserve_range=True)
        # label3 = skimage.transform.resize(label, (label.shape[0] // 4, label.shape[1] // 4),
        #                                   order=0, mode='reflect', preserve_range=True)
        # label4 = skimage.transform.resize(label, (label.shape[0] // 8, label.shape[1] // 8),
        #                                   order=0, mode='reflect', preserve_range=True)
        # label5 = skimage.transform.resize(label, (label.shape[0] // 16, label.shape[1] // 16),
        #                                   order=0, mode='reflect', preserve_range=True)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        depth = np.expand_dims(depth, 0).astype(np.float32)
        return {'image': torch.from_numpy(image).float(),
                'depth': torch.from_numpy(depth).float(),
                'label': torch.from_numpy(label).float(),
                # 'label2': torch.from_numpy(label2).float(),
                # 'label3': torch.from_numpy(label3).float(),
                # 'label4': torch.from_numpy(label4).float(),
                # 'label5': torch.from_numpy(label5).float()
                }
