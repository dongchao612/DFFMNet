import os
import imageio
from PIL import Image
from torch.utils.data import Dataset
from DFFMNet_transforms import *
import torchvision.transforms as transforms

img_dir_train_file = './data_SUNRGBD/img_dir_train.txt'
depth_dir_train_file = './data_SUNRGBD/depth_dir_train.txt'
label_dir_train_file = './data_SUNRGBD/label_train.txt'
img_dir_test_file = './data_SUNRGBD/img_dir_test.txt'
depth_dir_test_file = './data_SUNRGBD/depth_dir_test.txt'
label_dir_test_file = './data_SUNRGBD/label_test.txt'


class SUNRGBD(Dataset):
    def __init__(self, transform=None, phase_train=True, data_dir=None):
        self.phase_train = phase_train
        self.transform = transform
        try:
            with open(img_dir_train_file, 'r') as f:
                self.img_dir_train = f.read().splitlines()
            with open(depth_dir_train_file, 'r') as f:
                self.depth_dir_train = f.read().splitlines()
            with open(label_dir_train_file, 'r') as f:
                self.label_dir_train = f.read().splitlines()

            with open(img_dir_test_file, 'r') as f:
                self.img_dir_test = f.read().splitlines()
            with open(depth_dir_test_file, 'r') as f:
                self.depth_dir_test = f.read().splitlines()
            with open(label_dir_test_file, 'r') as f:
                self.label_dir_test = f.read().splitlines()
        except:
            print("开始生成txt文件...")
            if data_dir is None:
                data_dir = r'D:\Document\github\data_set\sun_rgbd'

            self.img_dir_train = []
            self.depth_dir_train = []
            self.label_dir_train = []

            self.img_dir_test = []
            self.depth_dir_test = []
            self.label_dir_test = []

            depthpath = os.path.join(data_dir, os.listdir(data_dir)[0])
            imagepath = os.path.join(data_dir, os.listdir(data_dir)[1])
            labelpath = os.path.join(data_dir, os.listdir(data_dir)[-1])

            itempath_train = os.path.join(imagepath, "train")
            itempath_test = os.path.join(imagepath, "test")
            # print(itempath_train,itempath_test)
            for item in os.listdir(itempath_train):
                self.img_dir_train.append(os.path.join(itempath_train, item))
            for item in os.listdir(itempath_test):
                self.img_dir_test.append(os.path.join(itempath_test, item))

            itempath_train = os.path.join(depthpath, "train")
            itempath_test = os.path.join(depthpath, "test")
            # print(itempath_train,itempath_test)
            for item in os.listdir(itempath_train):
                self.depth_dir_train.append(os.path.join(itempath_train, item))
            for item in os.listdir(itempath_test):
                self.depth_dir_test.append(os.path.join(itempath_test, item))

            itempath_train = os.path.join(labelpath, "train")
            itempath_test = os.path.join(labelpath, "test")
            # print(itempath_train,itempath_test)
            for item in os.listdir(itempath_train):
                self.label_dir_train.append(os.path.join(itempath_train, item))
            for item in os.listdir(itempath_test):
                self.label_dir_test.append(os.path.join(itempath_test, item))

            with open(img_dir_train_file, 'w') as f:
                f.write('\n'.join(self.img_dir_train))
            with open(depth_dir_train_file, 'w') as f:
                f.write('\n'.join(self.depth_dir_train))
            with open(label_dir_train_file, 'w') as f:
                f.write('\n'.join(self.label_dir_train))
            with open(img_dir_test_file, 'w') as f:
                f.write('\n'.join(self.img_dir_test))
            with open(depth_dir_test_file, 'w') as f:
                f.write('\n'.join(self.depth_dir_test))
            with open(label_dir_test_file, 'w') as f:
                f.write('\n'.join(self.label_dir_test))

    def __len__(self):
        if self.phase_train:
            return len(self.img_dir_train)
        else:
            return len(self.img_dir_test)

    def __getitem__(self, idx):
        if self.phase_train:
            img_dir = self.img_dir_train
            depth_dir = self.depth_dir_train
            label_dir = self.label_dir_train
        else:
            img_dir = self.img_dir_test
            depth_dir = self.depth_dir_test
            label_dir = self.label_dir_test

        image = imageio.v2.imread(img_dir[idx])
        depth = imageio.v2.imread(depth_dir[idx])
        label = imageio.v2.imread(label_dir[idx])
        # (530, 730, 3) (530, 730) (530, 730)
        # print(image.shape, depth.shape, label.shape)
        sample = {'image': image, 'depth': depth, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    train_data = SUNRGBD(phase_train=True, transform=transforms.Compose([scaleNorm(), ToTensor(), Normalize()]))
    print(train_data.__len__())  # 5285
    print(train_data[0]["image"].shape, train_data[0]["depth"].shape, train_data[0]["label"].shape)

    test_data = SUNRGBD(phase_train=False, transform=transforms.Compose([scaleNorm(), ToTensor(), Normalize()]))
    print(test_data.__len__())  # 5050
    print(test_data[0]["image"].shape, test_data[0]["depth"].shape, test_data[0]["label"].shape)

    print(train_data.__len__() + test_data.__len__())  # 10335

    # import matplotlib.pyplot as plt
    # import random
    #
    # plt.figure(figsize=(15, 10))
    #
    # ix = random.randint(0, train_data.__len__() - 1)
    # plt.subplot(2, 3, 1)
    # plt.imshow(train_data[ix]["image"])
    #
    # plt.subplot(2, 3, 2)
    # plt.imshow(train_data[ix]["depth"], cmap='gray')
    #
    # plt.subplot(2, 3, 3)
    # plt.imshow(train_data[ix]["label"])
    #
    # ix = random.randint(0, train_data.__len__() - 1)
    # plt.subplot(2, 3, 4)
    # plt.imshow(test_data[ix]["image"])
    #
    # plt.subplot(2, 3, 5)
    # plt.imshow(test_data[ix]["depth"], cmap='gray')
    #
    # plt.subplot(2, 3, 6)
    # plt.imshow(test_data[ix]["label"])
    #
    # plt.savefig("datasetSUNRGBD.png")
    # plt.show()
