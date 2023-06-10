import os
import imageio
from torch.utils.data import Dataset, random_split
from MFFMNet_transforms import *
import torchvision.transforms as transforms

image_w = 640
image_h = 480

img_dir_file = './data_NYUv2/img_dir.txt'
depth_dir_file = './data_NYUv2/depth_dir.txt'
label_dir_file = './data_NYUv2/label.txt'


class NYUv2(Dataset):
    def __init__(self, transform=None, phase_train=True, data_dir=None):
        self.phase_train = phase_train
        self.transform = transform
        try:
            with open(img_dir_file, 'r') as f:
                self.img_dir = f.read().splitlines()
            with open(depth_dir_file, 'r') as f:
                self.depth_dir = f.read().splitlines()
            with open(label_dir_file, 'r') as f:
                self.label_dir = f.read().splitlines()
        except:
            print("开始生成txt文件...")
            if data_dir is None:
                data_dir = r'D:\Document\github\data_set\NYUv2'
            self.img_dir = []
            self.depth_dir = []
            self.label_dir = []

            depthpath = os.path.join(data_dir, os.listdir(data_dir)[0])
            imagepath = os.path.join(data_dir, os.listdir(data_dir)[1])
            labelpath = os.path.join(data_dir, os.listdir(data_dir)[-1])

            for item in os.listdir(depthpath):
                self.depth_dir.append(os.path.join(depthpath, item))
            for item in os.listdir(imagepath):
                self.img_dir.append(os.path.join(imagepath, item))
            for item in os.listdir(labelpath):
                self.label_dir.append(os.path.join(labelpath, item))

            with open(img_dir_file, 'w') as f:
                f.write('\n'.join(self.img_dir))
            with open(depth_dir_file, 'w') as f:
                f.write('\n'.join(self.depth_dir))
            with open(label_dir_file, 'w') as f:
                f.write('\n'.join(self.label_dir))

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img_dir = self.img_dir
        depth_dir = self.depth_dir
        label_dir = self.label_dir

        label = imageio.v2.imread(label_dir[idx])
        depth = imageio.v2.imread(depth_dir[idx])
        image = imageio.v2.imread(img_dir[idx])


        sample = {'image': image, 'depth': depth, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    data = NYUv2(transform=transforms.Compose([scaleNorm(), RandomScale((1.0, 1.4)), RandomHSV((0.9, 1.1),
                                                                                                     (0.9, 1.1),
                                                                                                     (25, 25)),
                                                     RandomCrop(image_h, image_w),
                                                     RandomFlip(),
                                                     ToTensor(),
                                                     Normalize()]), phase_train=True)
    print(data[0]["image"].shape)  # torch.Size([3, 480, 640])
    print(data[0]["depth"].shape)  # torch.Size([1, 480, 640])
    print(data[0]["label"].shape)  # torch.Size([480, 640])
    print(data.__len__())  # 1449

    # import matplotlib.pyplot as plt
    #
    # data = NYUv2()
    # plt.subplot(1, 3, 1)
    # plt.imshow(data[0]["image"])
    #
    # plt.subplot(1, 3, 2)
    # plt.imshow(data[0]["depth"])
    #
    # plt.subplot(1, 3, 3)
    # plt.imshow(data[0]["label"])
    #
    # plt.savefig("datasetNYUv2.png")
    # plt.show()
