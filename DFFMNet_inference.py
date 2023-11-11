import imageio
import torch
import torch.optim
from skimage.transform import resize
from torchvision.transforms import Normalize

import RedNet_model
from utils import utils
from utils.utils import load_ckpt

device = torch.device("cuda:0" if False and torch.cuda.is_available() else "cpu")
image_w = 640
image_h = 480


def inference(rgb_path, depth_path, last_ckpt):
    model = RedNet_model.RedNet(pretrained=False)

    load_ckpt(model, None, last_ckpt, device)

    # 开始eval
    model.eval()
    model.to(device)

    image = imageio.v2.imread(rgb_path)
    depth = imageio.v2.imread(depth_path)

    # 打开图片
    # Bi-linear
    image = resize(image, (image_h, image_w), order=1, mode='reflect', preserve_range=True)
    # Nearest-neighbor
    depth = resize(depth, (image_h, image_w), order=1, mode='reflect', preserve_range=True)

    image = image / 255
    image = torch.from_numpy(image).float()
    depth = torch.from_numpy(depth).float()

    image = image.permute(2, 0, 1)
    depth.unsqueeze_(0)

    image = Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])(image)
    depth = Normalize(mean=[19050],
                                             std=[9650])(depth)

    image = image.to(device).unsqueeze_(0)
    depth = depth.to(device).unsqueeze_(0)

    pred = model(image, depth)
    print(image.shape,depth.shape)
    print(pred)


    output = utils.color_label_nyuv2(torch.max(pred, 1)[1] + 1)[0]

    path = rgb_path.split("/")[3] + rgb_path.split("/")[4] + rgb_path.split("/")[-1][0:-4]
    outpath = last_ckpt.split('/')[-1][0:-7]

    imageio.imsave(f'result/{path}.png', image[0].cpu().numpy().transpose((1, 2, 0)))
    imageio.imsave(f'result/{path}_{outpath}.png', output.cpu().numpy().transpose((1, 2, 0)))


if __name__ == '__main__':
    from MFFMNet_data_NYUv2 import img_dir_file, depth_dir_file

    with open(img_dir_file, 'r') as f:
        img_dir_test = f.read().splitlines()
    with open(depth_dir_file, 'r') as f:
        depth_dir_test = f.read().splitlines()
    # print(img_dir_test.__len__())# 5050
    import random

    result = random.sample(range(0, 1449), 20)
    # print(result)

    path = [f'./model/ckpt_epoch_{i}.00.pth' for i in [5]]

    for i in result:
        rgb_path = img_dir_test[i]
        depth_path = depth_dir_test[i]
        for p in path:
            inference(rgb_path, depth_path, p)


