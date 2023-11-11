import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchvision.transforms import Compose

from DFFMNet_data_SUNRGBD import SUNRGBD
from DFFMNet_data_NYUv2 import NYUv2

from DFFMNet_model import MFFMNet
from DFFMNet_transforms import ToTensor, Normalize
from tensorboardX import SummaryWriter
import torch
from utils.utils import color_label, color_label_nyuv2, color_label_sun

import matplotlib.pyplot as plt
if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    device = 'cpu'
    batch_size = 2
    workers = 2
    epochs = 5

    lr = 0.0001
    momentum = 0.9
    weight_decay = 0.8
    amp = False
    NYUv2_class = 40 + 1

    train_data = NYUv2(transform=Compose([ToTensor(), Normalize()]), phase_train=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True,
                              drop_last=True, persistent_workers=True)

    model = MFFMNet(NYUv2_class, pretrained_model=None, norm_layer=nn.BatchNorm2d).to(
        device)

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    loss_fn = CrossEntropyLoss()

    writer = SummaryWriter("logs")

    print("device={}".format(device))

    model.train()
    e_loss_lst = []
    gobal_step = 0
    for e in range(epochs):
        e_loss = 0
        for batch_i, data in enumerate(train_loader):
            image = data['image'].to(device)
            depth = data['depth'].to(device)
            target = data['label'].to(device)

            output = model(image, depth)[0]
            # print(torch.unique(torch.argmax(output, 1)))
            loss = loss_fn(output, target.long())

            writer.add_images('img', image, gobal_step)
            writer.add_images('label', target, gobal_step)
            writer.add_images('output', color_label(torch.argmax(output, 1), color_label_nyuv2), gobal_step)  # 没有+1
            gobal_step = gobal_step + 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("epoch = {}\tbatch = {}\t loss={}".format(e, batch_i, loss))

        print("epoch {} agv loss = {}".format(e, e_loss / train_loader.__len__()))
        e_loss_lst.append(e_loss / train_loader.__len__())

        # 模型参数（官方推荐）
        torch.save(model.state_dict(), "save_model/model.pth")
        plt.plot(e_loss_lst)
        plt.savefig("loss_{}.png".format(epochs))

        writer.close()
