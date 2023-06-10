import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from MFFMNet_data_SUNRGBD import SUNRGBD
from MFFMNet_model import MFFMNet
from MFFMNet_transforms import *

if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    batch_size = 2
    data_dir = "/path/to/SUNRGB-D"
    workers = 2
    epochs = 500

    lr = 0.0001
    momentum = 0.9
    weight_decay = 0.8
    amp = False
    NYUv2_class = 40

    train_data = SUNRGBD(transform=transforms.Compose([scaleNorm(), ToTensor(), Normalize()]), phase_train=True,
                         data_dir=data_dir)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True,drop_last=True,
                              persistent_workers=True)

    model = MFFMNet(NYUv2_class, pretrained_model=None, norm_layer=nn.BatchNorm2d).to(
        device)

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    loss_fn = CrossEntropyLoss()

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)

    print("device={}".format(device))

    testSize = 128
    model.train()
    e_loss_lst = []
    for e in range(1):
        e_loss = 0
        for batch_i, data in enumerate(train_loader):
            image = data['image'].to(device)  # torch.Size([2, 3, 480, 640])
            depth = data['depth'].to(device)  # torch.Size([2, 1, 480, 640])

            target = data['label'].to(device)  # torch.Size([2, 480, 640])

            output = model(image, depth)[0]  # torch.Size([2, 37, 480, 640])

            loss = loss_fn(output, target.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("epoch = {}\tbatch = {}\t loss={}".format(e, batch_i, loss))
            break
        # print("epoch {} agv loss = {}".format(e, e_loss / train_loader.__len__()))
        # e_loss_lst.append(e_loss / train_loader.__len__())

        # torch.save(model, "save_model/model.pt")
        # plt.plot(e_loss_lst)
        # plt.savefig("loss_{}.png".format(epochs))
