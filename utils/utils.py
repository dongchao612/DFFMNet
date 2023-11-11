import numpy as np
import torch

color_label_sun = [(0, 0, 0),
                   # 0=background
                   (148, 65, 137), (255, 116, 69), (86, 156, 137),
                   (202, 179, 158), (155, 99, 235), (161, 107, 108),
                   (133, 160, 103), (76, 152, 126), (84, 62, 35),
                   (44, 80, 130), (31, 184, 157), (101, 144, 77),
                   (23, 197, 62), (141, 168, 145), (142, 151, 136),
                   (115, 201, 77), (100, 216, 255), (57, 156, 36),
                   (88, 108, 129), (105, 129, 112), (42, 137, 126),
                   (155, 108, 249), (166, 148, 143), (81, 91, 87),
                   (100, 124, 51), (73, 131, 121), (157, 210, 220),
                   (134, 181, 60), (221, 223, 147), (123, 108, 131),
                   (161, 66, 179), (163, 221, 160), (31, 146, 98),
                   (99, 121, 30), (49, 89, 240), (116, 108, 9),
                   (161, 176, 169), (80, 29, 135), (177, 105, 197),
                   (139, 110, 246)]

color_label_nyuv2 = [
    (0, 0, 0),
    # 0=background
    (127, 20, 22), (9, 128, 64), (127, 128, 51), (40, 41, 115), (125, 39, 125), (0, 128, 128),
    (127, 127, 127), (57, 16, 18),
    (191, 32, 38), (65, 128, 61), (191, 128, 43), (67, 41, 122), (192, 27, 128), (64, 128, 127),
    (191, 127, 127), (28, 64, 28),
    (127, 66, 28), (47, 180, 74), (127, 192, 66), (29, 67, 126), (128, 64, 127), (47, 183, 127),
    (127, 192, 127), (65, 65, 25),
    (191, 67, 38), (75, 183, 73), (190, 192, 49), (64, 64, 127), (193, 65, 128), (74, 187, 127),
    (192, 192, 127), (11, 17, 60),
    (127, 21, 66), (0, 128, 65), (127, 127, 63), (47, 65, 154), (117, 64, 153), (8, 127, 191),
    (127, 127, 189), (63, 9, 63)
]


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            # 创建混淆矩阵
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            # 寻找GT中为目标的像素索引
            k = (a >= 0) & (a < n)
            # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def compute(self):
        h = self.mat.float()
        # 计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)
        acc_global = torch.diag(h).sum() / h.sum()

        # 计算每个类别的准确率
        acc = torch.diag(h) / h.sum(1)

        # 计算每个类别预测与真实目标的iou
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
            acc_global.item() * 100,
            ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
            ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
            iu.mean().item() * 100)


def color_label(label, label_colours):
    label = label.clone().cpu().data.numpy()
    colored_label = np.vectorize(lambda x: label_colours[int(x)])

    colored = np.asarray(colored_label(label)).astype(np.float32)
    colored = colored.squeeze()

    try:
        return torch.from_numpy(colored.transpose([1, 0, 2, 3]))
    except ValueError:
        return torch.from_numpy(colored[np.newaxis, ...])


import time
import torch.backends.cudnn as cudnn

def compute_speed(model, rgb_size, depth_size, device, iteration=100):
    torch.cuda.set_device(device)
    cudnn.benchmark = True

    model.eval()
    model = model.cuda()

    rgb = torch.randn(*rgb_size, device=device)
    depth = torch.randn(*depth_size, device=device)
    for _ in range(50):
        model(rgb, depth)

    print('=========Speed Testing=========')
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iteration):
        model(rgb, depth)
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    speed_time = elapsed_time / iteration * 1000
    fps = iteration / elapsed_time

    print('Elapsed Time: [%.2f s / %d iter]' % (elapsed_time, iteration))
    print('Speed Time: %.2f ms / iter   FPS: %.2f' % (speed_time, fps))
    return speed_time, fps