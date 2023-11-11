# TODO
import torch
from tensorboardX.utils import make_grid

from utils import utils
pred_scales = [torch.randn(1, 40, 480, 640)]
# print(utils.color_label_nyuv2)
print(torch.max(pred_scales[0][:3], 1)[1].shape)

grid_image = make_grid(utils.color_label_nyuv2(torch.max(pred_scales[0][:3], 1)[1] + 1), 3)


