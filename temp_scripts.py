import torch
from nets.efficientnet import EfficientNet
from nets.efficientdet import BiFPN

if __name__ == '__main__':
    input_tensor = torch.rand(size=(1, 3, 640, 640))
    model = EfficientNet.from_pretrained('efficientnet-b1')
    c3, c4, c5 = model.out_channels
    out = model(input_tensor)
    fpn = BiFPN(c3, c4, c5, 64, 3)
    out = fpn(out)
    for item in out:
        print(item.shape)
