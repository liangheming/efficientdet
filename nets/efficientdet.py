import torch
import math
from torch import nn
from nets.efficientnet import EfficientNet
from utils.retinanet import BoxCoder
from nets.efficientnet.utils import Conv2dDynamicSamePadding, MaxPool2dDynamicSamePadding, MemoryEfficientSwish


class DWSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=False, norm=True, act=True):
        super(DWSConv2d, self).__init__()

        self.conv = nn.Sequential(
            Conv2dDynamicSamePadding(in_channels, in_channels, kernel_size, stride, groups=in_channels, bias=bias),
            Conv2dDynamicSamePadding(in_channels, out_channels, 1, 1, bias=bias)
        )
        self.norm = norm
        self.act = act
        if norm:
            self.bn = nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3)
        if act:
            self.ac = MemoryEfficientSwish()

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.bn(x)
        if self.act:
            x = self.ac(x)
        return x


class ScaleWeight(nn.Module):
    def __init__(self, weight_num=2, init_val=1.0, requires_grad=True, eps=1e-4):
        super(ScaleWeight, self).__init__()
        weights = torch.ones(size=(weight_num,)) * init_val
        self.eps = eps
        self.weights = nn.Parameter(weights, requires_grad=requires_grad)

    def forward(self, xs):
        assert len(self.weights) == len(xs)
        positive_weights = self.weights.relu()
        positive_weights = positive_weights / positive_weights.sum() + self.eps
        ret = 0.
        for w, x in zip(positive_weights, xs):
            ret += (w * x)
        return ret


class BiFPNLayer(nn.Module):
    def __init__(self, c3, c4, c5, inner_channels, weight_inputs=True, first=False):
        super(BiFPNLayer, self).__init__()
        self.first = first
        if self.first:
            self.c3_latent = nn.Sequential(
                Conv2dDynamicSamePadding(c3, inner_channels, 1),
                nn.BatchNorm2d(inner_channels, momentum=0.01, eps=1e-3)
            )
            self.c4_latent = nn.Sequential(
                Conv2dDynamicSamePadding(c4, inner_channels, 1),
                nn.BatchNorm2d(inner_channels, momentum=0.01, eps=1e-3)
            )

            self.c5_latent = nn.Sequential(
                Conv2dDynamicSamePadding(c5, inner_channels, 1),
                nn.BatchNorm2d(inner_channels, momentum=0.01, eps=1e-3)
            )

            self.c5_to_p6 = nn.Sequential(
                Conv2dDynamicSamePadding(c5, inner_channels, 1),
                nn.BatchNorm2d(inner_channels, momentum=0.01, eps=1e-3),
                MaxPool2dDynamicSamePadding(3, 2)
            )

            self.p6_to_p7 = nn.Sequential(
                MaxPool2dDynamicSamePadding(3, 2)
            )

            self.c4_latent_re = nn.Sequential(
                Conv2dDynamicSamePadding(c4, inner_channels, 1),
                nn.BatchNorm2d(inner_channels, momentum=0.01, eps=1e-3)
            )

            self.c5_latent_re = nn.Sequential(
                Conv2dDynamicSamePadding(c5, inner_channels, 1),
                nn.BatchNorm2d(inner_channels, momentum=0.01, eps=1e-3)
            )

        self.p6_0 = DWSConv2d(inner_channels, inner_channels, act=False)
        self.p6_0_scale = ScaleWeight(2, requires_grad=weight_inputs)

        self.p5_0 = DWSConv2d(inner_channels, inner_channels, act=False)
        self.p5_0_scale = ScaleWeight(2, requires_grad=weight_inputs)

        self.p4_0 = DWSConv2d(inner_channels, inner_channels, act=False)
        self.p4_0_scale = ScaleWeight(2, requires_grad=weight_inputs)

        self.p3_1 = DWSConv2d(inner_channels, inner_channels, act=False)
        self.p3_1_scale = ScaleWeight(2, requires_grad=weight_inputs)

        self.p4_1 = DWSConv2d(inner_channels, inner_channels, act=False)
        self.p4_1_scale = ScaleWeight(3, requires_grad=weight_inputs)

        self.p5_1 = DWSConv2d(inner_channels, inner_channels, act=False)
        self.p5_1_scale = ScaleWeight(3, requires_grad=weight_inputs)

        self.p6_1 = DWSConv2d(inner_channels, inner_channels, act=False)
        self.p6_1_scale = ScaleWeight(3, requires_grad=weight_inputs)

        self.p7_1 = DWSConv2d(inner_channels, inner_channels, act=False)
        self.p7_1_scale = ScaleWeight(2, requires_grad=weight_inputs)

        self.up_sample = nn.UpsamplingNearest2d(scale_factor=2)
        self.down_sample = MaxPool2dDynamicSamePadding(3, 2)
        self.act = MemoryEfficientSwish()

    def forward(self, xs):
        c3, c4, c5 = xs[:3]
        if self.first:
            p3_in = self.c3_latent(c3)
            p4_in = self.c4_latent(c4)
            p5_in = self.c5_latent(c5)
            p6_in = self.c5_to_p6(c5)
            p7_in = self.p6_to_p7(p6_in)
        else:
            p3_in = c3
            p4_in = c4
            p5_in = c5
            p6_in, p7_in = xs[3:]
        p6_0 = self.p6_0(self.act(self.p6_0_scale([self.up_sample(p7_in), p6_in])))
        p5_0 = self.p5_0(self.act(self.p5_0_scale([self.up_sample(p6_0), p5_in])))
        p4_0 = self.p4_0(self.act(self.p4_0_scale([self.up_sample(p5_0), p4_in])))
        p3_out = self.p3_1(self.act(self.p3_1_scale([self.up_sample(p4_0), p3_in])))
        if self.first:
            # reference to https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
            p4_in = self.c4_latent_re(c4)
            p5_in = self.c5_latent_re(c5)
        p4_out = self.p4_1(self.act(self.p4_1_scale([self.down_sample(p3_out), p4_0, p4_in])))
        p5_out = self.p5_1(self.act(self.p5_1_scale([self.down_sample(p4_out), p5_0, p5_in])))
        p6_out = self.p6_1(self.act(self.p6_1_scale([self.down_sample(p5_out), p6_0, p6_in])))
        p7_out = self.p7_1(self.act(self.p7_1_scale([self.down_sample(p6_out), p7_in])))
        return [p3_out, p4_out, p5_out, p6_out, p7_out]


class BiFPN(nn.Module):
    def __init__(self, c3, c4, c5, inner_channels, repeat_times, weight_input=True):
        super(BiFPN, self).__init__()
        self.bi_fpn = list()
        for i in range(repeat_times):
            if i == 0:
                self.bi_fpn.append(BiFPNLayer(c3, c4, c5, inner_channels, weight_input, first=True))
            else:
                self.bi_fpn.append(BiFPNLayer(c3, c4, c5, inner_channels, weight_input, first=False))
        self.bi_fpn = nn.Sequential(*self.bi_fpn)

    def forward(self, xs):
        for fpn in self.bi_fpn:
            xs = fpn(xs)
        return xs


class ClsHead(nn.Module):
    def __init__(self, in_channel,
                 inner_channel,
                 num_layers,
                 num_anchors=9, num_cls=80):
        super(ClsHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_cls = num_cls
        self.bones = list()
        for i in range(num_layers):
            if i == 0:
                conv = DWSConv2d(in_channel, inner_channel)
            else:
                conv = DWSConv2d(inner_channel, inner_channel)
            self.bones.append(conv)
        self.bones = nn.Sequential(*self.bones)
        self.cls = nn.Conv2d(inner_channel, self.num_anchors * self.num_cls, 3, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.cls.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, x):
        x = self.bones(x)
        x = self.cls(x)
        bs, _, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous() \
            .view(bs, h, w, self.num_anchors, self.num_cls) \
            .view(bs, -1, self.num_cls)
        return x


class RegHead(nn.Module):
    def __init__(self, in_channel, inner_channel=256, num_layers=4, num_anchors=9, ):
        super(RegHead, self).__init__()
        self.num_anchors = num_anchors
        self.bones = list()
        for i in range(num_layers):
            if i == 0:
                conv = DWSConv2d(in_channel, inner_channel)
            else:
                conv = DWSConv2d(inner_channel, inner_channel)
            self.bones.append(conv)
        self.bones = nn.Sequential(*self.bones)
        self.reg = nn.Conv2d(inner_channel, self.num_anchors * 4, 3, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.bones(x)
        x = self.reg(x)
        bs, _, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous() \
            .view(bs, h, w, self.num_anchors, 4) \
            .view(x.size(0), -1, 4)
        return x


default_anchor_sizes = [32., 64., 128., 256., 512.]
default_strides = [8, 16, 32, 64, 128]
default_anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
default_anchor_ratios = [0.5, 1., 2.]


class DetHead(nn.Module):
    def __init__(self,
                 inner_channel,
                 conv_depth,
                 num_cls=80,
                 fpn_layers_num=5,
                 anchor_sizes=None,
                 anchor_scales=None,
                 anchor_ratios=None,
                 strides=None):
        super(DetHead, self).__init__()
        self.num_cls = num_cls
        self.fpn_layers_num = fpn_layers_num
        if anchor_sizes is None:
            anchor_sizes = default_anchor_sizes
        self.anchor_sizes = anchor_sizes
        if anchor_scales is None:
            anchor_scales = default_anchor_scales
        self.anchor_scales = anchor_scales
        if anchor_ratios is None:
            anchor_ratios = default_anchor_ratios
        self.anchor_ratios = anchor_ratios
        if strides is None:
            strides = default_strides
        self.strides = strides
        self.anchor_nums = len(self.anchor_scales) * len(self.anchor_ratios)
        self.anchors = [torch.zeros(size=(0, 4))] * self.fpn_layers_num
        self.cls_head = ClsHead(inner_channel, inner_channel, conv_depth, self.anchor_nums, num_cls)
        self.reg_head = RegHead(inner_channel, inner_channel, conv_depth, self.anchor_nums, )
        self.box_coder = BoxCoder()

    def build_anchors_delta(self, size=32.):
        """
        :param size:
        :return: [anchor_num, 4]
        """
        scales = torch.tensor(self.anchor_scales).float()
        ratio = torch.tensor(self.anchor_ratios).float()
        scale_size = (scales * size)
        w = (scale_size[:, None] * ratio[None, :].sqrt()).view(-1) / 2
        h = (scale_size[:, None] / ratio[None, :].sqrt()).view(-1) / 2
        delta = torch.stack([-w, -h, w, h], dim=1)
        return delta

    def build_anchors(self, feature_maps):
        """
        :param feature_maps:
        :return: list(anchor) anchor:[all,4] (x1,y1,x2,y2)
        """
        assert self.fpn_layers_num == len(feature_maps)
        assert len(self.anchor_sizes) == len(feature_maps)
        assert len(self.anchor_sizes) == len(self.strides)
        anchors = list()
        for stride, size, feature_map in zip(self.strides, self.anchor_sizes, feature_maps):
            # 9*4
            anchor_delta = self.build_anchors_delta(size)
            _, _, ny, nx = feature_map.shape
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            # h,w,4
            grid = torch.stack([xv, yv, xv, yv], 2).float()
            anchor = (grid[:, :, None, :] + 0.5) * stride + anchor_delta[None, None, :, :]
            anchor = anchor.view(-1, 4)
            anchors.append(anchor)
        return anchors

    def forward(self, xs):
        cls_outputs = list()
        reg_outputs = list()
        for j, x in enumerate(xs):
            cls_outputs.append(self.cls_head(x))
            reg_outputs.append(self.reg_head(x))
        if self.anchors[0] is None or self.anchors[0].shape[0] != cls_outputs[0].shape[1]:
            with torch.no_grad():
                anchors = self.build_anchors(xs)
                assert len(anchors) == len(self.anchors)
                for i, anchor in enumerate(anchors):
                    self.anchors[i] = anchor.to(xs[0].device)
        if self.training:
            return cls_outputs, reg_outputs, self.anchors
        else:
            predicts_list = list()
            for cls_out, reg_out, anchor in zip(cls_outputs, reg_outputs, self.anchors):
                scale_reg = self.box_coder.decoder(reg_out, anchor)
                predicts_out = torch.cat([scale_reg, cls_out], dim=-1)
                predicts_list.append(predicts_out)
            return predicts_list


class EfficientDet(nn.Module):
    def __init__(self,
                 anchor_scales=None,
                 anchor_ratios=None,
                 strides=None,
                 num_cls=80,
                 compound_coef=0,
                 weights_path=None):
        super(EfficientDet, self).__init__()
        self.backbones = EfficientNet.from_pretrained('efficientnet-b{:d}'.format(compound_coef),
                                                      weights_path=weights_path)
        c3, c4, c5 = self.backbones.out_channels
        bifpn_w = int(64 * (1.35 ** compound_coef))
        bifpn_d = int(3 + compound_coef)
        head_d = int(3 + (compound_coef / 3))
        anchor_expand_factor = [3., 4., 4., 4., 4., 4., 5., 5.]
        if strides is None:
            strides = default_strides
        anchor_sizes = [anchor_expand_factor[int(compound_coef)] * stride_item for stride_item in strides]
        self.neck = BiFPN(c3, c4, c5, bifpn_w, bifpn_d)
        self.head = DetHead(
            bifpn_w,
            head_d,
            num_cls,
            5,
            anchor_sizes,
            anchor_scales,
            anchor_ratios,
            strides
        )

    def forward(self, x):
        c3, c4, c5 = self.backbones(x)
        p3, p4, p5, p6, p7 = self.neck([c3, c4, c5])
        out = self.head([p3, p4, p5, p6, p7])
        return out

# if __name__ == '__main__':
#     input_tensor = torch.rand(size=(4, 3, 512, 512)).float()
#     net = EfficientDet()
#     mcls_output, mreg_output, manchor = net(input_tensor)
#     for cls_out, reg_out, anchor_out in zip(mcls_output, mreg_output, manchor):
#         print(cls_out.shape, reg_out.shape, anchor_out.shape)
#     net.eval()
#     out = net(input_tensor)
#     for item in out:
#         print(item.shape)
