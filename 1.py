import torch
import torchvision
import torchvision.models
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch import nn
from eca_module import eca_layer
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math
import torchvision
import torch.optim as optim  # Pytoch常用的优化方法都封装在torch.optim里面
import matplotlib.pyplot as plt
from SE_weight_module import SEWeightModule
from attention import *
from foloss import FocalLoss
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from collections import OrderedDict
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(120),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((120, 120)),  # 我这里是缩放成120*120进行处理的，您可以缩放成你需要的比例，但是不建议修改，因为会影响全连接成的输出
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}


def main():
    train_data = torchvision.datasets.ImageFolder(root="./data/train",
                                                  transform=data_transform["train"])

    traindata = DataLoader(dataset=train_data, batch_size=16, shuffle=True, num_workers=0)

    test_data = torchvision.datasets.ImageFolder(root="./data/val",
                                                 transform=data_transform["val"])

    train_size = len(train_data)  # 求出训练集的长度
    test_size = len(test_data)  # 求出测试集的长度
    print(train_size)
    print(test_size)
    testdata = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 如果有GPU就使用GPU，否则使用CPU
    print("using {} device.".format(device))

    class IntermediateLayerGetter(nn.ModuleDict):
        def __init__(self, model, return_layers):
            # 首先判断 return_layer中的key 是否在model中
            if not set(return_layers).issubset([name for name, _ in model.named_children()]):
                raise ValueError("return_layers are not present in model")

            orig_return_layers = return_layers
            return_layers = {str(k): str(v) for k, v in return_layers.items()}
            layers = OrderedDict()

            # 遍历模型子模块按顺序存入有序字典
            # 只保存layer4及其之前的结构，舍去之后不用的结构
            for name, module in model.named_children():
                layers[name] = module
                if name in return_layers:
                    del return_layers[name]
                if not return_layers:
                    break

            super(IntermediateLayerGetter, self).__init__(layers)
            self.return_layers = orig_return_layers

        def forward(self, x):
            out = OrderedDict()
            # 依次遍历模型的所有子模块，并进行正向传播，
            # 收集layer1, layer2, layer3, layer4的输出
            for name, module in self.items():
                x = module(x)
                if name in self.return_layers:
                    out_name = self.return_layers[name]
                    out[out_name] = x
            return out

    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, in_channel, out_channel, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
            super(Bottleneck, self).__init__()

            self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(out_channel)

            self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channel)

            self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion, kernel_size=1,
                                   stride=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample

        def forward(self, x):
            identify = x
            if self.downsample is not None:
                identify = self.downsample(x)

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            out += identify
            out = self.relu(out)

            return out

    class ResNet(nn.Module):
        # 仅包含分类器前面的模型结构
        def __init__(self, block, blocks_num, num_classes=4):
            super(ResNet, self).__init__()

            self._norm_layer = nn.BatchNorm2d

            self.in_channel = 64

            self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(self.in_channel)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, blocks_num[0])
            self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
            self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
            self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        def _make_layer(self, block, channel, block_num, stride=1):
            norm_layer = self._norm_layer
            downsample = None
            if stride != 1 or self.in_channel != channel * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                    norm_layer(channel * block.expansion))

            layers = []
            layers.append(block(self.in_channel, channel, downsample=downsample,
                                stride=stride, norm_layer=norm_layer))
            self.in_channel = channel * block.expansion

            for _ in range(1, block_num):
                layers.append(block(self.in_channel, channel, norm_layer=norm_layer))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            return x

    class MaxPool_Block(nn.Module):

        def forward(self, x, names):
            names.append('pool')
            # input, kernel_size, stride, padding
            x.append(F.max_pool2d(x[-1], 1, 2, 0))
            return x, names

    class FPN(nn.Module):
        def __init__(self, backbone_output_channels_list, out_channels, maxpool_blocks=True):
            super(FPN, self).__init__()

            # 用来调整resnet输出特征矩阵(layer1,2,3,4)的channel（kernel_size=1）
            self.inner_blocks = nn.ModuleList()
            # 对调整后的特征矩阵使用3x3的卷积核来得到对应的预测特征矩阵
            self.layer_blocks = nn.ModuleList()

            for backbone_channels in backbone_output_channels_list:
                inner_block_module = nn.Conv2d(backbone_channels, out_channels, 1)
                layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)

                self.inner_blocks.append(inner_block_module)
                self.layer_blocks.append(layer_block_module)
            # 初始化权重参数
            for m in self.children():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight, a=1)
                    nn.init.constant_(m.bias, 0)

            self.extra_blocks = maxpool_blocks

        def forward(self, x):
            # 对应backbone中layer1、2、3、4特征图的输出的key
            names = list(x.keys())
            # 对应backbone中layer1、2、3、4特征图的输出的value
            x = list(x.values())
            # backbone中layer4的输出调整到指定维度上 channels: 2048->256
            last_inner = self.inner_blocks[-1](x[-1])
            # results中保存着经过3*3conv后的每个预测特征层
            results = []
            # layer4
            results.append(self.layer_blocks[-1](last_inner))
            # layer4+layer3
            layer3_inner = self.inner_blocks[2](x[2])
            layer3_shape = layer3_inner.shape[-2:]
            last_layer3 = F.interpolate(last_inner, size=layer3_shape, mode="nearest")
            last_layer3_sum = layer3_inner + last_layer3
            results.insert(0, self.layer_blocks[2](last_layer3_sum))
            # layer3+layer2
            layer2_inner = self.inner_blocks[1](x[1])
            layer2_shape = layer2_inner.shape[-2:]
            layer3_layer2 = F.interpolate(last_layer3_sum, size=layer2_shape, mode="nearest")
            layer3_layer2_sum = layer2_inner + layer3_layer2
            results.insert(0, self.layer_blocks[1](layer3_layer2_sum))
            # layer2+layer1
            layer1_inner = self.inner_blocks[0](x[0])
            layer1_shape = layer1_inner.shape[-2:]
            layer2_layer1 = F.interpolate(layer3_layer2_sum, size=layer1_shape, mode="nearest")
            layer2_layer1_sum = layer1_inner + layer2_layer1
            results.insert(0, self.layer_blocks[0](layer2_layer1_sum))
            # results 存储着FPN后特征图从大到小的 key:0,1,2,3,4. value(H W shape):[1/4, 1/8, 1/16, 1/32, 1/64]
            if self.extra_blocks:
                results, names = self.extra_blocks(results, names)
            # out: key: 0,1,2,3,pool
            out = OrderedDict([(k, v) for k, v in zip(names, results)])

            return out

    class Backbone_FPN(nn.Module):
        def __init__(self, backbone, return_layers, in_channels_list, out_channels):
            super(Backbone_FPN, self).__init__()

            maxpool_block = MaxPool_Block()
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
            self.fpn = FPN(
                backbone_output_channels_list=in_channels_list,
                out_channels=out_channels,
                maxpool_blocks=maxpool_block,
            )

            self.out_channels = out_channels

        def forward(self, x):
            x = self.body(x)
            x = self.fpn(x)

            return x

    def ResNet50_FPN_backbone():
        resnet_backbone = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=4)
        return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        in_channels_list = [256, 512, 1024, 2048]
        out_channels = 256

        return Backbone_FPN(resnet_backbone, return_layers, in_channels_list, out_channels)

    if __name__ == '__main__':
        alexnet = ResNet50_FPN_backbone()
        print(alexnet)

    def conv3x3(in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

    class nECABasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
            super(nECABasicBlock, self).__init__()
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(planes, planes, 1)
            self.bn2 = nn.BatchNorm2d(planes)
            self.eca = eca_layer(planes, k_size)
            self.nam = NAM(planes)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.eca(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

            return out

    class nECABottleneck(nn.Module):
        expansion = 4

        def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
            super(nECABottleneck, self).__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * 4)
            self.relu = nn.ReLU(inplace=True)
            self.eca = eca_layer(planes * 4, k_size)
            self.nam = NAM(planes * 4)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)
            out = self.eca(out)
            out = self.nam(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

            return out

    class ResNet(nn.Module):

        def __init__(self, block, layers, num_classes=4, k_size=[3, 3, 3, 3]):
            self.inplanes = 64
            super(ResNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0], int(k_size[0]))
            self.layer2 = self._make_layer(block, 128, layers[1], int(k_size[1]), stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], int(k_size[2]), stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], int(k_size[3]), stride=2)
            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        def _make_layer(self, block, planes, blocks, k_size, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample, k_size))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, k_size=k_size))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x

    def eca_resnet18(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False):
        """Constructs a ResNet-18 model.
        Args:
            k_size: Adaptive selection of kernel size
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            num_classes:The classes of classification
        """
        model = ResNet(nECABasicBlock, [2, 2, 2, 2], num_classes=num_classes, k_size=k_size)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        return model

    def eca_resnet34(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False):
        """Constructs a ResNet-34 model.
        Args:
            k_size: Adaptive selection of kernel size
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            num_classes:The classes of classification
        """
        model = ResNet(nECABasicBlock, [3, 4, 6, 3], num_classes=num_classes, k_size=k_size)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        return model

    def eca_resnet50(k_size=[3, 3, 3, 3], num_classes=4, pretrained=False):
        """Constructs a ResNet-50 model.
        Args:
            k_size: Adaptive selection of kernel size
            num_classes:The classes of classification
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        print("Constructing eca_resnet50......")
        model = ResNet(nECABottleneck, [3, 4, 6, 3], num_classes=num_classes, k_size=k_size)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        return model

    def eca_resnet101(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False):
        """Constructs a ResNet-101 model.
        Args:
            k_size: Adaptive selection of kernel size
            num_classes:The classes of classification
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = ResNet(nECABottleneck, [3, 4, 23, 3], num_classes=num_classes, k_size=k_size)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        return model

    def eca_resnet152(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False):
        """Constructs a ResNet-152 model.
        Args:
            k_size: Adaptive selection of kernel size
            num_classes:The classes of classification
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = ResNet(nECABottleneck, [3, 8, 36, 3], num_classes=num_classes, k_size=k_size)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        return model

    resnet = eca_resnet50()
    x = torch.randn(64, 3, 224, 224)
    X = resnet(x)
    print(X.shape)

    class Block(nn.Module):
        r""" ConvNeXt Block. There are two equivalent implementations:
        (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
        (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
        We use (2) as we find it slightly faster in PyTorch

        Args:
            dim (int): Number of input channels.
            drop_path (float): Stochastic depth rate. Default: 0.0
            layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        """

        def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
            super().__init__()
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
            self.norm = LayerNorm(dim, eps=1e-6)
            self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
            self.act = nn.GELU()
            self.pwconv2 = nn.Linear(4 * dim, dim)
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        def forward(self, x):
            input = x
            x = self.dwconv(x)
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
            x = self.norm(x)
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.pwconv2(x)
            if self.gamma is not None:
                x = self.gamma * x
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

            x = input + self.drop_path(x)
            return x

    class ConvNeXt(nn.Module):
        r""" ConvNeXt
            A PyTorch impl of : `A ConvNet for the 2020s`  -
              https://arxiv.org/pdf/2201.03545.pdf

        Args:
            in_chans (int): Number of input image channels. Default: 3
            num_classes (int): Number of classes for classification head. Default: 1000
            depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
            dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
            drop_path_rate (float): Stochastic depth rate. Default: 0.
            layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
            head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        """

        def __init__(self, in_chans=3, num_classes=4,
                     depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                     layer_scale_init_value=1e-6, head_init_scale=1.,
                     ):
            super().__init__()

            self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
            stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            )
            self.downsample_layers.append(stem)
            for i in range(3):
                downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
            dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
            cur = 0
            for i in range(4):
                stage = nn.Sequential(
                    *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                            layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )
                self.stages.append(stage)
                cur += depths[i]

            self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
            self.head = nn.Linear(dims[-1], num_classes)

            self.apply(self._init_weights)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

        def _init_weights(self, m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                nn.init.constant_(m.bias, 0)

        def forward_features(self, x):
            for i in range(4):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
            return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

        def forward(self, x):
            x = self.forward_features(x)
            x = self.head(x)
            return x

    class LayerNorm(nn.Module):
        r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
        The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
        shape (batch_size, height, width, channels) while channels_first corresponds to inputs
        with shape (batch_size, channels, height, width).
        """

        def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
            self.eps = eps
            self.data_format = data_format
            if self.data_format not in ["channels_last", "channels_first"]:
                raise NotImplementedError
            self.normalized_shape = (normalized_shape,)

        def forward(self, x):
            if self.data_format == "channels_last":
                return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            elif self.data_format == "channels_first":
                u = x.mean(1, keepdim=True)
                s = (x - u).pow(2).mean(1, keepdim=True)
                x = (x - u) / torch.sqrt(s + self.eps)
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
                return x

    model_urls = {
        "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
        "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
        "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
        "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
        "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
        "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
        "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
        "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
        "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
    }

    @register_model
    def convnext_tiny(pretrained=False, in_22k=False, **kwargs):
        model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
        if pretrained:
            url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
            model.load_state_dict(checkpoint["model"])
        return model

    @register_model
    def convnext_small(pretrained=False, in_22k=False, **kwargs):
        model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
        if pretrained:
            url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
        return model

    @register_model
    def convnext_base(pretrained=False, in_22k=False, **kwargs):
        model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
        if pretrained:
            url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
        return model

    @register_model
    def convnext_large(pretrained=False, in_22k=False, **kwargs):
        model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
        if pretrained:
            url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
        return model

    @register_model
    def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
        model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
        if pretrained:
            assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
            url = model_urls['convnext_xlarge_22k']
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
        return model

    rt = convnext_small()
    x = torch.randn(64, 3, 224, 224)
    X = rt(x)
    print(X.shape)

    VGGnet = convnext_small()  # 这里的7要修改成自己数据集的种类

    RN = eca_resnet50()

    alexnet1 = eca_resnet50()

    mlps = [RN.to(device), alexnet1.to(device), VGGnet.to(device)]  # 建立一个数组，将三个模型放入

    epoch = 280  # 训练轮数
    LR = 0.0001  # 学习率，我这里对于三个模型设置的是一样的学习率，事实上根据模型的不同设置成不一样的效果最好
    a = [{"params": mlp.parameters()} for mlp in mlps]  # 依次读取三个模型的权重
    optimizer = torch.optim.Adam(a, lr=LR)  # 建立优化器
    loss_function = FocalLoss()  # 构建损失函数

    train_loss_all = [[], [], []]
    train_accur_all = [[], [], []]
    ronghe_train_loss = []  # 融合模型训练集的损失
    ronghe_train_accuracy = []  # 融合模型训练集的准确率

    test_loss_all = [[], [], []]
    test_accur_all = [[], [], []]

    ronghe_test_loss = []  # 融合模型测试集的损失
    ronghe_test_accuracy = []  # 融合模型测试集的准确

    for i in range(epoch):  # 遍历开始进行训练
        train_loss = [0, 0, 0]  # 因为三个模型，初始化三个0存放模型的结果

        train_accuracy = [0.0, 0.0, 0.0]  # 同上初始化三个0，存放模型的准确率
        for mlp in range(len(mlps)):
            mlps[mlp].train()  # 遍历三个模型进行训练
        train_bar = tqdm(traindata)  # 构建进度条，训练的时候有个进度条显示

        pre1 = []  # 融合模型的损失
        vote1_correct = 0  # 融合模型的准确率
        for step, data in enumerate(train_bar):  # 遍历训练集

            img, target = data

            length = img.size(0)

            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            for mlp in range(len(mlps)):  # 对三个模型依次进行训练
                mlps[mlp].train()
                outputs = mlps[mlp](img)
                loss1 = loss_function(outputs, target)  # 求损失
                outputs = torch.argmax(outputs, 1)

                loss1.backward()  # 反向传播
                train_loss[mlp] += abs(loss1.item()) * img.size(0)
                accuracy = torch.sum(outputs == target)

                pre_num1 = outputs.cpu().numpy()
                # print(pre_num1.shape)
                train_accuracy[mlp] = train_accuracy[mlp] + accuracy

                pre1.append(pre_num1)

            arr1 = np.array(pre1)

            pre1.clear()  # 将pre进行清空

            result1 = [Counter(arr1[:, i]).most_common(1)[0][0] for i in
                       range(length)]  # 对于每张图片，统计三个模型其中，预测的那种情况最多，就取最多的值为融合模型预测的结果，即为投票
            # 投票的意思，因为是三个模型，取结果最多的
            vote1_correct += (result1 == target.cpu().numpy()).sum()

            optimizer.step()  # 更新梯度

        losshe = 0
        for mlp in range(len(mlps)):
            print("epoch：" + str(i + 1), "模型" + str(mlp) + "的损失和准确率为：",
                  "train-Loss：{} , train-accuracy：{}".format(train_loss[mlp] / train_size,
                                                             train_accuracy[mlp] / train_size))
            train_loss_all[mlp].append(train_loss[mlp] / train_size)
            train_accur_all[mlp].append(train_accuracy[mlp].double().item() / train_size)
            losshe += train_loss[mlp] / train_size
        losshe /= 3
        print("epoch: " + str(i + 1) + "集成模型训练集的正确率" + str(vote1_correct / train_size))
        print("epoch: " + str(i + 1) + "集成模型训练集的损失" + str(losshe))
        ronghe_train_loss.append(losshe)
        ronghe_train_accuracy.append(vote1_correct / train_size)

        test_loss = [0, 0, 0]
        test_accuracy = [0.0, 0.0, 0.0]

        for mlp in range(len(mlps)):
            mlps[mlp].eval()
        with torch.no_grad():
            pre = []
            vote_correct = 0
            test_bar = tqdm(testdata)
            vote_correct = 0
            for data in test_bar:

                length1 = 0
                img, target = data
                length1 = img.size(0)

                img, target = img.to(device), target.to(device)

                for mlp in range(len(mlps)):
                    outputs = mlps[mlp](img)

                    loss2 = loss_function(outputs, target)
                    outputs = torch.argmax(outputs, 1)

                    test_loss[mlp] += abs(loss2.item()) * img.size(0)

                    accuracy = torch.sum(outputs == target)
                    pre_num = outputs.cpu().numpy()

                    test_accuracy[mlp] += accuracy

                    pre.append(pre_num)
                arr = np.array(pre)
                pre.clear()  # 将pre进行清空
                result = [Counter(arr[:, i]).most_common(1)[0][0] for i in
                          range(length1)]  # 对于每张图片，统计三个模型其中，预测的那种情况最多，就取最多的值为融合模型预测的结果，
                vote_correct += (result == target.cpu().numpy()).sum()
        losshe1 = 0
        for mlp in range(len(mlps)):
            print("epoch：" + str(i + 1), "模型" + str(mlp) + "的损失和准确率为：",
                  "test-Loss：{} , test-accuracy：{}".format(test_loss[mlp] / test_size, test_accuracy[mlp] / test_size))
            test_loss_all[mlp].append(test_loss[mlp] / test_size)
            test_accur_all[mlp].append(test_accuracy[mlp].double().item() / test_size)
            losshe1 += test_loss[mlp] / test_size
        losshe1 /= 3
        print("epoch: " + str(i + 1) + "集成模型测试集的正确率" + str(vote_correct / test_size))
        print("epoch: " + str(i + 1) + "集成模型测试集的损失" + str(losshe1))
        ronghe_test_loss.append(losshe1)
        ronghe_test_accuracy.append(vote_correct / test_size)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    # for mlp in range(len(mlps)):
    plt.plot(range(epoch), ronghe_train_loss,
             "ro-", label="Train loss")
    plt.plot(range(epoch), ronghe_test_loss,
             "bs-", label="test loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(range(epoch), ronghe_train_accuracy,
             "ro-", label="Train accur")
    plt.plot(range(epoch), ronghe_test_accuracy,
             "bs-", label="test accur")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()

    torch.save(alexnet1.state_dict(), "alexnet.pth")
    torch.save(RN.state_dict(), "lenet1.pth")
    torch.save(VGGnet.state_dict(), "VGGnet.pth")

    print("模型已保存")


if __name__ == '__main__':
    main()
