import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)

        # if stride == 2:
        #     stride = (1, 2, 2)

        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        # print('stride {}'.format(stride))
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Freehand(nn.Module):
    def __init__(self, block, layers, shortcut_type='B', cardinality=32, num_classes1=21, num_classes2=28, gpus=0):
        self.inplanes = 64
        self.device = torch.device("cuda:{}".format(gpus))
        super(Freehand, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type, cardinality)
        self.layer2 = self._make_layer(block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(block, 1024, layers[3], shortcut_type, cardinality, stride=2)

        self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.conv2 = nn.Conv3d(in_channels=2048, out_channels=128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))

        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(cardinality * 16 * block.expansion, num_classes1)
        self.fc2 = nn.Linear(cardinality * 16 * block.expansion, num_classes2)
        self.dropout1 = nn.Dropout(p=0.25, inplace=False)

        self.attention = nn.Sequential(
            nn.BatchNorm3d(2048),
            nn.Conv3d(in_channels=2048, out_channels=1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0),
            nn.ReLU(),
            nn.Conv3d(in_channels=1024, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def encode(self, x):
        h1 = self.relu(x)
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, us_imgs, opf_imgs, neighbour_slice):
        n, c, m, h, w = us_imgs.shape
        us_mix = torch.empty((n, c + 1, m - 1, h, w))
        for i in range(len(us_imgs)):
            us_pack = us_imgs[i][0]
            us_pack_new = torch.empty((c + 1, m - 1, h, w))
            imgs1 = us_pack[:-1]
            imgs2 = us_pack[1:]
            idx = 0
            for img1, img2 in zip(imgs1, imgs2):
                us_pack_new[0][idx] = img1
                us_pack_new[1][idx] = img2
                idx += 1
            us_mix[i] = us_pack_new
        x = torch.cat([us_mix.to(self.device), opf_imgs], 1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        at_map = self.attention(x)
        x = x * at_map
        mp = self.relu(x)

        x = self.avgpool(mp)
        x = x.view(x.size(0), -1)

        t = self.fc1(x[:, :1024])
        r = self.fc2(x[:, 1024:])

        batch = x.shape[0]
        t = t.reshape(batch, (neighbour_slice - 1), 3)
        r = r.reshape(batch, (neighbour_slice - 1), 3)
        return t, r

def network(args):
    model = Freehand(ResNeXtBottleneck, [3, 4, 6, 3])
    model.conv1 = nn.Conv3d(in_channels=5, out_channels=64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
    num_ftrs = model.fc1.in_features
    model.fc1 = nn.Linear(num_ftrs, (args.neighbour_slice - 1) * 3)
    num_ftrs = model.fc2.in_features
    model.fc2 = nn.Linear(num_ftrs, (args.neighbour_slice - 1) * 3)
    return model