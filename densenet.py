import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class DenseNet3(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0, numgroups = 1):
        super(DenseNet3, self).__init__()
        self.numgroups = numgroups
        in_planes = 2 * growth_rate #24
        n = (depth - 4) / 3 #12
        if bottleneck == True:
            n = n / 2 #6
            block = BottleneckBlock
        else:
            block = BasicBlock
        # 1st conv before any dense block

        if num_classes == 10 or num_classes == 100: #CIFAR-10/100
            for i in range(numgroups):
                exec( 'self.conv1_{idx} = nn.Conv2d(3, in_planes, kernel_size=3, stride=1, \
                               padding=1, bias=False)'.format(idx=i) )
                exec( 'self.pool1_{idx} = nn.MaxPool2d(1, stride=1, padding=0)'.format(idx=i) )
        elif num_classes == 1000: #ImageNet
            for i in range(numgroups):
                exec( 'self.conv1_{idx} = nn.Conv2d(3, in_planes, kernel_size=7, stride=2, \
                               padding=3, bias=False)'.format(idx=i) )
                exec( 'self.pool1_{idx} = nn.MaxPool2d(3, stride=2, padding=1)'.format(idx=i) )



        # 1st block
        for i in range(numgroups):
            exec('self.block1_{idx} = DenseBlock(n, in_planes, growth_rate, block, dropRate)'.format(idx=i))
        in_planes = int(in_planes+n*growth_rate)
        for i in range(numgroups):
            exec('self.trans1_{idx} = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)'.format(idx=i))
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        for i in range(numgroups):
            exec('self.block2_{idx} = DenseBlock(n, in_planes, growth_rate, block, dropRate)'\
                 .format(idx=i))
        in_planes = int(in_planes+n*growth_rate)
        for i in range(numgroups):
            exec('self.trans2_{idx} = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)),\
                dropRate=dropRate)'.format(idx=i))
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        for i in range(numgroups):
            exec('self.block3_{idx} = DenseBlock(n, in_planes, growth_rate, block, dropRate)'.format(idx=i))
        in_planes = int(in_planes+n*growth_rate)
        # global average pooling and classifier
        for i in range(numgroups):
            exec('self.bn1_{idx} = nn.BatchNorm2d(in_planes)'.format(idx=i))
        for i in range(numgroups):
            exec('self.relu_{idx} = nn.ReLU(inplace=True)'.format(idx=i))
        self.fc = nn.Linear(in_planes*numgroups, num_classes)
        self.in_planes = in_planes*numgroups

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        catlist = []
        for i in range(self.numgroups):
            exec('out_{idx} = self.conv1_{idx}(x)'.format(idx=i))
            exec('out_{idx} = self.pool1_{idx}(out_{idx})'.format(idx=i))
            exec('out_{idx} = self.block1_{idx}(out_{idx})'.format(idx=i))
            exec('out_{idx} = self.trans1_{idx}(out_{idx})'.format(idx=i))
            exec('out_{idx} = self.block2_{idx}(out_{idx})'.format(idx=i))
            exec('out_{idx} = self.trans2_{idx}(out_{idx})'.format(idx=i))
            exec('out_{idx} = self.block3_{idx}(out_{idx})'.format(idx=i))
            exec('out_{idx} = self.bn1_{idx}(out_{idx})'.format(idx=i))
            exec('out_{idx} = self.relu_{idx}(out_{idx})'.format(idx=i))
            exec('out_{idx} = F.avg_pool2d(out_{idx}, 8)'.format(idx=i))
            exec('out_{idx} = out_{idx}.view(-1, self.in_planes/self.numgroups)'.format(idx=i))
            exec('catlist.append(out_{idx})'.format(idx=i))
        out = torch.cat(catlist, 1)
        #out = catlist[0]
        return self.fc(out)
