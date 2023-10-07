import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

class InceptionA(nn.Module):

    def __init__(self, input_channels, pool_features):
        super().__init__()
        self.branch1x1 = BasicConv2d(input_channels, 32, kernel_size=1)

        self.branch5x5 = nn.Sequential(
            ConvBN(input_channels, 32, kernel_size=1),
            ConvBN(32, 48, kernel_size=3)
        )

        self.branch3x3 = nn.Sequential(
            ConvBN(input_channels, 32, kernel_size=1),
            ConvBN(32, 48, kernel_size=3),
            ConvBN(48, 64, kernel_size=5)
        )

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=1, stride=1),
            ConvBN(input_channels, pool_features, kernel_size=1)
        )

    def forward(self, x):
        #
        #x -> 1x1(same)
        branch1x1 = self.branch1x1(x)

        #x -> 1x1 -> 5x5(same)
        branch5x5 = self.branch5x5(x)
        #branch5x5 = self.branch5x5_2(branch5x5)

        #x -> 1x1 -> 3x3 -> 3x3(same)
        branch3x3 = self.branch3x3(x)

        #x -> pool -> 1x1(same)
        branchpool = self.branchpool(x)  #torch.Size([48, 32, 22, 116])

        outputs = [branch1x1, branch5x5, branch3x3, branchpool]

        return torch.cat(outputs, 1)

#downsample
#Factorization into smaller convolutions
class InceptionB(nn.Module):

    def __init__(self, input_channels):
        super().__init__()

        self.branch3x3 = BasicConv2d(input_channels, 64, kernel_size=3, stride=2)

        self.branch3x3stack = nn.Sequential(
            BasicConv2d(input_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=2)
        )

        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):

        #x - > 3x3(downsample)
        branch3x3 = self.branch3x3(x)

        #x -> 3x3 -> 3x3(downsample)
        branch3x3stack = self.branch3x3stack(x)

        #x -> avgpool(downsample)
        branchpool = self.branchpool(x)

        #"""We can use two parallel stride 2 blocks: P and C. P is a pooling
        #layer (either average or maximum pooling) the activation, both of
        #them are stride 2 the filter banks of which are concatenated as in
        #figure 10."""
        outputs = [branch3x3, branch3x3stack, branchpool]

        return torch.cat(outputs, 1)


#Factorizing Convolutions with Large Filter Size
class InceptionC(nn.Module):
    def __init__(self, input_channels, channels_7x7):
        super().__init__()
        self.branch1x1 = BasicConv2d(input_channels, 64, kernel_size=1)

        c7 = channels_7x7

        #In theory, we could go even further and argue that one can replace any n × n
        #convolution by a 1 × n convolution followed by a n × 1 convolution and the
        #computational cost saving increases dramatically as n grows (see figure 6).
        self.branch7x7 = nn.Sequential(
            BasicConv2d(input_channels, c7, kernel_size=1),
            BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(c7, 64, kernel_size=(1, 7), padding=(0, 3))
        )

        self.branch7x7stack = nn.Sequential(
            BasicConv2d(input_channels, c7, kernel_size=1),
            BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(c7, 64, kernel_size=(1, 7), padding=(0, 3))
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, 64, kernel_size=1),
        )

    def forward(self, x):

        #x -> 1x1(same)
        branch1x1 = self.branch1x1(x)

        #x -> 1layer 1*7 and 7*1 (same)
        branch7x7 = self.branch7x7(x)

        #x-> 2layer 1*7 and 7*1(same)
        branch7x7stack = self.branch7x7stack(x)

        #x-> avgpool (same)
        branchpool = self.branch_pool(x)

        outputs = [branch1x1, branch7x7, branch7x7stack, branchpool]

        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
    
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),)

class InceptionV3(nn.Module):

    def __init__(self, num_classes=100):
        super().__init__()
        # self.Conv2d_1a_3x3 = BasicConv2d(16, 32, kernel_size=3)
        # self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        # self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3)
        self.Conv2d_1a_3x3 = ConvBN(16, 32, kernel_size=3)
        self.Conv2d_2a_3x3 = ConvBN(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = ConvBN(32, 64, kernel_size=3)
        #self.Conv2d_3b_1x1 = BasicConv2d(64, 128, kernel_size=1)
        #self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3) #torch.Size([48, 192, 20, 114])

        #naive inception module
        self.Mixed_5b = InceptionA(64, pool_features=32)
        self.Mixed_5c = InceptionA(176, pool_features=16)
        self.Mixed_5d = InceptionA(160, pool_features=8)

        # #downsample
        # self.Mixed_6a = InceptionB(152) #torch.Size([10, 312, 10, 57])
        self.Mixed_6a = nn.Conv2d(in_channels=152, out_channels=312, kernel_size=(5, 6), stride=(2, 2), padding=(1, 1))
        
        self.Mixed_6b = InceptionC(312, channels_7x7=128)
        self.Mixed_6c = InceptionC(256, channels_7x7=160)
        self.Mixed_6d = InceptionC(256, channels_7x7=160)
        # self.Mixed_6e = InceptionC(768, channels_7x7=192)

        # #downsample
        # self.Mixed_7a = InceptionD(768)

        # self.Mixed_7b = InceptionE(1280)
        # self.Mixed_7c = InceptionE(2048)

        #6*6 feature size
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.dropout = nn.Dropout2d() 
        #self.linear = nn.Linear(2048, num_classes)
        self.conv = nn.Conv2d(256,128,3)
        self.conv2 = ConvBN(128,64,3)
        # self.conv3 = ConvBN(64,32,3)
        # self.conv4 = ConvBN(32,16,3)

    def forward(self, x):
        
        #32 -> 30
        x = self.Conv2d_1a_3x3(x) 
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x) #torch.Size([48, 64, 22, 116])
        #x = self.Conv2d_3b_1x1(x)
        #x = self.Conv2d_4a_3x3(x)  #torch.Size([2, 192, 18, 112])

        #30 -> 30
        
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)   #torch.Size([2, 152,22,116])
        #

        
        #30 -> 14
        #Efficient Grid Size Reduction to avoid representation
        #bottleneck
        x = self.Mixed_6a(x) #torch.Size([10, 312, 10, 57])
        
        #import pdb;pdb.set_trace()

        # x = self.conv3(x)
        # x = self.conv4(x)
        #14 -> 14
        #"""In practice, we have found that employing this factorization does not
        #work well on early layers, but it gives very good results on medium
        #grid-sizes (On m × m feature maps, where m ranges between 12 and 20).
        #On that level, very good results can be achieved by using 1 × 7 convolutions
        #followed by 7 × 1 convolutions."""
        x = self.Mixed_6b(x) #torch.Size([24, 768, 10, 57])
        x = self.Mixed_6c(x) #
        x = self.Mixed_6d(x) #torch.Size([24, 256, 10, 57])
        x = self.conv(x)
        x = self.conv2(x)
        #x = self.Mixed_6e(x)

        #14 -> 6
        #Efficient Grid Size Reduction
        # x = self.Mixed_7a(x)

        #6 -> 6
        #We are using this solution only on the coarsest grid,
        #since that is the place where producing high dimensional
        #sparse representation is the most critical as the ratio of
        #local processing (by 1 × 1 convolutions) is increased compared
        #to the spatial aggregation."""
        # x = self.Mixed_7b(x)
        # x = self.Mixed_7c(x) #torch.Size([2, 2048, 3, 27])

        #6 -> 1
        # x = self.avgpool(x)  
        #x = self.dropout(x)  #torch.Size([48, 288, 20, 114])
        #import pdb;pdb.set_trace()
        #x = x.view(x.size(0), -1)
        #x = self.linear(x)
        return x


def inceptionv3():
    return InceptionV3()

class Joints(nn.Module):
    def __init__(self):
        super(Joints, self).__init__()
        #self.conv1 =  
        self.joints= nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )
        self.linear1 = nn.Linear(64, 21*2)
        self.linear2 = nn.Linear(64, 21)
    def forward(self, x):
        
        #x = self.linear3(x) #
        x = self.joints(x) #
        twod = torch.sigmoid(self.linear1(x)) #torch.Size([100, 42])
        threed = torch.sigmoid(self.linear2(x))
        return twod,threed

class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x   

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(2048, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        # self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=5,stride=1, padding=0, dilation=1 )
        self.conv1 = nn.Conv2d(64, 32, kernel_size=5,stride=1, padding=0, dilation=1 )
        self.conv3 = nn.Conv2d(16, 4, kernel_size=5,stride=1, padding=0, dilation=1 )
        self.conv4 = nn.Conv2d(4, 2, kernel_size=5,stride=1, padding=0, dilation=1 )
        self.conv5 = nn.Conv2d(2, 1, kernel_size=3,stride=1, padding=0, dilation=1 )
        self.score_fr = nn.Conv2d(4096, 2, 1)
        self.upscore = nn.ConvTranspose2d(64, 64, 8, stride=4,
                                          bias=False)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        #
        x = self.linear(z)

        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=4)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)  #torch.Size([1, 64, 32, 32])
        x = self.layer1(x) ##torch.Size([1, 64, 32, 32]) 
        mask = self.upscore(x)
        mask = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(mask)))))
        mask = torch.sigmoid(mask)
        #mask = mask.view(x.size(0),1,114,114) #
        return mask

class CSI_AutoEncoder(nn.Module):
    def __init__(self):
        super(CSI_AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = 6, out_channels = 3, kernel_size = 1, stride = 1, padding = 1, groups = 3, bias=False), # [b,3,20,114]
            nn.Conv2d(3,16,kernel_size = 3, stride = 1, padding = 1, groups = 1, bias = True), # [b,16,22,112] [b,16,20,114]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),)
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1), #[b,16,11,56] [b,16,10,57]

        #     nn.Conv2d(16,32,kernel_size = 3, stride = 1, padding = 1, groups = 1, bias = True), # [b,32,7,24] [b,32,10,57]
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(negative_slope=0.3, inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1), #[b,32,7,24] [b,16,5,29]

        #     nn.Conv2d(32,64,kernel_size = 3, stride = 1, padding = 1, groups = 1, bias = True), # [b,64,9,6]
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(negative_slope=0.3, inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1), #[b,64,5,3]

        #     nn.Conv2d(64,128,kernel_size = 3, stride = 1, padding = 1, groups = 1, bias = True), # [b,64,9,6]
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(negative_slope=0.3, inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1), #[b,64,5,3]

        #     nn.Sigmoid(),
        # )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (4,3), stride=2, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 6, 3, stride=1, padding = 1),
            nn.Tanh(),
        )
        self.fc = nn.Sequential(
            nn.Linear(6*20*114, 12544),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout2d() 
        self.fj = nn.Sequential(
            nn.Linear(64*8*55, 2048),
            nn.ReLU(),
        )
        self.joint = Joints()
        self.decmask = ResNet18Dec()
        self.incep = InceptionV3()
        #self.convv = ConvBN()

    def forward(self, x):
        # 
        
        encoded = self.encoder(x) # [b, 128,2,8]  #[48, 16, 22, 116]
        
        x = self.incep(encoded)
        
        # joints decode
        #import pdb;pdb.set_trace()
        joints_encode = self.fj(x.view(x.size(0),-1))
        twod,threed = self.joint(joints_encode)
        # mask decode
        # decoded = self.decoder(encoded)
        # decoded = decoded.view(decoded.size(0), -1)
        # decoded_mask = self.fc(decoded)
        # decoded_mask = decoded_mask.view(decoded_mask.size(0), 112,112)

        mask_encode = self.fj(x.view(x.size(0),-1))
        decoded_mask = self.decmask(mask_encode)

        return encoded, decoded_mask, twod,threed
   