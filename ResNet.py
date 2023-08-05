import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleNeck(nn.Module):
    expansion = 4 # expansion can strongthen the ability of extracting feature and decrease compute amount
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1) -> None:
        super(BottleNeck, self).__init__()
        self.identity_downsample = identity_downsample
        self.bottleneck_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        
    def forward(self, x):
        x_identity = x.clone()
        out = self.bottleneck_layer(x)
        if self.identity_downsample is not None:
            x_identity = self.identity_downsample(x_identity)
        out = out + x_identity
        out = F.relu(out)
        return out
# identity_downsample is used when input channels is not equal to output channels
class ResBlock(nn.Module):
    # expansion is used to control output channel of Resblock or bottleneck
    # put in here cause if ResBlock has not init, I can call ResBlock to get expansion directly 
    expansion = 1
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1) -> None:
        super(ResBlock, self).__init__()
        self.identity_downsample = identity_downsample
        self.res_funcion = nn.Sequential(
            # if stride = 1, then input 2d shape is the same as output 2d shape 
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False), # if kernel size is single number, that means the kernel is square
            nn.BatchNorm2d(out_channels), # add batch norm after conv, can avoid cov. shift
            nn.ReLU(), 
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()

    """
    F.relu() is a function, can directly use in calculate
    nn.ReLU() is a class, need to be called to a obj, then we could use
    """
    def forward(self, x):
        x_identity = x.clone()
        print(f'Origin x\'s shape : {x.shape}')
        out = self.res_funcion(x)
        print(f'After res_block x\'s shape : {x.shape}')
        if self.identity_downsample is not None:
            x_identity = self.identity_downsample(x_identity)
        print(f'After downsampling x\'s shape : {x.shape}')
        out += x_identity
        out = F.relu(out) 
        return out
class ResNet(nn.Module):
    def __init__(self, res_block, layer_list, input_channels, num_classes=2) -> None:
        super(ResNet, self).__init__()
        # in_channel is relative to ResBlock
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # each layer is a ResBlock
        self.conv2 = self._make_layer(res_block, layer_list[0], channels=64, stride=1)
        self.conv3 = self._make_layer(res_block, layer_list[1], channels=128, stride=2)
        self.conv4 = self._make_layer(res_block, layer_list[2], channels=256, stride=2)
        self.conv5 = self._make_layer(res_block, layer_list[3], channels=512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # (1, 1) is kernel size, this can fit differnet size of input 
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.avgpool(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out
    def _make_layer(self, res_block, blocks, channels, stride):
        identity_downsample = None
        layers = []
        # Each ResNet block does identity downsample, it's meaning is keep the output shape same as input shape
        # if stride != 1, it means that down sampling, 
        # and if self.in_channels != channels*res_block.expansion means channels num is not equal, cannot add directly, so need conv
        if stride != 1 or self.in_channels != channels*res_block.expansion:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels*res_block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(channels*res_block.expansion)
            )
        # only first res_block need down sample and stride = 2, 
        layers.append(res_block(self.in_channels, channels, identity_downsample, stride))
        self.in_channels = channels*res_block.expansion
        for i in range(blocks-1):
            layers.append(res_block(self.in_channels, channels)) 
        return nn.Sequential(*layers) # * can parse list into argument, pass to nn.Sequential

def ResNet18(input_channels, num_classes):
    return ResNet(ResBlock, [2, 2, 2, 2], input_channels, num_classes)
        
def ResNet50(input_channels, num_classes):
    return ResNet(BottleNeck, [3, 4, 6, 3], input_channels, num_classes)
        
def ResNet152(input_channels, num_classes):
    return ResNet(BottleNeck, [3, 8, 36, 3], input_channels, num_classes)
        