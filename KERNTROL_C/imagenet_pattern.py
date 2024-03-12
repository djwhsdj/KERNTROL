from __future__ import print_function
import os
import argparse
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune
import torch.nn.init as init
import torchvision
from torchvision import datasets, transforms
import utils
from utils import *
# from tqdm import tqdm
import pickle
import logging
import random
# import pandas as pd
# from thop import profile 

import torch.backends.cudnn as cudnn

##### Settings #########################################################################                      3x3에서는 patdnn을 따라가도록?
parser = argparse.ArgumentParser(description='Pytorch PatDNN training')
parser.add_argument('--dir',        default='/Data',           help='dataset root')
parser.add_argument('--model',      default='ResNet',          help = 'select model : VGG9_Q, ResNet_Q, Wide_ResNet_Q')
parser.add_argument('--dataset',    default='imagenet',          help='select dataset')
parser.add_argument('--batchsize',  default=512, type=int,      help='set batch size')
parser.add_argument('--lr',         default=1e-4, type=float,   help='set learning rate') # 6e-5
parser.add_argument('--epoch',      default=80, type=int,      help='set epochs') # 60
parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--optimizer', default='SGD', type=str, help='[adam, SGD]')
parser.add_argument('--no-cuda',    default=False, action='store_true', help='disables CUDA training')
parser.add_argument('--u_th',    default=0, type=float, help='upper threshold of row utilization') # 0.25, 0.3, 0.4
parser.add_argument('--l_th',    default=0, type=float, help='lower threshold of row utilization') # 0.25, 0.3, 0.4
parser.add_argument('--GPU', type=int, default=2) 
parser.add_argument('--ab', type=int, default=32)
parser.add_argument('--wb', type=int, default=32)
parser.add_argument('--seed', type=int, default=1992)
parser.add_argument('--ar', type=int, default=0)
parser.add_argument('--ac', type=int, default=0)
parser.add_argument('--scale', type=int, default=4)
parser.add_argument('--original', type=int, default=1)
parser.add_argument('--entry', type=int, default=6, help = '6: 6-entry, 4: 4-entry')
parser.add_argument('--save', type=int, default=0, help = 'model save   0:no, 1:do')
parser.add_argument('--pretrain', type=int, default=1, help = 'pretrained model   0:no, 1:do')
args = parser.parse_args()
print(args)

GPU_NUM = args.GPU # GPU

m_bit = args.wb
real_ac = int(args.ac/args.wb * m_bit)


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= str(GPU_NUM)  # Set the GPU 2 to use
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
print ('Current cuda device ', device)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(args.seed)



logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BasicBlock__(nn.Module):
    expansion = 1

    def __init__(self, a_bit, w_bit, in_planes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock__, self).__init__()

        self.w_bit = w_bit
        self.a_bit = a_bit

        self.act1 = Activate(self.a_bit)
        self.act2 = Activate(self.a_bit)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = Conv2d_Q_(self.w_bit, in_planes, planes, kernel_size=kernel, padding=(1,1), stride=stride, bias=False)
        self.bn1 = SwitchBatchNorm2d(self.w_bit, planes)
        self.conv2 = Conv2d_Q_(self.w_bit, planes, planes, kernel_size=kernel, padding=padding, stride=1, bias=False) 
        self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)
    
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act2(out)

        return out
class ResNet__(nn.Module):

    def __init__(self, a_bit, w_bit, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet__, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.w_bit = w_bit
        self.a_bit = a_bit

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                # if isinstance(m, Bottleneck):
                #     nn.init.constant_(m.bn3.weight, 0)
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),)

        layers = []
        layers.append(block(self.a_bit, self.w_bit, self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.a_bit, self.w_bit, self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)
    
kernel = 3
padding = 1

if args.original == 0:
    name = 'Random' # 6 entry

elif args.original == 1:
    name = 'PatDNN'

elif args.original == 2:
    name = 'Pconv'


print('pretrained model uploading...')
directory_save = './save/ResNet/imagenet/original/2023/80/lr0.1000_wb8_ab32_ar8xac32.pt'
model = ResNet__(args.ab, args.wb, block=BasicBlock__, layers=[2,2,2,2], num_classes=1000).cuda()
model.load_state_dict(torch.load(directory_save), strict = False)

pattern_dic = []
entry = 3

pattern_dic_noprune = [[1,1,1,1,1,1,1,1,1]]

pruned_weight = 9 - args.entry
if args.original == 1 :
    unpruned_weight = args.entry - 1 # 가운데 제외시키려고
for nam, p in model.named_parameters():
    if nam.split('.')[-1] == "weight" and len(p.shape) == 4 and 'shortcut' not in nam and '0' not in nam and 'layer' in nam: # and '0' not in nam
        print(nam, p.size())
        if args.original == 0: # random
            candi_list = []
            ran_list = [0,1,2,3,4,5,6,7,8]
            one_list = [1,1,1,1,1,1,1,1,1]
            sample = random.sample(ran_list, pruned_weight)
            for sam in sample :
                one_list[sam] = 0
            mask = one_list

        elif args.original == 1: # patdnn
            tensor = torch.abs(torch.sum(p, (0,1)))
            tensor = tensor.cpu().detach().numpy()
            tensor[1][1] = -10000

            for i in range(unpruned_weight):
                tensor = np.where(tensor == np.max(tensor), -10000, tensor)
            
            tensor = np.where(tensor==-10000, 1, 0).astype(np.int32)
            tensor = tensor.reshape(1,-1)
            mask = tensor[0].tolist()

        pattern_dic.append(mask)





class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, a_bit, w_bit, in_planes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()

        self.w_bit = w_bit
        self.a_bit = a_bit

        self.act1 = Activate(self.a_bit)
        self.act2 = Activate(self.a_bit)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if args.original == 1 or args.original == 0:
            if in_planes != planes:
                self.conv1 = Conv2d_Q_(self.w_bit, in_planes, planes, kernel_size=kernel, padding=(1,1), stride=stride, bias=False)
                self.bn1 = SwitchBatchNorm2d(self.w_bit, planes)
                self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)
                if planes == 64:
                    pat1 = torch.tile(torch.tensor(pattern_dic[0]).reshape(3,3), (planes, planes, 1,1)).cuda()
                    count1 = counting(pattern_dic[0], 3, 3)
                    count2 = counting(pattern_dic[0], 4, 4)
                    count3 = counting(pattern_dic[0], 5, 5)
                    logger.info(pattern_dic[0])

                elif planes == 128:
                    pat1 = torch.tile(torch.tensor(pattern_dic[2]).reshape(3,3), (planes, planes, 1,1)).cuda()
                    count1 = counting(pattern_dic[2], 3, 3)
                    count2 = counting(pattern_dic[2], 4, 4)
                    count3 = counting(pattern_dic[2], 5, 5)
                    logger.info(pattern_dic[2])

                elif planes == 256:
                    pat1 = torch.tile(torch.tensor(pattern_dic[4]).reshape(3,3), (planes, planes, 1,1)).cuda()  
                    count1 = counting(pattern_dic[4], 3, 3)
                    count2 = counting(pattern_dic[4], 4, 4)
                    count3 = counting(pattern_dic[4], 5, 5)
                    logger.info(pattern_dic[4])
                else:
                    pat1 = torch.tile(torch.tensor(pattern_dic_noprune[0]).reshape(3,3), (planes, planes, 1,1)).cuda()  
                    count1 = counting(pattern_dic_noprune[0], 3, 3)
                    count2 = counting(pattern_dic_noprune[0], 3, 3)
                    count3 = counting(pattern_dic_noprune[0], 3, 3)
                    logger.info(pattern_dic_noprune[0])
                    
                logger.info('reduced rows: if pw==3: %d, pw==4: %d, pw==5: %d'%(count1*planes, count2*planes, count3*planes))

                self.conv1 = Conv2d_Q_(self.w_bit, in_planes, planes, kernel_size=(3,3), padding=(1,1), stride=stride, bias=False)
                self.bn1 = SwitchBatchNorm2d(self.w_bit, planes)
                self.conv2 = Conv2d_Q_mask(self.w_bit, planes, planes, kernel_size=3, pat=pat1, padding=padding, stride=1, bias=False) 
                self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)     
                    
            else: 
                logger.info('='*50)
                logger.info('in_planes: %d,  out_plaens: %d'%(in_planes, planes))
                if planes == 64:
                    pat1 = torch.tile(torch.tensor(pattern_dic[0]).reshape(3,3), (planes, in_planes, 1,1)).cuda()
                    count1 = counting(pattern_dic[0], 3, 3)
                    count2 = counting(pattern_dic[0], 4, 4)
                    count3 = counting(pattern_dic[0], 5, 5)
                    logger.info(pattern_dic[0])

                    pat2 = torch.tile(torch.tensor(pattern_dic[1]).reshape(3,3), (planes, in_planes, 1,1)).cuda()
                    count4 = counting(pattern_dic[1], 3, 3)
                    count5 = counting(pattern_dic[1], 4, 4)
                    count6 = counting(pattern_dic[1], 5, 5)
                    logger.info(pattern_dic[1])

                elif planes == 128:
                    pat1 = torch.tile(torch.tensor(pattern_dic[2]).reshape(3,3), (planes, in_planes, 1,1)).cuda()
                    count1 = counting(pattern_dic[2], 3, 3)
                    count2 = counting(pattern_dic[2], 4, 4)
                    count3 = counting(pattern_dic[2], 5, 5)
                    logger.info(pattern_dic[2])

                    pat2 = torch.tile(torch.tensor(pattern_dic[3]).reshape(3,3), (planes, in_planes, 1,1)).cuda()
                    count4 = counting(pattern_dic[3], 3, 3)
                    count5 = counting(pattern_dic[3], 4, 4)
                    count6 = counting(pattern_dic[3], 5, 5)
                    logger.info(pattern_dic[4])

                elif planes == 256:
                    pat1 = torch.tile(torch.tensor(pattern_dic[4]).reshape(3,3), (planes, in_planes, 1,1)).cuda()
                    count1 = counting(pattern_dic[4], 3, 3)
                    count2 = counting(pattern_dic[4], 4, 4)
                    count3 = counting(pattern_dic[4], 5, 5)
                    logger.info(pattern_dic[4])

                    pat2 = torch.tile(torch.tensor(pattern_dic[5]).reshape(3,3), (planes, in_planes, 1,1)).cuda()
                    count4 = counting(pattern_dic[5], 3, 3)
                    count5 = counting(pattern_dic[5], 4, 4)
                    count6 = counting(pattern_dic[5], 5, 5)
                    logger.info(pattern_dic[5])
                
                else:
                    pat1 = torch.tile(torch.tensor(pattern_dic_noprune[0]).reshape(3,3), (planes, planes, 1,1)).cuda()  
                    count1 = counting(pattern_dic_noprune[0], 3, 3)
                    count2 = counting(pattern_dic_noprune[0], 3, 3)
                    count3 = counting(pattern_dic_noprune[0], 3, 3)
                    logger.info(pattern_dic_noprune[0])

                    pat2 = torch.tile(torch.tensor(pattern_dic_noprune[0]).reshape(3,3), (planes, planes, 1,1)).cuda()  
                    count4 = counting(pattern_dic_noprune[0], 3, 3)
                    count5 = counting(pattern_dic_noprune[0], 3, 3)
                    count6 = counting(pattern_dic_noprune[0], 3, 3)
                    logger.info(pattern_dic_noprune[0])
                    
                logger.info('conv1 reduced rows: if pw==3: %d, pw==4: %d, pw==5: %d'%(count1*in_planes, count2*in_planes, count3*in_planes))
                logger.info('conv2 reduced rows: if pw==3: %d, pw==4: %d, pw==5: %d'%(count4*in_planes, count5*in_planes, count6*in_planes))

                self.bn1 = SwitchBatchNorm2d(self.w_bit, planes)
                self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)
                self.conv1 = Conv2d_Q_mask(self.w_bit, planes, planes, kernel_size=3, pat=pat1, padding=padding, stride=1, bias=False) 
                self.conv2 = Conv2d_Q_mask(self.w_bit, planes, planes, kernel_size=3, pat=pat2, padding=padding, stride=1, bias=False) 


        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act2(out)

        return out
    
class ResNet(nn.Module):

    def __init__(self, a_bit, w_bit, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.w_bit = w_bit
        self.a_bit = a_bit

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                # if isinstance(m, Bottleneck):
                #     nn.init.constant_(m.bn3.weight, 0)
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),)

        layers = []
        layers.append(block(self.a_bit, self.w_bit, self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.a_bit, self.w_bit, self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)




kernel = 3
padding = 1




directory = './log/%s/%s/%s/%s/%d'%(args.model, args.dataset, name, args.seed, args.entry)
if not os.path.isdir(directory):
    os.makedirs(directory)
file_name = directory + '/lr%.4f_wb%d_ab%d_lth%.2f_ar%dxac%d.log'%(args.lr, args.wb, args.ab, args.l_th, args.ar, args.ac)


file_handler = logging.FileHandler(file_name)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

args.workers = 16

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")



def eval(model, test_loader):
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img = img.to(device)
            out = model(img)
            pred = out.max(1)[1].detach().cpu().numpy()
            target = target.cpu().numpy()
            correct += (pred==target).sum()
            total += len(target)
    return correct / total




def train_model(model, train_loader, test_loader):
    logger.info("="*100)
    best_acc = -1

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[45,60,70], gamma=0.1) # 30,60,85,95,105 # 45,60,70


    print(f'TRAINING START!')
    for epoch in range(args.epoch):
        model.train()
        cnt = 0
        loss_sum = 0

        for i, (img, target) in enumerate(train_loader):
            cnt += 1
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        
        
        # scheduler.step()
        loss_sum = loss_sum / cnt
        model.eval()
        acc = eval(model, test_loader)
 

        print(f'Epochs : {epoch+1}, Accuracy : {acc}')
        logger.info("Epoch %d/%d, Acc=%.4f"%(epoch+1, args.epoch, acc))
        
        scheduler.step()

        if acc > best_acc :
            best_acc = acc
            print('Best accuracy is updated! at epoch %d/%d: %.4f '%(epoch+1, args.epoch, best_acc))
            logger.info('Best accuracy is : %.4f '%(best_acc))

    logger.info('Final best accuracy is : %.4f '%(best_acc))

    

def main():
    train_loader, test_loader = data_loader(args.dir, args.dataset, args.batchsize, args.workers)

    model = ResNet(args.ab, args.wb, block=BasicBlock, layers=[2,2,2,2], num_classes=1000).cuda()
    model.load_state_dict(torch.load('./save/ResNet/imagenet/original/2023/80/lr0.1000_wb8_ab32_ar8xac32.pt'), strict=False)
        


    # print(model)
    train_model(model, train_loader, test_loader)

    # input = torch.randn(1, 3, 32, 32).cuda()
    # flops, params = profile(model, inputs=(input,))


if __name__=='__main__':
  main()




