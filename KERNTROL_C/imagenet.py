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
from utils import *
import logging
import random
import torch.optim as optim

import torch.backends.cudnn as cudnn
from torchvision import models # for pretrained model
import torchvision.models as models_


##### Settings #########################################################################                      3x3에서는 patdnn을 따라가도록?
parser = argparse.ArgumentParser(description='Pytorch PatDNN training')
parser.add_argument('--dir',        default='/Data',           help='dataset root')
parser.add_argument('--model',      default='ResNet',          help = 'select model : ResNet_Q, Wide_ResNet_Q')
parser.add_argument('--dataset',    default='imagenet',          help='select dataset')
parser.add_argument('--batchsize',  default=512, type=int,      help='set batch size')
parser.add_argument('--lr',         default=1e-4, type=float,   help='set learning rate') # 6e-5
parser.add_argument('--epoch',      default=20, type=int,      help='set epochs') # 60
parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay') # before 5e-4
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--optimizer', default='SGD', type=str, help='[adam, SGD]')
parser.add_argument('--no-cuda',    default=False, action='store_true', help='disables CUDA training')
parser.add_argument('--u_th',    default=0, type=float, help='upper threshold of row utilization') # 0.25, 0.3, 0.4
parser.add_argument('--l5_th',    default=0, type=float, help='lower threshold of row utilization') # 0.25, 0.3, 0.4
parser.add_argument('--l4_th',    default=0.25, type=float, help='lower threshold of row utilization') # 0.25, 0.3, 0.4
parser.add_argument('--GPU', type=int, default=2) 
parser.add_argument('--ab', type=int, default=32)
parser.add_argument('--wb', type=int, default=32)
parser.add_argument('--seed', type=int, default=231113)
parser.add_argument('--ar', type=int, default=2048)
parser.add_argument('--ac', type=int, default=1024)
parser.add_argument('--scale', type=int, default=4)
parser.add_argument('--original', type=int, default=1, help = '1.: Conv2D,   2.: Switched Conv2D')
parser.add_argument('--mode', type=str, default='same', help = '1: same,  2: bound,  3:inter')
parser.add_argument('--save', type=int, default=0, help = 'model save   0:no, 1:do')
parser.add_argument('--pretrain', type=int, default=1, help = 'pretrained model   0:no, 1:do')
args = parser.parse_args()
print(args)

GPU_NUM = args.GPU # GPU
# device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(device) # change allocation of current GPU
# print ('Current cuda device ', torch.cuda.current_device()) # check
m_bit = args.wb
real_ac = int(args.ac/args.wb * m_bit)

args.workers = 16

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

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, a_bit, w_bit, in_planes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()

        self.w_bit = w_bit
        self.a_bit = a_bit

        self.act1 = Activate(self.a_bit)
        self.act2 = Activate(self.a_bit)
        if args.dataset == 'imagenet':
            if in_planes == 64:
                image_col, image_row = 56, 56
            elif in_planes == 128:
                image_col, image_row = 28, 28
            elif in_planes == 256:
                image_col, image_row = 14, 14
            elif in_planes == 512:
                image_col, image_row = 7, 7

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if args.original == 1:
            self.conv1 = Conv2d_Q_(self.w_bit, in_planes, planes, kernel_size=kernel, padding=(1,1), stride=stride, bias=False)
            self.bn1 = SwitchBatchNorm2d(self.w_bit, planes)
            self.conv2 = Conv2d_Q_(self.w_bit, planes, planes, kernel_size=kernel, padding=padding, stride=1, bias=False) 
            self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)
        
        else : 
            if in_planes != planes:
                self.conv1 = Conv2d_Q_(self.w_bit, in_planes, planes, kernel_size=kernel, padding=(1,1), stride=stride, bias=False)
                self.bn1 = SwitchBatchNorm2d(self.w_bit, planes)
                self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)

                pw = int(SDK(image_col, image_row, 3, 3, planes, planes, args.ar, args.ac))
                # print(in_planes, planes, pw)

                logger.info("planes = %d, planes = %d , pw = %d"%(planes, planes, pw))

                if pw == kernel : 
                    MK = None
                else :
                    if pw == 5 :
                        mask = pattern_gen(pw, kernel, args.l5_th)
                    else :
                        mask = pattern_gen(pw, kernel, args.l5_th)
                    W = pw - kernel 
                    MK = kernel + 2*1*W # 1 is stride
                    logger.info(mask)

                    
                    if args.original == 2:
                        pat_ours_pw5 = pattern_gen_v1(MK, pw, kernel, args.l5_th, args.u_th, planes, planes, mode=args.mode)
                        for i in range(len(pat_ours_pw5)):
                            logger.info(pat_ours_pw5[i][0][0])
                        pat_ours_pw4 = torch.tile(mask, (planes, planes, 1, 1)).cuda()

                    if args.original == 4:
                        pat_sdk = pattern_gen_sdk(MK, pw, kernel, mask, planes, planes)
                        for i in range(len(pat_sdk)):
                            logger.info(pat_sdk[i][0][0])
                st = pw - kernel + 1

                if args.original == 2:
                    if pw != kernel :
                        if pw == 5:
                            self.conv2 = SwitchedConv2d_update_ours(self.w_bit, planes, planes, pat=pat_ours_pw5, pw=pw, lth=args.l5_th, kernel_size=MK, padding=padding, stride=st, bias=False)
                        elif pw == 4:
                            self.conv2 = SwitchedConv2d_update_ours(self.w_bit, planes, planes, pat=pat_ours_pw4, pw=pw, lth=args.l5_th, kernel_size=MK, padding=padding, stride=st, bias=False)
                    else :
                        self.conv2 = Conv2d_Q_(self.w_bit, planes, planes, kernel_size=kernel, padding=padding, stride=1, bias=False) 

                elif args.original == 4:
                    if pw != kernel :
                        self.conv2 = SwitchedConv2d_update_sdk(self.w_bit, planes, planes, pat=pat_sdk, pw=pw, kernel_size=MK, padding=padding, stride=st, bias=False)
                    else :
                        self.conv2 = Conv2d_Q_(self.w_bit, planes, planes, kernel_size=kernel, padding=padding, stride=1, bias=False)   
                
            else: 
                self.bn1 = SwitchBatchNorm2d(self.w_bit, planes)
                self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)

                pw = int(SDK(image_col, image_row, 3, 3, planes, planes, args.ar, args.ac))

                logger.info("in_planes = %d, planes = %d , pw = %d"%(in_planes, planes, pw))
                if pw == kernel :  # 3 is kernel size
                    MK = None
                else :
                    W = pw - kernel # pattern_index
                    MK = kernel + 2*1*W # 1 is stride
                    if pw == 5 :
                        mask = pattern_gen(pw, kernel, args.l5_th)
                    else :
                        mask = pattern_gen(pw, kernel, args.l5_th)

                    logger.info(mask)
                    
                    if args.original == 2:
                        pat_ours_pw5 = pattern_gen_v1(MK, pw, kernel, args.l5_th, args.u_th, in_planes, planes, mode=args.mode)
                        for i in range(len(pat_ours_pw5)):
                            logger.info(pat_ours_pw5[i][0][0])
                        pat_ours_pw4 = torch.tile(mask, (planes, in_planes, 1, 1)).cuda()

                    if args.original == 4:
                        pat_sdk = pattern_gen_sdk(MK, pw, kernel, mask, in_planes, planes)
                        for i in range(len(pat_sdk)):
                            logger.info(pat_sdk[i][0][0])

                st = pw - kernel + 1         

                if args.original == 2:
                    if kernel != pw:   # 아래 라인에 planes -> in_planes 수정
                        if pw == 5 :
                            self.conv1 = SwitchedConv2d_update_ours(self.w_bit, planes, planes, pat=pat_ours_pw5, pw=pw, lth=args.l5_th, kernel_size=MK, padding=padding, stride=st, bias=False)
                            self.conv2 = SwitchedConv2d_update_ours(self.w_bit, planes, planes, pat=pat_ours_pw5, pw=pw, lth=args.l5_th, kernel_size=MK, padding=padding, stride=st, bias=False)
                        elif pw == 4:
                            self.conv1 = SwitchedConv2d_update_ours(self.w_bit, planes, planes, pat=pat_ours_pw4, pw=pw, lth=args.l5_th, kernel_size=MK, padding=padding, stride=st, bias=False)
                            self.conv2 = SwitchedConv2d_update_ours(self.w_bit, planes, planes, pat=pat_ours_pw4, pw=pw, lth=args.l5_th, kernel_size=MK, padding=padding, stride=st, bias=False)
                    else :
                        self.conv1 = Conv2d_Q_(self.w_bit, planes, planes, kernel_size=kernel, padding=padding, stride=1, bias=False) 
                        self.conv2 = Conv2d_Q_(self.w_bit, planes, planes, kernel_size=kernel, padding=padding, stride=1, bias=False) 

                elif args.original == 4:
                    if kernel != pw:
                        self.conv1 = SwitchedConv2d_update_sdk(self.w_bit, planes, planes, pat=pat_sdk, pw=pw, kernel_size=MK, padding=padding, stride=st, bias=False)
                        self.conv2 = SwitchedConv2d_update_sdk(self.w_bit, planes, planes, pat=pat_sdk, pw=pw, kernel_size=MK, padding=padding, stride=st, bias=False)
                    else :
                        self.conv1 = Conv2d_Q_(self.w_bit, planes, planes, kernel_size=kernel, padding=padding, stride=1, bias=False) 
                        self.conv2 = Conv2d_Q_(self.w_bit, planes, planes, kernel_size=kernel, padding=padding, stride=1, bias=False)        
        
        '''
        self.conv1 = Conv2d_Q_(self.w_bit, in_planes, planes, kernel_size=kernel, padding=(1,1), stride=stride, bias=False)
        self.bn1 = SwitchBatchNorm2d(self.w_bit, planes)
        self.act1 = Activate(self.a_bit)

        self.conv2 = Conv2d_Q_(self.w_bit, planes, planes, kernel_size=kernel, padding=padding, stride=1, bias=False) 
        self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)
        self.act2 = Activate(self.a_bit)
        '''

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



mode = args.mode

if args.original == 2:
    name = 'KERNTROLC'
    kernel = 3
    padding = 1

elif args.original == 4:
    name = 'KERNTROLM'
    kernel = 3
    padding = 1

directory = './log/%s/%s/%s/%d/%s'%(args.model, args.dataset, name, args.seed, args.epoch)
if not os.path.isdir(directory):
    os.makedirs(directory)
file_name = directory + '/wb%d_ab%d_l5th%.2f_ar%dxac%d.log'%(args.wb, args.ab, args.l5_th, args.ar, args.ac)


if args.save == 1:
    directory_save = './save/%s/%s/%s/%d/%d'%(args.model, args.dataset, name, args.seed, args.epoch)
    if not os.path.isdir(directory_save):
        os.makedirs(directory_save)


file_handler = logging.FileHandler(file_name)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


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
        
        
        scheduler.step()
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
            if args.save == 1:
                if args.original == 1 or args.original == 2 or args.original == 4:
                    print('model save')
                    logger.info('model save')
                    if args.original == 1:
                        torch.save(model.state_dict(), directory_save + '/lr%.4f_wb%d_ab%d_ar%dxac%d.pt'%(args.lr, args.wb, args.ab, args.ar, args.ac))
                    else :
                        torch.save(model.state_dict(), directory_save + '/lr%.4f_wb%d_ab%d_lth%.2f_ar%dxac%d.pt'%(args.lr, args.wb, args.ab, args.l5_th, args.ar, args.ac))

    logger.info('Final best accuracy is : %.4f '%(best_acc))

    

def main():
    train_loader, test_loader = data_loader(args.dir, args.dataset, args.batchsize, args.workers)

    if args.model == 'ResNet':
        if args.dataset == 'imagenet':
            num_class = 1000
        model = ResNet(args.ab, args.wb, block=BasicBlock, layers=[2,2,2,2], num_classes=num_class).cuda()
        if args.pretrain == 1:
            if args.original == 2:
                # model.load_state_dict(torch.load('./save/ResNet/imagenet/KERNTROLC/2023/80/lr0.1000_wb8_ab32_lth0.00_ar2048xac1024.pt'), strict=False)
                model.load_state_dict(torch.load('/workspace/KERNTROL_/save/ResNet/imagenet/KERNTROLC/lr0.1000_wb8_ab32_lth0.00_ar2048xac1024_66.27.pt'), strict=False)
            elif args.original == 4:
                model.load_state_dict(torch.load('./save/ResNet/imagenet/KERNTROLM/2023/80/lr0.1000_wb8_ab32_lth0.00_ar2048xac1024.pt'), strict=False)
            print('model is loaded')


    train_model(model, train_loader, test_loader)

if __name__=='__main__':
  main()






