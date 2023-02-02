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
# import utils
# from utils import *
# from tqdm import tqdm
# import pickle
from utils_1 import *
import logging
import random
# import pandas as pd
# from thop import profile 

import torch.backends.cudnn as cudnn

# import os
# os.environ['CUDA_VISIBLE_DEVICES']='사용할GPU_UUID'



##### Settings #########################################################################                      3x3에서는 patdnn을 따라가도록?
parser = argparse.ArgumentParser(description='Pytorch PatDNN training')
parser.add_argument('--dir',        default='/Data',           help='dataset root')
parser.add_argument('--model',      default='Wide_ResNet_Q',          help = 'select model : ResNet_Q, Wide_ResNet_Q')
parser.add_argument('--dataset',    default='cifar100',          help='select dataset')
parser.add_argument('--batchsize',  default=512, type=int,      help='set batch size')
parser.add_argument('--lr',         default=0.1, type=float,   help='set learning rate') # 6e-5
parser.add_argument('--epoch',      default=200, type=int,      help='set epochs') # 60
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--optimizer', default='SGD', type=str, help='[adam, SGD]')
parser.add_argument('--no-cuda',    default=False, action='store_true', help='disables CUDA training')
parser.add_argument('--u_th',    default=0.5, type=float, help='upper threshold of row utilization') # 0.25, 0.3, 0.4
parser.add_argument('--l5_th',    default=0.25, type=float, help='lower threshold of row utilization') # 0.25, 0.3, 0.4
parser.add_argument('--l4_th',    default=0.25, type=float, help='lower threshold of row utilization') # 0.25, 0.3, 0.4
parser.add_argument('--GPU', type=int, default=2) 
parser.add_argument('--ab', type=int, default=32)
parser.add_argument('--wb', type=int, default=2)
parser.add_argument('--seed', type=int, default=1992)
parser.add_argument('--ar', type=int, default=512)
parser.add_argument('--ac', type=int, default=512)
parser.add_argument('--scale', type=int, default=4)
parser.add_argument('--original', type=int, default=1, help = '1.: Conv2D,   2.: Switched Conv2D')
parser.add_argument('--mode', type=str, default='same', help = '1: same,  2: bound,  3:inter')
parser.add_argument('--save', type=int, default=0, help = 'model save   0:no, 1:do')
args = parser.parse_args()
print(args)

GPU_NUM = args.GPU # GPU
# device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(device) # change allocation of current GPU
# print ('Current cuda device ', torch.cuda.current_device()) # check
m_bit = args.wb
real_ac = int(args.ac/args.wb * m_bit)


# os.environ["CUDA_VISIBLE_DEVICES"]= str(GPU_NUM)  # Set the GPU 2 to use
# if args.GPU == 0 :  
#     os.environ['CUDA_VISIBLE_DEVICES']='MIG-ffeb7d21-314a-5375-9837-8d1bb9f18872'
# elif args.GPU == 1:
#     os.environ['CUDA_VISIBLE_DEVICES']='MIG-d064e803-fa01-50f8-b10a-bcda0f02b03c'
# elif args.GPU == 2:
#     os.environ['CUDA_VISIBLE_DEVICES']='MIG-ae73c3ad-1cef-56bd-ab31-3bf18c0f8e62'

    
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= str(GPU_NUM)  # Set the GPU 2 to use
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
print ('Current cuda device ', device)

# device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
# print ('Current cuda device ', device)

# ranint = random.randint(0, 10000000)
## seed 1992일 때, 우리의 방법이 제일 좋음
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



### WRN 16-4 만드는중.
class Wide_BasicBlock_Q(nn.Module):
    expansion = 1
    def __init__(self, a_bit, w_bit, in_planes, planes, stride=1):
        super().__init__()
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.act1 = Activate(self.a_bit)
        self.act2 = Activate(self.a_bit)


        self.dropout = nn.Dropout(0.3) # p = 0.3
        global image_col, image_row
        # if args.dataset == 'ResNet_Q':
        if in_planes == 16 or in_planes == 64:
            image_col, image_row = 32, 32
        elif in_planes == 128:
            image_col, image_row = 16, 16
        elif in_planes == 256:
            image_col, image_row = 8, 8
        
        
        if args.original == 1:
            self.conv1 = Conv2d_Q_(self.w_bit, in_planes, planes, kernel_size=kernel, padding=(1,1), stride=stride, bias=False)
            self.bn1 = SwitchBatchNorm2d(self.w_bit, in_planes)
            self.conv2 = Conv2d_Q_(self.w_bit, planes, planes, kernel_size=kernel, padding=padding, stride=1, bias=False) 
            self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)

        elif args.original == 3:
            self.conv1 = Conv2d_Q_(self.w_bit, in_planes, planes, kernel_size=kernel, padding=(1,1), stride=stride, bias=False)
            self.bn1 = SwitchBatchNorm2d(self.w_bit, in_planes)
            self.conv2 = Conv2d_Q_(self.w_bit, planes, planes, kernel_size=kernel, padding=padding, stride=1, bias=False) 
            self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)
        else : 
            if in_planes != planes:
                self.conv1 = Conv2d_Q_(self.w_bit, in_planes, planes, kernel_size=(3,3), padding=(1,1), stride=stride, bias=False)
                self.bn1 = SwitchBatchNorm2d(self.w_bit, in_planes)

                pw = int(SDK(image_col, image_row, 3, 3, planes, planes, args.ar, args.ac))
                # if pw == 5:
                #     uth = 0.15
                # else:
                #     uth = 0.25

                logger.info("planes = %d, planes = %d , pw = %d"%(planes, planes, pw))


                # ### weight mask 만드는 함수
                # windows = pw - kernel + 1
                # weight_mask = torch.zeros(MK, MK).cuda()
                # for i in range(0, windows) :
                #     for j in range(0, windows) :
                #         weight_mask[i:i+pw, j:j+pw] = weight_mask[i:i+pw, j:j+pw] + mask

                # weight_mask = torch.where(weight_mask>=1, 1, 0)
                # ####

                if pw == kernel : 
                    MK = None
                else :
                    if pw == 5 :
                        mask = pattern_gen(pw, kernel, args.l5_th)
                    else :
                        mask = pattern_gen(pw, kernel, args.l4_th)
                    W = pw - kernel 
                    MK = kernel + 2*1*W # 1 is stride
                    logger.info(mask)

                    
                    if args.original == 2:
                        pat_ours_pw5 = pattern_gen_v1(MK, pw, kernel, args.l5_th, args.u_th, planes, planes, mode=args.mode)
                        for i in range(len(pat_ours_pw5)):
                            logger.info(pat_ours_pw5[i][0][0])
                        pat_ours_pw4 = torch.tile(mask, (planes, planes, 1, 1)).cuda()
                    
                    '''
                    if args.original == 2:
                        if pw == 5 :
                            pat_ours = pattern_gen_sdk(MK, pw, kernel, mask, planes, planes)
                            for i in range(len(pat_ours)):
                                logger.info(pat_ours[i][0][0])
                        elif pw == 4 :
                            pat_ours = torch.tile(mask, (planes, planes, 1, 1)).cuda()
                            logger.info(pat_ours[0][0][0])
                    '''
                    if args.original == 4:
                        pat_sdk = pattern_gen_sdk(MK, pw, kernel, mask, planes, planes)
                        for i in range(len(pat_sdk)):
                            logger.info(pat_sdk[i][0][0])

                st = pw - kernel + 1

                if args.original == 2:
                    if pw != kernel :
                        if pw == 5:
                            self.conv2 = SwitchedConv2d_update_ours(self.w_bit, planes, planes, pat=pat_ours_pw5, pw=pw, lth=args.l5_th, kernel_size=MK, padding=padding, stride=st, bias=False)
                            # self.conv2 = SwitchedConv2d_update_sdk(self.w_bit, planes, planes, pat=pat_ours, pw=pw, kernel_size=MK, padding=padding, stride=st, bias=False)
                        elif pw == 4:
                            self.conv2 = SwitchedConv2d_update_ours(self.w_bit, planes, planes, pat=pat_ours_pw4, pw=pw, lth=args.l4_th, kernel_size=MK, padding=padding, stride=st, bias=False)
                            # self.conv2 = SwitchedConv2d_update_ours(self.w_bit, planes, planes, pat=pat_ours, pw=pw, lth=args.l4_th, kernel_size=MK, padding=padding, stride=st, bias=False)
                    else :
                        self.conv2 = Conv2d_Q_(self.w_bit, planes, planes, kernel_size=kernel, padding=padding, stride=1, bias=False) 

                elif args.original == 4:
                    if pw != kernel :
                        self.conv2 = SwitchedConv2d_update_sdk(self.w_bit, planes, planes, pat=pat_sdk, pw=pw, kernel_size=MK, padding=padding, stride=st, bias=False)
                    else :
                        self.conv2 = Conv2d_Q_(self.w_bit, planes, planes, kernel_size=kernel, padding=padding, stride=1, bias=False) 

                self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)
                
            else: 
                self.bn1 = SwitchBatchNorm2d(self.w_bit, in_planes)
                self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)
                # self.bn1 = nn.BatchNorm2d(in_planes)

                pw = int(SDK(image_col, image_row, 3, 3, planes, planes, args.ar, args.ac))
                # pw = int(args.u_th) # 임시로 셋팅
                # if pw == 5:
                #     uth = 0.15
                # else:
                #     uth = 0.25

                ### weight mask 만드는 함수
                # windows = pw - kernel + 1
                # weight_mask = torch.zeros(MK, MK).cuda()
                # for i in range(0, windows) :
                #     for j in range(0, windows) :
                #         weight_mask[i:i+pw, j:j+pw] = weight_mask[i:i+pw, j:j+pw] + mask

                # weight_mask = torch.where(weight_mask>=1, 1, 0)
                # ####



                logger.info("in_planes = %d, planes = %d , pw = %d"%(in_planes, planes, pw))
                if pw == kernel :  # 3 is kernel size
                    MK = None
                else :
                    W = pw - kernel # pattern_index
                    MK = kernel + 2*1*W # 1 is stride
                    if pw == 5 :
                        mask = pattern_gen(pw, kernel, args.l5_th)
                    else :
                        mask = pattern_gen(pw, kernel, args.l4_th)
                    # mask = pattern_gen(pw, kernel, args.l_th)
                    # mask = pattern_gen(pw, kernel, uth)
                    logger.info(mask)
                    
                    if args.original == 2:
                        pat_ours_pw5 = pattern_gen_v1(MK, pw, kernel, args.l5_th, args.u_th, in_planes, planes, mode=args.mode)
                        for i in range(len(pat_ours_pw5)):
                            logger.info(pat_ours_pw5[i][0][0])
                        pat_ours_pw4 = torch.tile(mask, (planes, in_planes, 1, 1)).cuda()
                    '''
                    if args.original == 2:
                        if pw == 5 :
                            pat_ours = pattern_gen_sdk(MK, pw, kernel, mask, planes, planes)
                            for i in range(len(pat_ours)):
                                logger.info(pat_ours[i][0][0])
                        elif pw == 4 :
                            pat_ours = torch.tile(mask, (planes, planes, 1, 1)).cuda()

                            logger.info(pat_ours[0][0][0])
                    '''
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
                            # self.conv1 = SwitchedConv2d_update_sdk(self.w_bit, planes, planes, pat=pat_ours, pw=pw, kernel_size=MK, padding=padding, stride=st, bias=False)
                            # self.conv2 = SwitchedConv2d_update_sdk(self.w_bit, planes, planes, pat=pat_ours, pw=pw, kernel_size=MK, padding=padding, stride=st, bias=False)
                        elif pw == 4:
                            self.conv1 = SwitchedConv2d_update_ours(self.w_bit, planes, planes, pat=pat_ours_pw4, pw=pw, lth=args.l4_th, kernel_size=MK, padding=padding, stride=st, bias=False)
                            self.conv2 = SwitchedConv2d_update_ours(self.w_bit, planes, planes, pat=pat_ours_pw4, pw=pw, lth=args.l4_th, kernel_size=MK, padding=padding, stride=st, bias=False)
                            # self.conv1 = SwitchedConv2d_update_ours(self.w_bit, planes, planes, pat=pat_ours, pw=pw, lth=args.l4_th, kernel_size=MK, padding=padding, stride=st, bias=False)
                            # self.conv2 = SwitchedConv2d_update_ours(self.w_bit, planes, planes, pat=pat_ours, pw=pw, lth=args.l4_th, kernel_size=MK, padding=padding, stride=st, bias=False)
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


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.BatchNorm2d(self.expansion * in_planes),
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                )

    def forward(self, x):
        out = self.conv1(self.act1(self.bn1(x)))
        out = self.dropout(out)
        out = self.act2(self.bn2(out))
        out = self.conv2(out)
        out += self.shortcut(x)  # x used here
        return out


class Wide_ResNet_Q(nn.Module):
    def __init__(self, a_bit, w_bit, block, num_blocks, scale, num_classes=10): 
        super().__init__()

        self.in_planes = 16
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.act = Activate(self.a_bit)
        nStages = [16, 16*scale, 32*scale, 64*scale]
        self.bn1 = SwitchBatchNorm2d(self.w_bit, nStages[3])
        
        self.layers = nn.Sequential(
            nn.Conv2d(3, nStages[0], kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(self.in_planes),
            nn.ReLU(),
            
            *self._make_layer(block, nStages[1], num_blocks[0], stride=1), 
            *self._make_layer(block, nStages[2], num_blocks[1], stride=2),
            *self._make_layer(block, nStages[3], num_blocks[2], stride=2),
        )

        # mask_prune(self.layers)
        self.fc = nn.Linear(nStages[3], num_classes) 

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            # Full precision
            # option is 'A': Use F.pad
            # option is 'B': Use Conv+BN
            layers.append(block(self.a_bit, self.w_bit, self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        out = self.act(self.bn1(out))
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out 


mode = args.mode



if args.original == 1 :
    name = 'original'
    kernel = 3
    padding = (1,1) # 원래는 1

elif args.original == 2:
    name = 'ours'
    kernel = 3
    padding = 1

elif args.original == 3:
    name = '5x5_layer1_original'
    kernel = 5
    padding = 2

elif args.original == 4:
    name = 'SDK'
    kernel = 3
    padding = 1



directory = './log/%s/%s/%s/%s/%d/%s/%.2f/%.4f'%(args.model, args.dataset, name, args.seed, args.epoch, args.mode, args.u_th, args.lr)
if not os.path.isdir(directory):
    os.makedirs(directory)
file_name = directory + '/wb%d_ab%d_l5th%.2f_l4th%.2f_ar%dxac%d.log'%(args.wb, args.ab, args.l5_th, args.l4_th, args.ar, args.ac)


if args.save == 1:
    directory_save = './save/%s/%s/%s/%s/%d/%s'%(args.model, args.dataset, name, args.seed, args.epoch, args.mode)
    if not os.path.isdir(directory_save):
        os.makedirs(directory_save)


file_handler = logging.FileHandler(file_name)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

args.workers = 1

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")



#########################################################################################################
# ResNet code modified from original of [https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py]
# Modified version for our experiment.


'''
ResNet-20 Quantization
'''

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class BasicBlock_Q(nn.Module):
    expansion = 1
    def __init__(self, a_bit, w_bit, in_planes, planes, stride=1, option='A'):
        super().__init__()
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.act1 = Activate(self.a_bit)
        self.act2 = Activate(self.a_bit)
        # self.act1 = nn.ReLU()
        # self.act2 = nn.ReLU()

        global image_col, image_row
        # if args.dataset == 'ResNet_Q':
        if in_planes == 16:
            image_col, image_row = 32, 32
        elif in_planes == 32:
            image_col, image_row = 16, 16
        elif in_planes == 64:
            image_col, image_row = 8, 8
        
        
        if args.original == 1:
            self.conv1 = Conv2d_Q_(self.w_bit, in_planes, planes, kernel_size=kernel, padding=(1,1), stride=stride, bias=False)
            self.bn1 = SwitchBatchNorm2d(self.w_bit, planes)
            self.conv2 = Conv2d_Q_(self.w_bit, planes, planes, kernel_size=kernel, padding=padding, stride=1, bias=False) 
            self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)

        elif args.original == 3:
            self.conv1 = Conv2d_Q_(self.w_bit, in_planes, planes, kernel_size=kernel, padding=(1,1), stride=stride, bias=False)
            self.bn1 = SwitchBatchNorm2d(self.w_bit, planes)
            self.conv2 = Conv2d_Q_(self.w_bit, planes, planes, kernel_size=kernel, padding=padding, stride=1, bias=False) 
            self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)

        else:
            if in_planes != planes:
                self.conv1 = Conv2d_Q_(self.w_bit, in_planes, planes, kernel_size=(3,3), padding=(1,1), stride=stride, bias=False)
                self.bn1 = SwitchBatchNorm2d(self.w_bit, planes)

                pw = int(SDK(image_col, image_row, 3, 3, planes, planes, args.ar, args.ac))
                logger.info("planes = %d, planes = %d , pw = %d"%(planes, planes, pw))
                W = pw - kernel 
                MK = kernel + 2*1*W # 1 is stride

                if pw == kernel : 
                    MK = None
                else :
                    if pw == 5 :
                        mask = pattern_gen(pw, kernel, args.l5_th)
                    else :
                        mask = pattern_gen(pw, kernel, args.l4_th)
                    logger.info(mask)
                    if args.original == 2:
                        pat_ours_pw5 = pattern_gen_v1(MK, pw, kernel, args.l5_th, 0, planes, planes, mode='same')
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
                            self.conv2 = SwitchedConv2d_update_ours(self.w_bit, planes, planes, pat=pat_ours_pw4, pw=pw, lth=args.l4_th, kernel_size=MK, padding=padding, stride=st, bias=False)
                    else :
                        self.conv2 = Conv2d_Q_(self.w_bit, planes, planes, kernel_size=kernel, padding=padding, stride=1, bias=False) 

                elif args.original == 4:
                    if pw != kernel :
                        self.conv2 = SwitchedConv2d_update_sdk(self.w_bit, planes, planes, pat=pat_sdk, pw=pw, kernel_size=MK, padding=padding, stride=st, bias=False)
                    else :
                        self.conv2 = Conv2d_Q_(self.w_bit, planes, planes, kernel_size=kernel, padding=padding, stride=1, bias=False) 

                self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)


            else: 
                self.bn1 = SwitchBatchNorm2d(self.w_bit, in_planes)
                self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)

                pw = int(SDK(image_col, image_row, 3, 3, planes, planes, args.ar, args.ac))
                logger.info("planes = %d, planes = %d , pw = %d"%(in_planes, planes, pw))
                if pw == kernel :  # 3 is kernel size
                    MK = None
                else :
                    W = pw - kernel # pattern_index
                    MK = kernel + 2*1*W # 1 is stride
                    if pw == 5 :
                        mask = pattern_gen(pw, kernel, args.l5_th)
                    else :
                        mask = pattern_gen(pw, kernel, args.l4_th)
                    # mask = pattern_gen(pw, kernel, args.l_th)
                    # mask = pattern_gen(pw, kernel, uth)
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
                    if kernel != pw:  
                        if pw == 5 :
                            self.conv1 = SwitchedConv2d_update_ours(self.w_bit, planes, planes, pat=pat_ours_pw5, pw=pw, lth=args.l5_th, kernel_size=MK, padding=padding, stride=st, bias=False)
                            self.conv2 = SwitchedConv2d_update_ours(self.w_bit, planes, planes, pat=pat_ours_pw5, pw=pw, lth=args.l5_th, kernel_size=MK, padding=padding, stride=st, bias=False)
                        elif pw == 4:
                            self.conv1 = SwitchedConv2d_update_ours(self.w_bit, planes, planes, pat=pat_ours_pw4, pw=pw, lth=args.l4_th, kernel_size=MK, padding=padding, stride=st, bias=False)
                            self.conv2 = SwitchedConv2d_update_ours(self.w_bit, planes, planes, pat=pat_ours_pw4, pw=pw, lth=args.l4_th, kernel_size=MK, padding=padding, stride=st, bias=False)
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



        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                '''
                For CIFAR10 ResNet paper uses option A.
                '''
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)  # x used here
        out = self.act2(out)
        return out

# ResNet code modified from original of [https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py]
# Modified version for our experiment.
class ResNet20_Q(nn.Module):
    def __init__(self, a_bit, w_bit, block, num_blocks, scale, num_classes=10): 
        super().__init__()
        self.in_planes = 16 # Resnet

        self.w_bit = w_bit
        self.a_bit = a_bit
        self.act = Activate(self.a_bit)
        # self.act = nn.ReLU()

        self.layers = nn.Sequential(
            nn.Conv2d(3, self.in_planes, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(self.in_planes),
            Activate(self.a_bit),
            # nn.ReLU(),
            
            *self._make_layer(block, 16, num_blocks[0], stride=1),
            *self._make_layer(block, 32, num_blocks[1], stride=2),
            *self._make_layer(block, 64, num_blocks[2], stride=2),
        )

        # mask_prune(self.layers)
        self.fc = nn.Linear(64, num_classes) 

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            # Full precision
            # option is 'A': Use F.pad 
            # option is 'B': Use Conv+BN
            layers.append(block(self.a_bit, self.w_bit, self.in_planes, planes, stride, option='B'))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out 



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
    if args.dataset == 'cifar10' :
        T = 30
    elif args.dataset =='cifar100':
        T = 50
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T, 1e-4)


    print(f'TRAINING START!')
    for epoch in range(args.epoch):
        model.train()
        cnt = 0
        loss_sum = 0

        '''
        Scheduler를 사용하지 않는 경우 적용

        if args.optimizer == 'adam' :
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'prune' : # patdnn에서 사용한 optimizer
            optimizer = PruneAdam(model.named_parameters(), lr=args.lr) 
        elif args.optimizer == 'SGD' :
            if args.dataset == 'svhn' :
                optimizer = optim.SGD(model.parameters(), lr=cf_learning_rate_svhn(args.lr, epoch), momentum=0.9, weight_decay=args.weight_decay)
            else:
                # optimizer = optim.SGD(model.parameters(), lr=cf_learning_rate(args.lr, epoch), momentum=0.9, weight_decay=args.weight_decay)
                 optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        '''


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
        # print(model.layers)
        if epoch % 20 == 0:
            logger.info("*"*100)
            logger.info("Epoch %d/%d"%(epoch, args.epoch))
            logger.info("Layer 1 Weight")
            # logger.info(model.layers[4].conv1.weight[0][0])
            logger.info(torch.sum(model.layers[4].conv1.weight, dim=[0,1]))
            logger.info("Layer 1 Gradient")
            # logger.info(model.layers[4].conv1.weight.grad[0][0])
            logger.info(torch.sum(model.layers[4].conv1.weight.grad, dim=[0,1]))
            logger.info("="*50)
            logger.info("Layer 2 Weight")
            # logger.info(model.layers[6].conv1.weight[0][0])
            logger.info(torch.sum(model.layers[6].conv1.weight, dim=[0,1]))
            logger.info("Layer 2 Gradient")
            # logger.info(model.layers[6].conv1.weight.grad[0][0])
            logger.info(torch.sum(model.layers[6].conv1.weight.grad, dim=[0,1]))
            logger.info("="*50)
            logger.info("Layer 3 Weight")
            # logger.info(model.layers[8].conv1.weight[0][0])
            logger.info(torch.sum(model.layers[8].conv1.weight, dim=[0,1]))
            logger.info("Layer 3 Gradient")
            # logger.info(model.layers[8].conv1.weight.grad[0][0])
            logger.info(torch.sum(model.layers[8].conv1.weight.grad, dim=[0,1]))
            logger.info("*"*100)

        # print(model.layers[4].conv1.weight[0][0])
        print(f'Epochs : {epoch+1}, Accuracy : {acc}')
        logger.info("Epoch %d/%d, Acc=%.4f"%(epoch+1, args.epoch, acc))
        
        scheduler.step()

        # result_list['train_acc'].append(acc)
        
        # data_frame = pd.DataFrame(data=result_list, index=range(0, epoch + 1))
        # if args.wb == 1 :
        #     csv_name = './log/resnet20/cifar10/scale1_1bit/%s_%s_%.4f_%d_%d_wb%d_ab%d_%.2f_256x256.csv'%(args.model, name, args.lr, args.epoch, args.seed, args.wb, args.ab, args.th)
        # elif args.wb == 2 :
        #     csv_name = './log/resnet20/cifar10/scale1_2bit/%s_%s_%.4f_%d_%d_wb%d_ab%d_%.2f_256x256.csv'%(args.model, name, args.lr, args.epoch, args.seed, args.wb, args.ab, args.th)
        # else :
        #     csv_name = './log/resnet20/cifar10/scale1_4bit/%s_%s_%.4f_%d_%d_wb%d_ab%d_%.2f_256x256.csv'%(args.model, name, args.lr, args.epoch, args.seed, args.wb, args.ab, args.th)
        # data_frame.to_csv(csv_name, index_label='epoch')        

        if acc > best_acc :
            best_acc = acc
            print('Best accuracy is updated! at epoch %d/%d: %.4f '%(epoch+1, args.epoch, best_acc))
            logger.info('Best accuracy is : %.4f '%(best_acc))
            if args.save == 1:
                if args.original == 1 or args.original == 2 or args.original == 4:
                    print('model save')
                    logger.info('model save')
                    if args.original == 1:
                        torch.save(model.state_dict(), directory_save + '/lr%.4f_wb%d_ab%d_.pt'%(args.lr, args.wb, args.ab))
                    else :
                        torch.save(model.state_dict(), directory_save + '/lr%.4f_wb%d_ab%d_lth%.2f_uth%.2f_ar%dxac%d.pt'%(args.lr, args.wb, args.ab, args.l_th, args.u_th, args.ar, args.ac))

    logger.info('Final best accuracy is : %.4f '%(best_acc))

    

def main():
    train_loader, test_loader = data_loader(args.dir, args.dataset, args.batchsize, args.workers)

    if args.model == 'VGG9_Q' :
        model = VGG9_Q(args.ab, args.wb, num_classes=100).cuda()

    elif args.model == 'ResNet_Q' :
        if args.dataset == 'cifar10' :
            model = ResNet20_Q(args.ab, args.wb, block=BasicBlock_Q, num_blocks=[3,3,3], scale=1, num_classes=10).cuda()
        elif args.dataset == 'cifar100' :
            model = ResNet20_Q(args.ab, args.wb, block=BasicBlock_Q, num_blocks=[3,3,3], scale=1, num_classes=100).cuda()

    elif args.model == 'Wide_ResNet_Q' :
        if args.dataset == 'cifar10' or args.dataset == 'svhn':
            model = Wide_ResNet_Q(args.ab, args.wb, block=Wide_BasicBlock_Q, num_blocks=[2,2,2], scale=args.scale, num_classes=10).cuda()
        elif args.dataset == 'cifar100' :
            model = Wide_ResNet_Q(args.ab, args.wb, block=Wide_BasicBlock_Q, num_blocks=[2,2,2], scale=args.scale, num_classes=100).cuda()


    train_model(model, train_loader, test_loader)

    # input = torch.randn(1, 3, 32, 32).cuda()
    # flops, params = profile(model, inputs=(input,))


if __name__=='__main__':
  main()






