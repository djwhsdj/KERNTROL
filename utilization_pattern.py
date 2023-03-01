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
parser.add_argument('--model',      default='Wide_ResNet_Q',          help = 'select model : VGG9_Q, ResNet_Q, Wide_ResNet_Q')
parser.add_argument('--dataset',    default='cifar100',          help='select dataset')
parser.add_argument('--batchsize',  default=512, type=int,      help='set batch size')
parser.add_argument('--lr',         default=0.1, type=float,   help='set learning rate') # 6e-5
parser.add_argument('--epoch',      default=102, type=int,      help='set epochs') # 60
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--optimizer', default='SGD', type=str, help='[adam, SGD]')
parser.add_argument('--no-cuda',    default=False, action='store_true', help='disables CUDA training')
parser.add_argument('--u_th',    default=0, type=float, help='upper threshold of row utilization') # 0.25, 0.3, 0.4
parser.add_argument('--l_th',    default=0, type=float, help='lower threshold of row utilization') # 0.25, 0.3, 0.4
parser.add_argument('--GPU', type=int, default=2) 
parser.add_argument('--ab', type=int, default=4)
parser.add_argument('--wb', type=int, default=1)
parser.add_argument('--seed', type=int, default=1992)
parser.add_argument('--ar', type=int, default=2014)
parser.add_argument('--ac', type=int, default=512)
parser.add_argument('--scale', type=int, default=4)
parser.add_argument('--original', type=int, default=1)
parser.add_argument('--entry', type=int, default=6, help = '6: 6-entry, 4: 4-entry')
args = parser.parse_args()
print(args)

GPU_NUM = args.GPU # GPU
# device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(device) # change allocation of current GPU
# print ('Current cuda device ', torch.cuda.current_device()) # check
m_bit = args.wb
real_ac = int(args.ac/args.wb * m_bit)

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
# os.environ["CUDA_VISIBLE_DEVICES"]= str(GPU_NUM)  # Set the GPU 2 to use
# device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
# print ('Current cuda device ', device)

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




kernel = 3
padding = 1

if args.original == 0:
    name = 'Random' # 6 entry

elif args.original == 1:
    name = 'PatDNN'

elif args.original == 2:
    name = 'Pconv'



class LambdaLayer__(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class BasicBlock_Q__(nn.Module):
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
        
    
        self.conv1 = Conv2d_Q_(self.w_bit, in_planes, planes, kernel_size=kernel, padding=(1,1), stride=stride, bias=False)
        self.bn1 = SwitchBatchNorm2d(self.w_bit, planes)
        self.conv2 = Conv2d_Q_(self.w_bit, planes, planes, kernel_size=kernel, padding=padding, stride=1, bias=False) 
        self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)



        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                '''
                For CIFAR10 ResNet paper uses option A.
                '''
                self.shortcut = LambdaLayer__(lambda x:
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
class ResNet20_Q__(nn.Module):
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




class Wide_BasicBlock_Q__(nn.Module):
    expansion = 1
    def __init__(self, a_bit, w_bit, in_planes, planes, stride=1):
        super().__init__()
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.act1 = Activate(self.a_bit)
        self.act2 = Activate(self.a_bit)

        self.dropout = nn.Dropout(0.3) # p = 0.3
        
        self.conv1 = Conv2d_Q_(self.w_bit, in_planes, planes, kernel_size=kernel, padding=(1,1), stride=stride, bias=False)
        self.bn1 = SwitchBatchNorm2d(self.w_bit, in_planes)
        self.conv2 = Conv2d_Q_(self.w_bit, planes, planes, kernel_size=kernel, padding=padding, stride=1, bias=False) 
        self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)

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


class Wide_ResNet_Q__(nn.Module):
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



kernel = 3
padding = 1
if args.dataset == 'cifar100' :
    directory_save = './save/Wide_ResNet_Q/cifar100/original/%d/200/lr0.1000_wb2_ab8_.pt'%(args.seed)
    model = Wide_ResNet_Q__(args.ab, args.wb, block=Wide_BasicBlock_Q__, num_blocks=[2,2,2], scale=args.scale, num_classes=100).cuda()

if args.dataset == 'cifar10' :
    directory_save = './save/ResNet_Q/cifar10/original/%d/100/lr0.1000_wb1_ab4_.pt'%(args.seed)
    model = ResNet20_Q__(args.ab, args.wb, block=BasicBlock_Q__, num_blocks=[3,3,3], scale=1, num_classes=10).cuda()

model.load_state_dict(torch.load(directory_save), strict = False)
# model.eval()

pattern_dic = []
entry = 3

pruned_weight = 9 - args.entry
if args.original == 1 :
    unpruned_weight = args.entry - 1 # 가운데 제외시키려고
for nam, p in model.named_parameters():
    if nam.split('.')[-1] == "weight" and len(p.shape) == 4 and 'shortcut' not in nam and '0' not in nam:
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


        elif args.original == 2: # pconv
            tensor = torch.abs(torch.sum(p, (0,1)))
            tensor = tensor.cpu().detach().numpy()

            row1 = tensor[0][1] + tensor[1][0] + tensor[1][1] + tensor[1][2] 
            row3 = tensor[0][1] + tensor[1][1] + tensor[1][2] + tensor[2][1] 
            col1 = tensor[1][0] + tensor[1][1] + tensor[1][2] + tensor[2][1] 
            col3 = tensor[0][1] + tensor[1][0] + tensor[1][1] + tensor[2][1] 

            val = [row1, row3, col1, col3]
            min_val = min(val)

            if val.index(min_val) == 0:
                mask = [0,1,0,1,1,1,0,0,0]
            elif val.index(min_val) == 1:
                mask = [0,1,0,0,1,1,0,1,0]
            elif val.index(min_val) == 2:
                mask = [0,0,0,1,1,1,0,1,0]
            else :
                mask = [0,1,0,1,1,0,0,1,0]


        pattern_dic.append(mask)



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
        
    

        if in_planes != planes:
            logger.info('='*50)
            logger.info('planes: %d,  out_plaens: %d'%(planes, planes))
            if in_planes == 16:
                pat1 = torch.tile(torch.tensor(pattern_dic[0]).reshape(3,3), (planes, planes, 1,1)).cuda()
                count1 = counting(pattern_dic[0], 3, 3)
                count2 = counting(pattern_dic[0], 4, 4)
                count3 = counting(pattern_dic[0], 5, 5)
                logger.info(pattern_dic[0])

            elif in_planes == 32:
                pat1 = torch.tile(torch.tensor(pattern_dic[6]).reshape(3,3), (planes, planes, 1,1)).cuda()
                count1 = counting(pattern_dic[6], 3, 3)
                count2 = counting(pattern_dic[6], 4, 4)
                count3 = counting(pattern_dic[6], 5, 5)
                logger.info(pattern_dic[6])

            elif in_planes == 64:
                pat1 = torch.tile(torch.tensor(pattern_dic[12]).reshape(3,3), (planes, planes, 1,1)).cuda()  
                count1 = counting(pattern_dic[12], 3, 3)
                count2 = counting(pattern_dic[12], 4, 4)
                count3 = counting(pattern_dic[12], 5, 5)
                logger.info(pattern_dic[12])

            logger.info('reduced rows: if pw==3: %d, pw==4: %d, pw==5: %d'%(count1*planes, count2*planes, count3*planes))

            self.conv1 = Conv2d_Q_(self.w_bit, in_planes, planes, kernel_size=(3,3), padding=(1,1), stride=stride, bias=False)
            self.bn1 = SwitchBatchNorm2d(self.w_bit, planes)
            self.conv2 = Conv2d_Q_mask(self.w_bit, planes, planes, kernel_size=3, pat=pat1, padding=padding, stride=1, bias=False) 
            self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)
            
        else: 
            logger.info('='*50)
            logger.info('in_planes: %d,  out_plaens: %d'%(in_planes, planes))
            if planes == 16:
                pat1 = torch.tile(torch.tensor(pattern_dic[1]).reshape(3,3), (planes, in_planes, 1,1)).cuda()
                count1 = counting(pattern_dic[1], 3, 3)
                count2 = counting(pattern_dic[1], 4, 4)
                count3 = counting(pattern_dic[1], 5, 5)
                logger.info(pattern_dic[1])

                pat2 = torch.tile(torch.tensor(pattern_dic[2]).reshape(3,3), (planes, in_planes, 1,1)).cuda()
                count4 = counting(pattern_dic[2], 3, 3)
                count5 = counting(pattern_dic[2], 4, 4)
                count6 = counting(pattern_dic[2], 5, 5)
                logger.info(pattern_dic[2])

            elif planes == 32:
                pat1 = torch.tile(torch.tensor(pattern_dic[7]).reshape(3,3), (planes, in_planes, 1,1)).cuda()
                count1 = counting(pattern_dic[7], 3, 3)
                count2 = counting(pattern_dic[7], 4, 4)
                count3 = counting(pattern_dic[7], 5, 5)

                logger.info(pattern_dic[7])
                pat2 = torch.tile(torch.tensor(pattern_dic[8]).reshape(3,3), (planes, in_planes, 1,1)).cuda()
                count4 = counting(pattern_dic[8], 3, 3)
                count5 = counting(pattern_dic[8], 4, 4)
                count6 = counting(pattern_dic[8], 5, 5)
                logger.info(pattern_dic[8])

            elif planes == 64:
                pat1 = torch.tile(torch.tensor(pattern_dic[13]).reshape(3,3), (planes, in_planes, 1,1)).cuda()
                count1 = counting(pattern_dic[13], 3, 3)
                count2 = counting(pattern_dic[13], 4, 4)
                count3 = counting(pattern_dic[13], 5, 5)
                logger.info(pattern_dic[13])

                pat2 = torch.tile(torch.tensor(pattern_dic[14]).reshape(3,3), (planes, in_planes, 1,1)).cuda()
                count4 = counting(pattern_dic[14], 3, 3)
                count5 = counting(pattern_dic[14], 4, 4)
                count6 = counting(pattern_dic[14], 5, 5)
                logger.info(pattern_dic[14])

            logger.info('conv1 reduced rows: if pw==3: %d, pw==4: %d, pw==5: %d'%(count1*in_planes, count2*in_planes, count3*in_planes))
            logger.info('conv2 reduced rows: if pw==3: %d, pw==4: %d, pw==5: %d'%(count4*in_planes, count5*in_planes, count6*in_planes))

            self.bn1 = SwitchBatchNorm2d(self.w_bit, planes)
            self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)
            self.conv1 = Conv2d_Q_mask(self.w_bit, planes, planes, kernel_size=3, pat=pat1, padding=padding, stride=1, bias=False) 
            self.conv2 = Conv2d_Q_mask(self.w_bit, planes, planes, kernel_size=3, pat=pat2, padding=padding, stride=1, bias=False) 



        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                '''
                For CIFAR10 ResNet paper uses option A.
                '''
                self.shortcut = LambdaLayer__(lambda x:
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

        if in_planes == 16 or in_planes == 64:
            image_col, image_row = 32, 32
        elif in_planes == 128:
            image_col, image_row = 16, 16
        elif in_planes == 256:
            image_col, image_row = 8, 8
        
        
        if in_planes != planes:
            logger.info('='*50)
            logger.info('in_planes: %d,  out_plaens: %d'%(planes, planes))
            if in_planes == 16:
                pat1 = torch.tile(torch.tensor(pattern_dic[0]).reshape(3,3), (planes, planes, 1,1)).cuda()
                count1 = counting(pattern_dic[0], 3, 3)
                count2 = counting(pattern_dic[0], 4, 4)
                count3 = counting(pattern_dic[0], 5, 5)
                logger.info(pattern_dic[0])

            elif in_planes == 64:
                pat1 = torch.tile(torch.tensor(pattern_dic[3]).reshape(3,3), (planes, planes, 1,1)).cuda()
                count1 = counting(pattern_dic[3], 3, 3)
                count2 = counting(pattern_dic[3], 4, 4)
                count3 = counting(pattern_dic[3], 5, 5)
                logger.info(pattern_dic[3])

            elif in_planes == 128:
                pat1 = torch.tile(torch.tensor(pattern_dic[6]).reshape(3,3), (planes, planes, 1,1)).cuda()  
                count1 = counting(pattern_dic[6], 3, 3)
                count2 = counting(pattern_dic[6], 4, 4)
                count3 = counting(pattern_dic[6], 5, 5)
                logger.info(pattern_dic[6])

            logger.info('reduced rows: if pw==3: %d, pw==4: %d, pw==5: %d'%(count1*planes, count2*planes, count3*planes))

            self.conv1 = Conv2d_Q_(self.w_bit, in_planes, planes, kernel_size=(3,3), padding=(1,1), stride=stride, bias=False)
            self.bn1 = SwitchBatchNorm2d(self.w_bit, in_planes)
            self.conv2 = Conv2d_Q_mask(self.w_bit, planes, planes, kernel_size=3, pat=pat1, padding=padding, stride=1, bias=False) 
            self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)
            
        else: 

            ################ 패턴 수정
            logger.info('='*50)
            logger.info('in_planes: %d,  out_plaens: %d'%(in_planes, planes))
            if planes == 64:
                pat1 = torch.tile(torch.tensor(pattern_dic[1]).reshape(3,3), (planes, in_planes, 1,1)).cuda()
                count1 = counting(pattern_dic[1], 3, 3)
                count2 = counting(pattern_dic[1], 4, 4)
                count3 = counting(pattern_dic[1], 5, 5)
                logger.info(pattern_dic[1])
                pat2 = torch.tile(torch.tensor(pattern_dic[2]).reshape(3,3), (planes, in_planes, 1,1)).cuda()
                count4 = counting(pattern_dic[2], 3, 3)
                count5 = counting(pattern_dic[2], 4, 4)
                count6 = counting(pattern_dic[2], 5, 5)
                logger.info(pattern_dic[2])

            elif planes == 128:
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

            elif planes == 256:
                pat1 = torch.tile(torch.tensor(pattern_dic[7]).reshape(3,3), (planes, in_planes, 1,1)).cuda()
                count1 = counting(pattern_dic[7], 3, 3)
                count2 = counting(pattern_dic[7], 4, 4)
                count3 = counting(pattern_dic[7], 5, 5)
                logger.info(pattern_dic[7])
                pat2 = torch.tile(torch.tensor(pattern_dic[8]).reshape(3,3), (planes, in_planes, 1,1)).cuda()
                count4 = counting(pattern_dic[8], 3, 3)
                count5 = counting(pattern_dic[8], 4, 4)
                count6 = counting(pattern_dic[8], 5, 5)
                logger.info(pattern_dic[8])

            logger.info('conv1 reduced rows: if pw==3: %d, pw==4: %d, pw==5: %d'%(count1*in_planes, count2*in_planes, count3*in_planes))
            logger.info('conv2 reduced rows: if pw==3: %d, pw==4: %d, pw==5: %d'%(count4*in_planes, count5*in_planes, count6*in_planes))

            self.bn1 = SwitchBatchNorm2d(self.w_bit, in_planes)
            self.bn2 = SwitchBatchNorm2d(self.w_bit, planes)
            self.conv1 = Conv2d_Q_mask(self.w_bit, planes, planes, kernel_size=3, pat=pat1, padding=padding, stride=1, bias=False) 
            self.conv2 = Conv2d_Q_mask(self.w_bit, planes, planes, kernel_size=3, pat=pat2, padding=padding, stride=1, bias=False) 

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
        # print(out)
        # print(out.shape)
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




directory = './log/%s/%s/%s/%s/%d'%(args.model, args.dataset, name, args.seed, args.entry)
if not os.path.isdir(directory):
    os.makedirs(directory)
file_name = directory + '/lr%.4f_wb%d_ab%d_lth%.2f_ar%dxac%d.log'%(args.lr, args.wb, args.ab, args.l_th, args.ar, args.ac)


file_handler = logging.FileHandler(file_name)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

args.workers = 2

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

    if args.dataset == 'cifar10' :
        T = 30
    elif args.dataset =='cifar100':
        T = 60
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T, 1e-4)


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

        if acc > best_acc :
            best_acc = acc
            print('Best accuracy is updated! at epoch %d/%d: %.4f '%(epoch+1, args.epoch, best_acc))
            logger.info('Best accuracy is : %.4f '%(best_acc))

    logger.info('Final best accuracy is : %.4f '%(best_acc))

    

def main():
    train_loader, test_loader = data_loader(args.dir, args.dataset, args.batchsize, args.workers)

    if args.model == 'Wide_ResNet_Q' :
        if args.dataset == 'cifar10' or args.dataset == 'svhn':
            model = Wide_ResNet_Q(args.ab, args.wb, block=Wide_BasicBlock_Q, num_blocks=[2,2,2], scale=args.scale, num_classes=10).cuda()
        elif args.dataset == 'cifar100' :
            model = Wide_ResNet_Q(args.ab, args.wb, block=Wide_BasicBlock_Q, num_blocks=[2,2,2], scale=args.scale, num_classes=100).cuda()
   
    elif args.model == 'ResNet_Q' :
        if args.dataset == 'cifar10' :
            model = ResNet20_Q(args.ab, args.wb, block=BasicBlock_Q, num_blocks=[3,3,3], scale=1, num_classes=10).cuda()

    train_model(model, train_loader, test_loader)

    # input = torch.randn(1, 3, 32, 32).cuda()
    # flops, params = profile(model, inputs=(input,))


if __name__=='__main__':
  main()




