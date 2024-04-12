import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, SVHN, MNIST
import torchvision.datasets as datasets

from re import L
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.modules.batchnorm import BatchNorm2d

# import scipy
import time



def counting (mask, pwr, pwh) :
    mask = np.array(mask)
    mask = mask.reshape(3,3)
    kr, kh = 3,3

    cnt = 0

    kernel = []
    for i in range(kr*kh) :
        kernel.append(i)
    
    pw = []
    for j in range(pwr*pwh) :
        pw.append([])
        
    for a in range(pwh-kh+1) :
        for b in range(pwr-kr+1) :
            for c in range(len(kernel)) :
                divider = c // 3
                residue = c % 3
                pw_idx = (divider+a)*pwr+(residue+b)
                pw[pw_idx].append(kernel[c])
    
    zero_list = []
    for j in range(kr) :
        for k in range(kh) :
            cal = mask[j, k].sum()
            if cal == 0 :
                idx = j*kr + k
                zero_list.append(idx)

    for q in range(len(pw)) :
        for j in zero_list :
            if j in pw[q] :
                pw[q].remove(j)

    for m in pw :
        if m == [] :
            cnt+=1


    return cnt



def SDK (image_col, image_row, filter_col, filter_row, in_channel, out_channel, \
                    array_row, array_col) :
    
    row_vector = filter_row * filter_col * in_channel
    col_vector = out_channel
    
    used_row = math.ceil(row_vector/array_row)
    used_col = math.ceil(col_vector/array_col)
    
    new_array_row = array_row * used_row
    new_array_col = array_col * used_col

    cycle = []
    w = [] # pw 크기
    w.append(filter_row*filter_col)
    cycle.append(used_row*used_col*(image_row-filter_row+1)*(image_col-filter_col+1))
    
    i=0
    while True :
        i += 1
        pw_row = filter_row + i - 1 
        pw_col = filter_col + i - 1
        pw = pw_row * pw_col
        if pw*in_channel <= new_array_row and i * i * out_channel <= new_array_col :
            parallel_window_row = math.ceil((image_row - (filter_row + i) + 1)/i) + 1
            parallel_window_col = math.ceil((image_col - (filter_col + i) + 1)/i) + 1
            
            if parallel_window_row * parallel_window_row * used_row * used_col <= cycle[0] :
                del cycle[0]
                del w[0]
                cycle.append(parallel_window_row * parallel_window_col * used_row * used_col)
                w.append(pw)
        else :
            break
        
    return  math.sqrt(w[0])
                        

def SDK_a (image_col, image_row, filter_col, filter_row, in_channel, out_channel, \
                    array_row, array_col) :
    

    # initialize
    cycle = [100000]
    w = [filter_row*filter_col] # pw 크기
    num_windows, ar, ac = [0], [0], [0]
    
    i=0
    while True :
        i += 1
        pw_row = filter_row + i - 1 
        pw_col = filter_col + i - 1

        parallel_window_row = math.ceil((image_row - (filter_row + i) + 1)/i) + 1
        parallel_window_col = math.ceil((image_col - (filter_col + i) + 1)/i) + 1

        ARC = math.ceil((pw_row*pw_col*in_channel)/array_row)
        ACC = math.ceil((i*i*out_channel)/array_col)
        if parallel_window_row * parallel_window_row * ARC * ACC <= cycle[0] :
            del cycle[0]
            del w[0]
            del ar[0]
            del ac[0]
            del num_windows[0]
            
            num_w = parallel_window_row * parallel_window_col
            num_windows.append(num_w)
            ar.append(ARC)
            ac.append(ACC)
            cycle.append(num_w * ARC * ACC)
            w.append(pw_row*pw_col)


        if pw_row >= image_row :
            break
    
    return int(math.sqrt(w[0]))


class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification.
    Refer to https://github.com/JiahuiYu/slimmable_networks/blob/master/utils/loss_ops.py
    """
    def forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        cross_entropy_loss = cross_entropy_loss.mean()
        return cross_entropy_loss
        
class Activate(nn.Module):
    def __init__(self, a_bit, quantize=True):
        super(Activate, self).__init__()
        self.abit = a_bit
        # Since ReLU is not differentible at x=0, changed to GELU
        self.acti = nn.ReLU(inplace=True)
        self.quantize = quantize
        if self.quantize:
            self.quan = activation_quantize_fn(self.abit)

    def forward(self, x):
        if self.abit == 32:
            x = self.acti(x)
        else:
            x = torch.clamp(x, 0.0, 1.0)
        if self.quantize:
            x = self.quan(x)
        return x

class activation_quantize_fn(nn.Module):
    def __init__(self, a_bit):
        super(activation_quantize_fn, self).__init__()
        self.abit = a_bit
        assert self.abit <= 8 or self.abit == 32

    def forward(self, x):
        if self.abit == 32:
            activation_q = x
        else:
            activation_q = qfn.apply(x, self.abit)
        return activation_q


class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        n = float(2**k - 1)
        out = torch.round(input * n) / n
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit):
        super(weight_quantize_fn, self).__init__()
        self.wbit = w_bit
        assert self.wbit <= 8 or self.wbit == 32

    def forward(self, x):
        if self.wbit == 32:
            E = torch.mean(torch.abs(x)).detach()
            weight = torch.tanh(x)
            weight = weight / torch.max(torch.abs(weight))
            weight_q = weight * E
        else:
            E = torch.mean(torch.abs(x)).detach()
            weight = torch.tanh(x)
            weight = weight / 2 / torch.max(torch.abs(weight)) + 0.5
            weight_q = 2 * qfn.apply(weight, self.wbit) - 1
            weight_q = weight_q * E
        return weight_q



class SwitchBatchNorm2d(nn.Module):
    """Adapted from https://github.com/JiahuiYu/slimmable_networks
    """
    def __init__(self, w_bit, num_features):
        super(SwitchBatchNorm2d, self).__init__()
        self.w_bit = w_bit
        self.bn_dict = nn.ModuleDict()
        # for i in self.bit_list:
        #     self.bn_dict[str(i)] = nn.BatchNorm2d(num_features)
        self.bn_dict[str(w_bit)] = nn.BatchNorm2d(num_features, eps=1e-4)

        self.abit = self.w_bit
        self.wbit = self.w_bit
        if self.abit != self.wbit:
            raise ValueError('Currenty only support same activation and weight bit width!')

    def forward(self, x):
        x = self.bn_dict[str(self.abit)](x)
        return x

class SwitchBatchNorm2d_(SwitchBatchNorm2d) : ## 만든거
    def __init__(self, w_bit, num_features) :
        super(SwitchBatchNorm2d_, self).__init__(num_features=num_features, w_bit=w_bit)
        self.w_bit = w_bit      
        # return SwitchBatchNorm2d_
    


def batchnorm2d_fn(w_bit):
    class SwitchBatchNorm2d_(SwitchBatchNorm2d):
        def __init__(self, num_features, w_bit=w_bit):
            super(SwitchBatchNorm2d_, self).__init__(num_features=num_features, w_bit=w_bit)

    return SwitchBatchNorm2d_


class Conv2d_Q(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2d_Q, self).__init__(*args, **kwargs)


class Conv2d_Q_(Conv2d_Q): ## original
    def __init__(self, w_bit, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                    bias=False):
        super(Conv2d_Q_, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.w_bit = w_bit
        self.quantize_fn = weight_quantize_fn(self.w_bit)

    def forward(self, input):
        weight_q = self.quantize_fn(self.weight) 
        return F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)



class Conv2d_Q_mask(Conv2d_Q): ## original
    def __init__(self, w_bit, in_channels, out_channels, kernel_size, pat, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv2d_Q_mask, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.w_bit = w_bit
        self.quantize_fn = weight_quantize_fn(self.w_bit)
        self.pat = pat

    def forward(self, input):
        tmp_weight = self.weight * self.pat
        weight_q = self.quantize_fn(tmp_weight) 
        weight_q = weight_q * self.pat
        return F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)







class weight_masking(nn.Module):
    def __init__(self, line):
        super(weight_masking, self).__init__()

        self.line = line

    def forward(self, x):
        weight = x.weight.detach().cpu()
        layer_shape = weight.size()
        array = np.ones(layer_shape)
        if self.line == 1 :
            array[:,:,0,:] = 0
        else :
            array[:,:,2,:] = 0

        x.weight.data = np.multiply(weight, array)

        return x.float().cuda()

'''
# Option 1: write directly to data
wfx.weight.data = wfx.weight * mask_use

# Option 2: convert result to nn.Parameter and write to weight
wfx.weight = nn.Parameter(wfx.weight * mask_use)
'''

def pattern_gen(pw, k, thresh):

    idx = 0
    mask = torch.zeros(pw, pw).cuda()
    for i in range(pw-k+1) : # col 
        for j in range(pw-k+1) : # row
            idx += 1
            mask[i:i+k, j:j+k]+=1
    
    mask = mask/idx
    mask = torch.where(mask <= thresh, 0, 1)

    return mask


def pattern_gen_v1(MK, pw, k, lo_thresh, up_thresh, in_planes, planes, mode='bound'):

    idx = 0
    mask = torch.zeros(pw, pw, dtype=torch.double)
    for i in range(pw-k+1) : # col 
        for j in range(pw-k+1) : # row
            idx += 1
            mask[i:i+k, j:j+k]+=1
    
    mask = mask/idx
    mask = torch.where(mask <= lo_thresh, 0., mask)
    mask_bound = torch.where(mask <= lo_thresh, 0., 1.).type(torch.int)

    if mode == 'bound':
        mask = torch.where(mask >= up_thresh, 1., mask)
        pat = torch.where(mask <1, 0., mask)
    elif mode == 'inter':
        mask1 = torch.where(mask <= up_thresh, 1., mask)
        mask2 = torch.where(mask > lo_thresh, 1., mask)
        mask = mask1 * mask2
        pat = torch.where(mask <1, 0., mask)
    
    elif mode == 'same':
        mask = mask
        pat = torch.where(mask>0, 1., 0.)

    mask = mask.type(torch.int)
    # 마스크 생성

    kern = torch.ones(k, k)

    if MK == 5 :
      a = 1
    elif MK == 7 :
      a = 2

    pad = (a,a,a,a)
    kern = F.pad(kern, pad, "constant", 0)

    st = pw - k
    list1 = []
    a = torch.Tensor([])
    for idx_height in range(pw-k+1) : # height 
        for idx_width in range(pw-k+1) : # width
            if mode == 'inter':
                b = kern[st-idx_height:st-idx_height+pw, st-idx_width:st-idx_width+pw] + pat
                b = torch.where(b > 1., 1., b)
                b = b * mask_bound
                b = torch.where(b >= 1., 1., 0.)
            
            elif mode == 'bound':
                b = kern[st-idx_height:st-idx_height+pw, st-idx_width:st-idx_width+pw] + pat
                b = b * mask_bound
                b = torch.floor(b).type(torch.int)
                b = torch.where(b >= 1, 1, 0)

            elif mode == 'same':
                b = pat

            b =  b.type(torch.float)
            x = torch.tile(b, (planes, in_planes, 1, 1)).cuda()
            # print(x)
            list1.append(x)
        
    return torch.stack(list1, dim=0)




def pattern_gen_v2(MK, pw, k, pat, planes, mode='bound'):

    kern = torch.ones(k, k)

    if MK == 5 :
      a = 1
    elif MK == 7 :
      a = 2

    pad = (a,a,a,a)
    mask = F.pad(kern, pad, "constant", 0)

    st = pw - k
    list1 = []
    a = torch.Tensor([])
    for idx_height in range(pw-k+1) : # height 
        for idx_width in range(pw-k+1) : # width
            if mode == 'bound':
              b = mask[st-idx_height:st-idx_height+pw, st-idx_width:st-idx_width+pw] + pat
              b = torch.where(b >= 1, 1, b)
            elif mode == 'inter' or mode == 'same':
              b = pat
              
            x = torch.tile(b, (planes, planes, 1, 1)).cuda()
            list1.append(x)
        
    return torch.stack(list1, dim=0)



def pattern_gen(pw, k, thresh):

    idx = 0
    mask = torch.zeros(pw, pw).cuda()
    for i in range(pw-k+1) : # col 
        for j in range(pw-k+1) : # row
            idx += 1
            mask[i:i+k, j:j+k]+=1
    
    mask = mask/idx
    mask = torch.where(mask <= thresh, 0, 1)

    return mask


def pattern_gen_sdk(MK, pw, k, pat, in_planes, planes):

    kern = torch.ones(k, k)

    if MK == 5 :
      a = 1
    elif MK == 7 :
      a = 2

    pad = (a,a,a,a)
    mask = F.pad(kern, pad, "constant", 0).cuda()

    st = pw - k
    list1 = []
    a = torch.Tensor([])
    for idx_height in range(pw-k+1) : # height 
        for idx_width in range(pw-k+1) : # width
            # print(pat)
            b = mask[st-idx_height:st-idx_height+pw, st-idx_width:st-idx_width+pw] * pat
            x = torch.tile(b, (planes, in_planes, 1, 1)).cuda()
            list1.append(x)
            
    return torch.stack(list1, dim=0)


    


def idx_gen(pw, k):

    size = (pw - k + 1)^2
    idx = []
    for i in range(size):
        idx.append(i)

    return idx



def slicing_conv(idx, windows, pw, weight, pat) :
    height = int(idx // windows) # 몫
    width = int(idx % windows) # 나머지

    slicing = pw-3
    # print(pat.shape)
    # print(weight[:,:,slicing-height:slicing-height+pw, slicing-width:slicing-width+pw].shape)
    ad_weight = weight[:,:,slicing-height:slicing-height+pw, slicing-width:slicing-width+pw] * pat
    # print(ad_weight)


    return ad_weight


class SwitchedConv2d_update_ours(Conv2d_Q_) :
    def __init__(self, w_bit, in_channels, out_channels, kernel_size, pat, pw, lth, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(SwitchedConv2d_update_ours, self).__init__(w_bit, in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.w_bit = w_bit
        self.pat = pat
        self.lth = lth
        self.pw = pw
        self.quantize_fn = weight_quantize_fn(self.w_bit)
        self.idx = (self.pw - 3 + 1)**2 # 3 is kernel size



    def forward(self, input) :
        nofwindows = int(self.idx)
        windows = int(math.sqrt(nofwindows))
        s = input.size()

        if self.pw == 5 :
            if s[3] == 32 or s[3] == 8: 
                p2d = (0, 1, 0, 1)
                input = F.pad(input, p2d, "constant", 0) # effectively zero padding, 
            elif s[3] == 16:
                p2d = (0, 2, 0, 2)
                input = F.pad(input, p2d, "constant", 0) # effectively zero padding, 
            
            
        
        weight_q = self.quantize_fn(self.weight)
        if self.pw == 4 and self.lth == 0.5 :
            weight_q[:,:,1:4,1:4] = self.quantize_fn(self.weight[:,:,1:4, 1:4])
        elif self.pw == 5 and self.lth >= 0.4 :
            weight_q[:,:,1:6,1:6] = self.quantize_fn(self.weight[:,:,1:6, 1:6])


        if nofwindows == 4 :
            # start = time.time()
            ad_weight1 = slicing_conv(0, windows, self.pw, weight_q, self.pat[0])
            mask1_out1 = F.conv2d(input, ad_weight1, self.bias, self.stride, self.padding, self.dilation, self.groups)

            ad_weight2 = slicing_conv(1, windows, self.pw, weight_q, self.pat[1])
            mask1_out2 = F.conv2d(input, ad_weight2, self.bias, self.stride, self.padding, self.dilation, self.groups)

            ad_weight3 = slicing_conv(2, windows, self.pw, weight_q, self.pat[2])
            mask1_out3 = F.conv2d(input, ad_weight3, self.bias, self.stride, self.padding, self.dilation, self.groups)

            ad_weight4 = slicing_conv(3, windows, self.pw, weight_q, self.pat[3])
            mask1_out4 = F.conv2d(input, ad_weight4, self.bias, self.stride, self.padding, self.dilation, self.groups)
            
            a = torch.stack([mask1_out1, mask1_out2, mask1_out3, mask1_out4], dim=-1).reshape(-1,s[1],s[2]//2,s[2]//2,2,2).permute(0,1,2,4,3,5).reshape(-1, s[1], s[2], s[3])

            return a 

        elif nofwindows == 9 :
            ad_weight1 = slicing_conv(0, windows, self.pw, weight_q, self.pat[0])
            mask1_out1 = F.conv2d(input, ad_weight1, self.bias, self.stride, 1, self.dilation, self.groups)
            ad_weight2 = slicing_conv(1, windows, self.pw, weight_q, self.pat[1])
            mask1_out2 = F.conv2d(input, ad_weight2, self.bias, self.stride, 1, self.dilation, self.groups)
            ad_weight3 = slicing_conv(2, windows, self.pw, weight_q, self.pat[2])
            mask1_out3 = F.conv2d(input, ad_weight3, self.bias, self.stride, 1, self.dilation, self.groups)


            ad_weight4 = slicing_conv(3, windows, self.pw, weight_q, self.pat[3])
            mask1_out4 = F.conv2d(input, ad_weight4, self.bias, self.stride, 1, self.dilation, self.groups)

            ad_weight5 = slicing_conv(4, windows, self.pw, weight_q, self.pat[4])
            mask1_out5 = F.conv2d(input, ad_weight5, self.bias, self.stride, 1, self.dilation, self.groups)

            ad_weight6 = slicing_conv(5, windows, self.pw, weight_q, self.pat[5])
            mask1_out6 = F.conv2d(input, ad_weight6, self.bias, self.stride, 1, self.dilation, self.groups)
            
            ad_weight7 = slicing_conv(6, windows, self.pw, weight_q, self.pat[6])
            mask1_out7 = F.conv2d(input, ad_weight7, self.bias, self.stride, 1, self.dilation, self.groups)

            ad_weight8 = slicing_conv(7, windows, self.pw, weight_q, self.pat[7])
            mask1_out8 = F.conv2d(input, ad_weight8, self.bias, self.stride, 1, self.dilation, self.groups)

            ad_weight9 = slicing_conv(8, windows, self.pw, weight_q, self.pat[8])
            mask1_out9 = F.conv2d(input, ad_weight9, self.bias, self.stride, 1, self.dilation, self.groups)
            
            a = torch.stack([mask1_out1, mask1_out2, mask1_out3, mask1_out4, mask1_out5, mask1_out6, mask1_out7, mask1_out8, mask1_out9], dim=-1).reshape(-1, s[1], mask1_out1.size(2), mask1_out1.size(3), 3, 3).permute(0,1,2,4,3,5).reshape(-1, s[1], mask1_out1.size(2)*3, mask1_out1.size(3)*3)

            if a.shape[3] == 54:
                p1d = (1, 1, 1, 1)
                a = F.pad(a, p1d, "constant", 0) # effectively zero padding, 

            
            if a.shape[3] == 33 :
                a = a[:,:,:32,:32]
            elif a.shape[3] == 18 :
                a = a[:,:,:16,:16]
            elif a.shape[3] == 9:
                a = a = a[:,:,:8,:8]


            return a


class SwitchedConv2d_update_sdk(Conv2d_Q_) :
    def __init__(self, w_bit, in_channels, out_channels, kernel_size, pat, pw, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(SwitchedConv2d_update_sdk, self).__init__(w_bit, in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.w_bit = w_bit
        self.pat = pat
        self.pw = pw
        self.quantize_fn = weight_quantize_fn(self.w_bit)
        self.idx = (self.pw - 3 + 1)**2 # 3 is kernel size


    def forward(self, input) :
        nofwindows = int(self.idx)
        windows = int(math.sqrt(nofwindows))
        s = input.size()

        if self.pw == 5 :
            if input.size()[3] == 32 or input.size()[3] == 8: 
                p2d = (0, 1, 0, 1)
                input = F.pad(input, p2d, "constant", 0) # effectively zero padding, 
            elif input.size()[3] == 16:
                p2d = (0, 2, 0, 2)
                input = F.pad(input, p2d, "constant", 0) # effectively zero padding, 
            


        weight_q = self.quantize_fn(self.weight)
        if self.pw == 5 :
            weight_q[:,:,2:5,2:5] = self.quantize_fn(self.weight[:,:,2:5, 2:5])
        elif self.pw == 4 :
            weight_q[:,:,1:4,1:4] = self.quantize_fn(self.weight[:,:,1:4, 1:4])

        if nofwindows == 4 :
            ad_weight1 = slicing_conv(0, windows, self.pw, weight_q, self.pat[0])
            mask1_out1 = F.conv2d(input, ad_weight1, self.bias, self.stride, self.padding, self.dilation, self.groups)
            
            ad_weight2 = slicing_conv(1, windows, self.pw, weight_q, self.pat[1])
            mask1_out2 = F.conv2d(input, ad_weight2, self.bias, self.stride, self.padding, self.dilation, self.groups)



            ad_weight3 = slicing_conv(2, windows, self.pw, weight_q, self.pat[2])
            mask1_out3 = F.conv2d(input, ad_weight3, self.bias, self.stride, self.padding, self.dilation, self.groups)

            ad_weight4 = slicing_conv(3, windows, self.pw, weight_q, self.pat[3])
            mask1_out4 = F.conv2d(input, ad_weight4, self.bias, self.stride, self.padding, self.dilation, self.groups)
            
            a = torch.stack([mask1_out1, mask1_out2, mask1_out3, mask1_out4], dim=-1).reshape(-1,s[1],s[2]//2,s[2]//2,2,2).permute(0,1,2,4,3,5).reshape(-1, s[1], s[2], s[3])


            return a 

        elif nofwindows == 9 :
            ad_weight1 = slicing_conv(0, windows, self.pw, weight_q, self.pat[0])
            mask1_out1 = F.conv2d(input, ad_weight1, self.bias, 3, 1, self.dilation, self.groups)

            ad_weight2 = slicing_conv(1, windows, self.pw, weight_q, self.pat[1])
            mask1_out2 = F.conv2d(input, ad_weight2, self.bias, 3, 1, self.dilation, self.groups)

            ad_weight3 = slicing_conv(2, windows, self.pw, weight_q, self.pat[2])
            mask1_out3 = F.conv2d(input, ad_weight3, self.bias, 3, 1, self.dilation, self.groups)


            ad_weight4 = slicing_conv(3, windows, self.pw, weight_q, self.pat[3])
            mask1_out4 = F.conv2d(input, ad_weight4, self.bias, 3, 1, self.dilation, self.groups)

            ad_weight5 = slicing_conv(4, windows, self.pw, weight_q, self.pat[4])
            mask1_out5 = F.conv2d(input, ad_weight5, self.bias, 3, 1, self.dilation, self.groups)

            ad_weight6 = slicing_conv(5, windows, self.pw, weight_q, self.pat[5])
            mask1_out6 = F.conv2d(input, ad_weight6, self.bias, 3, 1, self.dilation, self.groups)
            

            ad_weight7 = slicing_conv(6, windows, self.pw, weight_q, self.pat[6])
            mask1_out7 = F.conv2d(input, ad_weight7, self.bias, 3, 1, self.dilation, self.groups)

            ad_weight8 = slicing_conv(7, windows, self.pw, weight_q, self.pat[7])
            mask1_out8 = F.conv2d(input, ad_weight8, self.bias, 3, 1, self.dilation, self.groups)

            ad_weight9 = slicing_conv(8, windows, self.pw, weight_q, self.pat[8])
            mask1_out9 = F.conv2d(input, ad_weight9, self.bias, 3, 1, self.dilation, self.groups)


            a = torch.stack([mask1_out1, mask1_out2, mask1_out3, mask1_out4, mask1_out5, mask1_out6, mask1_out7, mask1_out8, mask1_out9], dim=-1).reshape(-1, s[1], mask1_out1.size(2), mask1_out1.size(3), 3, 3).permute(0,1,2,4,3,5).reshape(-1, s[1], mask1_out1.size(2)*3, mask1_out1.size(3)*3)
            if a.shape[3] == 54:
                p1d = (1, 1, 1, 1)
                a = F.pad(a, p1d, "constant", 0) # effectively zero padding, 

            if a.shape[3] == 33 :
                a = a[:,:,:32,:32]
            elif a.shape[3] == 18 :
                a = a[:,:,:16,:16]
                # print(a.shape)
            elif a.shape[3] == 9:
                a = a = a[:,:,:8,:8]

            return a


class Linear_Q(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(Linear_Q, self).__init__(*args, **kwargs)

class Linear_Q_(Linear_Q): ## 만든거
    def __init__(self, w_bit, in_features, out_features, bias=True):
        super(Linear_Q_, self).__init__(in_features, out_features, bias=bias)
        self.w_bit = w_bit
        self.quantize_fn = weight_quantize_fn(self.w_bit)

    def forward(self, input, order=None):
        weight_q = self.quantize_fn(self.weight)
        return F.linear(input, weight_q, self.bias)




def data_loader(dir, dataset, batch_size, workers):
    if dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_set = datasets.CIFAR10(root=dir, train=True, download=True, transform=train_transform)
        val_set = datasets.CIFAR10(root=dir, train=False, download=False, transform=val_transform)


    elif dataset == 'svhn' :
        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        val_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        train_set = datasets.SVHN(root=dir, split='train', download=True, transform=train_transform)
        val_set = datasets.SVHN(root=dir, split='test', download=False, transform=val_transform)

    elif dataset == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])
        train_set = datasets.CIFAR100(root=dir, train=True, download=True, transform=train_transform)
        val_set = datasets.CIFAR100(root=dir, train=False, download=False, transform=val_transform)


        
    elif dataset == 'imagenet':
        # traindir = os.path.join(dir, 'ImageNet2012/train')
        # valdir = os.path.join(dir, 'ImageNet2012/val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        train_set = datasets.ImageFolder(os.path.join(dir, 'train'), transform=train_transform)
        val_set = datasets.ImageFolder(os.path.join(dir, 'val'), transform=val_transform)



    elif dataset == 'mnist':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        train_set = datasets.MNIST(root=dir, train=True, download=True, transform=train_transform)
        val_set = datasets.MNIST(root=dir, train=False, download=False, transform=val_transform)
           
    else:
        assert False, 'No Such Dataset'
        
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    return train_loader, val_loader


class SwitchedConv2d_update_test(Conv2d_Q_) :
    def __init__(self, w_bit, in_channels, out_channels, kernel_size, pat, pw, lth, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(SwitchedConv2d_update_test, self).__init__(w_bit, in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.pat = pat
        self.lth = lth
        self.pw = pw
        self.idx = (self.pw - 3 + 1)**2 # 3 is kernel size



    def forward(self, input) :
        nofwindows = int(self.idx)
        windows = int(math.sqrt(nofwindows))
        s = input.size()
        
        weight_q = self.weight

        if nofwindows == 4 :
            ad_weight1 = slicing_conv(0, windows, self.pw, weight_q, self.pat[0])
            mask1_out1 = F.conv2d(input, ad_weight1, self.bias, self.stride, self.padding, self.dilation, self.groups)

            ad_weight2 = slicing_conv(1, windows, self.pw, weight_q, self.pat[1])
            mask1_out2 = F.conv2d(input, ad_weight2, self.bias, self.stride, self.padding, self.dilation, self.groups)

            ad_weight3 = slicing_conv(2, windows, self.pw, weight_q, self.pat[2])
            mask1_out3 = F.conv2d(input, ad_weight3, self.bias, self.stride, self.padding, self.dilation, self.groups)

            ad_weight4 = slicing_conv(3, windows, self.pw, weight_q, self.pat[3])
            mask1_out4 = F.conv2d(input, ad_weight4, self.bias, self.stride, self.padding, self.dilation, self.groups)
            
            a = torch.stack([mask1_out1, mask1_out2, mask1_out3, mask1_out4], dim=-1).reshape(-1,s[1],s[2]//2,s[2]//2,2,2).permute(0,1,2,4,3,5).reshape(-1, s[1], s[2], s[3])

            return a 

        
