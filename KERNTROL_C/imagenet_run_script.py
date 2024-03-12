import os
import time
import random

print('input the mode for training mode')
mode = int(input())


seed_candidate = [1016, 1106, 2023, 1992, 52398213, 38982158, 37988277, 72975320, 51044796, 72392303]
# original model test
'''
ResNet-20
lr 1e-1 ~ 1e-4
512 x 512
256 x 256

WRN16-4
lr 1e-1 ~ 1e-4
2048 x 1024
1024 x 1024
'''


#########################################
# ResNet-20/cifar10/ab=4/wb=1/T=30/100epochs/lr=1e-1 ~ 1e-4 cosine scheduler / weight_decay = 5e-4
# WRN16-4/cifar100/ab=8/wb=2/T=60/200epochs/lr=1e-1 ~ 1e-4 cosine scheduler / weight_decay = 5e-4
# ResNet-18/imagenet/ab=8/wb=8/multi-step 50, 80, 105 deacy x0.1 /lr =1 e-1 / weight decay = 1e-5

# [52365648, 62292156, 1016]
'''
230216
WRN16-4 
orignal
KERNTROL-M
KERNTROL-C
Pattern-base
'''


if mode == 0 : # [45,60,70] // 80 lr 0.1 // batch 512 // 이걸로 고정
    original = [2]
    epoch = [20] 
    wb = [8]
    ab = [32]
    seed_candidate = [2023]
    model = 'ResNet'
    dataset = ['imagenet']
    GPU = 2
    l5_th = [0.25]
    ar = [2048]
    ac = 1024
    save = 0
    pretrained = 1

elif mode == 1 : # [45,60,70] // 80 lr 0.1 // batch 512 // 이걸로 고정
    original = [2]
    epoch = [20] 
    wb = [8]
    ab = [32]
    seed_candidate = [2023]
    model = 'ResNet'
    dataset = ['imagenet']
    GPU = 1
    l5_th = [0.40]
    ar = [2048]
    ac = 1024
    save = 0
    pretrained = 1

elif mode == 2 : # [45,60,70] // 80 lr 0.1 // batch 512 // 이걸로 고정
    original = [4]
    epoch = [20] 
    wb = [8]
    ab = [32]
    seed_candidate = [2023]
    model = 'ResNet'
    dataset = ['imagenet']
    GPU = 2
    l5_th = [0.45]
    ar = [2048]
    ac = 1024
    save = 0
    pretrained = 1

elif mode == 4 : # [45,60,70] // 80 lr 0.1 // batch 512 // 이걸로 고정
    original = [4]
    epoch = [80] 
    wb = [8]
    ab = [32]
    seed_candidate = [2023]
    model = 'ResNet'
    dataset = ['imagenet']
    GPU = 3
    l5_th = [0]
    ar = [2048]
    ac = 1024
    save = 1
    pretrained = 0

#########################################



for seed in seed_candidate :
    for wbb in wb :
        for abb in ab :
            for dset in dataset:
                for epo in epoch :
                    for l5_thr in l5_th :
                        for arr in ar:
                            for ori in original :
                                os.system( 'python imagenet.py' + ' --dataset ' + str(dset) + ' --model ' +str(model) + ' --original '+ str(ori) 
                                + ' --GPU ' + str(GPU) + ' --epoch ' + str(epo) + ' --ab ' + str(abb) + ' --wb ' + str(wbb) + ' --seed ' + str(seed) + ' --ar ' 
                                + str(arr) + ' --ac ' + str(ac) + ' --l5_th ' + str(l5_thr) + ' --save ' + str(save)+ ' --pretrain ' +str(pretrained))

time.sleep(10)
