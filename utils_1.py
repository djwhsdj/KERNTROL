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
# ResNet-20/cifar10/ab=4/wb=1/T=30/100epochs
# WRN16-4/cifar100/ab=8/wb=2/T=60/200epochs

if mode == 1 : # 
    original = [1]
    epoch = [100] 
    wb = [1]
    ab = [4]
    seed_candidate = [1016, 62292156]
    model = 'ResNet_Q'
    dataset = ['cifar10']
    GPU = 1
    u_th = [0]
    l5_th = [0]
    l4_th = [0]
    ar = [2048]
    ac = 1024
    mode = ['same']

elif mode == 2 : # 
    original = [2,4]
    epoch = [202] 
    wb = [2]
    ab = [8]
    seed_candidate = [52365648]
    model = 'Wide_ResNet_Q'
    dataset = ['cifar100']
    GPU = 2
    u_th = [0]
    l5_th = [0, 0.15, 0.25, 0.4, 0.5]
    l4_th = [0.25]
    ar = [2048]
    ac = 1024
    mode = ['same']

elif mode == 3 : # 
    original = [2,4]
    epoch = [202] 
    wb = [2]
    ab = [8]
    seed_candidate = [52365648]
    model = 'Wide_ResNet_Q'
    dataset = ['cifar100']
    GPU = 3
    u_th = [0]
    l5_th = [0, 0.15, 0.25, 0.4, 0.5]
    l4_th = [0.5]
    ar = [2048]
    ac = 1024
    mode = ['same']
#########################################



for seed in seed_candidate :
    for wbb in wb :
        for abb in ab :
            for dset in dataset:
                for epo in epoch :
                    for u_thr in u_th :
                        for l4_thr in l4_th :
                            for l5_thr in l5_th :
                                for arr in ar:
                                    for ori in original :
                                        for mod in mode :
                                            os.system( 'python3 utilization_0729.py' + ' --dataset ' + str(dset) + ' --model ' +str(model) + ' --original '+ str(ori) 
                                            + ' --GPU ' + str(GPU) + ' --epoch ' + str(epo) + ' --ab ' + str(abb) + ' --wb ' + str(wbb) + ' --seed ' + str(seed) + ' --ar ' 
                                            + str(arr) + ' --ac ' + str(ac) + ' --u_th ' + str(u_thr) + ' --l5_th ' + str(l5_thr)+ ' --l4_th ' + str(l4_thr) + ' --mode ' + str(mod))

time.sleep(10)

