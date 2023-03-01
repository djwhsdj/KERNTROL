import os
import time
import random

print('input the mode for training mode')
mode = int(input())


seed_candidate = [1016, 1106, 2023, 1992, 52398213, 38982158, 37988277, 72975320, 51044796, 72392303]


'''
original
0: pairs all layer
1: 5x5 pw all kerntrol -c
2: 4x4 pw all kerntrol -c
3: 5x5 pw all kerntrol
4: 4x4 pw all kerntrol

'''

# 모든 case에 대해 inpu padding
if mode == 1 : 
    original = [1] 
    optimizer = 'SGD' 
    lr = [0.1]
    epoch = [100] 
    wb = [1]
    ab = [4]
    seed_candidate = [1016, 62292156]
    model = 'ResNet_Q'
    dataset = ['cifar10']
    GPU = 0
    l_th = [0, 0.15, 0.25, 0.4, 0.45]
    ar = [512]
    ac = 256

elif mode == 2 :
    original = [1]
    optimizer = 'SGD' 
    lr = [0.1]
    epoch = [100] 
    wb = [1]
    ab = [4]
    seed_candidate = [1106]
    model = 'ResNet_Q'
    dataset = ['cifar10']
    GPU = 1
    l_th = [0, 0.15, 0.25, 0.4, 0.45]
    ar = [512]
    ac = 256

elif mode == 3 :
    original = [1]
    optimizer = 'SGD' 
    lr = [0.1]
    epoch = [100] 
    wb = [1]
    ab = [4]
    seed_candidate = [1992]
    model = 'ResNet_Q'
    dataset = ['cifar10']
    GPU = 2
    l_th = [0, 0.15, 0.25, 0.4, 0.45]
    ar = [512]
    ac = 256

elif mode == 4 :
    original = [1]
    optimizer = 'SGD' 
    lr = [0.1]
    epoch = [100] 
    wb = [1]
    ab = [4]
    seed_candidate = [52365648]
    model = 'ResNet_Q'
    dataset = ['cifar10']
    GPU = 3
    l_th = [0, 0.15, 0.25, 0.4, 0.45]
    ar = [512]
    ac = 256

#########################################



for seed in seed_candidate :
    for ori in original :
        for wbb in wb :
            for abb in ab :
                for dset in dataset:
                    for lrr in lr :
                        for epo in epoch :
                            for l_thr in l_th :
                                for arr in ar:
                                    os.system( 'python3 utilization_0113_others.py' + ' --dataset ' + str(dset) + ' --optimizer ' + str(optimizer) + ' --model ' +str(model) + ' --original '+ str(ori) 
                                    + ' --GPU ' + str(GPU) + ' --lr ' + str(lrr) + ' --epoch ' + str(epo) + ' --ab ' + str(abb) + ' --wb ' + str(wbb) + ' --seed ' + str(seed) + ' --ar ' 
                                    + str(arr) + ' --ac ' + str(ac) + ' --l_th ' + str(l_thr))

time.sleep(10)
