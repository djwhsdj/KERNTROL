import os
import time
import random

print('input the mode for training mode')
mode = int(input())




'''
original = MODE
1: Baseline
2: KERNTROL
ab: acitvation bit precision
wb: weight bit precision
l_th: threshold
ar: array row
ac: array column
The model can be changed with 'ResNet-Q, WRN16-4'
'''

if mode == 1 : # 
    original = [2]
    epoch = [100] 
    wb = [1]
    ab = [4]
    seed_candidate = [1106, 1992]
    model = 'ResNet_Q'
    dataset = ['cifar10']
    GPU = 3
    l5_th = [0.15, 0.25, 0.4, 0.45]
    # l4_th = [0]
    ar = [512]
    ac = 256
    save = 0

elif mode == 2 : # 
    original = [4]
    epoch = [100] 
    wb = [1]
    ab = [4]
    seed_candidate = [1106, 1992]
    model = 'ResNet_Q'
    dataset = ['cifar10']
    GPU = 3
    l5_th = [0.15, 0.25, 0.4, 0.45]
    # l4_th = [0.25]
    ar = [512]
    ac = 256
    save = 0

elif mode == 3 : # 
    original = [4]
    epoch = [200] 
    wb = [2]
    ab = [8]
    seed_candidate = [1106, 1992]
    model = 'Wide_ResNet_Q'
    dataset = ['cifar100']
    GPU = 2
    l5_th = [0.15, 0.25, 0.4, 0.45]
    ar = [2048]
    ac = 1024
    save = 0

elif mode == 4 : # 
    original = [1]
    epoch = [100] 
    wb = [1]
    ab = [4]
    seed_candidate = [1106, 1992]
    model = 'ResNet_Q'
    dataset = ['cifar10']
    GPU = 0
    l5_th = [0]
    # l4_th = [0]
    ar = [2048]
    ac = 1024
    save = 1
#########################################



for seed in seed_candidate :
    for wbb in wb :
        for abb in ab :
            for dset in dataset:
                for epo in epoch :
                    for l5_thr in l5_th :
                        for arr in ar:
                            for ori in original :
                                os.system( 'python3 utilization_0729.py' + ' --dataset ' + str(dset) + ' --model ' +str(model) + ' --original '+ str(ori) 
                                + ' --GPU ' + str(GPU) + ' --epoch ' + str(epo) + ' --ab ' + str(abb) + ' --wb ' + str(wbb) + ' --seed ' + str(seed) + ' --ar ' 
                                + str(arr) + ' --ac ' + str(ac) + ' --l5_th ' + str(l5_thr) + ' --save ' + str(save))

time.sleep(10)
