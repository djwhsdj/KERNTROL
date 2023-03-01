import os
import time
import random

print('input the mode for training mode')
mode = int(input())


seed_candidate = [1016, 1106, 2023, 1992, 52398213, 38982158, 37988277, 72975320, 51044796, 72392303]

'''
if args.original == 0:
    name = 'Random' # 6 entry

elif args.original == 1:
    name = 'PatDNN'

elif args.original == 2:
    name = 'Pconv'
'''

if mode == 1 : # 
    original = [0,1]
    epoch = [100] 
    wb = [1]
    ab = [4]
    seed_candidate = [1106, 1992]
    model = 'ResNet_Q'
    dataset = ['cifar10']
    GPU = 0
    entries = [4,5,6,7,8]

elif mode == 2 : # 
    original = [2]
    epoch = [100] 
    wb = [1]
    ab = [4]
    seed_candidate = [1106, 1992]
    model = 'ResNet_Q'
    dataset = ['cifar10']
    GPU = 0
    entries = [4]

elif mode == 5 : # 
    original = [0]
    epoch = [200] 
    wb = [2]
    ab = [8]
    seed_candidate = [1106, 1992]
    model = 'Wide_ResNet_Q'
    dataset = ['cifar100']
    GPU = 3
    entries = [4,5,6,7,8]
#########################################



for seed in seed_candidate :
    for wbb in wb :
        for abb in ab :
            for dset in dataset:
                for epo in epoch :
                    for ori in original :
                        for en in entries:
                            os.system( 'python3 utilization_pattern.py' + ' --dataset ' + str(dset) + ' --model ' +str(model) + ' --original '+ str(ori) 
                            + ' --GPU ' + str(GPU)  + ' --epoch ' + str(epo) + ' --ab ' + str(abb) + ' --wb ' + str(wbb) + ' --seed ' + str(seed) + ' --entry ' + str(en))

time.sleep(10)
