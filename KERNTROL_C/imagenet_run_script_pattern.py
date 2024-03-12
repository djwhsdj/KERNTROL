import os
import time
import random

print('input the mode for training mode')
mode = int(input())


# seed_candidate = [1016, 1106, 2023, 1992, 52398213, 38982158, 37988277, 72975320, 51044796, 72392303]

'''
if args.original == 0:
    name = 'Random' # 6 entry

elif args.original == 1:
    name = 'PatDNN'

elif args.original == 2:
    name = 'Pconv'
'''

if mode == 0 : # 
    original = [0]
    epoch = [20] 
    wb = [8]
    ab = [32]
    seed_candidate = [2023]
    model = 'ResNet18_Q'
    dataset = ['imagenet']
    GPU = 1
    entries = [6]

elif mode == 1 : # 
    original = [0]
    epoch = [20] 
    wb = [8]
    ab = [32]
    seed_candidate = [2023]
    model = 'ResNet18_Q'
    dataset = ['imagenet']
    GPU = 2
    entries = [4]

#########################################



for seed in seed_candidate :
    for wbb in wb :
        for abb in ab :
            for dset in dataset:
                for epo in epoch :
                    for ori in original :
                        for en in entries:
                            os.system( 'python imagenet_pattern.py' + ' --dataset ' + str(dset) + ' --model ' +str(model) + ' --original '+ str(ori) 
                            + ' --GPU ' + str(GPU)  + ' --epoch ' + str(epo) + ' --ab ' + str(abb) + ' --wb ' + str(wbb) + ' --seed ' + str(seed) + ' --entry ' + str(en))

time.sleep(10)
