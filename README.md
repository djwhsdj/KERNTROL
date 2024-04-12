# KERNTROL: Kernel Shape Control Toward Ultimate Memory Utilization for In-Memory Convolutional Weight Mapping
---

## Requirements
+ python3.x+
+ Pytorch
+ Numpy

## Usage of KERNTROL-M

### utils.py
* This code includes functions of quantization, counting, mapping algorithms and KERNTROL layer.

### utilization_run_script.py
* This code operates a baseline and our proposed KERNTROL-M.

### utilization_others_script.py
* This code is for operating the network assuming that all convolutional layers have the same parallel window size (5x5, 4x4).

### utilization_pattern_run_script.py
* This code operates the pattern-based pruning methods (PatDNN, Random, PConv) with various entries.

* Note that, for a fair comparison, we assumed that all patterns have the row-wise pattern dimension.


## Usage of KERNTROL-C (in folder KERNTROL-C) 
### This in ResNet-18 with ImageNet dataset
 
### imagenet_run_script.py
* This code operates a baseline and our proposed KERNTROL-M and KERNTROL-C.

### imagenet_run_script_pattern.py
* This code operates the N-entry pattern-based pruning methods.

### utils.py
* This code includes functions of quantization, counting, mapping algorithms and KERNTROL layers.



## Mapping methods

### Im2col (Image to column)
You can read the original pdf [here](https://dl.acm.org/doi/10.1145/2964284.2967243)

### SDK (Shift and Duplicate Kernel)
You can read the original pdf [here](https://ieeexplore.ieee.org/document/9104658)

### Kernel shape control with empty mask (KERNTROL-M)
You can read the original pdf [here](https://ieeexplore.ieee.org/abstract/document/10323749)

### Kernel shape control with compensatory weights (KERNTROL-C)
You can read the original pdf [here](https://ieeexplore.ieee.org/abstract/document/10443813)

