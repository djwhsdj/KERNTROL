# Kernel Shape Control for Row-Efficient Convolution on Processing-In-Memory Arrays
---
## Abstract
Processing-in-memory (PIM) architectures have been highlighted as one of the most viable options to meet the growing need for faster and more power-efficient computation. Paired with a convolutional weight mapping scheme, PIM arrays can accelerate various deep convolutional neural networks (CNNs) and the applications that adopt them. Recently, shift and duplicate kernel (SDK) convolutional weight mapping scheme was proposed, achieving up to 50% throughput improvement over the prior arts. However, the traditional pattern-based pruning methods, which were adopted for row-skipping and computing cycle reduction, are not optimal for the latest SDK mapping due to structural irregularity caused by the shifted and duplicated kernels. To address this issue, we propose kernel shape control (KERNTROL), a method in which kernel shapes are controlled/omitted depending on their mapped columns with the purpose of fostering a structural regularity that is favorable in achieving a high row-skipping ratio and model accuracy. In comparison to the latest pattern-based pruning approaches, KERNTROL achieves up to 36.4% improvement in the compression rate, and 38.6% in array utilization with maintaining the original model accuracy.

## Requirements
+ python3.x+
+ Pytorch
+ Numpy

## Usage

### utils.py
* This code includes functions of quantization, couting, mapping algorithm and KERNTROL.

### utilization_run_script.py
* This code operates a baseline and our proposed ones (KERNTROL, KERNTROL).

### utilization_others_script.py
* This code is for operating the network assuming that all convolutional layer have the same parallel window size (5x5, 4x4).

### utilization_pattern_run_script.py
* This code operates the pattern-based pruning methods (PatDNN, Random, PConv) with various entries.

* Note that, for a fair comparison, we assumed that all pattern have the block-wise pattern dimension.


## Mapping methods

### Im2col (Image to column)
You can read the original pdf [here](https://dl.acm.org/doi/10.1145/2964284.2967243)

Each kernel with size KxKxIC (where K is kernel, IC is input channel) is unrolled into the column. A kernel-sized window in an input feature map (IFM) is convolved with the kernel.


### SDK (Shift and Duplicate Kernel)
You can read the original pdf [here](https://ieeexplore.ieee.org/document/9104658)

This mapping computes multiple windows instead of single window simultaneously in each cycle. To reuses the input data, this method forms the parallel window that is a set of windows. Thus, it obtains multiple output feature maps (OFMs) by utilizing the PIM array.
