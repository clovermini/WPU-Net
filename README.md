# WPU-Net
WPU-Net: Boundary learning by using weighted propagation in convolution network.
<!-- This is the pytorch implementation of algorithm in paper https://arxiv.org/pdf/1905.09226.pdf. -->

## Abstract
Deep learning has driven great progress in natural and biological image processing. However, in materials science and engineering, there are often some flaws and indistinctions in material microscopic images induced from complex sample preparation, even due to the material itself, hindering the detection of target objects. In this work, we propose WPU-net that redesign the architecture and weighted loss of U-Net to force the network to integrate information from adjacent slices and pay more attention to the topology in this boundary detection task. Experiments demonstrate that the proposed method achieves promising performance and outperforms state-of-the-art methods. Moreover, we present a materials microscopic image dataset with the goal of advancing the state-of-the-art in image processing for materials sciences.

## Environment

    python 3.6
    pytorch 1.0
    gala (for evaluation)

gala is installed according to https://github.com/janelia-flyem/gala.


## DataSet and Running

We opened up the material dataset we used in paper experiment. Hoping more and more researchers can participate in image processing of material science, promoting the development of materials science. You can download it from https://github.com/Keep-Passion/pure_iron_grain_data_sets.

Usage Demo:

    # generate datasets
    python data_generator.py
    
    # inference
    python main.py

    # train WPU-Net
    python segmentation/trainer.py --input="<path to your dataset>" --bs=24 --loss="<abw/cbw>" --epochs 500 --ml

Pre-train parameters download:  
For wpunet segmentation, you can download at [Baidu Pan](https://pan.baidu.com/s/13LXR25eWwgd-UbKIhLsGvA) (The key is '4yx7') or [Google Drive](https://drive.google.com/file/d/1Gc2j-DrJhX0E4fnvRItf95o0BXWQa-wr/view?usp=sharing), you should unzip it at './segmentation/'.   

## Visualization

The example results of WPU-Net algorithm is shown as follows: 

<p align = "center">
<img src="https://raw.githubusercontent.com/clovermini/MarkdownPhotos/master/WPUnet.png">
</p>

## Citation
<!--
If you find our work is useful for your research, Please be kind to cite it. Thanks. 

    Ma B, Liu C, Wei X, et al. WPU-Net: Boundary learning by using weighted propagation in convolution network[J]. arXiv preprint arXiv:1905.09226, 2019.
-->
This paper is in submission....
