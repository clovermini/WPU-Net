# WPU-Net
WPU-Net: Boundary learning by using weighted propagation in convolution network.
<!-- This is the pytorch implementation of algorithm in paper https://arxiv.org/pdf/1905.09226.pdf. -->

## Abstract
Deep learning has driven great progress in natural and biological image processing. However, in materials science and engineering, there are often some flaws and indistinctions in material microscopic images induced from complex sample preparation, even due to the material itself, hindering the detection of target objects. In this work, we propose WPU-net that redesign the architecture and weighted loss of U-Net to force the network to integrate information from adjacent slices and pay more attention to the topology in this boundary detection task. Then, the WPU-net was applied into a typic material example, i.e., the grain boundary detection of polycrystalline material. Experiments demonstrate that the proposed method achieves promising performance and outperforms state-of-the-art methods. Besides, we propose a new method for object tracking between adjacent slices, which can effectively reconstruct the 3D structure of the whole material. Finally, we present a materials microscopic image dataset with the goal of advancing the state-of-the-art in image processing for materials sciences.

## Environment

    python 3.6
    pytorch 1.0
    gala (for evaluation)

gala is installed according to https://github.com/janelia-flyem/gala.


## DataSet and Running

We opened up the material dataset we used in paper experiment. Hoping more and more researchers can participate in image processing of material science, promoting the development of materials science. You can download it from https://github.com/Keep-Passion/pure_iron_grain_data_sets.

Usage Demo:

    # inference
    python main.py

    # train WPU-Net
    python segmentation/trainer.py --input="<path to your dataset>" --bs=24 --loss="<abw/cbw>" --epochs 500 --ml

Pre-train parameters download:  
For wpunet segmentation, you can download at [Baidu Pan](https://pan.baidu.com/s/1_xCiSQe0tXhDP0cMnUPp5A) (The key is 'ttah') or [Google Drive](https://drive.google.com/file/d/1Gc2j-DrJhX0E4fnvRItf95o0BXWQa-wr/view?usp=sharing), you should unzip it at './segmentation/'.   
For grain track, you can download at [Baidu Pan](https://pan.baidu.com/s/1hBVOt21wxi_8HUOlPyg_Vw) (The key is 'k6b1') or [Google Drive](https://drive.google.com/file/d/1dhwSwmxDKBwub9Wi4DPXpJotnHrNOyaL/view?usp=sharing), you should unzip it at './grain_track/'.   

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
