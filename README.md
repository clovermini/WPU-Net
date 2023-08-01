# WPU-Net
Boundary learning by using weighted propagation in convolution network.
This is the PyTorch implementation of the algorithm in [paper](https://doi.org/10.1016/j.jocs.2022.101709).

## Abstract
In material science, image segmentation is of great significance for quantitative analysis of microstructures. Here, we propose a novel Weighted Propagation Convolution Neural Network based on U-Net (WPU-Net) to detect boundary in poly-crystalline microscopic images. We introduce spatial consistency into network to eliminate the defects in raw microscopic image. And we customize adaptive boundary weight for each pixel in each grain, so that it leads the network to preserve grain鈥檚 geometric and topological characteristics. Moreover, we provide our dataset with the goal of advancing the development of image processing in materials science. Experiments demonstrate that the proposed method achieves promising performance in both of objective and subjective assessment. In boundary detection task, it reduces the error rate by 7%, which outperforms state-of-the-art methods by a large margin.}

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

If you find our work useful for your research, Please be kind to cite it. Thanks. 

    Wei Liu, Jiahao Chen, Chuni Liu, Xiaojuan Ban, Boyuan Ma, Hao Wang, Weihua Xue, Yu Guo. Boundary learning by using weighted propagation in convolution network[J]. Journal of Computational Science, 62, 2022, 101709. https://doi.org/10.1016/j.jocs.2022.101709.

Or

    @article{LIU2022101709,
    title = {Boundary learning by using weighted propagation in convolution network},
    journal = {Journal of Computational Science},
    volume = {62},
    pages = {101709},
    year = {2022},
    issn = {1877-7503},
    doi = {https://doi.org/10.1016/j.jocs.2022.101709},
    url = {https://www.sciencedirect.com/science/article/pii/S1877750322001077},
    author = {Wei Liu and Jiahao Chen and Chuni Liu and Xiaojuan Ban and Boyuan Ma and Hao Wang and Weihua Xue and Yu Guo},
    keywords = {Material microscopic image segmentation, Convolution neural network, Loss function},
    abstract = {In material science, image segmentation is of great significance for quantitative analysis of microstructures. Here, we propose a novel Weighted Propagation Convolution Neural Network based on U-Net (WPU-Net) to detect boundary in poly-crystalline microscopic images. We introduce spatial consistency into network to eliminate the defects in raw microscopic image. And we customize adaptive boundary weight for each pixel in each grain, so that it leads the network to preserve grain鈥檚 geometric and topological characteristics. Moreover, we provide our dataset with the goal of advancing the development of image processing in materials science. Experiments demonstrate that the proposed method achieves promising performance in both of objective and subjective assessment. In boundary detection task, it reduces the error rate by 7%, which outperforms state-of-the-art methods by a large margin.}
    }
