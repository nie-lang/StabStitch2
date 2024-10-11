# <p align="center">StabStitch++: Unsupervised Online Video Stitching with Spatiotemporal Bidirectional Warps

## Introduction

[Lang Nie](https://nie-lang.github.io/)<sup>1</sup>, [Chunyu Lin](https://faculty.bjtu.edu.cn/8549/)<sup>1</sup>, [Kang Liao](https://kangliao929.github.io/)<sup>2</sup>, [Yun Zhang](http://zhangyunnet.cn/academic/index.html)<sup>3</sup>, [Shuaicheng Liu](http://www.liushuaicheng.org/)<sup>4</sup>, [Yao Zhao](https://faculty.bjtu.edu.cn/5900/)<sup>1</sup>

<sup>1</sup> Beijing Jiaotong University  {nielang, cylin, yzhao}@bjtu.edu.cn

<sup>2</sup> Nanyang Technological University

<sup>3</sup> Communication University of Zhejiang 

<sup>4</sup> University of Electronic Science and Technology of China


> ### Feature
> Compared with the conference version (StabStitch), the main contributions of StabStitch++ are as follows:
> 
> 1. We propose a differentiable bidirectional decomposition module to carry out bidirectional warping on a virtual middle plane, which evenly spreads warping burdens across both views. It benefits both image and video stitching, demonstrating universality and scalability.
>    
> 2. A new warp smoothing model is presented to simultaneously encourage content alignment, trajectory smoothness, and online collaboration. Different from StabStitch that sacrifices alignment for stabilization, the new model makes no compromise and optimizes both of them in the online mode. 
![image](https://github.com/nie-lang/StabStitch2/blob/main/figure.jpg)
The above figure shows the difference between StabStitch and StabStitch++.
> 
## Video
We will release a video of our results on YouTube.

## 📝 Changelog

- [x] 2024.10.11: The repository of StabStitch++ is created.
- [ ] Release the video of our results.
- [ ] Release the collected traditional datasets.
- [ ] Release the testing code and pre-trained models.
- [ ] Release the training code.

## Dataset 
For the StabStitch-D dataset, please refer to [StabStitch](https://github.com/nie-lang/StabStitc). 

For the collected traditional datasets, we will release them soon.

## Code
We plan to release the code in about two weeks.


## Meta
If you have any questions about this project, please feel free to drop me an email.

NIE Lang -- nielang@bjtu.edu.cn
```
@inproceedings{nie2024eliminating,
author="Nie, Lang and Lin, Chunyu and Liao, Kang and Zhang, Yun and Liu, Shuaicheng and Ai, Rui and Zhao, Yao",
title="Eliminating Warping Shakes for Unsupervised Online Video Stitching",
booktitle="ECCV",
year="2024",
pages="390--407"
}
```

## References
[1] L. Nie, C. Lin, K. Liao, Y. Zhang, S. Liu, R. Ai, Y. Zhao. Eliminating Warping Shakes for Unsupervised Online Video Stitching. ECCV, 2024.   
[2] L. Nie, C. Lin, K. Liao, S. Liu, and Y. Zhao. Parallax-Tolerant Unsupervised Deep Image Stitching. ICCV, 2023.   
[3] S. Liu, P. Tan, L. Yuan, J. Sun, and B. Zeng. Meshflow: Minimum latency online video stabilization. ECCV, 2016.  
