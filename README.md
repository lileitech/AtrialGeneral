# AtrialGeneral

## Overview
The repository contains the core codes of "[AtrialGeneral: Domain Generalization for Left Atrial Segmentation of Multi-Center LGE MRIs](https://arxiv.org/abs/2106.08727)".
The code includes four baseline models (U-Net, U-Net++, DeepLab v3+ and MAnet) and three model generalization schemes.
The schemes include histogram matching (HM), mutual information based disentangled representation, and random style transfer.

<img src="https://github.com/Marie0909/AtrialGeneral/edit/main/AtrialGeneral.png" width="300" height="450" />

## Dataset
The dataset employed in this work is from:
[ISBI 2012: Left Atrium Fibrosis and Scar Segmentation Challenge](http://atriaseg2018.cardiacatlas.org/) and
[MICCAI 2018: Atrial Segmentation Challenge](http://www.cardiacatlas.org/challenges/left-atrium-fibrosis-and-scar-segmentation-challenge/)

## Cite
If this code is useful for you, please kindly cite this work via:

@article{li2021atrialgeneral,  
  title={AtrialGeneral: Domain Generalization for Left Atrial Segmentation of Multi-Center LGE MRIs},  
  author={Li, Lei and Zimmer, Veronika A and Schnabel, Julia A and Zhuang, Xiahai},  
  journal={arXiv preprint arXiv:2106.08727},  
  year={2021}  
}  

If you have any questions, please contact lilei.sky@sjtu.edu.cn.
