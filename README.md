# BKC-Net-pytorch
This repository is for the paper "BKC-Net: Bi-knowledge Contrastive Learning for Renal Tumor Diagnosis on 3D CT images".

The Bi-knowledge Contrastive Network (BKC-Net) is used for renal tumor diagnosis on 3D CT images. It has two consecutive learning processes:
1) The focus-perceive learning focuses on the tumors by the segmentation branch and perceives the related surroundings via self-perception pooling in the perception
branch, thus accomodating to various degrees of enhancement caused by task-independent features. 
2) The bi-knowledge contrastive learning integrates two kinds of knowledge and contrasts both to discover more subtle differences among classes.


<p align="center"><img width="100%" src="fig/frame10.png" /></p>
