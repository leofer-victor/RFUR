# Deep Learning-based Freehand 3D Ultrasound Reconstruction with External Camera

## Introduction

3D ultrasound (US) has the potential to enhance the accuracy and speed of doctors' diagnoses by providing a volumetric perception. Compared to native 3D US, freehand 3D US reconstruction demonstrates its advantages in terms of flexibility and lightweightness. Recently, methodologies and systems primarily attempt to do the reconstruction purely based on 2D US image sequences, utilizing convolutional neural networks (CNNs) to determine sequence positioning. However, extracting out-of-plane motions solely from 2D US images has proved to be challenging and error-prone. Specifically, most of the existing systems lack the ability to perceive global motions, which limits their applications to only certain predefined scanning patterns. In this paper, we propose a deep learning-based freehand 3D ultrasound reconstruction approach combined with an external camera to capture the tendency for large-scale motion. The corresponding features from the US images are utilized to determine the fine local transformations, while the RGB images from an external camera attached to the US probe are used to provide global motion awareness for the 3D reconstruction network. The mounted camera is positioned to gain a view of the scanning target from an exterior perspective. In order to better represent the moving tendency of the probe, the optical flows of the RGB images are calculated. Two consecutive US images, together with one optical flow image from the external camera are concatenated, and a sequence of such combinations is fed into a 3D-CNN network to obtain continuous pose transformations. As the scanning progresses, we concatenate all transformations to derive the global trajectory. Notably, our work represents the first instance of fusing the optical flow of an external camera and US images for US sequences motion estimation. Experimental results demonstrate that the proposed method outperforms the baseline method.

## Environment

Set your environment by anaconda

```python
pip install -r ./requirements.txt
```

## Running

You can make a change to the train.sh and test.sh and run the training and testing by

```
./train.sh
./test.sh
```

