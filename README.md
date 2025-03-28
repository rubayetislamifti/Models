# Model Installation Guide

## Table of Contents
1. [Introduction](#introduction)
2. [DeepGazeI](#deepgazei)
   - [Installation Steps](#installation-steps)
   - [Update Image Folder Path](#update-image-folder-path)
   - [Notes](#notes)
3. [ML-NET](#ml-net)
   - [Installation Steps](#installation-steps-1)
   - [Update Image Folder Path](#update-image-folder-path-1)
4. [TranSalNet-Res](#transalnet-res)
   - [Installation Steps](#installation-steps-2)
   - [Update Image Folder Path](#update-image-folder-path-2)
5. [Conclusion](#conclusion)

## Introduction
This document provides installation instructions for our models: **DeepGazeI**, **ML-NET**, and **TranSalNet-Res**.

## DeepGazeI
### Installation Steps
**DeepGazeI** is our first model. To install it, follow these steps:

1. Clone the repository from GitHub:
   ```bash
   git clone git@github.com:rubayetislamifti/Models.git
   ```
2. Install the required dependencies:
   ```bash
   pip install scipy
   pip install torch
   pip install opencv-python
   ```

### Update Image Folder Path
Replace the following directory:
```
I:\Saliency4asd\Saliency4asd\ASD_FixMaps
```
with your actual path:
```
path/to/your/folder
```

### Notes:
- `deepgazeI.py` is used for processing a single image.
- `DeepGazeI2.py` and `DeepGazeI3.py` are used for batch processing of images in `TD_FixMaps` and `ASD_FixMaps`.

## ML-NET
### Installation Steps
**ML-NET** is our second model. Follow these steps to install it:

1. Clone the repository from GitHub:
   ```bash
   git clone git@github.com:rubayetislamifti/Models.git
   ```
2. Install the required dependency:
   ```bash
   pip install torch
   ```

### Update Image Folder Path
Replace the following directory:
```
/content/drive/MyDrive/DIP/TD_FixMaps
```
with your actual path:
```
path/to/your/folder
```

## TranSalNet-Res
### Installation Steps
**TranSalNet-Res** is our third model. Follow these steps to install it:

1. Clone the repository from GitHub:
   ```bash
   git clone git@github.com:rubayetislamifti/Models.git
   ```
2. Install the required dependencies:
   Your IDE should automatically detect and install the required dependencies.

### Update Image Folder Path
Replace the following directory:
```
E:/Digital Image Processing/Assignment 3/TranSalNet-Res/Datasets/Saliency_map_TD
```
with your actual path:
```
path/to/your/folder
```

## Conclusion
Following these steps will ensure a smooth installation of all three models. Make sure to update the paths accordingly before running the scripts.

