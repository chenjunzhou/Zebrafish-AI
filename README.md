# Zebrafish AI: Zebrafish Segmentation and Statistical Analysis User Guide

This document provides instructions on how to use a Python-based zebrafish segmentation model and MATLAB code for statistical analysis.

# I. Zebrafish Segmentation Model

## 1. Environment Configuration

To execute this code, an environment with Python>=3.6.0 and PyTorch>=1.8.0 is required. The following is a concise tutorial for configuring the necessary environment."

- Install Miniconda or Anaconda

  Miniconda: https://docs.conda.io/projects/miniconda/en/latest/
  Anaconda: https://www.anaconda.com/download
- Creating a Virtual Environment

  ```
  conda create --name pytorch python=3.8
  conda activate pytorch
  ```
- Install PyTorch

  PyTorch: https://pytorch.org/get-started/locally/

  Verify PyTorch Installation:

  ```
  import torch

  print(torch.__version__)

  print(torch.cuda.is_available())
  ```
- Clone repo and install requirements.txt

  ```
  git clone https://github.com/...
  cd ECA-ResXUnet
  pip install -r requirements.txt
  ```

## 2. train

Create Custom Dataset and run train.py

```
- datasets
    |—— CCV
    |   |—— train
    |   |   |—— images
    |   |   |   |—— 1.jpg
    |   |   |   |—— 2.jpg
    |   |   |—— masks
    |   |   |   |—— 1.jpg
    |   |   |   |—— 2.jpg
    |   |—— val
    |        |—— images
    |        |   |—— 3.jpg
    |        |   |—— 4.jpg
    |        |—— masks
    |        |   |—— 3.jpg
    |        |   |—— 4.jpg
    |—— CV

```

```
CUDA_VISIBLE_DEVICES=0 python train.py --dataDir datasets/ --batch_size 8 --size 416 1024 --region_list CCV brain_area
```

## 3. Inference

Download weights and run inference.py

```
- weights
    |—— CCV
    |   |—— CCV_best_model.pth
    |—— CV
        |—— CV_best_model.pth
```

```
CUDA_VISIBLE_DEVICES=0 python inference.py --weights weights/ --savedir ./output --imagedir images/
```

- weights: model weights
- imagedir: image path for segmentation
- savedir: result save path

# II. Matlab Code for Statistical Analysis of Zebrafish Segmentation Results

### Prepare Segmentation Results

Ensure that you have run the zebrafish segmentation model and obtained the segmented zebrafish images.

### Download Statistical Analysis Code

Download the MATLAB code for statistical analysis of zebrafish segmentation from the code repository.

## 1. SIVP

```
- SIVP
    |—— SIV_budding_num_2.m
    |—— SIV_Fluorescent.m
    |—— SIV_hole_area.m
    |—— SIV_other.m
    |—— SIV_function_all.m
    |—— minboundrect.m
    |—— main.m
    |—— Inpoly.m
```

1.1 Download the above functions under the same folder

1.2 Open main.m with matlab, modify the pathname

- scrDir：the folder used to store the pending pictures(This folder path can also be unmodified, and when you run the code, the folder option to select will pop up, and then select the folder where the image to be processed is located)
- output_xlsx_path: Save the Excel file path of the extracted features, the path must be modified, and after the run, the Excel sheet will be under the folder of the path

1.3 Open SIV_function_all.m with matlab, modify the image format(png or bmp)

1.4 Run main.m, the results are saved in the output path

## 2. CBV

```
- CBV
    |—— CBV_branch_point.m
    |—— CBV_Fluorescent.m
    |—— CBV_Vessel_area.m
    |—— CBV_branch_point_100.m
    |—— CBV_function_all.m
    |—— minboundrect.m
    |—— main.m
    |—— Inpoly.m
```

2.1 Download the above functions under the same folder

2.2 Open main.m with matlab, modify the pathname

- scrDir：the folder used to store the pending pictures.(This folder path can also be unmodified, and when you run the code, the folder option to select will pop up, and then select the folder where the image to be processed is located)
- output_xlsx_path: Save the Excel file path of the extracted features, the path must be modified, and after the run, the Excel sheet will be under the folder of the path.

2.3 Open CBV_function_all.m with matlab, modify the image format(png or bmp)

2.4 Run main.m, the results are saved in the output path

## 3. CCV

```
- Matlab
    |—— CCV.m
    |—— Globularity.m
    |—— minboundrect.m
    |—— Inpoly.m
```

3.1 Put three functions(Globularity.m,minboundrect.m,Inpoly.m) in the folder where the image is located

3.2 run the function,the results are saved in the output path

## 4. DA,PCV,ISV,CVP

```
- Matlab
    |—— DA.m
    |—— PCV.m
    |—— ISV.m
    |—— CVP.m
    |—— minboundrect.m
    |—— Inpoly.m
```

4.1 Put two functions(minboundrect.m,Inpoly.m) in the folder where the image is located

4.2 Open DA.m with matlab, modify the pathname and run the function,the results are saved in the output path

4.3 Open PCV.m with matlab, modify the pathname and run the function,the results are saved in the output path

4.4 Open ISV.m with matlab, modify the pathname and run the function,the results are saved in the output path

4.5 Open CVP.m with matlab, modify the pathname and run the function,the results are saved in the output path

Run the MATLAB code for statistical analysis.

## 5. View Statistical Results

After running the statistical analysis code, the program will generate the statistical results, which will be displayed in the MATLAB command window or saved to the specified output path.

Follow the above steps to use the zebrafish segmentation model and statistical analysis code. For more detailed instructions, please refer to the documentation in the code repository or contact the developers for support.
