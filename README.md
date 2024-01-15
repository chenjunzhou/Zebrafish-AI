# Zebrafish AI: Zebrafish Segmentation and Statistical Analysis

This document outlines the use of a Python-based zebrafish segmentation model and MATLAB code for statistical analysis.

## I. Zebrafish Segmentation Model

### 1. Environment Configuration

To use this code, ensure your environment meets the following requirements: Python>=3.6.0 and PyTorch>=1.8.0. Follow these steps to set up the necessary environment:

- **Install Miniconda or Anaconda**
  - Miniconda: [Miniconda Documentation](https://docs.conda.io/projects/miniconda/en/latest/)
  - Anaconda: [Anaconda Download](https://www.anaconda.com/download)

- **Creating a Virtual Environment**

  ```markdown
  conda create --name pytorch python=3.8
  conda activate pytorch
  ```

- **Install PyTorch**
  - PyTorch: [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
  
  Verify PyTorch Installation:

  ```python
  import torch

  print(torch.__version__)
  print(torch.cuda.is_available())
  ```

- **Clone Repository and Install Dependencies**

  ```shell
  git clone https://github.com/...
  cd ECA-ResXUnet
  pip install -r requirements.txt
  ```

### 2. Inference

Use pre-trained model weights for segmenting zebrafish images:

1. **Download Weights from one of the following links:**
   - Baidu Netdisk: [Download Link](https://pan.baidu.com/s/180stNFemiUNkSvrAJ9A60g?pwd=0f9e)
   - Dropbox: [Download Link](https://www.dropbox.com/scl/fi/r3qa1etm793yhxnir63i1/weights.zip?rlkey=typpdp8oz7l11wvpw31fl04yy&dl=0)

  ```markdown
  - weights
      |—— CBV
      |   |—— CV_best_model.pth
      |—— CCV
      |   |—— CCV_best_model.pth
  ```

2. **Run Inference Script**

  ```shell
  CUDA_VISIBLE_DEVICES=0 python inference.py --weights weights/ --savedir ./output --imagedir images/
  ```

  - `weights`: model weights directory
  - `imagedir`: directory for images to be segmented
  - `savedir`: directory to save results

### 3. Training the Model

Train the model with your custom dataset:

- **Create a Custom Dataset and Run train.py**

  ```markdown


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
      |—— CBV
  ```

  ```shell
  # Train the CCV model
  CUDA_VISIBLE_DEVICES=0 python train.py --dataDir datasets/ --batch_size 8 --size 416 1024 --region_list CCV brain_area
  ```

  `region_list` specifies the segmentation regions of interest. You can add more regions by listing additional names.

## II. MATLAB Code for Statistical Analysis of Zebrafish Segmentation Results

### Prepare Segmentation Results

First, ensure you have segmented zebrafish images using the model.

### Download Statistical Analysis Code

Download the MATLAB code for zebrafish segmentation analysis from the repository.

### 1. SIVP

  ```markdown
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

  Follow these steps:

  1.1 Download and place all functions in the same folder.

  1.2 Modify `main.m` in MATLAB, adjusting the pathname:
   - `scrDir`: folder containing images for processing.
   - `output_xlsx_path`: path to save the extracted features in an Excel file.

  1.3 Modify `SIV_function_all.m` for the image format (png or bmp).

  1.4 Run `main.m`. Results are saved in the specified output path.

### 2. CBV

  ```markdown
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

  Follow similar steps as for SIVP.

### 3. CCV

  ```markdown
  - Matlab
      |—— CCV.m
      |—— Globularity.m
      |—— minboundrect.m
      |—— Inpoly.m
  ```

  Place the functions in the image directory and run `CCV.m`.

### 4. DA, PCV, ISV, CVP

  ```markdown
  - Matlab
      |—— DA.m
      |—— PCV.m
      |—— ISV.m
      |—— CVP.m
      |—— minboundrect.m
      |—— Inpoly.m
  ```

  Run each MATLAB script after modifying the pathname.

### 5. View Statistical Results

Results will be displayed in the MATLAB command window or saved to the specified output path.

Follow these instructions

to effectively utilize the zebrafish segmentation model and perform statistical analysis. For more detailed instructions or assistance, please refer to the documentation in the code repository or contact the developers.