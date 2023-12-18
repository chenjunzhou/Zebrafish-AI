
## Environment Configuration

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

## train

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


## Inference

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
