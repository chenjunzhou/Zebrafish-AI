# Download weights

## Download from Baidu Netdisk or Dropbox

[Baidu netdisk Link](https://pan.baidu.com/s/180stNFemiUNkSvrAJ9A60g?pwd=0f9e)
[Dropbox Link](https://www.dropbox.com/scl/fi/r3qa1etm793yhxnir63i1/weights.zip?rlkey=typpdp8oz7l11wvpw31fl04yy&dl=0)   

## Structure of the weights

```
- weights
    |—— CCV
    |   |—— CCV_best_model.pth
    |—— CV
        |—— CV_best_model.pth
```

## Run inference.py

```
CUDA_VISIBLE_DEVICES=0 python inference.py --weights weights/ --savedir ./output --imagedir images/
```

- weights: model weights
- imagedir: image path for segmentation
- savedir: result save path
