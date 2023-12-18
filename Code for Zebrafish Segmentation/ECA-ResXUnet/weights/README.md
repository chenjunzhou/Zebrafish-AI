## Download weights

### Download from dropbox
```
```

### Download from baidu netdisk
```
Link：https://pan.baidu.com/s/1dswnR1TKClGOUoCk6jD_aA?pwd=1kvl 
```
```
- weights
    |—— CCV
    |   |—— CCV_best_model.pth
    |—— CV
        |—— CV_best_model.pth
```


## run inference.py
```
CUDA_VISIBLE_DEVICES=0 python inference.py --weights weights/ --savedir ./output --imagedir images/
```

- weights: model weights
- imagedir: image path for segmentation
- savedir: result save path
