#!/usr/bin/env python
# coding: utf-8
import argparse
import sys
from dataset.datasets import *
import segmentation_models_pytorch as smp
import torch
from tools import *
DEVICE = 'cuda:0'

def get_args():
    parser = argparse.ArgumentParser(description='Fluorescence with ECA-ResXUnet !')
    parser.add_argument('--weights', default='', help='weights path')
    parser.add_argument('--savedir', default='', help='save dir')
    parser.add_argument('--imagedir', default='', help='image path')
    parser.add_argument('--size', type=int, default=[416,1024], nargs='+', help='image size for train')
    parser.add_argument('--region_list', type=str, nargs='+', default=['CCV', 'CV'], help='region list')
    parser.add_argument('--encoder', default='se_resnext50_32x4d', help='model encoder')
    parser.add_argument('--encoder_weights', default='imagenet', help='pretrained weights')
    parser.add_argument('--classes', default=['region'], help='CLASSES')
    parser.add_argument('--activation', default='sigmoid', help='sigmoid activation')

    args = parser.parse_args()
    print(args)
    return args

list_dir_endswith = lambda x, y: list(filter(lambda z: z.endswith(y), os.listdir(x)))

list_model = lambda x: list_dir_endswith(x, '.pth')
list_dir = lambda x: filter(lambda y: os.path.isdir(join(x, y)), os.listdir(x))
abs_dir_list = lambda r, dirs: [join(r, d) for d in dirs]

def all_model(path_list):
    for p in path_list: yield torch.load(p, map_location='cuda:0')

def list_all_relative_dirs(p):
    cur_dirs = list(list_dir(p)) 
    for c in cur_dirs:
        for cc in list_all_relative_dirs(join(p, c)):
            cur_dirs += [join(c, cc)]
    return cur_dirs


def batch_inference_in_dir(src_dir, dest_dir, model):
    print(src_dir, dest_dir)
    make_dirs(dest_dir)
    img_path_list = []
    for e in ['.bmp', '.png', '.jpg','tif']:
        img_path_list += list(abs_dir_list(src_dir, list_dir_endswith(src_dir, e)))

    if len(img_path_list)==0:
        return
    inference_dataset = ZebrafishDataset(img_path_list, None, classes=None,
                                         augmentation=get_validation_augmentation(args.size),
                                         preprocessing=get_preprocessing(preprocessing_fn))
    for id in range(len(inference_dataset)):
        img_augmented = inference_dataset[id]
        if img_augmented is None:
            continue
        img_path = inference_dataset.images_fps[id]
        img_org = cv_imread(img_path)
        name = os.path.basename(img_path)
        print(name)

        x_tensor = torch.from_numpy(img_augmented).to(DEVICE).unsqueeze(0)


        pr_mask= model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy())
        pr_mask_org_size = cv2.resize(pr_mask, (img_org.shape[1], img_org.shape[0]), interpolation=cv2.INTER_CUBIC)
        pr_mask_org_size_threshold = pr_mask_org_size.round().astype(np.uint8) * 255

        # draw bbox
        contours, hierarchy = cv2.findContours(pr_mask_org_size_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("Warning: can't find the region in ", img_path)
            continue
        # apply mask on the original image and extract out rect of detected region
        cnt = get_largest_contour(contours)
        mask = make_mask_by_contours(img_org.shape, img_org.dtype, [cnt])
        img_masked = np.where(mask, img_org, 0)
        mask_rect = extract_contour_rect(pr_mask_org_size_threshold, cnt)
        # output the img_rect and mask_rect
        img_rect = extract_contour_rect(img_masked, cnt)
        os.makedirs(os.path.join(dest_dir, 'mask'), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, 'img_region'), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, 'mask_region'), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, 'img_seg'), exist_ok=True)

        # save mask
        cv2.imencode(name[-4:], pr_mask_org_size_threshold)[1].tofile(os.path.join(dest_dir, 'mask', os.path.basename(img_path)))
        cv2.imencode(name[-4:], mask_rect)[1].tofile(os.path.join(dest_dir, 'mask_region', os.path.basename(img_path)))

        # save image region
        cv2.imencode(name[-4:], img_rect)[1].tofile(
            os.path.join(dest_dir,'img_region', os.path.basename(img_path)))
        cv2.imencode(name[-4:], img_masked)[1].tofile(
            os.path.join(dest_dir, 'img_seg', os.path.basename(img_path)))


def inference_all_dirs(src_dir, model_dir):
    rd = list_all_relative_dirs(src_dir)
    src_dir_list = [src_dir] + abs_dir_list(src_dir, rd)
    all_model_path = abs_dir_list(model_dir, list_model(model_dir))
    bid = batch_inference_in_dir
    model_name = lambda p: os.path.basename(p).split('_')[0]
    for i, m in enumerate(all_model(all_model_path)):
        print('load model')
        if isinstance(m, torch.nn.DataParallel):
            m = m.module
        dest_dir_list = [dest_dir] + abs_dir_list(dest_dir + '_' + model_name(all_model_path[i]), rd)
        list(map(lambda s, d: bid(s, d, m), src_dir_list, dest_dir_list))

if __name__ == "__main__":
    args = get_args()
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, args.encoder_weights)

    for name in args.region_list:
        weights = args.weights
        src_dir = args.imagedir
        model_path = join(weights, name)
        dest_path = args.savedir
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        dest_dir = join(dest_path,name)
        inference_all_dirs(src_dir, model_path)
    exit(0)


