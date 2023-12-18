# -*- coding: UTF-8 -*-
import os
import argparse
import sys
import math
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset.datasets import *
from tools import *
from loss import cldice

def get_args():
    parser = argparse.ArgumentParser(description='Fluorescence with ECA-ResUnext !')
    parser.add_argument('--dataDir', default='datasets', help='dataset directory')
    parser.add_argument('--saveDir', default='weights', help='model save dir')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size for train')
    parser.add_argument('--size', type=int, default=[416,1024], nargs='+', help='image size for train')
    parser.add_argument('--region_list', type=str, nargs='+', default=['brain_area'], help='region list')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--nEpochs', type=int, default=150, help='number of epochs to train')
    parser.add_argument('--save_epoch', type=int, default=1, help='number of epochs to save model')
    parser.add_argument('--train_phase', default='end_to_end', help='train phase')
    parser.add_argument('--encoder', default='se_resnext50_32x4d', help='model encoder')
    parser.add_argument('--encoder_weights', default='imagenet', help='pretrained weights')
    parser.add_argument('--classes', default=['region'], help='CLASSES')
    parser.add_argument('--activation', default='sigmoid', help='sigmoid activation')

    args = parser.parse_args()
    print(args)
    return args

def zebrafish_train():
    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights,
        classes=len(args.classes),
        activation=args.activation,
    )
    convert_relu_to_PRELU(model)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Let's use",torch.cuda.device_count(),"GPUs!")
    else:
        device = torch.device("cpu")
        print('Using CPU!')
    model = torch.nn.DataParallel(model)
    model.to(device)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, args.encoder_weights)

    for name in args.region_list:
        img_train_dirs = [os.path.join(args.dataDir, name, 'train/images/')]
        mask_train_dirs = [os.path.join(args.dataDir, name, 'train/masks/')]
        img_val_dirs = [os.path.join(args.dataDir, name, 'val/images/')]
        mask_val_dirs = [os.path.join(args.dataDir, name, 'val/masks/')]

        # load dataset
        def zbft():
            return ZebrafishDataset(image_train, mask_train, classes=None,
                                    augmentation=get_training_augmentation(args.size),
                                    preprocessing=get_preprocessing(preprocessing_fn)
                                    )
        def zbfv():
            return ZebrafishDataset(image_val, mask_val, classes=None,
                                    augmentation=get_validation_augmentation(args.size),
                                    preprocessing=get_preprocessing(preprocessing_fn)
                                    )

        # begin batch training
        for img_train_dir, mask_train_dir, img_val_dir, mask_val_dir, name in \
                zip(img_train_dirs, mask_train_dirs, img_val_dirs, mask_val_dirs, args.region_list):
            # begin train the dataset
            image_train, mask_train = make_images_masks_list_with_random_shuffle(img_train_dir, mask_train_dir)
            image_val, mask_val = make_images_masks_list_with_random_shuffle(img_val_dir, mask_val_dir)
            preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, args.encoder_weights)
            train_dataset = zbft()
            valid_dataset = zbfv()

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
            # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
            # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

            loss = cldice.soft_dice_cldice()
            metrics = [
                smp.utils.metrics.Fscore(threshold=0.5)
            ]
            lr = args.lr
            optimizer = torch.optim.Adam([
                dict(params=model.parameters(), lr=0.0001),
            ])

            # create epoch runners
            # it is a simple loop of iterating over dataloader`s samples
            train_epoch = smp.utils.train.TrainEpoch(
                model,
                loss=loss,
                metrics=metrics,
                optimizer=optimizer,
                device=device,
                verbose=True,
            )

            valid_epoch = smp.utils.train.ValidEpoch(
                model,
                loss=loss,
                metrics=metrics,
                device=device,
                verbose=True,
            )
            max_score = 0
            # save model
            os.makedirs(args.saveDir, exist_ok=True)
            best_model_path = join(args.saveDir, name + '_best_model.pth')
            latest_model_path = join(args.saveDir, name + '_latest_model.pth')
            for i in range(0, args.nEpochs):
                print('\nEpoch: {}'.format(i))
                train_logs = train_epoch.run(train_loader)
                valid_logs = valid_epoch.run(valid_loader)

                # do something (save model, change lr, etc.)
                torch.save(model, latest_model_path)
                if max_score < valid_logs['fscore']:
                    max_score = valid_logs['fscore']
                    torch.save(model, best_model_path)
                    print('Model saved!')

                lr = lr * math.pow((1 - i/150),0.9)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

# Replacement activation function
def convert_relu_to_PRELU(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.PReLU())
        else:
            convert_relu_to_PRELU(child)

if __name__ == "__main__":
    DATA_PATH = ''
    args = get_args()
    zebrafish_train()

    