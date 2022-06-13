#%%
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 312  # 3120 originally
IMAGE_WIDTH = 416  # 4160 originally
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_MODEL = False
TRAIN_IMG_DIR = "dataset/train/images/"
TRAIN_MASK_DIR = "dataset/train/masks/"
VAL_IMG_DIR = "dataset/validation/images/"
VAL_MASK_DIR = "dataset/validation/masks/"
TEST_IMG_DIR = "dataset/test/images"
TEST_MASK_DIR = "dataset/test/masks"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def load_dataset():

    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transform = test_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    train_loader, val_loader, test_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        test_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader

def train(train_loader, val_loader, model):

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    if TRAIN_MODEL:

        for epoch in range(NUM_EPOCHS):
            train_fn(train_loader, model, optimizer, loss_fn, scaler)

            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            filename_ckp = "ckps/ckp_{}.pth.tar".format(epoch)
            save_checkpoint(checkpoint,filename_ckp)

            # check accuracy
            check_accuracy(val_loader, model, device=DEVICE)

            # print some examples to a folder
            save_predictions_as_imgs(
                val_loader, model, epoch, folder="saved_images/", device=DEVICE
            )
    
def test(test_loader, model):

    model.eval()
    for idx, (x, y) in enumerate(test_loader):
        x = x.to(device=DEVICE)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            print(preds.shape)
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(np.transpose(x[0,:,:,:].cpu(),(1,2,0)))
        plt.subplot(1,2,2)
        plt.imshow(preds[0,0,:,:].cpu(),'gray')
        plt.show()

if __name__ == "__main__":

    train_loader, val_loader, test_loader = load_dataset()
    
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)    
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    train(train_loader, val_loader, model)
    test(test_loader, model)

# %%
