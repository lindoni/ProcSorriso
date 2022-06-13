#%%
from pathlib import Path
import os
import cv2 as cv
import matplotlib.pyplot as plt

path_val_imgs = Path('dataset/val_images')
path_val_masks = Path('dataset/val_masks')
path_pred_masks = Path('dataset/pred_masks')
path_results = Path('dataset/results')


patient_ids = os.listdir(path_val_imgs)

for patient_id in patient_ids:    
    patient_id = patient_id.split(".jpg")[0]

    path_img = os.path.join(path_val_imgs, patient_id + ".jpg") #sem corante 
    image_ipv = cv.imread(path_img,cv.IMREAD_COLOR)
    image_ipv = cv.cvtColor(image_ipv,cv.COLOR_BGR2RGB)

    path_val_mask = os.path.join(path_val_masks, patient_id + "_mask.png") #sem corante 
    val_mask = cv.imread(path_val_mask,cv.IMREAD_GRAYSCALE)    

    path_pred_mask = os.path.join(path_pred_masks, patient_id + "_pred.png") #sem corante 
    pred_mask = cv.imread(path_pred_mask,cv.IMREAD_GRAYSCALE)
    pred_mask = cv.resize(pred_mask,(4160,3120))
    
    fig = plt.figure(figsize = (15,5))
    plt.suptitle("TARGET - patient_id: "+patient_id)
    plt.subplot(1,3,1)
    plt.imshow(image_ipv)
    plt.subplot(1,3,2)
    plt.imshow(val_mask,'gray')
    plt.subplot(1,3,3)
    plt.imshow(image_ipv)
    plt.imshow(val_mask, alpha=0.3)
    plt.show()
    path_comp = os.path.join(path_results, patient_id+".png")
    fig.savefig(path_comp)

    fig = plt.figure(figsize = (15,5))
    plt.suptitle("PRED - patient_id: "+patient_id)
    plt.subplot(1,3,1)
    plt.imshow(image_ipv)
    plt.subplot(1,3,2)
    plt.imshow(pred_mask,'gray')
    plt.subplot(1,3,3)
    plt.imshow(image_ipv)
    plt.imshow(pred_mask, alpha=0.3)
    plt.show()
    path_comp = os.path.join(path_results, patient_id+"_pred.png")
    fig.savefig(path_comp)


    #image_ipv_masked = image_ipv[val_mask]
    #plt.figure()

#%%