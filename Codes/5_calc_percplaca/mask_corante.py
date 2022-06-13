#%%
import numpy as np
from pathlib import Path
import os
import cv2 as cv
import matplotlib.pyplot as plt

#%matplotlib qt
#plt.close('all')

#%%
path_folder_ipc = Path('dataset/ipc')
path_folder_train_ipc = Path('{}/images'.format(path_folder_ipc))
path_folder_mask_ipc = Path('{}/masks'.format(path_folder_ipc))

patient_ids = [ name for name in os.listdir(path_folder_train_ipc)]
patient_ids = ['18']
for patient_id in patient_ids:
    
    patient_id = str(patient_id)
    patient_ipc = os.path.join(path_folder_train_ipc, patient_id + ".jpg")
    image_ipc = cv.imread(patient_ipc,cv.IMREAD_COLOR) #UNCHANGED não lê informação de rotação automática EXIF metadata
    image_ipc = cv.cvtColor(image_ipc,cv.COLOR_BGR2RGB)

    patient_ipc_mask = os.path.join(path_folder_mask_ipc, patient_id + "_mask.png")
    mask_ipc = cv.imread(patient_ipc_mask,cv.IMREAD_GRAYSCALE) #UNCHANGED não lê informação de rotação automática EXIF metadata   
    
    mask_ipc_tile = np.tile(np.expand_dims(mask_ipc,axis=2),(1,1,3))
    image_masked = image_ipc*mask_ipc_tile   


    plt.figure(figsize = (15,5))
    plt.subplot(1,3,1)
    plt.title("RED")
    plt.imshow(image_masked[:,:,0],'gray')
    plt.subplot(1,3,2)
    plt.title("GREEN")
    plt.imshow(image_masked[:,:,1],'gray')
    plt.subplot(1,3,3)
    plt.title("BLUE")
    plt.imshow(image_masked[:,:,2],'gray')
    plt.show()

    placa = image_masked[:,:,1].copy()
    placa[placa<150]=0
    all_px = np.count_nonzero(mask_ipc)
    placa_px = np.count_nonzero(placa)
    placa_perc = placa_px/all_px

    fig = plt.figure(figsize = (15,5))
    plt.suptitle("patient_id: "+patient_id)
    plt.subplot(1,3,1)
    plt.imshow(image_ipc)
    plt.subplot(1,3,2)
    plt.imshow(image_masked)
    plt.subplot(1,3,3)
    plt.title("% placa: {:.2f}".format(placa_perc*100))
    plt.imshow(placa,'gray')
    plt.show()
    

#%%
path_folder_ipv = Path('dataset/ipv')
path_folder_train_ipv = Path('{}/images'.format(path_folder_ipv))
path_folder_mask_ipv = Path('{}/masks'.format(path_folder_ipv))

patient_ids = [ name for name in os.listdir(path_folder_train_ipv)]
patient_ids = ['18']
for patient_id in patient_ids:
    
    patient_id = str(patient_id)
    patient_ipv = os.path.join(path_folder_train_ipv, patient_id + ".jpg")
    image_ipv = cv.imread(patient_ipv,cv.IMREAD_COLOR) #UNCHANGED não lê informação de rotação automática EXIF metadata
    image_ipv = cv.cvtColor(image_ipv,cv.COLOR_BGR2RGB)

    patient_ipv_mask = os.path.join(path_folder_mask_ipv, patient_id + "_mask.png")
    mask_ipv = cv.imread(patient_ipv_mask,cv.IMREAD_GRAYSCALE) #UNCHANGED não lê informação de rotação automática EXIF metadata   
    
    mask_ipv_tile = np.tile(np.expand_dims(mask_ipv,axis=2),(1,1,3))
    image_masked = image_ipv*mask_ipv_tile   

    plt.figure(figsize = (15,5))
    plt.subplot(1,3,1)
    plt.title("RED")
    plt.imshow(image_masked[:,:,0],'gray')
    plt.subplot(1,3,2)
    plt.title("GREEN")
    plt.imshow(image_masked[:,:,1],'gray')
    plt.subplot(1,3,3)
    plt.title("BLUE")
    plt.imshow(image_masked[:,:,2],'gray')
    plt.show()
    img_th =cv.inRange(image_masked,(20, 100, 100), (30, 255, 255))
    placa = image_masked[:,:,1].copy()
    placa[placa<150]=0
    all_px = np.count_nonzero(mask_ipv)
    placa_px = np.count_nonzero(placa)
    placa_perc = placa_px/all_px

    fig = plt.figure(figsize = (15,5))
    plt.suptitle("patient_id: "+patient_id)
    plt.subplot(1,3,1)
    plt.imshow(image_ipv)
    plt.subplot(1,3,2)
    plt.imshow(image_masked)
    plt.subplot(1,3,3)
    plt.title("% placa: {:.2f}".format(placa_perc*100))
    plt.imshow(placa,'gray')
    plt.show()
#%%
path_folder_ipv = Path('dataset/ipc_CycleGAN')
path_folder_train_ipv = Path('{}/images'.format(path_folder_ipv))
path_folder_mask_ipv = Path('{}/masks'.format(path_folder_ipv))

patient_ids = [ name for name in os.listdir(path_folder_train_ipv)]
#patient_ids = ['18']
for patient_id in patient_ids:
    patient = patient_id.split(".png")[0]
    patient_id = str(patient_id)
    patient_ipv = os.path.join(path_folder_train_ipv, patient+ ".png")
    image_ipv = cv.imread(patient_ipv,cv.IMREAD_COLOR) #UNCHANGED não lê informação de rotação automática EXIF metadata
    image_ipv = cv.cvtColor(image_ipv,cv.COLOR_BGR2RGB)

    patient_ipv_mask = os.path.join(path_folder_mask_ipv, patient+ "_mask.png")
    mask_ipv = cv.imread(patient_ipv_mask,cv.IMREAD_GRAYSCALE) #UNCHANGED não lê informação de rotação automática EXIF metadata   
    mask_ipv = cv.resize(mask_ipv,(image_ipv.shape[1],image_ipv.shape[0]))
    mask_ipv = np.tile(np.expand_dims(mask_ipv,axis=2),(1,1,3))
    
   
    image_masked = image_ipv*mask_ipv

    plt.figure(figsize = (15,5))
    plt.subplot(1,3,1)
    plt.title("RED")
    plt.imshow(image_masked[:,:,0],'gray')
    plt.subplot(1,3,2)
    plt.title("GREEN")
    plt.imshow(image_masked[:,:,1],'gray')
    plt.subplot(1,3,3)
    plt.title("BLUE")
    plt.imshow(image_masked[:,:,2],'gray')
    plt.show()
    img_th =cv.inRange(image_masked,(20, 100, 100), (30, 255, 255))
    placa = image_masked[:,:,1].copy()
    placa[placa<150]=0
    all_px = np.count_nonzero(mask_ipv)
    placa_px = np.count_nonzero(placa)
    placa_perc = placa_px/all_px

    fig = plt.figure(figsize = (15,5))
    plt.suptitle("patient_id: "+patient_id)
    plt.subplot(1,3,1)
    plt.imshow(image_ipv)
    plt.subplot(1,3,2)
    plt.imshow(image_masked)
    plt.subplot(1,3,3)
    plt.title("% placa: {:.2f}".format(placa_perc*100))
    plt.imshow(placa,'gray')
    plt.show()

# %%
