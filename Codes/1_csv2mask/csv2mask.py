#%%
import numpy as np
from pathlib import Path
import os
import cv2 as cv
import matplotlib.pyplot as plt
import csv
import json

def makedir(path2create):
    """Create directory if it does not exists.""" 
    error = 1    
    if not os.path.exists(path2create):
        os.makedirs(path2create)
        error = 0    
    return error

path_folder_all = Path('dataset/all') #to read ipc e ipv
path_folder_ipc = Path('dataset/ipc') #to write ipc
path_folder_ipv = Path('dataset/ipv') #to write ipv
path_folder_ipc_cut = Path('dataset/ipc_cut') #to write ipc cut
path_folder_ipv_cut = Path('dataset/ipv_cut') #to write ipv cut
makedir(path_folder_ipc)
makedir(path_folder_ipv)
makedir(path_folder_ipc_cut)
makedir(path_folder_ipv_cut)

path_folder_train_ipc = Path('{}/train_ipc/images'.format(path_folder_ipc))
makedir(path_folder_train_ipc)
path_folder_mask_ipc = Path('{}/train_ipc/masks'.format(path_folder_ipc))
makedir(path_folder_mask_ipc)
path_folder_masked_ipc = Path('{}/train_ipc/masked_images'.format(path_folder_ipc))
makedir(path_folder_masked_ipc)

path_folder_train_ipv = Path('{}/train_ipv/images'.format(path_folder_ipv))
makedir(path_folder_train_ipv)
path_folder_mask_ipv = Path('{}/train_ipv/masks'.format(path_folder_ipv))
makedir(path_folder_mask_ipv)
path_folder_masked_ipv = Path('{}/train_ipv/masked_images'.format(path_folder_ipv))
makedir(path_folder_masked_ipv)

path_folder_train_ipc_cut = Path('{}/train_ipc/images'.format(path_folder_ipc_cut))
makedir(path_folder_train_ipc_cut)
path_folder_mask_ipc_cut = Path('{}/train_ipc/masks'.format(path_folder_ipc_cut))
makedir(path_folder_mask_ipc_cut)
path_folder_masked_ipc_cut = Path('{}/train_ipc/masked_images'.format(path_folder_ipc_cut))
makedir(path_folder_masked_ipc_cut)

path_folder_train_ipv_cut = Path('{}/train_ipv/images'.format(path_folder_ipv_cut))
makedir(path_folder_train_ipv_cut)
path_folder_mask_ipv_cut = Path('{}/train_ipv/masks'.format(path_folder_ipv_cut))
makedir(path_folder_mask_ipv_cut)
path_folder_masked_ipv_cut = Path('{}/train_ipv/masked_images'.format(path_folder_ipv_cut))
makedir(path_folder_masked_ipv_cut)

if False:
    # Replace " " by "_" in file_names
    file_names = os.listdir(path_folder)
    for file_name in file_names:
        new_name = file_name.replace(" ","_")
        os.rename(os.path.join(path_folder,file_name), os.path.join(path_folder,new_name))

patient_ids = [ name for name in os.listdir(path_folder_all) if os.path.isdir(os.path.join(path_folder_all, name))]
patient_ids = list(map(int, patient_ids))
patient_ids.sort()
#patient_ids = [45] #comment to use all

for patient_id in patient_ids:
    
    patient_id = str(patient_id)

    patient_ipv = os.path.join(path_folder_all, patient_id, patient_id + "_IPV.jpg") #sem corante
    patient_ipc = os.path.join(path_folder_all, patient_id, patient_id + "_IPC.jpg") #com corante
    patient_csv = os.path.join(path_folder_all,patient_id, patient_id + "_csv.csv") #polígono segmentação

    image_ipc = cv.imread(patient_ipc,cv.IMREAD_COLOR) #UNCHANGED não lê informação de rotação automática EXIF metadata
    image_ipc = cv.cvtColor(image_ipc,cv.COLOR_BGR2RGB)
    image_ipv = cv.imread(patient_ipv,cv.IMREAD_COLOR) #UNCHANGED não lê informação de rotação automática EXIF metadata
    image_ipv = cv.cvtColor(image_ipv,cv.COLOR_BGR2RGB) 

    # Opens csv created by (exported as csv) https://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html
    # Build coordinates list to fill later
    data = []
    try:
        with open(patient_csv) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:            
                    line_count += 1
                else:
                    data.append(row)
                    line_count += 1
        try:
            data_ipc = '{""name'+data[0][0].split("name")[1]
            data_ipc = data_ipc.replace('""','"')
            data_ipc = data_ipc.split('","{}"')[0]
            data_ipv = '{""name'+data[1][0].split("name")[1]
            data_ipv = data_ipv.replace('""','"')
            data_ipv = data_ipv.split('","{}"')[0]
        except:
            try:
                data_ipc = data[0][5]
                data_ipv = data[1][5]
            except:
                print("Erro no split")
        
        image_size = (1000,3000)

        #Sem corante
        data_dic_ipv = json.loads(data_ipv)
        coord_x_ipv = np.array(data_dic_ipv["all_points_x"]) 
        coord_y_ipv = np.array(data_dic_ipv["all_points_y"])
        coords_ipv = np.array(list(zip(coord_x_ipv,coord_y_ipv)))
        col_min_ipv = (coord_x_ipv.min()+(coord_x_ipv.max()-coord_x_ipv.min())//2)-image_size[1]//2
        row_min_ipv = (coord_y_ipv.min()+(coord_y_ipv.max()-coord_y_ipv.min())//2)-image_size[0]//2


        #Com corante
        data_dic_ipc = json.loads(data_ipc)
        coord_x_ipc = np.array(data_dic_ipc["all_points_x"]) 
        coord_y_ipc = np.array(data_dic_ipc["all_points_y"])
        coords_ipc = np.array(list(zip(coord_x_ipc,coord_y_ipc)))
        col_min_ipc = (coord_x_ipc.min()+(coord_x_ipc.max()-coord_x_ipc.min())//2)-image_size[1]//2
        row_min_ipc = (coord_y_ipc.min()+(coord_y_ipc.max()-coord_y_ipc.min())//2)-image_size[0]//2

        if True:

            #IPC -- Creates mask from coordinates and save it as image

            img_bg = np.zeros_like(image_ipc)
            mask_ipc = cv.fillPoly(img_bg, pts =[coords_ipc], color=(1,1,1))
            masked_ipc = image_ipc*mask_ipc

            # Original image
            mask_path = os.path.join(path_folder_mask_ipc, patient_id+"_mask.png")
            img_path = os.path.join(path_folder_train_ipc, patient_id+".jpg")
            maskedimg_path = os.path.join(path_folder_masked_ipc, patient_id+".jpg")

            cv.imwrite(mask_path, mask_ipc*255)
            cv.imwrite(img_path, cv.cvtColor(image_ipc, cv.COLOR_RGB2BGR))
            cv.imwrite(maskedimg_path, cv.cvtColor(masked_ipc, cv.COLOR_RGB2BGR))

            # Cropped image
            image_ipc = image_ipc[row_min_ipc:row_min_ipc+image_size[0],col_min_ipc:col_min_ipc+image_size[1],:]
            mask_ipc  = mask_ipc[row_min_ipc:row_min_ipc+image_size[0],col_min_ipc:col_min_ipc+image_size[1],:]
            masked_ipc = masked_ipc[row_min_ipc:row_min_ipc+image_size[0],col_min_ipc:col_min_ipc+image_size[1],:]

            mask_path_cut = os.path.join(path_folder_mask_ipc_cut, patient_id+"_mask.png")
            img_path_cut = os.path.join(path_folder_train_ipc_cut, patient_id+".jpg")
            maskedimg_path_cut = os.path.join(path_folder_masked_ipc_cut, patient_id+".jpg")

            cv.imwrite(mask_path_cut, mask_ipc*255)
            cv.imwrite(img_path_cut, cv.cvtColor(image_ipc, cv.COLOR_RGB2BGR))
            cv.imwrite(maskedimg_path_cut, cv.cvtColor(masked_ipc, cv.COLOR_RGB2BGR))

            #IPV -- Creates mask from coordinates and save it as image
            img_bg = np.zeros_like(image_ipv)
            mask_ipv = cv.fillPoly(img_bg, pts =[coords_ipv], color=(1,1,1))
            masked_ipv = image_ipv*mask_ipv

            # Original image
            mask_path = os.path.join(path_folder_mask_ipv, patient_id+"_mask.png")
            img_path = os.path.join(path_folder_train_ipv, patient_id+".jpg")
            maskedimg_path = os.path.join(path_folder_masked_ipv, patient_id+".jpg")

            cv.imwrite(mask_path, mask_ipv*255)
            cv.imwrite(img_path, cv.cvtColor(image_ipv, cv.COLOR_RGB2BGR))
            cv.imwrite(maskedimg_path, cv.cvtColor(masked_ipv, cv.COLOR_RGB2BGR))
            
            # Cropped image

            image_ipv = image_ipv[row_min_ipv:row_min_ipv+image_size[0],col_min_ipv:col_min_ipv+image_size[1],:]
            mask_ipv  = mask_ipv[row_min_ipv:row_min_ipv+image_size[0],col_min_ipv:col_min_ipv+image_size[1],:]
            masked_ipv = masked_ipv[row_min_ipv:row_min_ipv+image_size[0],col_min_ipv:col_min_ipv+image_size[1],:]

            mask_path_cut = os.path.join(path_folder_mask_ipv_cut, patient_id+"_mask.png")
            img_path_cut = os.path.join(path_folder_train_ipv_cut, patient_id+".jpg")
            maskedimg_path_cut = os.path.join(path_folder_masked_ipv_cut, patient_id+".jpg")

            cv.imwrite(mask_path_cut, mask_ipv*255)
            cv.imwrite(img_path_cut, cv.cvtColor(image_ipv, cv.COLOR_RGB2BGR))
            cv.imwrite(maskedimg_path_cut, cv.cvtColor(masked_ipv, cv.COLOR_RGB2BGR))


            #Plot cropped images
            fig = plt.figure(figsize = (15,5))
            plt.suptitle("patient_id: "+patient_id)
            plt.subplot(1,3,1)
            plt.imshow(image_ipc)
            plt.subplot(1,3,2)
            plt.imshow(image_ipc)
            plt.imshow(mask_ipc, alpha=0.3)
            plt.subplot(1,3,3)
            plt.imshow(masked_ipc)
            plt.show()
            path_comp = os.path.join(path_folder_ipc_cut, patient_id+"_comp.png")
            fig.savefig(path_comp)

            fig = plt.figure(figsize = (15,5))
            plt.suptitle("patient_id: "+patient_id)
            plt.subplot(1,3,1)
            plt.imshow(image_ipv)
            plt.subplot(1,3,2)
            plt.imshow(image_ipv)
            plt.imshow(mask_ipv, alpha=0.3)
            plt.subplot(1,3,3)
            plt.imshow(masked_ipv)
            plt.show()
            path_comp = os.path.join(path_folder_ipv_cut, patient_id+"_comp.png")
            fig.savefig(path_comp)

        if False:
            fig = plt.figure(figsize = (15,5))
            plt.suptitle("patient_id: "+patient_id)
            plt.subplot(1,3,1)
            plt.imshow(image_ipv)
            plt.subplot(1,3,2)
            plt.imshow(mask_ipv,'gray')
            plt.subplot(1,3,3)
            plt.imshow(image_ipv)
            plt.imshow(mask_ipv, alpha=0.3)
            plt.show()
            path_comp = os.path.join(path_folder_ipv, patient_id+"_comp.png")
            fig.savefig(path_comp)
    except:
        print("Erro csv: "+ str(patient_id))

# %%
