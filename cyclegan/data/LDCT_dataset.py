import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import os
import random
import pandas as pd
import pydicom
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
#from Noise import add_gaussian_noise, poisson, calculate_psnr, saltpepper, calcola_media, blurred, combinations_noise, change_3d, normalize_img, norm_prova, calcola_unpaired, calcola_L1
#from piq import brisque
import pyiqa
import csv
import time


class LDCTDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser

        Returns:
            the modified parser.
        """
        ####
        parser.add_argument('--annotation_A', type=str, default='annotations/ldct_lowdose.csv',
                            help='Path of the csv_file containing the directories of the images')
        parser.add_argument('--annotation_B', type=str, default='annotations/ldct_fulldose.csv',
                            help='Path of the csv_file containing the directories of the images')
        parser.add_argument('--window_width', type=int, default=1400,
                            help='Window width specifies the rage of HU values to display')
        parser.add_argument('--window_center', type=int, default=-400,
                            help='It specifies the center of the selected HU window')
        parser.add_argument('--height', type=int, default=512, help='Height of the image')
        parser.add_argument('--width', type=int, default=512, help='Width of the image')

        return parser
    
    @staticmethod
    def convert_in_hu(dicom_file):  # apply a linear transformations to get the HU values (y = m*x + q)-> trasformo i pixel in HU attraverso i metadati (intercept e slope)
        image = dicom_file.pixel_array  
        intercept = dicom_file.RescaleIntercept
        slope = dicom_file.RescaleSlope
        image = slope * image + intercept
        return image #mi restituisce l'immagine in HU

    @staticmethod
    def normalize_img(x, lower=None, upper=None, data_range='-11'):  #x immagine da normalizzare, lower e upper valori minimi e massimi immagine, normalizzazione tra -1 e 1
        if np.max(x)!=np.min(x):
            
            #print ('uguali', np.max(x))
        #else:
            x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))  # map between 0 and 1
        if data_range == '01':
            return x_norm
        else:
            return (2 * x_norm) - 1  #map between -1 and 1 (i valori minimi dei pixel saranno -1, i massimi 1)

    @staticmethod
    def plot_img(x, pname):
        x = x.detach().cpu().numpy()
        x = x[0, :, :]
        plt.imshow(x, cmap='gray')
        plt.title(pname)
        plt.axis('off')
        plt.show()

    def __init__(self, opt): # annotation_file_A, annotation_file_B, height,width, window_width, window_center, opt): #file annotation, dimensioni immagine, finestra di clip
        # self.annotations_file = annotation_file
        BaseDataset.__init__(self, opt)
        self.annotations_A = pd.read_csv(opt.annotation_A) #legge il file annotations
        self.annotations_B = pd.read_csv(opt.annotation_B)

        self.window_width = opt.window_width
        self.window_center = opt.window_center
        #Resize shape
        self.height = opt.height
        self.width = opt.width
        self.A_size = len(self.annotations_A)  # get the size of dataset A
        self.B_size = len(self.annotations_B)  # get the size of dataset B
        self.dataset_len = max(self.A_size, self.B_size)
        #self.plot_verbose = opt.plot_verbose
        # btoA = self.opt.direction == 'BtoA'

        self.plot_verbose = False

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, A_paths 
            A (tensor)       -- an image in the input domain
            A_paths (str)    -- image paths
        """
       
        img_path_A = self.annotations_A['img_path'].iloc[index]  #prende i path alle immagini
        img_name_A=self.annotations_A['img_name'].iloc[index] + img_path_A.split('/')[-1].replace('.dcm', '')
        img_raw_A = pydicom.dcmread(img_path_A, force=True) #legge l'immagine dal percorso
        img_A = self.transforms(img_raw_A) #pre-processing immagine 

        img_path_B = self.annotations_B['img_path'].iloc[index]  #prende i path alle immagini
        img_name_B=self.annotations_B['img_name'].iloc[index] + img_path_B.split('/')[-1].replace('.dcm', '')
        img_raw_B = pydicom.dcmread(img_path_B, force=True) #legge l'immagine dal percorso
        img_B = self.transforms(img_raw_B) #pre-processing immagine 

        return {'A': img_A, 'B': img_B, 'A_paths': img_path_A,  'B_paths': img_path_B, 'A_name': img_name_A, 'B_name': img_name_B, "modality_mapping": {"A": "LDCT","B": 'HDCT'}}
    
    #Return dataset length
    def __len__(self):
        """Return the total number of images in the dataset.
        """
        return self.dataset_len

    #Select the display window based on: window center and window width
    def window_image(self, hu_img): 
        img_w = hu_img.copy()
        img_min = self.window_center - self.window_width // 2
        img_max = self.window_center + self.window_width // 2
        img_w[img_w < img_min] = img_min
        img_w[img_w > img_max] = img_max
        return img_w #restituisce l'immagine in una finestra specifica

    #Apply preprocessing
    def transforms(self, dicom, tensor_output=True):
        x = self.convert_in_hu(dicom)  #converte le immagini in HU
        x = self.window_image(x)  #seleziona la finestra specifica
        x = self.normalize_img(x) #normalizza
         
        x = cv2.resize(x, (self.height, self.width))  #sceglie le dimensioni dell'immagine
        
        if tensor_output:  #se tensor_output
            x = torch.from_numpy(x)  #converte in tensore torch
            x = x.unsqueeze(dim=0)  #aggiunge una dimensione all'inizio del tensore
            return x.float()
        else:
            return x.astype('float32')

    
