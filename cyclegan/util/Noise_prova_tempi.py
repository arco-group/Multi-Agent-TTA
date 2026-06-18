import numpy as np
#from utils import dataset_ufficiale
#from Dataset import CTDataset
from torch.utils.data import DataLoader
import torch
import cv2
from itertools import combinations
import random
#import pyiqa

#AGGIUNTA GAUSSIAN NOISE
def add_gaussian_noise(image, std):
    image=image.cpu()
    np.random.seed(100)
    noise = np.random.normal(0, std, image.shape)
    noise.reshape(image.shape)  
    noisy_image = image + noise
    return noisy_image

#AGGIUNTA POSSOIN NOISE
def poisson(image, level):
    image=image.cpu()
    np.random.seed(100)
    noisy = np.random.poisson(level, size=image.shape) 
    poisson_img=image+noisy
    #poisson_img = np.clip(poisson_img, -1, 1)
    return poisson_img

#AGGIUNTA SALT&PEPPER 
# def saltpepper(image, amount):
#     image=image.cpu()
#     s_vs_p = 0.5
#     out = np.copy(image)
#       # Salt mode
#     num_salt = np.ceil(amount * image.numel() * s_vs_p)  #numero di pixel da convertire in salt  #mi da massimo 26.215
#     num_pepper = np.ceil(amount* image.numel() * (1. - s_vs_p))  #numero di pixel da convertire in pepper
#     num_1=0
#     num_2=0
#     while num_1<num_salt:
#      # valori = np.linspace(0, 512, numero_di_valori)
#      # valori=np.linspace(0, 512, numero_di_valori)  #me li da come array, per prenderne uno faccio valori[0]
#      #coords_1=valori[num_1]
#      #coords_2=valori[num_2]
#       coords_1 = np.random.randint(0, image.shape[1] - 1) #numero random da 0 a 512
#       coords_2 = np.random.randint(0, image.shape[2] -1)  #numero random da 0 a 512
#       num_1 +=1
#       out[0][coords_1][coords_2] = 1  #aggiunta di salt
#     while num_2<num_pepper:
#       coords_11 = np.random.randint(0, image.shape[1] - 1) #numero random da 0 a 512
#       coords_22 = np.random.randint(0, image.shape[2] -1)  #numero random da 0 a 512
#       num_2 +=1
#       out[0][coords_11][coords_22] = -1  #aggiunta di pepper
#     out=torch.from_numpy(out)
#     return out

def saltpepper(image, amount):
    image=image.cpu()
    s_vs_p = 0.5
    out = np.copy(image)
      # Salt mode
    num_salt = np.ceil(amount * image.numel() * s_vs_p)  #numero di pixel da convertire in salt  #mi da massimo 26.215
    num_pepper = np.ceil(amount* image.numel() * (1. - s_vs_p))  #numero di pixel da convertire in pepper
    num_1=0
    num_2=0
    num_salt=int(num_salt)
    valori_float_coord1=np.linspace(0, 511, num_salt) #me li da come array, per prenderne uno faccio valori[0]
    valori_coord1 = valori_float_coord1.astype(int)
    #print('valori_coord1', valori_coord1)
    valori_float_coord2=np.linspace(0, 400, num_salt) #me li da come array, per prenderne uno faccio valori[0]
    valori_coord2 = valori_float_coord2.astype(int)
    #print('valori_coord2', valori_coord2)
    while num_1<num_salt:
     # valori = np.linspace(0, 512, numero_di_valori)
      coords_1=valori_coord1[num_1]
      coords_2=valori_coord2[num_1]
      num_1 +=1
      out[0][coords_1][coords_2] = 1  #aggiunta di salt
    num_pepper=int(num_pepper)
    valori_pepper_float_coord1=np.linspace(0, 300, num_pepper)
    valori_pepper_coord1 = valori_pepper_float_coord1.astype(int)
    #print ('valori_pepper_coord1', valori_pepper_coord1)
    valori_pepper_float_coord2=np.linspace(0, 500, num_pepper)
    valori_pepper_coord2 = valori_pepper_float_coord2.astype(int)
    #print ('valori_pepper_coord2', valori_pepper_coord2)
    while num_2<num_pepper:
      coords_11=valori_pepper_coord1[num_2]
      coords_22=valori_pepper_coord2[num_2]
      num_2 +=1
      out[0][coords_11][coords_22] = -1  #aggiunta di pepper
    out=torch.from_numpy(out)
    return out





#AGGIUNTA BLURRED NOISE 
def blurred(image, kernel):
  image=image.cpu()
  image=image.numpy()
  blurred = cv2.GaussianBlur(image, (kernel, kernel), 0)
  blurred=torch.from_numpy(blurred)
  return blurred

#FUNZIONE PER ESTRARRE UN NUMERO ALL'INTERNO DELL'INTERVALLO
def random_function(start, end, step): 
    num_steps = int((end - start) / step) + 1  #calcola il numero di step nell'intervallo [start, end] 
    random_step = random.randint(0, num_steps - 1)  #genera un numero casuale da 0 al numero di step
    random_value = start + random_step * step  #moltiplica il numero casuale per il numero di step totali e li somma allo start
    return random_value  #restituisce un numero casuale all'interno dell'intervallo selezionato

#FUNZIONE COMBINAZIONE RUMORI
def combinations_noise(image):
  image=image.cpu()
  all_noise=['gaussian', 'poisson', 'sp', 'blurred']  
  immagine_corrotta=[]
  for r in range (2, 5):  #da 2 a 4 rumori combinati
    output=list(combinations(all_noise, r)) #combina i rumori, prima tra 2, poi 3, poi 4, e crea una lista 
    for i in output:
      #print ('i', i)
      #Se il rumore è nell'i-esimo elemento della lista viene applicato, altrimenti no
      if 'gaussian' in i:
        #considero il rumore minimo per sigma 
        sigma=0.02  #random_function(0.02, 0.5, 0.02) #per ogni rumore viene estratto un valore randomico all'interno dell'intervallo selezionato
        gauss_img=add_gaussian_noise(image, sigma)
      else:
        gauss_img=image
      if 'poisson' in i:
        l=0.02    #random_function(0.02, 0.5, 0.02)
        poisson_img=poisson (image, l)
      else: 
        poisson_img=image
      if 'sp' in i:
        am=0.008     #random_function(0.008, 0.2, 0.008) 
        sp_img=saltpepper(image, am)
      else:
        sp_img=image
      if 'blurred' in i:
        k=5      #random_function(5, 101, 4)
        img_blurred = blurred(image, k)
      else:
        img_blurred=image
        
      new_image=(gauss_img+poisson_img+sp_img+img_blurred)/4 #calcolo la media tra le immagini ottenute 
      immagine_corrotta.append(new_image)  #metto tutte le immagini rumorose nella lista
  #print(len(immagine_corrotta))               
  return immagine_corrotta  # (11x([1 512 512]))-> esce una lista di array che rappresentano l'immagine corrotta 11 volte 

#CALCOLO PSNR
def calculate_psnr(image1, image2):
    # Converte i tensori in float32
    image1 = image1.float()
    image2 = image2.float()
    mse = torch.mean((image1 - image2) ** 2)            #differenza tra le due immagini al quadrato
    if mse == 0:
        return float('inf')                             #se sono uguali mse=0, restituisce inf
    dynamic_range = torch.max(image1) - torch.min(image1)
    psnr = 10 * torch.log10((dynamic_range ** 2) / mse)   #calcolo psnr
    return psnr

#CALCOLO MEDIA
def calcola_media (lista):
  somma = sum(lista)
  media = somma / len(lista)
  return media


#CAMBIA FORMATO IMMAGINE DA [1 1 SIZE SIZE] A [1 3 SIZE SIZE]
def change_3d(image):
    image_3d = torch.empty((image.shape[0], 3, image.shape[2], image.shape[3]), dtype=torch.float32)  #creo un tensore vuoto di dimensioni [1 3 512 512]
    im = image[0, 0, :, :]  #immagine a  singolo canale (grayscale)-> prende l'immagine, canale unico
    #assegno lo stesso valore (im) ai 3 canali della nuova immagine 
    image_3d[0, 0, :, :] = im  
    image_3d[0, 1, :, :] = im  
    image_3d[0, 2, :, :] = im  
    return image_3d



def normalize_img(x, lower=None, upper=None, data_range='-11'):  #x immagine da normalizzare, lower e upper valori minimi e massimi immagine, normalizzazione tra -1 e 1
        x_norm = (x - torch.min(x)) / (torch.max(x) - torch.min(x))  # map between 0 and 1
        if data_range == '01':
            return x_norm
        else:
            return (2 * x_norm) - 1  #map between -1 and 1 (i valori minimi dei pixel saranno -1, i massimi 1)

def norm_prova (image):
   mean=torch.mean(image)
   std=torch.std(image)
   img=(image-mean)/std
   return img



def calcola_unpaired(metrica, batch):
   if metrica=='niqe':
      niqe = pyiqa.create_metric('niqe', device=torch.device("cuda:0"), as_loss=True, loss_reduction='none')                 #non la uso come funzione di perdita
      niqe_score = niqe(batch)
      return niqe_score
   
   if metrica=='brisque':
      brisque=pyiqa.create_metric('brisque', device=torch.device("cuda:0"), as_loss=True, loss_reduction='none') 
      brisque_score=brisque(batch)
      return brisque_score
   if metrica=='paq2piq':
      new_3d=change_3d(batch)
      paq2piq = pyiqa.create_metric('paq2piq', device=torch.device("cuda:0"), as_loss=True, loss_reduction='none') 
      paq2piq_score=paq2piq(new_3d)
      return paq2piq_score
   if metrica=='ilniqe':
      new_3d=change_3d(batch)
      il_niqe=pyiqa.create_metric('ilniqe', device=torch.device("cuda:0"), as_loss=True, loss_reduction='none') 
      il_score=il_niqe(new_3d)
      return il_score 
   if metrica=='musiq-ava':
      new_3d=change_3d(batch)
      musiq_ava=pyiqa.create_metric('musiq-ava', device=torch.device("cuda:0"), as_loss=True, loss_reduction='none') 
      musiq_ava_score=musiq_ava(new_3d)
      return musiq_ava_score
   if metrica=='musiq-koniq':
      new_3d=change_3d(batch)
      musiq_koniq=pyiqa.create_metric('musiq-koniq', device=torch.device("cuda:0"), as_loss=True, loss_reduction='none') 
      musiq_koniq_score=musiq_koniq(new_3d)
      return musiq_koniq_score
   if metrica=='musiq-paq2piq':
      new_3d=change_3d(batch)
      musiq_paq2piq=pyiqa.create_metric('musiq-paq2piq', device=torch.device("cuda:0"), as_loss=True, loss_reduction='none') 
      musiq_paq2piq_score=musiq_paq2piq(new_3d)
      return musiq_paq2piq_score
   if metrica=='musiq-spaq':
      new_3d=change_3d(batch)
      musiq_spaq=pyiqa.create_metric('musiq-spaq', device=torch.device("cuda:0"), as_loss=True, loss_reduction='none') 
      musiq_spaq_score=musiq_spaq(new_3d)
      return musiq_spaq_score
   if metrica=='dbcnn':
      new_3d=change_3d(batch)
      dbcnn=pyiqa.create_metric('dbcnn', device=torch.device("cuda:0"), as_loss=True, loss_reduction='none') 
      dbcnn_score=dbcnn(new_3d)
      return dbcnn_score
   if metrica=='hyperiqa':
      new_3d=change_3d(batch)
      hyper=pyiqa.create_metric('hyperiqa', device=torch.device("cuda:0"), as_loss=True, loss_reduction='none') 
      hyper_score=hyper(new_3d)
      return hyper_score
   if metrica=='clipiqa':
      new_3d=change_3d(batch)
      clip=pyiqa.create_metric('clipiqa', device=torch.device("cuda:0"), as_loss=True, loss_reduction='none') 
      clip_score=clip(new_3d)
      return clip_score
   if metrica=='nima':
      new_3d=change_3d(batch)
      nima=pyiqa.create_metric('nima', device=torch.device("cuda:0"), as_loss=True, loss_reduction='none') 
      nima_score=nima(new_3d)
      return nima_score
   



def calcola_L1 (image, target):
   loss=torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')
   l1=loss(image, target).item()
   return (l1)