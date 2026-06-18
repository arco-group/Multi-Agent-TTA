import torch
import torch.nn as nn
import torch.optim as optim
from models.UNet import UNet
from models.Autoencoder_model import AENet
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
# from util.visualizer import Visualizer
import pandas as pd
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

torch.manual_seed(42)  # garantisce la riproducibilità
torch.cuda.manual_seed(42)
from options.base_options import BaseOptions
from tqdm import tqdm

# scriviamo una funzione per ricalcolare tnet_dim a seconda di cosa c'è in return_layers
def compute_tnet_dim(opt):
    layers_to_dim = {'input': 1, 'first_conv': 64, 'second_conv': 128, 'third_conv': 256, 'resnet_block_1': 256,
                     'resnet_block_2': 256, 'resnet_block_3': 256, 'resnet_block_4': 256, 'final_output': 1}
    tnet_dim = []
    for layer in opt.return_layers:
        tnet_dim.append(layers_to_dim[layer])
    opt.tnet_dim = tnet_dim

def save_images(real_image, reconstructed_image, epoch, path, name, batch_size):
    path = os.path.join(path, f'epoch{epoch}')
    path = os.path.join(path, 'saved_images')

    # Converte i tensori in numpy array e poi in immagini PIL
    real_image = (real_image.squeeze().cpu().detach().numpy())
    reconstructed_image = (reconstructed_image.squeeze().cpu().detach().numpy())

    # se il batch size è maggiore di 1, ripete il processo per tutte le immagini
    if batch_size > 1:
        # se batch_size è maggiore della dimensione dell'ultimo batch, riduci il batch_size per evitare errori
        if batch_size > 8:
            batch_size = 8
        if batch_size > real_image.shape[0]:
            batch_size = real_image.shape[0]
        # se è maggiore di 8 lo limito a 8
        for i in range(batch_size):
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(real_image[i], cmap='gray')
            plt.subplot(1, 2, 2)
            plt.imshow(reconstructed_image[i], cmap='gray')
            os.makedirs(path, exist_ok=True)
            plt.savefig(f'{path}/{name}_epoch_{epoch}_img_{i}.png')
    else:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(real_image, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed_image, cmap='gray')
        # Crea directory se non esiste
        os.makedirs(path, exist_ok=True)
        plt.savefig(f'{path}/{name}_epoch_{epoch}.png')

def plot_ae_losses(loss_list, num_epochs, path, name):

    epochs = np.arange(1, num_epochs + 1)  # crea un array con il numero di epoche

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_list, label='AE Loss', marker='o')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Autoencoder Losses Over Epochs')
    plt.legend()
    plt.grid(True)
    #plt.show()
    # salviamo il grafico
    plt.savefig(os.path.join(path, f'train_ind_losses_{name}.png'))
    return name


def training_AE(AENet, task_model, dataset, opt, AE_to_train='all'):
    # AE=AENet.AENet[AE_to_train]
    if opt.model == 'pix2pix':
        task_model.set_requires_grad([task_model.netG, task_model.netD], False)
    else:
        task_model.set_requires_grad([task_model.netG_A, task_model.netG_B, task_model.netD_A, task_model.netD_B], False)
    # il task model è freezato durante addestramento di AE (vanno freezati i singoli pezzi perchè nel codice base model, è scritto così.
    AENet.set_requires_grad(AENet.AENet, True)  # verranno trainati solo AE
    # il .AENet contiene la lista dei 4 AE che devono essere trainati

    task_model.eval()  # metto il task model in modalità inferenza
    AENet.train()  # metto l'autoencoder in modalità trainingc

    if AE_to_train == 'all':
        ae_indices = range(len(AENet.AENet))
    else:
        ae_indices = [AE_to_train]
    weights = opt.__dict__.get('weights', [1] * len(AENet.AENet))  # TODO: da capire la questione weights

    loss_list_epoch = {i: [] for i in ae_indices}
    loss_list_batch = {i: [] for i in ae_indices}

    for epoch in range(opt.total_epoch):
        for batch in tqdm(dataset.dataloader): # qui cicliamo sul dataloader in modo da avere le batch
            task_model.set_input(batch)  # imposta il modello
            outputs = task_model.forward(return_layers=opt.return_layers)  # passo forward del task model

            for i in ae_indices:
                ae = AENet.AENet[i]
                optimizer = AENet.optimizers[i]

                optimizer.zero_grad()

                index = opt.return_layers[i]
                side_out = outputs[index]

                # prepara input
                if len(AENet.AENetMatch[i]) == 2:
                    side_out_cat_orig = torch.cat([side_out[0], side_out[1]], dim=1)
                    side_out_cat = torch.cat([AENet.addnoise(side_out[0]), AENet.addnoise(side_out[1])], dim=1)
                else:
                    side_out_cat_orig = side_out
                    side_out_cat = AENet.addnoise(side_out)

                # forward AE + loss
                if i == len(AENet.AENetMatch) - 1 and opt.use_gt:
                    label_noise = AENet.addnoise(task_model.real_B.unsqueeze(1))
                    ae_out = ae(label_noise, side_out=False)
                    level_loss = weights[i] * AENet.AELoss(ae_out, task_model.real_B.unsqueeze(1))
                else:
                    ae_out = ae(side_out_cat, side_out=False)
                    level_loss = weights[i] * AENet.AELoss(ae_out, side_out_cat_orig)

                # backward & step
                level_loss.backward()
                optimizer.step()

                # log
                loss_list_batch[i].append(level_loss.item())

        AENet.update_learning_rate()  # update learning rates in the beginning of every epoch.

        # stampa loss per ogni AE
        for i in ae_indices:
            print(f"[Epoch {epoch+1}] AE_{i+1} loss: {loss_list_batch[i][-1]:.6f}")
            loss_list_epoch[i].append(loss_list_batch[i][-1])

        if epoch % 5 == 0 or epoch == opt.total_epoch - 1:
            for i in ae_indices:
                AENet.save_networks_AE(epoch, i)

                # Salva immagini reali e ricostruite ogni 5 epoche
                with torch.no_grad():  # Disabilita il calcolo dei gradienti per la valutazione
                    fake_images_B = task_model.fake_B  # Immagini reali output
                    real_images_A = task_model.real_A  # Immagini reali input
                    ae_reconstructed_B = AENet.AENet[-1](
                        AENet.addnoise(fake_images_B))  # Passa attraverso l'AE per ottenere le ricostruzioni
                    ae_reconstructed_A = AENet.AENet[0](AENet.addnoise(real_images_A))

                    # Salva le immagini
                    if i == 0:
                        save_images(real_images_A, ae_reconstructed_A, epoch, AENet.save_dir, 'input', opt.batch_size)
                    elif i == 8:
                        save_images(fake_images_B, ae_reconstructed_B, epoch, AENet.save_dir, 'output', opt.batch_size)
    return loss_list_epoch, loss_list


if __name__ == '__main__':
    opt = TrainOptions().parse()
    compute_tnet_dim(opt)
    print(opt.tnet_dim)
    task_model = create_model(opt)  # creo il task model
    dataset = create_dataset(opt)  # ci da il dataloader
    AENet = AENet(opt)  # creo l'autoencoder
  
    #NB: ricorda che la p2p non ha net.G_a ma solo net.G
    if opt.model == 'pix2pix':
        load_path = os.path.join(opt.checkpoints_dir.replace('_rec_models', ""), opt.name,'latest_net_G.pth')
    else:
        load_path = os.path.join(opt.checkpoints_dir.replace('_rec_models', ""), opt.name,'latest_net_G_A.pth')
        
    state_dict = torch.load(load_path, map_location=str(task_model.device)) # dizionario che ha per chiave il nome del layer
    # e per valore i pesi

    if opt.model == 'pix2pix':
        if isinstance(task_model.netG, torch.nn.DataParallel): # se parallelizzo il modello (uso più di 1 gpu)
            task_model.netG.module.load_state_dict(state_dict) # parallelizzo ogni rete sulla gpu ( T model ha 4 reti)
        else:
            task_model.netG.load_state_dict(state_dict)
    else:
        if isinstance(task_model.netG_A, torch.nn.DataParallel): # se parallelizzo il modello (uso più di 1 gpu)
            task_model.netG_A.module.load_state_dict(state_dict) # parallelizzo ogni rete sulla gpu ( T model ha 4 reti)
        else:
            task_model.netG_A.load_state_dict(state_dict)

    # AE_to_train = int(opt.AE_to_train)
    loss_list_epoch, loss_list = training_AE(AENet, task_model, dataset, opt, opt.AE_to_train)

