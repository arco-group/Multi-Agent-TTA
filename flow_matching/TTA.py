from models.adaptor_3 import DTTAnorm, ANet
import torch
import torch.nn as nn
import torch.optim as optim
from models.UNet import UNet
from models.Autoencoder_model import AENet
import time
import pandas as pd
import os
import torch
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import random
from tqdm import tqdm
import warnings
import re
import matplotlib.pyplot as plt
from train_monitoring_agent import compute_tnet_dim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr, structural_similarity as compute_ssim

warnings.filterwarnings("ignore")

torch.manual_seed(0)  # garantisce la riproducibilità
torch.cuda.manual_seed(0)
np.random.seed(0)


def l2_reg_ortho(model, lambda_l2=1e-4):
    l2_loss = torch.tensor(0.0, device='cuda')
    for param in model.parameters():
        if param.requires_grad:
            l2_loss += torch.norm(param, p=2) ** 2  # norma L2 dei parametri
    return lambda_l2 * l2_loss


def TTA_rndm_50(adaptors, opt, task_model, save_dir, batch, x_minus_one, t_tensor, scheduler, rec_loss, stable=False, return_layers=None, plot=False, psnr_score_no_tta=0.0):
    opt.criteria = 'PSNR'
    n = len(return_layers[1:-1])  # i ricostruttori escluso il primo e l'ultimo
    indexs = [i for i in range(n)]
    tutte_combinazioni = set()  # è un insieme vuoto senza duplicati
    # numero di combinazioni casuali da provare
    num_random_comb = opt.__dict__.get('num_random_comb', 50)  # numero di combinazioni casuali da provare
    while len(tutte_combinazioni) < num_random_comb:
        r = random.randint(1, len(indexs))  # Lunghezza casuale della combinazione
        tutte_combinazioni.add(tuple(sorted(random.sample(indexs, r))))  # aggiungo la combinazione alla lista
    tutte_combinazioni = list(tutte_combinazioni)  # Converti il set in una lista se serve

    tutte_combinazioni = [tuple([l for l in indexs])] # SE ABLATION SULLE COMBINAZIONI
    adaptors.set_requires_grad([adaptors.adpNet, adaptors.conv], True)  #
    adaptors.train()
    orthw = opt.__dict__.get('orthw', 1)

    rec_loss = round(rec_loss.item(), 4)

    loss_config_output = pd.DataFrame(columns=['config', 'loss_output', 'loss_tot', 'PSNR'])
    calcola_mae = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')

    for comb in tutte_combinazioni:
        chosen_comb = list(comb)  # prendo una combinazione
        chosen_comb = sorted(chosen_comb)  # la ordino
        comb_to_print = chosen_comb
        chosen_comb = [0] + [x + 1 for x in chosen_comb] + [n + 1]
        chosen_comb = [return_layers[i] for i in chosen_comb]  # scelgo i layer corrispondenti
        opt.return_layers = chosen_comb
        compute_tnet_dim(opt)

        load_path_weights_AE = os.path.join(opt.diff_ckpt + '_rec_models', 'epoch49')
        AE = AENet(opt)
        for i in range(len(opt.return_layers)):
            name = opt.return_layers[i]
            elem = opt.tnet_dim[i]
            load_path_weights = os.path.join(load_path_weights_AE, f'AE_{name}_49.pt')

            state_dict = torch.load(load_path_weights, map_location=str(
                AE.device))  # dizionario che ha per chiave il nome del layer e per valore i pesi

            AE.AENet[i].load_state_dict(state_dict['AE_weights'])  # così carico il dizionario nel modello
            AE.set_requires_grad(AE.AENet[i], False)
        adaptors.reset(default=True)  # resetta per ogni campione i pesi

        prev_loss = float('inf')  # loss precedente, diamo un valore molto grande
        loss_tot = []
        loss_output = []

        #save_dir = "prova"
        #os.makedirs(save_dir, exist_ok=True)

        for epoch in range(opt.n_epochs):  # numero di iterazioni per ogni campione
            outputs = adaptors(batch, task_model, x_minus_one, t_tensor, scheduler)

            loss = 0  # somma delle 4 loss (una per ogni adaptor)
            print('---------------------------------')

            for i in range(len(AE.AENet)):
                index = opt.return_layers[i]  # prendo l'indice del layer (chiave del dizionario outputs)
                side_out = outputs[index]  # side_out è l'output del task network
                level_loss = 0  # loss del singolo reconstruction model

                if len(AE.AENetMatch[i]) == 2:  # concatenate features from the same level
                    side_out_cat = torch.cat([side_out[0], side_out[1]], dim=1)
                else:
                    side_out_cat = side_out

                ae_out = AE.AENet[i](side_out_cat, side_out=False)

                # Normalizzazione per stabilizzare la loss
                scale = side_out_cat.pow(2).mean().sqrt().detach()
                level_loss = AE.AELoss(ae_out, side_out_cat) / (scale + 1e-6)

                print(f'loss {i} epoch {epoch}: {level_loss}')
                loss += level_loss  # somma delle loss dei vari livelli

            """
            gt = batch['B'].to('cuda')

            # 3) Loss diretta pred vs GT (con normalizzazione come nel tuo codice)
            level_loss = AE.AELoss(outputs['final_output'][0], gt)
            loss += level_loss
            # Se usi la loss standard:
            # data_loss = criterion(pred, gt)
            print(f'epoch {epoch}: data_loss={level_loss.item():.6f}')
            """

            loss_output.append(level_loss.data.item())  # loss dell'output
            org_loss = orthw * l2_reg_ortho(adaptors.conv)  # orthogonal regularization
            loss += org_loss
            loss_tot.append(loss.data.item())

            adaptors.optimizer_ANet.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adaptors.parameters(), max_norm=10.0)
            adaptors.optimizer_ANet.step()
            adaptors.save_networks(str(epoch))

            """
            ae_img = outputs['final_output'][0].detach().cpu()
            if ae_img.ndim == 3:  # (C, H, W)
                ae_img = ae_img[0]  # se è multi-canale, prendi il primo
            plt.imshow(ae_img, cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"epoch_{epoch:03d}.png"), bbox_inches='tight', pad_inches=0)
            plt.close()

            plt.imshow(batch['B'][0][0].cpu(), cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"real.png"), bbox_inches='tight', pad_inches=0)
            plt.close()
            """

            # Early stopping
            if prev_loss < loss:
                break
            else:
                prev_loss = loss

        min_index = loss_output.index(min(loss_output))  # indice dell'epoca che ha prodotto la perdita minima
        used_comb = [str(x + 1) for x in comb_to_print]  # prendo la combinazione usata
        used_comb = '_'.join(used_comb)  # la trasformo in stringa
        adaptors.load_networks(str(min_index))  # carico i pesi del modello con la loss minore
        adaptors.save_networks(used_comb, config=True)  # salvo i pesi del modello con la combinazione usata

        with torch.no_grad():
            outputs = adaptors(batch, task_model, x_minus_one, t_tensor, scheduler)
        
        real_B = torch.Tensor(batch['B']).unsqueeze(0)  # campione reale
        fake_B = outputs[opt.return_layers[-1]]  # campione adattato

        # OUTPUT
        gt_array = real_B.cpu().detach().numpy()
        pred_array = fake_B.cpu().detach().numpy()
        mask = gt_array != 0
        psnr_score = compute_psnr(gt_array[mask], pred_array[mask], data_range=gt_array.max() - gt_array.min())        
        loss_config_output = loss_config_output._append(
            {'config': used_comb, 'loss_output': loss_output[min_index],
                'loss_tot': loss_tot[min_index] / len(comb), 'PSNR': psnr_score}, ignore_index=True)

    # seleziono per il campione la migliore combinazione, cioè quella che minimizza la loss di output.
    min_loss = loss_config_output[opt.criteria].max()  # prendo la loss minima
    # prendo la combinazione corrispondente alla loss minima
    used_comb = loss_config_output[loss_config_output[opt.criteria] == min_loss]['config'].values[0]            

    adaptors.load_networks(used_comb, config=True)  # carico i pesi del modello con la combinazione usata
    # return layers deve diventare quella corretta
    chosen_comb = used_comb.split('_')  # divide la stringa in interi
    chosen_comb = [int(x) - 1 for x in
                    chosen_comb]  # int(x)- 1 effettua questa conversione, rendendo gli indici compatibili con il codice.
    chosen_comb = sorted(chosen_comb)  # ordina
    chosen_comb = [0] + [x + 1 for x in chosen_comb] + [n + 1]  # aggiunge il primo e l'ultimo layer
    chosen_comb = [return_layers[i] for i in chosen_comb]
    opt.return_layers = chosen_comb  # scelgo i layer corrispondenti
    compute_tnet_dim(opt)

    with torch.no_grad():
            outputs = adaptors(batch, task_model, x_minus_one, t_tensor, scheduler)
    real_B = torch.Tensor(batch['B']).unsqueeze(0)  # campione reale
    fake_B = outputs[opt.return_layers[-1]]  # campione adattato
    # OUTPUT
    gt_array = real_B.cpu().detach().numpy()
    pred_array = fake_B.cpu().detach().numpy()
    mask = gt_array != 0
    
    psnr_score = compute_psnr(gt_array[mask], pred_array[mask], data_range=gt_array.max() - gt_array.min())  
    ssim_score = compute_ssim(gt_array[mask], pred_array[mask], data_range=gt_array.max() - gt_array.min())
    mae_score = calcola_mae(torch.Tensor(gt_array), torch.Tensor(pred_array)).item()

    selected_loss_output = float(loss_config_output[loss_config_output['config'] == used_comb]['loss_output'].values[0])

    return used_comb, ssim_score, mae_score, psnr_score, min_loss, selected_loss_output
