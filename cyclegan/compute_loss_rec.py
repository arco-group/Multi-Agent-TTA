import torch
import torch.nn as nn
import torch.optim as optim
from models.UNet import UNet
from models.Autoencoder_model import AENet
import time
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
# from util.visualizer import Visualizer
import pandas as pd
import os
from util import html
from util.visualizer import save_images
import torch
from models import find_model_using_name
from util.visualizer import calcola_mse, calculate_psnr, calculate_ssim
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
import re
import time
from pathlib import Path

# from util.functional import normalize
# from util.Variable import Variable
# questo script serve per la fase di TTA, in cui vengono trainati gli Adaptor e freezati Task network e Autoencoder

#torch.cuda.set_per_process_memory_fraction(0.25, 0)


torch.manual_seed(0)  # Ensure reproducibility.
torch.cuda.manual_seed(0)
np.random.seed(0)
from options.base_options import BaseOptions
from tqdm import tqdm


def compute_tnet_dim(opt):
    layers_to_dim = {'input': 1, 'first_conv': 64, 'second_conv': 128, 'third_conv': 256, 'resnet_block_1': 256,
                     'resnet_block_2': 256, 'resnet_block_3': 256, 'resnet_block_4': 256, 'final_output': 1}
    tnet_dim = []
    for layer in opt.return_layers:
        tnet_dim.append(layers_to_dim[layer])
    opt.tnet_dim = tnet_dim


def l2_reg_ortho(model, lambda_l2=1e-4):
    l2_loss = torch.tensor(0.0, device='cuda')
    for param in model.parameters():
        if param.requires_grad:
            l2_loss += torch.norm(param, p=2) ** 2  # norma L2 dei parametri
    return lambda_l2 * l2_loss


def plot_losses(loss_tot, num_epochs, path, j):  # j è l'indice del ciclo sulle batch
    print('LOSSES:', loss_tot)
    print('EPOCHE:', num_epochs)

    epochs = np.arange(1, num_epochs + 1)  # crea un array con il numero di epoche
    print('EPOCHE LISTA:', epochs)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_tot, label='Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(' Losses Over Epochs')
    plt.legend()
    plt.grid(True)
    # salviamo il grafico

    # se la cartella non esiste la crea
    if not os.path.exists(os.path.join(path, 'losses')):
        os.makedirs(os.path.join(path, 'losses'))
    path = os.path.join(path, 'losses')  # cartella in cui salvare il grafico

    plt.savefig(os.path.join(path, f'losses_{j}'))


def compute_loss_rec(opt, task_model, rec_models=None, stable=False, AENet=None, save_every=100):

    task_model.set_requires_grad([task_model.netG_A, task_model.netG_B, task_model.netD_A, task_model.netD_B], False)

    task_model.eval()

    # Keep the reconstruction models in evaluation mode.
    for subnets in rec_models:
        subnets.eval()

    # stablize training by pre-train histogram manipulator (if needed)
    loss_rec = pd.DataFrame(
        columns=['img_name', 'loss_AE_0', 'loss_AE_1', 'loss_AE_2', 'loss_AE_3', 'loss_AE_4', 'loss_AE_5', 'loss_AE_6',
                 'loss_AE_7', 'loss_AE_8'])

    os.makedirs(opt.results_dir, exist_ok=True)

    for j, batch in enumerate(tqdm(dataset)):
        task_model.set_input(batch)
        outputs = task_model.forward(return_layers=opt.return_layers)  # Forward pass of the task model.
        task_model.compute_visuals()
        visuals = task_model.get_current_visuals()  # get image results
        img_path = task_model.get_image_paths()
        loss_rec = loss_rec._append(
            {'campione': img_path[0] , 'loss_AE_0': np.nan, 'loss_AE_1': np.nan, 'loss_AE_2': np.nan, 'loss_AE_3': np.nan,
             'loss_AE_4': np.nan, 'loss_AE_5': np.nan, 'loss_AE_6': np.nan, 'loss_AE_7': np.nan, 'loss_AE_8': np.nan},
            ignore_index=True) 
        
        for i in range(len(rec_models)):
            index = opt.return_layers[i]  # Index of the feature map in outputs.
            side_out = outputs[index]  # Output of the task network.
                
            if len(AENet.AENetMatch[i]) == 2:
                side_out = torch.cat([side_out[0], side_out[1]], dim=1)
            else:
                # Use separate features.
                side_out = side_out
            ae_out = AENet.AENet[i](side_out, side_out=False)  # Reconstruction-model output in domain B.
            rec_loss = weights[i] * AENet.AELoss(ae_out, side_out)  # Reconstruction loss without adaptation.
            loss_rec.loc[j, f'loss_AE_{i}'] = rec_loss.data.item()
        # Save the CSV periodically so progress is not lost if the job stops.
        if (j + 1) % save_every == 0:
            loss_rec.to_csv(os.path.join(opt.results_dir, 'loss_rec.csv'), index=False)

    loss_rec.to_csv(os.path.join(opt.results_dir, f'loss_rec.csv'), index=False)


def resolve_checkpoint_path(candidate_path, opt, layer_name=None, model_kind='ae'):
    """
    Resolve a checkpoint path even if the run folder name used to build
    `results_dir` does not match the folder that actually stores trained weights.
    """
    candidate = Path(candidate_path)
    if candidate.exists():
        return str(candidate)

    search_roots = []
    if getattr(opt, 'checkpoints_dir', None):
        search_roots.append(Path(opt.checkpoints_dir))
    if getattr(opt, 'results_dir', None):
        search_roots.append(Path(opt.results_dir).parent)

    patterns = []
    if model_kind == 'ae' and layer_name is not None:
        patterns.append(f'epoch49/AE_{layer_name}_49.pt')
    if model_kind == 'g':
        patterns.extend(['latest_net_G.pth', 'latest_net_G_A.pth'])

    for root in search_roots:
        for pattern in patterns:
            matches = list(root.rglob(pattern))
            if matches:
                return str(matches[0])

    raise FileNotFoundError(f'Could not resolve checkpoint path for {candidate_path}')

if __name__ == '__main__':
    opt = TrainOptions().parse()
    return_layers = opt.return_layers
    compute_tnet_dim(opt)
    task_model = create_model(opt)  # creo il task model
    dataset = create_dataset(opt)  # ci da il dataloader

    AENet = AENet(opt)
    weights = opt.__dict__.get('weights', [1] * len(AENet.AENet))
    for i in range(len(opt.return_layers)):
        layer = opt.return_layers[i]
        elem = opt.tnet_dim[i]
        
        pattern = r"/rec_loss_[^/]*/"

        #load_path_weights = os.path.join(opt.results_dir.replace('rec_loss/', ""), f'epoch49/AE_{layer}_49.pt')
        load_path_weights = os.path.join(re.sub(pattern, "/", opt.results_dir), f'epoch49/AE_{layer}_49.pt')

        load_path_weights = resolve_checkpoint_path(load_path_weights, opt, layer_name=layer, model_kind='ae')
        state_dict = torch.load(load_path_weights, map_location=str(
            AENet.device))  # State dict containing one entry per layer.

        AENet.AENet[i].load_state_dict(state_dict)  # Load the state dict into the model.
        AENet.set_requires_grad(AENet.AENet[i], False)

    rec_models = AENet.AENet
    
    pattern_2 = r"_rec_models/rec_loss_[^/]*/"

    load_path_model = os.path.join(re.sub(pattern_2, "/", opt.results_dir), 'latest_net_G_A.pth')

    
    load_path_model = resolve_checkpoint_path(load_path_model, opt, model_kind='g')
    state_dict = torch.load(load_path_model,
                            map_location=str(task_model.device))  # State dict for the task model weights.
    
    
    if isinstance(task_model.netG_A, torch.nn.DataParallel): # se parallelizzo il modello (uso più di 1 gpu)
        task_model.netG_A.module.load_state_dict(state_dict) # parallelizzo ogni rete sulla gpu ( T model ha 4 reti)
    else:
        task_model.netG_A.load_state_dict(state_dict)

    
    # Load the reconstruction block weights.
    loss_list = compute_loss_rec(opt, task_model, rec_models, stable=False, AENet=AENet)  # adaptor
