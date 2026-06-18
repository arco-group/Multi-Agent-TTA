import torch
import torch.nn as nn
import torch.optim as optim
from models.UNet import UNet
from models.Autoencoder_model import AENet
import time
import pandas as pd
import os
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
from src.code.Mri2DSlice_dataset import Mri2DSlicedataset
from src.code import networks
from monai.utils import set_determinism
from monai.networks.schedulers import RFlowScheduler
import re
import time
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
from TTA import TTA_rndm_50
from train_monitoring_agent import compute_tnet_dim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr, structural_similarity as compute_ssim
torch.manual_seed(0)  # garantisce la riproducibilità
torch.cuda.manual_seed(0)
np.random.seed(0)
from tqdm import tqdm
import argparse

torch.manual_seed(42)  # garantisce la riproducibilità
torch.cuda.manual_seed(42)
set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_GPUS = torch.cuda.device_count()


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


def compute_loss_rec(opt, task_model, rec_models=None, stable=False, AENet=None):
    for p in task_model.parameters():
        p.requires_grad = False
    task_model.eval()
    # questa parte sotto va modificata
    for subnets in rec_models:
        subnets.eval()

    os.makedirs(opt.results_dir, exist_ok=True)
    loss_csv_path = os.path.join(opt.results_dir, opt.loss_csv_name)
    header_written = os.path.exists(loss_csv_path)
    rows_buffer = []

    existing = set()
    if os.path.exists(loss_csv_path):
        existing_df = pd.read_csv(loss_csv_path)
        if 'campione' in existing_df.columns and 'img_name' not in existing_df.columns:
            existing_df = existing_df.rename(columns={'campione': 'img_name'})
        if 'img_name' in existing_df.columns:
            existing = set(existing_df['img_name'].astype(str).tolist())

    scheduler_initialized = False

    num_batches = len(train_loader)
    processed = 0
    skipped = 0
    pbar = tqdm(train_loader, total=num_batches, desc="batches", unit="batch")
    for j, batch in enumerate(pbar):
        data = batch
        img_paths = data['A_paths']
        if isinstance(img_paths, (list, tuple)):
            img_names = [str(p) for p in img_paths]
        else:
            img_names = [str(img_paths)]

        condition_batch = torch.as_tensor(data['A'])
        gt_batch = torch.as_tensor(data['B'])
        if condition_batch.dim() == 3:
            condition_batch = condition_batch.unsqueeze(0)
            gt_batch = gt_batch.unsqueeze(0)

        # Skip samples already processed (resume-safe with mixed batches)
        keep_indices = [i for i, name in enumerate(img_names) if name not in existing]
        skipped += (len(img_names) - len(keep_indices))
        if len(keep_indices) == 0:
            pbar.set_postfix(processed=processed, skipped=skipped)
            continue
        if len(keep_indices) < len(img_names):
            keep_idx = torch.as_tensor(keep_indices, device=condition_batch.device)
            condition_batch = condition_batch.index_select(0, keep_idx)
            gt_batch = gt_batch.index_select(0, keep_idx)
            img_names = [img_names[i] for i in keep_indices]

        if not scheduler_initialized:
            num_inference_steps = opt.diff_step
            img_numel = condition_batch.shape[-1] * condition_batch.shape[-2]
            scheduler.set_timesteps(
                num_inference_steps=num_inference_steps,
                device=DEVICE,
                input_img_size_numel=img_numel,
            )
            scheduler_initialized = True

        x = torch.randn_like(condition_batch).to(DEVICE, non_blocking=True)
        condition_batch = condition_batch.to(DEVICE, non_blocking=True)
        gt_batch = gt_batch.to(DEVICE, non_blocking=True)

        next_timesteps = torch.cat(
            (
                scheduler.timesteps[1:],
                torch.tensor([0], dtype=scheduler.timesteps.dtype, device=scheduler.timesteps.device),
            )
        )
        # Step fino al n-1 del Task Model Iterativo
        with torch.inference_mode():
            for t, next_t in zip(scheduler.timesteps[:-1], next_timesteps[:-1]):
                t_tensor = t.view(1).long()
                model_input = torch.cat([x, condition_batch], dim=1)
                predicted_velocity = task_model(x=model_input, timesteps=t_tensor, context=None)
                x, _ = scheduler.step(predicted_velocity, t, x, next_t)

            # Last timestep for feature extraction
            t = scheduler.timesteps[-1]
            next_t = next_timesteps[-1]
            t_tensor = t.view(1).long()
            model_input = torch.cat([x, condition_batch], dim=1)
            predicted_velocity, outputs = task_model(
                x=model_input, timesteps=t_tensor, context=None, return_layers=True
            )
            x, _ = scheduler.step(predicted_velocity, t, x, next_t)

            # Patch missing entries in outputs
            outputs[opt.return_layers[-1]] = x
            outputs[opt.return_layers[0]] = outputs[opt.return_layers[0]][:, 1, :, :].unsqueeze(1)

            index = opt.return_layers[-1]  # prendo l'indice del layer (chiave del dizionario outputs)
            side_out = outputs[index]  # side_out è l'output del task network
            ae_out = AENet.AENet[-1](side_out, side_out=False)  # uscita del modello di ricostruzione -> dominio B
            # Per-sample MSE: identico al caso batch_size=1, ma preserva i valori per campione
            per_sample_loss = (ae_out - side_out).pow(2).mean(dim=(1, 2, 3))

        for i, img_name in enumerate(img_names):
            rows_buffer.append(
                {'img_name': img_name, 'loss_AE_output': float(per_sample_loss[i].item())}
            )
            existing.add(img_name)
            processed += 1
        pbar.set_postfix(processed=processed, skipped=skipped)

        if len(rows_buffer) >= opt.csv_flush:
            row_df = pd.DataFrame(rows_buffer)
            row_df.to_csv(loss_csv_path, mode='a', header=not header_written, index=False)
            header_written = True
            rows_buffer = []

    if rows_buffer:
        row_df = pd.DataFrame(rows_buffer)
        row_df.to_csv(loss_csv_path, mode='a', header=not header_written, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', required=False, type=str)
    parser.add_argument('--output_dir', required=False, type=str)
    parser.add_argument('--diff_ckpt', required=False, default=None, type=str)
    parser.add_argument('--ae_ckpt', required=False, default=None, type=str)

    parser.add_argument('--experiment_name', required=False, type=str)
    parser.add_argument('--annotation_A', required=False, type=str)
    parser.add_argument('--annotation_B', required=False, type=str)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--lr', default=1.5e-5, type=float)

    parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--mri_modalities', default=["t1n", "t1c", "t2w", "t2f", "LDCT", "HDCT"], help='which MRI modality to use', nargs='+', type=str)
    parser.add_argument('--slice_range', type=int, nargs=2, default=[0, 999],help='Range of slice indices to include, e.g., --slice_range 30 128')
    parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count, we save the model by <epoch_count>, if 0 we dont load the ckpt ...')
    parser.add_argument('--n_epochs_fixed', type=int, default=100, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--total_epoch', dest='total_epoch', default=100, type=int, help='total numer of epochs for training AE')
    parser.add_argument('--return_layers', default=[i for i in range(12)], help='which rec models to connect', nargs='+', type=int)
    parser.add_argument('--aenet_dim', type=int, default=64, help='feature AE')
    parser.add_argument('--aelr', default=0.001, type=float)
    parser.add_argument('--alr', default=0.0001, type=float)

    parser.add_argument('--aenet_style', default="per_layer", type=str)
    parser.add_argument('--AE_to_train', default='all')
    parser.add_argument('--diff_step', type=int, default=10, help='number of steps for reverse diffusion process in inference')
    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--under_sample_dataset', action="store_true", help='True undersample the dataset deleting one slice every three')
    parser.add_argument('--phase', type=str, default=None, help='train or test, if None dont split')

    parser.add_argument('--usedtta', dest='usedtta', action='store_true', default=False, help='use dtta in MEDIA paper')
    parser.add_argument('--seq', dest='seq', type=lambda x: list(map(int, x.split(','))), help='the 1x1 conv seq to be used in A-Net')
    parser.add_argument('--wo', dest='orthw', default=1, type=float, help='orthogonal weights in training ANet')
    parser.add_argument('--model', default='flow_matching')
    parser.add_argument('--criteria', default='loss_output', type=str, help= 'criteria')
    parser.add_argument('--csv_flush', default=200, type=int, help='rows to buffer before appending to csv')
    parser.add_argument('--loss_csv_name', default='loss_rec.csv', type=str,
                        help='name of the csv file saved inside results_dir')
    parser.add_argument('--input_range_01', action="store_true", help='If set, keep input in [0,1] (no [-1,1] remap)')
    
    opt = parser.parse_args()

    opt.gpu_ids = [0]
    opt.output_dir = opt.diff_ckpt + '_rec_models'
    opt.checkpoints_dir = opt.output_dir
    opt.results_dir = opt.diff_ckpt + '_rec_loss'
    opt.name = opt.experiment_name
    return_layers = opt.return_layers
    compute_tnet_dim(opt)

    task_model = networks.init_ddpm(opt.diff_ckpt).to(DEVICE)
    scheduler = RFlowScheduler(
        num_train_timesteps=1000,
        use_discrete_timesteps=False,
        sample_method="uniform",
        use_timestep_transform=True,
        base_img_size_numel=256 * 256,
        spatial_dim=2,
    )

    dataset = Mri2DSlicedataset(opt)
    prefetch_factor = 2 if opt.num_workers > 0 else None
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,  # number of parallel process to load data from files
        drop_last=True,
        pin_memory=True,
        persistent_workers=opt.num_workers > 0,
        prefetch_factor=prefetch_factor,
    )

    AENet = AENet(opt)

    load_path_weights_AE = os.path.join(opt.diff_ckpt + '_rec_models', 'epoch49')

    for i in range(len(opt.return_layers)):
        name = opt.return_layers[i]
        elem = opt.tnet_dim[i]

        load_path_weights = os.path.join(load_path_weights_AE, f'AE_{name}_49.pt')

        state_dict = torch.load(load_path_weights, map_location=str(AENet.device))  # dizionario che ha per chiave il nome del layer e per valore i pesi

        AENet.AENet[i].load_state_dict(state_dict['AE_weights'])  # così carico il dizionario nel modello
        AENet.set_requires_grad(AENet.AENet[i], False)

    rec_models = AENet.AENet
    
    loss_list = compute_loss_rec(opt, task_model, rec_models, stable=False, AENet=AENet)  # adaptor
