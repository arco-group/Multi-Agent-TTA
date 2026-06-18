import torch
import torch.nn as nn
import torch.optim as optim
import time
from models import create_model
from models.Autoencoder_model import AENet
import pandas as pd
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from src.code.Mri2DSlice_dataset import Mri2DSlicedataset
from src.code import networks
from monai.utils import set_determinism
from monai.networks.schedulers import RFlowScheduler
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import warnings
import csv
warnings.filterwarnings("ignore")

torch.manual_seed(42)  # garantisce la riproducibilità
torch.cuda.manual_seed(42)
set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_GPUS = torch.cuda.device_count()

# scriviamo una funzione per ricalcolare tnet_dim a seconda di cosa c'è in return_layers
def compute_tnet_dim(opt):
    layers_to_dim = {'0': 1, '1': 64, '2': 64, '3': 128, '4': 128,
                     '5': 256, '6': 256, '7': 256, '8': 128, '9': 128, '10': 64, '11': 1}
    tnet_dim = []
    for layer in opt.return_layers:
        tnet_dim.append(layers_to_dim[str(layer)])
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


def plot_ae_losses(loss_list, graph_type, path, name):
    epochs = np.arange(0, len(loss_list))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_list, label='AE Loss', marker='o')

    plt.xlabel(graph_type)
    plt.ylabel('Loss')
    plt.title('Autoencoder Losses Over Epochs')
    plt.legend()
    plt.grid(True)
    os.makedirs(path, exist_ok=True)

    plt.savefig(os.path.join(path, f'train_ind_losses_{name}.png'))
    return name


def training_AE(AENet, task_model, scheduler, dataloader, opt, AE_to_train='all'):
    """
    Train one or all Autoencoders (AEs) for reconstruction using diffusion-based task model.
    Supports checkpoint resume, multi-AE optimization, and loss/image logging.

    Args:
        AENet: wrapper object containing list of AEs, optimizers, schedulers.
        task_model: frozen diffusion model providing intermediate features.
        scheduler: flow matching scheduler (RFlowScheduler)
        dataloader: data iterator.
        opt: training options.
        AE_to_train: 'all' to train every AE, or int index of a specific AE.
    """
    # -------------------------------
    # Resume checkpoints if any
    # -------------------------------
    if opt.epoch_count != 0:
        start_epoch = opt.epoch_count + 1
        ae_indices = range(len(AENet.AENet)) if AE_to_train == 'all' else [AE_to_train]
        for i in ae_indices:
            ckpt_path = os.path.join(
                opt.output_dir, opt.experiment_name,
                f'epoch{opt.epoch_count}', f'AE_{i}_{opt.epoch_count}.pt'
            )
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=DEVICE)
                AENet.AENet[i].load_state_dict(ckpt['AE_weights'])
                AENet.optimizers[i].load_state_dict(ckpt['optimizer'])
                AENet.schedulers[i].load_state_dict(ckpt['scheduler'])
                print(f"✅ Restored AE_{i} from epoch {opt.epoch_count}")
    else:
        start_epoch = 0

    # -------------------------------
    # Setup
    # -------------------------------
    task_model.eval()
    for p in task_model.parameters(): 
        p.requires_grad = False
    scheduler_ready = False
    next_timesteps = None
    AENet.train()

    ae_indices = range(len(AENet.AENet)) if AE_to_train == 'all' else [AE_to_train]
    loss_files = {
        i: os.path.join(AENet.save_dir, f'loss_{i}.csv') for i in ae_indices
    }
    loss_lists = {i: [] for i in ae_indices}
    loss_lists_epoch = {i: [] for i in ae_indices}

    # Create loss logs if missing
    for i in ae_indices:
        lf = loss_files[i]
        os.makedirs(os.path.dirname(lf), exist_ok=True)
        if not os.path.exists(lf):
            with open(lf, 'w', newline='') as f:
                csv.writer(f).writerow(["step", "epoch", "loss"])

    # -------------------------------
    # Training loop
    # -------------------------------
    for epoch in range(start_epoch, opt.total_epoch):
        print(f"\n🟢 Starting epoch {epoch+1}/{opt.total_epoch}")
        step = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            x = torch.randn_like(batch['A']).to(DEVICE)
            condition_batch = batch['A'].to(DEVICE)
            gt_batch = batch['B'].to(DEVICE)

            # ----------- Forward diffusion (one per batch) -----------
            with torch.no_grad():
                if not scheduler_ready:
                    img_numel = condition_batch.shape[-1] * condition_batch.shape[-2]
                    scheduler.set_timesteps(
                        num_inference_steps=opt.diff_step,
                        device=DEVICE,
                        input_img_size_numel=img_numel,
                    )
                    next_timesteps = torch.cat(
                        (
                            scheduler.timesteps[1:],
                            torch.tensor(
                                [0],
                                dtype=scheduler.timesteps.dtype,
                                device=scheduler.timesteps.device,
                            ),
                        )
                    )
                    scheduler_ready = True

                for t, next_t in zip(scheduler.timesteps[:-1], next_timesteps[:-1]):
                    t_tensor = torch.tensor([t], device=DEVICE).long()
                    model_input = torch.cat([x, condition_batch], dim=1)
                    predicted_velocity = task_model(x=model_input, timesteps=t_tensor, context=None)
                    x, _ = scheduler.step(predicted_velocity, t, x, next_t)

                # Last timestep for feature extraction
                t = scheduler.timesteps[-1]
                next_t = next_timesteps[-1]
                t_tensor = torch.tensor([t], device=DEVICE).long()
                model_input = torch.cat([x, condition_batch], dim=1)
                predicted_velocity, outputs = task_model(
                    x=model_input, timesteps=t_tensor, context=None, return_layers=True
                )
                x, _ = scheduler.step(predicted_velocity, t, x, next_t)

                # Patch missing entries in outputs
                outputs[opt.return_layers[-1]] = x
                outputs[opt.return_layers[0]] = outputs[opt.return_layers[0]][:, 1, :, :].unsqueeze(1)

            # ----------- Train all selected AEs -----------
            for i in ae_indices:
                ae = AENet.AENet[i]
                optimizer = AENet.optimizers[i]
                optimizer.zero_grad()

                index = opt.return_layers[i]
                side_out = outputs[index]

                # Preprocessing
                if len(AENet.AENetMatch[i]) == 2:
                    side_out_cat_orig = torch.cat([side_out[0], side_out[1]], dim=1)
                    side_out_cat = side_out_cat_orig
                else:
                    side_out_cat_orig = side_out
                    side_out_cat = side_out

                ae_out = ae(side_out_cat, side_out=False)
                level_loss = AENet.AELoss(ae_out, side_out_cat_orig)

                level_loss.backward()
                optimizer.step()

                loss_value = level_loss.item()
                loss_lists[i].append(loss_value)

                # Write incremental CSV log
                with open(loss_files[i], 'a', newline='') as f:
                    csv.writer(f).writerow([step, None, loss_value])

                # Periodic step plot
                if step % 200 == 0:
                    plot_ae_losses(loss_lists[i], 'steps', AENet.save_dir, f"{opt.return_layers[i]}_steps")

                # Periodic image save
                if i == 0 or i == 11:
                    if step % 800 == 0:
                        with torch.no_grad():
                            real_images = outputs[opt.return_layers[i]].float().to(DEVICE)
                            ae_reconstructed = ae(real_images)
                            save_images(real_images, ae_reconstructed, epoch, AENet.save_dir,
                                        f'AE{i}_step{step}', opt.batch_size)

            step += 1

        # -------------------------------
        # End of epoch: schedulers + saving
        # -------------------------------
        for sch in AENet.schedulers:
            sch.step()

        for i in ae_indices:
            AENet.save_networks_AE(epoch, i, step)

            # Update epoch info in CSV
            with open(loss_files[i], 'r', newline='') as f:
                rows = list(csv.reader(f))
            last = rows[-1]
            last[1] = str(epoch)
            rows[-1] = last
            with open(loss_files[i], 'w', newline='') as f:
                csv.writer(f).writerows(rows)

            # Epoch-level loss plot
            loss_lists_epoch[i].append(loss_lists[i][-1])
            plot_ae_losses(loss_lists_epoch[i], 'epochs', AENet.save_dir, f"{opt.return_layers[i]}_epochs")

            print(f"💾 AE_{i} saved for epoch {epoch}, last loss={loss_lists[i][-1]:.6f}")

    return loss_lists_epoch, loss_lists



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', required=False, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--diff_ckpt', required=False, default=None, type=str)
    parser.add_argument('--experiment_name', required=True, type=str)
    parser.add_argument('--annotation_A', required=False, type=str)
    parser.add_argument('--annotation_B', required=False, type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--n_epochs', default=5000, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1.5e-5, type=float)
    parser.add_argument('--diff_loss_weight', type=float, default=1.0)

    parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--mri_modalities', default=["t1n", "t1c", "t2w", "t2f"], help='which MRI modality to use', nargs='+', type=str)
    parser.add_argument('--slice_range', type=int, nargs=2, default=[0, 999],help='Range of slice indices to include, e.g., --slice_range 30 128')

    parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count, we save the model by <epoch_count>, if 0 we dont load the ckpt ...')
    parser.add_argument('--n_epochs_fixed', type=int, default=100, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--total_epoch', dest='total_epoch', default=100, type=int, help='total numer of epochs for training AE')
    parser.add_argument('--return_layers', default=[i for i in range(12)], help='which rec models to connect', nargs='+', type=int)
    parser.add_argument('--aenet_dim', type=int, default=64, help='feature AE')
    parser.add_argument('--aelr', default=0.001, type=float)
    parser.add_argument('--aenet_style', default="per_layer", type=str)
    parser.add_argument('--AE_to_train', default='all')
    parser.add_argument('--diff_step', type=int, default=10, help='number of steps for reverse diffusion process in inference')
    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--under_sample_dataset', action="store_true", help='True undersample the dataset deleting one slice every three')
    parser.add_argument('--phase', type=str, default=None, help='train or test, if None dont split')
    parser.add_argument('--input_range_01', action="store_true", help='If set, keep input in [0,1] (no [-1,1] remap)')

    opt = parser.parse_args()
    opt.gpu_ids = [0]
    opt.checkpoints_dir = opt.output_dir
    opt.name = opt.experiment_name

    compute_tnet_dim(opt)
    print(opt.tnet_dim)

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
    train_loader = DataLoader(dataset=dataset,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            num_workers=opt.num_workers, # number of parallel process to load data from files
                            drop_last=True,
                            pin_memory=True)

    AENet = AENet(opt)  # creo l'autoencoder

    loss_list_epoch, loss_list = training_AE(AENet, task_model, scheduler, train_loader, opt, opt.AE_to_train)  # train AE

    # salvo le loss in un file .csv
    df = pd.DataFrame(loss_list)
    df.to_csv(os.path.join(AENet.save_dir, f'train_ind_losses_{opt.return_layers[AE_to_train]}.csv'), index=False) # per ogni AE ho un file che contiene 742*50 loss
