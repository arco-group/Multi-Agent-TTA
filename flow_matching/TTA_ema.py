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
from itertools import islice
import re
import time
import argparse
from models.adaptor_3 import DTTAnorm, ANet
torch.manual_seed(0)  # garantisce la riproducibilità
torch.cuda.manual_seed(0)
np.random.seed(0)
from tqdm import tqdm
from src.code.Mri2DSlice_dataset import Mri2DSlicedataset
from src.code import networks
from monai.utils import set_determinism
from monai.networks.schedulers import RFlowScheduler
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
from TTA import TTA_rndm_50
from train_monitoring_agent import compute_tnet_dim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr, structural_similarity as compute_ssim

#torch.cuda.set_per_process_memory_fraction(0.33, 0)  # Limit to 30% of GPU 0

torch.manual_seed(42)  # garantisce la riproducibilità
torch.cuda.manual_seed(42)
set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_GPUS = torch.cuda.device_count()
thresholds = {
    "flow_matching": {
        #"LDCT": {"LDCT_HDCT": 0},
        "LDCT": {"LDCT_HDCT": 0.0038},
        "BraTS": {"t1n_t2w": 0.0024},
        #"BraTS": {"t1n_t2w": 0},

}
}

datasets = ['OASIS', 'UCSF', 'UPENN', 'FDG', 'LDCT', 'BraTS']

def extract_threshold(loss_history, sample_size, percentile, with_replacement=True):
    if not loss_history:
        return None
    if sample_size <= 0 or sample_size >= len(loss_history):
        window = list(loss_history)
    else:
        window = list(loss_history[-sample_size:])
    return float(np.percentile(window, percentile))


def run_inference(task_model, AENet, adaptors, dataset, opt, save_dir, thr=0):
    print(save_dir)
    for p in task_model.parameters():
        p.requires_grad = False
    task_model.eval()

    return_layers = opt.return_layers

    # questa parte sotto va modificata
    for subnets in AENet.AENet:
        subnets.eval()

    mae, psnr, ssim_list = [], [], []
    df = pd.DataFrame(columns=['img_name', 'SSIM', 'MAE', 'PSNR', 'rec_loss', 'threshold_used', 'loss_for_threshold', 'threshold_after', 'loss_source'])
    df_tta = pd.DataFrame(columns=['img_name', 'SSIM', 'MAE', 'PSNR', 'rec_loss', 'threshold_used', 'loss_output', 'config'])

    os.makedirs(save_dir, exist_ok=True)
    csv_path_no_tta = os.path.join(save_dir, 'metrics_no_tta_ema.csv')
    csv_path_tta = os.path.join(save_dir, 'metrics_tta_ema.csv')

    no_tta_columns = ['img_name', 'SSIM', 'MAE', 'PSNR', 'rec_loss', 'threshold_used', 'loss_for_threshold', 'threshold_after', 'loss_source']
    tta_columns = ['img_name', 'SSIM', 'MAE', 'PSNR', 'rec_loss', 'threshold_used', 'loss_output', 'config']
    calcola_mae = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')


    try:
        df_existing_no_tta = pd.read_csv(csv_path_no_tta) if os.path.exists(csv_path_no_tta) else pd.DataFrame(columns=no_tta_columns)
    except pd.errors.EmptyDataError:
        df_existing_no_tta = pd.DataFrame(columns=no_tta_columns)

    try:
        df_existing_tta = pd.read_csv(csv_path_tta) if os.path.exists(csv_path_tta) else pd.DataFrame(columns=tta_columns)
    except pd.errors.EmptyDataError:
        df_existing_tta = pd.DataFrame(columns=tta_columns)

    processed_img_names = set(df_existing_no_tta['img_name'].astype(str)) if 'img_name' in df_existing_no_tta else set()
    loss_history = df_existing_no_tta['loss_for_threshold'].dropna().astype(float).tolist()
    processed_count = len(df_existing_no_tta)
    if df_existing_no_tta['threshold_after'].dropna().size > 0:
        current_threshold = float(df_existing_no_tta['threshold_after'].dropna().iloc[-1])
    else:
        current_threshold = float(thr)

    initial_wait_raw = getattr(opt, 'tta_ema_t', None)
    if initial_wait_raw is None:
        initial_wait_raw = opt.tta_ema_j
    initial_wait = max(1, int(initial_wait_raw))
    update_stride = max(1, int(opt.tta_ema_j))
    next_threshold_update = initial_wait
    while next_threshold_update <= processed_count:
        next_threshold_update += update_stride

    write_header_no_tta = not os.path.exists(csv_path_no_tta)
    write_header_tta = not os.path.exists(csv_path_tta)

    for idx, data in tqdm(enumerate(dataset)):
        img_path = data['A_paths']     # get image paths
        img_name = img_path
        if img_name in processed_img_names:
            opt.return_layers = return_layers
            continue
        
        condition_batch = torch.Tensor(data['A']).unsqueeze(0)
        gt_batch = torch.Tensor(data['B']).unsqueeze(0)

        B, C, H, W = condition_batch.shape
        num_inference_steps = opt.diff_step
        img_numel = condition_batch.shape[-1] * condition_batch.shape[-2]
        scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            device=DEVICE,
            input_img_size_numel=img_numel,
        )
        x = torch.randn_like(condition_batch).to(DEVICE)
        condition_batch = condition_batch.to(DEVICE)
        gt_batch = gt_batch.to(DEVICE)

        next_timesteps = torch.cat(
            (
                scheduler.timesteps[1:],
                torch.tensor([0], dtype=scheduler.timesteps.dtype, device=scheduler.timesteps.device),
            )
        )
        progress = tqdm(
            zip(scheduler.timesteps, next_timesteps),
            total=min(len(scheduler.timesteps), len(next_timesteps)),
            desc="Flow Matching Sampling",
        )
        # Step fino al n-1 del Task Model Iterativo
        with torch.no_grad():
            for t, next_t in zip(scheduler.timesteps[:-1], next_timesteps[:-1]):
                t_tensor = torch.tensor([t], device=DEVICE).long()
                model_input = torch.cat([x, condition_batch], dim=1)
                predicted_velocity = task_model(x=model_input, timesteps=t_tensor, context=None)
                x, _ = scheduler.step(predicted_velocity, t, x, next_t)

            x_minus_one = x

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

        pred_denoised = x
        gt_array = gt_batch.cpu().detach().numpy()
        pred_array = pred_denoised.cpu().detach().numpy()
        mask = gt_array != 0

        psnr_score = compute_psnr(gt_array[mask], pred_array[mask], data_range=gt_array.max() - gt_array.min())
        ssim_score = compute_ssim(gt_array[mask], pred_array[mask], data_range=gt_array.max() - gt_array.min())
        mae_score = calcola_mae(torch.Tensor(gt_array), torch.Tensor(pred_array)).item()

        index = opt.return_layers[-1]  # prendo l'indice del layer (chiave del dizionario outputs)
        side_out = outputs[index]  # side_out è l'output del task network   
            
        # use seperate features
        side_out = side_out
        ae_out = AENet.AENet[-1](side_out, side_out=False)  # uscita del modello di ricostruzione -> dominio B
        rec_loss = AENet.AELoss(ae_out, side_out)  # loss dei ricostruttori senza adaptation
        ae_out = AENet.AENet[-1](side_out, side_out=False)  # uscita del modello di ricostruzione -> dominio B
        rec_loss = AENet.AELoss(ae_out, side_out)  # loss dei ricostruttori senza adaptation
        rec_loss_value = float(rec_loss.item())

        threshold_before = current_threshold
        loss_for_threshold = rec_loss_value
        loss_source = 'pre'
        tta_config = ''
        tta_loss = np.nan

        row = {
            'img_name': img_path,
            'SSIM': np.round(ssim_score, 4),
            'MAE': np.round(mae_score, 4),
            'PSNR': np.round(psnr_score, 4),
            'rec_loss': np.round(rec_loss_value, 6),
            'threshold_used': np.round(threshold_before, 6),
        }

        if rec_loss_value > threshold_before:
            used_comb, ssim_score, mae_score, psnr_score, min_loss, selected_loss = TTA_rndm_50(adaptors, opt, task_model, save_dir, data, x_minus_one, t_tensor, scheduler, rec_loss, return_layers=opt.return_layers, psnr_score_no_tta=psnr_score)


            tta_config = used_comb
            tta_loss = float(selected_loss)

            if opt.tta_ema_loss_source == 'post' and not np.isnan(tta_loss):
                loss_for_threshold = tta_loss
                loss_source = 'post'
            else:
                loss_for_threshold = rec_loss_value
                loss_source = 'pre'

            row_tta = {
                'img_name': img_path,
                'SSIM': np.round(ssim_score, 4),
                'MAE': np.round(mae_score, 4),
                'PSNR': np.round(psnr_score, 4),
                'rec_loss': np.round(rec_loss_value, 6),
                'threshold_used': np.round(threshold_before, 6),
                'loss_output': np.round(tta_loss, 6),
                'config': used_comb
            }
            pd.DataFrame([row_tta]).to_csv(
                csv_path_tta,
                mode='a',
                header=write_header_tta,
                index=False
            )
            write_header_tta = False

        loss_history.append(loss_for_threshold)
        processed_count += 1

        if processed_count >= next_threshold_update:
            percentile_threshold = extract_threshold(
                loss_history,
                opt.tta_ema_k,
                float(opt.tta_ema_percentile)
            )
            if percentile_threshold is not None:
                alpha = float(opt.tta_ema_alpha)
                current_threshold = alpha * percentile_threshold + (1.0 - alpha) * current_threshold
                next_threshold_update += update_stride

        row.update({
            'loss_for_threshold': np.round(loss_for_threshold, 6),
            'threshold_after': np.round(current_threshold, 6),
            'loss_source': loss_source
        })

        pd.DataFrame([row]).to_csv(
            csv_path_no_tta,
            mode='a',
            header=write_header_no_tta,
            index=False
        )
        write_header_no_tta = False

        processed_img_names.add(img_name)
        opt.return_layers = return_layers

    summary = {}
    df_no_tta = pd.read_csv(csv_path_no_tta) if os.path.exists(csv_path_no_tta) else pd.DataFrame()
    df_tta = pd.read_csv(csv_path_tta) if os.path.exists(csv_path_tta) else pd.DataFrame()

    if not df_no_tta.empty:
        summary.update({
            'total_samples': len(df_no_tta),
            'no_tta_mae_mean': float(df_no_tta['MAE'].mean()),
            'no_tta_psnr_mean': float(df_no_tta['PSNR'].mean()),
            'no_tta_ssim_mean': float(df_no_tta['SSIM'].mean())
        })

    if not df_tta.empty:
        summary.update({
            'tta_samples': len(df_tta),
            'tta_mae_mean': float(df_tta['MAE'].mean()),
            'tta_psnr_mean': float(df_tta['PSNR'].mean()),
            'tta_ssim_mean': float(df_tta['SSIM'].mean())
        })

        triggered_subset = df_no_tta[df_no_tta['img_name'].isin(df_tta['img_name'])]
        if not triggered_subset.empty:
            summary.update({
                'no_tta_triggered_mae_mean': float(triggered_subset['MAE'].mean()),
                'no_tta_triggered_psnr_mean': float(triggered_subset['PSNR'].mean()),
                'no_tta_triggered_ssim_mean': float(triggered_subset['SSIM'].mean())
            })

    summary_path = os.path.join(save_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        import json
        json.dump(summary, f, indent=2)


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
    parser.add_argument('--mri_modalities', default=["t1n", "t1c", "t2w", "t2f"], help='which MRI modality to use', nargs='+', type=str)
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
    parser.add_argument('--diff_step', type=int, default=1, help='number of steps for reverse diffusion process in inference')
    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--under_sample_dataset', action="store_true", help='True undersample the dataset deleting one slice every three')
    parser.add_argument('--phase', type=str, default=None, help='train or test, if None dont split')

    parser.add_argument('--usedtta', dest='usedtta', action='store_true', default=False, help='use dtta in MEDIA paper')
    parser.add_argument('--seq', dest='seq', type=lambda x: list(map(int, x.split(','))), help='the 1x1 conv seq to be used in A-Net')
    parser.add_argument('--wo', dest='orthw', default=1, type=float, help='orthogonal weights in training ANet')
    parser.add_argument('--model', default='flow_matching')
    parser.add_argument('--criteria', default='loss_output', type=str, help= 'criteria')

    parser.add_argument('--tta_ema_j', type=int, default=60, help='number of processed samples between threshold updates for TTA_ema')
    parser.add_argument('--tta_ema_k', type=int, default=30, help='number of past samples considered when recomputing the adaptive threshold in TTA_ema')
    parser.add_argument('--tta_ema_t', type=int, default=None, help='number of processed samples before first threshold update for TTA_ema')
    parser.add_argument('--tta_ema_percentile', type=float, default=95.0, help='percentile used to derive the percentile target during TTA_ema updates')
    parser.add_argument('--tta_ema_alpha', type=float, default=0.2, help='EMA smoothing factor applied when updating the threshold in TTA_ema')
    parser.add_argument('--tta_ema_loss_source', type=str, default='post', choices=['pre', 'post'], help='which loss to use for threshold updates when TTA_ema is performed (pre or post adaptation)')

    parser.add_argument('--input_range_01', action="store_true", help='If set, keep input in [0,1] (no [-1,1] remap)')


    opt = parser.parse_args()
    opt.gpu_ids = [0]
    opt.output_dir = opt.diff_ckpt + '_rec_models'
    opt.checkpoints_dir = opt.output_dir
    opt.results_dir = opt.diff_ckpt + '_TTA'
    opt.name = opt.experiment_name

    return_layers = opt.return_layers
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

    total_iters = 0                # the total number of training iterations

    AENet = AENet(opt)

    load_path_weights_AE = os.path.join(opt.diff_ckpt + '_rec_models', 'epoch49')

    for i in range(len(opt.return_layers)):
        name = opt.return_layers[i]
        elem = opt.tnet_dim[i]

        load_path_weights = os.path.join(load_path_weights_AE, f'AE_{name}_49.pt')

        state_dict = torch.load(load_path_weights, map_location=str(AENet.device))  # dizionario che ha per chiave il nome del layer e per valore i pesi

        AENet.AENet[i].load_state_dict(state_dict['AE_weights'])  # così carico il dizionario nel modello
        AENet.set_requires_grad(AENet.AENet[i], False)
    
    adaptors = ANet(opt).cuda()

     # --- inferenza ---
    ds = [d for d in datasets if d in opt.dataroot][0]
    source_ds = [d for d in datasets if d in opt.checkpoints_dir][0]
    output_dir = os.path.join(opt.results_dir.replace('_rec_models', '_TTA'), 'TTA_ema_1step_all_rec', ds)
    adaptors.save_dir=output_dir
    adaptors.save_dir_config = output_dir
    thr = thresholds[opt.model][source_ds]['_'.join(opt.mri_modalities)]
    run_inference(task_model, AENet, adaptors, dataset, opt, output_dir, thr)
