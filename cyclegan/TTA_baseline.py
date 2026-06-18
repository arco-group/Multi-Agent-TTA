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
from models.adaptor_3 import DTTAnorm, ANet
torch.manual_seed(0)  # Ensure reproducibility.
torch.cuda.manual_seed(0)
np.random.seed(0)
from options.base_options import BaseOptions
from tqdm import tqdm

#from TTA import TTA_rndm_50, compute_tnet_dim
from TTA_all_rec import TTA_all, TTA_rndm_50, compute_tnet_dim


# torch.cuda.set_per_process_memory_fraction(0.33, 0)  # Limit to 30% of GPU 0

def run_inference(task_model, AENet, adaptors, dataset, opt, save_dir, thr=0):
    print(save_dir)
    task_model.set_requires_grad([task_model.netG_A, task_model.netG_B, task_model.netD_A, task_model.netD_B], False)

    task_model.eval()

    return_layers = opt.return_layers

    # Keep the reconstruction models in evaluation mode.
    for subnets in AENet.AENet:
        subnets.eval()

    mae, psnr, ssim_list = [], [], []
    df = pd.DataFrame(columns=['img_name', 'SSIM', 'MAE', 'PSNR'])
    df_tta = pd.DataFrame(columns=['img_name', 'SSIM', 'MAE', 'PSNR', 'config'])

    # Prepare output paths.
    os.makedirs(save_dir, exist_ok=True)
    csv_path_no_tta = os.path.join(save_dir, 'metrics_no_tta.csv')
    csv_path_tta = os.path.join(save_dir, 'metrics_tta.csv')

    no_tta_columns = ['img_name', 'SSIM', 'MAE', 'PSNR']
    tta_columns = ['img_name', 'SSIM', 'MAE', 'PSNR', 'config']

    try:
        df_existing_no_tta = pd.read_csv(csv_path_no_tta) if os.path.exists(csv_path_no_tta) else pd.DataFrame(columns=no_tta_columns)
    except pd.errors.EmptyDataError:
        df_existing_no_tta = pd.DataFrame(columns=no_tta_columns)

    try:
        df_existing_tta = pd.read_csv(csv_path_tta) if os.path.exists(csv_path_tta) else pd.DataFrame(columns=tta_columns)
    except pd.errors.EmptyDataError:
        df_existing_tta = pd.DataFrame(columns=tta_columns)

    processed_img_names = set(df_existing_no_tta['img_name'].astype(str)) if 'img_name' in df_existing_no_tta else set()

    # Write the header only once.
    write_header_no_tta = not os.path.exists(csv_path_no_tta)
    write_header_tta = not os.path.exists(csv_path_tta)

    for idx, data in tqdm(enumerate(dataset)):
        task_model.set_input(data)  # unpack data from data loader
        img_path = task_model.get_image_paths()     # get image paths
        img_name = img_path[0]

        if img_name in processed_img_names:
            opt.return_layers = return_layers
            continue

        # These three lines are required because we do not call the base_model test helper here.
        with torch.no_grad():
            outputs = task_model.forward(return_layers=opt.return_layers)  # forward pass of the task model

        task_model.compute_visuals()
        visuals = task_model.get_current_visuals()  # get image results

        mae_score=calcola_mse(visuals)
        mae.append(mae_score)
        psnr_score=calculate_psnr(visuals)
        psnr.append(psnr_score)
        ssim_score=calculate_ssim(visuals)
        ssim_list.append(ssim_score)
        
        index = opt.return_layers[-1]  # Index of the final feature map in outputs.
        side_out = outputs[index]  # Output of the task network.
            
        # Use the final-layer features directly.
        ae_out = AENet.AENet[-1](side_out, side_out=False)  # Reconstruction-model output in domain B.
        rec_loss = AENet.AELoss(ae_out, side_out)  # Reconstruction loss without adaptation.

        row = {
            'img_name': img_path[0],
            'SSIM': np.round(ssim_score, 4),
            'MAE': np.round(mae_score, 4),
            'PSNR': np.round(psnr_score, 4),
        }
        pd.DataFrame([row]).to_csv(
            csv_path_no_tta,
            mode='a',
            header=write_header_no_tta,
            index=False
        )
        write_header_no_tta = False
        processed_img_names.add(img_name)
        if np.round(rec_loss.item(),4) > thr:
            used_comb, ssim_score, mae_score, psnr_score, min_loss, _ = TTA_rndm_50(adaptors, opt, task_model, save_dir, data, rec_loss, return_layers=opt.return_layers, psnr_score_no_tta=psnr_score)
            #used_comb, ssim_score, mae_score, psnr_score, min_loss, _ = TTA_all(adaptors, opt, task_model, save_dir, data, rec_loss, return_layers=opt.return_layers, psnr_score_no_tta=psnr_score)

            
            row_tta = {
                'img_name': img_path[0],
                'SSIM': np.round(ssim_score, 4),
                'MAE': np.round(mae_score, 4),
                'PSNR': np.round(psnr_score, 4),
                'config': used_comb
            }
            pd.DataFrame([row_tta]).to_csv(
                csv_path_tta,
                mode='a',
                header=write_header_tta,
                index=False
            )
            write_header_tta = False

        opt.return_layers = return_layers

    # compute summary statistics using saved CSVs
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
    else:
        df_tta = pd.DataFrame(columns=tta_columns)

    summary_path = os.path.join(save_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        import json
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    return_layers = opt.return_layers
    compute_tnet_dim(opt)
    task_model = create_model(opt)  # creo il task model

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    task_model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations

    pattern = r"/TEST_[^/]*/"
    load_path_model = os.path.join(re.sub(pattern, "/", opt.results_dir.replace('_rec_models', "")), 'latest_net_G_A.pth')

    state_dict_model = torch.load(load_path_model,
                                  map_location=str(task_model.device))  # State dict for the task model weights.

    if isinstance(task_model.netG_A, torch.nn.DataParallel): # se parallelizzo il modello (uso più di 1 gpu)
        task_model.netG_A.module.load_state_dict(state_dict_model) # parallelizzo ogni rete sulla gpu ( T model ha 4 reti)
    else:
        task_model.netG_A.load_state_dict(state_dict_model)

    AENet = AENet(opt)

    pattern = r"/TEST_[^/]*/"
    load_path_weights_AE = os.path.join(re.sub(pattern, "/", opt.results_dir), 'epoch49')

    for i in range(len(opt.return_layers)):
        name = opt.return_layers[i]
        elem = opt.tnet_dim[i]

        load_path_weights = os.path.join(load_path_weights_AE, f'AE_{name}_49.pt')

        state_dict = torch.load(load_path_weights, map_location=str(
            AENet.device))  # State dict containing one entry per layer.

        AENet.AENet[i].load_state_dict(state_dict)  # Load the state dict into the model.
        AENet.set_requires_grad(AENet.AENet[i], False)
    
    adaptors = ANet(opt).cuda()

     # --- inferenza ---
    dataset_name = opt.dataset_name or os.path.basename(os.path.normpath(opt.dataroot))
    output_dir = os.path.join(opt.results_dir.replace('_rec_models', '_TTA'), 'TTA_baseline', dataset_name)
    adaptors.save_dir=output_dir
    adaptors.save_dir_config = output_dir
    run_inference(task_model, AENet, adaptors, dataset, opt, output_dir, float(opt.tta_threshold))
