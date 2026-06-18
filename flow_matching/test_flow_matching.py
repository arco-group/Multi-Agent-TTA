import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from monai.utils import set_determinism
from monai.networks.schedulers import RFlowScheduler
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as compute_psnr, structural_similarity as compute_ssim
from src.code.ldct_hdct_dataset import LDCTHDCTDataset
from src.code import networks
from src.code.Mri2DSlice_dataset import Mri2DSlicedataset
# from flow_matching.src.ldct_hdct_dataset import LDCTHDCTDataset


# -----------------------
# ✅ Set environment
# -----------------------
set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_GPUS = torch.cuda.device_count()


@torch.no_grad()
def run_inference_and_log(
        diffusion_model,
        condition_batch,
        gt_batch,
        writer,
        step,
        device,
        scheduler,
        num_inference_steps,
        csv_path,
        mri_modalities=None,
):
    diffusion_model.eval()
    B, C, H, W = condition_batch.shape

    img_numel = condition_batch.shape[-1] * condition_batch.shape[-2]
    scheduler.set_timesteps(
        num_inference_steps=num_inference_steps,
        device=device,
        input_img_size_numel=img_numel,
    )
    x = torch.randn_like(condition_batch).to(device)
    condition_batch = condition_batch.to(device)
    gt_batch = gt_batch.to(device)

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
    for t, next_t in progress:
        t_tensor = torch.tensor([t], device=device).long()
        model_input = torch.cat([x, condition_batch], dim=1)
        predicted_velocity = diffusion_model(x=model_input, timesteps=t_tensor, context=None)
        x, _ = scheduler.step(predicted_velocity, t, x, next_t)

    pred_denoised = x

    def norm_percentile(x, pmin=1, pmax=99):
        x = x.clone().to(torch.float32)
        B = x.shape[0]
        normed = torch.zeros_like(x)
        for i in range(B):
            x_i = x[i]
            min_val = torch.quantile(x_i, pmin / 100.0)
            max_val = torch.quantile(x_i, pmax / 100.0)
            x_i = torch.clamp(x_i, min=min_val, max=max_val)
            normed[i] = (x_i - min_val) / (max_val - min_val + 1e-8)
        return normed

    ld = condition_batch.cpu().detach()
    gt = gt_batch.cpu().detach()
    pred = pred_denoised.cpu().detach()
    error = norm_percentile(abs(pred - gt))

    to_save = B if B < 5 else 4

    fig, axes = plt.subplots(nrows=to_save, ncols=4, figsize=(8, 2.5 * to_save))
    if to_save == 1:
        axes = [axes]

    for i in range(B):
        images = [ld[i], gt[i], pred[i], error[i]]
        titles = [mri_modalities[0], mri_modalities[1], "Prediction", "Error"]

        # Extract arrays
        gt_array = gt[i][0].numpy()
        pred_array = pred[i][0].numpy()
        # Create a mask where gt is not zero
        mask = gt_array != 0

        # Compute metrics
        psnr = compute_psnr(gt_array[mask], pred_array[mask], data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        ssim = compute_ssim(gt_array[mask], pred_array[mask], data_range=gt[i][0].numpy().max() - gt[i][0].numpy().min())
        mse = np.mean((gt_array[mask] - pred_array[mask]) ** 2)
        print(psnr, ssim, mse)

        # Crea dizionario con metriche
        metrics = {
            'Sample': step * B + i,
            'MAE': mse,
            'PSNR': psnr,
            'SSIM': ssim
        }

        # Scrivi su CSV
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Sample', 'MAE', 'PSNR', 'SSIM'])
            writer.writerow(metrics)

        if i < to_save:
            if step % 100 == 0:
                for j in range(4):
                    ax = axes[i][j] if B > 1 else axes[0][j]
                    ax.set_axis_off()
                    ax.set_title(titles[j])
                    img = images[j].squeeze(0).cpu().numpy()
                    ax.imshow(img, cmap='hot' if titles[j] == "Error" else 'gray')

    if step % 100 == 0:
        plt.tight_layout()
        save_dir = os.path.join(opt.output_dir, opt.experiment_name, "test_latest", "images")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'{step}.png'))
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', type=str, required=False)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--diff_ckpt', type=str, required=True)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--mri_modalities', default=["t1n", "t1c", "t2w", "t2f"], help='which MRI modality to use', nargs='+', type=str)
    parser.add_argument('--slice_range', type=int, nargs=2, default=[0, 999],help='Range of slice indices to include, e.g., --slice_range 30 128')
    parser.add_argument('--diff_step', type=int, default=10, help='number of steps for reverse diffusion process in inference')
    parser.add_argument('--under_sample_dataset', action="store_true", help='True undersample the dataset deleting one slice every three')
    parser.add_argument('--phase', type=str, default=None, help='train or test, if None dont split')
    parser.add_argument('--input_range_01', action="store_true", help='If set, keep input in [0,1] (no [-1,1] remap)')

    opt = parser.parse_args()

    os.makedirs(opt.output_dir, exist_ok=True)

    #dataset = LDCTHDCTDataset(
    #)
    dataset = Mri2DSlicedataset(opt)

    
    test_loader = DataLoader(dataset=dataset,
                              batch_size=opt.batch_size,
                              shuffle=False, # To easily merge for TTA metrics
                              num_workers=opt.num_workers, # number of parallel process to load data from files
                              drop_last=True,
                              pin_memory=True)

    diffusion = networks.init_ddpm(opt.diff_ckpt).to(DEVICE)

    if NUM_GPUS > 1:
        diffusion = torch.nn.DataParallel(diffusion)

    scheduler = RFlowScheduler(
        num_train_timesteps=1000,
        use_discrete_timesteps=False,
        sample_method="uniform",
        use_timestep_transform=True,
        base_img_size_numel=256 * 256,
        spatial_dim=2,
    )

    writer = SummaryWriter(comment=opt.experiment_name)
    csv_path = os.path.join(opt.output_dir, opt.experiment_name, "test_latest")
    os.makedirs(csv_path, exist_ok=True)
    csv_path = os.path.join(opt.output_dir, opt.experiment_name, "test_latest", "metrics.csv")

    # Scrive l'header una sola volta (se il file non esiste)
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Sample', 'MAE', 'PSNR', 'SSIM'])
            writer.writeheader()

    for step, batch in enumerate(test_loader):
        run_inference_and_log(
            diffusion_model=diffusion,
            condition_batch=batch['A'],
            gt_batch=batch['B'],
            writer=writer,
            step=step,
            device=DEVICE,
            scheduler=scheduler,
            num_inference_steps=opt.diff_step,
            csv_path=csv_path,
            mri_modalities=opt.mri_modalities
        )
    txt_path = csv_path.replace(".csv", ".txt")

    # Carica il CSV
    df = pd.read_csv(csv_path)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['PSNR'])


    # Calcola media e deviazione standard
    ssim_mean = round(df['SSIM'].mean(), 4)
    ssim_std  = round(df['SSIM'].std(), 4)

    mae_mean  = round(df['MAE'].mean(), 4)
    mae_std   = round(df['MAE'].std(), 4)

    psnr_mean = round(df['PSNR'].mean(), 4)
    psnr_std  = round(df['PSNR'].std(), 4)

    # Scrive su txt
    with open(txt_path, 'w') as f:
        f.write(f"SSIM task model: {ssim_mean} ± {ssim_std}\n")
        f.write(f"MAE task model: {mae_mean} ± {mae_std}\n")
        f.write(f"PSNR task model: {psnr_mean} ± {psnr_std}\n")

    print(f"✅ Inference complete. Metrics saved to {csv_path}")
