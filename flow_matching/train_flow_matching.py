import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from monai.utils import set_determinism
from monai.networks.schedulers import RFlowScheduler
from tqdm import tqdm
import torchvision.utils as vutils
from src.code.ldct_hdct_dataset import LDCTHDCTDataset
from src.code import networks
import csv
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compute_psnr, structural_similarity as compute_ssim
from src.code.Mri2DSlice_dataset import Mri2DSlicedataset
from torch.optim import lr_scheduler

from types import MethodType
# -----------------------
# ✅ Set environment
# -----------------------
set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_GPUS = torch.cuda.device_count()


# -----------------------
# ✅ Log to tensorboard
# -----------------------
@torch.no_grad()
def sample_and_plot_batch_flow_matching(
        diffusion_model,
        condition_batch,
        gt_batch,
        writer,
        step,
        device,
        tag="Flow_Matching_Sampling",
        scheduler=None,
        num_inference_steps=10,
        mri_modalities=None,
        epoch=None,
        max_samples=4
):
    """
    Flow Matching sampling + tensorboard batch display with uncertainty.
    Plots [LD | GT | Prediction | Uncertainty] per row.
    """

    diffusion_model.eval()
    
    # --- 🟡 Limita il batch a max_samples ---
    if condition_batch.shape[0] > max_samples:
        condition_batch = condition_batch[:max_samples]
        gt_batch = gt_batch[:max_samples]

    B, C, H, W = condition_batch.shape

    if scheduler is None:
        raise ValueError("scheduler must be provided for Flow Matching sampling.")

    img_numel = condition_batch.shape[-1] * condition_batch.shape[-2]
    scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device, input_img_size_numel=img_numel)

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

        with autocast(enabled=True):
            predicted_velocity = diffusion_model(x=model_input, timesteps=t_tensor, context=None)

        x, _ = scheduler.step(predicted_velocity, t, x, next_t)

    pred_denoised = x

    # ---- Plotting ---- #
    def norm(x):
        x = x.clone()
        x -= x.amin(dim=(1, 2, 3), keepdim=True)
        x /= (x.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)
        return x

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

    fig, axes = plt.subplots(nrows=B, ncols=4, figsize=(8, 2.5 * B))
    if B == 1:
        axes = [axes]  # make iterable

    for i in range(B):
        images = [ld[i], gt[i], pred[i], error[i]]
        titles = [mri_modalities[0], mri_modalities[1], "Prediction", "Error"]

        for j in range(4):
            ax = axes[i][j] if B > 1 else axes[0][j]
            ax.set_axis_off()
            ax.set_title(titles[j])
            img = images[j].squeeze(0).cpu().numpy()
            ax.imshow(img, cmap='hot' if titles[j] == "Error" else 'gray')

    plt.tight_layout()
    save_dir = os.path.join(opt.output_dir, opt.experiment_name, f"Epoch_{epoch}")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{step}.png'))
    plt.close()


# -----------------------
# ✅ Training script
# -----------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', required=False, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--diff_ckpt', required=False, default=None, type=str)
    parser.add_argument('--experiment_name', required=True, type=str)
    parser.add_argument('--annotation_A', required=False, type=str)
    parser.add_argument('--annotation_B', required=False, type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--n_epochs', default=5000, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1.5e-5, type=float)
    parser.add_argument('--diff_loss_weight', type=float, default=1.0)

    parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--mri_modalities', default=["t1n", "t1c", "t2w", "t2f"], help='which MRI modality to use', nargs='+', type=str)
    parser.add_argument('--slice_range', type=int, nargs=2, default=[0, 999],help='Range of slice indices to include, e.g., --slice_range 30 128')

    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    parser.add_argument('--n_epochs_fixed', type=int, default=100, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--diff_step', type=int, default=10, help='number of steps for reverse diffusion process in inference')
    parser.add_argument('--under_sample_dataset', action="store_true", help='True undersample the dataset deleting one slice every three')
    parser.add_argument('--phase', type=str, default=None, help='train or test, if None dont split')


    opt = parser.parse_args()

    # -----------------------
    # ✅ Load dataset
    # -----------------------
    # Load the LDCT/HDCT dataset
    
    # dataset = LDCTHDCTDataset(
    # )

    dataset = Mri2DSlicedataset(opt)
    
    train_loader = DataLoader(dataset=dataset,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              num_workers=opt.num_workers, # number of parallel process to load data from files
                              drop_last=True,
                              pin_memory=True)

    # -----------------------
    # ✅ Load diffusion model
    # -----------------------
    diffusion = networks.init_ddpm(opt.diff_ckpt).to(DEVICE)
    
    if NUM_GPUS > 1:
        print(f"Using {NUM_GPUS} GPUs")
        diffusion = torch.nn.DataParallel(diffusion)

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=opt.lr)

    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs_fixed) / float(opt.n_epochs_decay + 1)
        return lr_l
        
    lr_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule) # Linear Scheduler

    scheduler = RFlowScheduler(
        num_train_timesteps=1000,
        use_discrete_timesteps=False,
        sample_method="uniform",
        use_timestep_transform=True,
        base_img_size_numel=256 * 256,
        spatial_dim=2,
    )

    inference_scheduler = RFlowScheduler(
        num_train_timesteps=1000,
        use_discrete_timesteps=False,
        sample_method="uniform",
        use_timestep_transform=True,
        base_img_size_numel=256 * 256,
        spatial_dim=2,
    )
    scaler = GradScaler()
    writer = SummaryWriter(comment=opt.experiment_name)

    global_counter = {'train': 0}

    # -----------------------
    # ✅ Training loop
    # -----------------------
    for epoch in range(opt.n_epochs):
        diffusion.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in progress_bar:
            img_A = batch["A"].to(DEVICE) 
            img_B = batch["B"].to(DEVICE)  

            noise = torch.randn_like(img_B)
            timesteps = scheduler.sample_timesteps(img_B)

            with autocast(enabled=True):
                optimizer.zero_grad(set_to_none=True)

                noisy_img_B = scheduler.add_noise(original_samples=img_B, noise=noise, timesteps=timesteps)
                noisy_image = torch.cat([noisy_img_B, img_A], dim=1)
                predicted_velocity = diffusion(x=noisy_image, timesteps=timesteps, context=None)

                # Compute loss (Rectified Flow / Flow Matching)
                loss = F.mse_loss(predicted_velocity.float(), (img_B - noise).float())
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Logging
            writer.add_scalar('train/loss', loss.item(), global_counter['train'])
            epoch_loss += loss.item()
            global_counter['train'] += 1
            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

            torch.cuda.empty_cache()
            if step % 1000 == 0:
                sample_and_plot_batch_flow_matching(
                    diffusion_model=diffusion.module if isinstance(diffusion, torch.nn.DataParallel) else diffusion,
                    condition_batch=img_A,
                    gt_batch=img_B,
                    writer=writer,
                    step=step,
                    device=DEVICE,
                    tag="Flow_Matching_Sampling",
                    scheduler=inference_scheduler,
                    num_inference_steps=opt.diff_step,
                    mri_modalities=opt.mri_modalities,
                    epoch=epoch,
                    max_samples=4
                    )

        writer.add_scalar('train/epoch_loss', epoch_loss / len(train_loader), epoch)

        if epoch % 5 == 0:
            # Save the model after each epoch.
            os.makedirs(os.path.join(opt.output_dir, opt.experiment_name, f"Epoch_{epoch}"), exist_ok=True)
            torch.save(diffusion.state_dict(), os.path.join(opt.output_dir, opt.experiment_name, f"Epoch_{epoch}",  f'diffusion-ep-{epoch}.pth'))

        old_lr = optimizer.param_groups[0]['lr']
        lr_scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

        # Save the model at the end of training.
        torch.save(diffusion.state_dict(), os.path.join(opt.output_dir, opt.experiment_name, f'latest.pth'))

    print("Training complete.")
