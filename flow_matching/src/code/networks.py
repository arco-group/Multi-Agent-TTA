import os
from typing import Optional
# from diffusers import UNet2DConditionModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from generative.networks.nets import (
    AutoencoderKL,
    PatchDiscriminator,
    DiffusionModelUNet,
    ControlNet
)

from src.code.diffusion_unet_modified_forward import ModifiedDiffusionModelUNet

# from .my_unet import DiffusionModelUNetAleatoricConcat
"""
def load_if(checkpoints_path: Optional[str], network: nn.Module) -> nn.Module:
    
    # Load pretrained weights if available.

    # Args:
    #    checkpoints_path (Optional[str]): path of the checkpoints
    #    network (nn.Module): the neural network to initialize 

    # Returns:
    #    nn.Module: the initialized neural network
    
    if checkpoints_path is not None:
        assert os.path.exists(checkpoints_path), 'Invalid path'
        print("Loading checkpoint...")
        network.load_state_dict(torch.load(checkpoints_path))
    return network
"""  

def load_if(checkpoints_path: Optional[str], network: nn.Module) -> nn.Module:
    """
    Load pretrained weights if available.

    Args:
        checkpoints_path (Optional[str]): path of the checkpoints
        network (nn.Module): the neural network to initialize 

    Returns:
        nn.Module: the initialized neural network
    """
    if checkpoints_path is not None:
        print("Loading checkpoint...")

        if not checkpoints_path.endswith('.pth'):
            checkpoints_path = checkpoints_path + '/latest.pth'

        # Load the checkpoint
        checkpoint = torch.load(checkpoints_path, map_location=torch.device("cpu"))

        # Remove 'module.' prefix if present
        new_checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}

        # Load updated state dict
        network.load_state_dict(new_checkpoint)
    
    return network


def init_ddpm(checkpoints_path: Optional[str] = None) -> nn.Module:
    ddpm = ModifiedDiffusionModelUNet(
        spatial_dims=2,  # 2D data (CT slices); use 3 for volumetric 3D CT
        in_channels=2,  # Concatenation of [x_t (noisy high-dose), x_ld (low-dose)] → 2 channels
        out_channels=1,  # Predict noise only; model doubles this to also predict log variance (output = 2 channels)
        num_res_blocks=(2, 2, 2, 2),  # Number of residual blocks at each U-Net level (encoder/decoder depth)
        num_channels=(64, 128, 128, 256),  # Number of feature channels at each U-Net level (controls width of the network)
        attention_levels=(False, False, True, True),  # Whether to use self-attention at each level (deeper levels benefit more)
        norm_num_groups=32,  # Number of groups for GroupNorm (default value used in most LDMs)
        norm_eps=1e-6,  # Epsilon for GroupNorm to avoid divide-by-zero issues
        resblock_updown=False,  # If True, uses residual blocks for up/downsampling (set False for standard blocks)
        num_head_channels=8,  # Number of channels per attention head (used only where attention is enabled)
        with_conditioning=False,  # IMPORTANT: Set to False because you are using spatial concatenation (not cross-attention)
        transformer_num_layers=1,  # Number of layers in the transformer blocks (only applies to attention-enabled levels)
        cross_attention_dim=None,  # Not needed since you're not using cross-attention
        upcast_attention=False,  # Set to True if you want full-precision attention (usually only needed for FP16 training stability)
        use_flash_attention=True,  # Set to True only if using xFormers + GPU to enable flash attention (faster, lower memory)
        dropout_cattn=0.0  # Dropout in the cross/self-attention layers (typically 0.0 unless overfitting)
    )
    return load_if(checkpoints_path, ddpm)








