from models.UNet import UNet
import torch.nn as nn
import numpy as np
import copy
import torch
import os
import torch.nn.functional as F
from models import networks
from generative.networks.nets.diffusion_model_unet import get_timestep_embedding, get_up_block

def init_weights(x):
    # torch.manual_seed(0)
    if type(x) == nn.Conv2d:
        nn.init.kaiming_normal_(x.weight.data)
        nn.init.zeros_(x.bias.data)

"""
def init_weights_eye(x, channel=64):
    # indentity init, only works for same input/output channel
    if type(x) == nn.Conv2d:
        eye = nn.init.eye_(torch.empty(x.weight.shape[0], x.weight.shape[1])).unsqueeze(-1).unsqueeze(-1)
        init_bias = nn.init.zeros_(torch.empty(x.weight.shape[0]))
        x.weight.data = eye
        x.bias.data = init_bias
"""

def init_weights_eye(m):
    if isinstance(m, nn.Conv2d):
        device = m.weight.device
        with torch.no_grad():
            eye = torch.eye(m.in_channels, device=device).view(m.in_channels, m.in_channels, 1, 1)
            m.weight.copy_(eye)
            m.bias.zero_()
            # print(f"[init_eye] Conv({m.in_channels}x{m.out_channels}) set to identity on {device}")

class DTTAnorm(nn.Module):  # è il metodo di Karani
    def __init__(self, opt):
        super(DTTAnorm, self).__init__()
        self.opt = opt  # TODO: aggiunto io
        self.usedtta = opt.usedtta  # TODO: aggiunto io

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self,x):
        x_ = self.conv1(x)
        scale = (torch.randn([1,16,1,1]) * 0.05 + 0.2).to(x_.device)
        x_ = torch.exp(-(x_**2) / (scale**2))
        x_ = self.conv2(x_)
        x_ = torch.exp(-(x_**2) / (scale**2))
        x_ = self.conv3(x_)
        return x_ + x


def init_weights_zero(m):
    if isinstance(m, nn.Conv2d):
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)

class AdpNetIrene(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(64),
            nn.Conv2d(64, 64, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(64),
            nn.Conv2d(64, 1, 1),
            nn.InstanceNorm2d(1)
        )
        self.net.apply(init_weights_zero)   # inizializza tutto a zero

    def forward(self, x):
        return x + self.net(x)


class ANet(nn.Module):
    def __init__(self, opt):
        """Adaptor Net: Default is for a 4-level UNet with fixed 64 channels
        Args:
            AENet: nn.Module, pre-trained auto-encoder for the source image
            channel: int, input feature channel of the affine transform
            seq: list->int, index of the affine matrix to be used
        """
        super(ANet, self).__init__()
        self.opt = opt  # TODO: aggiunto io
        # tnet_dim = opt.tnet_dim  # TODO: aggiunto io
        tnet_dim = [1, 64, 64, 128, 128, 256, 256, 256, 128, 128, 64, 1]
        self.ALoss=nn.MSELoss()
        self.save_dir = os.path.join(opt.results_dir, 'weights') # pesi per ogni epoca durante il training della singola config
        self.save_dir_config = os.path.join(opt.results_dir, 'config_weights') #pesi migliori per ogni config

        adpNet = None
        if self.opt.usedtta:  # se usedtta è True, allora usa DTTA, metodo di karani!!!
            adpNet = DTTAnorm(opt)
        seq = self.opt.seq

        self.conv = nn.ModuleList() # lista di layer per adattare
        feature_channel = tnet_dim[1:-1] # canali a livello delle feature
        self.channel = feature_channel
        nums = len(self.channel)
        self.nums = nums
        if seq is None:
            self.seq = np.arange(nums)
        else:
            self.seq = seq
        # use pre-contrast manipulation
        self.adpNet = adpNet
        if adpNet is None:
            """
            self.adpNet = nn.Sequential(
                nn.Conv2d(1, 64, 1),
                nn.LeakyReLU(negative_slope=0.2),
                nn.InstanceNorm2d(64),
                nn.Conv2d(64, 64, 1),
                nn.LeakyReLU(negative_slope=0.2),
                nn.InstanceNorm2d(64),
                nn.Conv2d(64, 1, 1),
                nn.LeakyReLU(negative_slope=0.2),
                nn.InstanceNorm2d(1)
            )
            self.adpNet.apply(init_weights)
            """
            self.adpNet = AdpNetIrene()
            # use feature affine transform
        for c in self.channel: #adaptor a liv feature
            convs = nn.Conv2d(c, c, 1)
            self.conv.append(convs)
        self.conv.apply(init_weights_eye)

        self.optimizer_ANet = torch.optim.Adam(self.parameters(), opt.alr)  # ottimizzatore per l'adattatore


    def reset(self, default=True):
        # reset the fine-tuned weights for a new test subject
        # np.random.seed(0)
        #torch.manual_seed(0)
        if default:
            self.conv.apply(init_weights_eye)
            # self.adpNet.apply(init_weights)
            self.adpNet.apply(init_weights_zero)
        else:
            path = os.environ.get("MULTI_AGENT_TTA_RESET_DIR", "checkpoints/reset")
            # carichiamo adpNet
            weight_path = os.path.join(path, 'adpNet.pth')
            if os.path.exists(weight_path):
                state_dict = torch.load(weight_path)
                self.adpNet.load_state_dict(state_dict)
            else:
                print(f"File {weight_path} not found")
            # carichiamo i conv
            for i in range(len(self.conv)):
                weight_path = os.path.join(path, f'conv_{i}.pth')
                if os.path.exists(weight_path):
                    state_dict = torch.load(weight_path)
                    self.conv[i].load_state_dict(state_dict)
                else:
                    print(f"File {weight_path} not found")
                    break
        self.cuda()

    def forward(self, batch, task_model, x_minus_one, timesteps, scheduler):
        """
        Forward for a 4-level UNet
        Args:
            TNet: nn.Module. The pretrained task network
            side_out: bool. If true, output every intermediate results
            seq: list->int or np array. Position of 1x1 convolution
        """
        # data img, pri a la passiamo nell'adaptor
        real_A = torch.Tensor(batch['A']).unsqueeze(0).to('cuda') # batch è un diz che contiene per ogni campione, le img di dominio A e B
        x = self.adpNet(real_A) # prendo l'immagine del dominio A e la passo all'adaptor

        outputs = {}  # Dizionario che conterrà i vari output richiesti sarebbe hs della DiffusionUnet  
        start = 0
        outputs[start]= x # immagine adattata
        start += 1

        t_emb = get_timestep_embedding(timesteps, task_model.block_out_channels[0])
        t_emb = t_emb.to(dtype=x.dtype)
        emb = task_model.time_embed(t_emb)

        # 3. initial convolution
        x = torch.cat([x_minus_one, x], dim=1)

        h = task_model.conv_in(x)

        if start in self.opt.return_layers:
            h=self.conv[start-1](h) #adattamento a livello di feature
            outputs[start] = h
        start += 1

        down_block_res_samples = [h]
        for downsample_block in task_model.down_blocks:
            h, res_samples = downsample_block(hidden_states=h, temb=emb, context=None)

            if start in self.opt.return_layers:
                h=self.conv[start-1](h) #adattamento a livello di feature
                outputs[start] = h
            start += 1

            for residual in res_samples:
                down_block_res_samples.append(residual)

        h = task_model.middle_block(hidden_states=h, temb=emb, context=None)
        if start in self.opt.return_layers:
            h=self.conv[start-1](h) #adattamento a livello di feature
            outputs[start] = h
        start += 1

        for upsample_block in task_model.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            h = upsample_block(hidden_states=h, res_hidden_states_list=res_samples, temb=emb, context=None)

            if start in self.opt.return_layers:
                h=self.conv[start-1](h) #adattamento a livello di feature
                outputs[start] = h
            start += 1

        final_output = task_model.out(h)
        final_output, _ = scheduler.step(final_output, timesteps, x_minus_one, 0)
        outputs[start] = final_output
        return outputs

    # prendo x reale, lo passo a T, mi fa il forward e mi da le uscite. poi adatto singolarmente ogni blocchettino.
    # final_output non viene adattato, perchè è l'output finale del task model.

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def save_networks(self, epoch, config=False):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        if not config:
            path = os.path.join(self.save_dir, epoch)
        else:
            path = os.path.join(self.save_dir_config, epoch)
        if not os.path.exists(path):
            os.makedirs(path)
        # salviamo adpNet
        weight_path = os.path.join(path, 'adpNet.pth')
        if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.adpNet.cpu().state_dict(), weight_path)
            self.adpNet.cuda()
        else:
            torch.save(self.adpNet.cpu().state_dict(), weight_path)
        # salviamo i conv
        for i in range(len(self.conv)):
            weight_path = os.path.join(path, f'conv_{i}.pth')
            if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(self.conv[i].cpu().state_dict(), weight_path)
                self.conv[i].cuda()
            else:
                torch.save(self.conv[i].cpu().state_dict(), weight_path)

    def load_networks(self, epoch, config=False):
        if not config:
            path = os.path.join(self.save_dir, epoch)
        else:
            path = os.path.join(self.save_dir_config, epoch)
        # carichiamo adpNet
        weight_path = os.path.join(path, 'adpNet.pth')
        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path)
            self.adpNet.load_state_dict(state_dict)
        else:
            print(f"File {weight_path} not found")
        # carichiamo i conv
        for i in range(len(self.conv)):
            weight_path = os.path.join(path, f'conv_{i}.pth')
            if os.path.exists(weight_path):
                state_dict = torch.load(weight_path)
                self.conv[i].load_state_dict(state_dict)
            else:
                print(f"File {weight_path} not found")
                break

