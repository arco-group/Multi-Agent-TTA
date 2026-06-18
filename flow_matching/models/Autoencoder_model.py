from models.UNet import UNet
import torch.nn as nn
import numpy as np
import copy
import torch
import os
import torch.nn.functional as F
from models import networks


# per trainare AE:
# script che fa:
# 1. definisce AE e task network T
# 2. carica i pesi del task network pretrained, e lo freeza
# 3. per ogni batch, la passa a T ottenendo le feature e le passa ad AE che restituisce output
# 4. calcola la loss tra le feature di T e le feature di AE
# NB: nello script degli AE viene fatto solo il passo forward, quindi dalle features di T si ottiene l'output di AE

class AENet(nn.Module):
    def __init__(self, opt):
        super(AENet, self).__init__()  # chiama il costruttore della classe torch.nn.Module
        self.opt = opt
        self.gpu_ids = opt.gpu_ids  # Aggiungi questa riga per definire gpu_ids
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Definisci il device (GPU o CPU)
        self.def_AENet()
        self.AELoss = nn.MSELoss()
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # creiamo un path che ha il nome della cartella in cui
        self.metric = 0
        
        # Definiamo gli ottimizzatori per ogni subnet e li aggiungiamo a una lista
        self.optimizers = []
        # set AENet optimizers (per girarlo su alvis con gpu)
        for subnets in self.AENet:  # subnet sono le reti di autoencoder
            params = []
            subnets.to(self.device)  # Trasferisci le subnet su GPU
            params.extend(list(subnets.parameters()))
            if opt.phase != 'test':
                self.optimizer_AENet = torch.optim.Adam(params, self.opt.aelr)  # ottimizzatore per l'autoencoder
                self.optimizers.append(self.optimizer_AENet) # Aggiungi l'ottimizzatore alla lista degli ottimizzatori
        if opt.phase != 'test':
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]


    def def_AENet(self):
        self.AENet = []
        n0 = self.opt.aenet_dim  # tipicamente 64

        if self.opt.aenet_style == "symmetric":
            self.AENetMatch = [[0]]
            for i in range(1, len(self.opt.tnet_dim) - 1):
                self.AENetMatch += [[i, -i - 1]]
            self.AENetMatch += [[-1]]

            for i in range(len(self.AENetMatch)):
                if len(self.AENetMatch[i]) == 1:
                    dims = self.opt.tnet_dim[i]
                    self.AENet += [UNet(inplane=dims,
                                        midplane=[n0 // 2, n0 // 4, n0 // 8],
                                        outplane=dims, skip=False, isn=True)]
                else:
                    dims = self.opt.tnet_dim[i] * 2
                    self.AENet += [UNet(inplane=dims,
                                        midplane=[n0, n0 // 2, n0 // 4],
                                        outplane=dims, skip=False, isn=True)]
        
        elif self.opt.aenet_style == "per_layer":
            self.AENetMatch = [[i] for i in range(len(self.opt.tnet_dim))]
            for dims in self.opt.tnet_dim:
                if dims == 1:
                    self.AENet += [UNet(inplane=dims, midplane=[n0 // 2, n0 // 4, n0 // 8], \
                        outplane=dims, skip=False, isn=True)]
                else:
                    self.AENet += [UNet(inplane=dims,
                                    midplane=[n0, n0 // 2, n0 // 4],
                                    outplane=dims, skip=False, isn=True)]
        
        else:
            raise ValueError(f"Unknown AENet style: {self.opt.aenet_style}")


    def forward(self, side_out):
        side_out = side_out.to(self.device)
        self.reconstructed_A = self.netAE_A(self.real_A)
        self.reconstructed_B = self.netAE_B(self.real_B)

        return self.reconstructed_A, self.reconstructed_B

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

    def addnoise(self, feat):
        """ Add noise to features for auto-encoders
        feats [batch, channel, H, W]
        """
        # read config from self.opt
        # random permute features
        if self.opt.feat_noise == False:
            return feat
        blks = [16, int(np.ceil(feat.shape[3] / feat.shape[2])) * 16]
        ratio = 0.25
        radius = [feat.shape[2] // blks[0] + 1, feat.shape[3] // blks[1] + 1]
        nums = np.round(blks[0] * blks[1] * ratio * ratio)
        wrong_labels = copy.deepcopy(feat)
        for i in range(feat.shape[0]):
            for _ in range(np.random.randint(nums)):
                rx = np.random.randint(1, radius[0] + 1)
                ry = np.random.randint(1, radius[1] + 1)
                mcx = np.random.randint(rx + 1, feat.shape[2] - rx - 1)
                mcy = np.random.randint(ry + 1, feat.shape[3] - ry - 1)
                mcx_src = np.random.randint(rx + 1, feat.shape[2] - rx - 1)
                mcy_src = np.random.randint(ry + 1, feat.shape[3] - ry - 1)
                wrong_labels[i, :, mcx - rx:mcx + rx, mcy - ry:mcy + ry] = feat[i, :, mcx_src - rx:mcx_src + rx,
                                                                           mcy_src - ry:mcy_src + ry]
        return wrong_labels

    def save_networks_AE(self, epoch, AE_to_train=None, step=None):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        path = os.path.join(self.save_dir, f'epoch{epoch}')
        if not os.path.exists(path):
            os.makedirs(path)
        if AE_to_train is None:
            for i in range(len(self.AENet)):
                # se la cartella non esiste, creala
                weight_path = os.path.join(path,f'AE_{self.opt.return_layers[i]}_{epoch}.pt')  # AE_feature su cui addestro_epoca

                if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(self.AENet[i].cpu().state_dict(), weight_path)
                    self.AENet[i].cuda(self.gpu_ids[0])
                else:
                    torch.save(self.AENet[i].cpu().state_dict(), weight_path)
        else:
            weight_path = os.path.join(path, f'AE_{self.opt.return_layers[AE_to_train]}_{epoch}.pt')

            if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
                if step is None:
                    torch.save(self.AENet[AE_to_train].cpu().state_dict(), weight_path)
                else:
                    torch.save(
                        {
                            "step": step,
                            "AE_weights": self.AENet[AE_to_train].cpu().state_dict(),
                            "optimizer": self.optimizers[AE_to_train].state_dict(),
                            "scheduler": self.schedulers[AE_to_train].state_dict(),
                        },
                        weight_path
                    )
                self.AENet[AE_to_train].cuda(self.gpu_ids[0])
            else:
                if step is None:
                    torch.save(self.AENet[AE_to_train].cpu().state_dict(), weight_path)
                else:
                    torch.save(
                        {
                            "step": step,
                            "AE_weights": self.AENet[AE_to_train].cpu().state_dict(),
                            "optimizer": self.optimizers[AE_to_train].state_dict(),
                            "scheduler": self.schedulers[AE_to_train].state_dict(),
                        },
                        weight_path
                    )


    #TODO:
    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_aelr = self.optimizers[0].param_groups[0]['lr'] # qui è lr perchè il dizionario param_groups ha per chiave lr
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        aelr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_aelr, aelr))

  
