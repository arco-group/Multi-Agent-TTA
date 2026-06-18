"""Test script for the flow-matching monitoring-agent branch."""
import os
import pandas as pd
from options.test_options import TestOptions
from util.visualizer import calcola_mse, calculate_psnr, calculate_ssim
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np
import torch
from models.Autoencoder_model import AENet
from collections import OrderedDict
from tqdm import tqdm 


try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', required=False, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--diff_ckpt', required=False, default=None, type=str)
    parser.add_argument('--experiment_name', required=True, type=str)
    parser.add_argument('--annotation_A', required=False, type=str)
    parser.add_argument('--annotation_B', required=False, type=str)
    parser.add_argument('--num_workers', default=0, type=int)
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
    parser.add_argument('--aenet_dim', type=int, default=64, help=' feature AE')
    parser.add_argument('--aelr', default=1.5e-5, type=float)
    parser.add_argument('--aenet_style', default="per_layer", type=str)
    parser.add_argument('--AE_to_train', type=int, default=11)
    parser.add_argument('--diff_step', type=int, default=10, help='number of steps for reverse diffusion process in inference')
    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--under_sample_dataset', action="store_true", help='True undersample the dataset deleting one slice every three')
    parser.add_argument('--phase', type=str, default=None, help='train or test, if None dont split')

    opt = parser.parse_args()
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    print('lunghezza', len(dataset))
    
    model = create_model(opt)  # create a model given opt.model and other options
    model = networks.init_ddpm(opt.diff_ckpt).to(DEVICE)

    # model.setup(opt)               # regular setup: load and print networks; create schedulers
    # visualizer = Visualizer()

    load_path_model = os.path.join(opt.diff_ckpt, 'latest.pth')
    
    state_dict_model = torch.load(load_path_model,
                                  map_location=str(model.device))  # State dict for the task model weights.

    
    AENet = AENet(opt)

    load_path_weights_AE_input = os.path.join(opt.output_dir.replace('TEST/', ""), 'epoch40/AE_0_40.pt')
    load_path_weights_AE_output = os.path.join(opt.output_dir.replace('TEST/', ""), 'epoch40/AE_11_40.pt')

    state_dict_input = torch.load(load_path_weights_AE_input, map_location=str(
        AENet.device))  # State dict containing one entry per layer.
   
    state_dict_output = torch.load(load_path_weights_AE_output,
                                   map_location=str(AENet.device))  # State dict for the output autoencoder.

    AENet.AENet[0].load_state_dict(state_dict_input)  # Load the state dict into the model.
    AENet.AENet[-1].load_state_dict(state_dict_output)

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name,
                               config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='Autoencoder')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name,
                           '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    if opt.eval:
        model.eval()
        for net in AENet.AENet:  # Put the autoencoder in evaluation mode.
            net.eval()

    mse_i = []
    psnr_i = []
    ssim_list_i = []
    mse_o = []
    psnr_o = []
    ssim_list_o = []

    df_i = pd.DataFrame(columns=['img_name', 'MAE', 'PSNR', 'SSIM'])
    df_o = pd.DataFrame(columns=['img_name', 'MAE', 'PSNR', 'SSIM'])

    for i, data in tqdm(enumerate(dataset)):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        outputs = model.forward(return_layers=opt.return_layers)  # Forward pass of the task model.
        input_real = outputs[
            opt.return_layers[0]]  # Return layer used by the task model.
        # AE input corresponds to real A.
        output_real = outputs[opt.return_layers[-1]]  # Fake B.

        input_rec = AENet.AENet[0](input_real, side_out=False)  # Feed the task-model output to the autoencoder.
        output_rec = AENet.AENet[-1](output_real, side_out=False)

        # visuals = AENet.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths

        # Create two dictionaries to evaluate the reconstruction performance of the two autoencoders.
        # The first, visual_input, uses real_A as real_B and the reconstruction from AE_0 as fake_B.
        visuals_input = OrderedDict()
        visuals_input['real_B'] = input_real
        visuals_input['fake_B'] = input_rec

        # The second, visual_output, uses the CycleGAN target-domain output as real_B
        # and the reconstruction from AE_3 as fake_B.
        visuals_output = OrderedDict()
        visuals_output['real_B'] = output_real
        visuals_output['fake_B'] = output_rec

        # TODO: use a dictionary when calling the metric functions.
        # AE 0 operates on the input branch.
        mae_score = calcola_mse(visuals_input)  # Computed per batch.
        mse_i.append(mae_score)
        psnr_score = calculate_psnr(visuals_input)
        psnr_i.append(psnr_score)
        ssim_score = calculate_ssim(visuals_input)
        ssim_list_i.append(ssim_score)

        row = {'img_name': img_path, 'MAE': mae_score, 'PSNR': psnr_score, 'SSIM': ssim_score}
        df_new = pd.DataFrame([row])
        df_i = pd.concat([df_i, df_new], ignore_index=True)

        # AE 8 che lavora su output
        mae_score = calcola_mse(visuals_output)  # Computed per batch.
        mse_o.append(mae_score)
        psnr_score = calculate_psnr(visuals_output)
        psnr_o.append(psnr_score)
        ssim_score = calculate_ssim(visuals_output)
        ssim_list_o.append(ssim_score)

        row = {'img_name': img_path, 'MAE': mae_score, 'PSNR': psnr_score, 'SSIM': ssim_score}
        df_new = pd.DataFrame([row])
        df_o = pd.concat([df_o, df_new], ignore_index=True)

        # print ('mse', mse)

        if i % 100 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
            save_images(webpage, visuals_input, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize,
                        use_wandb=opt.use_wandb, return_layers='/input')
            save_images(webpage, visuals_output, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize,
                        use_wandb=opt.use_wandb, return_layers='/output')

    webpage.save()  # save the HTML
    print('------ input ------')
    print('lunghezza', len(mse_i))
    mse_mean = np.mean(mse_i)
    mse_std = np.std(mse_i)
    psnr_mean = np.mean(psnr_i)
    psnr_std = np.std(psnr_i)
    ssim_mean = np.mean(ssim_list_i)
    ssim_std = np.std(ssim_list_i)
    print('ssim mean', ssim_mean)
    print('ssim std', ssim_std)
    print('mae mean', mse_mean)
    print('mae std', mse_std)
    print('psnr mean', psnr_mean)
    print('psnr std', psnr_std)
    # salvo i print in un file txt
    with open(os.path.join(opt.results_dir, 'input_AE_0.txt'), 'w') as f:
        f.write("SSIM_AE_0:" + str(ssim_mean) + '±' + str(ssim_std) + '\n')
        f.write("MAE_AE_0:" + str(mse_mean) + '±' + str(mse_std) + '\n')
        f.write("PSNR_AE_0:" + str(psnr_mean) + '±' + str(psnr_std) + '\n')

    print('------ output ------')
    print('lunghezza', len(mse_o))
    mse_mean = np.mean(mse_o)
    mse_std = np.std(mse_o)
    psnr_mean = np.mean(psnr_o)
    psnr_std = np.std(psnr_o)
    ssim_mean = np.mean(ssim_list_o)
    ssim_std = np.std(ssim_list_o)
    print('ssim std', ssim_mean)
    print('ssim std', ssim_std)
    print('mae mean', mse_mean)
    print('mae std', mse_std)
    print('psnr mean', psnr_mean)
    print('psnr std', psnr_std)
    # salvo i print in un file txt
    with open(os.path.join(opt.results_dir, 'output_AE_8.txt'), 'w') as f:
        f.write("SSIM_AE_8:" + str(ssim_mean) + '±' + str(ssim_std) + '\n')
        f.write("MAE_AE_8:" + str(mse_mean) + '±' + str(mse_std) + '\n')
        f.write("PSNR_AE_8:" + str(psnr_mean) + '±' + str(psnr_std) + '\n')

    df_i.to_csv(os.path.join(opt.results_dir, 'input_AE_0.csv'), index=False)  # salva il dataframe in un file csv
    df_o.to_csv(os.path.join(opt.results_dir, 'output_AE_8.csv'), index=False)  # salva il dataframe in un file csv
