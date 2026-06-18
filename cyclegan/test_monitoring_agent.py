"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
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
import re
# torch.cuda.set_per_process_memory_fraction(0.33, 0)


try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    print('lunghezza', len(dataset))
    model = create_model(opt)  # create a model given opt.model and other options

    # model.setup(opt)               # regular setup: load and print networks; create schedulers
    # visualizer = Visualizer()


    pattern = r"/TEST_[^/]*/"
    if opt.model == 'pix2pix':
        load_path_model = os.path.join(re.sub(pattern, "/", opt.results_dir.replace('_rec_models', "")), 'latest_net_G.pth')
    else:
        load_path_model = os.path.join(re.sub(pattern, "/", opt.results_dir.replace('_rec_models', "")), 'latest_net_G_A.pth')

    state_dict_model = torch.load(load_path_model,
                                  map_location=str(model.device))  # dizionario che ha per chiave il nome del layer

    if opt.model == 'pix2pix':
        if isinstance(model.netG, torch.nn.DataParallel): # se parallelizzo il modello (uso più di 1 gpu)
            model.netG.module.load_state_dict(state_dict_model) # parallelizzo ogni rete sulla gpu ( T model ha 4 reti)
        else:
            model.netG.load_state_dict(state_dict_model)
    else:
        if isinstance(model.netG_A, torch.nn.DataParallel): # se parallelizzo il modello (uso più di 1 gpu)
            model.netG_A.module.load_state_dict(state_dict_model) # parallelizzo ogni rete sulla gpu ( T model ha 4 reti)
        else:
            model.netG_A.load_state_dict(state_dict_model)

    AENet = AENet(opt)

    pattern = r"/TEST_[^/]*/"
    load_path_weights_AE_input = os.path.join(re.sub(pattern, "/", opt.results_dir), 'epoch49/AE_input_49.pt')
    load_path_weights_AE_output = os.path.join(re.sub(pattern, "/", opt.results_dir), 'epoch49/AE_final_output_49.pt')

    state_dict_input = torch.load(load_path_weights_AE_input, map_location=str(
        AENet.device))  # dizionario che ha per chiave il nome del layer e per valore i pesi
   
    state_dict_output = torch.load(load_path_weights_AE_output,
                                   map_location=str(AENet.device))  # dizionarioo che ha per chiave il nome del layer

    # TODO: corregere il caricamento dei pesi perchè ora li dobb caricare in pix2pix
    AENet.AENet[0].load_state_dict(state_dict_input)  # così carico il dizionario nel modello
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
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
        for net in AENet.AENet:  # metto l'autoencoder in modalità evaluation
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
        outputs = model.forward(return_layers=opt.return_layers)  # passo forward del task model
        input_real = outputs[
            opt.return_layers[0]]  # return layer, nel forward del task model, mi dice dove si attaccano
        # AE (è realA)
        output_real = outputs[opt.return_layers[-1]]  # fake B

        input_rec = AENet.AENet[0](input_real, side_out=False).clip(-1,1)  # diamo output task model ad autoencoder
        output_rec = AENet.AENet[-1](output_real, side_out=False).clip(-1,1)

        # visuals = AENet.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths

        """
        visuals_taskmodel = OrderedDict()
        visuals_taskmodel['real_B'] = data['B'].to('cuda')
        visuals_taskmodel['fake_B'] = output_real
        mae_score = calcola_mse(visuals_taskmodel)  # dovrebbe essere per ogni batch
        psnr_score = calculate_psnr(visuals_taskmodel)
        ssim_score = calculate_ssim(visuals_taskmodel)
        print(mae_score)
        print(psnr_score)
        print(ssim_score)
        """

        # Creo due diziomnari perfhè voglio valiutare le performance di ricostruziome di 2 AE:
        # il primo, visual_input ha come real_B l'input (real A) e come fake_B l'img ricostruita da AE_0
        visuals_input = OrderedDict()
        visuals_input['real_B'] = input_real
        visuals_input['fake_B'] = input_rec

        # il secondo, visual_output ha come real_B lp'immagine generata dalla cycle_gan del dominio target (fake B)
        # e come fake_B l'img ricostruita da AE_3
        visuals_output = OrderedDict()
        visuals_output['real_B'] = output_real
        visuals_output['fake_B'] = output_rec

        # TODO: per chiamare le funzioni che calcolano metriche, serve un dizionario
        # AE 0 che lavora su input
        mae_score = calcola_mse(visuals_input)  # dovrebbe essere per ogni batch
        mse_i.append(mae_score)
        psnr_score = calculate_psnr(visuals_input)
        psnr_i.append(psnr_score)
        ssim_score = calculate_ssim(visuals_input)
        ssim_list_i.append(ssim_score)

        row = {'img_name': img_path, 'MAE': mae_score, 'PSNR': psnr_score, 'SSIM': ssim_score}
        df_new = pd.DataFrame([row])
        df_i = pd.concat([df_i, df_new], ignore_index=True)

        # AE 8 che lavora su output
        mae_score = calcola_mse(visuals_output)  # dovrebbe essere per ogni batch
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
        f.write(f"SSIM_AE_0:{ssim_mean:.4f}±{ssim_std:.4f}\n")
        f.write(f"MAE_AE_0:{mse_mean:.4f}±{mse_std:.4f}\n")
        f.write(f"PSNR_AE_0:{psnr_mean:.4f}±{psnr_std:.4f}\n")

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
            f.write(f"SSIM_AE_8:{ssim_mean:.4f}±{ssim_std:.4f}\n")
            f.write(f"MAE_AE_8:{mse_mean:.4f}±{mse_std:.4f}\n")
            f.write(f"PSNR_AE_8:{psnr_mean:.4f}±{psnr_std:.4f}\n")

    df_i.to_csv(os.path.join(opt.results_dir, 'input_AE_0.csv'), index=False)  # salva il dataframe in un file csv
    df_o.to_csv(os.path.join(opt.results_dir, 'output_AE_8.csv'), index=False)  # salva il dataframe in un file csv


