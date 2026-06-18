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
from tqdm import tqdm

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

# torch.cuda.set_per_process_memory_fraction(0.25, 0)

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers,default: ./checkpoints, otherwise 
                                   # you must define opt.checkpoints_dir for a different checkpoints, like checkpoints_MRI or checkpoints_CT_PET

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    mae=[]
    psnr=[]
    ssim_list=[]
    df = pd.DataFrame(columns=['img_name', 'SSIM', 'MAE',  'PSNR'])
    for i, data in tqdm(enumerate(dataset)):
        # print (i)
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        #visuals['fake_B'] = model.real_A # solo per vedere img low e high dose
        img_path = model.get_image_paths()     # get image paths

        mae_score=calcola_mse(visuals)
        mae.append(mae_score)

        psnr_score=calculate_psnr(visuals)
        psnr.append(psnr_score)

        ssim_score=calculate_ssim(visuals)
        ssim_list.append(ssim_score)

        row = {'img_name': img_path, 'SSIM': ssim_score, 'MAE': mae_score, 'PSNR': psnr_score}
        df_new = pd.DataFrame([row])
        df = pd.concat([df, df_new], ignore_index=True)

        
       # print ('mse', mse)

        #if i % 5 == 0:  # save images to an HTML file
         #   print('processing (%04d)-th image... %s' % (i, img_path))
        # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)

        if i % 100 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb, return_layers='/input')


    webpage.save()  # save the HTML
    print ('lunghezza', len(mae))
    ssim_mean=round(np.mean(ssim_list),4)
    ssim_std = round(np.std(ssim_list), 4)

    mae_mean=round(np.mean(mae),4)
    mae_std=round(np.std(mae),4)

    psnr_mean = round(np.mean(psnr),4)
    psnr_std =round(np.std(psnr),4)
    
    txt_path = os.path.join(web_dir, 'metrics.txt')
    csv_path = os.path.join(web_dir, 'metrics.csv')

    #with open(os.path.join(opt.results_dir,  f'{opt.results_dir}.txt'), 'w') as f:
    with open(txt_path, 'w') as f:
        f.write("SSIM task model:" + str(ssim_mean) + '±' + str(ssim_std) + '\n')
        f.write("MAE task model:" + str(mae_mean) + '±' + str(mae_std) + '\n')
        f.write("PSNR task model:" + str(psnr_mean) + '±' + str(psnr_std) + '\n')

    df.to_csv(csv_path, index=False)
    #df.to_csv(os.path.join(opt.results_dir, f'{opt.results_dir}.csv'), index=False)
    #df.to_csv(os.path.join(data_dir, 'cyclegan_regressor_terzo_10_1.csv'), index=False)
