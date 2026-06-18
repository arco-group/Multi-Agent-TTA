import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from PIL import Image
import os
import numpy as np
import torch
import matplotlib.pyplot as plt


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

def save_images(batch, fake_image_B, epoch, path, epoch_iter):
    path = path + f'/epoch{epoch}' + '/saved_images'

    real_image_A = batch['A']
    real_image_B = batch['B']

    mapping_modality = batch['modality_mapping']

    # Da [-1, 1] a [0, 1]
    real_image_A = (real_image_A + 1) / 2
    real_image_B = (real_image_B + 1) / 2
    fake_image_B = (fake_image_B + 1) / 2

    if real_image_A.shape[0] > 4:
        real_image_A = real_image_A[:4]
        real_image_B = real_image_B[:4]
        fake_image_B = fake_image_B[:4]

    # ---- Plotting ---- #
    ld = real_image_A.cpu().detach()
    gt = real_image_B.cpu().detach()
    pred = fake_image_B.cpu().detach()
    error = norm_percentile(abs(pred - gt))
    B = real_image_A.shape[0]

    fig, axes = plt.subplots(nrows=B, ncols=4, figsize=(8, 2.5 * B))
    if B == 1:
        axes = [axes]  # make iterable

    for i in range(B):
        images = [ld[i], gt[i], pred[i], error[i]]
        titles = [mapping_modality['A'][i], mapping_modality['B'][i], "Prediction", "Error"]

        for j in range(4):
            ax = axes[i][j] if B > 1 else axes[0][j]
            ax.set_axis_off()
            ax.set_title(titles[j])
            img = images[j].squeeze(0).cpu().numpy()
            ax.imshow(img, cmap='hot' if titles[j] == "Error" else 'gray')

    plt.tight_layout()
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, f'{epoch_iter}.png'))
    plt.close()


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                save_images(data, model.fake_B, epoch, model.save_dir, epoch_iter)
                # visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
                
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
