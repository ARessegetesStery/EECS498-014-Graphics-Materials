import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import os, imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms.functional as TF
import imageio
import gdown
from nerf_model import VeryTinyNerfModel, get_rays, render
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(device)

def train(rawData, model, optimizer, n_iters=3000):
    """
    Train the Neural Radiance Field (NeRF) model. This function performs training over a specified number of iterations,
    updating the model parameters to minimize the difference between rendered and actual images.

    Parameters:
    model (torch.nn.Module): The NeRF model to be trained.
    optimizer (torch.optim.Optimizer): The optimizer used for training the model.
    n_iters (int): The number of iterations to train the model. Default is 3000.
    """

    rawData = np.load("tiny_nerf_data.npz", allow_pickle=True)
    images = rawData["images"]
    poses = rawData["poses"]
    focal = rawData["focal"]
    H, W = images.shape[1:3]
    H = int(H)
    W = int(W)

    testimg, testpose = images[99], poses[99]   
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    testimg = torch.Tensor(testimg).to(device)
    testpose = torch.Tensor(testpose).to(device)

    psnrs = []
    iternums = []

    plot_step = 500
    n_samples = 64   # Number of samples along each ray.

    for i in tqdm(range(n_iters)):
        # Randomly select an image from the dataset and use it as the target for training.
        images_idx = np.random.randint(images.shape[0])
        target = images[images_idx]
        pose = poses[images_idx]


        #############################################################################
        #                                   TODO: Task 5 A                          #
        #############################################################################
        # Perform training. Use mse loss for loss calculation and update the model parameter using the optimizer.
        # Compute the rays for the selected image.
        # Render the scene using the current model state.
        # 

        # rays_o, rays_d = ...
        # rgb, depth = ...

        # optimizer... 
        # image_loss = ...
        # image_loss.backward() # calculate the gradient w.s.t image_loss
        # optimizer.step() # do update

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        if i % plot_step == 0:
            torch.save(model.state_dict(), 'ckpt.pth')
            # Render a test image to evaluate the current model performance.
            with torch.no_grad():
                rays_o, rays_d = get_rays(H, W, focal, testpose)
                rgb, depth = render(model, rays_o, rays_d, near=1., far=7., n_samples=n_samples)
                loss = torch.nn.functional.mse_loss(rgb, testimg)
                # Calculate PSNR for the rendered image.
                psnr = mse2psnr(loss)

                psnrs.append(psnr.detach().cpu().numpy())
                iternums.append(i)

                # Plotting the rendered image and PSNR over iterations.
                plt.figure(figsize=(9, 3))

                plt.subplot(131)
                picture = rgb.cpu()  # Copy the rendered image from GPU to CPU.
                plt.imshow(picture)
                plt.title(f'RGB Iter {i}')

                plt.subplot(132)
                picture = depth.cpu() * (rgb.cpu().mean(-1)>1e-2)
                plt.imshow(picture, cmap='gray')
                plt.title(f'Depth Iter {i}')

                plt.subplot(133)
                plt.plot(iternums, psnrs)
                plt.title('PSNR')
                plt.savefig(f"training_results_{i}.png")

def main():
    # load the tiny nerf dataset
    rawData = np.load("tiny_nerf_data.npz", allow_pickle=True)
    images = rawData["images"]
    poses = rawData["poses"]
    focal = rawData["focal"]
    H, W = images.shape[1:3]
    H = int(H)
    W = int(W)
    print("Images: {}".format(images.shape))
    print("Camera Poses: {}".format(poses.shape))
    print("Focal Length: {:.4f}".format(focal))

    # show an example image
    testimg, testpose = images[99], poses[99]
    plt.imshow(testimg)
    plt.title('Dataset example')
    plt.savefig("input_data_example.png")


    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    testimg = torch.Tensor(testimg).to(device)
    testpose = torch.Tensor(testpose).to(device)


    # Test your rendering
    if os.path.exists('pretrained.pth') is False:
        url = 'https://drive.google.com/uc?id=1Mj9A7f4BsbPLw2cUGcUgCkDArTJz9q_h'
        output = 'pretrained.pth'
        gdown.download(url, output, quiet=False)

    nerf = VeryTinyNerfModel()
    nerf = nn.DataParallel(nerf).to(device)
    ckpt = torch.load('pretrained.pth')
    nerf.load_state_dict(ckpt)
    test_img_idx_list = [0, 40, 85]
    with torch.no_grad():
        test_img_idx = test_img_idx_list[-1]
        rays_o, rays_d = get_rays(H, W, focal, poses[test_img_idx])
        #############################################################################
        #                                   TODO: Task 4                            #
        #############################################################################
        # Render the scene using the current model state. You may want to use near = 2, far = 6, n_samples = 64 
        
        # rgb, depth = ..., 

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        plt.figure(figsize=(9,3))

        plt.subplot(131)
        picture = rgb.cpu()
        plt.title("RGB Prediction #{}".format(test_img_idx))
        plt.imshow(picture)

        plt.subplot(132)
        picture = depth.cpu() * (rgb.cpu().mean(-1)>1e-2)
        plt.imshow(picture, cmap='gray')
        plt.title("Depth Prediction #{}".format(test_img_idx))

        plt.subplot(133)
        plt.title("Ground Truth #{}".format(test_img_idx))
        plt.imshow(rawData["images"][test_img_idx])
        plt.savefig("rendering_test_results_provided_model.png")

    # Train your model
    # Feel free to commit the below before you really train something.
    nerf = VeryTinyNerfModel()
    nerf = nn.DataParallel(nerf).to(device)
    optimizer = torch.optim.Adam(nerf.parameters(), lr=5e-3, eps = 1e-7)
    train(rawData, nerf, optimizer)

    with torch.no_grad():
        for test_img_idx in test_img_idx_list:
            rays_o, rays_d = get_rays(H, W, focal, poses[test_img_idx])

            #############################################################################
            #                                   TODO: Task 5 B                          #
            #############################################################################
            # Render the scene using the current model state. You may want to use near = 2, far = 6, n_samples = 64.

            # rgb, depth = ...
            #############################################################################
            #                             END OF YOUR CODE                              #
            #############################################################################
            plt.figure(figsize=(9,3))

            plt.subplot(131)
            picture = rgb.cpu()
            plt.title("RGB Prediction #{}".format(test_img_idx))
            plt.imshow(picture)

            plt.subplot(132)
            picture = depth.cpu() * (rgb.cpu().mean(-1)>1e-2)
            plt.imshow(picture, cmap='gray')
            plt.title("Depth Prediction #{}".format(test_img_idx))

            plt.subplot(133)
            plt.title("Ground Truth #{}".format(test_img_idx))
            plt.imshow(rawData["images"][test_img_idx])
            plt.savefig(f"rendering_test_results_trained_model{test_img_idx}.png")

if __name__ == '__main__':
    main()