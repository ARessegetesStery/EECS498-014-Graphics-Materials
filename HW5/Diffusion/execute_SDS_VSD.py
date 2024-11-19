import os, sys, shutil
join = os.path.join
import sys
import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import io
from tqdm import tqdm
from datetime import datetime
import random
import imageio
from pathlib import Path
import shutil
import logging

# from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()  # disable warning

from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DDIMScheduler

IMG_EXTENSIONS = ['jpg', 'png', 'jpeg', 'bmp']


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from utils.sds_vsd_utils import (
                                    get_t_schedule, 
                                    get_loss_weights, 
                                    sds_vsd_grad_diffuser, 
                                    phi_vsd_grad_diffuser, 
                                    extract_lora_diffusers,
                                    predict_noise0_diffuser,
                                    update_curve,
                                    get_images,
                                    get_latents,
                                    get_optimizer,
                                )



def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    # parameters
    ### basics
    parser.add_argument('--seed', default=1024, type=int, help='global seed')
    parser.add_argument('--log_steps', type=int, default=50, help='Log steps')
    parser.add_argument('--log_progress', type=str2bool, default=False, help='Log progress')
    parser.add_argument('--log_gif', type=str2bool, default=False, help='Log gif')
    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4', help='Path to the model')
    current_datetime = datetime.now()
    parser.add_argument('--run_date', type=str, default=current_datetime.strftime("%Y%m%d"), help='Run date')
    parser.add_argument('--run_time', type=str, default=current_datetime.strftime("%H%M"), help='Run time')
    parser.add_argument('--store_dir', type=str, default='SDS_VSD_results', help='Working directory')
    parser.add_argument('--half_inference', type=str2bool, default=False, help='inference sd with half precision')
    parser.add_argument('--save_x0', type=str2bool, default=False, help='save predicted x0')
    parser.add_argument('--save_phi_model', type=str2bool, default=False, help='save save_phi_model, lora or simple unet')
    parser.add_argument('--load_phi_model_path', type=str, default='', help='phi_model_path to load')
    parser.add_argument('--use_mlp_particle', type=str2bool, default=False, help='use_mlp_particle as representation')  # Should be False all the time
    parser.add_argument('--init_img_path', type=str, default='', help='init particle from a known image path')
    ### sampling
    parser.add_argument('--num_steps', type=int, default=1000, help='Number of steps for random sampling')
    parser.add_argument('--t_end', type=int, default=980, help='largest possible timestep for random sampling')
    parser.add_argument('--t_start', type=int, default=20, help='least possible timestep for random sampling')
    parser.add_argument('--multisteps', default=1, type=int, help='multisteps to predict x0')
    parser.add_argument('--t_schedule', default='descend', type=str, help='t_schedule for sampling')
    parser.add_argument('--prompt', default="a photograph of an astronaut riding a horse", type=str, help='prompt')
    parser.add_argument('--n_prompt', default="", type=str, help='negative prompt')
    parser.add_argument('--height', default=512, type=int, help='height of image')
    parser.add_argument('--width', default=512, type=int, help='width of image')
    parser.add_argument('--rgb_as_latents', default=True, type=str2bool, help='width of image')
    parser.add_argument('--generation_mode', default='sds', type=str, help='sd generation mode')
    parser.add_argument('--batch_size', default=1, type=int, help='batch_size / overall number of particles')
    parser.add_argument('--particle_num_vsd', default=1, type=int, help='batch size for VSD training')
    parser.add_argument('--particle_num_phi', default=1, type=int, help='number of particles to train phi model')
    parser.add_argument('--guidance_scale', default=None, type=float, help='Scale for classifier-free guidance')
    parser.add_argument('--cfg_phi', default=1., type=float, help='Scale for classifier-free guidance of phi model')
    ### optimizing
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--betas', nargs='+', type=float, default=[0.9, 0.999], help='Betas for Adam optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for Adam optimizer')
    parser.add_argument('--phi_lr', type=float, default=0.0001, help='Learning rate for phi model')
    parser.add_argument('--phi_model', type=str, default='lora', help='models servered as epsilon_phi')
    parser.add_argument('--use_t_phi', type=str2bool, default=False, help='use different t for phi finetuning')
    parser.add_argument('--phi_update_step', type=int, default=1, help='phi finetuning steps in each iteration')
    parser.add_argument('--lora_vprediction', type=str2bool, default=False, help='use v prediction model for lora')
    parser.add_argument('--lora_scale', type=float, default=1.0, help='lora_scale of the unet cross attn')
    parser.add_argument('--use_scheduler', default=False, type=str2bool, help='use_scheduler for lr')
    parser.add_argument('--lr_scheduler_start_factor', type=float, default=1/3, help='Start factor for learning rate scheduler')
    parser.add_argument('--lr_scheduler_iters', type=int, default=300, help='Iterations for learning rate scheduler')
    parser.add_argument('--loss_weight_type', type=str, default='none', help='type of loss weight')
    parser.add_argument('--nerf_init', type=str2bool, default=False, help='initialize with diffusion models as mean predictor')
    parser.add_argument('--grad_scale', type=float, default=1., help='grad_scale for loss in vsd')
    args = parser.parse_args()


    # Pre-Define the parameters for use
    if args.generation_mode == "sds":
        args.num_steps = 500 
        args.log_steps = 50 
        args.lr = 0.03
        args.model_path = 'stabilityai/stable-diffusion-2-1-base' 
        args.loss_weight_type = '1m_alphas_cumprod'
        args.t_schedule = "random"
        args.prompt = "a photograph of an astronaut riding a horse" 
        args.height = 512 
        args.width = 512 
        # args.batch_size = 1 
        if args.guidance_scale is None:
            args.guidance_scale = 7.5
        print("args.guidance_scale is ", args.guidance_scale)
        args.log_progress = True 
        args.save_x0 = True 

    elif args.generation_mode == "vsd":
        args.num_steps = 500
        args.log_steps = 72
        args.seed = 1024 
        args.lr = 0.03 
        args.phi_lr = 0.0001 
        args.use_t_phi = True 
        args.model_path = 'stabilityai/stable-diffusion-2-1-base'
        args.loss_weight_type = '1m_alphas_cumprod' 
        args.t_schedule = 'random' 
        args.phi_model = 'lora' 
        args.lora_scale = 1.0
        args.lora_vprediction = False 
        args.prompt = "a photograph of an astronaut riding a horse" 
        args.height = 512 
        args.width = 512 
        # args.batch_size = 1 
        if args.guidance_scale is None:
            args.guidance_scale = 7.5
        print("args.guidance_scale is ", args.guidance_scale)
        args.log_progress = True 
        args.save_x0 = True 
        args.save_phi_model = True

    else:
        raise NotImplementedError("Please check the generation_mode you passed, and be careful the upper lower case.")



    # Create working directory
    args.run_id = args.run_date + '_' + args.run_time
    args.store_dir = f'{args.store_dir}-{args.run_id}'

    os.makedirs(args.store_dir, exist_ok=True)
    assert args.generation_mode in ['t2i', 'sds', 'vsd']
    assert args.phi_model in ['lora', 'unet_simple']

    # For sds and t2i, use only args.batch_size
    if args.generation_mode in ['t2i', 'sds']:
        args.particle_num_vsd = args.batch_size
        args.particle_num_phi = args.batch_size
    assert (args.batch_size >= args.particle_num_vsd) and (args.batch_size >= args.particle_num_phi)
    if args.batch_size > args.particle_num_vsd:
        print(f'use multiple ({args.batch_size}) particles!! Will get inconsistent x0 recorded')
    ### set random seed everywhere
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU.
    np.random.seed(args.seed)   # Numpy module.
    random.seed(args.seed)      # Python random module.
    torch.manual_seed(args.seed)
    return args


class nullcontext:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    

def main():
    #################################################################################
    #                                config & logger                                #
    #################################################################################
    args = get_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32 # use float32 by default
    image_name = args.prompt.replace(' ', '_')
    shutil.copyfile(__file__, join(args.store_dir, os.path.basename(__file__)))
    ### set up logger
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.basicConfig(filename=f'{args.store_dir}/std_{args.run_id}.log', filemode='w', 
                        format='%(asctime)s %(levelname)s --> %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(f'[INFO] Cmdline: '+' '.join(sys.argv))
    ### log basic info
    args.device = device
    logger.info(f'Using device: {device}; version: {str(torch.version.cuda)}')
    if device.type == 'cuda':
        logger.info(torch.cuda.get_device_name(0))
    logger.info("################# Arguments: ####################")
    for arg in vars(args):
        logger.info(f"\t{arg}: {getattr(args, arg)}")
    
    
    #################################################################################
    #                         load model & diffusion scheduler                      #
    #################################################################################
    logger.info(f'load models from path: {args.model_path}')
    # 1. Load the autoencoder model which will be used to decode the latents into image space. 
    vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae", torch_dtype=dtype)
    # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
    tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer", torch_dtype=dtype)
    text_encoder = CLIPTextModel.from_pretrained(args.model_path, subfolder="text_encoder", torch_dtype=dtype)
    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(args.model_path, subfolder="unet", torch_dtype=dtype)
    # 4. Scheduler
    scheduler = DDIMScheduler.from_pretrained(args.model_path, subfolder="scheduler", torch_dtype=dtype)

    if args.half_inference:
        unet = unet.half()
        vae = vae.half()
        text_encoder = text_encoder.half()
    unet = unet.to(device)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    # all variables in same device for scheduler.step()
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    
    if args.generation_mode == 'vsd':
        if args.phi_model == 'lora':
            vae_phi = vae
            ### unet_phi is the same instance as unet that has been modified in-place
            unet_phi, unet_lora_layers = extract_lora_diffusers(unet, device)      

            phi_params = list(unet_lora_layers.parameters())
        

    elif args.generation_mode == 'sds':
        unet_phi = None
        vae_phi = vae
    


    ### scheduler set timesteps
    num_train_timesteps = len(scheduler.betas)
    if args.generation_mode == 't2i':
        scheduler.set_timesteps(args.num_steps)
    else:
        scheduler.set_timesteps(num_train_timesteps)

    #################################################################################
    #                       initialize particles and text emb                       #
    #################################################################################
    ### get text embedding
    text_input = tokenizer([args.prompt] * max(args.particle_num_vsd, args.particle_num_phi), padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
                                [args.n_prompt] * max(args.particle_num_vsd, args.particle_num_phi), padding="max_length", max_length=max_length, return_tensors="pt"
                            )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings_vsd = torch.cat([uncond_embeddings[:args.particle_num_vsd], text_embeddings[:args.particle_num_vsd]])
    text_embeddings_phi = torch.cat([uncond_embeddings[:args.particle_num_phi], text_embeddings[:args.particle_num_phi]])
    ### init particles
    if args.rgb_as_latents:
        particles = torch.randn((args.batch_size, unet.config.in_channels, args.height // 8, args.width // 8))      # This branch

    particles = particles.to(device, dtype=dtype)


    #################################################################################
    #                           optimizer & lr schedule                             #
    #################################################################################
    
    ### weight loss     Important !!!
    loss_weights = get_loss_weights(scheduler.betas, args)


    ### optimizer
    # For a tensor, we can optimize the tensor directly
    particles.requires_grad = True
    particles_to_optimize = [particles]

    total_parameters = sum(p.numel() for p in particles_to_optimize if p.requires_grad)
    print(f'Total number of trainable parameters in particles: {total_parameters}; number of particles: {args.batch_size}')
    ### Initialize optimizer & scheduler
    if args.generation_mode == 'vsd':
        if args.phi_model in ['lora', 'unet_simple']:
            phi_optimizer = torch.optim.AdamW([{"params": phi_params, "lr": args.phi_lr}], lr=args.phi_lr)
            print(f'number of trainable parameters of phi model in optimizer: {sum(p.numel() for p in phi_params if p.requires_grad)}')
    optimizer = get_optimizer(particles_to_optimize, args)
    ### lr_scheduler
    if args.use_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, \
            start_factor=args.lr_scheduler_start_factor, total_iters=args.lr_scheduler_iters)

    #################################################################################
    #################################################################################
    #                             Main optimization loop                            #
    #################################################################################
    #################################################################################
    log_steps = []
    train_loss_values = []
    ave_train_loss_values = []
    if args.log_progress:
        image_progress = []
    first_iteration = True
    logger.info("################# Metrics: ####################")
    ######## t schedule #########
    chosen_ts = get_t_schedule(num_train_timesteps, args, loss_weights)
    pbar = tqdm(chosen_ts)
    

    #################################################################################
    #                                 MODE: SDS | VSD                               #
    #################################################################################
    ### sds text to image generation
    if args.generation_mode in ['sds', 'vsd']:
        cross_attention_kwargs = {'scale': args.lora_scale} if (args.generation_mode == 'vsd' and args.phi_model == 'lora') else {}
        for step, chosen_t in enumerate(pbar):
            # get latent of all particles
            latents = get_latents(particles, vae, args.rgb_as_latents, use_mlp_particle=args.use_mlp_particle)
            t = torch.tensor([chosen_t]).to(device)

            ######## q sample #########
            # random sample particle_num_vsd particles from latents
            indices = torch.randperm(latents.size(0))
            latents_sd = latents[indices[:args.particle_num_vsd]]      # Be Careful here, for multi VSD particles, particle_num_vsd > 1
            noise = torch.randn_like(latents_sd)
            noisy_latents = scheduler.add_noise(latents_sd, noise, t)

            ######## Do the gradient for latents!!! #########
            optimizer.zero_grad()


            # Predict the noise
            grad_, noise_pred, noise_pred_phi = sds_vsd_grad_diffuser(unet, noisy_latents, noise, text_embeddings_vsd, t, \
                                                    guidance_scale=args.guidance_scale, unet_phi=unet_phi, \
                                                        generation_mode=args.generation_mode, phi_model=args.phi_model, \
                                                            cross_attention_kwargs=cross_attention_kwargs, \
                                                                multisteps=args.multisteps, scheduler=scheduler, lora_v=args.lora_vprediction, \
                                                                    half_inference=args.half_inference, \
                                                                        cfg_phi=args.cfg_phi, grad_scale=args.grad_scale)

            ##################################### TODO #1: Code starts here #########################################
            # First, finish TODO in utils/sds_vsd_utils.py
            # With grad_, implement the rest of Formula (16) to obtain the gradient of L_SDS (In VSD, this is L_VSD)

            # Multiply grad_ with loss_weights at the timestep t (Hint: use breakpoint to check loss_weights data structure)
            grad_ = None

            ############################################ End Your Code Here ##############################################


            # Now, what you get is only a Gradient. We need to convert to loss.
            # Formula: d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
            target = (latents_sd - grad_).detach()
            loss = 0.5 * F.mse_loss(latents_sd, target, reduction="mean") / args.batch_size        # batch_size should be 1


            loss.backward()
            optimizer.step()
            if args.use_scheduler:
                lr_scheduler.step(loss)


            ######################################## Do the gradient for unet_phi (LoRA)!!! #########################################
            torch.cuda.empty_cache()
            if args.generation_mode == 'vsd':
                ## update the unet (phi) model (LoRA)
                for _ in range(args.phi_update_step):
                    phi_optimizer.zero_grad()
                    if args.use_t_phi:
                        # different t for phi finetuning
                        # t_phi = np.random.choice(chosen_ts, 1, replace=True)[0]
                        t_phi = np.random.choice(list(range(num_train_timesteps)), 1, replace=True)[0]
                        t_phi = torch.tensor([t_phi]).to(device)
                    else:
                        t_phi = t
                    # random sample particle_num_phi particles from latents
                    indices = torch.randperm(latents.size(0))
                    latents_phi = latents[indices[:args.particle_num_phi]]
                    noise_phi = torch.randn_like(latents_phi)
                    noisy_latents_phi = scheduler.add_noise(latents_phi, noise_phi, t_phi)
                    loss_phi = phi_vsd_grad_diffuser(unet_phi, noisy_latents_phi.detach(), noise_phi, text_embeddings_phi, t_phi, \
                                                     cross_attention_kwargs=cross_attention_kwargs, scheduler=scheduler, \
                                                        lora_v=args.lora_vprediction, half_inference=args.half_inference)
                    loss_phi.backward()
                    phi_optimizer.step()
            ################################################################################################################################

            ### Store loss and step
            train_loss_values.append(loss.item())
            ### update pbar
            pbar.set_description(f'Loss: {loss.item():.6f}, sampled t : {t.item()}')

            optimizer.zero_grad()
            ######## Evaluation and log metric #########
            if args.log_steps and (step % args.log_steps == 0 or step == (args.num_steps-1)):
                log_steps.append(step)
                # save current img_tensor
                # scale and decode the image latents with vae
                tmp_latents = 1 / vae.config.scaling_factor * latents_sd.clone().detach()
                if args.save_x0:
                    # Note: In SDS, noise - noise_pred_phi will be zero; In VSD, it may be a little bit different
                    if args.generation_mode == 'sds':
                        pred_latents = scheduler.step(noise_pred, t, noisy_latents).pred_original_sample.to(dtype).clone().detach()
                    if args.generation_mode == 'vsd':
                        pred_latents = scheduler.step(noise_pred - noise_pred_phi + noise, t, noisy_latents).pred_original_sample.to(dtype).clone().detach()
                        pred_latents_phi = scheduler.step(noise_pred_phi, t, noisy_latents).pred_original_sample.to(dtype).clone().detach()
                

                with torch.no_grad():
                    image_ = vae.decode(tmp_latents).sample.to(torch.float32)
                    if args.save_x0:
                        image_x0 = vae.decode(pred_latents / vae.config.scaling_factor).sample.to(torch.float32)
                        if args.generation_mode == 'vsd':
                            image_x0_phi = vae_phi.decode(pred_latents_phi / vae.config.scaling_factor).sample.to(torch.float32)
                            image = image_ #torch.cat((image_, image_x0, image_x0_phi), dim=2)
                        else:
                            image =  image_ # torch.cat((image_, image_x0), dim=2) # Save Decoede VAE and 
                    else:
                        image = image_

                if args.log_progress:
                    image_progress.append((image/2+0.5).clamp(0, 1))
                save_image((image/2+0.5).clamp(0, 1), f'{args.store_dir}/{image_name}_image_step{step}_t{t.item()}.png')
                ave_train_loss_value = np.average(train_loss_values)
                ave_train_loss_values.append(ave_train_loss_value) if step > 0 else None
                logger.info(f'step: {step}; average loss: {ave_train_loss_value}')
                update_curve(train_loss_values, 'Train_loss', 'steps', 'Loss', args.store_dir, args.run_id)

            if first_iteration and device==torch.device('cuda'):
                global_free, total_gpu = torch.cuda.mem_get_info(0)
                logger.info(f'global free and total GPU memory: {round(global_free/1024**3,6)} GB, {round(total_gpu/1024**3,6)} GB')
                first_iteration = False


    #################################################################################
    #                                   save results                                #
    #################################################################################
    if args.log_gif:
        # make gif
        images = sorted(Path(args.store_dir).glob(f"*{image_name}*.png"))
        images = [imageio.imread(image) for image in images]
        imageio.mimsave(f'{args.store_dir}/{image_name}.gif', images, duration=0.3)
    if args.log_progress and args.batch_size == 1:
        concatenated_images = torch.cat(image_progress, dim=0)
        save_image(concatenated_images, f'{args.store_dir}/generated_progressive.png')
    # save final image
    if args.generation_mode == 't2i':
        image = image_
    else:
        image = get_images(particles, vae, args.rgb_as_latents, use_mlp_particle=args.use_mlp_particle)
    save_image((image/2+0.5).clamp(0, 1), f'{args.store_dir}/generated_image.png')

    if args.generation_mode in ['vsd'] and args.save_phi_model:
        if args.phi_model in ['lora']:
            unet_phi.save_attn_procs(save_directory=f'{args.store_dir}')
        elif args.phi_model in ['unet_simple']:
            unet_phi.save_pretrained(save_directory=f'{args.store_dir}')

#########################################################################################
if __name__ == "__main__":

    main()


