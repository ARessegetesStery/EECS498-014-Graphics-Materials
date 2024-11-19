import os, shutil, sys
import time
import cv2
import torch
import argparse


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from pipeline.pipeline_ddpm import DDPMPipeline
from pipeline.pipeline_ddim import DDIMPipeline
from utils.DPS_utils import InpaintingOperator, mask_generator, get_noise, get_dataset, get_dataloader


if __name__ == "__main__":


    # Setting
    model_id = "google/ddpm-celebahq-256"       # Human face Unconditional Pre-trained model weight

    # Prepare the folder
    store_path = "DPS_results"
    if os.path.exists(store_path):
        shutil.rmtree(store_path)
    os.makedirs(store_path)


    # Init the pipeline, We use DDIM by default
    pipeline = DDIMPipeline.from_pretrained(model_id).to("cuda")  


    # Init Inpainting operator
    operator = InpaintingOperator("cuda")
    mask_gen = mask_generator(mask_type = "random", mask_prob_range = [0.3, 0.7], image_size = 256) # Mask generator
    noiser = get_noise(name = "gaussian", sigma = 0.05)


    # Prepare the data loader 
    dataset = get_dataset(name="ffhq", root="__asset__/ffhq_fixed_samples/")
    data_loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)



    # Generate many images
    start_time = time.time()
    for idx, (ref_img, unnormalized_img) in enumerate(data_loader):

        ref_img = ref_img.to("cuda")
        unnormalized_img = unnormalized_img.to("cuda")

        # Set the mask for the pair dataset between 
        mask = mask_gen(ref_img)
        mask = mask[:, 0, :, :].unsqueeze(dim=0)


        # Prepare the measurement which is the A(x) + n
        y = operator.forward(ref_img, mask)
        y_unnormalized = operator.forward(unnormalized_img, mask)   # For the unnormalized pair dataset purpose
        y_n = noiser(y)     


        # Save the masked unnormalized image (for visualiza on the pair dataset)
        cv2.imwrite(os.path.join(store_path, "DPS_masked_input_image"+str(idx)+".png"), cv2.cvtColor(torch.permute(y_unnormalized[0], (1, 2, 0)).detach().cpu().numpy()*255, cv2.COLOR_BGR2RGB))


        # Execute the inference of the model
        # TODO: Change your num_inference_steps here
        image = pipeline(num_inference_steps = 1000, use_DPS = True, operator = operator, mask = mask, measurement = y_n)     # Note: the default inference step is 1000 for DDPM and 50 for DDIM (No need to input new value inside)


        # Save image
        image[0][0].save(os.path.join(store_path, "DPS_generated_image" + str(idx) + ".png"))
        

    end_time = time.time()
    total_time = end_time - start_time
    print("Total time spent is ", total_time, " second.")