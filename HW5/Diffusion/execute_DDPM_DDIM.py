import os, shutil, sys
import time
import argparse

# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from pipeline.pipeline_ddpm import DDPMPipeline
from pipeline.pipeline_ddim import DDIMPipeline


if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', "--scheduler_name", required=True, choices=["DDPM", "DDIM"])    
    args = parser.parse_args()

    # Setting
    model_id = "google/ddpm-celebahq-256"       # Human face Unconditional Pre-trained model weight
    num_images = 10
    scheduler_name = args.scheduler_name # DDIM or DDPM


    # Prepare the folder
    store_folder_name = scheduler_name + "_results"
    if os.path.exists(store_folder_name):
        shutil.rmtree(store_folder_name)
    os.makedirs(store_folder_name)


    # load model and scheduler
    if scheduler_name == "DDPM":
        model = DDPMPipeline.from_pretrained(model_id).to("cuda")   # Use .to("cuda") for GPU; else, this will be run in CPU and is very slow!
    elif scheduler_name == "DDIM":
        model = DDIMPipeline.from_pretrained(model_id).to("cuda")  


    # Generate num_images images by unconditional network
    start_time = time.time()
    for idx in range(num_images):
        # Execute the model with none of any condition as the input, aka. Unconditional Diffusion Model
        image = model()     # Note: the default inference step is 1000 for DDPM and 50 for DDIM (No need to input new value inside)

        # save image
        image[0][0].save(os.path.join(store_folder_name, scheduler_name+"_generated_image"+str(idx)+".png"))

    end_time = time.time()
    total_time = end_time - start_time
    print("Total time spent is ", total_time, " second.")