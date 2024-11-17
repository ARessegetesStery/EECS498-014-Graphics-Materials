import os, sys, shutil
from datasets import load_dataset


if __name__ == "__main__":
    hugginface_name = "huggan/few-shot-dog"   # huggan/few-shot-obama
    store_parent_path = "many_shot_dog"        # few_show_obama
    if os.path.exists(store_parent_path):
        shutil.rmtree(store_parent_path)
    os.makedirs(store_parent_path)

    train_dataset = load_dataset(hugginface_name, split="train")
    for idx, img in enumerate(train_dataset): 
        if idx == 1000: # Only consider 100 images
            break
        
        store_path = os.path.join(store_parent_path, str(idx) + ".png")
        img['image'].save(store_path)


    print("Finished!")

