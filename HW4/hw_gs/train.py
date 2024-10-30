import argparse
import os

import imageio
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from data_loader import Camera, load_data
from gs_model import GaussModel
from gs_render import GaussRenderer
from loss_util import l1_loss, ssim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--single_image_fitting",
        action="store_true",
        help="whether to fit a single image for debugging",
    )
    parser.add_argument(
        "--num_points", type=int, default=2**13, help="number of points to fit"
    )
    parser.add_argument(
        "--loss_lambda", type=float, default=0.2, help="weight for ssim loss"
    )
    args = parser.parse_args()

    all_c2w, all_gt, intrinsic = load_data(
        root="./chair", split_name="train", scale=0.18, scene_scale=1.0
    )

    # randomly initialize points
    points = np.random.randn(args.num_points, 3) * 3
    points_color = np.random.rand(points.shape[0], 3)

    if os.path.exists("result") is False:
        os.makedirs("result")

    if args.single_image_fitting:
        saving_path = "result/single_image_fitting"
    else:
        saving_path = "result/multi_image_fitting"

    if os.path.exists(saving_path) is False:
        os.makedirs(saving_path)

    gaussModel = GaussModel(sh_degree=3, debug=False)
    gaussModel.create_from_pcd(points, points_color)

    if args.single_image_fitting:
        total_step = 500 + 1
    else:
        total_step = 2000 + 1

    optimizer = optim.Adam(gaussModel.parameters(), lr=1e-2)

    gaussRender = GaussRenderer(H=144, W=144, active_sh_degree=3, white_bkgd=True)
    lr = 1e-2

    for i in tqdm(range(total_step)):
        if args.single_image_fitting:
            camera_idx = 0
        else:
            camera_idx = torch.randint(all_gt.shape[0], (1,))
        gt_img = all_gt[camera_idx].cuda().squeeze(0)
        c2w = all_c2w[camera_idx].cuda().squeeze(0)

        camera = Camera(
            144,
            144,
            intrinsic,
            c2w,
            znear=0.1,
            zfar=100.0,
            trans=np.array([0.0, 0.0, 0.0]),
            scale=1.0,
        )
        out = gaussRender(pc=gaussModel, camera=camera)

        rendered_image = out["render"]

        loss = (1.0 - args.loss_lambda) * l1_loss(
            rendered_image, gt_img
        ) + args.loss_lambda * (1.0 - ssim(rendered_image, gt_img))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            imageio.imwrite(
                os.path.join(saving_path, f"test_{i}.png"),
                np.uint8(255 * rendered_image.detach().cpu().numpy()),
            )
            tqdm.write(f"Step {i}, Loss    : {loss.item()}")

        if i % 500 == 0 and i != 0:
            lr /= 2
            optimizer = optim.Adam(gaussModel.parameters(), lr=lr)

    if not args.single_image_fitting:
        torch.save(
            gaussModel.state_dict(), os.path.join(saving_path, "trained_model.pth")
        )

        # create evaluation folder
        saving_path = "result/multi_image_fitting_test_views"
        if os.path.exists(saving_path) is False:
            os.makedirs(saving_path)
        # do evaluation
        all_c2w, all_gt, intrinsic = load_data(
            root="./chair",
            split_name="test",
            scale=0.18,
            scene_scale=1.0,
            n_images_interval=10,
        )
        for i in tqdm(range(all_c2w.shape[0])):
            gt_img = all_gt[i].cuda().squeeze(0)
            c2w = all_c2w[i].cuda().squeeze(0)

            camera = Camera(
                144,
                144,
                intrinsic,
                c2w,
                znear=0.1,
                zfar=100.0,
                trans=np.array([0.0, 0.0, 0.0]),
                scale=1.0,
            )
            out = gaussRender(pc=gaussModel, camera=camera)
            rendered_image = out["render"]
            imageio.imwrite(
                os.path.join(saving_path, f"test_{i}.png"),
                np.uint8(255 * rendered_image.detach().cpu().numpy()),
            )


if __name__ == "__main__":
    main()
