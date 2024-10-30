import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Union

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn


def parse_camera(params):
    H = params[:, 0]
    W = params[:, 1]
    intrinsics = params[:, 2:18].reshape((-1, 4, 4))
    c2w = params[:, 18:34].reshape((-1, 4, 4))
    return H, W, intrinsics, c2w


def to_viewpoint_camera(camera):
    """
    Parse a camera of intrinsic and c2w into a Camera Object
    """
    Hs, Ws, intrinsics, c2ws = parse_camera(camera.unsqueeze(0))
    camera = Camera(
        width=int(Ws[0]), height=int(Hs[0]), intrinsic=intrinsics[0], c2w=c2ws[0]
    )
    return camera


class Camera(nn.Module):
    def __init__(
        self,
        width,
        height,
        intrinsic,
        c2w,
        znear=0.1,
        zfar=100.0,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
    ):
        super().__init__()
        device = c2w.device
        self.znear = znear
        self.zfar = zfar
        self.focal_x, self.focal_y = intrinsic[0, 0], intrinsic[1, 1]
        self.FoVx = focal2fov(self.focal_x, width)
        self.FoVy = focal2fov(self.focal_y, height)
        self.image_width = int(width)
        self.image_height = int(height)
        self.world_view_transform = torch.linalg.inv(c2w).permute(1, 0)
        self.intrinsic = intrinsic
        self.c2w = c2w
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .to(device)
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


@dataclass
class Intrin:
    fx: Union[float, torch.Tensor]
    fy: Union[float, torch.Tensor]
    cx: Union[float, torch.Tensor]
    cy: Union[float, torch.Tensor]

    def scale(self, scaling: float):
        return Intrin(
            self.fx * scaling, self.fy * scaling, self.cx * scaling, self.cy * scaling
        )

    def get(self, field: str, image_id: int = 0):
        val = self.__dict__[field]
        return val if isinstance(val, float) else val[image_id].item()


def load_data(
    root: str,
    split_name: str,
    scale: float = 0.5,
    scene_scale: float = 1.0,
    n_images_interval: int = 2,
):
    '''
        This function is to load dataset
        Args:
            root (str):
            split_name (str): 
            scale (float):
            scene_scale (float):
            n_images_interval (int):
        Returns:
            all_c2w (): Camera to World transform
            all_gt (): GT images
            intrinsic (): Intrinsic matrix
    '''

    data_path = os.path.join(root, split_name)
    data_json = os.path.join(root, "transforms_" + split_name + ".json")
    print("LOAD DATA", data_path)

    j = json.load(open(data_json))

    # OpenGL -> OpenCV
    cam_trans = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.float32))

    all_c2w = []
    all_gt = []
    for frame in j["frames"]:
        fpath = os.path.join(data_path, os.path.basename(frame["file_path"]) + ".png")
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        c2w = c2w @ cam_trans  # To OpenCV

        im_gt = imageio.imread(fpath)
        if scale < 1.0:
            full_size = list(im_gt.shape[:2])
            rsz_h, rsz_w = (round(hw * scale) for hw in full_size)
            im_gt = cv2.resize(im_gt, (rsz_w, rsz_h), interpolation=cv2.INTER_AREA)

        all_c2w.append(c2w)
        all_gt.append(torch.from_numpy(im_gt))
    focal = float(0.5 * all_gt[0].shape[1] / np.tan(0.5 * j["camera_angle_x"]))
    all_c2w = torch.stack(all_c2w)
    all_c2w[:, :3, 3] *= scene_scale

    all_gt = torch.stack(all_gt).float() / 255.0
    if all_gt.size(-1) == 4:
        # Apply alpha channel
        all_gt = all_gt[..., :3] * all_gt[..., 3:] + (1.0 - all_gt[..., 3:])

    n_images, h_full, w_full, _ = all_gt.shape

    # Choose a subset of training images
    all_gt = all_gt[0::n_images_interval, ...]
    all_c2w = all_c2w[0::n_images_interval, ...]
    intrinsic = np.zeros((3, 3))
    intrinsic[0, 0] = focal
    intrinsic[1, 1] = focal
    intrinsic[2, 2] = 1.0

    return all_c2w, all_gt, intrinsic
