import numpy as np
import torch
import torch.nn as nn

from gs_render import build_scaling_rotation, inverse_sigmoid, strip_symmetric
from utils import RGB2SH


class GaussModel(nn.Module):
    """
    A Gaussian Model containing the optimizable parameters of gaussians.

    Attributes
        - _xyz: [N, 3], locations of gaussians' centers.
        - _feature_dc: [N, 1, 3], DC term (RGB colors) of features for SH coefficients.
        _feature_rest: [N, K, 3], rest features for SH coefficients.
        - _rotatoin: [N, 4], rotation of gaussians in quaternion representation.
        - _scaling: [N, 3], scaling of gaussians.
        - _opacity: [N, 1], opacity of gaussian
    """

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int = 3, debug=False):
        super().__init__()
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.setup_functions()
        self.debug = debug

    def create_from_pcd(self, pcd_xyz, pcd_color):
        """
        create the guassian model from a color point cloud
        """
        points = pcd_xyz
        colors = pcd_color

        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(colors)).float().cuda())

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = (
            torch.ones_like(torch.from_numpy(np.asarray(points[..., 0]))).float().cuda()
            * 0.015
        )
        scales = torch.log(dist2)[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        if self.debug:
            # easy for visualization
            colors = np.zeros_like(colors)
            opacities = inverse_sigmoid(
                0.9
                * torch.ones(
                    (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
                )
            )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        return self

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def save_ply(self, path):
        from plyfile import PlyData, PlyElement

        # import os
        # os.makedirs(os.path.dirname(path), exist_ok=True)
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]  # noqa: E741
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append(f"f_dc_{i}")
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append(f"f_rest_{i}")
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append(f"scale_{i}")
        for i in range(self._rotation.shape[1]):
            l.append(f"rot_{i}")
        return l
