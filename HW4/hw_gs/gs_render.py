import contextlib
import math
import pdb

import torch
import torch.nn as nn

from utils import eval_sh


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def homogeneous(points):
    """
    Converting points into homogenuous cooridnates.

    Args:
        - points: [..., 3]

    Return:
        - homogenuous coordinates: [..., 4]
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def build_rotation(r):
    """
    Building rotation matrix (R) given the quaternion.

    Args:
        - r: [B, 4], quaternion vector.

    Returns:
        - R: [B, 3, 3], rotation matrix.
    """
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    #############################################################################
    #                                   TODO: Task 1 A                          #
    #############################################################################
    # Get the rotation matrix from the quaternion
    # Here, we already help you initialize the Rotation Matrix to be (3x3) all zero matrix
    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return R


def build_scaling_rotation(scaling_vector, quaternion_vector):
    """
    Calculate the matrix L = R * S, where R is the rotation matrix and S is the scaling matrix.

    Args:
        - scaling_vector: [B, 3], scaling vector.
        - quaternion_vector: [B, 4], quaternion vector.

    Returns:
        - L: [B, 3, 3], the scaling-rotation matrix.
    """
    S = torch.zeros(
        (scaling_vector.shape[0], 3, 3), dtype=torch.float, device="cuda"
    )  # s.shape[0] is B (Batch size)
    R = build_rotation(quaternion_vector)

    #############################################################################
    #                                   TODO: Task 1 B                          #
    #############################################################################
    # Get the scaling matrix from the scaling vector
    # Hint: Check Formula 3 in the isntruction pdf

    # S = ...
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    L = R @ S  # L = R * S
    return L


def build_covariance_3d(s, r):
    """
    Build the 3D covariance matrix given the scaling and rotation vectors.

    Args:
        - s: [B, 3], scaling vector.
        - r: [B, 4], quaternion vector.

    Returns:
        - actual_covariance: [B, 3, 3], the 3D covariance
    """
    L = build_scaling_rotation(s, r)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance


def build_covariance_2d(mean3d, cov3d, viewmatrix, fov_x, fov_y, focal_x, focal_y):
    """
    Build the 2D covariance matrix given the 3D covariance matrix, gaussian centers (mean3d), and view matrix.

    Args:
        - mean3d: [B, 3], the mean (xyz location) of the gaussian.
        - cov3d: [B, 3, 3], the 3D covariance matrix.
        - viewmatrix: [4, 4], the view matrix.
        - fov_x: float, the field of view in x direction.
        - fov_y: float, the field of view in y direction.
        - focal_x: float, the focal length in x direction.
        - focal_y: float, the focal length in y direction.

    Returns:
        - cov2d: [B, 2, 2], the 2D covariance matrix.
    """
    # The following models the steps outlined by equations 29
    # and 31 in "EWA Splatting" (Zwicker et al., 2002).
    # Additionally considers aspect / scaling of viewport.
    # Transposes used to account for row-/column-major conventions.
    tan_fovx = math.tan(fov_x * 0.5)
    tan_fovy = math.tan(fov_y * 0.5)
    t = (mean3d @ viewmatrix[:3, :3]) + viewmatrix[-1:, :3]

    # truncate the influences of gaussians far outside the frustum.
    tx = (t[..., 0] / t[..., 2]).clip(min=-tan_fovx * 1.3, max=tan_fovx * 1.3) * t[
        ..., 2
    ]
    ty = (t[..., 1] / t[..., 2]).clip(min=-tan_fovy * 1.3, max=tan_fovy * 1.3) * t[
        ..., 2
    ]
    tz = t[..., 2]

    # Eq.29 locally affine transform
    # perspective transform is not affine so we approximate with first-order taylor expansion
    # notice that we multiply by the intrinsic so that the variance is at the sceen space
    J = torch.zeros(mean3d.shape[0], 3, 3).to(mean3d)
    J[..., 0, 0] = 1 / tz * focal_x
    J[..., 0, 2] = -tx / (tz * tz) * focal_x
    J[..., 1, 1] = 1 / tz * focal_y
    J[..., 1, 2] = -ty / (tz * tz) * focal_y
    W = viewmatrix[:3, :3].T  # transpose to correct viewmatrix

    #############################################################################
    #                                   TODO: Task 2                           #
    #############################################################################
    # Calculate the 2D covariance matrix cov2d
    # Hint: Check Args explaination for cov3d; For clean code of matrix multiplication, consider using @

    # cov2d = ...
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    # add low pass filter here according to E.q. 32
    filter = torch.eye(2, 2).to(cov2d) * 0.3
    return cov2d[:, :2, :2] + filter[None]


def projection_ndc(points, viewmatrix, projmatrix):
    points_o = homogeneous(points)  # object space
    points_h = points_o @ viewmatrix @ projmatrix  # screen space # RHS
    p_w = 1.0 / (points_h[..., -1:] + 0.000001)
    p_proj = points_h * p_w
    p_view = points_o @ viewmatrix
    in_mask = p_view[..., 2] >= 0.2
    return p_proj, p_view, in_mask


@torch.no_grad()
def get_radius(cov2d):
    """
    This function get the radius of an eclipse given the covariance matrix.
    It first calculate the eigenvalues of the covariance matrix, and then choose the larger one.
    Args:
        - cov2d: [B, 2, 2], covariance matrix.
    Returns:
        - radius: [B], radius of the eclipse.
    """
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
    mid = 0.5 * (cov2d[:, 0, 0] + cov2d[:, 1, 1])
    lambda1 = mid + torch.sqrt((mid**2 - det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2 - det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()


@torch.no_grad()
def get_rect(pix_coord, radii, width, height):
    """
    This function gets the rectangle that covers the ellipse.

    Args:
        - pix_coord: [B, 2], pixel coordinates of the ellipse.
        - radii: [B], radius of the ellipse.
        - width: int, width of the image.
        - height: int, height of the image.

    Returns:
        - rect_min: [B, 2], the minimum coordinates of the rectangle.
        - rect_max: [B, 2], the maximum coordinates of the rectangle
    """
    rect_min = pix_coord - radii[:, None]
    rect_max = pix_coord + radii[:, None]
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max


class GaussRenderer(nn.Module):
    """
    The guassian splatting renderer.
    """

    def __init__(self, H, W, active_sh_degree=3, white_bkgd=True, **kwargs):
        super().__init__()
        self.active_sh_degree = active_sh_degree
        self.debug = False
        self.white_bkgd = white_bkgd
        self.pix_coord = torch.stack(
            torch.meshgrid(torch.arange(H), torch.arange(W), indexing="xy"), dim=-1
        ).to("cuda")

    def build_color(self, means3D, shs, camera):
        rays_o = camera.camera_center
        rays_d = means3D - rays_o
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        color = eval_sh(self.active_sh_degree, shs.permute(0, 2, 1), rays_d)
        color = (color + 0.5).clip(min=0.0, max=1.0)
        return color

    def render(self, camera, means2D, cov2d, color, opacity, depths):
        # get the radius of ellipse
        radii = get_radius(cov2d)

        # get the rectangle coordinates that covers the ellipse
        rect = get_rect(
            means2D, radii, width=camera.image_width, height=camera.image_height
        )

        self.render_color = torch.ones(*self.pix_coord.shape[:2], 3).to("cuda")
        self.render_depth = torch.zeros(*self.pix_coord.shape[:2], 1).to("cuda")
        self.render_alpha = torch.zeros(*self.pix_coord.shape[:2], 1).to("cuda")

        TILE_SIZE = 16
        for h in range(0, camera.image_height, TILE_SIZE):
            for w in range(0, camera.image_width, TILE_SIZE):
                #############################################################################
                #                                   TODO: Task 3                           #
                #############################################################################
                # check if the 2D gaussian intersects with the tile 
                # To do so, we need to check if the rectangle of the 2D gaussian (rect) intersects with the tile

                # in_mask = .....
                #############################################################################
                #                             END OF YOUR CODE                              #
                #############################################################################

                if not in_mask.sum() > 0:
                    continue

                P = in_mask.sum()  # noqa F841
                tile_coord = self.pix_coord[
                    h : h + TILE_SIZE, w : w + TILE_SIZE
                ].flatten(0, -2)
                sorted_depths, index = torch.sort(depths[in_mask])
                sorted_means2D = means2D[in_mask][index]
                sorted_cov2d = cov2d[in_mask][index]  # P 2 2
                sorted_inverse_conv = sorted_cov2d.inverse()  # inverse of variance
                sorted_opacity = opacity[in_mask][index]
                sorted_color = color[in_mask][index]
                dx = tile_coord[:, None, :] - sorted_means2D[None, :]  # B P 2

                #############################################################################
                #                                   TODO: Task 4                           #
                #############################################################################
                #  In this block, you are expcted to implemente the splatting process in one tile.
                # You are expected to output `acc_alpha`, `tile_color`, and `tile_depth` which are the accumulated alpha, color, and depth in the tile.
                # To achieve it, you may want to follow the following logic:
                # Step 1: calculate the gaussian weights given `dx` and `sorted_inverse_conv`, which are the weights of each gaussian in the tile applying on each pixel. It follows the normal gaussian distribution formula.
                # It should have the shape [B, P], where B is the number of pixels in the tile, and P is the number of gaussians in the tile.
                # Step 2: calculate alpha. alpha = (gauss_weight[..., None] * sorted_opacity[None]).clip(max=0.99)
                # Step 3: calculate the accumulated alpha, color and depth.

                # gauss_weight = ... # Hint: Check step 1 in the instruction pdf
                # alpha = ... # Hint: Check step 2 in the instruction pdf
                # T = ... # Hint: Check Eq. (6) in the instruction pdf

                # acc_alpha =  ... # Hint: Check Eq. (8) in the instruction pdf
                # tile_color = ... # Hint: Check Eq. (5) in the instruction pdf
                # tile_depth = ... # Hint: Check Eq. (7) in the instruction pdf
                #############################################################################
                #                             END OF YOUR CODE                              #
                #############################################################################
                self.render_color[h : h + TILE_SIZE, w : w + TILE_SIZE] = (
                    tile_color.reshape(TILE_SIZE, TILE_SIZE, -1)
                )
                self.render_depth[h : h + TILE_SIZE, w : w + TILE_SIZE] = (
                    tile_depth.reshape(TILE_SIZE, TILE_SIZE, -1)
                )
                self.render_alpha[h : h + TILE_SIZE, w : w + TILE_SIZE] = (
                    acc_alpha.reshape(TILE_SIZE, TILE_SIZE, -1)
                )

        return {
            "render": self.render_color,
            "depth": self.render_depth,
            "alpha": self.render_alpha,
            "visiility_filter": radii > 0,
            "radii": radii,
        }

    def forward(self, camera, pc, **kwargs):
        means3D = pc.get_xyz
        opacity = pc.get_opacity
        scales = pc.get_scaling
        rotations = pc.get_rotation
        shs = pc.get_features

        prof = contextlib.nullcontext

        with prof("projection"):
            mean_ndc, mean_view, in_mask = projection_ndc(
                means3D,
                viewmatrix=camera.world_view_transform,
                projmatrix=camera.projection_matrix,
            )
            mean_ndc = mean_ndc[in_mask]
            mean_view = mean_view[in_mask]
            depths = mean_view[:, 2]

            means3D = means3D[in_mask]
            scales = scales[in_mask]
            rotations = rotations[in_mask]
            shs = shs[in_mask]
            opacity = opacity[in_mask]

        with prof("build color"):
            color = self.build_color(means3D=means3D, shs=shs, camera=camera)
        with prof("build cov3d"):
            cov3d = build_covariance_3d(scales, rotations)

        with prof("build cov2d"):
            cov2d = build_covariance_2d(
                mean3d=means3D,
                cov3d=cov3d,
                viewmatrix=camera.world_view_transform,
                fov_x=camera.FoVx,
                fov_y=camera.FoVy,
                focal_x=camera.focal_x,
                focal_y=camera.focal_y,
            )

            mean_coord_x = ((mean_ndc[..., 0] + 1) * camera.image_width - 1.0) * 0.5
            mean_coord_y = ((mean_ndc[..., 1] + 1) * camera.image_height - 1.0) * 0.5
            means2D = torch.stack([mean_coord_x, mean_coord_y], dim=-1)
        with prof("render"):
            rets = self.render(
                camera=camera,
                means2D=means2D,
                cov2d=cov2d,
                color=color,
                opacity=opacity,
                depths=depths,
            )

        return rets
