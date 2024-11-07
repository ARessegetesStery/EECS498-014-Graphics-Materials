import torch
import numpy as np
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_rays(H, W, focal, pose):
    """
    This function generates camera rays for each pixel in an image. It calculates the origin and direction of rays
    based on the camera's intrinsic parameters (focal length) and extrinsic parameters (pose).
    The rays are generated in world coordinates, which is crucial for the NeRF rendering process.

    Parameters:
    H (int): Height of the image in pixels.
    W (int): Width of the image in pixels.
    focal (float): Focal length of the camera.
    pose (torch.Tensor): Camera pose matrix of size 4x4.

    Returns:
    tuple: A tuple containing two elements:
        rays_o (torch.Tensor): Origins of the rays in world coordinates.
        rays_d (torch.Tensor): Directions of the rays in world coordinates.
    """
    # Create a meshgrid of image coordinates (i, j) for each pixel in the image.
    u, v = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32)
    )
    u = u.t()
    v = v.t()

    #############################################################################
    #                                   TODO: Task 1                            #
    ############################################################################# 
    # Step 1: Calculate the direction vectors for each ray originating from the camera center in the camera coordinate.
    # We assume the camera looks towards -z. 
    # The y direction and H direction in figure 2 is opposite, so it should also be multiplied with -1
    #  Note: the focal length here is in the unit of pixels. The coordinates are normalized with respect to the focal length (which means that should be divided by focal).
    # Plus, Cast to the device (cuda)
    
    # dirs = 
    
    # Step 2: Transform the direction vectors (dirs) from camera coordinates to world coordinates.
    # The provided pose is camera-to-world matrix. Please note the expected rays_d should have the shape of [N, 3], where N is the number of rays.

    # rays_d = 

    # Step 3: Normalize the direction vectors on the last dimension (only 3 channels) to ensure they have unit length (Hint: check torch.norm and look at keepdim parameter of it).

    # rays_d = ...
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    # The ray origins (rays_o) are set to the camera position, given by the translation part (last column) of the pose matrix.
    # The position is expanded to match the shape of rays_d for broadcasting.
    rays_o = pose[:3, -1].expand(rays_d.shape)

    # Return the origins and directions of the rays.
    return rays_o, rays_d



def positional_encoder(p, L_embed=6):
    """
    This function applies positional encoding to the input tensor. Positional encoding is used in NeRF
    to allow the model to learn high-frequency details more effectively. It applies sinusoidal functions
    at different frequencies to the input.

    Parameters:
    p (torch.Tensor): The input tensor to be positionally encoded.
    L_embed (int): The number of frequency levels to use in the encoding. Defaults to 6.

    Returns:
    torch.Tensor: The positionally encoded tensor.
    """

    # Initialize a list with the input tensor.
    rets = [p]

    # Loop over the number of frequency levels.
    for i in range(L_embed):
        #############################################################################
        #                                   TODO: Task 2                            #
        #############################################################################
        # hint: following the order of (sin, cos), like for fn in [torch.sin, torch.cos]:

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################


    # Concatenate the original and encoded features along the last dimension.
    return torch.cat(rets, -1)



def cumprod_exclusive(tensor: torch.Tensor):
    """
    Compute the exclusive cumulative product of a tensor along its last dimension.
    'Exclusive' means that the cumulative product at each element does not include the element itself.
    This function is used in volume rendering to compute the product of probabilities
    along a ray, excluding the current sample point.

    Parameters:
    tensor (torch.Tensor): The input tensor for which to calculate the exclusive cumulative product.

    Returns:
    torch.Tensor: The tensor after applying the exclusive cumulative product.

    Example: 
    Input: tensor = torch.tensor([[2, 3, 4, 5]]),
    Output: tensor = torch.tensor([[1, 2, 2*3, 3*4]])

    Hint: Check torch.cumprod function. 
    """
    #############################################################################
    #                                   TODO: Task 3                            #
    #############################################################################
    # Compute the cumulative product along the last dimension of the tensor. Hint: Check torch.cumprod
    
    # cumprod = ...
    
    # Roll the elements along the last dimension by one position. Hint: Check torch.roll
    # This shifts the cumulative products to make them exclusive.

    # cumprod = ...

    # Set the first element of the last dimension to 1, as the exclusive product of the first element is always 1.
    
    # Your code here
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return cumprod


class VeryTinyNerfModel(torch.nn.Module):
    """
    A very small implementation of a Neural Radiance Field (NeRF) model. This model is a simplified
    version of the standard NeRF architecture, it consists of a simple feedforward neural network with three linear layers.

    Parameters:
    filter_size (int): The number of neurons in the hidden layers. Default is 128.
    num_encoding_functions (int): The number of sinusoidal encoding functions. Default is 6.
    """

    def __init__(self, filter_size=128, num_encoding_functions=6):
        super(VeryTinyNerfModel, self).__init__()
        self.layer1 = torch.nn.Linear(3 + 3 * 2 * num_encoding_functions, filter_size)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        self.layer3 = torch.nn.Linear(filter_size, 4)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


def render(model, rays_o, rays_d, near, far, n_samples, rand=False):
    """
    Render a scene using a Neural Radiance Field (NeRF) model. This function samples points along rays,
    evaluates the NeRF model at these points, and applies volume rendering techniques to produce an image.

    Parameters:
    model (torch.nn.Module): The NeRF model to be used for rendering.
    rays_o (torch.Tensor): Origins of the rays.
    rays_d (torch.Tensor): Directions of the rays.
    near (float): Near bound for depth sampling along the rays.
    far (float): Far bound for depth sampling along the rays.
    n_samples (int): Number of samples to take along each ray.
    rand (bool): If True, randomize sample depths. Default is False.

    Returns:
    tuple: A tuple containing the RGB map and depth map of the rendered scene.
    """

    # Sample points along each ray, from 'near' to 'far'.
    t = torch.linspace(near, far, n_samples).to(device)
    if rand:
        mids = 0.5 * (t[..., 1:] + t[..., :-1])
        upper = torch.cat([mids, t[..., -1:]], -1)
        lower = torch.cat([t[..., :1], mids], -1)
        t_rand = torch.rand(t.shape).to(device)
        t = lower + (upper - lower) * t_rand

    #############################################################################
    #                                   TODO: Task 4 A                          #
    #############################################################################
    # Compute 3D coordinates of the sampled points along the rays.

    # points = ...
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    # Flatten the points and apply positional encoding.
    flat_points = torch.reshape(points, [-1, points.shape[-1]])
    flat_points = positional_encoder(flat_points)

    # Evaluate the model on the encoded points in chunks to manage memory usage.
    chunk = 1024 * 32
    raw = torch.cat([model(flat_points[i:i + chunk]) for i in range(0, flat_points.shape[0], chunk)], 0)
    raw = torch.reshape(raw, list(points.shape[:-1]) + [4])

    # Compute densities (sigmas) and RGB values from the model's output.
    sigma = F.relu(raw[..., 3])
    rgb = torch.sigmoid(raw[..., :3])

    #############################################################################
    #                                   TODO: Task 4 B                          #
    #############################################################################
    # Tries to solve the calculation from Matrix perspective instead of the For Loop
    # Perform volume rendering to obtain the weights of each point.

    # one_e_10 = ...
    # dists = ...
    # alpha = ...
    # weights = ...
    
    # Compute the weighted sum of RGB values along each ray to get the final pixel color.

    # rgb_map = ...

    # Compute the depth map as the weighted sum of sampled depths.

    # depth_map = ...
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return rgb_map, depth_map