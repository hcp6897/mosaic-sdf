import torch
import torch.nn as nn
import numpy as np

from einops import rearrange


class ShapeSampler(nn.Module):
    def __init__(self, shape_type):       
        super(ShapeSampler, self).__init__()

        self.shape_type = shape_type
        
    def forward(self, points):
        """
        This is a stub function that simulates sampling the SDF for different shapes.
        :param shape_type: A string indicating the type of shape ('sphere', 'cube', 'pyramid').
        :param points: A tensor of points in space (N, 3) we want to sample the SDF for.
        :return: A tensor representing the SDF values at the provided points.
        """
        if self.shape_type == "sphere":
            return self.sample_sphere(points)
        elif self.shape_type == "cube":
            return self.sample_cube(points)
        # elif shape_type == "pyramid":
        #     return self.sample_pyramid(points)
        else:
            raise ValueError("Unknown shape type.")
        
    def sample_sphere(self, points, radius=1.0):
        """
        Simulate SDF sampling for a sphere centered at the origin.
        :param points: Points at which to sample the SDF (N, 3).
        :param radius: Radius of the sphere.
        :return: SDF values at the provided points.
        """
        return torch.norm(points, dim=1) - radius

    def sample_cube(self, points, side_length=2.0):
        """
        Simulate SDF sampling for a cube centered at the origin.
        :param points: Points at which to sample the SDF (N, 3).
        :param side_length: Side length of the cube.
        :return: SDF values at the provided points.
        """
        half_side = side_length / 2
        max_dist = torch.max(torch.abs(points) - half_side, dim=1)[0]
        return max_dist
    
    # def sample_pyramid(self, points, height=2.0, base=2.0):
    #     """
    #     Simulate SDF sampling for a pyramid centered at the origin.
    #     :param points: Points at which to sample the SDF (N, 3).
    #     :param height: Height of the pyramid.
    #     :param base: Base length of the pyramid.
    #     :return: SDF values at the provided points.
    #     """
    #     # This is a simplified stub for the pyramid SDF.
    #     # A true SDF for a pyramid would involve more complex calculations.
    #     x_dist = torch.abs(points[:, 0]) - base/2
    #     y_dist = torch.abs(points[:, 1]) - base/2
    #     z_dist = points[:, 2] - height/2
    #     return torch.max(torch.max(x_dist, y_dist), z_dist)

