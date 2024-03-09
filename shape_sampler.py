import torch
import torch.nn as nn
import numpy as np

from einops import rearrange


class ShapeSampler(nn.Module):
    def __init__(self, shape_type):       
        super(ShapeSampler, self).__init__()

        self.shape_type = shape_type
        self.noise_scale = 0#noise_scale

        
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

    ### FPS below

    # def farthest_point_sampling(self, points, n_samples):
    #     """
    #     Perform farthest point sampling on a set of points.
    #     :param points: A tensor of points (N, 3).
    #     :param n_samples: The number of points to sample.
    #     :return: A tensor of sampled points (n_samples, 3).
    #     """
    #     farthest_pts = points[torch.randint(len(points), (1,))]
    #     distances = torch.norm(points - farthest_pts[0], dim=1)
    #     for _ in range(1, n_samples):
    #         farthest_pts = torch.cat((farthest_pts, points[torch.argmax(distances, dim=0, keepdim=True)]), dim=0)
    #         distances = torch.min(distances, torch.norm(points - farthest_pts[-1], dim=1))
    #     return farthest_pts

    # def add_noise_to_positions(self, positions):
    #     """
    #     Add noise to grid positions to simulate non-ideal initial placements.
    #     :param positions: A tensor of positions (N, 3).
    #     :return: Positions with added noise.
    #     """
    #     noise = torch.randn_like(positions) * self.noise_scale
    #     return positions + noise

    # # Example usage within ShapeSampler
    # def initialize_grids_with_noise(self, n_grids, device='cpu'):
    #     # This is a simplified example that assumes access to boundary points
    #     boundary_points = self.get_boundary_points(shape_type=self.shape_type, n_points=1000, device=device)
    #     sampled_points = self.farthest_point_sampling(boundary_points, n_grids)
    #     noisy_positions = self.add_noise_to_positions(sampled_points)
    #     return noisy_positions

    # def get_boundary_points(self, shape_type, n_points, device):
    #     # Placeholder method to simulate boundary points for 'sphere' and 'cube'
    #     if shape_type == "sphere":
    #         # Simulate sphere boundary points
    #         points = torch.randn(n_points, 3, device=device)
    #         points = points / torch.norm(points, dim=1, keepdim=True)
    #     elif shape_type == "cube":
    #         # Simulate cube boundary points
    #         points = torch.rand(n_points, 3, device=device) * 2 - 1
    #     else:
    #         raise ValueError("Unknown shape type.")
    #     return points