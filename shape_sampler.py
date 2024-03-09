from abc import abstractmethod
import torch
import torch.nn as nn
import numpy as np

from einops import rearrange

from trimesh.caching import tracked_array
from pysdf import SDF
from pytorch3d.io import load_obj


class ShapeSampler(nn.Module):
    def __init__(self, vertices, faces):       
        super(ShapeSampler, self).__init__()

        self.vertices = vertices
        self.faces = faces

        tv = tracked_array(vertices.numpy())
        tf = tracked_array(faces.verts_idx.numpy())

        self.sdf_fun = SDF(tv, tf)

        self.noise_scale = 0#noise_scale


    @abstractmethod
    def from_file(file_path, device='cpu'):
        vertices, faces, aux = load_obj(file_path, device=device)
        return ShapeSampler(vertices, faces)
    
        
    def forward(self, points):   
        # add check if points are not tensor, then bypassing numpy() conversion
        np_points = points.detach().cpu().numpy()
        return torch.tensor(self.sdf_fun(np_points)).to(points.device)
    
    

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