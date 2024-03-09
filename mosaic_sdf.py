import torch
import torch.nn as nn
import numpy as np

from einops import rearrange

from shape_sampler import ShapeSampler

class MosaicSDF(nn.Module):
    def __init__(self, grid_resolution=7, n_grids=1024):
        """
        Initialize the MosaicSDF representation.
        
        :param shape_sampler: Shape Sampler facade.
        :param n_grids: Number of local grids.
        :param grid_resolution: Resolution of each grid (assumed cubic for simplicity).
        """
        super(MosaicSDF, self).__init__()
        
        # self.shape_sampler = shape_sampler

        self.n_grids = n_grids
        # Assuming volume_centers, scales, and sdf_values are learnable parameters
        self.volume_centers = nn.Parameter(torch.rand((self.n_grids, 3)) * 2 - 1)  # Initialize randomly within [-1, 1]
        
        self.k = grid_resolution

        min_rand_scale, max_rand_scale = .01, 1.

        self.scales = nn.Parameter(torch.rand((n_grids,)) * (max_rand_scale - min_rand_scale) + min_rand_scale)

        # self.mosaic_sdf_values = torch.randn(n_grids, grid_resolution, grid_resolution, grid_resolution)
        init_mosaic_sdf_values = torch.randn(n_grids, grid_resolution, grid_resolution, grid_resolution)
        self.register_buffer('mosaic_sdf_values', init_mosaic_sdf_values)


    def update_sdf_values(self, shape_sampler: ShapeSampler):
        self.mosaic_sdf_values = self._compute_local_sdf(shape_sampler)


    def forward(self, points):
        """
        Compute the SDF values at given points using the Mosaic-SDF representation.
        
        :param points: Tensor of points where SDF values are to be computed (N, 3).
        :return: SDF values at the provided points.
        """
        

        points_sdf = self._compute_point_sdf(points)
        
        return points_sdf

    
    def _compute_local_sdf(self, shape_sampler: ShapeSampler):
                
        in_grid_offsets = torch.linspace(-.5, .5, self.k)

        x, y, z = torch.meshgrid(in_grid_offsets, in_grid_offsets, in_grid_offsets, indexing='ij')

        grid_offsets = torch.stack([x, y, z], dim=-1).reshape((-1, 3)).to(self.scales.device)

        scaled_grid_offsets = self.scales[:, None, None] * grid_offsets[None, ...]

        grid_points = self.volume_centers[:, None, :] + scaled_grid_offsets

        batched_grid_points = rearrange(grid_points, 'n k3 d -> (n k3) d', d=3) 
        sdf_values = shape_sampler(batched_grid_points)#[:, None] 
        sdf_values = rearrange(sdf_values, '(n k1 k2 k3) -> n k1 k2 k3', n=self.n_grids, k1=self.k, k2=self.k, k3=self.k)
        # sdf_values = rearrange(sdf_values, '(n k1 k2 k3) d-> n k1 k2 k3 d', n=self.n_grids, k1=self.k, k2=self.k, k3=self.k, d=1)

        return sdf_values


    def _compute_trilinear_interpolation_weights(self, relative_positions):
        
        # Step 1: Create a tensor with each cell's value being its relative position within the tensor
        rel_positions = torch.linspace(0, 1, steps=self.k, device=relative_positions.device)
        grid_coords = torch.stack(
            torch.meshgrid(rel_positions, rel_positions, rel_positions, indexing='ij'), 
            dim=-1)

        # Step 2: Expand dims for broadcasting
        grid_coords_expanded = grid_coords[None, None, ...]  # Shape: (1, 1, K, K, K, 3)
        rel_pos_expanded = relative_positions[:, :, None, None, None, :] # Shape: (b, n, 1, 1, 1, 3)
        #.unsqueeze(2).unsqueeze(3).unsqueeze(4)  

        # Step 3: Compute relative offsets
        in_grid_rel_offsets = rel_pos_expanded - grid_coords_expanded  # Shape: (b, n, K, K, K, 3)

        # Step 4: Compute distances from offsets
        in_grid_distances = torch.linalg.norm(in_grid_rel_offsets, dim=-1)  # Shape: (b, n, K, K, K)

        # Step 5: Zero-out distances greater than 1, 
        in_grid_weights = torch.where(in_grid_distances <= 1, in_grid_distances, torch.zeros_like(in_grid_distances))
        # might replace above with use of clamp / saturate, pseudo:
        # distances_mask = 1 - torch.floor(torch.clamp(distances, 0, 1))
        # weights = distances * distances_mask

        # Step 6: Normalize weights
        in_grid_weights_sum = in_grid_weights.sum(dim=(2,3,4)) 
        in_grid_weights_normalized = in_grid_weights / in_grid_weights_sum[:, :, None, None, None]

        # set zeros where have nan because divided by zero
        # in_grid_weights_normalized[in_grid_weights_normalized != in_grid_weights_normalized] = 0
        in_grid_weights_normalized = torch.nan_to_num(in_grid_weights_normalized, nan=0.0)

        return in_grid_weights_normalized


    def _compute_point_sdf(self, points):
        
        points_expanded_to_grids = points[:, None, :]
        grids_expanded_to_points = self.volume_centers[None, ...]
        scales_expanded_to_points = self.scales[None, ..., None]

        grid_relative_positions = points_expanded_to_grids - grids_expanded_to_points
        grid_scaled_relative_positions = grid_relative_positions / scales_expanded_to_points

        interpolation_weights = self._compute_trilinear_interpolation_weights(grid_scaled_relative_positions)
        
        interpolation_values = self.mosaic_sdf_values[None, ...] * interpolation_weights
        interpolation_values = interpolation_values.sum(axis=(2,3,4))
        # Calculate each grid weight
        grid_scaled_relative_dist = torch.linalg.norm(grid_scaled_relative_positions, axis=-1)
        
        # w_i_hat
        grid_weight = torch.relu(1 - grid_scaled_relative_dist)
        grid_weight = grid_weight / grid_weight.sum(axis=-1, keepdim=True)
        grid_weight = torch.nan_to_num(grid_weight, nan=0.0)
        
        point_sdf = torch.sum(interpolation_values * grid_weight, axis=-1)

        return point_sdf