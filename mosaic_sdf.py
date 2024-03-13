import torch
import torch.nn as nn
import numpy as np

from einops import rearrange

from utils import min_l2_distance
from shape_sampler import ShapeSampler

class MosaicSDF(nn.Module):
    def __init__(self, grid_resolution=7, n_grids=1024, 
                 volume_centers=None, volume_scales=None, mosaic_scale_multiplier=1,
                 device='cuda'):
        """
        Initialize the MosaicSDF representation.
        
        :param shape_sampler: Shape Sampler facade.
        :param n_grids: Number of local grids.
        :param grid_resolution: Resolution of each grid (assumed cubic for simplicity).
        """
        super(MosaicSDF, self).__init__()
        self.eps = 1e-4

        # self.shape_sampler = shape_sampler
        self.out_of_reach_const = 1

        self.n_grids = n_grids
        self.k = grid_resolution

        
        if volume_centers is None:
            volume_centers = torch.rand((self.n_grids, 3)) * 2 - 1
        # Assuming volume_centers, scales, and sdf_values are learnable parameters
        self.volume_centers = nn.Parameter(volume_centers)  
        
        if volume_scales is None:
            # min_rand_scale, max_rand_scale = .15, .25
            # volume_scales = torch.rand((n_grids,)) * (max_rand_scale - min_rand_scale) + min_rand_scale
            
            mean_min_l2_dist = min_l2_distance(volume_centers.to(device)).mean()
            volume_scales = torch.ones((n_grids,), device=device) * mean_min_l2_dist

        self.scales = nn.Parameter(volume_scales * mosaic_scale_multiplier)
        
        init_mosaic_sdf_values = torch.randn(n_grids, grid_resolution, grid_resolution, grid_resolution)
        self.register_buffer('mosaic_sdf_values', init_mosaic_sdf_values)


    def forward(self, points):
        """
        Compute the SDF values at given points using the Mosaic-SDF representation.
        
        :param points: Tensor of points where SDF values are to be computed (N, 3).
        :return: SDF values at the provided points.
        """

        points_sdf = self._compute_point_sdf(points)
        
        return points_sdf


    def _compute_trilinear_interpolation_weights(self, point_relative_positions):
        
        # Step 1: Create a tensor with each cell's value being its relative position within the tensor
        coord_span = torch.linspace(-1, 1, steps=self.k, device=point_relative_positions.device)
        coord_step = 2 / (self.k - 1)
        grid_coords = torch.stack(
            torch.meshgrid(coord_span, coord_span, coord_span, indexing='ij'), 
            dim=-1)

        # Step 2: Expand dims for broadcasting
        grid_coords_expanded = grid_coords[None, None, ...]  # Shape: (1, 1, K, K, K, 3)
        rel_pos_expanded = point_relative_positions[:, :, None, None, None, :] # Shape: (b, n, 1, 1, 1, 3)
        #.unsqueeze(2).unsqueeze(3).unsqueeze(4)  

        # Step 3: Compute relative offsets
        in_grid_rel_offsets = rel_pos_expanded - grid_coords_expanded  # Shape: (b, n, K, K, K, 3)
        in_grid_rel_offsets = in_grid_rel_offsets / coord_step
        # print('in_grid_rel_offsets')
        # print(in_grid_rel_offsets.shape)
        # # print(in_grid_rel_offsets[:,:,1:3, 1:3, 1:3, :])
        # print(in_grid_rel_offsets)
        # Step 4: Compute distances from offsets
        in_grid_distances = torch.linalg.norm(in_grid_rel_offsets, dim=-1, ord=2)  # Shape: (b, n, K, K, K)
        # in_grid_distances, _ = in_grid_rel_offsets.abs().max(dim=-1)
        # print(f'in_grid_distances:\n{in_grid_distances}')
        
        # Step 5: Zero-out distances greater than 1, 
        in_grid_weights = torch.where(in_grid_distances<= 1.0, 
                                      1 - in_grid_distances, 
                                      torch.zeros_like(in_grid_distances))
        # print(f'in_grid_weights: {in_grid_weights}')
        # might replace above with use of clamp / saturate, pseudo:
        # distances_mask = 1 - torch.floor(torch.clamp(distances, 0, 1))
        # weights = distances * distances_mask

        # Step 6: Normalize weights
        in_grid_weights_sum = in_grid_weights.sum(dim=(2,3,4))[:, :, None, None, None]
        # in_grid_weights_normalized = in_grid_weights / in_grid_weights_sum

        # # set zeros where have nan because divided by zero
        # # in_grid_weights_normalized[in_grid_weights_normalized != in_grid_weights_normalized] = 0
        # in_grid_weights_normalized = torch.nan_to_num(in_grid_weights_normalized, nan=0.0)

        in_grid_weights_normalized = torch.where(
            in_grid_weights_sum > self.eps, 
            in_grid_weights / in_grid_weights_sum, torch.zeros_like(in_grid_weights))
        
        # print(f'in_grid_weights_normalized: {in_grid_weights_normalized}')
        
        return in_grid_weights_normalized


    def _compute_point_sdf(self, points):
        
        points_expanded_to_grids = points[:, None, :]
        points_expanded_to_grids = points_expanded_to_grids#.detach()
        grids_expanded_to_points = self.volume_centers[None, ...]
        scales_expanded_to_points = self.scales[None, ..., None]

        grid_relative_pos_to_point = (points_expanded_to_grids - grids_expanded_to_points) / scales_expanded_to_points
        # print('grid_relative_pos_to_point')
        # print(grid_relative_pos_to_point)
        interpolation_weights = self._compute_trilinear_interpolation_weights(grid_relative_pos_to_point)
        # print(f'interpolation_weights: {interpolation_weights}')
        # print('interpolation_weights')
        # print(interpolation_weights.shape)
        # print(interpolation_weights)
        
        interpolation_values = self.mosaic_sdf_values[None, ...] * interpolation_weights
        interpolation_values = interpolation_values.sum(axis=(2,3,4))
        debug_interpolation_weights = interpolation_weights.sum(axis=(1,2,3,4))[:,None]
        # print('debug_interpolation_weights')
        # print(debug_interpolation_weights.shape)
        # Calculate each grid weight
        
        grid_normalized_relative_dist = torch.linalg.norm(
            grid_relative_pos_to_point, dim=-1, ord=2)
        # grid_normalized_relative_dist, _ = grid_relative_pos_to_point.abs().max(dim=-1)
        # print(f'grid_normalized_relative_dist: {grid_normalized_relative_dist}')
        # print('grid_normalized_relative_dist')
        # print(grid_normalized_relative_dist.shape)
        # print(grid_normalized_relative_dist)
        # grid_normalized_relative_dist = torch.linalg.norm(grid_normalized_relative_positions, axis=-1)
        
        # w_i_hat
        grid_weight = torch.relu(1 - grid_normalized_relative_dist + self.eps)
        
        sum_of_weights = grid_weight.sum(axis=-1, keepdim=True)
        
        # TODO kernel crashes if I not detach normalized_grid_weight, maybe because of bug in autograd
        # normalized_grid_weight = torch.nan_to_num(
        #     (grid_weight / sum_of_weights).detach()
        #     , nan=0.0)
        normalized_grid_weight = torch.where(
            sum_of_weights > self.eps, 
            grid_weight / sum_of_weights, torch.zeros_like(grid_weight))

        # print('norm sum:', normalized_grid_weight)
        # Use a mask to identify areas with very low weight sums (not covered by grids)
        # uncovered_mask = sum_of_weights < self.eps
        # DEBUG
        # print('sum_of_weights.shape')
        # print(sum_of_weights.shape)
        # print('debug_interpolation_weights.shape')
        # print(debug_interpolation_weights.shape)
        # print('uncovered_mask.shape')
        # print(uncovered_mask.shape)
        
        # print('0:', uncovered_mask.view(-1).shape)
        # print('mask:', uncovered_mask)
        # For uncovered areas, use out_of_reach_const; otherwise, compute as before
        
        # print('a:', points.shape[:1])
        
        # point_sdf_ = torch.nan_to_num(
        #     torch.sum(interpolation_values * normalized_grid_weight, axis=-1),
        #     nan=self.out_of_reach_const)
        # print('b:', point_sdf_.shape)

        if True:
            # uncovered_mask = (sum_of_weights < self.eps) | (debug_interpolation_weights < self.eps)
            uncovered_mask = sum_of_weights < self.eps

            point_sdf = torch.where(
                uncovered_mask.view(-1),
                torch.full(points.shape[:1], self.out_of_reach_const, device=points.device),
                torch.sum(interpolation_values * normalized_grid_weight, axis=-1)
            )
        else:
            point_sdf = torch.sum(interpolation_values * normalized_grid_weight, axis=-1)
        # print('c:', point_sdf.shape)
        # print('d:', torch.sum(interpolation_values * normalized_grid_weight, axis=-1).shape)
        # print('e:', torch.full(points.shape[:1], self.out_of_reach_const, device=points.device).shape)

        return point_sdf
    

    # Update SDF values
    def update_sdf_values(self, shape_sampler: ShapeSampler):
        self.mosaic_sdf_values = self._compute_local_sdf(shape_sampler)

    
    def _compute_local_sdf(self, shape_sampler: ShapeSampler):
                
        in_grid_offsets = torch.linspace(-1, 1, self.k)

        x, y, z = torch.meshgrid(in_grid_offsets, in_grid_offsets, in_grid_offsets, indexing='ij')

        grid_offsets = (
            torch.stack([x, y, z], dim=-1)
                .reshape((-1, 3))
                .to(self.scales.device)
        )
        
        scaled_grid_offsets = self.scales[:, None, None] * grid_offsets[None, ...]

        grid_points = self.volume_centers[:, None, :] + scaled_grid_offsets

        batched_grid_points = rearrange(grid_points, 'n k3 d -> (n k3) d', d=3) 

        sdf_values = shape_sampler(batched_grid_points)

        sdf_values = rearrange(sdf_values, '(n k1 k2 k3) -> n k1 k2 k3', n=self.n_grids, k1=self.k, k2=self.k, k3=self.k)
        
        return sdf_values
    
