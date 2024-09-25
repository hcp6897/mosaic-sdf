import unittest
import torch
import torch.nn as nn
import numpy as np

from mosaicSDF.mosaic_sdf import MosaicSDF

class TestMosaicSDF(unittest.TestCase):
    
    def setUp(self):
        # Example setup for a test case
        self.grid_resolution = 3
        self.n_grids = 4  # Reduced for simplicity
        self.mosaic_sdf = MosaicSDF(grid_resolution=self.grid_resolution, n_grids=self.n_grids)
        # Customize initialization for testing
        # self.mosaic_sdf.volume_centers = nn.Parameter(torch.tensor([[0., 0., 0.], [1., 1., 1.], [-1., -1., -1.], [2., 2., 2.]]))
        volume_centers = torch.tensor([
            [0.5, 0.5, 0.5], 
            # [-0.5, -0.5, -0.5], 
            # [0.5, -0.5, 0.5], 
            # [-0.5, 0.5, -0.5]
            ])
        self.mosaic_sdf.volume_centers = nn.Parameter(volume_centers)

        scales = torch.ones((volume_centers.shape[0])) * .5
        self.mosaic_sdf.scales = nn.Parameter(scales)
        
        init_mosaic_sdf_values = torch.zeros((volume_centers.shape[0], self.grid_resolution, self.grid_resolution, self.grid_resolution))
        init_mosaic_sdf_values[0] = torch.tensor([
            []
        ])
        # init_mosaic_sdf_values[1] += 2  # Second grid, SDF values of 2, and so on
        # init_mosaic_sdf_values[2] += 3
        # init_mosaic_sdf_values[3] += 4
        self.mosaic_sdf.register_buffer('mosaic_sdf_values', init_mosaic_sdf_values)

        
    def test_forward(self):
        # Test the forward function with a known input and check the output
        input_points = torch.tensor([
            # [0., 0., 0.], 
            # [1., 1., 1.]
            [0.5, 0.5, 0.5], 
            [-0.5, -0.5, -0.5], 
            ])
        expected_sdf_values = torch.tensor(
            [1, 0], 
            dtype=torch.float32)  # Fill in based on expected logic
        actual_sdf_values = self.mosaic_sdf(input_points)
        self.assertTrue(
            torch.allclose(expected_sdf_values, actual_sdf_values, atol=1e-6), 
            actual_sdf_values)

# Add more test methods as needed

if __name__ == '__main__':
    unittest.main()

