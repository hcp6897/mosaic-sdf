from abc import abstractmethod
import torch
import torch.nn as nn
import numpy as np

from einops import rearrange

from trimesh.caching import tracked_array
from pysdf import SDF
from pytorch3d.io import load_obj
import point_cloud_utils as pcu
from utils import to_tensor, to_numpy

# TODO test point_mesh_distance from PT3D
# TODO this class should be split into one that provides SDF and another that does operations on meshes
class ShapeSampler(nn.Module):
    def __init__(self, vertices, verts_idx, normalize_shape=True, 
                 sdf_func=None, sdf_value_scaler=1):       
        super(ShapeSampler, self).__init__()
            
        self.vertices = vertices
        self.verts_idx = verts_idx

        if normalize_shape:
            self.vertices, self.norm_center_offset, self.norm_max_extent = ShapeSampler.normalize_vertices(self.vertices)

        self.np_vertices = to_numpy(self.vertices)
        self.np_verts_idx = to_numpy(self.verts_idx)

        self.sdf_func = sdf_func
        self.sdf_value_scaler = sdf_value_scaler
        
        if self.sdf_func is None:
            tv = tracked_array(self.vertices.cpu().numpy())
            tf = tracked_array(self.verts_idx.cpu().numpy())
            self.sdf_value_scaler = -1
            self.sdf_func = SDF(tv, tf)
        

        self.noise_scale = 0 #noise_scale


    def forward(self, points):   
        # print('ShapeSampler.forward')
        # add check if points are not tensor, then bypassing numpy() conversion
        np_points = to_numpy(points)
        # return torch.tensor(self.sdf_fun(np_points)).to(points.device) 
        return torch.tensor(self.sdf_func(np_points) * self.sdf_value_scaler).to(points.device) 


    def sample_n_random_points(self, n_points, rand_offset=None, random_seed=42):
        
        # print('3 ->', vertices.shape, verts_idx.shape)

        f_i, bc = pcu.sample_mesh_random(self.np_vertices, 
                                         self.np_verts_idx, 
                                         num_samples=n_points, 
                                         random_seed=random_seed)
        # print(f_i)
        # print(bc)
        # print('4 ->', f_i.shape, bc.shape)

        # Use the face indices and barycentric coordinate to compute sample positions and normals
        v_sampled = pcu.interpolate_barycentric_coords(self.np_verts_idx, f_i, bc, self.np_vertices)
        v_sampled = to_tensor(v_sampled, device=self.vertices.device)
        
        if rand_offset is not None:
            v_sampled += (torch.rand_like(v_sampled) - .5) * rand_offset * 2
        
        return v_sampled


    @abstractmethod
    def from_file(file_path, device='cpu', make_watertight=True, wt_resolution=10_000, **kwargs):
        # vertices, faces, aux = load_obj(file_path, device=device)
        vertices, verts_idx = pcu.load_mesh_vf(file_path)
        # print('1 ->', vertices.shape, verts_idx.shape)

        if make_watertight:
            vertices, verts_idx = pcu.make_mesh_watertight(vertices, verts_idx, resolution=wt_resolution)
        # print('2 ->', vertices.shape, verts_idx.shape)
        return ShapeSampler(to_tensor(vertices, device), 
                            to_tensor(verts_idx, device),
                            **kwargs)
    
    
    @abstractmethod
    def scale_offset_mesh(mesh, offset, scale):
        scaled_verts = mesh.verts_list()[0] * scale
        # Translate the mesh
        translated_verts = scaled_verts + offset
        # Create a new mesh with the transformed vertices and the same faces
        new_mesh = mesh.clone()
        new_mesh = new_mesh.update_padded(new_verts_padded=translated_verts.unsqueeze(0))
        return new_mesh


    @abstractmethod
    def normalize_vertices(vertices, center_offset=None, max_extent=None):
        # Further adjust to ensure the shape is centered at the origin
        if center_offset is None:
            center_offset = vertices.mean(dim=0)
        vertices -= center_offset

        # Calculate the scale factor as the max extent in any dimension
        if max_extent is None:
            max_extent = torch.abs(vertices).max()
        # Normalize vertices to fit within the [-1, 1] range
        normalized_vertices = vertices / max_extent
        
        return normalized_vertices, center_offset, max_extent


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




    # def compute_sdf_gradient(self, points, delta=1e-4):
    #     """
    #     Approximate the gradient of the SDF at given points using central differences.
        
    #     Args:
    #     - points: Tensor of shape (N, 3) representing N points in 3D space.
    #     - delta: A small offset used for finite differences.
        
    #     Returns:
    #     - grad: Tensor of shape (N, 3) representing the approximate gradient of the SDF at each point.
    #     """
    #     device = points.device
    #     N, D = points.shape
    #     grad = torch.zeros_like(points, requires_grad=False, device=device)
        
    #     for i in range(D):
    #         # Create a basis vector for the i-th dimension
    #         offset = torch.zeros(D, device=device)
    #         offset[i] = delta
            
    #         # Compute SDF at slightly offset points
    #         sdf_plus = self.forward(points + offset)
    #         sdf_minus = self.forward(points - offset)
            
    #         # Approximate the derivative using central differences
    #         grad[:, i] = (sdf_plus - sdf_minus) / (2 * delta)
        
    #     return grad