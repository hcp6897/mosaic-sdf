from abc import abstractmethod
import torch
import torch.nn as nn
import numpy as np

from trimesh.caching import tracked_array
from pysdf import SDF
import point_cloud_utils as pcu
from utils import to_tensor, to_numpy


# TODO test point_mesh_distance from PT3D
# TODO this class should be split into one that provides SDF and another that does operations on meshes
class ShapeSampler(nn.Module):

    def __init__(self, vertices, verts_idx, normalize_shape=True, sdf_func=None, sdf_value_scaler=1):       
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
        
        self.noise_scale = 0    # noise_scale


    def forward(self, points):   
        # add check if points are not tensor, then bypassing numpy() conversion
        np_points = to_numpy(points)

        return torch.tensor(self.sdf_func(np_points) * self.sdf_value_scaler).to(points.device) 


    def sample_n_random_points(self, n_points, rand_offset=None, random_seed=42):
        """
        Random sample points.
        """
        f_i, bc = pcu.sample_mesh_random(
            self.np_vertices, 
            self.np_verts_idx, 
            num_samples=n_points, 
            random_seed=random_seed
        )

        # Use the face indices and barycentric coordinate to compute sample positions and normals
        v_sampled = pcu.interpolate_barycentric_coords(self.np_verts_idx, f_i, bc, self.np_vertices)
        v_sampled = to_tensor(v_sampled, device=self.vertices.device)
        
        if rand_offset is not None:
            v_sampled += (torch.rand_like(v_sampled) - .5) * rand_offset * 2
        
        return v_sampled


    @abstractmethod
    def from_file(file_path, device='cpu', make_watertight=True, wt_resolution=10_000, **kwargs):
        vertices, verts_idx = pcu.load_mesh_vf(file_path)

        if make_watertight:
            vertices, verts_idx = pcu.make_mesh_watertight(vertices, verts_idx, resolution=wt_resolution)
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

