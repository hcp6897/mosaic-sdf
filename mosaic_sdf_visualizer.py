from abc import abstractmethod
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    BlendParams,
    Materials,
    AmbientLights
)
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.renderer.mesh.textures import Textures
from pytorch3d.structures.meshes import join_meshes_as_scene

from mosaic_sdf import MosaicSDF
from shape_sampler import ShapeSampler

from pytorch3d.io import load_obj
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes
import numpy as np
from utils import to_numpy, to_tensor

class MosaicSDFVisualizer:
    def __init__(self, mosaic_sdf: MosaicSDF, shape_sampler: ShapeSampler, device, template_mesh_path:str):
        self.mosaic_sdf = mosaic_sdf
        self.shape_sampler = shape_sampler
        self.device = device #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        R, T = look_at_view_transform(5, 45, 30)
        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        
        # self.lights = PointLights(device=self.device, location=[
        #     [0.0, 0.0, -3.0],
        #     # [0.0, 0.0, 3.0],
        #     # [-3.0, 0.0, 0],
        #     # [3.0, 0.0, 0],
        #     ])
        self.lights = AmbientLights(device=self.device, 
                                    ambient_color=((1.0, 1.0, 1.0),),
                                    # ambient_intensity=(0.5,)
                                    )

        
        self.raster_settings = RasterizationSettings(
            image_size=512, 
            blur_radius=0.0, 
            faces_per_pixel=1,
        )
        
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device, 
                cameras=self.cameras,
                lights=self.lights
            )
        )
        
        self.load_meshes_to_show(template_mesh_path)

    
    def load_meshes_to_show(self, template_mesh_path):
        template_vertices, template_faces, template_aux = load_obj(template_mesh_path,  device=self.device)
        
        template_vertices, _, _ = ShapeSampler.normalize_vertices(template_vertices)
        

        self.template_mesh = MosaicSDFVisualizer.create_mesh_from_verts(template_vertices, 
                                                    template_faces, vert_colors = [.8, 0, 0],
                                                  device=self.device)

        self.boundary_mesh = MosaicSDFVisualizer.create_mesh_from_verts(template_vertices, 
                                                    template_faces, vert_colors = [0, 0, .5],
                                                  device=self.device)

        self.shape_target_mesh = MosaicSDFVisualizer.create_mesh_from_verts(self.shape_sampler.vertices, 
                                                  self.shape_sampler.faces, 
                                                  vert_colors = [0, .3, 0],
                                                  device=self.device)
    

    @abstractmethod
    def create_mesh_from_verts(vertices, faces, vert_colors = [.6, 0, 0], device='cpu'):
        total_verts = vertices.shape[0]
        verts_rgb = torch.ones((1, total_verts, 3), device=device)  # White color for all vertices
        verts_rgb *= torch.tensor(vert_colors, device=device)
        # Initialize the textures with the corrected verts_rgb
        textures = Textures(verts_rgb=verts_rgb)

        # Create the mesh
        return Meshes(verts=[vertices.to(device)], 
                      faces=[faces.verts_idx.to(device)], 
                      textures=textures).to(device)


    def create_state_meshes(self, show_mosaic_grids=True, 
                            show_target_mesh=True,
                            show_boundary_mesh=True,
                            show_rasterized_sdf_mesh=True
                            ):
        """
        Visualizes the MosaicSDF grids as semi-transparent cubes.
        """
        volume_centers = self.mosaic_sdf.volume_centers
        scales = self.mosaic_sdf.scales
                
        # materials = Materials(
        #     device=self.device,
        #     specular_color=[[0.0, 1.0, 0.0]],
        #     shininess=10.0
        # )        

        all_meshes = []
        if show_target_mesh:
            all_meshes.append(self.shape_target_mesh)


        if show_rasterized_sdf_mesh:
            all_meshes.append(MosaicSDFVisualizer.rasterize_sdf(self.mosaic_sdf, device=self.device))


        if show_mosaic_grids:
            for center, scale in zip(volume_centers, scales):
                new_mesh = ShapeSampler.scale_offset_mesh(self.template_mesh, center, scale)
                all_meshes.append(new_mesh)
        
        if show_boundary_mesh:
            all_meshes.append(self.boundary_mesh)
            
        combined_mesh = join_meshes_as_scene(all_meshes)

        return combined_mesh


    def render_meshes(self, meshes):
        mesh_alpha = .2
        blend_params = BlendParams(sigma=1e-4, gamma=mesh_alpha, background_color=(0, 0, 0))

        images = self.renderer(meshes_world=meshes, cameras=self.cameras,
                            #    materials=materials
                               blend_params=blend_params

                               )
        
        return images


    def plot_meshes(self):
                
        with torch.no_grad():
            meshes = self.create_state_meshes()
            vis = self.render_meshes(meshes)
            # vis.shape
            plt.figure(figsize=(4, 4))
            plt.imshow(vis[0, ..., :3].cpu().numpy())
            plt.axis("off")



    def rasterize_sdf(sdf_func, resolution=16, device = 'cpu', 
                      vert_colors=[.25, .25, .25],
                      sdf_scaler=1,
                      extra_sdf_offset=[0,0,0]):
        
        # # Assuming 'mosaic_sdf' is your MosaicSDF instance and 'resolution' is the desired grid resolution
        grid_points = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, resolution),
            torch.linspace(-1, 1, resolution),
            torch.linspace(-1, 1, resolution)
        ), dim=-1).reshape(-1, 3)#.to(device)
        
        # if sdf_func is None:
        #     sdf_func = self.mosaic_sdf

        sdf_values = sdf_func(grid_points.to(device))
        sdf_values *= sdf_scaler

        sdf_volume = to_numpy(sdf_values.reshape(resolution, resolution, resolution))
        
        # Run marching cubes to get vertices, faces, and normals
        sdf_verts, sdf_faces, normals, values = marching_cubes(sdf_volume, level=0)
        
        sdf_offset = np.ones(3) * resolution / 2
        sdf_max_span = np.ones(3) * resolution / 2

        sdf_verts, _, _ = ShapeSampler.normalize_vertices(sdf_verts, sdf_offset, sdf_max_span)
        sdf_verts += np.array(extra_sdf_offset)

        # faces = faces + 1  # skimage has 0-indexed faces, while PyTorch3D expects 1-indexed
        # print(verts)
        # Convert to PyTorch tensors
        sdf_verts = torch.tensor(sdf_verts, dtype=torch.float32)
        # sdf_verts = torch.tensor(sdf_verts.copy(), dtype=torch.float32)

        # verts = ShapeSampler.normalize_shape(verts)
        sdf_faces = torch.tensor(sdf_faces.copy(), dtype=torch.int64)

        total_verts = sdf_verts.shape[0]
        verts_rgb = torch.ones((1, total_verts, 3), device=device)
        verts_rgb *= torch.tensor(vert_colors, device=device)
        # verts_rgb *= torch.tensor(verts, device=device)
        # Initialize the textures with the corrected verts_rgb
        textures = Textures(verts_rgb=verts_rgb)

        # Create a PyTorch3D mesh
        mesh = Meshes(verts=[sdf_verts], faces=[sdf_faces], textures=textures).to(device)
        
        return mesh
