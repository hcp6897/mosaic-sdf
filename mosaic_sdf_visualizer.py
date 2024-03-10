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

        self.template_mesh = self.create_mesh_from_verts(template_vertices, 
                                                    template_faces, vert_colors = [.8, 0, 0])

        self.boundary_mesh = self.create_mesh_from_verts(template_vertices, 
                                                    template_faces, vert_colors = [0, 0, .5])

        self.shape_target_mesh = self.create_mesh_from_verts(self.shape_sampler.vertices, 
                                                  self.shape_sampler.faces, 
                                                  vert_colors = [0, .3, 0])
    

    def create_mesh_from_verts(self, vertices, faces, vert_colors = [.6, 0, 0]):
        total_verts = vertices.shape[0]
        verts_rgb = torch.ones((1, total_verts, 3), device=self.device)  # White color for all vertices
        verts_rgb *= torch.tensor(vert_colors, device=self.device)
        # Initialize the textures with the corrected verts_rgb
        textures = Textures(verts_rgb=verts_rgb)

        # Create the mesh
        return Meshes(verts=[vertices.to(self.device)], faces=[faces.verts_idx.to(self.device)], 
                      textures=textures).to(self.device)


    def create_mosaic_grid_meshes(self):
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

        all_meshes = [self.shape_target_mesh]


        def scale_offset_mesh(mesh, offset, scale):
            scaled_verts = self.template_mesh.verts_list()[0] * scale
            # Translate the mesh
            translated_verts = scaled_verts + offset
            # Create a new mesh with the transformed vertices and the same faces
            new_mesh = mesh.clone()
            new_mesh = new_mesh.update_padded(new_verts_padded=translated_verts.unsqueeze(0))
            return new_mesh


        for center, scale in zip(volume_centers, scales):
            new_mesh = scale_offset_mesh(self.template_mesh, center, scale)
            all_meshes.append(new_mesh)
        
        new_mesh = scale_offset_mesh(self.boundary_mesh, 0, 2)
        all_meshes.append(new_mesh)
            
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
            meshes = self.create_mosaic_grid_meshes()
            vis = self.render_meshes(meshes)
            # vis.shape
            plt.figure(figsize=(4, 4))
            plt.imshow(vis[0, ..., :3].cpu().numpy())
            plt.axis("off")



    def rasterize_mosaic_sdf(self, resolution=8, device = 'cpu'):
        
        # # Assuming 'mosaic_sdf' is your MosaicSDF instance and 'resolution' is the desired grid resolution
        grid_points = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, resolution),
            torch.linspace(-1, 1, resolution),
            torch.linspace(-1, 1, resolution)
        ), dim=-1).reshape(-1, 3)#.to(device)

        sdf_values = self.mosaic_sdf(grid_points.to(self.device))
        sdf_values = torch.clamp(sdf_values, -1, 1)
        sdf_volume = sdf_values.reshape(resolution, resolution, resolution).to(device).numpy()

        # Run marching cubes to get vertices, faces, and normals
        verts, faces, normals, values = marching_cubes(sdf_volume, level=0)
        # faces = faces + 1  # skimage has 0-indexed faces, while PyTorch3D expects 1-indexed

        # Convert to PyTorch tensors
        verts = torch.tensor(verts.copy(), dtype=torch.float32)
        faces = torch.tensor(faces.copy(), dtype=torch.int64)

        total_verts = verts.shape[0]
        verts_rgb = torch.ones((1, total_verts, 3), device=device)
        verts_rgb *= torch.tensor([.75, 0, 0], device=device)
        # verts_rgb *= torch.tensor(verts, device=device)
        # Initialize the textures with the corrected verts_rgb
        textures = Textures(verts_rgb=verts_rgb)

        # Create a PyTorch3D mesh
        mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
        return mesh
