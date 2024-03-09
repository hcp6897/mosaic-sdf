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
    Materials
)
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.renderer.mesh.textures import Textures
from pytorch3d.structures.meshes import join_meshes_as_scene

from mosaic_sdf import MosaicSDF
from shape_sampler import ShapeSampler

from pytorch3d.io import load_obj

class MosaicSDFVisualizer:
    def __init__(self, mosaic_sdf: MosaicSDF, shape_sampler: ShapeSampler, device, template_mesh_path:str):
        self.mosaic_sdf = mosaic_sdf
        self.shape_sampler = shape_sampler
        self.device = device #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        R, T = look_at_view_transform(2.7, 0, 180)
        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        
        self.lights = PointLights(device=self.device, location=[[0.0, 0.0, -3.0]])
        
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
                                                    template_faces, vert_colors = [.6, 0, 0])

        self.shape_target_mesh = self.create_mesh_from_verts(self.shape_sampler.vertices, 
                                                  self.shape_sampler.faces, 
                                                  vert_colors = [0, .6, 0])
    

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

        for center, scale in zip(volume_centers, scales):

            scaled_verts = self.template_mesh.verts_list()[0] * scale
            # Translate the mesh
            translated_verts = scaled_verts + center
            # Create a new mesh with the transformed vertices and the same faces
            new_mesh = self.template_mesh.clone()
            new_mesh = new_mesh.update_padded(new_verts_padded=translated_verts.unsqueeze(0))
            
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


    def visualize_shape(self):
        # Fetch the shape mesh from ShapeSampler
        mesh = self.shape_sampler.get_mesh()
        
        # Render the shape mesh
        rendered_shape = self.renderer(mesh)
        
        return rendered_shape

    def visualize(self):
        # Combine visualization of grids and shape
        self.visualize_grids()
        shape_image = self.visualize_shape()
        
        # Placeholder for displaying the rendered image; integrate with your image display tool/framework
        # For example, if using IPython display: display(Image(tensor_to_image(shape_image)))
        pass
