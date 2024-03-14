import point_cloud_utils as pcu
from pytorch3d.io import load_obj
from mosaic_sdf_visualizer import MosaicSDFVisualizer
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
import torch
from shape_sampler import ShapeSampler
from mosaic_sdf import MosaicSDF

def compare_shapes(shape_sampler: ShapeSampler, 
                   visualizer: MosaicSDFVisualizer, 
                   mosaic_sdf: MosaicSDF,
                   resolution = 32, device='cuda',
                   show_mosaic_grids = False, 
                   show_gt_sdf=True, show_gt_mesh=True, show_mosaic_sdf=True, **kwargs):

    with torch.no_grad():
        plot_dict = {}
         
        if show_gt_sdf:
            plot_dict['gt_sdf_mesh'] = MosaicSDFVisualizer.rasterize_sdf(
                sdf_func=shape_sampler.forward, resolution=resolution, sdf_scaler=-1, 
                extra_sdf_offset=[2,0, 0], vert_colors=[0, .5, 0], 
                **kwargs)
        if show_gt_mesh:
            plot_dict['gt_mesh'] = visualizer.create_state_meshes(
                mosaic_sdf=mosaic_sdf,
                show_mosaic_grids=False,
                show_target_mesh=True,
                show_boundary_mesh=kwargs.get('show_boundary_mesh', False),
                resolution=resolution,
                show_rasterized_sdf_mesh=False,
                vert_colors=[0, 0, .5],
                offset_vertices=torch.tensor([-2,0,0], device=device), 
                **kwargs
                )
        if show_mosaic_sdf:
            plot_dict['mosaic_meshes'] = visualizer.create_state_meshes(
                mosaic_sdf=mosaic_sdf,
                show_mosaic_grids=show_mosaic_grids,
                show_target_mesh=False,
                show_boundary_mesh=kwargs.get('show_boundary_mesh', False),
                resolution=resolution,
                vert_colors=[.5, .5, 0], 
                **kwargs
                )
        
        # Render the plotly figure
        subplots = {
            # "subplot1": plot_dict
        }
        for k, v in plot_dict.items():
            subplots[k] = {'plot':v}
        fig = plot_scene(subplots, ncols=len(plot_dict))
        fig.show()    
    


def copy_spheres_to_points(v_sampled, other_meshes_to_show, device):
    # copy spheres to sampled point
    sphere_mesh_path = 'data/sphere.obj'

    v_sphere, f_sphere = pcu.load_mesh_vf(sphere_mesh_path)

    scale_sphere = .05
    v_sphere *= scale_sphere

    spheres_to_show = {}
    for p in v_sampled:
        v_sphere_p = v_sphere + p
        sphere_mesh = MosaicSDFVisualizer.create_mesh_from_verts(v_sphere_p, f_sphere, vert_colors = [0, .8, 0], device=device)
        spheres_to_show[f'm_{p}'] = sphere_mesh
    

    # Render the plotly figure
    fig = plot_scene({
        "subplot1": {
            # "wt_mesh": wt_mesh,
            "source_mesh": other_meshes_to_show,
            **spheres_to_show
        }
    })
    fig.show() 

