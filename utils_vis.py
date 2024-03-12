import point_cloud_utils as pcu
from pytorch3d.io import load_obj
from mosaic_sdf_visualizer import MosaicSDFVisualizer
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene


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