import torch
from optimizer import MosaicSDFOptimizer

from ray import tune
from ray.train import RunConfig, CheckpointConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from ray.data import DataContext
from ray.train import Checkpoint
import os


cube_mesh_path = 'data/cube.obj'
cube_wireframe_path = 'data/cube_wireframe.obj'

sdf_shape_path = 'data/chain.obj'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


## Config

config = {   
    'device': device,
    # 'shape_sampler': shape_sampler,  # Adjust accordingly
    'shape_path': os.path.abspath(sdf_shape_path),  # Adjust accordingly
    
    # mosaicSDF params
    'grid_resolution': 7,
    # 'n_grids': 1024,
    # 'n_grids': 128,
    # 'n_grids': 8,
    'n_grids': tune.grid_search([ 256 ]),
    
    # 'points_random_spread': tune.grid_search([.01, .03, .05]),
    'points_random_spread': .03,

    'val_points_random_spread': .03,
    # 'mosaic_scale_multiplier': 3,
    'mosaic_scale_multiplier': tune.grid_search([ 2, 3 ]),
    
    # optimizer params
    # 'lr': 1e-4,
    'lr': tune.grid_search([ 1e-3, 1e-4]),
    'weight_decay': 0.0,
    "b1": tune.grid_search([0.5, 0.9]),
    "b2": .999,

    # lerp between l1 and l2 losses
    'lambda_val': .1,
    
    # optimization params
    # 'n_epochs':  tune.choice([1, 4, 8, 16]),
    # 'n_epochs':  tune.grid([ 8 ]),
    
    'n_epochs':  2,

    'points_in_epoch': 4096,
    'points_sample_size': 32,
    # 'gradient_accumulation_steps': 1,
    'gradient_accumulation_steps': tune.grid_search([1, 4, 8]),

    'eval_every_nth_points': 512,
    'val_size': 2048,
    'points_sample_size_eval_scaler': 4, # can sample faster during eval

    'project_name': 'mosaicSDF_chain',
    'log_to_wandb': True, 
    'log_to_console': False,
    
    # other debug stuff
    'output_graph': False,
    'points_random_sampling': False
}



# from ray.air.integrations.mlflow import MLflowLoggerCallback
# mlflow_tracking_uri = "http://localhost:5000"

driver_ctx = DataContext.get_current()
trainable_with_resources = tune.with_resources(MosaicSDFOptimizer, {"cpu": 10, "gpu": 1})

wb_exclude = ['training_iteration',
              'timestamp',
              'time_since_restore',
              'iterations_since_restore']

tune_storage_path = f"./out/tune/{config['project_name']}"
tune_storage_path = os.path.abspath(tune_storage_path)

tuner = tune.Tuner(
    trainable_with_resources,
    run_config=RunConfig(
        storage_path=tune_storage_path, 
        name=config['project_name'],
        stop={"training_iteration": config['n_epochs']},
        callbacks= [
         #   WandbLoggerCallback(project="hello_tune", excludes=wb_exclude),
            # MLflowLoggerCallback(
            #         tracking_uri=mlflow_tracking_uri,
            #         experiment_name=experiment_name,
            #         save_artifact=False,
            #     )
         ],
        checkpoint_config=CheckpointConfig(
            num_to_keep=10,
            checkpoint_frequency=0,  #checkpoint disabled!
            checkpoint_at_end=True
        )),
    
    tune_config=tune.TuneConfig(
        max_concurrent_trials=1,
        scheduler=ASHAScheduler(metric='val_loss', mode="min", grace_period=10),
    ),
    param_space=config
)


results = tuner.fit()

# continue_tune = False
# if not continue_tune:
#     results = tuner.fit()
# else:
#     print('resuming')
#     restored_tuner = tune.Tuner.restore(
#         Checkpoint.from_directory(os.path.join(tune_storage_path, config['project_name'])).path,
#         trainable=trainable_with_resources,
#         # Re-specify the `param_space` to update the object references.
#         param_space=config,
#         resume_unfinished=True,
#     )




