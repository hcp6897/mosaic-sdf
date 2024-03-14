import time
import numpy as np
import torch
import torch.nn as nn
from ray import tune
from shape_sampler import ShapeSampler
from mosaic_sdf import MosaicSDF
import os
import sys
from torchviz import make_dot
from utils import to_numpy, to_tensor
import wandb
from pathlib import Path

class MosaicSDFOptimizer(tune.Trainable):
    def setup(self, config):
        
        # choosing right seed value is crucial )))
        torch.manual_seed(42) 

        self.device = config['device']
        self.shape_sampler: ShapeSampler = ShapeSampler.from_file(config['shape_path'], device=self.device)

        n_grids=config.get('n_grids', 1024)

        volume_centers = self.shape_sampler.sample_n_random_points(n_grids)

        self.model = MosaicSDF(
            grid_resolution=config['grid_resolution'],
            n_grids=n_grids,
            volume_centers=volume_centers,
            mosaic_scale_multiplier=config['mosaic_scale_multiplier']
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['lr'],
            betas=(config['b1'], config['b2']),
            weight_decay=config['weight_decay'],
        )

        self.lambda_val = config['lambda_val']
        self.val_size = config['val_size']

        self.config = config
        self.points_sample_size = config['points_sample_size']
        self.points_random_sampling = config['points_random_sampling']
        self.points_random_spread = self.config['points_random_spread']
        
        self.model.update_sdf_values(self.shape_sampler)

        if config['log_to_wandb']:
            wandb.init(project=config['project_name'], config=config)



    def compute_loss(self, points_x, points_y):
     
        # Compute SDF values for the points using the Mosaic-SDF representation
        fx_xj = self.model.forward(points_x)
        fs_xj = self.shape_sampler.forward(points_x).to(self.device)
        
        # L1 Loss - discrepancy in SDF values
        l1_loss = torch.norm(fx_xj - fs_xj, p=1) / len(points_x)
        
    
        ## L2 losses 
        fx_xj_l2 = self.model.forward(points_y)
        ## Compute gradients (first derivatives) for the points
        fx_yj_grad = torch.autograd.grad(outputs=fx_xj_l2, 
                                            inputs=points_y, 
                                        grad_outputs=torch.ones_like(fx_xj_l2), 
                                        create_graph=True)[0]
        
        # todo check other values
        grad_delta = 1e-2

        fs_yj_grad = self.compute_gradient_numerically(points_y, self.shape_sampler.forward, delta=grad_delta)

        # L2 Loss - discrepancy in gradients
        l2_loss = torch.norm(fx_yj_grad - fs_yj_grad, p=2) / len(points_y)
        
        # Combined Loss
        loss = (1 - self.lambda_val) * l1_loss + self.lambda_val * l2_loss
        
        return loss, l1_loss, l2_loss #, an_num_grad_loss


    def step(self):
        
        epoch_time_start = time.time()
        self.model.train()
        d = 3
        gradient_accumulation_steps = self.config['gradient_accumulation_steps']

        test_points = self.shape_sampler.sample_n_random_points(self.val_size * 2,         
                        rand_offset=self.config['val_points_random_spread'],
                        random_seed=42)
        
        n_steps = self.config['points_in_epoch'] // self.points_sample_size
        eval_every_nth_step = self.config['eval_every_nth_points'] // self.points_sample_size
        
        for iteration in range(n_steps):
            
            if self.points_random_sampling:
                points_x = (torch.rand((self.points_sample_size, d), 
                                    device=self.device, requires_grad=False) - 1) * 2
                points_y = (torch.rand((self.points_sample_size, d), 
                                    device=self.device, requires_grad=True) - 1) * 2
            else:
                points = self.shape_sampler.sample_n_random_points(self.points_sample_size * 2,         
                                                                   rand_offset=self.points_random_spread,
                                                                   random_seed=iteration)

                
                points_x = points[:points.shape[0] // 2]
                points_y = points[points.shape[0] // 2:]
                points_y.requires_grad=True           

            
            self.optimizer.zero_grad()
            loss, l1_loss, l2_loss = self.compute_loss(points_x, points_y)
            
            loss = loss / gradient_accumulation_steps  # Scale loss
            loss = loss * self.config.get('loss_scaler', 1)

            loss.backward()

            if (iteration + 1) % gradient_accumulation_steps == 0 or iteration == n_steps - 1:
                self.optimizer.step()  # Update parameters
                self.optimizer.zero_grad()  # Reset gradients
                self.model.update_sdf_values(self.shape_sampler)

            stats = {
                'step': iteration,
                'train_loss': loss.item(),
                'train_l1_loss': l1_loss.item(),
                'train_l2_loss': l2_loss.item(),
                # 'an_num_loss': an_num_grad_loss.item()
            }
            
            if (
                (iteration + 1) % eval_every_nth_step == 0 
                or iteration == n_steps - 1
            ):
            
                self.model.eval()
                # with torch.no_grad():

                val_losses = []
                val_l1_losses = []
                val_l2_losses = []
                
                for points in torch.split(test_points, 
                                        self.points_sample_size * self.config['points_sample_size_eval_scaler'] * 2):
                    points_x = points[:points.shape[0] // 2]
                    points_y = points[points.shape[0] // 2:]
                    points_y.requires_grad=True

                    loss, l1_loss, l2_loss = self.compute_loss(points_x, points_y)
                    
                    val_losses.append(loss.item())
                    val_l1_losses.append(l1_loss.item())
                    val_l2_losses.append(l2_loss.item())

                    self.optimizer.zero_grad()
            
                stats = {
                    **stats,
                    'val_loss': np.mean(val_losses),
                    'val_l1_loss': np.mean(val_l1_losses),
                    'val_l2_loss': np.mean(val_l2_losses),
                }
                
                if self.config.get('log_to_console', True):
                    sys.stdout.write(f"\nIteration {stats['step']}, val Loss: {stats['val_loss']:.4f}, "
                        f"val L1: {stats['val_l1_loss']:.4f}, val L2: {stats['val_l2_loss']:.4f} ||| "
                        f"train Loss: {stats['train_loss']:.4f} "
                        f"train L1: {stats['train_l1_loss']:.4f}, train L2: {stats['train_l2_loss']:.4f} "
                        #   f"An2Num Loss: {stats['an_num_loss']:.4f}"
                        )
                    sys.stdout.flush()
                
                if self.config['log_to_wandb'] and iteration != n_steps - 1:
                    wandb.log(stats)
                
                
                self.model.train()
            
            
            if iteration == 0 and self.config['output_graph']:
                # Visualization after computing the loss
                graph = make_dot(loss, params=dict(self.model.named_parameters()))
                graph.render(f'out/computation_graph_{iteration}', format='png')  # Saves the graph as 'computation_graph.png'


        stats['epoch_time'] = time.time() - epoch_time_start
        if self.config['log_to_wandb'] and iteration != n_steps - 1:
            wandb.log(stats)
        
        return stats


        
        

    def save_checkpoint(self, checkpoint_dir):
        # dir_path = os.path.join(checkpoint_dir, self.config['project_name'])
        dir_path = checkpoint_dir
        Path(dir_path).mkdir(parents=True, exist_ok=True)

        checkpoint_path = os.path.join(dir_path, "model.pth")

        print(f'model saved to: {checkpoint_path}')
        torch.save(self.model.state_dict(), checkpoint_path)
        

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))



    def compute_gradient_numerically(self, points, forward_func, delta=1e-4):
        """
        Approximate the gradient of the SDF at given points using central differences.
        
        Args:
        - points: Tensor of shape (N, 3) representing N points in 3D space.
        - delta: A small offset used for finite differences.
        
        Returns:
        - grad: Tensor of shape (N, 3) representing the approximate gradient of the SDF at each point.
        """
        device = points.device
        N, D = points.shape
        # grad = torch.zeros_like(points, requires_grad=False, device=device)
        grad = torch.zeros_like(points, device=device)
        
        for i in range(D):
            # Create a basis vector for the i-th dimension
            offset = torch.zeros(D, device=device)
            offset[i] = delta
            
            # Compute SDF at slightly offset points
            sdf_plus = forward_func(points + offset)
            sdf_minus = forward_func(points - offset)
            
            # Approximate the derivative using central differences
            grad[:, i] = (sdf_plus - sdf_minus) / (2 * delta)
        
        return grad