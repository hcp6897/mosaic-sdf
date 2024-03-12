import numpy as np
import torch
import torch.nn as nn
from ray import tune
from shape_sampler import ShapeSampler
from mosaic_sdf import MosaicSDF
import os
from torchviz import make_dot
from utils import to_numpy, to_tensor

class MosaicSDFOptimizer:
    def __init__(self, config):
        self.device = config['device'] #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.shape_sampler: ShapeSampler = config['shape_sampler']

        n_grids=config.get('n_grids', 1024)

        volume_centers = torch.tensor(
            self.shape_sampler.sample_n_random_points(n_grids),
            device=self.device
        )

        self.model = MosaicSDF(
            grid_resolution=config.get('grid_resolution', 7),
            n_grids=n_grids,
            volume_centers=volume_centers
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get('lr', 1e-3),
            weight_decay=config.get('weight_decay', 0),
        )

        self.lambda_val = config['lambda_val']
        self.num_iterations = config['num_iterations']
        self.val_size = config['val_size']

        # self.criterion = nn.MSELoss()  # Example loss function; adjust as needed
        self.config = config
        self.points_sample_size = config.get('points_sample_size', 64)
        self.points_random_sampling = config.get('points_random_sampling', False)
        self.points_random_spread = self.config.get('points_random_spread', .03)


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
        
        grad_delta = 1e-2

        # fx_yj_grad_num = self.compute_gradient_numerically(points_y, self.model.forward, delta=grad_delta)
        # fx_yj_grad = fx_yj_grad_num
        # an_num_grad_loss = torch.mean(torch.abs(fx_yj_grad_num - fx_yj_grad))
        
        # print(fx_yj_grad[:3])
        # fs_yj_grad = self.shape_sampler.compute_sdf_gradient(points_y, delta=1e-4)
        fs_yj_grad = self.compute_gradient_numerically(points_y, self.shape_sampler.forward, delta=grad_delta)
        # print(fs_yj_grad[:3])

        # L2 Loss - discrepancy in gradients
        l2_loss = torch.norm(fx_yj_grad - fs_yj_grad, p=2) / len(points_y)
        
        # Combined Loss
        loss = l1_loss + self.lambda_val * l2_loss
        
        return loss, l1_loss, l2_loss #, an_num_grad_loss


    def train(self):
        self.model.train()
        # total_loss = 0
        d = 3

        test_points = self.shape_sampler.sample_n_random_points(self.val_size * 2,         
                        rand_offset=self.points_random_spread,
                        random_seed=42)
        
        for iteration in range(self.num_iterations):
            
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

            self.model.update_sdf_values(self.shape_sampler)

            self.optimizer.zero_grad()
            loss, l1_loss, l2_loss = self.compute_loss(points_x, points_y)
            
            # loss = l1_loss  

            loss.backward()

            if iteration == 0 and self.config['output_graph']:
                # Visualization after computing the loss
                graph = make_dot(loss, params=dict(self.model.named_parameters()))
                graph.render(f'out/computation_graph_{iteration}', format='png')  # Saves the graph as 'computation_graph.png'


            self.optimizer.step()
            stats = {
                'step': iteration,
                'train_loss': loss.item(),
                'train_l1_loss': l1_loss.item(),
                'train_l2_loss': l2_loss.item(),
                # 'an_num_loss': an_num_grad_loss.item()
            }
            if iteration % self.config.get('print_loss_iterations', 1) == 0:
                
                self.model.eval()

                val_losses = []
                val_l1_losses = []
                val_l2_losses = []
                
                for points in torch.split(test_points, self.points_sample_size):
                    points_x = points[:points.shape[0] // 2]
                    points_y = points[points.shape[0] // 2:]
                    points_y.requires_grad=True

                    loss, l1_loss, l2_loss = self.compute_loss(points_x, points_y)
                    
                    val_losses.append(loss.item())
                    val_l1_losses.append(l1_loss.item())
                    val_l2_losses.append(l2_loss.item())
                
                stats = {
                    **stats,
                    'val_loss': np.mean(val_losses),
                    'val_l1_loss': np.mean(val_l1_losses),
                    'val_l2_loss': np.mean(val_l2_losses),
                }
                
                print(f"Iteration {stats['step']}, val Loss: {stats['val_loss']:.4f}, "
                      f"val L1: {stats['val_l1_loss']:.4f}, val L2: {stats['val_l2_loss']:.4f} ||| "
                      f"train Loss: {stats['train_loss']:.4f} "
                      f"train L1: {stats['train_l1_loss']:.4f}, train L2: {stats['train_l2_loss']:.4f} "
                    #   f"An2Num Loss: {stats['an_num_loss']:.4f}"
                      )
                
                self.model.train()
            
            
        # tune.report(loss=average_loss)  
        # wandb.report(stats)  


        
        

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
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