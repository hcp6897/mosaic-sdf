import torch
import torch.nn as nn
from ray import tune
from shape_sampler import ShapeSampler
from mosaic_sdf import MosaicSDF
import os
from torchviz import make_dot

class MosaicSDFOptimizer:
    def __init__(self, config):
        self.device = config['device'] #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.shape_sampler: ShapeSampler =config['shape_sampler']
                                  
        self.model = MosaicSDF(
            grid_resolution=config.get('grid_resolution', 7),
            n_grids=config.get('n_grids', 1024),
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get('lr', config.get('lr', 1e-3))
        )

        self.lambda_val = config['lambda_val']
        self.num_iterations = config['num_iterations']

        # self.criterion = nn.MSELoss()  # Example loss function; adjust as needed
        self.config = config
        self.points_sample_size = config.get('points_sample_size', 64)


    def compute_loss(self, points_x, points_y):
        
        self.model.update_sdf_values(self.shape_sampler)

        # Compute SDF values for the points using the Mosaic-SDF representation
        fx_xj = self.model.forward(points_x)
        fs_xj = self.shape_sampler.forward(points_x).to(self.device)
        
        # L1 Loss - discrepancy in SDF values
        l1_loss = torch.norm(fx_xj - fs_xj, p=1) / len(points_x)
        
        if False:
            # synthetic_output = torch.sum(self.model.volume_centers ** 2)
            synthetic_output = self.model.volume_centers * points_y

            # Target gradient: aiming for gradients to approach 0
            target_grad = torch.zeros_like(self.model.volume_centers)

            # Compute gradients of the synthetic function's output w.r.t. volume_centers
            volume_centers_grad = torch.autograd.grad(outputs=synthetic_output, 
                                                    #   inputs=self.model.volume_centers,
                                                      inputs=points_y,
                                                    grad_outputs=torch.ones_like(synthetic_output),
                                                    create_graph=True)[0]

            # L2 Loss on the difference between computed gradients and target gradients
            l2_loss = torch.norm(volume_centers_grad - target_grad, p=2)
            loss = l2_loss
        

        # L2 losses 
        fx_xj_l2 = self.model.forward(points_y)
        # Compute gradients (first derivatives) for the points
        fx_yj_grad = torch.autograd.grad(outputs=fx_xj_l2, 
                                            inputs=points_y, 
                                        grad_outputs=torch.ones_like(fx_xj_l2), 
                                        create_graph=True)[0]
        
        fs_yj_grad = self.shape_sampler.compute_sdf_gradient(points_y, delta=1e-4)
        
        # L2 Loss - discrepancy in gradients
        l2_loss = torch.norm(fx_yj_grad - fs_yj_grad, p=2) / len(points_y)
        
        # Combined Loss
        loss = l1_loss + self.lambda_val * l2_loss
        
        return loss


    def train(self):
        self.model.train()
        total_loss = 0
        d = 3
    
        for iteration in range(self.num_iterations):

            points_x = torch.rand((self.points_sample_size, d), 
                                  device=self.device, requires_grad=False) * 2 - 1
            points_y = torch.rand((self.points_sample_size, d), 
                                  device=self.device, requires_grad=True) * 2 - 1
            

            self.optimizer.zero_grad()
            loss = self.compute_loss(points_x, points_y)
            loss.backward()

            if iteration == 0 and self.config['output_graph']:
                # Visualization after computing the loss
                graph = make_dot(loss, params=dict(self.model.named_parameters()))
                graph.render(f'out/computation_graph_{iteration}', format='png')  # Saves the graph as 'computation_graph.png'

            self.optimizer.step()

            if iteration % self.config.get('print_loss_iterations', 1) == 0:
                print(f"Iteration {iteration}, Loss: {loss.item():.4f}")
            
            total_loss += loss.item()
            # print(f"loss: {total_loss}")

        # average_loss = total_loss / len(train_loader)
        # tune.report(loss=average_loss)  # Reporting to Ray Tune


        
        

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))
