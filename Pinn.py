import torch
from Net import Net
# from Net_1 import Net
import torch
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
class PINN:
    def __init__(self,hidden_structure):
        self.spatial_dim = 3
        self.net = Net(self.spatial_dim, hidden_structure)
        self.collocation_points = None
        self.initial_condition_points = None
        self.l_scale = 1
        self.t_scale = 1
        self.source_stdev = 1
        self.source_value = 1
    # def set_location(self, source_locs, max_vals, source_values = None, kappa = 1e-2, sigma = .025, trainable = False):
    #     self.source_locs = source_locs
    #     #figure out if there should be 3 individual scales, or just 1 across x,y,z
    #     self.l_scale = max(max_vals[1:])
    #     self.source_locs_scaled = source_locs / self.l_scale
    #     self.t_scale = max_vals[0]
    #     self.t_max = max_vals[0]
    #     if source_values is not None:
    #         self.source_mixture_hm = Gaussian_Mixture(self.source_locs_scaled,[[sigma]*self.spatial_dim for _ in range(len(source_values))],source_values,trainable)
    #         self.q = self.source_mixture_hm.magnitude
    #     else:
    #         self.q = [0] * len(source_locs)

    def set_default_collocation_points(self,collocation_points):
        self.collocation_points = collocation_points

    def forward(self,input_tensor,scaled=False):
        if scaled:
            return self.net(input_tensor)
        else:
            temp = input_tensor.clone()
            temp[:,1:] = temp[:,1:]/self.l_scale
            temp[:,0] = temp[:,0]/self.t_scale
        return self.net(input_tensor)

    def scale_tensor(self, loc_tensor, wind_tensor = None):
        loc_temp = loc_tensor.clone()
        loc_temp[:,1:] = loc_temp[:,1:]/self.l_scale
        loc_temp[:,0] = loc_temp[:,0]/self.t_scale
        if wind_tensor != None:
            wind_temp = wind_tensor.clone()/self.l_scale/self.t_scale
        return loc_temp, wind_temp


    def compute_pde_loss(self, tx, wind_vector, source_term = None, scaled = False):
        # assumes v has shape (1, spatial_dim)
        # assumes source_loc has shape (1, spatial_dim)
        # assumes source_value is a scalar
        if scaled == False:
            # tx, wind_vector = self.scale_tensor(tx, wind_vector)
            pass

        batch_size = tx.shape[0]
        spatial_dim =self.spatial_dim

        tx.requires_grad_()
        u = self.net(tx)
        u_x = torch.autograd.grad(outputs=u, inputs=tx, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True, allow_unused=True)[0]
        u_xx = torch.autograd.grad(outputs=u_x, inputs=tx, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True, allow_unused=True)[0]
        u_x[:,1:] *= 1/self.l_scale
        u_xx *= 1/self.l_scale**2
        u_x[:,0:1] *= 1/self.t_scale

        assert u.shape == (batch_size,1)

        assert u_x.shape == (batch_size, spatial_dim+1)
        assert u_xx.shape == (batch_size, spatial_dim+1)

        laplace_term = torch.sum(u_xx[:,1:], dim=1).view(batch_size, 1)
        assert wind_vector.shape == u_x[:,1:3].shape
        velocity_term = torch.sum(wind_vector*u_x[:,1:3], dim=1).view(batch_size, 1)

        assert laplace_term.shape == (batch_size, 1)
        assert velocity_term.shape == (batch_size, 1)

        source_term = self.evaluate_source(tx)

        assert source_term.shape == (batch_size, 1)
        # compute loss
        assert u_x[:,0:1].shape == velocity_term.shape
        kappa = 1*1e-5

        assert u_x[:,0:1].shape ==velocity_term.shape == laplace_term.shape == source_term.shape

        pde_loss = torch.mean( torch.square((u_x[:,0:1] + velocity_term - kappa * laplace_term - source_term) ))
        return pde_loss
    

    def compute_negative_loss(self,points):
        u = self.forward(points)
        return torch.mean((torch.abs(u)-u)**2)
    
    def compute_data_loss(self,data,data_values):

        assert data.shape == (data_values.shape[0],self.spatial_dim+1)
        assert len(data.shape) == 2
        assert data_values.shape[1] == 1
        return torch.mean(torch.square(self.forward(data,False) - data_values))

    def train(self,num_epochs, initial_condition = None, collocation = None):
        if initial_condition == None:
            initial_condition = self.initial_condition_points
        if collocation == None:
            collocation = self.collocation_points
        


    def evaluate_source(self,tx):
        X = tx[:,1:]
        n = X.shape[0]
        source_stdev = torch.tensor([self.source_stdev]).repeat((n,1))
        # assert source_stdev.shape == (len(x)*self.num_gaussian,self.spatial_dim)
        assert source_stdev.shape == (n,1)
        # res = self.magnitude[i]*1/(((2*torch.pi)**(self.spatial_dim/2))*torch.prod(source_stdev,dim=1))*torch.exp(-torch.sum(torch.square(x - source_pts)/(2*source_stdev**2),dim=1))
        res = self.source_value/(((2*torch.pi)**(3/2))*source_stdev**3)*torch.exp(torch.sum(-(X - 0)**2/(2*source_stdev**2),dim=1)).reshape(-1,1)
        assert res.shape == (n,1)
        return res
    

# print(pinn.evaluate_source(torch.tensor([[0,0,0],[1,1,1]])))

X = torch.tensor([[0.,0.,0.,0],[1.,1.,1.,1.]])
# print(X)
# print(PINN(X))
