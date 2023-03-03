# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Author: Andreas Look, Andreas.Look@de.bosch.com


import torch
import torch.nn as nn
from .layers import LinearMM, ReluMM, DropoutMM
FloatTensor = torch.FloatTensor

class BaseNSDE(nn.Module):
    """
    Parent class for NSDE models. 
    Provides the BMM algorithm. 
    
    Parameters:
        - dt (float): timestep 
    """
    def __init__(self, dt=.1):
        super().__init__()
        self.dt = FloatTensor([dt])
        
    def drift(self, x):
        raise NotImplementedError("Drift not implemented.")

    def drift_moments(self, m, P):
        raise NotImplementedError("Drift-Moment not implemented.")
        
    def diffusion(self, x):
        raise NotImplementedError("Diffusion not implemented.")
        
    def diffusions_moments(self, m, P):
        raise NotImplementedError("Diffusion-Moment not implemented.")
        
    def expected_gradient(self):
        return self.exp_jac
                
    def diffusion_central_moment(self, m, P):
        m, P = self.diffusions_moments(m, P)
        P_central = P + m.unsqueeze(-1)@m.unsqueeze(-1).transpose(1,2) # Eq. 12
        P_central = torch.diag_embed(torch.diagonal(P_central, 0, 1, 2))
        return P_central          
    
    def next_moments(self, m, P):
        """
        Moment matching mode. Applies the method from our paper. 
        
        Parameters:
            - m (torch.FloatTensor): mean with shape (batch_size, d_in)
            - P (torch.FloatTensor): covariance with (batch_size, d_in, d_in)
            
        Returns:
            - m_nxt (torch.FloatTensor): output mean with shape (batch_size, d_in)
            - P_nxt (torch.FloatTensor): output covariance with (batch_size, d_in, d_in)
        """
        f_m, P_ff = self.drift_moments(m, P)
        f_m = f_m*self.dt
        P_ff = P_ff*(self.dt**2) # Cov(f)                           
        L_P_central = self.diffusion_central_moment(m, P)*self.dt # E[LL^T]

        Fx = self.exp_jac # Expected Jacobian
        P_xf = Fx@P*(self.dt) # Cov(x,f), Eq. 13
        
        P_nxt = P + P_ff + P_xf + P_xf.transpose(1,2) + L_P_central # Eq. 20
        m_nxt = m + f_m
        return m_nxt, P_nxt
    
    
    def forward(self, x):
        """
        Monte Carlo mode. Applies Euler Maruyama discretization.
        
        Parameters: 
            - x (torch.FloatTensor): input with shape (batch_size, d_in)
        
        Returns: 
            - y (torch.FloatTensor): output with shape (batch_size, d_in)
        """
        noise = torch.randn(*x.shape,1)*torch.sqrt(self.dt)
        y = x + self.drift(x)*self.dt + (torch.diag_embed(self.diffusion(x))@noise).view(-1, self.d)
        return y
    
    
    
class NSDE(BaseNSDE):
    """
    Neural SDE with 3 layer drift and 2 layer diffusion function.
    
    Parameters:
        - d  (int): dimensionality 
        - dt (float): timestep 
        - n_hidden (int): number of hidden neurons
        - p  (float): 1- dropoutrate
    """
    
    def __init__(self, d=2, dt=.05, n_hidden=50, p=0.8):
        super().__init__()
        
        self.d = d
        self.dt = FloatTensor([dt])
        self.n_hidden = n_hidden
        self.p=p
        
        self.f1 = LinearMM(self.d, self.n_hidden)
        self.f2 = LinearMM(self.n_hidden, self.n_hidden)
        self.f3 = LinearMM(self.n_hidden, self.d)
        
        self.L1 = LinearMM(self.d, self.n_hidden)
        self.L2 = LinearMM(self.n_hidden, self.d)
        
        self.relu = ReluMM()
        self.dropout = DropoutMM(p=p)
        
              
    def drift(self, x):
        """
        Drift function in Monte Carlo mode.
        
        Parameters: 
            - x (torch.FloatTensor): input with shape (batch_size, d_in)
        
        Returns: 
            - y (torch.FloatTensor): output with shape (batch_size, d_in)
        """
        y = self.f1(x)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.f2(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.f3(y)        
        return y
            
    def drift_moments(self, m, P):
        """
        Drift function in moment matching mode. Stores the expected jacobian in self.exp_jac.
        
        Parameters:
            - m (torch.FloatTensor): mean with shape (batch_size, d_in)
            - P (torch.FloatTensor): covariance with (batch_size, d_in, d_in)
            
        Returns:
            - m_nxt (torch.FloatTensor): output mean with shape (batch_size, d_in)
            - P_nxt (torch.FloatTensor): output covariance with (batch_size, d_in, d_in)
        """
        jac = self.f1.jac(m, P)
        m_nxt, P_nxt = self.f1.next_moments(m, P)
            
        jac = self.relu.jac(m_nxt, P_nxt)@jac
        m_nxt, P_nxt = self.relu.next_moments(m_nxt, P_nxt)
        
        jac = self.dropout.jac(m_nxt, P_nxt)@jac
        m_nxt, P_nxt = self.dropout.next_moments(m_nxt, P_nxt) 
        
        jac = self.f2.jac(m_nxt, P_nxt)@jac
        m_nxt, P_nxt = self.f2.next_moments(m_nxt, P_nxt)
                
        jac = self.relu.jac(m_nxt, P_nxt)@jac
        m_nxt, P_nxt = self.relu.next_moments(m_nxt, P_nxt)
        
        jac = self.dropout.jac(m_nxt, P_nxt)@jac
        m_nxt, P_nxt = self.dropout.next_moments(m_nxt, P_nxt) 
        
        jac = self.f3.jac(m_nxt, P_nxt)@jac
        m_nxt, P_nxt = self.f3.next_moments(m_nxt, P_nxt)
        
        self.exp_jac = jac # expected Jacobian
        
        return m_nxt, P_nxt
          
    def diffusion(self, x):
        """
        Diagonal diffusion function in Monte Carlo mode.
        
        Parameters: 
            - x (torch.FloatTensor): input with shape (batch_size, d_in)
        
        Returns: 
            - y (torch.FloatTensor): output with shape (batch_size, d_in)
        """
        
        y = self.L1(x)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.L2(y)
        y = self.relu(y)
        return y
    
    def diffusions_moments(self, m, P):
        """
        iagonal diffusion function in moment matching mode.
        
        Parameters:
            - m (torch.FloatTensor): mean with shape (batch_size, d_in)
            - P (torch.FloatTensor): covariance with (batch_size, d_in, d_in)
            
        Returns:
            - m_nxt (torch.FloatTensor): output mean with shape (batch_size, d_in)
            - P_nxt (torch.FloatTensor): output covariance with (batch_size, d_in, d_in)
        """
        m_nxt, P_nxt = self.L1.next_moments(m, P)
        m_nxt, P_nxt = self.relu.next_moments(m_nxt, P_nxt)
        m_nxt, P_nxt = self.dropout.next_moments(m_nxt, P_nxt)
        m_nxt, P_nxt = self.L2.next_moments(m_nxt, P_nxt)
        m_nxt, P_nxt = self.relu.next_moments(m_nxt, P_nxt)
        return m_nxt, P_nxt  