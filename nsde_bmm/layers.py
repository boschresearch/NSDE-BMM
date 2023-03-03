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
#
# The source code for the class ReluMM and HeavisideMM is derived from 
# Deterministic Variational Inferencen (https://github.com/microsoft/deterministic-variational-inference)
# Copyright (c) 2018 Microsoft Corporation. All rights reserved.
# licensed under MIT License
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.


import torch
import torch.nn as nn
import numpy as np
   
class BaseLayer(nn.Module):
    """
    Parent class for all moment matching layers.
    Provides input (mean and covariance) checking.
    """
    
    def __init__(self):
        super().__init__()
        
    def _assert_dim(self, m, P):
        assert m.dim()==2, "Expect 2 dimensional mean: batch_size x dim"
        assert P.dim()==3, "Expect 3 dimensional Covariance: batch_size x dim x dim"
    
class NonLinearLayer(BaseLayer):
    """
    Parent class for nonlinear moment matching layers. 
    Provides necessary constants for downstream computations. 
    """
    
    def __init__(self):
        super().__init__()
        self.register_buffer('_eps', torch.FloatTensor([1e-5]))
        self.register_buffer('_one', torch.FloatTensor([1.0]))
        self.register_buffer('_one_ovr_sqrt2pi', torch.FloatTensor([1.0 / np.sqrt(2.0 * np.pi)]))
        self.register_buffer('_one_ovr_sqrt2', torch.FloatTensor([1.0 / np.sqrt(2.0)]))
        self.register_buffer('_one_ovr_2', torch.FloatTensor([1.0/2.0]))
        self.register_buffer('_two', torch.FloatTensor([2.0]))
        self.register_buffer('_twopi', torch.FloatTensor([2.0 * np.pi]))
          
    
class LinearMM(BaseLayer):
    """
    Linear layer with Monte Carlo and moment matching mode.
       
    Args:
        - d_in (int): input dimensionality
        - d_out (int): output dimensionality
    """
    
    def __init__(self, d_in, d_out):
        super().__init__()
        
        self.d_in = d_in
        self.d_out = d_out
        
        self._layer = torch.nn.Linear(d_in, d_out)
        
    def jac(self, m, P):
        """
        Calculates the expected Jacobian.
        
        Parameters:
            - m (torch.FloatTensor): mean with shape (batch_size, d_in)
            - P (torch.FloatTensor): covariance with (batch_size, d_in, d_in)
            
        Returns: 
            - jac (torch.FloatTensor): jacobian with (batch_size, d_out, d_in)
        """
        batch_size, _ = m.shape
        jac = self._layer.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        return jac
    
    def next_moments(self, m, P):
        """
        Moment matching mode.
        
        Parameters:
            - m (torch.FloatTensor): mean with shape (batch_size, d_in)
            - P (torch.FloatTensor): covariance with (batch_size, d_in, d_in)
            
        Returns:
            - m_nxt (torch.FloatTensor): output mean with shape (batch_size, d_out)
            - P_nxt (torch.FloatTensor): output covariance with (batch_size, d_out, d_out)
        """
        self._assert_dim(m, P)
        
        m_nxt = self._layer(m)
        P_nxt = self._layer.weight.unsqueeze(0)@P@self._layer.weight.unsqueeze(0).transpose(1,2)
        return m_nxt, P_nxt
        
    def forward(self, x):
        """
        Monte Carlo mode.
        
        Parameters: 
            - x (torch.FloatTensor): input with shape (batch_size, d_in)
        
        Returns: 
            - y (torch.FloatTensor): output with shape (batch_size, d_out)
        """
        return self._layer(x)
 
    
class DropoutMM(BaseLayer):
    """
    Dropout layer with Monte Carlo and moment matching mode.
       
    Args:
        - p (float): probability of element to be kept, i.e. propability of drawing a 1.
    """
    
    def __init__(self, p=0.9):
        super().__init__()
        
        self.p=p
        self._layer = torch.nn.Dropout(1-self.p)
        
    def jac(self, m, P):
        """
        Calculates the expected Jacobian.
        
        Parameters:
            - m (torch.FloatTensor): mean with shape (batch_size, d_in)
            - P (torch.FloatTensor): covariance with (batch_size, d_in, d_in)
            
        Returns: 
            - jac (torch.FloatTensor): jacobian with (batch_size, d_in, d_in)
        """
        batch_size, d_in = m.shape
        jac = torch.eye(d_in).unsqueeze(0).repeat(batch_size, 1, 1)
        return jac
    
        
    def next_moments(self, m, P):
        """
        Moment matching mode.
        
        Parameters:
            - m (torch.FloatTensor): mean with shape (batch_size, d_in)
            - P (torch.FloatTensor): covariance with (batch_size, d_in, d_in)
            
        Returns:
            - m_nxt (torch.FloatTensor): output mean with shape (batch_size, d_in)
            - P_nxt (torch.FloatTensor): output covariance with (batch_size, d_in, d_in)
        """
        self._assert_dim(m, P)
        
        m_nxt = m
        
        m_square = m**2
        P_diag = torch.diagonal(P, 0, 1,2)
        P_diag_out = -P_diag*(self.p**2) + P_diag*self.p + self.p*(1-self.p)*m_square
        P_diag_out = torch.diag_embed(P_diag_out)
        P_nxt = P*(self.p**2) + P_diag_out
        P_nxt = P_nxt/(self.p**2) # scale back to orignal mean
        return m_nxt, P_nxt
        
    def forward(self, x):
        """
        Monte Carlo mode.
        
        Parameters: 
            - x (torch.FloatTensor): input with shape (batch_size, d_in)
        
        Returns: 
            - y (torch.FloatTensor): output with shape (batch_size, d_in)
        """
        return self._layer(x)    
    
    
class ReluMM(NonLinearLayer):
    """
    ReLU layer with Monte Carlo and moment matching mode.
    """
    def __init__(self):
        super().__init__()
        
        self._activation = torch.nn.ReLU()
        self._heaviside = HeavisideMM()
        
    def _standard_gaussian(self, x):
        """
        line 17-18 from
        https://github.com/microsoft/deterministic-variational-inference/blob/master/bayes_util.py
        """
        return self._one_ovr_sqrt2pi * torch.exp(-x*x * self._one_ovr_2)

    def _gaussian_cdf(self, x):
        """
        line 20-21 from
        https://github.com/microsoft/deterministic-variational-inference/blob/master/bayes_util.py
        """
        return self._one_ovr_2 * (self._one + torch.erf(x * self._one_ovr_sqrt2))

    def _softrelu(self, x):
        """
        line 23-24 from
        https://github.com/microsoft/deterministic-variational-inference/blob/master/bayes_util.py
        """
        return self._standard_gaussian(x) + x * self._gaussian_cdf(x)
      
    def _g(self, rho, mu1, mu2):
        """
        line 26-36 from
        https://github.com/microsoft/deterministic-variational-inference/blob/master/bayes_util.py
        """
        one_plus_sqrt_one_minus_rho_sqr = (self._one + torch.sqrt(self._one - rho*rho))
        a = torch.asin(rho) - rho / one_plus_sqrt_one_minus_rho_sqr
        safe_a = torch.abs(a) + self._eps
        safe_rho = torch.abs(rho) + self._eps

        A = a / self._twopi
        sxx = safe_a * one_plus_sqrt_one_minus_rho_sqr / safe_rho
        one_ovr_sxy = (torch.asin(rho) - rho) / (safe_a * safe_rho)
    
        return A * torch.exp(-(mu1*mu1 + mu2*mu2) / (self._two * sxx) + one_ovr_sxy * mu1 * mu2)
    
    def _delta(self, rho, mu1, mu2):
        """
        line 38-39 from
        https://github.com/microsoft/deterministic-variational-inference/blob/master/bayes_util.py
        """
        return self._gaussian_cdf(mu1) * self._gaussian_cdf(mu2) + self._g(rho, mu1, mu2)
    
    def relu_full_covariance(self, P, mu, P_diag):
        """
        line 39-47 from
        https://github.com/microsoft/deterministic-variational-inference/blob/master/bayes_layers.py
        """
        mu1 = mu.unsqueeze(-1)
        mu2 = mu1.permute(0,2,1)

        s11s22 = P_diag.unsqueeze(2) *  P_diag.unsqueeze(1)
        rho = P / (torch.sqrt(s11s22) + self._eps)
        rho = torch.clamp(rho, -1/(1+1e-5), 1/(1+1e-5))

        return P * self._delta(rho, mu1, mu2)   
    
    
    def jac(self, m, P):
        """
        Calculates the expected Jacobian.
        
        Parameters:
            - m (torch.FloatTensor): mean with shape (batch_size, d_in)
            - P (torch.FloatTensor): covariance with (batch_size, d_in, d_in)
            
        Returns: 
            - jac (torch.FloatTensor): jacobian with (batch_size, d_in, d_in)
        """
        return torch.diag_embed(self._heaviside.next_mean(m, P))
    
    def next_moments(self, m, P):
        """
        Moment matching mode.
        
        Parameters:
            - m (torch.FloatTensor): mean with shape (batch_size, d_in)
            - P (torch.FloatTensor): covariance with (batch_size, d_in, d_in)
            
        Returns:
            - m_nxt (torch.FloatTensor): output mean with shape (batch_size, d_in)
            - P_nxt (torch.FloatTensor): output covariance with (batch_size, d_in, d_in)
            
        line 35-37, 49, and 51  from
        https://github.com/microsoft/deterministic-variational-inference/blob/master/bayes_layers.py
        """
        self._assert_dim(m, P)
        
        P_diag = torch.diagonal(P, offset=0, dim1=1, dim2=2)
        P_diag_sqrt = torch.sqrt(P_diag)
        mu = m / (P_diag_sqrt + self._eps)

        m_nxt = P_diag_sqrt * self._softrelu(mu)
        P_nxt = self.relu_full_covariance(P, mu, P_diag)
        return m_nxt, P_nxt
        
    def forward(self, x):
        """
        Monte Carlo mode.
        
        Parameters: 
            - x (torch.FloatTensor): input with shape (batch_size, d_in)
        
        Returns: 
            - y (torch.FloatTensor): output with shape (batch_size, d_in)
        """
        return self._activation(x)
    
class HeavisideMM(NonLinearLayer):
    """
    Heaviside layer with moment matching mode only. Is necessary for Jacobian calculation of the ReLU layer.
    """
    
    def __init__(self):
        super().__init__()
        
    def _gaussian_cdf(self, x):
        """
        line 20-21 from
        https://github.com/microsoft/deterministic-variational-inference/blob/master/bayes_util.py
        """
        return self._one_ovr_2 * (self._one + torch.erf(x * self._one_ovr_sqrt2))
    
    def heavy_g(self, rho, mu1, mu2):
        """
        line 41-50 from
        https://github.com/microsoft/deterministic-variational-inference/blob/master/bayes_util.py
        """
        
        sqrt_one_minus_rho_sqr = torch.sqrt(self._one - rho*rho)
        a = torch.asin(rho)
        safe_a = torch.abs(a) + self._eps
        safe_rho = torch.abs(rho) + self._eps

        A = a / self._twopi
        sxx = safe_a * sqrt_one_minus_rho_sqr / safe_rho
        sxy = safe_a * sqrt_one_minus_rho_sqr 
        sxy = sxy*(self._one + sqrt_one_minus_rho_sqr) / (rho * rho)
        return A * torch.exp(-(mu1*mu1 + mu2*mu2) / (self._two * sxx) + mu1*mu2/sxy)
   
    def _heaviside_covariance(self, P, mu, P_diag):
        """
        line 87-95 from
        https://github.com/microsoft/deterministic-variational-inference/blob/master/bayes_layers.py
        """
        mu1 = mu.unsqueeze(-1)
        mu2 = mu1.permute(0,2,1)

        s11s22 = P_diag.unsqueeze(2) *  P_diag.unsqueeze(1)
        rho = P / (torch.sqrt(s11s22) + self._eps)
        rho = torch.clamp(rho, -1/(1+1e-5), 1/(1+1e-5))
        return self.heavy_g(rho, mu1, mu2)
    
    def next_moments(self, m, P):
        """
        Moment matching mode.
        
        Parameters:
            - m (torch.FloatTensor): mean with shape (batch_size, d_in)
            - P (torch.FloatTensor): covariance with (batch_size, d_in, d_in)
            
        Returns:
            - m_nxt (torch.FloatTensor): output mean with shape (batch_size, d_in)
            - P_nxt (torch.FloatTensor): output covariance with (batch_size, d_in, d_in)
        
        line 84-85, 97, and 99 from
        https://github.com/microsoft/deterministic-variational-inference/blob/master/bayes_layers.py
        """
        self._assert_dim(m, P)
        
        P_diag = torch.diagonal(P, offset=0, dim1=1, dim2=2)
        P_diag_sqrt = torch.sqrt(P_diag)
        mu = m / (P_diag_sqrt + self._eps)
        
        m_nxt = self._gaussian_cdf(mu)
        P_nxt = self._heaviside_covariance(P, mu, P_diag)
        return m_nxt, P_nxt
    
    def next_mean(self, m, P):
        """
        Calculates the next mean. Omits covariance calculation.
        
        Parameters:
            - m (torch.FloatTensor): mean with shape (batch_size, d_in)
            - P (torch.FloatTensor): covariance with (batch_size, d_in, d_in)
            
        Returns: 
            - m_nxt (torch.FloatTensor): output mean with shape (batch_size, d_in)
        
        line 84-85, and 97 from
        https://github.com/microsoft/deterministic-variational-inference/blob/master/bayes_layers.py
        """
        self._assert_dim(m, P)
        
        P_diag = torch.diagonal(P, offset=0, dim1=1, dim2=2)
        P_diag_sqrt = torch.sqrt(P_diag)
        mu = m / (P_diag_sqrt + self._eps)
        
        m_nxt = self._gaussian_cdf(mu)
        return m_nxt
        
    def forward(self, x):
        raise Exception("No Forward Mode") 
    