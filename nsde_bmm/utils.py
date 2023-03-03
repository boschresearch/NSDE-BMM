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
import numpy as np

def mc_moments(X):
    """Calculates the first two moments of X.
    
    Parameters:
        - X (torch.FloatTensor): set of samples with shape (timesteps, num_samples, dim)
    
    Returns:
        - m (torch.FloatTensor): MC approximation of mean with (timesteps, dim)
        - P (torch.FloatTensor): MC approximation of covariance with (timesteps, dim, dim)
    """
    m = X.mean(1)
    delta = X - m.unsqueeze(1)
    P = delta.reshape(-1, delta.shape[-1], 1) @ delta.reshape(-1, 1, delta.shape[-1])
    P = P.reshape(*delta.shape, delta.shape[-1]).sum(1)/(delta.shape[1]-1)
    return m, P

def neg_log_likelihood(y, m, P):
    """Calculates negative log likelihood.
    
    Parameters:
        - y (torch.FloatTensor): observation with shape (batch_size, dim)
        - m (torch.FloatTensor): mean with shape (batch_size, dim)
        - P (torch.FloatTensor): covariance with shape (batch_size, dim, dim)
    
    Returns:
        - nll (torch.FloatTensor): nll with shape (batch_size, 1)
    """
    batch_size, dim = y.shape
    device = y.device
    
    EPS = torch.FloatTensor([1e-4]).to(device) 
       
    chol = torch.linalg.cholesky(P) 

    B = torch.eye(dim).expand(batch_size, dim, dim)
    inv = torch.linalg.solve(chol, B) 
    inv = inv.transpose(1,2)@inv  

    det = torch.prod(torch.diagonal(chol, 0, 1, 2)**2, 1).view(batch_size, 1, 1) + EPS

    l2_distance = (y - m).view(batch_size, 1, dim)
    mhb = l2_distance@inv@l2_distance.transpose(1,2) 

    nll =  mhb + torch.log(det)
    return nll.reshape(batch_size, 1)
