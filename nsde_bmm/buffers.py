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

import numpy as np

class Generator(object):
    """ Data generator with random shuffling.
    
    Parameters:
        X (np.array): dataset with shape (num_rollouts, timesteps, dim)
    """
    
    def __init__(self, X):
        self.X = X
        
    def __call__(self, length, batch_size):
        """
        Sample snippets.
        
        Parameters:
            length (int): snippet length
            batch_size(int): batch_size
            
        Returns:
            samples (np.array): samples from X with shape (batch_size, length, dim)
        """
        start_id = np.random.randint(0, self.X.shape[0]-length, batch_size)
        start_id = np.array([range(i,i+length) for i in start_id])
        
        batch_id = np.random.randint(0, self.X.shape[1], batch_size)
        batch_id = np.repeat(batch_id, length)
        samples = self.X[start_id.flatten(), batch_id].reshape(batch_size, length, self.X.shape[-1])
        return samples