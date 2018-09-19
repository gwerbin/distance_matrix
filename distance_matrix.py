#    DistanceMatrix class implementation
#    Copyright (C) 2018 Greg Werbin
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import scipy.sparse as sps
from scipy.spatial.distance import squareform

def _check_distance_dim(data):
    # https://github.com/scipy/scipy/blob/v1.1.0/scipy/spatial/distance.py#L2040-L2048
    s = data.shape
    if not len(s) == 1:
        raise ValueError('data must be 1 dimensional')

    n = s[0]
    d = int(np.ceil(np.sqrt(n * 2)))
    if d * (d - 1) != n * 2:
        raise ValueError('Incompatible vector size. It must be a binomial '
                         'coefficient (n choose 2) for some integer n >= 2.')

    return d


def _upper_ij2k(i, j, n):
    # https://stackoverflow.com/a/27088560/2954547
    return (n*(n-1)//2) - (n-i)*((n-i)-1)//2 + j - i - 1


def _upper_k2ij(k, n):
    # https://stackoverflow.com/a/27088560/2954547
    i = n - 2 - int(np.sqrt(-8*k + 4*n*(n-1)-7)/2.0 - 0.5)
    j = k + i + 1 - n*(n-1)//2 + (n-i)*((n-i)-1)//2
    return i,j


def _check_ij(ij, n):
        if len(ij) > 2:
            raise IndexError('too many indices for array')
        if len(ij) < 2:
            raise IndexError('not enough indices for array')

        i, j = ij
        #if i >= n:
            #raise IndexError('index {} out of bounds for axis 0 with size {}'.format(i, n))
        #if j >= n:
            #raise IndexError('index {} out of bounds for axis 0 with size {}'.format(j, n))
        
        return i, j


class DistanceMatrix():
    """ Treat a flat array like a distance matrix """
    def __init__(self, data):
        data = np.ascontiguousarray(data)
        d = _check_distance_dim(data)
        self.values = data
        self._size = d**2
        self._shape = (d, d)

    @property
    def T(self):
        return self

    @property
    def flat(self):
        # TODO: this might be tricky
        raise NotImplementedError()

    @property
    def imag(self):
        return self.__class__(self.values.imag)

    @property
    def real(self):
        return self.__class__(self.values.real)

    @property
    def size(self):
        return self._size

    @property
    def ndim(self):
        return 2

    @property
    def shape(self):
        return self._shape

    def __len__(self):
        return self.shape[0]
    
    def __getitem__(self, ij):
        n = len(self.values)
        i,j = ij = _check_ij(ij, n)
        if i == j:
            return 0

        ij = tuple(sorted(ij))
        k = _upper_ij2k(*ij, n)
        return self.values[k]

    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html
    #def __getattr__(self, a):
        #if hasattr(self, a):
            #return getattr(self, a)
        #else:
            #return getattr(self.values, a)

    def toarray(self, upper=True):
        arr = np.zeros((self.shape), dtype=self.values.dtype)
        #n = len(self)
        #row, col = _upper_k2ij(np.arange(n), n)
        indices = np.triu_indices if upper else np.tril_indices
        row, col = indices(self.shape[0], k=1)
        arr[row, col] = self.values
        return arr
    
    def tosparse(self, upper=True):
        #n = len(self)
        #row, col = _upper_k2ij(np.arange(n), n)
        indices = np.triu_indices if upper else np.tril_indices
        row, col = indices(self.shape[0], k=1)
        return sps.coo_matrix((self.values, (row, col)), self.shape)
