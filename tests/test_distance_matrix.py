#  DistanceMatrix class tests
#  Copyright (C) 2018 Greg Werbin
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from functools import partial

import hypothesis as h
import hypothesis.strategies as hs
import hypothesis.extra.numpy as hnp
import numpy as np
import scipy.sparse as sps
from scipy.spatial.distance import squareform, pdist

from distance_matrix import DistanceMatrix


## Hypothesis strategies ##

floats_notnull = partial(hs.floats, allow_nan=False, allow_infinity=False)
complex_notnull = partial(hs.complex_numbers, allow_nan=False, allow_infinity=False)

# TODO: fix overflow/underflow
data_strategy_real = hnp.arrays(
    np.float64, #hnp.floating_dtypes(),
    hs.tuples(hs.integers(min_value=2, max_value=50),
              hs.integers(min_value=2, max_value=5)),  # hnp.array_shapes
    floats_notnull()
)

# TODO: fix overflow/underflow
data_strategy_complex = hnp.arrays(
    np.complex128, #hnp.complex_number_dtypes(),
    hs.tuples(hs.integers(min_value=2, max_value=50), hs.just(1)),
    complex_notnull()
)

#data_strategy = data_strategy_real
data_strategy = hs.one_of(data_strategy_real, data_strategy_complex)


# TODO: number of elements must be binomial coef
#real_distance_strategy = hnp.arrays(
#    hnp.floating_dtypes(),
#    hs.integers(min_value=1, max_value=1000),
#    floats_notnull()
#)

# TODO: number of elements must be binomial coef
# Distance between complex points must have a nonnegative real part:
# https://gaurish4math.wordpress.com/tag/complex-distance/
#complex_distance_strategy = hnp.arrays(
#    hnp.complex_number_dtypes(),
#    hs.integers(min_value=1, max_value=1000),
#    hs.builds(complex, floats_notnull(min_value=0), floats_notnull())
#)

# TODO: number of elements must be binomial coef
# flat_distance_strategy = hs.one_of(real_distance_strategy, complex_distance_strategy)


## Tests ##

@h.given(data_strategy)
def test_conversions(y):
    x = pdist(y)
    d = DistanceMatrix(x)
    s = squareform(x)
    np.testing.assert_array_equal(d.toarray(), np.triu(s))
    assert (d.tosparse() != sps.coo_matrix(np.triu(s))).nnz == 0


@h.given(data_strategy)
def test_complex(y):
    x = pdist(y)
    d = DistanceMatrix(x)
    np.testing.assert_array_equal(d.imag.values, DistanceMatrix(x.imag).values)
    np.testing.assert_array_equal(d.real.values, DistanceMatrix(x.real).values)


@h.given(data_strategy)
def test_attrs(y):
    x = pdist(y)
    s = squareform(x)
    d = DistanceMatrix(x)

    assert d.T == d
    # np.testing.assert_array_equal(d.flat[0], x[0])
    assert d.size == s.size
    assert d.ndim == 2
    assert d.shape == s.shape
    assert len(d) == d.shape[0]


@h.given(data_strategy)
def test_indexing(y):
    x = pdist(y)
    s = squareform(x)
    d = DistanceMatrix(x)

    assert d[0, 0] == 0
    assert d[0, 1] == d[1, 0]
    assert d[0, 1] == s[0, 1]
    assert d[1, 0] == s[1, 0]
