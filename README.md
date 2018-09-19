Treat a "flat" distance matrix kind of like a Numpy array

```python
import numpy as np
from scipy.spatial.distance import pdist
from distance_matrix import DistanceMatrix

x = np.array([
    [1, 2, 1],
    [0, 3, 7],
    [2, 2, 5],
    [4, 6, 4]
])

d = pdist(x)

dm = DistanceMatrix(d)

print(dm[1, 1])
print(dm[2, 0])
print(dm[0, 2])
print(dm.values)
print(len(dm))
print(dm.shape)
print(dm.size)
assert dm.T.values == dm.T
```
