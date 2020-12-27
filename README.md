# HasedKNN
LSH based KNN inspired from LSH Attention (Reformer: The Efficient Transformer)

## Last Stable Release
```sh
$ pip install lshnn
```


## Usage example
From a jupyter notebook run
```python

from LSHNN import LSHNN

# Fetch dataset
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# Run LSH based KNN
knn = LSHNN(bucket_size=8, number_of_universes=20)

# Fit
knn.fit(X)

# Find ID's of 10 nearest neighbors
id = knn.find(vectorIdx=10, corpus=X, k=10)
```
