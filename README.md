# LSHKNN
LSH based KNN inspired from LSH Attention (Reformer: The Efficient Transformer)

## Usage example
From a jupyter notebook run
```python
from pyadlml.dataset import fetch_amsterdam

# Fetch dataset
data = fetch_amsterdam(cache=True)

# plot the persons activity density distribution over one day
from pyadlml.dataset.plot.activities import ridge_line
ridge_line(data.df_activities)

# plot the signal cross correlation between devices
from pyadlml.dataset.plot.devices import heatmap_cross_correlation
heatmap_cross_correlation(data.df_devices)

# create a raw representation with 20 second timeslices
from pyadlml.preprocessing import DiscreteEncoder, LabelEncoder
enc_dat = DiscreteEncoder(rep='raw', t_res='20s')
raw = enc_dat.fit_transform(data.df_devices)

# label the datapoints with the corresponding activity
lbls = LabelEncoder(raw).fit_transform(data.df_activities)

X = raw.values
y = lbls.values

# from here on do all the other fancy machine learning stuff you already know
from sklearn import svm
clf = svm.SVC()
clf.fit(X, y)
...
```
