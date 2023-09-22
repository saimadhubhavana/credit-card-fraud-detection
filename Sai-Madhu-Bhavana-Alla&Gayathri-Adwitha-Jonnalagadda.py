#!/usr/bin/env python
# coding: utf-8

# In[7]:


get_ipython().run_line_magic('pylab', 'inline')
import os
from functools import reduce

import h5py
import numpy as np
import pandas as pd

# In[8]:


def hdf5(path, data_key = "data", target_key = "target", flatten = True):

    with h5py.File(path, 'r') as hf:
        train = hf.get('train')
        X_tr = train.get(data_key)[:]
        y_tr = train.get(target_key)[:]
        test = hf.get('test')
        X_te = test.get(data_key)[:]
        y_te = test.get(target_key)[:]
        if flatten:
            X_tr = X_tr.reshape(X_tr.shape[0], reduce(lambda a, b: a * b, X_tr.shape[1:]))
            X_te = X_te.reshape(X_te.shape[0], reduce(lambda a, b: a * b, X_te.shape[1:]))
    return X_tr, y_tr, X_te, y_te


# In[9]:


X_tr, y_tr, X_te, y_te = hdf5(r"C:\Users\bhava\OneDrive\Desktop\ML\archive\usps.h5")
X_tr.shape, X_te.shape


# In[10]:


num_samples = 3
num_classes = len(set(y_tr))

classes = set(y_tr)
num_classes = len(classes)
fig, ax = plt.subplots(num_samples, num_classes, sharex = True, sharey = True, figsize=(num_classes, num_samples))

for label in range(num_classes):
    class_idxs = np.where(y_tr == label)
    for i, idx in enumerate(np.random.randint(0, class_idxs[0].shape[0], num_samples)):
        ax[i, label].imshow(X_tr[class_idxs[0][idx]].reshape([16, 16]), 'gray')
        ax[i, label].set_axis_off()


# In[11]:


from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_neighbors=2, weights='distance', n_jobs=-1)
knn_clf.fit(X=X_tr, y=y_tr)


# In[13]:


accuracy = knn_clf.score(X=X_te, y=y_te)
print("The accuracy of KNN-classifier for USPS data set is: ", accuracy)





