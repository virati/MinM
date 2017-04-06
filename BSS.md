+++
date = "2016-12-25T13:38:07-05:00"
title = "Some Blind Source Separation"
draft = false

+++

# Blind Source Separations

## Introduction
Blind source separation is the problem of trying to split out independent processes that are generating data. Doing this without a priori information about the system/s generating the data is the "blind" part of this.

A common example of this type of problem is trying to identify the number of people speaking in a noisy room with a certain number of microphones. Each microphone picks up each speaker, but to varying degrees. With information about where the microphones are, this problem is not so "blind". Without information about where the microphones are, this problem becomes "blind", but not insurmountable.

## Generate our data
We know the properties that our data needs:

* Gaussian with noise
* Multimodal

Since we'll be dealing with timeseries in other notebooks, we'll focus our conversation around timeseries knowing that the principles are generalizable.


```python
%reset
import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as sig
```

    Once deleted, variables cannot be recovered. Proceed (y/[n])? y



```python
#We'll deal with 2D data

#Simple cross example
mean1 = [0,0]
cov1 = [[0,0.7],[-3.5,6]]
cov2 = [[0,-0.7],[3.5,6]]
mean2 = [4,0]
x,y = np.random.multivariate_normal(mean1,cov1,100).T
u,v = np.random.multivariate_normal(mean2,cov2,100).T

plt.plot(x,y,'x')
plt.plot(u,v,'x',color='r')
plt.axis('equal')
plt.show()
```


![png](/imgs/output_3_0.png)


We made a dataset where two independent processes are observed. We know it's two independent processes because we *made them from scratch using two separate function calls to multivariate normal*. Of course, since it's all pseudorandom number generation, might have to make sure the seeds are different for each call, but I'm not sure that's how it's supposed to work.

We have a dataset with two independent processes. We want to now study this and *find* these processes from data where we won't know where each datapoint is actually coming from.

This set is actually very easy to see *visually* but let's do the process from the ground up. We start with linear approaches, move to ICA, then to gaussian processes.

### Principle Component Analysis
First we'll do a PCA on the aggregate dataset. This will give us two components: a component in the direction of maximal variance, and another one orthogonal to that


```python
data = np.vstack((np.hstack((x,u)),np.hstack((y,v))))
plt.plot(data[0,:],data[1,:],'x')
plt.axis('equal')
plt.show()
```


![png](/imgs/output_6_0.png)



```python
from sklearn.decomposition import PCA as sklPCA

skl_PCA = sklPCA(n_components=2)
skl_Xform = skl_PCA.fit_transform(data.T)

plt.plot(skl_Xform[:,0],skl_Xform[:,1],'o')
plt.axis('equal')
plt.show()

pcs = skl_PCA.components_

plt.figure()
ax=plt.axes()

plt.plot(data[0,:],data[1,:],'x')
ax.arrow(0,0,5*pcs[0,0],5*pcs[1,0],color='r',head_width=0.5)
ax.arrow(0,0,2*pcs[0,1],2*pcs[1,1],color='g',head_width=0.5)
#plt.plot(pcs[0,:])
plt.axis('equal')
plt.show()
```


![png](/imgs/output_7_0.png)



![png](/imgs/output_7_1.png)


So, according to PCA, we've got two components in our data. One going one direction, the other going the other direction. But, we have to look at the eigenvalues for each of these components to see how much of the variance is explained for each.

### Independent Component Analysis
ICA should give us the two components themselves, though since one component is 2d symmetric, not sure what will happen there...

### Gaussian Mixture Models
GMM should give us the two gaussian! Let's just go for it


```python
from sklearn import mixture
from matplotlib.colors import LogNorm

clf = mixture.GaussianMixture(n_components=2,covariance_type='full')
clf.fit(data.T)

xd = np.linspace(-20,20)
yd = np.linspace(-20,20)
Xd,Yd = np.meshgrid(xd,yd)
XX = np.array([Xd.ravel(),Yd.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(Xd.T.shape)

CS = plt.contour(Xd,Yd,Z,norm=LogNorm(vmin=1.0,vmax=1000),levels=np.logspace(0,2,20))
plt.scatter(data[0,:],data[1,:],.8)
plt.axis('equal')
plt.axis('tight')
plt.show()
```


![png](/imgs/output_11_0.png)

