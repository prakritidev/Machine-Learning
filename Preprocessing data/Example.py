
# This code is available on http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing

'''
While working in datasets Standardization is a very common requirement for ML estimators

They might behave badly if the individual features do not more or less look like standard normally distributed data: Gaussian with zero mean and unit variance.
 
Why we do mean removal and variance scaling?

This answer is best for explaning this question. 

-> https://math.stackexchange.com/questions/317114/what-is-the-purpose-of-subtracting-the-mean-from-data-when-standardizing

In short : 

	1. By subtracting the mean, we remove the influence of our choice. 
	2. Division by σ removes the units: we get a unitless quantity ("z-score") which is independent of the scale. 

We can so this by using preprocessing.StandardScaler
'''
from sklearn import preprocessing
import numpy as np
X = np.array([[ 1., -1.,  2.],
              [ 2.,  0.,  0.],
              [ 0.,  1., -1.]])
scaler = preprocessing.StandardScaler().fit(X)

print(scaler)
print('Mean')
print(scaler.mean_) 
print()                                     
print('Scale')
print(scaler.scale_)                                       
data = scaler.transform(X) 
print()
print('Scaled data has zero mean and unit variance')
print(data)

# transforming one row.

print()
print('transformation of 1 row')

print(scaler.transform([[-1.,  1., 0.]]))

'''
Output:

StandardScaler(copy=True, with_mean=True, with_std=True)
Mean
[ 1.          0.          0.33333333]

Scale
[ 0.81649658  0.81649658  1.24721913]

Scaled data has zero mean and unit variance
[[ 0.         -1.22474487  1.33630621]
 [ 1.22474487  0.         -0.26726124]
 [-1.22474487  1.22474487 -1.06904497]]

transformation of 1 row
[[-2.44948974  1.22474487 -0.26726124]]
'''


