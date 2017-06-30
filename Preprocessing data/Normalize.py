# Normalization is the process of scaling individual samples to have unit norm
# This process can be useful if you plan to use a quadratic form
'''

What is Norm?

It's the total size or length of all vectors in a vector space or matrix. 
Length can be calculation using Distance formula 

L1 - Norm(Manhattan norm)
L2 - Norm(Eucledian Doistance)

For more Detailed explanantion of Norms refer this -> https://rorasa.wordpress.com/2012/05/13/l0-norm-l1-norm-l2-norm-l-infinity-norm/

More on Distance -> http://www.improvedoutcomes.com/docs/WebSiteDocs/Clustering/Clustering_Parameters/Manhattan_Distance_Metric.htm

'''


from sklearn import preprocessing
import numpy as np

X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]
X_normalized = preprocessing.normalize(X, norm='l2')

print(X_normalized)         