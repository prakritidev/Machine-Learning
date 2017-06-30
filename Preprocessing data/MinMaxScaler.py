 #This transformation is often used as an alternative to zero mean, unit variance scaling.
from sklearn import preprocessing
import numpy as np

X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
print('intializting MinmAxScaler')
min_max_scaler1 = preprocessing.MinMaxScaler(feature_range=(0.0, 0.5 )) 
X_train_minmax1 = min_max_scaler1.fit_transform(X_train)
print(X_train_minmax1)

print('intializting MinmAxScaler')
print('Setting different range')
min_max_scaler2 = preprocessing.MinMaxScaler(feature_range=(0.1, 0.8 )) 
X_train_minmax2 = min_max_scaler2.fit_transform(X_train)
print(X_train_minmax2)